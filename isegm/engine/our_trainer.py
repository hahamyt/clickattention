import os
import random
import logging
from collections import defaultdict
from copy import deepcopy
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import distributed as dist

from isegm.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.misc import save_checkpoint
from isegm.utils.distributed import get_sampler, reduce_loss_dict
from .optimizer import get_optimizer_with_layerwise_decay, get_optimizer
from torch.cuda.amp import autocast as autocast, GradScaler

# from ray import tune

from ..inference.utils import ExponentialMovingAverage
from ..model.pcgrad import PCGrad, GradVacAMP

scaler = GradScaler()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class ISTrainer(object):
    def __init__(self, model, cfg, model_cfg, loss_cfg,
                 trainset, valset,
                 optimizer='adam',
                 optimizer_params=None,
                 image_dump_interval=200,
                 checkpoint_interval=15,
                 tb_dump_period=25,
                 max_interactive_points=0,
                 lr_scheduler=None,
                 metrics=None,
                 additional_val_metrics=None,
                 net_inputs=('images', 'points'),
                 max_num_next_clicks=0,
                 prev_mask_drop_prob=0.0,
                 rank=0
                 ):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.rank = rank

        self.max_interactive_points = max_interactive_points
        self.loss_cfg = loss_cfg
        self.val_loss_cfg = deepcopy(loss_cfg)
        self.tb_dump_period = tb_dump_period
        self.net_inputs = net_inputs
        self.max_num_next_clicks = max_num_next_clicks

        self.prev_mask_drop_prob = prev_mask_drop_prob

        if cfg.distributed:
            cfg.batch_size //= cfg.ngpus
            cfg.val_batch_size //= cfg.ngpus

        if metrics is None:
            metrics = []
        self.train_metrics = metrics
        self.val_metrics = deepcopy(metrics)
        if additional_val_metrics is not None:
            self.val_metrics.extend(additional_val_metrics)

        self.checkpoint_interval = checkpoint_interval
        self.image_dump_interval = image_dump_interval
        self.task_prefix = ''
        self.sw = None

        self.trainset = trainset
        self.valset = valset

        if self.is_master:
            logger.info(f'Dataset of {trainset.get_samples_number()} samples was loaded for training.')
            logger.info(f'Dataset of {valset.get_samples_number()} samples was loaded for validation.')

        self.train_data = DataLoader(
            trainset, cfg.batch_size,
            sampler=get_sampler(trainset, shuffle=True, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers,
        )

        self.val_data = DataLoader(
            valset, cfg.val_batch_size,
            sampler=get_sampler(valset, shuffle=False, distributed=cfg.distributed),
            drop_last=True, pin_memory=True,
            num_workers=cfg.workers
        )

        model = self._load_weights(model)

        if cfg.model_ema:
            if self.is_master:
                print("Using EMA")
            adjust = torch.cuda.device_count() * cfg.batch_size * cfg.model_ema_steps / 230  # torch.cuda.device_count()
            alpha = 1.0 - cfg.model_ema_decay
            alpha = min(1.0, alpha * adjust)
            self.ema_net = ExponentialMovingAverage(model.module if cfg.distributed else model,
                                                    device=next(model.parameters()).device, decay=1.0 - alpha)

        self.device = cfg.device
        self.net = model
        self.lr = optimizer_params['lr']

        self.optim = get_optimizer(self.net, optimizer, optimizer_params)

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        # self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)

    def run(self, num_epochs, start_epoch=None, validation=True):
        if start_epoch is None:
            start_epoch = self.cfg.start_epoch

        if self.is_master:
            logger.info(f'Starting Epoch: {start_epoch}')
            logger.info(f'Total Epochs: {num_epochs}')

        for epoch in range(start_epoch, num_epochs):
            self.training(epoch)
            # if validation:
            #    self.validation(epoch)

    def training(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        if self.cfg.distributed:
            self.train_data.sampler.set_epoch(epoch)

        log_prefix = 'Train' + self.task_prefix.capitalize()
        # file=self.tqdm_out,ncols=80
        tbar = tqdm(self.train_data) \
            if self.is_master else self.train_data

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        self.net.train()
        # for m in self.net.feature_extractor.modules():
        #        m.eval()

        train_loss = 0.0
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.train_data) + i
            if self.model_cfg['use_fp16']:
                self.optim.zero_grad()
                with autocast():
                    loss, losses_logging, splitted_batch_data, outputs = self.batch_forward(batch_data)
                    scaler.scale(loss)  # 解决报错问题
                    loss.backward()
                    scaler.step(self.optim)
                    scaler.update()
            else:
                loss, losses_logging, splitted_batch_data, outputs = self.batch_forward(batch_data)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            # 检查无梯度的参数
            # if self.is_master:
            #     for name, param in self.net.named_parameters():
            #         if param.grad is None:
            #             print(name)
            losses_logging['overall'] = loss

            reduce_loss_dict(losses_logging)

            train_loss += losses_logging['overall'].item()

            if self.cfg['model_ema'] and i % self.cfg.model_ema_steps == 0:
                self.ema_net.update_parameters(self.net)
                if epoch < 2:  # 在2个EPOCH前去掉权重
                    # Reset ema buffer to keep copying weights during warmup period
                    self.ema_net.n_averaged.fill_(0)

            if self.is_master:
                for loss_name, loss_value in losses_logging.items():
                    self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}',
                                       value=loss_value.item(),
                                       global_step=global_step)

                for k, v in self.loss_cfg.items():
                    if '_loss' in k and hasattr(v, 'log_states') and self.loss_cfg.get(k + '_weight', 0.0) > 0:
                        v.log_states(self.sw, f'{log_prefix}Losses/{k}', global_step)

                self.sw.add_scalar(tag=f'{log_prefix}States/learning_rate',
                                   value=self.lr if not hasattr(self, 'lr_scheduler') else self.lr_scheduler.get_lr()[
                                       -1],
                                   global_step=global_step)
                if 'search_param' not in self.model_cfg.keys():
                    tbar.set_description(
                        f'Epoch {epoch}, training loss {train_loss / (i + 1):.4f}, ema iou {self.train_metrics[0]._ema_iou:.4f}')
                for metric in self.train_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for metric in self.train_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}',
                                   value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

            if isinstance(self.checkpoint_interval, (list, tuple)):
                checkpoint_interval = [x for x in self.checkpoint_interval if x[0] <= epoch][-1][1]
            else:
                checkpoint_interval = self.checkpoint_interval
            # if 'search_param' not in self.model_cfg.keys():
            save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,  # self.ema_net,
                            epoch=epoch, distributed=self.cfg.distributed)

            if epoch % checkpoint_interval == 0:
                save_checkpoint(self.net, self.cfg.CHECKPOINTS_PATH, prefix=self.task_prefix,  # self.ema_net,
                                epoch=None, distributed=self.cfg.distributed)

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

    def validation(self, epoch):
        if self.sw is None and self.is_master:
            self.sw = SummaryWriterAvg(log_dir=str(self.cfg.LOGS_PATH),
                                       flush_secs=10, dump_period=self.tb_dump_period)

        log_prefix = 'Val' + self.task_prefix.capitalize()
        tbar = tqdm(self.val_data, file=self.tqdm_out, ncols=100) if self.is_master else self.val_data

        for metric in self.val_metrics:
            metric.reset_epoch_stats()

        val_loss = 0
        losses_logging = defaultdict(list)

        self.net.eval()
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.val_data) + i
            loss, batch_losses_logging, splitted_batch_data, outputs = \
                self.batch_forward(batch_data, validation=True)

            batch_losses_logging['overall'] = loss
            reduce_loss_dict(batch_losses_logging)
            for loss_name, loss_value in batch_losses_logging.items():
                losses_logging[loss_name].append(loss_value.item())

            val_loss += batch_losses_logging['overall'].item()

            if self.is_master:
                tbar.set_description(f'Epoch {epoch}, validation loss: {val_loss/(i + 1):.4f}')
                for metric in self.val_metrics:
                    metric.log_states(self.sw, f'{log_prefix}Metrics/{metric.name}', global_step)

        if self.is_master:
            for loss_name, loss_values in losses_logging.items():
                self.sw.add_scalar(tag=f'{log_prefix}Losses/{loss_name}', value=np.array(loss_values).mean(),
                                   global_step=epoch, disable_avg=True)

            for metric in self.val_metrics:
                self.sw.add_scalar(tag=f'{log_prefix}Metrics/{metric.name}', value=metric.get_epoch_value(),
                                   global_step=epoch, disable_avg=True)

    def batch_forward(self, batch_data, validation=False):
        metrics = self.val_metrics if validation else self.train_metrics
        losses_logging = dict()

        with torch.set_grad_enabled(not validation):
            batch_data = {k: v.to(self.rank) if self.cfg.distributed else v.to(self.device) for k, v in batch_data.items()}
            image, gt_mask, points = batch_data['images'], batch_data['instances'], batch_data['points']

            prev_output = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
            loss = 0.0

            if random.random() < 0.02:
                # 相当于重头开始点击，不从数据集中采样，特点就是只生成一次点击，对应的previous为空，符合实际情况
                points[:] = -1
                points = get_next_points(prev_output, gt_mask, points)
            else:
                # 用于生成 previous mask 所需的迭代次数
                if min(torch.sum(points[:, :24, 0] > 0, dim=1)) > 1:
                    num_iters = random.randint(1, self.max_num_next_clicks)
                else:
                    num_iters = random.randint(0, self.max_num_next_clicks - 1)

                points, prev_output = self.find_next_n_points(
                    image,
                    gt_mask,
                    points,
                    prev_output,
                    num_iters,
                    not validation
                )

            net_input = torch.cat((image, prev_output), dim=1) if self.model_cfg['with_prev_mask'] else image
            # 第二次进入模型，输入上一次预测的mask，以及新的至多三个点击点
            output = self.net(net_input, points)

            loss = self.add_loss('instance_loss', loss, losses_logging, validation,
                                 lambda: (output['instances'], batch_data['instances']))

            if True in self.model_cfg['use_attn_weight']:
                loss = self.add_loss('affinity_loss', loss, losses_logging, validation,
                                     lambda: (output['instances'], output['attns'], output['attnW'], output['pos_click_len']))

                loss = self.add_loss('attenW_loss', loss, losses_logging, validation,
                                         lambda: (output['attnW'], batch_data['instances'], output['pos_click_len']))

            if self.is_master:
                with torch.no_grad():
                    for m in metrics:
                        m.update(*(output.get(x) for x in m.pred_outputs),
                                 *(batch_data[x] for x in m.gt_outputs))
        return loss, losses_logging, batch_data, output

    def add_loss(self, loss_name, total_loss, losses_logging, validation, lambda_loss_inputs):
        loss_cfg = self.loss_cfg if not validation else self.val_loss_cfg
        loss_weight = loss_cfg.get(loss_name + '_weight', 0.0)
        if loss_weight > 0.0:
            loss_criterion = loss_cfg.get(loss_name)
            loss = loss_criterion(*lambda_loss_inputs())
            loss = torch.mean(loss)
            losses_logging[loss_name] = loss
            loss = loss_weight * loss

            total_loss = total_loss + loss

        return total_loss

    def _load_weights(self, net):
        if self.cfg.weights is not None:
            if os.path.isfile(self.cfg.weights):
                load_weights(net, self.cfg.weights)
                self.cfg.weights = None
            else:
                raise RuntimeError(f"=> no checkpoint found at '{self.cfg.weights}'")
        elif self.cfg.resume_exp is not None:
            checkpoints = list(self.cfg.CHECKPOINTS_PATH.glob(f'{self.cfg.resume_prefix}*.pth'))
            assert len(checkpoints) == 1

            checkpoint_path = checkpoints[0]
            logger.info(f'Load checkpoint from path: {checkpoint_path}')
            load_weights(net.module if self.cfg.distributed else net, str(checkpoint_path))
        return net

    def find_next_n_points(self, image, gt_mask, points, prev_output,
                           num_points, eval_mode=False, grad=False):

        with torch.set_grad_enabled(grad):
            for click_indx in range(num_points):

                if eval_mode:
                    self.net.eval()

                net_input = torch.cat((image, prev_output), dim=1) if self.model_cfg['with_prev_mask'] else image

                prev_output = torch.sigmoid(self.net(net_input, points)['instances'])

                points = get_next_points(prev_output, gt_mask, points)

                if eval_mode:
                    self.net.train()

            if self.model_cfg['with_prev_mask'] and self.prev_mask_drop_prob > 0 and num_points > 0:
                zero_mask = np.random.random(
                    size=prev_output.size(0)) < self.prev_mask_drop_prob
                prev_output[zero_mask] = \
                    torch.zeros_like(prev_output[zero_mask])

        return points, prev_output

    @property
    def is_master(self):
        return self.rank == 0

def get_next_points(pred, gt, points, pred_thresh=0.49):
    pred = pred.detach().cpu().numpy()[:, 0, :, :]
    gt = gt.cpu().numpy()[:, 0, :, :] > 0.5

    fn_mask = np.logical_and(gt, pred < pred_thresh)
    fp_mask = np.logical_and(np.logical_not(gt), pred > pred_thresh)

    fn_mask = np.pad(fn_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    fp_mask = np.pad(fp_mask, ((0, 0), (1, 1), (1, 1)), 'constant').astype(np.uint8)
    num_points = points.size(1) // 2
    points = points.clone()

    for bindx in range(fn_mask.shape[0]):
        fn_mask_dt = cv2.distanceTransform(fn_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]
        fp_mask_dt = cv2.distanceTransform(fp_mask[bindx], cv2.DIST_L2, 5)[1:-1, 1:-1]

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        dt = fn_mask_dt if is_positive else fp_mask_dt
        inner_mask = dt > max(fn_max_dist, fp_max_dist) / 2.0
        indices = np.argwhere(inner_mask)
        if len(indices) > 0:
            coords = indices[np.random.randint(0, len(indices))]
            order = max(points[bindx, :, 2].max(), 0) + 1
            if is_positive:
                loc = torch.argwhere(points[bindx, :num_points, 2] < 0)
                loc = loc[0, 0] if len(loc) > 0 else num_points - 1
                points[bindx, loc, 0] = float(coords[0])
                points[bindx, loc, 1] = float(coords[1])
                points[bindx, loc, 2] = float(order)
            else:
                loc = torch.argwhere(points[bindx, num_points:, 2] < 0)
                loc = loc[0, 0] + num_points if len(loc) > 0 else 2 * num_points - 1
                points[bindx, loc, 0] = float(coords[0])
                points[bindx, loc, 1] = float(coords[1])
                points[bindx, loc, 2] = float(order)

    return points


def load_weights(model, path_to_weights):
    current_state_dict = model.state_dict()
    new_state_dict = torch.load(path_to_weights, map_location='cpu')['state_dict']

    new_keys = set(new_state_dict.keys())
    old_keys = set(current_state_dict.keys())
    print('=' * 10)
    print(' unexpected: ', new_keys - old_keys)
    print(' lacked: ', old_keys - new_keys)
    print('=' * 10)
    current_state_dict.update(new_state_dict)
    model.load_state_dict(current_state_dict, strict=False)
