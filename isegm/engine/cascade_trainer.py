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

from isegm.utils.log import logger, TqdmToLogger, SummaryWriterAvg
from isegm.utils.misc import save_checkpoint
from isegm.utils.distributed import get_sampler, reduce_loss_dict, get_dp_wrapper
from .optimizer import get_optimizer
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
                 click_models=None,
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

        self.click_models = click_models
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

        self.optim = get_optimizer(model, optimizer, optimizer_params)

        if self.model_cfg['use_pcgrad']:
            # self.optim = PCGrad(2, self.optim, scaler=scaler, reduction='sum', cpu_offload= False)
            # For Gradient Vaccine
            self.optim = GradVacAMP(2, self.optim, torch.device('cuda:{}'.format(rank)), scaler=scaler,
                                    beta=1e-2, reduction='sum', cpu_offload=False)

        model = self._load_weights(model)

        if cfg.multi_gpu and (not cfg.distributed):
            model = get_dp_wrapper(cfg.distributed)(model, device_ids=cfg.gpu_ids,
                                                    output_device=cfg.gpu_ids[0])

        self.device = cfg.device
        self.net = model.to(self.device) if (cfg.multi_gpu and (not cfg.distributed)) else model
        self.lr = optimizer_params['lr']

        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(optimizer=self.optim)
            if cfg.start_epoch > 0:
                for _ in range(cfg.start_epoch):
                    self.lr_scheduler.step()

        self.tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        if self.click_models is not None:
            for click_model in self.click_models:
                for param in click_model.parameters():
                    param.requires_grad = False
                click_model.to(self.device)
                click_model.eval()

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

        # , file=self.tqdm_out, ncols=100, ascii=True, position=0, leave=False) \
        tbar = tqdm(self.train_data) \
            if (self.is_master and 'search_param' not in self.model_cfg.keys()) else self.train_data

        for metric in self.train_metrics:
            metric.reset_epoch_stats()

        self.net.train()
        # for m in self.net.feature_extractor.modules():
        #        m.eval()

        train_loss = 0.0
        for i, batch_data in enumerate(tbar):
            global_step = epoch * len(self.train_data) + i

            loss, losses_logging, splitted_batch_data, outputs = self.batch_forward(batch_data)

            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

            # if self.is_master:
            #     for name, param in self.net.named_parameters():
            #         if param.grad is None:
            #             print(name)
            losses_logging['overall'] = loss

            reduce_loss_dict(losses_logging)

            train_loss += losses_logging['overall'].item()

            if self.cfg['model_ema'] and i % self.cfg.model_ema_steps == 0:
                self.ema_net.update_parameters(self.net)
                if epoch < 2:  
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

        # if 'search_param' in self.model_cfg.keys():
        #     tune.report(loss=(train_loss / (i + 1)), iou=self.train_metrics[0]._ema_iou)

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

            if random.random() < 0.0001:
                points[:] = -1
                points = get_next_points_removeall(prev_output, gt_mask, points,1)

            loss = 0.0
            num_iters = random.randint(1, self.max_num_next_clicks)
            for click_indx in range(num_iters):
                # v2
                net_input = torch.cat((image, prev_output), dim=1) \
                    if self.model_cfg['with_prev_mask'] else image

                output = self.net(net_input, points)

                loss = self.add_loss('instance_loss', loss, losses_logging, validation,
                                     lambda: (output['instances'], batch_data['instances']))

                loss = self.add_loss('affinity_loss', loss, losses_logging, validation,
                                     lambda: (
                                     output['instances'], output['attns'], output['attnW'], output['pos_click_len']))

                loss = self.add_loss('attenW_loss', loss, losses_logging, validation,
                                     lambda: (output['attnW'], batch_data['instances'], output['pos_click_len']))

                prev_output = torch.sigmoid(output['instances']).detach()
                if click_indx < num_iters - 1:
                    # points = get_next_points(prev_output,
                    points = get_next_points_removeall(prev_output,
                                             gt_mask, points, click_indx+1)

                if self.model_cfg['with_prev_mask']  and self.prev_mask_drop_prob > 0:
                    zero_mask = np.random.random(size=prev_output.size(0)) < self.prev_mask_drop_prob
                    prev_output[zero_mask] = torch.zeros_like(prev_output[zero_mask])

            if self.is_master:
                with torch.no_grad():
                    for m in metrics:
                        m.update(*(output.get(x) for x in m.pred_outputs),
                                 *(batch_data[x] for x in m.gt_outputs))

        batch_data['points'] = points
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
            if self.model_cfg['use_pcgrad']:
                total_loss.append(loss)
            else:
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

    @property
    def is_master(self):
        return self.rank == 0

def get_next_points_removeall(pred, gt, points, click_indx, pred_thresh=0.49, remove_prob=0.0):
    assert click_indx > 0
    pred = pred.cpu().numpy()[:, 0, :, :]
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
            if np.random.rand() < remove_prob:
                points[bindx] = points[bindx] * 0.0 - 1.0
            if is_positive:
                points[bindx, num_points - click_indx, 0] = float(coords[0])
                points[bindx, num_points - click_indx, 1] = float(coords[1])
                points[bindx, num_points - click_indx, 2] = float(click_indx)
            else:
                points[bindx, 2 * num_points - click_indx, 0] = float(coords[0])
                points[bindx, 2 * num_points - click_indx, 1] = float(coords[1])
                points[bindx, 2 * num_points - click_indx, 2] = float(click_indx)

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
