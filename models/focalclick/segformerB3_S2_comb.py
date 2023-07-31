from isegm.utils.exp_imports.default import *

MODEL_NAME = 'segformerB3_S2_cclvs'
import os
from isegm.data.datasets.lvis import LvisDataset
from isegm.data.compose import ComposeDataset, ProportionalComposeDataset
from isegm.data.aligned_augmentation import AlignedAugmentator
from isegm.engine.focalclick_trainer import ISTrainer
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def main(cfg, rank=0):
    # TODO: 修改的，只输出model cfg
    model, model_cfg = init_model(cfg, rank)
    train(model, cfg, model_cfg, rank)


def init_model(cfg, rank):
    # 配置文件
    model_cfg = {
        'crop_size': (256, 256),
        'num_max_points': 24,
        'with_prev_mask': True,
        "use_cross_attn": True,
        "use_affinity_loss": [False, False, True, False],
        "use_attn_weight": [False, False, True, False],
        'use_pcgrad': True,
        'use_attn_weight_noise': [True, False, True, True],
        'lr': 1e-3,
        'optim': 'adamw'
    }

    model = SegFormerModel(pipeline_version='s2', model_version='b3',
                           use_leaky_relu=True, use_rgb_conv=False, use_disks=True, norm_radius=5,
                           binary_prev_mask=False,
                           with_aux_output=True, **model_cfg)
    model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.SEGFORMER_B3)
    # TODO:新改的
    if cfg.distributed:
        torch.cuda.set_device(rank)  # 解决卡0多出很多 731MB 显存占用的情况
        model = DDP(model.to(rank), device_ids=[rank], output_device=rank, find_unused_parameters=False)
    else:
        model.to(cfg.device)

    return model, model_cfg


def train(model, cfg, model_cfg, rank):
    cfg.batch_size = 28 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg['crop_size']

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0

    loss_cfg.instance_refine_loss = WFNL(alpha=0.5, gamma=2, w=0.5)
    loss_cfg.instance_refine_loss_weight = 1.0

    loss_cfg.trimap_loss = nn.BCEWithLogitsLoss()  # NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.trimap_loss_weight = 1.0

    loss_cfg.trimap_loss = nn.BCEWithLogitsLoss()  # NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.trimap_loss_weight = 1.0

    if (True in model_cfg['use_affinity_loss']):
        loss_cfg.affinity_loss = SegformerAffinityEnergyLoss(class_num=1, loss_index=model_cfg['use_affinity_loss'])
        loss_cfg.affinity_loss_weight = 0.05

    if (True in model_cfg['use_attn_weight']):
        loss_cfg.attenW_loss = AttWeightLoss(model_cfg['use_attn_weight'])
        loss_cfg.attenW_loss_weight = 0.1

    color_augmentator = Compose([
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    train_augmentator = AlignedAugmentator(ratio=[0.3, 1.3], target_size=crop_size, flip=True,
                                           distribution='Gaussian', gs_center=0.8, gs_sd=0.4,
                                           color_augmentator=color_augmentator
                                           )

    val_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.25)),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiPointSampler(model_cfg['num_max_points'], prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2,
                                       use_hierarchy=False,
                                       first_click_center=True)

    trainset_cclvs = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split='train',
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=30000,
        stuff_prob=0.1
    )

    trainset_ytbvos = YouTubeDataset(
        cfg.YTBVOS_PATH,
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=-1,
    )

    trainset_ade20k = ADE20kDataset(
        cfg.ADE20K_PATH,
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=-1,
        with_part=False
    )

    trainset_saliency = SaliencyDataset(
        [cfg.MSRA10K_PATH, cfg.DUT_TR_PATH, cfg.DUT_TE_PATH],
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=-1,
    )

    trainset_hflicker = HFlickerDataset(
        cfg.HFLICKER_PATH,
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=-1,
    )

    trainset_thin = ThinObjectDataset(
        cfg.THINOBJECT_PATH,
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=-1,
    )

    trainset_comb = ProportionalComposeDataset(
        [trainset_cclvs, trainset_saliency, trainset_thin, trainset_ytbvos, trainset_hflicker, trainset_ade20k],
        [0.35, 0.10, 0.1, 0.2, 0.1, 0.15],
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=30000,
    )

    valset = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split='val',
        augmentator=val_augmentator,
        min_object_area=1000,
        points_sampler=points_sampler,
        epoch_len=2000
    )


    optimizer_params = {
        'lr': model_cfg['lr'], 'betas': (0.9, 0.999), 'eps': 1e-8,
        # 'weight_decay':1e-8#''momentum' : 0.9, 'nesterov' : True #
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[190, 210], gamma=0.1)

    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset_comb, valset,
                        optimizer=model_cfg['optim'],
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 50), (200, 5)],
                        image_dump_interval=500,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg['num_max_points'],
                        max_num_next_clicks=3,
                        rank=rank)
    trainer.run(num_epochs=230)

