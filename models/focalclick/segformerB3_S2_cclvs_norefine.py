from isegm.utils.exp_imports.default import *

MODEL_NAME = 'segformerB3_S2_cclvs_norefine'

from isegm.data.aligned_augmentation import AlignedAugmentator
from isegm.engine.our_trainer import ISTrainer
# from isegm.engine.cascade_trainer import ISTrainer
from torch.nn.parallel import DistributedDataParallel as DDP

def main(cfg, rank=0):
    model, model_cfg = init_model(cfg, rank)
    train(model, cfg, model_cfg, rank)

def init_model(cfg, rank):
    # 配置文件
    model_cfg = {
        'crop_size': (512, 512),
        'num_max_points':24,
        'with_prev_mask':True,
        "use_attn_weight": [False, True, True, True],
        'lr': 5e-4,
        'optim': 'adamw',
        'use_fp16': False
    }
    torch.set_float32_matmul_precision('high')  # 使用中等精度计算32位矩阵乘法，最高精度的是highest
    model = SegFormerModelNoRefine(  pipeline_version = 's2', model_version = 'b3',
                       use_leaky_relu=True, use_rgb_conv=False, use_disks=True, norm_radius=5, binary_prev_mask=False,
                       with_aux_output=True, **model_cfg)

    model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.SEGFORMER_B3)
    # model.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.SEGFORMER_B3)
    # model = torch.compile(model)
    if cfg.distributed:
        torch.cuda.set_device(rank)     # 解决卡0多出很多 731MB 显存占用的情况
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model.to(rank), device_ids=[rank], output_device=rank,
                    broadcast_buffers=False,
                    find_unused_parameters=True)    # 因为不计算一次点击的损失，所以导致有些参数没有
    else:
        if not cfg.multi_gpu:
            model.to(cfg.device)

    return model, model_cfg

def train(model, cfg, model_cfg, rank):
    cfg.batch_size = 28 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg['crop_size']

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0

    if (True in model_cfg['use_attn_weight']):
        loss_cfg.affinity_loss = DiscriminateiveAffinityLoss(class_num=1, loss_index=model_cfg['use_attn_weight'])
        loss_cfg.affinity_loss_weight = 1.0

        loss_cfg.attenW_loss = AttWeightLoss(model_cfg['use_attn_weight'])
        loss_cfg.attenW_loss_weight = 1.0

    color_augmentator = Compose([
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    train_augmentator = AlignedAugmentator(ratio=[0.3,1.3], target_size=crop_size, flip=True, 
                                            distribution='Gaussian', gs_center = 0.8, gs_sd = 0.4,
                                            color_augmentator = color_augmentator
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
        epoch_len=30000,          # TODO
        stuff_prob=0.2,
    )


    valset = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split='val',
        augmentator=val_augmentator,
        min_object_area=1000,
        points_sampler=points_sampler,
        epoch_len=2000,
    )


    optimizer_params = {
        'lr': model_cfg['lr'], 'betas': (0.9, 0.999),  'eps': 1e-8,
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[160, 210], gamma=0.1)

    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset_cclvs, valset,
                        optimizer=model_cfg['optim'],
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 50), (200, 5)],
                        image_dump_interval=500,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg['num_max_points'],
                        max_num_next_clicks=3,
                        rank=rank)
    trainer.run(num_epochs=500)
'''
Eval results for model: 146
-----------------------------------------------------------------------------------------------
|  Pipeline   |  Dataset  | NoC@80% | NoC@85% | NoC@90% |>=20@85% |>=20@90% | SPC,s |  Time   |
-----------------------------------------------------------------------------------------------
|  NoRefine   |   DAVIS   |  2.84   |  3.78   |  4.76   |   30    |   51    | 0.143 | 0:03:54 |
|  NoRefine   |  D585_SP  |  1.38   |  1.60   |  2.23   |   13    |   21    | 0.170 | 0:03:41 |
|  NoRefine   | D585_ZERO |  3.10   |  3.93   |  5.86   |   45    |   88    | 0.109 | 0:06:15 |






'''