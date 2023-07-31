from isegm.utils.exp_imports.default import *
MODEL_NAME = 'segformerB3_S2_cclvs_baseline'
import os
from isegm.data.datasets.lvis import LvisDataset
from isegm.data.compose import ComposeDataset,ProportionalComposeDataset
from isegm.data.aligned_augmentation import AlignedAugmentator
from isegm.engine.our_trainer import ISTrainer
# from isegm.engine.cascade_trainer import ISTrainer
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

def main(cfg, rank=0):
    model, model_cfg = init_model(cfg, rank)
    train(model, cfg, model_cfg, rank)

def init_model(cfg, rank):
    # 配置文件
    model_cfg = {
        'crop_size': (256, 256),
        'num_max_points':24,
        'with_prev_mask':True,
        "use_affinity_loss": [False, False, False, False],
        "use_attn_weight": [False, False, False, False],
        'use_pcgrad': False,
        'lr': 1e-3,
        'optim': 'adamw',
        "cross_kvq": 'q' , # 加权后的特征作为kv 还是作为 q
        "trimap":False
    }

    model = SegFormerModelNoRefine(  pipeline_version = 's2', model_version = 'b3',
                       use_leaky_relu=True, use_rgb_conv=False, use_disks=True, norm_radius=5, binary_prev_mask=False,
                       with_aux_output=True, **model_cfg)
    model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.SEGFORMER_B3)
    # model.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.SEGFORMER_B3)

    if cfg.distributed:
        torch.cuda.set_device(rank)     # 解决卡0多出很多 731MB 显存占用的情况
        model = DDP(model.to(rank), device_ids=[rank], output_device=rank,
                    broadcast_buffers=True,
                    find_unused_parameters=False)
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

    # loss_cfg.trimap_loss = nn.BCEWithLogitsLoss() #NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    # loss_cfg.trimap_loss_weight = 1.0

    if (True in model_cfg['use_affinity_loss']):
        loss_cfg.affinity_loss = DiscriminateiveAffinityLoss(class_num=1, loss_index=model_cfg['use_affinity_loss'])
        loss_cfg.affinity_loss_weight = 1.0

    if (True in model_cfg['use_attn_weight']):
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
        'lr': model_cfg['lr'], 'betas': (0.9, 0.999),  'eps': 1e-8, # 'weight_decay':1e-8#''momentum' : 0.9, 'nesterov' : True #
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[160, 210], gamma=0.1)

    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset_cclvs, valset,
                        optimizer=model_cfg['optim'],
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        # checkpoint_interval=[(0, 50), (200, 5)],
                        image_dump_interval=500,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg['num_max_points'],
                        max_num_next_clicks=3,
                        rank=rank)
    trainer.run(num_epochs=230)
'''

Eval results for model: 075  ->  分辨率：512x512
-----------------------------------------------------------------------------------------------
|  Pipeline   |  Dataset  | NoC@80% | NoC@85% | NoC@90% |>=20@85% |>=20@90% | SPC,s |  Time   |
-----------------------------------------------------------------------------------------------
|  NoRefine   |   DAVIS   |  3.15   |  4.38   |  5.39   |   43    |   53    | 0.088 | 0:02:42 |
|  NoRefine   |  D585_SP  |  1.53   |  2.01   |  2.97   |   19    |   33    | 0.102 | 0:02:57 |
|  NoRefine   | D585_ZERO |  4.19   |  5.79   |  8.00   |   90    |   141   | 0.080 | 0:06:16 |


Eval results for model: 075  ->  分辨率：768x768
-----------------------------------------------------------------------------------------------
|  Pipeline   |  Dataset  | NoC@80% | NoC@85% | NoC@90% |>=20@85% |>=20@90% | SPC,s |  Time   |
-----------------------------------------------------------------------------------------------
|  NoRefine   |   DAVIS   |  3.20   |  4.20   |  5.25   |   29    |   51    | 0.181 | 0:05:27 |
|  NoRefine   |  D585_SP  |  1.43   |  2.03   |  2.82   |   12    |   19    | 0.230 | 0:06:19 |
|  NoRefine   | D585_ZERO |  3.82   |  4.74   |  6.81   |   47    |   90    | 0.242 | 0:16:04 |

Eval results for model: 075_5 ->  分辨率：768x768
-----------------------------------------------------------------------------------------------
|  Pipeline   |  Dataset  | NoC@80% | NoC@85% | NoC@90% |>=20@85% |>=20@90% | SPC,s |  Time   |
-----------------------------------------------------------------------------------------------
|  NoRefine   |   DAVIS   |  3.28   |  4.28   |  5.26   |   29    |   51    | 0.318 | 0:09:36 |
|  NoRefine   |  D585_SP  |  1.48   |  1.63   |  2.25   |   14    |   24    | 0.301 | 0:06:36 |
|  NoRefine   | D585_ZERO |  3.96   |  4.85   |  6.91   |   45    |   91    | 0.130 | 0:08:47 |




Eval results for model: 075  ->  分辨率：1024x1024
-----------------------------------------------------------------------------------------------
|  Pipeline   |  Dataset  | NoC@80% | NoC@85% | NoC@90% |>=20@85% |>=20@90% | SPC,s |  Time   |
-----------------------------------------------------------------------------------------------
|  NoRefine   |   DAVIS   |  3.43   |  4.10   |  5.27   |   22    |   49    | 0.212 | 0:06:25 |
|  NoRefine   |  D585_SP  |  1.40   |  2.14   |  2.79   |   10    |   14    | 0.215 | 0:05:51 |
|  NoRefine   | D585_ZERO |  3.88   |  4.62   |  6.36   |   40    |   68    | 0.205 | 0:12:43 |

Eval results for model: 090  ->  分辨率：1024x1024
-----------------------------------------------------------------------------------------------
|  Pipeline   |  Dataset  | NoC@80% | NoC@85% | NoC@90% |>=20@85% |>=20@90% | SPC,s |  Time   |
-----------------------------------------------------------------------------------------------
|  NoRefine   |   DAVIS   |  3.41   |  4.12   |  5.32   |   22    |   49    | 0.215 | 0:06:34 |
|  NoRefine   |  D585_SP  |  1.42   |  2.16   |  2.83   |    8    |   14    | 0.216 | 0:05:57 |
|  NoRefine   | D585_ZERO |  3.88   |  4.66   |  6.31   |   43    |   68    | 0.207 | 0:12:46 |

Eval results for model: 116  ->  分辨率：1024x1024
-----------------------------------------------------------------------------------------------
|  Pipeline   |  Dataset  | NoC@80% | NoC@85% | NoC@90% |>=20@85% |>=20@90% | SPC,s |  Time   |
-----------------------------------------------------------------------------------------------
|  NoRefine   |   DAVIS   |  3.43   |  4.14   |  5.32   |   23    |   49    | 0.211 | 0:06:28 |
|  NoRefine   |  D585_SP  |  1.42   |  2.19   |  2.87   |    9    |   16    | 0.213 | 0:05:57 |
|  NoRefine   | D585_ZERO |  3.89   |  4.66   |  6.30   |   42    |   66    | 0.207 | 0:12:42 |

'''