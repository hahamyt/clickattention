# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# from visualizer import get_local
# get_local.activate()

from isegm.model.modeling.segformer.models import Block, OverlapPatchEmbed, AttnWeightBlock
import math

# import visdom
# viz = visdom.Visdom()

# from isegm.model.modeling.unet2d.unet2d_cond_blocks import get_down_block

class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], side_dims=3,
                 **kwargs,
                 ):
        super().__init__()

        self.attn_weight = kwargs['use_attn_weight']

        self._kwargs = kwargs

        self.num_classes = num_classes
        self.depths = depths
        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        self.patch_embed_side1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=side_dims,
                                                   embed_dim=embed_dims[0])

        # transformer encoder
        if self.attn_weight[0]:
            self.attn_weight1 = AttnWeightBlock(embed_dims[0])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(   
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0], use_weight=self.attn_weight[0],
            **kwargs,)
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        if self.attn_weight[1]:
            self.attn_weight2 = AttnWeightBlock(embed_dims[1])
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1],use_weight=self.attn_weight[1],
            **kwargs,)
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        if self.attn_weight[2]:
            self.attn_weight3 = AttnWeightBlock(embed_dims[2])
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2],use_weight=self.attn_weight[2],
            **kwargs,)
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        if self.attn_weight[3]:
            self.attn_weight4 = AttnWeightBlock(embed_dims[3])
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3],use_weight=self.attn_weight[3],
            **kwargs,)
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            # logger = get_root_logger()
            # todo
            pass
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def pt2patchidx(self, points, im_sz, patch_sz=4):
        idxs_list = []
        for i in range(4):
            stage_patch_sz = patch_sz * (2 ** i)
            patchs = torch.div(points, stage_patch_sz, rounding_mode='floor')
            idxs = patchs[:, :, 0] * (im_sz // stage_patch_sz) + patchs[:, :, 1]
            idxs = idxs.int()
            idxs_list.append(idxs)

        return idxs_list

    def forward_features(self, x, additional_feature, points):
        B = x.shape[0]
        idxs_list = self.pt2patchidx(points, x.shape[2])

        outs = []
        attns = []; attns_tmp = []         # 用于存放注意力响应，计算Affinity 损失
        attnW = []                               # 用于存放特征权重

        # stage 1
        x, H, W = self.patch_embed1(x)         # B, 4096, 64
        if additional_feature is not None:
            af, H, W = self.patch_embed_side1(additional_feature)
            x = x + af
        weights = self.attn_weight1(x, idxs_list[0])['res'] if self.attn_weight[0] else None
        for i, blk in enumerate(self.block1):
            x_dict = blk(x, H, W, weights, self.attn_weight[0])
            x = x_dict['x']
            attns_tmp.append(x_dict['attn'])
        if self.attn_weight[0] and weights is not None:  # weights 为None的情形只在评估时会发生，因此不会影响训练
            attnW.append(weights.reshape(B, 1, W, W))
        x = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        attns.append(attns_tmp);attns_tmp = []
        # viz.heatmap(attnW[0][0, :, :, :].squeeze(), win='attnW0')

        # stage 2
        x, H, W = self.patch_embed2(x)          # B, 1024, 128
        weights = self.attn_weight2(x, idxs_list[1])['res'] if self.attn_weight[1] else None
        for i, blk in enumerate(self.block2):
            x_dict = blk(x, H, W, weights, self.attn_weight[1])
            x = x_dict['x']
            attns_tmp.append(x_dict['attn'])
        if self.attn_weight[1] and weights is not None:  # weights 为None的情形只在评估时会发生，因此不会影响训练
            attnW.append(weights.reshape(B, 1, W, W))
        x = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        attns.append(attns_tmp);attns_tmp = []
        # viz.heatmap(attnW[1][0, :, :, :].squeeze(), win='attnW1')

        # stage 3
        x, H, W = self.patch_embed3(x)          # B, 256, 320
        weights = self.attn_weight3(x, idxs_list[2])['res'] if self.attn_weight[2] else None
        for i, blk in enumerate(self.block3):
            x_dict = blk(x, H, W, weights, self.attn_weight[2])
            x = x_dict['x']
            attns_tmp.append(x_dict['attn'])
        if self.attn_weight[2] and weights is not None:  # weights 为None的情形只在评估时会发生，因此不会影响训练
            attnW.append(weights.reshape(B, 1, W, W))
        x = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        attns.append(attns_tmp);attns_tmp = []
        # viz.heatmap(attnW[2][0, :, :, :].squeeze(), win='attnW2')

        # stage 4
        x, H, W = self.patch_embed4(x)
        weights = self.attn_weight4(x, idxs_list[3]) if self.attn_weight[3] else None
        pos_click_len = weights['pos_click_len'] if self.attn_weight[3] else None
        weights = weights['res']  if self.attn_weight[3] else None
        for i, blk in enumerate(self.block4):
            x_dict = blk(x, H, W, weights, self.attn_weight[3])
            x = x_dict['x']
            attns_tmp.append(x_dict['attn'])
        if self.attn_weight[3] and weights is not None:  # weights 为None的情形只在评估时会发生，因此不会影响训练
            attnW.append(weights.reshape(B, 1, W, W))
        x = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        attns.append(attns_tmp);attns_tmp = []

        return {'outs': outs, 'attns': attns, 'attnW':attnW, "idxs_list":idxs_list,  "pos_click_len":pos_click_len}

    def forward(self, x, additional_feature, *args):
        x = self.forward_features(x, additional_feature, *args)
        return x

# @BACKBONES.register_module()
class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            # qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[2, 1, 1, 1],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)

# @BACKBONES.register_module()
class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

# @BACKBONES.register_module()
class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

# @BACKBONES.register_module()
class mit_b3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)

# @BACKBONES.register_module()
class mit_b4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)

# @BACKBONES.register_module()
class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, **kwargs)
