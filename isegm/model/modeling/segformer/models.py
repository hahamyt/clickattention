import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
# import xformers.ops as xops


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class NormWithEmbedding(nn.Module):
    def __init__(self,
                 in_channels,
                 use_scale_shift=True):
        super().__init__()
        self.use_scale_shift = use_scale_shift
        self.norm = nn.LayerNorm(in_channels)

        embedding_output = in_channels * 2 if use_scale_shift else in_channels
        self.embedding_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_channels, embedding_output))

        self.norm_x = nn.LayerNorm(in_channels)
        self.norm_scale = nn.LayerNorm(in_channels)
        self.norm_shift = nn.LayerNorm(in_channels)

    def forward(self, x, y, SepNorm=False):
        embedding = self.embedding_layer(y)
        if self.use_scale_shift:
            scale, shift = torch.chunk(embedding, 2, dim=2)
            if SepNorm:
                x = self.norm_x(x) * (1 + self.norm_scale(scale)) + self.norm_shift(shift)
            else:
                x = x * (1 + scale) + shift
                x = self.norm(x)
        else:
            x = self.norm(x + embedding)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 return_attn=False, is_scale=True):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

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

    def forward(self, x, H, W, bias=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn_ = attn.softmax(dim=-1)
        # if bias is not None:
        #     attn_ = torch.mul(attn_, bias.unsqueeze(1))

        attn = self.attn_drop(attn_)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

class AttnWeightBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp_x = Mlp(in_features=in_dim, hidden_features=4*in_dim, out_features=2*in_dim ,drop=0.5)
        self.mlp_z = Mlp(in_features=in_dim, hidden_features=4*in_dim, out_features=2*in_dim ,drop=0.5)
        # self.act = nn.GELU()
        # self.norm = nn.LayerNorm(2*in_dim)
        # self.mlp_s = nn.Linear(2*in_dim, 1)
        # self.weight_drop = nn.Dropout(0.1)
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

    def forward(self, x, idxs):
        W = int(math.sqrt(x.shape[1]))
        x_ = self.mlp_x(x, W, W)             # 96 x 1024 x 64
        x_ = x_.transpose(1, 2)                # 96  x 64 x 1024
        B, n_p = idxs.shape
        res = []
        pos_click_len = []
        for b_z in range(B):
            x_search = x_[b_z:b_z + 1, :, :]          # 1 x 64 x 1024
            x_b = x[b_z:b_z + 1, :, :]          # 1 x 1024 x 32
            idx_pos = idxs[b_z, : n_p // 2][idxs[b_z, : n_p // 2] >= 0].contiguous().type(torch.long)   # 注意这里需要保证大于等于 0
            pos_click_len.append(torch.tensor(len(idx_pos)).view(-1).to(x.device))
            if len(idx_pos) <= 1:       # 点击次数小于等于一次
                res.append(torch.ones(x.shape[1]).to(x.device)+ 0.0 * torch.sum(x_))
                continue

            x_pos = x_b[:, idx_pos, :].mean(dim=1, keepdim=True)        # 1 x 1 x 32
            x_pos = self.mlp_z(x_pos, 1, 1).transpose(1, 2)
            # 第一个参数维度： (1, 128, 4096),，第二个参数维度：（1, 128, 1）
            res_pos = F.conv1d(x_search, x_pos)              # 1 x 1 x 4096
            res_pos = F.layer_norm(res_pos, normalized_shape=res_pos.shape)
            res_pos = F.sigmoid(res_pos)
            # res_pos = (res_pos-res_pos.min())/(res_pos.max()-res_pos.min())
            res.append(res_pos.squeeze())

        res = torch.vstack(res).unsqueeze(-1)
        pos_click_len = torch.vstack(pos_click_len)
        # res = self.weight_drop(res)
        return {'res':res, 'pos_click_len':pos_click_len}
        # W = int(math.sqrt(x.shape[1]))
        # x_ = self.mlp_x(x, W, W)
        # B, n_p = idxs.shape     # 这里的np不准
        #
        # # 从x 中切出查询样本（注意不是 x_）
        # x_pos = [];none_b = []
        # for b_z in range(B):
        #     x_b = x[None, b_z, :, :]
        #     idx_pos = idxs[b_z, : n_p//2][idxs[b_z, : n_p//2]>=0].contiguous().type(torch.long)     # 64(256个) -> 87; 128（1024个） -> 171; 256 -> 339; 384 -> 507
        #
        #     if len(idx_pos) == 0:
        #         return None
        #     x_pos.append(x_b[:, idx_pos, :].mean(dim=1, keepdim=True))
        #
        # x_pos = torch.vstack(x_pos)
        # x_pos = self.mlp_z(x_pos, 1, 1)
        # x_ = x_ + x_pos
        # x_ = self.norm(x_)
        # x_ = self.act(x_)
        # res = torch.sigmoid(self.mlp_s(x_))
        # return res

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, use_weight=False,  **kwargs,):
        super().__init__()

        self._kwargs = kwargs
        self.use_weight = use_weight

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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


    def forward(self, x, H, W, w = None, use_weight=False):
        if use_weight:
            x_, attn = self.attn(self.norm1(x), H, W, bias=w)
            x = x + self.drop_path(x_)
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
            return {'x':x, 'attn':attn}
        else:
            x_ , attn= self.attn(self.norm1(x), H, W)
            x = x + self.drop_path(x_)
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
            return {'x': x, 'attn': attn}

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

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

    def forward(self, x):
        #x = self.dual_norm(x)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous() # n c h w -> n c h*w -> n h*w c
        x = self.norm(x)
        return x, H, W

