import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple

# import visdom
#
# viz = visdom.Visdom(env='test')

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=1,
        embed_dim=48,
        no_patch_embed_bias=False,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False if no_patch_embed_bias else True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4).contiguous()
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, hidden_features=1536, embed_dim=768, drop=0.):
        super().__init__()

        self.proj = nn.Linear(input_dim, embed_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.act(x)
        x = self.drop(x)
        x = self.proj(x)

        return x.clamp(min=0, max=4)


class GaussianDistribution2d(nn.Module):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 mask_in_chans=1,
                 n_heads=12,
                 point_in_chans=3,
                 embed_dim=48,
                 mlp_dim=192,
                 dropout=0.1,
                 drop_path_rate=0.0,
                 n_layers=6,
                 no_patch_embed_bias=False,
                 n_points=48
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5
        # self.vars_pos = nn.Parameter(torch.ones([24,])*8, requires_grad=True)
        # self.vars_neg = nn.Parameter(torch.ones([24,])*8, requires_grad=True)
        n_patch = (img_size // patch_size) ** 2
        self.n_patch = n_patch
        self.n_points = n_points
        self.proj_linear = nn.Linear(point_in_chans, embed_dim)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=mask_in_chans,
            embed_dim=embed_dim,
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [
                Block(embed_dim, n_heads, mlp_dim, dropout, dpr[i])
                for i in range(n_layers)
            ]
        )

        self.proj_mask = nn.Parameter(self.scale * torch.randn(embed_dim, embed_dim))
        self.proj_points = nn.Parameter(self.scale * torch.randn(embed_dim, embed_dim))

        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.mask_norm = nn.LayerNorm(self.n_points)

        self.mlp = MLP(input_dim=n_patch * n_points, embed_dim=n_points)

        self.var_bias = torch.tensor(4)
        # self.norm = nn.LayerNorm()

    def norm(self, x):
        x = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        return x

    def get_gaussian_distribution_2d(self, points, batchsize, rows, cols, vars, phase='pos'):
        b_sz = points.shape[0]
        num_points = points.shape[1]
        points = points.view(b_sz, -1, points.size(2))
        # points = points.view(-1, points.size(2))
        points, points_order = torch.split(points, [2, 1], dim=2)

        row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
        col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)
        coord_rows, coord_cols = torch.meshgrid(row_array, col_array)

        # var = self.mlp(points)

        if phase == 'pos':
            var = vars[:, :num_points] + self.var_bias
            # var = self.mlp(points)
        else:
            var = vars[:, num_points:] + self.var_bias
        outs = []
        for id_b in range(batchsize):
            point = points[id_b,:,:]
            valid_point = torch.max(point, dim=1, keepdim=False)[0] > 0
            valid_p = point[valid_point]
            valid_num = valid_p.shape[0]
            var_b = var[id_b, :]
            if valid_num > 0:
                coord_x = coord_rows.unsqueeze(0).repeat([valid_num, 1, 1])
                coord_y = coord_cols.unsqueeze(0).repeat([valid_num, 1, 1])
                var_valid = var_b[:valid_p.shape[0]]
                h_out = torch.zeros(rows, cols).to('cuda')
                for id_p in range(valid_num):
                    mu_i = valid_p[id_p, :]
                    dist = (coord_x[id_p, :,: ] - mu_i[0]).pow(2) \
                        + (coord_y[id_p, :,: ] - mu_i[1]).pow(2)
                    h = (1 / (2 * torch.pi * var_valid[id_p].pow(2))) * torch.exp(-dist / (2 * var_valid[id_p].pow(2)))
                    h_out = h_out + h / valid_num
                h_out = self.norm(h_out)
                # viz.images(h_out, win='h', opts={"title": 'h'})
            else:
                h_out = torch.zeros(rows, cols).to('cuda')
            outs.append(h_out.unsqueeze(0).unsqueeze(0))
        outs = torch.vstack(outs)

        return outs


    def forward(self, x, mask, points):
        # points[B,48,3] 上pos 下neg
        # var = self.mlp(coords)

        mask_patch = self.patch_embed(mask)
        mask_embed = mask_patch.flatten(2).transpose(1, 2)

        points = points.float()
        points_embed = self.proj_linear(points)

        mask_embed_L = mask_embed.shape[1]

        feat = torch.cat([mask_embed, points_embed], dim=1)

        for blk in self.blocks:
            feat = blk(feat)
        feat = self.decoder_norm(feat)

        mask_feat, points_feat = feat[:, :mask_embed_L], feat[:, mask_embed_L:]
        mask_feat = mask_feat @ self.proj_mask
        points_feat = points_feat @ self.proj_points

        mask_feat = mask_feat / mask_feat.norm(dim=-1, keepdim=True)
        points_feat = points_feat / points_feat.norm(dim=-1, keepdim=True)

        point_mask = mask_feat @ points_feat.transpose(1, 2)
        point_mask = self.mask_norm(point_mask)

        vars = self.mlp(point_mask)

        pos_points, neg_points = torch.chunk(points, 2, dim=1)

        pos_map = self.get_gaussian_distribution_2d(pos_points, x.shape[0], x.shape[2], x.shape[3], vars, phase='pos')
        # viz.images(pos_map[0, :, :, :], win='pos', opts={"title": 'pos'})
        neg_map = self.get_gaussian_distribution_2d(neg_points, x.shape[0], x.shape[2], x.shape[3], vars, phase='neg')
        # viz.images(neg_map[0, :, :, :], win='neg', opts={"title": 'neg'})
        coord_features = torch.cat([pos_map,neg_map], dim=1)

        return coord_features
#
# if __name__ == '__main__':
#     x = torch.randn([17,3,256,256])
#     point = torch.zeros([17,48,3])
#     point[:, 3:7, :] = 3
#
#     g = GaussianDistribution2d()
#     a = g(x, point)
