import torch.nn as nn
import torch
import torch.nn.functional as F
from isegm.utils.serialization import serialize
from .is_model import ISModel
from isegm.model.ops import DistMaps, ScaleLayer, BatchImageNormalize
from isegm.model.modifiers import LRMult
from .is_segformer_model import XConvBnRelu, ConvModule
from .modeling.segformer.segformer_model import SegFormer
import torchvision.ops.roi_align as roi_align
from isegm.model.ops import DistMaps
from isegm.model.GaussianDistribution2d import GaussianDistribution2d


# import visdom
# viz = visdom.Visdom(env='test')


class SegFormerModelNoRefine(ISModel):
    @serialize
    def __init__(self, feature_stride=4, backbone_lr_mult=0.1,
                 norm_layer=nn.BatchNorm2d, pipeline_version='s1', model_version='b0',
                 **kwargs):
        super().__init__(norm_layer=norm_layer, **kwargs)


        self.pipeline_version = pipeline_version
        self.model_version = model_version
        self.feature_extractor = SegFormer(self.model_version, **kwargs)
        self.feature_extractor.backbone.apply(LRMult(backbone_lr_mult))

        if self.pipeline_version == 's1':
            base_radius = 3
        else:
            base_radius = 5

        self.dist_maps_base = DistMaps(norm_radius=base_radius, spatial_scale=1.0,
                                       cpu_mode=False, use_disks=True)

    def get_coord_features(self, image, prev_mask, points):
        coord_features = self.dist_maps_base(image, points)
        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)
        return coord_features

    def backbone_forward(self, image, side_feature, *args):
        outs = self.feature_extractor(image, side_feature, *args)
        outs['instances'] = outs['pred']
        outs['instances_aux'] = outs['pred']
        return outs

    def get_click_gaussian_distribution(self, image, prev_mask, points):
        coord_features = self.gaussian_distribution(image, prev_mask, points)

        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)

        return coord_features

    def forward(self, image, points):

        image, prev_mask = self.prepare_input(image)
        coord_features = self.get_coord_features(image, prev_mask, points)
        click_map = coord_features[:, 1:, :, :]
        #
        # viz.images(click_map[:5, :1, :, :], win='pos', opts={"title": 'pos'})
        # viz.images(click_map[:5, 1:, :, :], win='neg', opts={"title": 'neg'})

        if self.pipeline_version == 's1':
            small_image = F.interpolate(image, scale_factor=0.5, mode='bilinear', align_corners=True)
            small_coord_features = F.interpolate(coord_features, scale_factor=0.5, mode='bilinear', align_corners=True)
            points = torch.div(points, 2, rounding_mode='floor')  # 修改以适应计算Patch出错的问题
        else:
            small_image = image
            small_coord_features = coord_features

        # small_coord_features = self.maps_transform(small_coord_features)
        outputs = self.backbone_forward(small_image, small_coord_features, points)

        outputs['click_map'] = click_map
        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)
        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                                 mode='bilinear', align_corners=True)

        return outputs

    def load_pretrained_weights(self, path_to_weights= ' '):
        backbone_state_dict = self.state_dict()
        pretrained_state_dict = torch.load(path_to_weights, map_location='cpu')['state_dict']
        ckpt_keys = set(pretrained_state_dict.keys())
        own_keys = set(backbone_state_dict.keys())
        missing_keys = own_keys - ckpt_keys
        unexpected_keys = ckpt_keys - own_keys
        print('Missing Keys: ', missing_keys)
        print('Unexpected Keys: ', unexpected_keys)
        backbone_state_dict.update(pretrained_state_dict)
        self.load_state_dict(backbone_state_dict, strict= False)

