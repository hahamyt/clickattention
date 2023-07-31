import torch
from models.focalclick.segformerB3_S2_cclvs_norefine import init_model
from thop import profile

model, model_cfg = init_model(None, None)

input = torch.randn(1, 4, 512, 512)
points = torch.randn(1, 48, 3)

flops, params = profile(model, inputs=(input, points))

print('flops: ', flops, 'params: ', params)
print('flops: %.2f G, params: %.2f M' % (flops / 1000000000.0, params / 1000000.0))


