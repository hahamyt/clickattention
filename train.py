import os
import argparse
import importlib.util
import numpy as np
import random
import cv2
import torch
from isegm.utils.exp import init_experiment

import warnings
warnings.filterwarnings("ignore")
# torch.set_warn_always(False)
# 正向传播时：开启自动求导的异常侦测
# torch.autograd.set_detect_anomaly(True)

os.environ['NUMEXPR_MAX_THREADS'] = '80'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 是否进行参数搜索
search_param = False

def main():
    # seed_torch()
    args = parse_args()
    if search_param:
        args.model_path = 'models/focalclick/segformerB3_S2_cclvs_ray.py'

    if args.temp_model_path:
        model_script = load_module(args.temp_model_path)
    else:
        model_script = load_module(args.model_path)

    args.distributed = 'WORLD_SIZE' in os.environ

    model_base_name = getattr(model_script, 'MODEL_NAME', None) #  'segformerB3_S2_cclvs_ddp'  #

    cfg = init_experiment(args, model_base_name)

    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    model_script.main(cfg)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default='models/focalclick/segformerB3_S2_cclvs_norefine.py', type=str,
    # parser.add_argument('--model_path', default='models/plainvit_huge448_cocolvis.py', type=str,
                        help='Path to the model script.')

    parser.add_argument('--model_base_name', default='segformerB3_S2_cclvs_norefine', type=str, )
    # parser.add_argument('--model_base_name', default='cocolvis_plainvit_huge448', type=str, )

    parser.add_argument('--exp-name', type=str, default='baseline',
                        help='Here you can specify the name of the experiment. '
                             'It will be added as a suffix to the experiment folder.')

    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='Dataloader threads.')

    parser.add_argument('--batch-size', type=int, default=2,
                        help='You can override model batch size by specify positive number.')

    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5', required=False,
                        help='Ids of used GPUs. You should use either this argument or "--ngpus".')

    parser.add_argument('--resume-exp', type=str, default=None, #'001_32-16-8',
                        help='The prefix of the name of the experiment to be continued. '
                             'If you use this field, you must specify the "--resume-prefix" argument.')

    parser.add_argument('--resume-prefix', type=str, default=None,
                        help='The prefix of the name of the checkpoint to be loaded.')

    parser.add_argument('--start-epoch', type=int, default=1,
                        help='The number of the starting epoch from which training will continue. '
                             '(it is important for correct logging and learning rate)')

    parser.add_argument('--epochs', type=int, default=-1,
                        help='You can override model epochs by specify positive number.')

    parser.add_argument('--weights', type=str, default=None,
                        help='Model weights will be loaded from the specified path if you use this argument.')

    parser.add_argument('--temp-model-path', type=str, default='',
                        help='Do not use this argument (for internal purposes).')

    parser.add_argument("--model-ema", action="store_true",default=False,  help="enable tracking Exponential Moving Average of model parameters")
    parser.add_argument("--model-ema-steps", type=int, default=32,
                        help="the number of iterations that controls how often to update the EMA model (default: 32)",)
    parser.add_argument("--model-ema-decay", type=float, default=0.99998,
                        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",)

    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument('--layerwise-decay', action='store_true',
                        help='layer wise decay for transformer blocks.')

    parser.add_argument('--upsample', type=str, default='x1',
                        help='upsample the output.')

    parser.add_argument('--random-split', action='store_true',
                        help='random split the patch instead of window split.')

    return parser.parse_args()

def seed_torch(seed=1029):
    # torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_module(script_path):
    spec = importlib.util.spec_from_file_location("model_script", script_path)
    model_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_script)

    return model_script


if __name__ == '__main__':
    main()

