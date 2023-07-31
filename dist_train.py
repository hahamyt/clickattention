import datetime
import os
import argparse
import importlib.util
import numpy as np
import random
import torch.multiprocessing as mp
import torch
from isegm.utils.exp import init_experiment
import torch.distributed as dist
import warnings
warnings.filterwarnings("ignore")

devices = '0,1,2,3,4'
world_size = 5
os.environ['CUDA_VISIBLE_DEVICES'] = devices
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29507'
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'      # 可以让报错信息更加准确
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy('file_system')

def main(rank, world_size, cfg, args):
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    model_script = load_module(args.model_path)
    model_script.main(cfg, rank)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default='models/focalclick/segformerB3_S2_cclvs_norefine.py', type=str,
    # parser.add_argument('--model_path', default='models/focalclick/segformerB0_S1_cclvs.py', type=str,
                        help='Path to the model script.')

    parser.add_argument('--model_base_name', default='segformerB3_S2_cclvs_norefine', type=str,)
    # parser.add_argument('--model_base_name', default='segformerB0_S1_cclvs', type=str,)

    parser.add_argument('--exp-name', type=str, default='2chn_out', #  --resume-exp 156_cascade_train --resume-prefix 015 --start-epoch 16
                        help='Here you can specify the name of the experiment. '
                             'It will be added as a suffix to the experiment folder.')

    parser.add_argument('--workers', type=int, default=2,
                        metavar='N',help='Dataloader threads.')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='You can override model batch size by specify positive number.')

    parser.add_argument('--gpus', type=str, default=devices, required=False,
                        help='Ids of used GPUs. You should use either this argument or "--ngpus".')

    parser.add_argument('--resume-exp', type=str, # default='040_16_with_noise',
                        help='The prefix of the name of the experiment to be continued. '
                             'If you use this field, you must specify the "--resume-prefix" argument.')

    parser.add_argument('--resume-prefix', type=str, #default='017',
                        help='The prefix of the name of the checkpoint to be loaded.')

    parser.add_argument('--start-epoch', type=int, default=1,
                        help='The number of the starting epoch from which training will continue. '
                             '(it is important for correct logging and learning rate)')

    parser.add_argument('--weights', type=str, default=None,
                        help='Model weights will be loaded from the specified path if you use this argument.')

    parser.add_argument('--temp-model-path', type=str, default='',
                        help='Do not use this argument (for internal purposes).')

    # 分布式训练部分
    parser.add_argument('--init_method', type=str, default='env://')
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', 0))

    parser.add_argument("--model-ema", action="store_true",default=False,  help="enable tracking Exponential Moving Average of model parameters")
    parser.add_argument("--model-ema-steps", type=int, default=32,
                        help="the number of iterations that controls how often to update the EMA model (default: 32)",)
    parser.add_argument("--model-ema-decay", type=float, default=0.99998,
                        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",)

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
    args = parse_args()

    # model_base_name = getattr(model_script, 'MODEL_NAME', None)

    args.distributed = True
    cfg = init_experiment(args, args.model_base_name)

    mp.spawn(main,
        args=(world_size,cfg, args),
        nprocs=world_size,
        join=True)

