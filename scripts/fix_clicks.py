import argparse
import torch
from pathlib import Path
import tqdm

from isegm.inference import utils
from isegm.utils.exp import load_config_file
from isegm.inference.evaluation import evaluate_sample

def parse_args():
    parser = argparse.ArgumentParser()

    group_device = parser.add_mutually_exclusive_group()
    group_device.add_argument('--gpus', type=str, default='0,1',
                              help='ID of used GPU.')
    group_device.add_argument('--cpu', action='store_true', default=False,
                              help='Use only CPU for inference.')

    parser.add_argument('--datasets', type=str, default='GrabCut,Berkeley,DAVIS,SBD,PascalVOC',
                        help='List of datasets on which the model should be tested. '
                             'Datasets are separated by a comma. Possible choices: '
                             'GrabCut, Berkeley, DAVIS, SBD, PascalVOC')

    parser.add_argument('--config-path', type=str, default='../config.yml',
                        help='The path to the config file.')

    args = parser.parse_args()
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f"cuda:{args.gpus.split(',')[0]}")

    if (args.iou_analysis or args.print_ious) and args.min_n_clicks <= 1:
        args.target_iou = 1.01
    else:
        args.target_iou = max(0.8, args.target_iou)

    cfg = load_config_file(args.config_path, return_edict=True)
    cfg.EXPS_PATH = Path(cfg.EXPS_PATH)

    return args, cfg


def main():
    args, cfg = parse_args()

    for dataset_name in args.datasets.split(','):
        dataset = utils.get_dataset(dataset_name, cfg)
        for index in tqdm(range(len(dataset)), leave=False):
            sample = dataset.get_sample(index)

            click_list, sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask, sample.init_mask, predictor,
                                            sample_id=index, vis= vis, save_dir = save_dir,
                                            index = index, **kwargs)



if __name__ == '__main__':
    main()
