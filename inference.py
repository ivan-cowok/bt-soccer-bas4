#Global imports
import argparse
import os
import sys
import subprocess
# Disable HF "Xet" download backend (native lib causes 0xC0000005 on Windows
# when timm fetches pretrained weights). Must be set before huggingface_hub
# is imported anywhere downstream.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
import torch
import numpy as np
from torch.utils.data import DataLoader


#Local imports
from util.io import load_json
from model.model import AdaSpot
from util.eval import inference
from dataset.frame import ActionSpotInferenceDataset
from util.dataset import load_classes, load_elements
from util.constants import STRIDE, STRIDE_SNB


def get_args():
    #Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--video_path', type=str, help='Path to video file for inference', required=True)
    parser.add_argument('--inference_threshold', type=float, default=0.2, help='Threshold for inference')
    parser.add_argument('--seed', type=int, default=1)
    
    return parser.parse_args()

def dict_to_namespace(d):
    if isinstance(d, dict):
        return argparse.Namespace(**{
            k: dict_to_namespace(v) for k, v in d.items()
        })
    return d

def update_args(args, config):

    #Update arguments with config file
    args.paths = dict_to_namespace(config['paths'])
    args.data = dict_to_namespace(config['data'])
    args.data.store_dir = args.paths.save_dir + '/store_data'
    args.paths.save_dir = os.path.join(args.paths.save_dir, args.model_name + '-' + str(args.seed)) # allow for multiple seeds
    args.data.frame_dir = args.paths.frame_dir
    args.data.save_dir = args.paths.save_dir
    args.training = dict_to_namespace(config['training'])
    args.model = dict_to_namespace(config['model'])
    args.model.clip_len = args.data.clip_len
    args.model.dataset = args.data.dataset
    args.model.num_classes = args.data.num_classes

    return args

def _can_load_timm_pretrained_backbone(feature_arch):
    base_arch = str(feature_arch).rsplit('_', 1)[0]
    timm_name = {
        'rny002': 'regnety_002',
        'rny004': 'regnety_004',
        'rny006': 'regnety_006',
        'rny008': 'regnety_008',
    }.get(base_arch, None)
    if timm_name is None:
        return True
    code = (
        "import timm; "
        f"timm.create_model('{timm_name}', pretrained=True); "
        "print('ok')"
    )
    try:
        proc = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception as exc:
        print(f'[backbone] pretrained probe failed with exception: {exc!r}')
        return False
    if proc.returncode != 0:
        print(
            '[backbone] pretrained timm probe failed '
            f'(returncode={proc.returncode}); forcing pretrained_backbone=False.'
        )
        return False
    return True


def main(args):

    config_path = args.model_name.split('_')[0] + '/' + args.model_name + '.json'
    config = load_json(os.path.join('config', config_path))
    args = update_args(args, config)

    # Initialize backbone from HF/timm ImageNet weights as a safety net before loading
    # the full AdaSpot checkpoint. Any keys missing or shape-mismatched in the checkpoint
    # then keep their pretrained ImageNet values instead of being random.
    if 'pretrained_backbone' in config['model']:
        args.model.pretrained_backbone = bool(config['model']['pretrained_backbone'])
    else:
        args.model.pretrained_backbone = True
    bb_path = getattr(args.model, 'pretrained_backbone_path', None)
    if bb_path:
        if not os.path.isabs(bb_path):
            bb_path = os.path.join(os.getcwd(), bb_path)
        args.model.pretrained_backbone_path = bb_path
        if not os.path.exists(bb_path):
            raise FileNotFoundError(
                f'model.pretrained_backbone_path does not exist: {bb_path}'
            )
    _offline = os.environ.get('HF_HUB_OFFLINE', '').lower() in ('1', 'true', 'yes') or \
        os.environ.get('TRANSFORMERS_OFFLINE', '').lower() in ('1', 'true', 'yes')
    if _offline:
        if getattr(args.model, 'pretrained_backbone', False):
            print('Warning: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE is set; forcing pretrained_backbone=False.')
        args.model.pretrained_backbone = False
    elif getattr(args.model, 'pretrained_backbone', False) and not bb_path:
        if not _can_load_timm_pretrained_backbone(args.model.feature_arch):
            args.model.pretrained_backbone = False
    print(
        f'[backbone] feature_arch={args.model.feature_arch} '
        f'pretrained_backbone={bool(getattr(args.model, "pretrained_backbone", False))}'
    )

    # Get classes and elements (optional data.active_class_names: subset of class.txt)
    active = getattr(args.data, 'active_class_names', None)
    if active is not None and len(active) == 0:
        active = None
    classes = load_classes(
        os.path.join('data', args.data.dataset, 'class.txt'),
        active_class_names=active,
    )
    if args.data.dataset == 'f3set':
        elements = load_elements(os.path.join('data', args.data.dataset, 'elements.txt'))
    else:
        elements = None

    n_fg = len(classes)
    if args.model.num_classes != n_fg:
        print(
            f'Note: setting model.num_classes from {args.model.num_classes} to {n_fg} '
            'to match the loaded class list.'
        )
        args.model.num_classes = n_fg

    # Model
    model = AdaSpot(args_model=args.model, args_training=args.training, classes=classes, elements=elements)

    print('START INFERENCE')
    # Use strict=False so a checkpoint trained with a different softic
    # configuration (e.g. softic_proj.* present/absent) still loads. Any
    # actually-required tensors that are missing will be obvious from the
    # printed report.
    _ckpt = torch.load(os.path.join(
        os.getcwd(), args.paths.save_dir, 'checkpoint_epoch_0005.pt'),
        map_location='cpu')
    _tgt_keys = set(model.state_dict().keys())
    _src_keys = set(_ckpt.keys())
    _missing = _tgt_keys - _src_keys
    _unexpected = _src_keys - _tgt_keys
    model.load(_ckpt, strict=False)
    if _missing or _unexpected:
        print(
            f'[load] checkpoint partial load: missing={len(_missing)}, '
            f'unexpected={len(_unexpected)}'
        )
        if _missing:
            print('  missing keys (sample):',
                  ', '.join(sorted(_missing)[:6]))
        if _unexpected:
            print('  unexpected keys (sample):',
                  ', '.join(sorted(_unexpected)[:6]))
    model.clean_modules() # clean modules to remove unnecessary parameters for inference and speed up evaluation

    stride = STRIDE
    if args.data.dataset == 'soccernetball':
        stride = STRIDE_SNB

    inf_dataset_kwargs = {
        'clip_len': args.data.clip_len, 'overlap_len': args.data.clip_len // 2, 'stride': stride, 'dataset': args.data.dataset, 'size': (args.model.hr_dim[1], args.model.hr_dim[0])
    }

    inference_dataset = ActionSpotInferenceDataset(args.video_path, **inf_dataset_kwargs)

    inference_loader = DataLoader(
        inference_dataset, batch_size = args.training.batch_size,
        shuffle = False, num_workers = args.training.num_workers,
        pin_memory = True, drop_last = False)

    inference(model, inference_loader, classes, threshold = args.inference_threshold)
    
    print('CORRECTLY FINISHED INFERENCE STEP')


if __name__ == '__main__':
    main(get_args())