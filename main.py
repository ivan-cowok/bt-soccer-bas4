#!/usr/bin/env python3
"""
File containing the main training script for T-DEED.
"""

# Standard imports (do not force HF_HUB_OFFLINE here: it breaks timm ImageNet
# weights unless they are already cached; set HF_HUB_OFFLINE=1 yourself if needed.)
import os
import torch
import numpy as np
import random
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import time

#Local imports
from util.io import load_json, store_json
from dataset.datasets import get_datasets
from dataset.frame import ActionSpotVideoDataset
from util.constants import LABELS_SNB_PATH, STRIDE, STRIDE_SNB, EVAL_SPLITS
from util.eval import evaluate, evaluate_SNB
from model.model import AdaSpot
from model.impl.sam import build_sam

def get_args():
    #Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='soccernetball')
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

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    
    sched1 = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                    total_iters=args.warm_up_epochs * num_steps_per_epoch)
    sched2 = CosineAnnealingLR(optimizer,
                    num_steps_per_epoch * cosine_epochs)
    return args.num_epochs, SequentialLR(optimizer, schedulers=[sched1, sched2],
                    milestones=[args.warm_up_epochs * num_steps_per_epoch])

def check_model_dims(data_args):
    """
    Ensure that model input dimensions are in the correct format (list of 2 ints)
    """
    hr_dim = data_args.hr_dim
    lr_dim = data_args.lr_dim
    hr_crop = data_args.hr_crop
    lr_crop = data_args.lr_crop

    for dim in [hr_dim, lr_dim, hr_crop, lr_crop]:
        if isinstance(dim, list):
            if len(dim) != 2:
                raise ValueError('Dimensions must be a list of 2 ints')
            if not all(isinstance(x, int) for x in dim):
                raise ValueError('Dimensions must be a list of 2 ints')
        else:
            raise ValueError('Dimensions must be a list of 2 ints')
    
    return

# Module-level epoch counter so worker_init_fn is picklable on Windows (spawn).
_current_epoch = 0

def worker_init_fn(worker_id):
    random.seed(worker_id + _current_epoch * 100)

def main(args):
    
    #Set seed
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


    config_path = args.model_name.split('_')[0] + '/' + args.model_name + '.json'
    config = load_json(os.path.join('config', config_path))
    args = update_args(args, config)

    # timm RegNet ImageNet weights come from Hugging Face; many networks block that.
    # Opt-in only: set "pretrained_backbone": true under "model" in the JSON to download.
    if 'pretrained_backbone' in config['model']:
        args.model.pretrained_backbone = bool(config['model']['pretrained_backbone'])
    else:
        args.model.pretrained_backbone = False
        print(
            'model.pretrained_backbone not in config: using False (no ImageNet download). '
            'Add "pretrained_backbone": true under "model" to enable timm/HF backbone weights.'
        )

    _offline = os.environ.get('HF_HUB_OFFLINE', '').lower() in ('1', 'true', 'yes') or \
        os.environ.get('TRANSFORMERS_OFFLINE', '').lower() in ('1', 'true', 'yes')
    if _offline:
        if getattr(args.model, 'pretrained_backbone', False):
            print(
                'Warning: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE is set; forcing pretrained_backbone=False.'
            )
        args.model.pretrained_backbone = False

    print(
        f'[backbone] feature_arch={args.model.feature_arch} '
        f'pretrained_backbone={bool(getattr(args.model, "pretrained_backbone", False))}'
    )

    # Check labels path for SN-BAS
    if args.data.dataset == 'soccernetball':
        if not os.path.exists(LABELS_SNB_PATH): # check that the path exists
            raise ValueError('Labels path for SN-BAS does not exist. Please update the LABELS_SNB_PATH constant in util/constants.py with the correct path to the labels file for SN-BAS.')

    # Check dimensions
    check_model_dims(args.model)

    # Get classes + train, val, datasets (+ val frames for map evaluation) + elements (for F3Set if necessary)
    classes, train_data, val_data, val_data_frames, elements = get_datasets(args.data, only_test = args.training.only_test)

    # Foreground class count must match the prediction head (class list / optional mask).
    n_fg = len(classes)
    if args.model.num_classes != n_fg:
        print(
            f'Note: setting model.num_classes from {args.model.num_classes} to {n_fg} '
            'to match the loaded class list (see class.txt and optional active_class_names).'
        )
        args.model.num_classes = n_fg

    # PyTorch: prefetch_factor only valid when num_workers > 0
    nw = args.training.num_workers
    prefetch = 1 if nw > 0 else None

    # Dataloaders
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=args.training.batch_size, # It is already random in get one
        pin_memory=True, num_workers=nw,
        prefetch_factor=prefetch, worker_init_fn=worker_init_fn)

    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=args.training.batch_size,
        pin_memory=True, num_workers=nw,
        prefetch_factor=prefetch, worker_init_fn=worker_init_fn)
                
    # Model
    model = AdaSpot(args_model=args.model, args_training=args.training, classes=classes, elements=elements)

    # Load pretrained weights from another checkpoint (e.g. SoccerNet Ball) if specified.
    # Keys with mismatched shapes (e.g. final FC head with different num_classes) are skipped.
    init_ckpt = getattr(args.model, 'init_checkpoint', None)
    if init_ckpt:
        if not os.path.isabs(init_ckpt):
            init_ckpt = os.path.join(os.getcwd(), init_ckpt)
        print(f'Loading init weights from: {init_ckpt}')
        src_state = torch.load(init_ckpt, map_location='cpu')
        tgt_state = model.state_dict()
        # Keep only keys that exist in the target and have the same shape
        filtered = {k: v for k, v in src_state.items()
                    if k in tgt_state and tgt_state[k].shape == v.shape}
        skipped = [k for k in src_state if k not in filtered]
        tgt_state.update(filtered)
        model.load(tgt_state, strict=False)
        print(f'  Loaded {len(filtered)} tensors, skipped {len(skipped)} (shape mismatch or new keys)')
        if skipped:
            print('  Skipped keys: ' + ', '.join(skipped[:10]) + ('...' if len(skipped) > 10 else ''))

    # Optimizer and scaler
    optimizer, scaler = model.get_optimizer({'lr': args.training.learning_rate})

    # Optional SAM / ASAM wrapper (no-op when training.sam is 'none' or absent).
    # ASAM is recommended for small datasets with class imbalance; it doubles
    # the per-step compute (two forward+backward passes).
    sam_mode = str(getattr(args.training, 'sam', 'none')).lower()
    sam_rho = float(getattr(args.training, 'sam_rho', 0.5))
    sam = build_sam(model._get_params(), optimizer, mode=sam_mode, rho=sam_rho)
    if sam is not None:
        adaptive_str = 'ASAM' if sam_mode == 'asam' else 'SAM'
        print(
            f'[optim] {adaptive_str} enabled (rho={sam_rho}). '
            f'Each training step now performs 2 forward+backward passes.'
        )
    else:
        print('[optim] SAM/ASAM disabled (training.sam == "none").')

    # Gradient accumulation: simulate a larger effective batch on a single
    # GPU. ``effective_batch = batch_size * grad_accum_steps``. The LR
    # scheduler must count *optimizer* steps, not micro-batches, so we
    # divide ``len(train_loader)`` by the accumulation count below.
    grad_accum_steps = max(1, int(getattr(args.training, 'grad_accum_steps', 1)))
    if grad_accum_steps > 1:
        eff_batch = args.training.batch_size * grad_accum_steps
        print(
            f'[optim] Gradient accumulation enabled: '
            f'micro_batch={args.training.batch_size} x '
            f'grad_accum_steps={grad_accum_steps} '
            f'-> effective batch size = {eff_batch}'
        )
        if sam is not None:
            print(
                '[optim] Note: with SAM/ASAM + grad accum, each micro-batch '
                'still does its own first/second pass (per-micro-batch '
                'perturbation); the sharp gradient is averaged across '
                f'{grad_accum_steps} micro-batches before the optimizer step.'
            )
    else:
        print('[optim] Gradient accumulation disabled (grad_accum_steps=1).')

    # Training loop
    if not args.training.only_test:

        # Warmup schedule. ``num_steps_per_epoch`` is the number of
        # *optimizer* steps per epoch, which is what the LR scheduler
        # ticks on. With grad accum, that is fewer than the number of
        # micro-batches in the loader. Use ceiling division so the
        # residual step at the end of each epoch (when len(loader) is
        # not divisible by grad_accum_steps) is also accounted for.
        num_steps_per_epoch = max(
            1,
            (len(train_loader) + grad_accum_steps - 1) // grad_accum_steps,
        )
        num_epochs, lr_scheduler = get_lr_scheduler(
            args.training, optimizer, num_steps_per_epoch)
        
        losses = []
        best_criterion = 0 if args.training.criterion == 'map' else float('inf')
        epoch = 0

        print('START TRAINING EPOCHS')
        # Freeze both backbones for the first N epochs so only the temporal
        # model, GRU, and prediction heads adapt to the new class set first.
        freeze_epochs = getattr(args.training, 'freeze_backbone_epochs', 0)
        if freeze_epochs > 0:
            for p in model._model.lowres_backbone.parameters():
                p.requires_grad = False
            for p in model._model.highres_backbone.parameters():
                p.requires_grad = False
            print(f'Backbones frozen for first {freeze_epochs} epoch(s).')

        for epoch in range(epoch, num_epochs):
            global _current_epoch
            _current_epoch = epoch

            # Unfreeze backbones once freeze period ends
            if freeze_epochs > 0 and epoch == freeze_epochs:
                for p in model._model.lowres_backbone.parameters():
                    p.requires_grad = True
                for p in model._model.highres_backbone.parameters():
                    p.requires_grad = True
                print(f'Epoch {epoch}: backbones unfrozen - full fine-tuning.')
            
            # Train epoch
            time_train0 = time.time()
            train_loss = model.epoch(
                train_loader, optimizer, scaler, lr_scheduler=lr_scheduler,
                sam=sam, grad_accum_steps=grad_accum_steps)
            time_train1 = time.time()
            time_train = time_train1 - time_train0
            
            # Val epoch
            time_val0 = time.time()
            val_loss = model.epoch(val_loader)
            time_val1 = time.time()
            time_val = time_val1 - time_val0

            better = False
            val_mAP = 0
            if args.training.criterion == 'loss':
                if val_loss <= best_criterion:
                    best_criterion = val_loss
                    better = True
            elif args.training.criterion == 'map':
                if epoch >= args.training.start_val_epoch:
                    time_map0 = time.time()
                    val_mAP = evaluate(model, val_data_frames, 'VAL', classes,
                                        printed=False, test=False)
                    time_map1 = time.time()
                    time_map = time_map1 - time_map0
                    if val_mAP >= best_criterion:
                        best_criterion = val_mAP
                        better = True
            
            #Printing info epoch
            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}'.format(
                epoch, train_loss, val_loss))
            if (args.training.criterion == 'map') & (epoch >= args.training.start_val_epoch):
                print('Val mAP: {:0.5f}'.format(val_mAP))
                if better:
                    print('New best mAP epoch!')
            print('Time train: ' + str(int(time_train // 60)) + 'min ' + str(np.round(time_train % 60, 2)) + 'sec')
            print('Time val: ' + str(int(time_val // 60)) + 'min ' + str(np.round(time_val % 60, 2)) + 'sec')
            if (args.training.criterion == 'map') & (epoch >= args.training.start_val_epoch):
                print('Time map: ' + str(int(time_map // 60)) + 'min ' + str(np.round(time_map % 60, 2)) + 'sec')
            else:
                time_map = 0

            losses.append({
                'epoch': epoch, 'train': train_loss, 'val': val_loss,
                'val_mAP': val_mAP
            })

            if args.paths.save_dir is not None:

                # Store losses
                os.makedirs(args.paths.save_dir, exist_ok=True)
                store_json(os.path.join(args.paths.save_dir, 'loss.json'), losses, pretty=True)

                # Full weights every epoch (same tensor dict as checkpoint_best.pt)
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        os.getcwd(), args.paths.save_dir,
                        f'checkpoint_epoch_{epoch:04d}.pt'))

                # Best-so-far weights (when criterion improves)
                if better:
                    torch.save(
                        model.state_dict(),
                        os.path.join(os.getcwd(), args.paths.save_dir, 'checkpoint_best.pt'))

    print('START INFERENCE')
    # Load best model
    model.load(torch.load(os.path.join(
        os.getcwd(), args.paths.save_dir, 'checkpoint_best.pt')))
    model.clean_modules() # clean modules to remove unnecessary parameters for inference and speed up evaluation

    eval_splits = EVAL_SPLITS

    for split in eval_splits:
        data_dir = getattr(args.data, 'data_dir', None) or os.path.join('data', args.data.dataset)
        split_path = os.path.join(data_dir, '{}.json'.format(split))

        stride = STRIDE
        if args.data.dataset == 'soccernetball':
            stride = STRIDE_SNB

        if os.path.exists(split_path):
            
            val_dataset_kwargs = {
                'classes': classes, 'frame_dir': args.data.frame_dir, 'clip_len': args.data.clip_len, 'dataset': args.data.dataset, 
                'stride': stride, 'overlap_len': args.data.clip_len // 2
                }
            split_data = ActionSpotVideoDataset(split_path, **val_dataset_kwargs)

            pred_file = None
            if args.paths.save_dir is not None:
                pred_file = os.path.join(
                    args.paths.save_dir, 'pred-{}'.format(split))
            
            mAPs, tolerances = evaluate(model, split_data, split.upper(), classes, pred_file, printed = True, test = True)

            if args.data.dataset == 'soccernetball':
                results = evaluate_SNB(LABELS_SNB_PATH, '/'.join(pred_file.split('/')[:-1]) + '/preds', split = split, metric = 'at1', classes = classes)
                
                print('aMAP@1: ', results['a_mAP'] * 100)
                print('Average mAP per class: ')
                print('-----------------------------------')
                for i in range(len(results["a_mAP_per_class"])):
                    print("    " + list(classes.keys())[i] + ": " + str(np.round(results["a_mAP_per_class"][i] * 100, 2)))

                results_2 = evaluate_SNB(LABELS_SNB_PATH, '/'.join(pred_file.split('/')[:-1]) + '/preds', split = split, metric = 'at2', classes = classes)
                print('aMAP@2: ', results_2['a_mAP'] * 100)
                print('Average mAP@2 per class: ')
                print('-----------------------------------')
                for i in range(len(results_2["a_mAP_per_class"])):
                    print("    " + list(classes.keys())[i] + ": " + str(np.round(results_2["a_mAP_per_class"][i] * 100, 2)))

    print('CORRECTLY FINISHED TRAINING AND INFERENCE')




if __name__ == '__main__':
    main(get_args())