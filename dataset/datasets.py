# Standard imports
import os

# Local imports
from util.dataset import load_classes, load_elements
from util.constants import STRIDE, STRIDE_SNB, OVERLAP
from dataset.frame import ActionSpotDataset, ActionSpotVideoDataset


def get_datasets(args, only_test = False):
    active = getattr(args, 'active_class_names', None)
    if active is not None and len(active) == 0:
        active = None
    class_path = os.path.join('data', args.dataset, 'class.txt')
    classes = load_classes(class_path, active_class_names=active)
    if active is not None:
        print(
            f'Using active_class_names subset ({len(classes)} foreground): ',
            list(classes.keys()),
        )
    elements = None
    if args.dataset == 'f3set':
        elements = load_elements(os.path.join('data', args.dataset, 'elements.txt'))

    if only_test:
        return classes, None, None, None, elements

    dataset_len = args.epoch_num_frames // args.clip_len
    val_frames = getattr(args, 'val_epoch_num_frames', None)
    if val_frames is not None:
        val_dataset_len = int(val_frames) // args.clip_len
    else:
        val_dataset_len = dataset_len

    if args.dataset == 'soccernetball':
        stride = STRIDE_SNB
    else:
        stride = STRIDE
    overlap = OVERLAP

    dataset_kwargs = {
        'classes': classes, 'frame_dir': args.frame_dir, 'store_dir': args.store_dir, 'store_mode': args.store_mode,
        'clip_len': args.clip_len, 'dataset_len': dataset_len, 'dataset': args.dataset, 'stride': stride,
        'overlap': overlap, 'mixup': args.mixup, 'elements': elements
    }

    print('Dataset size (train):', dataset_len)
    print('Dataset size (val):', val_dataset_len)

    train_data = ActionSpotDataset(
        os.path.join('data', args.dataset, 'train.json'), **dataset_kwargs)
    train_data.print_info()

    dataset_kwargs['mixup'] = False  # Disable mixup for validation
    dataset_kwargs['dataset_len'] = val_dataset_len

    val_data = ActionSpotDataset(
        os.path.join('data', args.dataset, 'val.json'), **dataset_kwargs)
    val_data.print_info()

    val_dataset_kwargs = {
        'classes': classes, 'frame_dir': args.frame_dir, 'clip_len': args.clip_len, 'dataset': args.dataset, 
        'stride': stride, 'overlap_len': 0
    }
    val_data_frames = ActionSpotVideoDataset(
        os.path.join('data', args.dataset, 'val.json'), **val_dataset_kwargs)
    val_data_frames.print_info()
        
    return classes, train_data, val_data, val_data_frames, elements