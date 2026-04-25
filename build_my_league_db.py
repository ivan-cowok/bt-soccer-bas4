"""
Build AdaSpot training data for soccernetball dataset.

Reads all Labels-ball.json under E:/Database/44/soccer_data,
performs a 90/10 train/val split (seeded), and writes:
  - data/soccernetball/class.txt
  - data/soccernetball/train.json
  - data/soccernetball/val.json
  - config/soccernetball/soccernetball.json

Config sets epoch_num_frames (train) and val_epoch_num_frames (val) separately so
each val epoch can use fewer random clip steps than train (see dataset/datasets.py).

(No test split / test.json — validation only.)

Also patches util/constants.py with the correct LABELS_SNB_PATH and GAMES_SNB.

Quiet by default (one summary line). Use -v / --verbose for label distribution,
per-class counts, split details, paths, and the full "next steps" banner.

Override per-epoch frame budgets: --train-epoch-frames N --val-epoch-frames M
(steps per epoch ~= N // clip_len with clip_len=100 in the emitted config).

Class masking: add to the emitted JSON under "data" a key
  "active_class_names": ["pass", "pass_received", ...]
with names exactly as in class.txt. Other labels are treated as background.
Set "num_classes" to the length of that list (or rely on training code to sync).
"""

import argparse
import json, os, random
from collections import Counter

# Per-epoch random-clip budgets (steps per epoch ~= value // clip_len; clip_len=100 in config below).
EPOCH_NUM_FRAMES_TRAIN = 200000  # train e.g. 2000 steps if batch_size=1
EPOCH_NUM_FRAMES_VAL = 100000  # val e.g. 1000 steps; omit key in JSON to match train (see datasets.py)

# ── CLI ───────────────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description='Build data/soccernetball from Labels-ball.json files.')
_parser.add_argument(
    '-v', '--verbose',
    action='store_false',
    help='Print label counts, per-class distribution, and full next-steps banner.',
)
_parser.add_argument(
    '--train-epoch-frames',
    type=int,
    default=None,
    help='Frames budget per train epoch (default: EPOCH_NUM_FRAMES_TRAIN). Steps ~= value // clip_len.',
)
_parser.add_argument(
    '--val-epoch-frames',
    type=int,
    default=None,
    help='Frames budget per val epoch (default: EPOCH_NUM_FRAMES_VAL).',
)
_cli = _parser.parse_args()
VERBOSE = _cli.verbose
TRAIN_EPOCH_FRAMES = (
    _cli.train_epoch_frames
    if _cli.train_epoch_frames is not None
    else EPOCH_NUM_FRAMES_TRAIN
)
VAL_EPOCH_FRAMES = (
    _cli.val_epoch_frames if _cli.val_epoch_frames is not None else EPOCH_NUM_FRAMES_VAL
)

# ── Paths ─────────────────────────────────────────────────────────────────────

LABELS_ROOT   = r'/workspace/44/data/soccer_data'
#LABELS_ROOT   = r'E:/Database/44/soccer_data'
ADASPOT_ROOT  = os.path.dirname(os.path.abspath(__file__))
DATA_OUT_DIR  = os.path.join(ADASPOT_ROOT, 'data', 'soccernetball')
CFG_OUT_DIR   = os.path.join(ADASPOT_ROOT, 'config', 'soccernetball')
FRAME_DIR     = r'/workspace/44/data/soccer_data_frames'   # where frames will live
#FRAME_DIR     = r'E:/Database/44/soccer_data_frames'   # where frames will live
SAVE_DIR      = os.path.join(ADASPOT_ROOT, 'checkpoints', 'soccernetball')

FPS           = 25          # assumed frame rate for your clips
BUFFER_SECS   = 2           # extra seconds added after last annotation
SEED          = 42
TRAIN_RATIO   = 0.9

# ── Collect all clips ─────────────────────────────────────────────────────────
# glob does not follow symlinks; use os.walk with followlinks=True instead.
all_jsons = sorted(
    os.path.join(root, fname)
    for root, _dirs, files in os.walk(LABELS_ROOT, followlinks=True)
    for fname in files
    if fname == 'Labels-ball.json'
)
if VERBOSE:
    print(f'Found {len(all_jsons)} Labels-ball.json files')

clips       = []   # list of dicts: {video, num_frames, labels_data}
label_count = Counter()
skipped     = []

for jpath in all_jsons:
    # Relative video path (forward slashes, no trailing slash)
    rel = os.path.relpath(jpath, LABELS_ROOT)           # soccernetball\2026-2027\xxx\Labels-ball.json
    rel = os.path.dirname(rel).replace('\\', '/')        # soccernetball/2026-2027/xxx

    with open(jpath, encoding='utf-8') as f:
        data = json.load(f)
    anns = data.get('annotations', [])
    if not anns:
        skipped.append(rel)
        continue

    # Estimate num_frames from max position in ms
    max_pos_ms  = max(int(a['position']) for a in anns)
    num_frames  = int((max_pos_ms / 1000 + BUFFER_SECS) * FPS)

    for a in anns:
        label_count[a.get('label', '?')] += 1

    clips.append({
        'video':      rel,
        'num_frames': num_frames,
        '_anns':      anns,
    })

if VERBOSE:
    print(f'Usable clips: {len(clips)}  |  Skipped (empty): {len(skipped)}')
    print(f'Total annotations: {sum(label_count.values())}')
    print('Label distribution:')
    for lab, cnt in label_count.most_common():
        print(f'  {lab}: {cnt}')

# ── Class list (sorted by frequency, most common first) ──────────────────────
class_names = [lab for lab, _ in label_count.most_common()]
if VERBOSE:
    print(f'\nClasses ({len(class_names)}): {class_names}')

# ── 90/10 split (seeded) ─────────────────────────────────────────────────────
random.seed(SEED)
shuffled = clips[:]
random.shuffle(shuffled)
n_train   = round(len(shuffled) * TRAIN_RATIO)
train_clips = shuffled[:n_train]
val_clips   = shuffled[n_train:]
if VERBOSE:
    print(f'\nSplit -> train: {len(train_clips)}  val: {len(val_clips)}')

# ── Helper: strip internal field, write JSON ─────────────────────────────────
def to_json_entry(clip):
    return {'video': clip['video'], 'num_frames': clip['num_frames']}

def write_json(path, obj, pretty=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        else:
            json.dump(obj, f, ensure_ascii=False)

# ── Write data/soccernetball/ ────────────────────────────────────────────────────
os.makedirs(DATA_OUT_DIR, exist_ok=True)

# class.txt
class_txt = os.path.join(DATA_OUT_DIR, 'class.txt')
with open(class_txt, 'w', encoding='utf-8') as f:
    for name in class_names:
        f.write(name + '\n')
if VERBOSE:
    print(f'\nWrote {class_txt}')

# train.json / val.json only
write_json(os.path.join(DATA_OUT_DIR, 'train.json'),
           [to_json_entry(c) for c in train_clips])
write_json(os.path.join(DATA_OUT_DIR, 'val.json'),
           [to_json_entry(c) for c in val_clips])
obsolete_test = os.path.join(DATA_OUT_DIR, 'test.json')
if os.path.isfile(obsolete_test):
    os.remove(obsolete_test)
    if VERBOSE:
        print(f'Removed obsolete {obsolete_test}')
if VERBOSE:
    print(f'Wrote train.json ({len(train_clips)} videos)')
    print(f'Wrote val.json  ({len(val_clips)} videos)')

# ── Write config/soccernetball/soccernetball_finetune.json ──────────────────────────────
os.makedirs(CFG_OUT_DIR, exist_ok=True)
config = {
    "paths": {
        "frame_dir": FRAME_DIR,
        "save_dir":  SAVE_DIR
    },
    "data": {
        "dataset":               "soccernetball",
        "data_dir":              "data/soccernetball",
        "num_classes":           len(class_names),
        "clip_len":              100,
        "epoch_num_frames":      TRAIN_EPOCH_FRAMES,
        "val_epoch_num_frames":  VAL_EPOCH_FRAMES,
        "mixup":                 True,
        "store_mode":            "store"
    },
    "training": {
        "batch_size": 2,
        "grad_accum_steps": 4,
        "num_epochs": 40,
        "warm_up_epochs": 3,
        "start_val_epoch": 5,
        "learning_rate": 0.0002,
        "only_test": False,
        "criterion": "loss",
        "classification_loss": "ce",
        "num_workers": 4,
        "lowres_loss": True,
        "highres_loss": True,
        "freeze_backbone_epochs": 3,
        "skip_empty_clips": False,
        "sam": "none",
        "sam_rho": 2.0,
        "softic": False,
        "softic_lambda": 0.001,
        "softic_temperature": 0.1,
        "softic_feat_dim": 128,
        "softic_bank_size": 256,
        "softic_warmup_size": 32
    },

    "model": {
        "hr_dim":             [448, 796],
        "hr_crop":            [448, 796],
        #"hr_dim":             [360, 640],
        #"hr_crop":            [360, 640],
        "lr_dim":             [224, 398],
        "lr_crop":            [224, 398],
        "roi_size":           [112, 112],

        "feature_arch": "rny004_astrm",
        "astrm_reduction": 4,
        "astrm_kernel_size": 3,

        "blocks_temporal":    [True, True, True, True],
        "aggregation":        "max",
        "temporal_arch":      "gru",
        "threshold":          0.0,
        "padding":            "replicate",
        "roi_channel_reduce": "mean_max",
        "roi_spatial_increase": 10,
        "roi_size_step":      28,
        "use_full_hr":        True,
        "use_cbam":           True,
        "pretrained":         True,
        "pretrained_backbone": True,
        #"init_checkpoint":    "config/pretrained/SoccernetBall_Big/checkpoint_best.pt"
    }
}
cfg_path = os.path.join(CFG_OUT_DIR, 'soccernetball.json')
write_json(cfg_path, config, pretty=True)
if VERBOSE:
    print(f'\nWrote {cfg_path}')

# ── Patch util/constants.py ───────────────────────────────────────────────────
train_vids = [c['video'] for c in train_clips]
val_vids   = [c['video'] for c in val_clips]


def _fmt_py_list(videos, indent=8):
    pad = ' ' * indent
    inner = ',\n'.join(f"{pad}    {repr(v)}" for v in videos)
    return f"[\n{inner},\n{pad}]"


constants_path = os.path.join(ADASPOT_ROOT, 'util', 'constants.py')
new_constants = f'''\'\'\'
We define constants here that are used across the codebase.
\'\'\'

LABELS_SNB_PATH = {repr(LABELS_ROOT)}
STRIDE = 1
STRIDE_SNB = 2
EVAL_SPLITS = ['val']
OVERLAP = 0.9
F3SET_ELEMENTS = [2, 3, 3, 3, 7, 8, 2, 4]
DEFAULT_PAD_LEN = 5
FPS_SNB = {FPS}

GAMES_SNB = {{
        'train'     : {_fmt_py_list(train_vids)},
        'val'       : {_fmt_py_list(val_vids)},
        'test'      : [],
        'challenge' : [],
        }}

# Evaluation
TOLERANCES = [0, 1, 2, 4]
TOLERANCES_SNB = [6, 12]
WINDOWS = [1, 2]
WINDOWS_SNB = [6, 12]
INFERENCE_BATCH_SIZE = 4
'''

with open(constants_path, 'w', encoding='utf-8') as f:
    f.write(new_constants)
if VERBOSE:
    print(f'Patched {constants_path}')

# ── Summary (quiet: one line; verbose: full banner) ────────────────────────────
_clip = config['data']['clip_len']
_train_steps = TRAIN_EPOCH_FRAMES // _clip
_val_steps = VAL_EPOCH_FRAMES // _clip
print(
    f'soccernetball: {len(train_clips)} train / {len(val_clips)} val | '
    f'{len(class_names)} classes | train_steps={_train_steps} val_steps={_val_steps} | '
    f'{DATA_OUT_DIR}'
)
if VERBOSE:
    print('\n' + '='*60)
    print('DONE. Next steps:')
    print('='*60)
    print(f'1. Extract video frames to:  {FRAME_DIR}')
    print('   Naming: frame0.jpg, frame1.jpg, ... at 25 fps')
    print('   Folder per clip:  <FRAME_DIR>/<video_path>/')
    print('   (e.g., E:\\Database\\44\\soccer_data_frames\\soccernetball\\2026-2027\\0484fd09bc994720bf73cb35545ea9\\frame0.jpg)')
    print('2. Train:')
    print('   python main.py --model_name soccernetball --seed 1')
    print('   (first run with store_mode="store", then switch to "load")')
    print('3. Inference:')
    print('   python visualize.py --video_path <your_clip.mp4>')
    print('   (update visualize.py CONFIG_PATH/CHECKPOINT_PATH if needed)')
