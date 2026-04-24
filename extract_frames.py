"""
Extract frames from all soccer clips for AdaSpot training.

Source videos : E:\Database\44\soccer_data\{video}\224.mp4
Output frames : E:\Database\44\soccer_data_frames\{video}\frame{N}.jpg
                (0-indexed, 25 fps, quality ~95%)

Default: if config/MyLeague/MyLeague_finetune.json exists next to this script,
frames are scaled to model hr_dim (e.g. 448x796) to match training — not full HD.

    python extract_frames.py                 # uses MyLeague_finetune.json hr_dim
    python extract_frames.py --native      # full native resolution (1920x1080 etc.)
    python extract_frames.py --hr_dim 448 796
    python extract_frames.py --config path/to/other.json

Re-extracting at a new size: do not use --skip_done (or delete old frame folders first).

(hr_dim in JSON is [height, width]; ffmpeg uses scale=width:height.)

Run:
    python extract_frames.py --skip_done     # skip clips that already have frame0.jpg
"""
import os
import json
import argparse
import subprocess
import sys
import shutil
from pathlib import Path


def resolve_ffmpeg(explicit):
    """
    Find ffmpeg. IDEs often launch Python without the user PATH where WinGet
    installs ffmpeg, so we probe common Windows locations after shutil.which.
    """
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return str(p.resolve())
        w = shutil.which(explicit)
        if w:
            return w

    w = shutil.which('ffmpeg')
    if w:
        return w

    candidates: list[Path] = []
    if sys.platform == 'win32':
        local = os.environ.get('LOCALAPPDATA', '')
        if local:
            pkgs = Path(local) / 'Microsoft' / 'WinGet' / 'Packages'
            if pkgs.is_dir():
                for gyan in pkgs.glob('Gyan.FFmpeg_*'):
                    for exe in gyan.glob('ffmpeg-*/bin/ffmpeg.exe'):
                        candidates.append(exe)
        program_files = os.environ.get('ProgramFiles', r'C:\Program Files')
        candidates.extend([
            Path(program_files) / 'ffmpeg' / 'bin' / 'ffmpeg.exe',
            Path(r'C:\ffmpeg\bin\ffmpeg.exe'),
        ])

    for c in candidates:
        try:
            if c.is_file():
                return str(c.resolve())
        except OSError:
            continue
    return None

VIDEO_ROOT  = Path(r"/workspace/44/data/soccer_data")
FRAMES_ROOT = Path(r"/workspace/44/data/soccer_data_frames")
DATA_DIR    = Path(r"/workspace/44/bt-soccer-bas4/data/soccernetball")

#VIDEO_ROOT  = Path(r"E:/Database/44/soccer_data")
#FRAMES_ROOT = Path(r"E:/Database/44/soccer_data_frames")
#DATA_DIR    = Path(r"data/soccernetball")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MY_LEAGUE_CONFIG = SCRIPT_DIR / 'config' / 'soccernetball' / 'soccernetball.json'
FPS = 25


def load_hr_dim_from_config(config_path: Path):
    """Return (H, W) from model.hr_dim, or None if missing."""
    if not config_path.is_file():
        return None
    with open(config_path, encoding='utf-8') as f:
        cfg = json.load(f)
    hr = cfg.get('model', {}).get('hr_dim')
    if not hr or len(hr) != 2:
        return None
    return int(hr[0]), int(hr[1])


def build_vf(hr_hw=None):
    """
    ffmpeg -vf chain: fps + optional scale.
    hr_hw: (H, W) as in config / torchvision; ffmpeg scale is W:H.
    """
    parts = [f'fps={FPS}']
    if hr_hw is not None:
        H, W = hr_hw
        parts.append(f'scale={W}:{H}:flags=lanczos')
    return ','.join(parts)

def all_videos():
    """Return sorted unique list of all video paths from train + val json."""
    videos = set()
    for fname in ('train.json', 'val.json'):
        with open(DATA_DIR / fname, encoding='utf-8') as f:
            for entry in json.load(f):
                videos.add(entry['video'])
    return sorted(videos)

def frames_done(out_dir: Path) -> bool:
    """Return True if at least frame0.jpg already exists."""
    return (out_dir / 'frame0.jpg').exists()

def extract(video_rel: str, skip_done: bool, ffmpeg: str, vf: str) -> bool:
    """Extract frames for one clip. Returns True on success."""
    src = VIDEO_ROOT / video_rel / '224p.mp4'
    out_dir = FRAMES_ROOT / video_rel

    if not src.exists():
        print(f"  [SKIP] source not found: {src}")
        return False

    if skip_done and frames_done(out_dir):
        print(f"  [DONE] {video_rel}")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(out_dir / 'frame%d.jpg')

    cmd = [
        ffmpeg,
        '-y',                        # overwrite
        '-i', str(src),
        '-vf', vf,
        '-start_number', '0',        # 0-indexed: frame0.jpg, frame1.jpg, ...
        '-q:v', '2',                 # JPEG quality (1=best, 31=worst; 2≈95%)
        out_pattern,
    ]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        err = result.stderr.decode('utf-8', errors='replace')[-500:]
        print(f"  [ERROR] {video_rel}\n    {err}")
        return False

    n_frames = len(list(out_dir.glob('frame*.jpg')))
    print(f"  [OK]   {video_rel}  ({n_frames} frames)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_done', action='store_true',
                        help='Skip clips where frame0.jpg already exists')
    parser.add_argument('--native', action='store_true',
                        help='Full video resolution (no scale). Overrides default hr_dim from config.')
    parser.add_argument('--ffmpeg', default=None,
                        help='Path to ffmpeg.exe (optional; auto-detected if omitted)')
    parser.add_argument('--config', type=str, default='config/soccernetball/soccernetball.json',
                        help='AdaSpot JSON (reads model.hr_dim). Default: MyLeague/MyLeague_finetune.json')
    parser.add_argument('--hr_dim', type=int, nargs=2, metavar=('H', 'W'), default=None,
                        help='Resize to H x W (model hr_dim order). Overrides --config.')
    args = parser.parse_args()

    hr_hw = None
    if args.native:
        hr_hw = None
    elif args.hr_dim is not None:
        hr_hw = (args.hr_dim[0], args.hr_dim[1])
    elif args.config:
        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = (SCRIPT_DIR / cfg_path).resolve()
        loaded = load_hr_dim_from_config(cfg_path)
        if loaded:
            hr_hw = loaded
        else:
            print(f"WARNING: could not read model.hr_dim from {cfg_path}; using native resolution.")
    else:
        # Default: My League finetune config -> hr_dim (448x796), not full HD
        if DEFAULT_MY_LEAGUE_CONFIG.is_file():
            loaded = load_hr_dim_from_config(DEFAULT_MY_LEAGUE_CONFIG)
            if loaded:
                hr_hw = loaded
                print(f"Default: using hr_dim from {DEFAULT_MY_LEAGUE_CONFIG}")
        if hr_hw is None:
            print("No hr_dim config found; extracting at native resolution. "
                  "Use --config or place config/MyLeague/MyLeague_finetune.json")

    vf = build_vf(hr_hw)
    if hr_hw:
        print(f"Resizing to hr_dim HxW = {hr_hw[0]}x{hr_hw[1]} (ffmpeg scale {hr_hw[1]}:{hr_hw[0]})")
    else:
        print("Extracting at native resolution (no scale).")

    ffmpeg_exe = resolve_ffmpeg(args.ffmpeg)
    if not ffmpeg_exe:
        print("ERROR: ffmpeg not found.")
        print("  Install with: winget install Gyan.FFmpeg")
        print("  Or pass: --ffmpeg \"C:\\path\\to\\ffmpeg.exe\"")
        sys.exit(1)

    try:
        subprocess.run([ffmpeg_exe, '-version'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        print(f"ERROR: ffmpeg at '{ffmpeg_exe}' failed to run.")
        sys.exit(1)

    print(f"Using ffmpeg: {ffmpeg_exe}")
    print(f"Video filter: {vf}\n")

    videos = all_videos()
    print(f"Total clips to process: {len(videos)}")
    print(f"Output root: {FRAMES_ROOT}\n")

    ok = fail = skip = 0
    for i, v in enumerate(videos, 1):
        print(f"[{i:3}/{len(videos)}] {v}")
        result = extract(v, args.skip_done, ffmpeg_exe, vf)
        if result is True:
            ok += 1
        elif result is False:
            # check if it was a skip due to already done
            out_dir = FRAMES_ROOT / v
            if args.skip_done and frames_done(out_dir):
                skip += 1
            else:
                fail += 1

    print(f"\nDone.  OK={ok}  Skipped={skip}  Failed={fail}")


if __name__ == '__main__':
    main()
