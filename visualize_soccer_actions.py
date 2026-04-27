#!/usr/bin/env python3
"""
Run AdaSpot (SoccerNet Ball) on a video and write an annotated video with detected actions.

The output video is taller than the input: a top and bottom margin hold all text; the
original frames are copied unchanged in the middle (nothing is drawn over the pitch).

Example:
  python visualize_soccer_actions.py ^
    --video path/to/match.mp4 ^
    --checkpoint "D:/work/44/AdaSpot-main/config/pretrained/SoccernetBall_Big" ^
    --output out_annotated.mp4
"""

import argparse
import os
import sys
import subprocess
# Disable HF "Xet" download backend (native lib causes 0xC0000005 on Windows
# when timm fetches pretrained weights). Must be set before huggingface_hub
# is imported anywhere downstream.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.frame import ActionSpotInferenceDataset
from model.model import AdaSpot
from util.constants import STRIDE_SNB
from util.dataset import load_classes
from util.eval import inference
from util.io import load_json, store_json_inference


def get_args():
    p = argparse.ArgumentParser(description="Visualize SoccerNet Ball action spotting on a video")
    p.add_argument("--video", type=str, default='D:/Data/333.mp4', help="Input video path")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (default: <video_stem>_adaspot.mp4)",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default='checkpoints/Soccernetball/SoccerNetBall-1/checkpoint_best.pt',
        help="Path to checkpoint_best.pt or a folder containing it (default: config/pretrained/SoccernetBall_Big)",
    )
    p.add_argument("--model_name", type=str, default="Soccernetball", help="Config name under config/<Dataset>/")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--inference_threshold", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=4, help="Lower if GPU runs out of memory")
    p.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional folder to save results_inference.json (same format as inference.py)",
    )
    p.add_argument(
        "--text-scale",
        type=float,
        default=1.35,
        help="Multiply all on-screen text size (default 1.35; use 1.8–2.0 if still hard to read)",
    )
    p.add_argument(
        "--fps-override",
        type=float,
        default=None,
        help="FPS for time labels (seconds = frame / FPS). If omitted, uses the file metadata; if missing, 25.",
    )
    return p.parse_args()


def dict_to_namespace(d):
    if isinstance(d, dict):
        return argparse.Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d


def update_args(args, config):
    args.paths = dict_to_namespace(config["paths"])
    args.data = dict_to_namespace(config["data"])
    args.data.store_dir = args.paths.save_dir + "/store_data"
    args.paths.save_dir = os.path.join(args.paths.save_dir, args.model_name + "-" + str(args.seed))
    args.data.frame_dir = args.paths.frame_dir
    args.data.save_dir = args.paths.save_dir
    args.training = dict_to_namespace(config["training"])
    args.model = dict_to_namespace(config["model"])
    args.model.clip_len = args.data.clip_len
    args.model.dataset = args.data.dataset
    args.model.num_classes = args.data.num_classes
    return args


def resolve_checkpoint(checkpoint_arg):
    """Accept .pt file or directory that contains checkpoint_best.pt (recursive)."""
    if checkpoint_arg is None:
        root = os.path.dirname(os.path.abspath(__file__))
        checkpoint_arg = os.path.join(root, "config", "pretrained", "SoccernetBall_Big")
    path = os.path.abspath(checkpoint_arg)
    if os.path.isfile(path) and path.endswith(".pt"):
        return path
    if os.path.isdir(path):
        direct = os.path.join(path, "checkpoint_best.pt")
        if os.path.isfile(direct):
            return direct
        for root, _dirs, files in os.walk(path):
            if "checkpoint_best.pt" in files:
                return os.path.join(root, "checkpoint_best.pt")
    raise FileNotFoundError(
        f"Could not find checkpoint_best.pt under: {checkpoint_arg}"
    )

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
        print(f"[backbone] pretrained probe failed with exception: {exc!r}")
        return False
    if proc.returncode != 0:
        print(
            "[backbone] pretrained timm probe failed "
            f"(returncode={proc.returncode}); forcing pretrained_backbone=False."
        )
        return False
    return True


def class_colors(classes):
    """Stable BGR colors per class index (1..N)."""
    palette = [
        (40, 40, 220),
        (220, 180, 40),
        (40, 220, 220),
        (220, 40, 180),
        (180, 220, 40),
        (40, 180, 220),
        (220, 40, 40),
        (180, 40, 220),
        (220, 220, 40),
        (80, 220, 80),
        (200, 120, 80),
        (100, 200, 200),
    ]
    out = {}
    for name, idx in classes.items():
        out[name] = palette[(idx - 1) % len(palette)]
    return out


def build_timeline(events_result, stride):
    """
    List of (absolute_frame, label, score) sorted by time.
    absolute_frame is the 0-based index in the *decoded* video stream (same as OpenCV read order).
    Model uses stride: predictions exist on frames 0, stride, 2*stride, … so peaks snap to those times.
    """
    out = []
    for e in events_result["events"]:
        abs_f = int(e["frame"]) * stride
        out.append((abs_f, e["label"], float(e["score"])))
    out.sort(key=lambda x: x[0])
    return out


def _margin_heights(fh, fw):
    """Fixed UI strips above/below the video — nothing is drawn on the frame itself."""
    short = min(fh, fw)
    top = max(96, int(short * 0.13))
    bottom = max(128, int(short * 0.20))
    return top, bottom


def _band_font_sizes(fw, fh, top_band, bottom_band, text_scale):
    short = min(fw, fh)
    base = max(0.75, short / 720.0) * text_scale
    return {
        "headline": min(2.4, 1.35 * base * (top_band / 110.0)),
        "meta": min(1.0, 0.72 * base),
        "log": min(1.05, 0.72 * base * (bottom_band / 150.0)),
        "thickness": max(2, int(round(3 * base))),
        "line_step": max(24, int(30 * base)),
    }


def _put_text_strong(img, text, org, font, font_scale, color, thickness):
    """Thick black stroke + color fill (readable on grass, crowds, and gray bars)."""
    x, y = int(org[0]), int(org[1])
    stroke = max(5, thickness + 5)
    cv2.putText(
        img,
        text,
        (x, y),
        font,
        font_scale,
        (0, 0, 0),
        stroke,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def compose_frame_with_margins(
    frame_bgr,
    frame_idx,
    timeline,
    colors,
    top_band,
    bottom_band,
    text_scale=1.0,
    timeline_window=10,
    fps=25.0,
    stride=1,
):
    """
    Build a taller frame: [ top UI band ][ original video, untouched ][ bottom UI band ].
    All text lives only in the margins — never on the pitch or players.
    """
    fh, fw = frame_bgr.shape[:2]
    H = fh + top_band + bottom_band
    W = fw
    canvas = np.full((H, W, 3), 28, dtype=np.uint8)
    canvas[top_band : top_band + fh, 0:fw] = frame_bgr

    cv2.rectangle(canvas, (0, 0), (W - 1, top_band - 1), (70, 70, 70), 1)
    cv2.rectangle(
        canvas, (0, top_band + fh), (W - 1, H - 1), (70, 70, 70), 1
    )

    fs = _band_font_sizes(fw, fh, top_band, bottom_band, text_scale)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fps = max(fps, 1e-6)
    t_now = frame_idx / fps
    meta = f"frame {frame_idx}  |  {t_now:.3f}s  (FPS={fps:.3f})"
    cv2.putText(
        canvas,
        meta,
        (16, 26),
        font,
        fs["meta"],
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    sync_hint = f"Peaks on frames multiple of {stride} (model stride); times = frame / FPS"
    cv2.putText(
        canvas,
        sync_hint,
        (16, 52),
        font,
        fs["meta"] * 0.72,
        (140, 140, 140),
        1,
        cv2.LINE_AA,
    )

    # Current action: only events close in time (shown in top margin)
    nearby = [
        (af, lab, sc)
        for af, lab, sc in timeline
        if abs(frame_idx - af) <= 12
    ]
    if nearby:
        af, lab, sc = min(nearby, key=lambda x: abs(x[0] - frame_idx))
        col = colors.get(lab, (255, 255, 255))
        t_ev = af / fps
        msg = f"{lab}    @ {t_ev:.3f}s (f{af})    score {sc:.2f}"
        y_head = int(top_band * 0.72)
        _put_text_strong(
            canvas, msg, (16, y_head), font, fs["headline"], col, fs["thickness"]
        )
    else:
        hint = "no action peak in this window"
        y_head = int(top_band * 0.72)
        cv2.putText(
            canvas,
            hint,
            (16, y_head),
            font,
            fs["headline"] * 0.45,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )

    line_step = fs["line_step"]
    max_lines = max(1, (bottom_band - 20) // line_step)
    recent = [(af, lab, sc) for af, lab, sc in timeline if af <= frame_idx][
        -min(timeline_window, max_lines) :
    ]

    y_base = top_band + fh + 16
    cv2.putText(
        canvas,
        "recent detections",
        (16, y_base + line_step),
        font,
        fs["log"] * 1.05,
        (210, 210, 210),
        fs["thickness"] - 1,
        cv2.LINE_AA,
    )
    for i, (af, lab, sc) in enumerate(recent):
        col = colors.get(lab, (255, 255, 255))
        t_ev = af / fps
        line = f"{t_ev:.3f}s  f{af}  {lab}  ({sc:.2f})"
        yy = y_base + line_step * (i + 2)
        if yy >= H - 8:
            break
        _put_text_strong(
            canvas, line, (16, yy), font, fs["log"], col, max(1, fs["thickness"] - 1)
        )

    return canvas


def main():
    args = get_args()
    if not os.path.isfile(args.video):
        print("Video not found:", args.video, file=sys.stderr)
        sys.exit(1)

    probe = cv2.VideoCapture(args.video)
    fps_raw = float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
    n_frames_meta = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    probe.release()

    if args.fps_override is not None:
        fps_display = float(args.fps_override)
        print(f"Using --fps-override={fps_display} for time labels (file reported {fps_raw:.4f} FPS).")
    elif fps_raw > 1e-3:
        fps_display = fps_raw
    else:
        fps_display = 25.0
        print(
            "Warning: could not read FPS from video; using 25.0 s for time labels. "
            "Set --fps-override to your true frame rate if times look wrong.",
            file=sys.stderr,
        )

    config_path = args.model_name.split("_")[0] + "/" + args.model_name + ".json"
    config = load_json(os.path.join("config", config_path))
    args_ns = update_args(argparse.Namespace(model_name=args.model_name, seed=args.seed), config)

    # Initialize backbone from HF/timm ImageNet weights as a safety net before loading
    # the full AdaSpot checkpoint. Any tensors missing/shape-mismatched in the checkpoint
    # keep ImageNet values instead of being random. Honors offline env vars.
    if "pretrained_backbone" in config["model"]:
        args_ns.model.pretrained_backbone = bool(config["model"]["pretrained_backbone"])
    else:
        args_ns.model.pretrained_backbone = True
    bb_path = getattr(args_ns.model, "pretrained_backbone_path", None)
    if bb_path:
        if not os.path.isabs(bb_path):
            bb_path = os.path.join(os.getcwd(), bb_path)
        args_ns.model.pretrained_backbone_path = bb_path
        if not os.path.exists(bb_path):
            raise FileNotFoundError(
                f"model.pretrained_backbone_path does not exist: {bb_path}"
            )
    _offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes") or \
        os.environ.get("TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes")
    if _offline:
        if getattr(args_ns.model, "pretrained_backbone", False):
            print("Warning: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE is set; forcing pretrained_backbone=False.")
        args_ns.model.pretrained_backbone = False
    elif getattr(args_ns.model, "pretrained_backbone", False) and not bb_path:
        if not _can_load_timm_pretrained_backbone(args_ns.model.feature_arch):
            args_ns.model.pretrained_backbone = False
    print(
        f"[backbone] feature_arch={args_ns.model.feature_arch} "
        f"pretrained_backbone={bool(getattr(args_ns.model, 'pretrained_backbone', False))}"
    )

    ckpt_path = resolve_checkpoint(args.checkpoint)
    active = getattr(args_ns.data, "active_class_names", None)
    if active is not None and len(active) == 0:
        active = None
    classes = load_classes(
        os.path.join("data", args_ns.data.dataset, "class.txt"),
        active_class_names=active,
    )
    n_fg = len(classes)
    if args_ns.model.num_classes != n_fg:
        print(
            f"Note: setting model.num_classes from {args_ns.model.num_classes} to {n_fg} "
            "to match the loaded class list."
        )
        args_ns.model.num_classes = n_fg
    colors = class_colors(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = AdaSpot(
        device=device,
        args_model=args_ns.model,
        args_training=args_ns.training,
        classes=classes,
        elements=None,
    )
    state = torch.load(ckpt_path, map_location=device)
    # strict=False so checkpoints trained with a different softic config
    # (which adds/removes ``softic_proj.*`` keys) still load. Anything
    # actually-required will show up in the report below.
    _tgt_keys = set(model.state_dict().keys())
    _src_keys = set(state.keys())
    _missing = _tgt_keys - _src_keys
    _unexpected = _src_keys - _tgt_keys
    model.load(state, strict=False)
    if _missing or _unexpected:
        print(
            f"[load] checkpoint partial load: missing={len(_missing)}, "
            f"unexpected={len(_unexpected)}"
        )
        if _missing:
            print("  missing keys (sample):",
                  ", ".join(sorted(_missing)[:6]))
        if _unexpected:
            print("  unexpected keys (sample):",
                  ", ".join(sorted(_unexpected)[:6]))
    model.clean_modules()

    stride = STRIDE_SNB
    inf_dataset_kwargs = {
        "clip_len": args_ns.data.clip_len,
        "overlap_len": args_ns.data.clip_len * 1 // 2,
        "stride": stride,
        "dataset": args_ns.data.dataset,
        "size": (args_ns.model.hr_dim[1], args_ns.model.hr_dim[0]),
    }
    inference_dataset = ActionSpotInferenceDataset(args.video, **inf_dataset_kwargs)
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    print("Running model (this may take a while)...")
    events_result, stride_out = inference(
        model,
        inference_loader,
        classes,
        threshold=args.inference_threshold,
        store_json_path=None,
    )
    assert stride_out == stride

    if args.save_json:
        os.makedirs(args.save_json, exist_ok=True)
        store_json_inference(args.save_json, events_result, stride=stride)
        print("Saved JSON to:", os.path.abspath(os.path.join(args.save_json, "results_inference.json")))

    timeline = build_timeline(events_result, stride)
    print(f"Detected {len(timeline)} action peaks after SNMS.")
    print(
        f"Timing: model stride={stride} (scores on every {stride}th frame). "
        f"Inference CAP_PROP_FRAME_COUNT={inference_dataset._video_len}; "
        f"time = frame_index / FPS (display FPS={fps_display:.4f})."
    )
    if n_frames_meta and abs(n_frames_meta - inference_dataset._video_len) > 2:
        print(
            f"Warning: probe reported {n_frames_meta} frames but dataset used {inference_dataset._video_len}; "
            "if timing drifts, re-encode the video or fix container metadata.",
            file=sys.stderr,
        )

    out_path = args.output
    if out_path is None:
        base, _ = os.path.splitext(args.video)
        out_path = base + "_adaspot.mp4"

    cap = cv2.VideoCapture(args.video)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    top_band, bottom_band = _margin_heights(fh, fw)
    out_h = fh + top_band + bottom_band
    print(
        f"Output layout: {top_band}px top bar + {fh}px video (no overlays) + {bottom_band}px bottom bar -> height {out_h}"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_display, (fw, out_h))
    if not writer.isOpened():
        print("Failed to open VideoWriter for:", out_path, file=sys.stderr)
        sys.exit(1)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_frame = compose_frame_with_margins(
            frame,
            frame_idx,
            timeline,
            colors,
            top_band,
            bottom_band,
            text_scale=args.text_scale,
            fps=fps_display,
            stride=stride,
        )
        writer.write(out_frame)
        frame_idx += 1

    cap.release()
    writer.release()
    if frame_idx != inference_dataset._video_len:
        print(
            f"Warning: decoded {frame_idx} frames while inference assumed {inference_dataset._video_len}; "
            "timeline may not match the last frames.",
            file=sys.stderr,
        )
    print("Wrote:", os.path.abspath(out_path))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
