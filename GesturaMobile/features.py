"""
features.py - single source of truth for turning MediaPipe hands into model input.

Used by tracker.py (recording), train_model.py (training) and test_model.py
(live inference), so the three can never drift out of sync.

Per frame we build a 132-value vector (2 hand slots x 66 values):
    63  hand SHAPE   : 21 landmarks (x,y,z), wrist at origin, divided by hand size
                       -> invariant to distance from camera and screen position
     2  wrist x,y    : where the hand is in the frame (needed for "forehead" vs "chin")
     1  hand scale   : apparent hand size (0 means "this slot has no hand")
Hands are ordered left->right by wrist x (MediaPipe handedness is unreliable
on mirrored video, so we don't use it).

A sign is a clip of SEQ_LEN frames covering WINDOW_SECONDS, resampled by TIME
(not by frame count) so the model behaves the same at 15, 30 or 60 FPS.

IMPORTANT: all frames are flipped (mirror view) BEFORE MediaPipe runs, in the
recorder and in the live demo. Keep it that way.
"""
import glob
import os

import numpy as np

SEQ_LEN = 30
WINDOW_SECONDS = 1.5
NUM_HANDS = 2
PER_HAND = 66
FRAME_DIM = NUM_HANDS * PER_HAND          # 132
MODEL_DIM = FRAME_DIM * 2                 # features + velocity = 264
SCALE_IDX = tuple(PER_HAND * (i + 1) - 1 for i in range(NUM_HANDS))   # (65, 131)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "clips")


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def hand_vector(landmarks):
    """21 MediaPipe landmarks -> 66-value vector, or None if degenerate."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    rel = pts - pts[0]
    scale = float(np.linalg.norm(rel[:, :2], axis=1).max())
    if scale < 1e-4:
        return None
    rel /= scale
    return np.concatenate([rel.ravel(), pts[0, :2], [scale]]).astype(np.float32)


def frame_features(hand_landmarks_list):
    """All hands detected in one frame -> fixed 132-value vector."""
    vecs = [hand_vector(h) for h in (hand_landmarks_list or [])]
    vecs = [v for v in vecs if v is not None]
    vecs.sort(key=lambda v: v[63])        # left -> right by wrist x
    out = np.zeros(FRAME_DIM, dtype=np.float32)
    for i, v in enumerate(vecs[:NUM_HANDS]):
        out[i * PER_HAND:(i + 1) * PER_HAND] = v
    return out


def resample_window(times, feats, end_time, seconds=WINDOW_SECONDS, n=SEQ_LEN):
    """Pick the n frames nearest to evenly spaced times in [end-seconds, end]."""
    times = np.asarray(times, dtype=np.float64)
    feats = np.asarray(feats, dtype=np.float32)
    targets = np.linspace(end_time - seconds, end_time, n)
    idx = np.abs(times[None, :] - targets[:, None]).argmin(axis=1)
    return feats[idx]


def hand_presence(window):
    """Fraction of frames in a (T,132) window that contain at least one hand."""
    return float((window[:, list(SCALE_IDX)] > 0).any(axis=1).mean())


def add_velocity(x):
    """(..., T, 132) -> (..., T, 264): append frame-to-frame differences."""
    x = np.asarray(x, dtype=np.float32)
    vel = np.diff(x, axis=-2, prepend=x[..., :1, :])
    return np.concatenate([x, vel], axis=-1)


# --------------------------------------------------------------------------- #
# Augmentation (training only)
# --------------------------------------------------------------------------- #
def augment_clip(clip, rng):
    """Random, realistic variation of one (SEQ_LEN,132) clip."""
    c = clip.reshape(SEQ_LEN, NUM_HANDS, PER_HAND).copy()
    present = c[:, :, PER_HAND - 1] > 0                       # (T, 2)
    shp = c[:, :, :63].reshape(SEQ_LEN, NUM_HANDS, 21, 3).copy()

    # 1) mirror (covers left-handed signers)
    if rng.random() < 0.5:
        shp[..., 0] *= -1
        c[:, :, 63] = np.where(present, 1.0 - c[:, :, 63], 0.0)
        both = present.all(axis=1)
        swapped = None
        if both.any():
            swapped = c[both][:, ::-1].copy()
            shp_both = shp[both][:, ::-1].copy()
        if swapped is not None:
            c[both] = swapped
            shp[both] = shp_both

    # 2) small in-plane rotation of the hand shape
    th = np.deg2rad(np.clip(rng.normal(0, 6), -15, 15))
    cs, sn = np.cos(th), np.sin(th)
    x, y = shp[..., 0].copy(), shp[..., 1].copy()
    shp[..., 0] = cs * x - sn * y
    shp[..., 1] = sn * x + cs * y

    # 3) landmark noise (only where a hand exists)
    mask = present[..., None, None]
    shp += rng.normal(0, 0.01, shp.shape).astype(np.float32) * mask

    c[:, :, :63] = shp.reshape(SEQ_LEN, NUM_HANDS, 63)

    # 4) position / size changes (different camera framing, distance)
    shift = rng.normal(0, 0.04, size=2).astype(np.float32)
    c[:, :, 63:65] += (shift + rng.normal(0, 0.004, (SEQ_LEN, NUM_HANDS, 2))) * present[..., None]
    c[:, :, 65] *= rng.uniform(0.85, 1.15)

    # 5) speed change + time shift
    center = (SEQ_LEN - 1) / 2
    f = rng.uniform(0.85, 1.15)
    t_shift = rng.uniform(-3, 3)
    idx = np.clip(np.round(center + (np.arange(SEQ_LEN) - center) * f + t_shift), 0, SEQ_LEN - 1).astype(int)
    c = c[idx]

    # 6) random tracking dropouts
    c[rng.random(SEQ_LEN) < 0.04] = 0.0

    return c.reshape(SEQ_LEN, FRAME_DIM).astype(np.float32)


# --------------------------------------------------------------------------- #
# Dataset on disk:  clips/<label>/<person>__<session>__<label>_<n>.npy
# --------------------------------------------------------------------------- #
def count_clips(data_dir=DATA_DIR):
    counts = {}
    if not os.path.isdir(data_dir):
        return counts
    for label in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, label)
        if os.path.isdir(d):
            counts[label] = len(glob.glob(os.path.join(d, "*.npy")))
    return counts


def load_dataset(data_dir=DATA_DIR, group_by="session"):
    """Returns X (N,T,132), y (N,) label strings, groups (N,) strings."""
    X, y, g = [], [], []
    if not os.path.isdir(data_dir):
        return np.zeros((0, SEQ_LEN, FRAME_DIM), np.float32), np.array([]), np.array([])
    for label in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, label)
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "*.npy"))):
            arr = np.load(f)
            if arr.shape != (SEQ_LEN, FRAME_DIM):
                print(f"  [skip] {f}: unexpected shape {arr.shape}")
                continue
            parts = os.path.basename(f).split("__")
            if len(parts) >= 3:
                group = parts[0] if group_by == "person" else "__".join(parts[:2])
            else:
                group = "unknown"
            X.append(arr.astype(np.float32))
            y.append(label)
            g.append(group)
    if not X:
        return np.zeros((0, SEQ_LEN, FRAME_DIM), np.float32), np.array([]), np.array([])
    return np.stack(X), np.array(y), np.array(g)
