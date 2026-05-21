from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_gray(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 2:
        return frame.astype(np.float32)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return np.tensordot(frame[..., :3].astype(np.float32), weights, axes=([-1], [0]))


def resize_frame(frame: np.ndarray, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(frame, size[::-1], interpolation=cv2.INTER_AREA).astype(np.float32)
    except Exception:
        from PIL import Image

        image = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
        image = image.resize(size[::-1], Image.BILINEAR)
        return np.asarray(image).astype(np.float32)


def load_video(
    path: str | Path,
    max_frames: int = 16,
    size: tuple[int, int] = (128, 128),
    grayscale: bool = True,
) -> np.ndarray:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError("load_video requires imageio. Install dependencies with `pip install -r requirements.txt`.") from exc

    frames = []
    reader = imageio.get_reader(str(path))
    for idx, frame in enumerate(reader):
        if idx >= max_frames:
            break
        if grayscale:
            frame = to_gray(frame)
        frame = resize_frame(frame, size)
        frames.append(frame)
    reader.close()
    if not frames:
        raise ValueError(f"No frames could be read from {path}")
    return np.stack(frames).astype(np.float32)


def save_gif(frames: np.ndarray, path: str | Path, fps: int = 8) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError("save_gif requires imageio. Install dependencies with `pip install -r requirements.txt`.") from exc

    path = Path(path)
    ensure_dir(path.parent)
    arr = np.asarray(frames)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    imageio.mimsave(str(path), list(arr), duration=1.0 / fps)


def mse(target: np.ndarray, pred: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    return float(np.mean((target - pred) ** 2))


def psnr(target: np.ndarray, pred: np.ndarray, max_val: float = 255.0) -> float:
    value = mse(target, pred)
    if value == 0:
        return math.inf
    return float(10.0 * math.log10((max_val * max_val) / value))


def residual_entropy(residual: np.ndarray) -> float:
    values = np.rint(np.asarray(residual)).astype(np.int32).ravel()
    _, counts = np.unique(values, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def summarize_prediction(target: np.ndarray, pred: np.ndarray, residual: np.ndarray) -> dict[str, float]:
    return {
        "mse": mse(target, pred),
        "psnr": psnr(target, pred),
        "residual_entropy": residual_entropy(residual),
    }


def plot_frame_grid(frames: Iterable[np.ndarray], titles: Iterable[str], path: str | Path) -> None:
    import matplotlib.pyplot as plt

    frames = list(frames)
    titles = list(titles)
    path = Path(path)
    ensure_dir(path.parent)

    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    if n == 1:
        axes = [axes]

    for ax, frame, title in zip(axes, frames, titles):
        ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_prediction(prev: np.ndarray, cur: np.ndarray, pred: np.ndarray, residual: np.ndarray, path: str | Path) -> None:
    abs_diff = np.abs(cur - prev)
    signed = np.clip(residual + 128, 0, 255)
    plot_frame_grid(
        [prev, cur, pred, abs_diff, signed],
        ["prev", "cur", "pred", "abs diff", "signed residual + 128"],
        path,
    )


def plot_motion_vectors(motion_vectors: np.ndarray, block_size: int, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    ensure_dir(path.parent)

    h_blocks, w_blocks, _ = motion_vectors.shape
    ys, xs = np.mgrid[0:h_blocks, 0:w_blocks]
    xs = xs * block_size + block_size / 2
    ys = ys * block_size + block_size / 2
    dx = motion_vectors[..., 1]
    dy = motion_vectors[..., 0]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.quiver(xs, ys, dx, dy, angles="xy", scale_units="xy", scale=1)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("motion vectors")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_translation_clip(
    num_frames: int = 12,
    size: tuple[int, int] = (128, 128),
    velocity: tuple[int, int] = (2, 4),
    square_size: int = 28,
) -> np.ndarray:
    height, width = size
    dy, dx = velocity
    frames = []
    for t in range(num_frames):
        frame = np.zeros((height, width), dtype=np.float32) + 20
        y0 = 30 + dy * t
        x0 = 20 + dx * t
        y1 = min(y0 + square_size, height)
        x1 = min(x0 + square_size, width)
        frame[max(y0, 0):y1, max(x0, 0):x1] = 220
        frames.append(frame)
    return np.stack(frames)


def make_flicker_clip(num_frames: int = 12, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    clip = make_translation_clip(num_frames=num_frames, size=size, velocity=(0, 3))
    for t in range(num_frames):
        if t % 2:
            clip[t] = np.clip(clip[t] + 45, 0, 255)
    return clip


def make_jitter_clip(num_frames: int = 12, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    height, width = size
    frames = []
    offsets = [(0, 0), (1, 4), (-2, 7), (2, 9), (-1, 13), (2, 15)]
    for t in range(num_frames):
        frame = np.zeros((height, width), dtype=np.float32) + 20
        oy, ox = offsets[t % len(offsets)]
        y0 = 34 + oy
        x0 = 22 + 4 * t + ox
        frame[y0:y0 + 28, x0:x0 + 28] = 220
        frames.append(frame)
    return np.stack(frames)


def make_blur_clip(num_frames: int = 12, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    clip = make_translation_clip(num_frames=num_frames, size=size, velocity=(1, 3))
    try:
        import cv2

        return np.stack([cv2.GaussianBlur(frame, (9, 9), 2.0) for frame in clip]).astype(np.float32)
    except Exception:
        return clip
