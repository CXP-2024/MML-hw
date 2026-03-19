#!/usr/bin/env python3
"""Mini JPEG-Style Compressor for Problem Set 1, Problem 3."""

import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, json

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ---- helpers ----

def dct2(block):
    """2D DCT (type-II, orthonormal)."""
    return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(block):
    """2D inverse DCT."""
    return idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

def load_gray(path):
    """Load image as float64 grayscale [0,255]."""
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float64)

def pad_to_multiple(img, block=8):
    """Pad image so dimensions are multiples of block size."""
    h, w = img.shape
    new_h = int(np.ceil(h / block) * block)
    new_w = int(np.ceil(w / block) * block)
    padded = np.zeros((new_h, new_w), dtype=img.dtype)
    padded[:h, :w] = img
    return padded, h, w

def blockwise_dct(img, block=8):
    """Apply block-wise DCT, return coefficient array of same shape."""
    h, w = img.shape
    coeffs = np.zeros_like(img)
    for i in range(0, h, block):
        for j in range(0, w, block):
            coeffs[i:i+block, j:j+block] = dct2(img[i:i+block, j:j+block])
    return coeffs

def blockwise_idct(coeffs, block=8):
    """Apply block-wise inverse DCT."""
    h, w = coeffs.shape
    recon = np.zeros_like(coeffs)
    for i in range(0, h, block):
        for j in range(0, w, block):
            recon[i:i+block, j:j+block] = idct2(coeffs[i:i+block, j:j+block])
    return recon

def quantize(coeffs, Q, block=8):
    """Divide by Q and round to int for each block."""
    h, w = coeffs.shape
    qcoeffs = np.zeros_like(coeffs)
    for i in range(0, h, block):
        for j in range(0, w, block):
            qcoeffs[i:i+block, j:j+block] = np.round(
                coeffs[i:i+block, j:j+block] / Q)
    return qcoeffs

def dequantize(qcoeffs, Q, block=8):
    """Multiply back by Q."""
    h, w = qcoeffs.shape
    coeffs = np.zeros_like(qcoeffs)
    for i in range(0, h, block):
        for j in range(0, w, block):
            coeffs[i:i+block, j:j+block] = qcoeffs[i:i+block, j:j+block] * Q
    return coeffs

def psnr(orig, recon):
    mse = np.mean((orig - recon) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0**2 / mse)

# ---- Quantization matrices ----

def simple_Q(p):
    """Q(i,j) = p*(i+j+1)"""
    ii, jj = np.meshgrid(np.arange(8), np.arange(8), indexing='ij')
    return p * (ii + jj + 1).astype(np.float64)

JPEG_QY = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99],
], dtype=np.float64)

# ---- Zigzag scan ----

def zigzag_order(n=8):
    """Return list of (i,j) in zigzag order for n x n block."""
    order = []
    for s in range(2 * n - 1):
        if s % 2 == 0:  # going up
            for i in range(min(s, n-1), max(s-n+1, 0)-1, -1):
                j = s - i
                if 0 <= j < n:
                    order.append((i, j))
        else:  # going down
            for j in range(min(s, n-1), max(s-n+1, 0)-1, -1):
                i = s - j
                if 0 <= i < n:
                    order.append((i, j))
    return order

ZZ = zigzag_order(8)

def zigzag_scan(block):
    """Flatten 8x8 block in zigzag order."""
    return np.array([block[i, j] for i, j in ZZ])

def run_length_encode(seq):
    """RLE: returns list of (value, run_length) pairs."""
    if len(seq) == 0:
        return []
    rle = []
    current = seq[0]
    count = 1
    for v in seq[1:]:
        if v == current:
            count += 1
        else:
            rle.append((current, count))
            current = v
            count = 1
    rle.append((current, count))
    return rle

# ---- Statistics ----

def compute_stats(qcoeffs, block=8):
    """Compute zero ratios overall / low-freq / high-freq."""
    h, w = qcoeffs.shape
    total = 0
    total_zero = 0
    low_total = 0
    low_zero = 0
    high_total = 0
    high_zero = 0
    for i in range(0, h, block):
        for j in range(0, w, block):
            blk = qcoeffs[i:i+block, j:j+block]
            for bi in range(8):
                for bj in range(8):
                    val = blk[bi, bj]
                    total += 1
                    if val == 0:
                        total_zero += 1
                    if bi + bj < 4:  # low freq
                        low_total += 1
                        if val == 0:
                            low_zero += 1
                    else:  # high freq
                        high_total += 1
                        if val == 0:
                            high_zero += 1
    return {
        'zero_ratio': total_zero / total,
        'low_zero_ratio': low_zero / low_total if low_total > 0 else 0,
        'high_zero_ratio': high_zero / high_total if high_total > 0 else 0,
    }

def compute_rle_length(qcoeffs, block=8):
    """Total RLE encoded sequence length (number of (value, run) pairs)."""
    h, w = qcoeffs.shape
    total_rle_len = 0
    for i in range(0, h, block):
        for j in range(0, w, block):
            blk = qcoeffs[i:i+block, j:j+block]
            zz = zigzag_scan(blk)
            rle = run_length_encode(zz)
            total_rle_len += len(rle)
    return total_rle_len

# ---- Main pipeline ----

def run_pipeline(img_path, img_name):
    """Run full pipeline on one image, return results dict."""
    img = load_gray(img_path)
    padded, orig_h, orig_w = pad_to_multiple(img)

    # 3.1 Patch-wise DCT
    coeffs = blockwise_dct(padded)

    # Visualize several example patches before and after DCT
    # Pick patches from diverse regions with different variance levels
    h_pad, w_pad = padded.shape
    # Only consider blocks within the original image (not padding)
    variances = []
    for i in range(0, orig_h - 7, 8):
        for j in range(0, orig_w - 7, 8):
            blk = padded[i:i+8, j:j+8]
            variances.append((np.var(blk), i, j))
    variances.sort(key=lambda x: x[0], reverse=True)

    # Pick 4 patches at different variance levels, ensuring spatial diversity
    def min_dist(pick_list, candidate):
        if not pick_list:
            return float('inf')
        return min(abs(candidate[1] - p[1]) + abs(candidate[2] - p[2]) for p in pick_list)

    n_blocks = len(variances)
    # Divide into 4 quartiles by variance and pick one from each, maximizing distance
    picks = []
    quartile_size = max(n_blocks // 4, 1)
    for q in range(4):
        start = q * quartile_size
        end = min(start + quartile_size, n_blocks)
        candidates = variances[start:end]
        # Filter out near-zero variance (padding artifacts)
        candidates = [c for c in candidates if c[0] > 0.5] or candidates[:5]
        # Pick the one farthest from already picked patches
        best = max(candidates, key=lambda c: min_dist(picks, c))
        picks.append(best)

    # --- Figure 1: Original image with patch locations marked ---
    colors = ['red', 'orange', 'cyan', 'lime']
    labels = ['High var (Q1)', 'Mid-high var (Q2)', 'Mid-low var (Q3)', 'Low var (Q4)']
    import matplotlib.patches as mpatches

    fig_loc, ax_loc = plt.subplots(1, 1, figsize=(12, 8))
    ax_loc.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax_loc.set_title(f'{img_name}: Original image with selected patch locations', fontsize=12)
    for idx, (var, pi, pj) in enumerate(picks[:4]):
        rect = mpatches.Rectangle((pj - 0.5, pi - 0.5), 8, 8,
                                   linewidth=2.5, edgecolor=colors[idx], facecolor='none')
        ax_loc.add_patch(rect)
        # Add label near the box
        ax_loc.text(pj + 10, pi + 4, f'{labels[idx]}\nvar={var:.0f}',
                    color=colors[idx], fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))
    ax_loc.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{img_name}_patch_locations.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Figure 2: Zoomed patches (spatial) + DCT log-magnitude ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, (var, pi, pj) in enumerate(picks[:4]):
        patch = padded[pi:pi+8, pj:pj+8]
        cpatch = coeffs[pi:pi+8, pj:pj+8]
        # Spatial domain (zoomed in)
        axes[0, idx].imshow(patch, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        axes[0, idx].set_title(f'{labels[idx]}\n({pi},{pj}), var={var:.0f}', fontsize=9,
                               color=colors[idx])
        axes[0, idx].axis('off')
        # Add a colored border to match the location map
        for spine in axes[0, idx].spines.values():
            spine.set_edgecolor(colors[idx])
            spine.set_linewidth(3)
            spine.set_visible(True)
        # DCT domain: use log(1 + |coeff|) for better visibility
        log_coeff = np.log1p(np.abs(cpatch))
        im = axes[1, idx].imshow(log_coeff, cmap='hot', interpolation='nearest')
        axes[1, idx].set_title(f'DCT log(1+|c|)', fontsize=9)
        axes[1, idx].axis('off')
        plt.colorbar(im, ax=axes[1, idx], fraction=0.046, pad=0.04)
    fig.suptitle(f'{img_name}: Zoomed 8x8 patches (top: spatial, bottom: DCT log-magnitude)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{img_name}_dct_patches.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {}

    # Strategy A: Simple Q with p in {4, 8, 16}
    for p in [4, 8, 16]:
        Q = simple_Q(p)
        qc = quantize(coeffs, Q)
        deq = dequantize(qc, Q)
        recon = blockwise_idct(deq)
        recon_crop = np.clip(recon[:orig_h, :orig_w], 0, 255)
        stats = compute_stats(qc)
        stats['psnr'] = psnr(img, recon_crop)
        stats['rle_length'] = compute_rle_length(qc)
        key = f'A_p{p}'
        results[key] = stats

        # Save reconstruction figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[1].imshow(recon_crop, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title(f'Reconstructed (Simple Q, p={p})\nPSNR={stats["psnr"]:.2f} dB')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'{img_name}_{key}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # Strategy B: JPEG QY with p in {0.5, 1, 2}
    for p in [0.5, 1, 2]:
        Q = p * JPEG_QY
        qc = quantize(coeffs, Q)
        deq = dequantize(qc, Q)
        recon = blockwise_idct(deq)
        recon_crop = np.clip(recon[:orig_h, :orig_w], 0, 255)
        stats = compute_stats(qc)
        stats['psnr'] = psnr(img, recon_crop)
        stats['rle_length'] = compute_rle_length(qc)
        key = f'B_p{str(p).replace(".", "_")}'
        results[key] = stats

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title('Original')
        axes[0].axis('off')
        axes[1].imshow(recon_crop, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title(f'Reconstructed (JPEG Q, p={p})\nPSNR={stats["psnr"]:.2f} dB')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'{img_name}_{key}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    return results


if __name__ == '__main__':
    base = os.path.dirname(__file__)
    all_results = {}
    for fname, label in [('p3_1.png', 'p3_1'), ('p3_2.png', 'p3_2')]:
        path = os.path.join(base, fname)
        print(f'Processing {fname} ...')
        res = run_pipeline(path, label)
        all_results[label] = res

    # Print results as tables
    for img_name, res in all_results.items():
        print(f'\n===== {img_name} =====')
        print(f'{"Setting":<12} {"ZeroRatio":>10} {"LowZero":>10} {"HighZero":>10} {"PSNR(dB)":>10} {"RLE_Len":>10}')
        for key in ['A_p4', 'A_p8', 'A_p16', 'B_p0_5', 'B_p1', 'B_p2']:
            s = res[key]
            print(f'{key:<12} {s["zero_ratio"]:>10.4f} {s["low_zero_ratio"]:>10.4f} {s["high_zero_ratio"]:>10.4f} {s["psnr"]:>10.2f} {s["rle_length"]:>10d}')

    # Save results as JSON for LaTeX
    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print('\nDone. Figures saved to', OUT_DIR)
