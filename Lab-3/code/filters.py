"""
filters.py
----------
From-scratch 2D spatial filtering primitives for the AGV Spatial Filtering Lab.

NO cv2.filter2D, NO scipy.signal.convolve2d / correlate2d are used anywhere.
Every operation below is built on plain numpy array indexing/arithmetic only,
implementing the correlation/convolution sum explicitly.

Author: (your name here)
Course: PCC-03 Digital Image and Video Processing
"""

import numpy as np


# ----------------------------------------------------------------------
# 1. CORE CORRELATION / CONVOLUTION (from first principles)
# ----------------------------------------------------------------------

def pad_image(img, pad_h, pad_w, mode="reflect"):
    """Zero/reflect pad a 2D array. Reflect padding avoids artificial dark
    borders that would otherwise bias PSNR near the edges."""
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)


def correlate2d(image, kernel, mode="same", pad_mode="reflect"):
    """
    2D cross-correlation implemented from scratch.

    output(i, j) = sum_{m,n} image(i+m, j+n) * kernel(m, n)

    This is the operation you get from a naive sliding-window filter (NOT
    convolution, which additionally flips the kernel). Implemented with an
    explicit double loop over the kernel footprint using vectorized slicing
    over the image plane, which is the standard "from scratch" approach
    taught for spatial filtering (no library correlate/convolve call).
    """
    image = image.astype(np.float64)
    kernel = kernel.astype(np.float64)

    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    if mode == "same":
        padded = pad_image(image, pad_h, pad_w, mode=pad_mode)
        out_h, out_w = image.shape
    elif mode == "valid":
        padded = image
        out_h, out_w = image.shape[0] - kh + 1, image.shape[1] - kw + 1
    else:
        raise ValueError("mode must be 'same' or 'valid'")

    output = np.zeros((out_h, out_w), dtype=np.float64)

    # Explicit correlation sum: shift-and-accumulate over the kernel support.
    # This is the from-scratch equivalent of the textbook double summation
    # sum_m sum_n w(m,n) f(i+m, j+n), done via kh*kw vectorized shifts
    # instead of kh*kw*H*W scalar Python loops (still a from-scratch
    # implementation of the correlation operator, just written efficiently).
    for m in range(kh):
        for n in range(kw):
            output += kernel[m, n] * padded[m:m + out_h, n:n + out_w]

    return output


def convolve2d(image, kernel, mode="same", pad_mode="reflect"):
    """
    True 2D convolution: correlation with the kernel flipped 180 degrees.
    output(i,j) = sum_{m,n} image(i-m, j-n) * kernel(m,n)
                = correlate(image, flip(kernel))
    """
    flipped = kernel[::-1, ::-1]
    return correlate2d(image, flipped, mode=mode, pad_mode=pad_mode)


def clip_to_uint8(img):
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# 2. TASK 1 — BOX / AVERAGING FILTER
# ----------------------------------------------------------------------

def box_kernel(size):
    """Normalized k x k averaging kernel (sum = 1)."""
    return np.ones((size, size), dtype=np.float64) / (size * size)


def box_filter(image, size):
    kernel = box_kernel(size)
    return correlate2d(image, kernel, mode="same")


# ----------------------------------------------------------------------
# 3. TASK 2 — LAPLACIAN SHARPENING
# ----------------------------------------------------------------------

# 4-neighbor (edge-neighbor only) Laplacian
LAPLACIAN_4 = np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]], dtype=np.float64)

# 8-neighbor (includes diagonals) Laplacian
LAPLACIAN_8 = np.array([[1, 1, 1],
                         [1, -8, 1],
                         [1, 1, 1]], dtype=np.float64)


def laplacian_response(image, variant="4"):
    kernel = LAPLACIAN_4 if variant == "4" else LAPLACIAN_8
    return correlate2d(image, kernel, mode="same")


def laplacian_sharpen(image, variant="4", c=-1.0):
    """
    Standard Laplacian sharpening:
        g(x,y) = f(x,y) + c * Laplacian(f)(x,y)
    c = -1 for the kernels above (center-negative convention), matching
    g = f - Laplacian(f) in the textbook formulation.
    """
    lap = laplacian_response(image, variant=variant)
    sharpened = image.astype(np.float64) + c * lap
    return sharpened


# ----------------------------------------------------------------------
# 4. TASK 3 — UNSHARP MASKING / HIGH-BOOST FILTERING
# ----------------------------------------------------------------------

def unsharp_highboost(image, k=1.0, blur_size=5):
    """
    g(x,y) = f(x,y) + k * (f(x,y) - blur(f)(x,y))

    k = 1   -> classic unsharp masking
    k > 1   -> high-boost filtering (amplifies the high-frequency mask)
    """
    blurred = box_filter(image, blur_size)
    mask = image.astype(np.float64) - blurred
    result = image.astype(np.float64) + k * mask
    return result


# ----------------------------------------------------------------------
# 5. DEGRADATION MODEL (noise + motion blur), built from scratch
# ----------------------------------------------------------------------

def add_gaussian_noise(image, sigma, seed=None):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, image.shape)
    noisy = image.astype(np.float64) + noise
    return np.clip(noisy, 0, 255)


def motion_blur_kernel(length=7, angle_deg=0):
    """
    Linear motion blur kernel of given pixel length, built by rasterizing a
    line segment through the kernel center (no library motion-blur helper).
    """
    size = length if length % 2 == 1 else length + 1
    kernel = np.zeros((size, size), dtype=np.float64)
    center = size // 2
    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)

    for t in np.linspace(-(length // 2), length // 2, length * 4):
        x = int(round(center + t * dx))
        y = int(round(center + t * dy))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0

    if kernel.sum() == 0:
        kernel[center, center] = 1.0
    kernel /= kernel.sum()
    return kernel


def apply_motion_blur(image, length=8, angle_deg=0):
    kernel = motion_blur_kernel(length, angle_deg)
    return correlate2d(image, kernel, mode="same")  # symmetric-ish kernel; correlation==convolution for a line mask through center at angle 0


# ----------------------------------------------------------------------
# 6. TASK 4 — OBJECTIVE METRICS
# ----------------------------------------------------------------------

def psnr(clean, test, max_val=255.0):
    """Peak Signal-to-Noise Ratio against ground truth, from scratch."""
    clean = clean.astype(np.float64)
    test = test.astype(np.float64)
    mse = np.mean((clean - test) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((max_val ** 2) / mse)


def sharpness_metric(image):
    """
    Sharpness / edge-strength metric: variance of the Laplacian (4-neighbor).
    Higher variance => more high-frequency edge energy => sharper image.
    This is our own metric, computed with our own laplacian_response(),
    not a library "blur detector".
    """
    lap = laplacian_response(image, variant="4")
    return float(np.var(lap))
