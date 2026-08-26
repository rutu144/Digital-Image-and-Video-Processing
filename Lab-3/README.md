# Digital Image and Video Processing

## Lab 3 — Spatial Filtering

### Restoring and Enhancing Degraded Onboard Camera Feed for an Autonomous Ground Vehicle (AGV)

---

## 📌 Overview

This laboratory experiment focuses on **spatial filtering for image restoration and enhancement**.

An Autonomous Ground Vehicle (AGV) uses an onboard camera for tasks such as:

* Obstacle detection
* Terrain classification
* Boundary detection
* Terrain segmentation

In outdoor environments, the camera image may be affected by:

* Sensor noise
* Atmospheric particles
* Low-light conditions
* Vibration-induced motion blur
* Loss of important object boundaries

Therefore, the image must be pre-processed before applying high-level computer vision algorithms.

In this lab, a clean grayscale image is intentionally degraded using **Gaussian noise and motion blur**. Different spatial-filtering techniques are then implemented **from scratch** to reduce noise and recover/enhance important edges.

The complete processing pipeline is:

```text
Clean Grayscale Image
        ↓
Generate Degraded Images
        ↓
Gaussian Noise / Motion Blur
        ↓
Noise Suppression
(Averaging Filter)
        ↓
Boundary Recovery
(Laplacian Sharpening)
        ↓
Image Enhancement
(Unsharp Masking / High-Boost)
        ↓
Objective Evaluation
(PSNR + Sharpness)
        ↓
Best Configuration
        ↓
Final AGV Pipeline
```

---

The main objectives of this laboratory are:

1. Implement **2D spatial correlation** from first principles.
2. Implement **2D convolution** from first principles.
3. Implement an averaging filter without using ready-made filtering functions.
4. Suppress Gaussian noise while preserving important image boundaries.
5. Implement Laplacian-based image sharpening.
6. Compare 4-neighbor and 8-neighbor Laplacian operators.
7. Implement unsharp masking.
8. Implement high-boost filtering.
9. Understand the effect of the boost factor `k`.
10. Evaluate processed images using objective image-quality metrics.
11. Use PSNR and a sharpness/edge-strength metric to select the best configuration.
12. Design a final two-stage AGV preprocessing pipeline.

---

# Problem Statement

An AGV operating in an outdoor environment receives camera frames that can contain sensor grain, atmospheric noise, and vibration-induced blur.

The objective of this laboratory is to design a spatial-filtering preprocessing pipeline that:

* Reduces unwanted noise.
* Recovers degraded boundaries.
* Enhances navigation-relevant edges.
* Preserves useful image details.
* Provides quantitative evidence for selecting filter parameters.

All major filtering operations must be implemented **from scratch** rather than using library shortcuts.

---

# Input Image

A single grayscale image representing an AGV forward-camera scene is required.

The selected image should preferably contain:

* Roads or paths
* Obstacles
* Buildings
* Trees
* Vehicles
* Clear object boundaries

A personal outdoor photograph converted to grayscale can also be used.

The original clean image is treated as the **ground-truth image** for quantitative evaluation.

---

# Test Data Generation

The clean grayscale image is programmatically degraded to simulate realistic AGV camera conditions.

## 1. Gaussian Noise

Two noise levels are generated:

```text
σ = 10
σ = 25
```

The noise represents sensor/thermal grain and atmospheric effects.

## 2. Motion Blur

Mild motion blur is introduced using a linear motion-blur kernel.

Required kernel length:

```text
7–9 pixels
```

The blur represents vibration or vehicle movement during image acquisition.

---

# Tasks

## Task 1 — Noise Suppression Using Averaging

A box/averaging filter is implemented from scratch.

The filtering operation must use the self-implemented `correlate2d()` function rather than a library filtering shortcut.

### Noise Levels

Two noisy images are used:

```text
Gaussian Noise σ = 10
Gaussian Noise σ = 25
```

### Kernel Sizes

The averaging filter is tested using:

```text
3 × 3
5 × 5
9 × 9
```

Therefore, six configurations are evaluated:

| Noise Level | Kernel Size |
| ----------- | ----------- |
| σ = 10      | 3×3         |
| σ = 10      | 5×5         |
| σ = 10      | 9×9         |
| σ = 25      | 3×3         |
| σ = 25      | 5×5         |
| σ = 25      | 9×9         |

### Observation

The results should be compared in terms of:

* Noise reduction
* Edge preservation
* Fine-detail preservation
* Overall image quality

As kernel size increases, noise suppression may improve, but excessive smoothing can remove important edges and details.

---

# Task 2 — Boundary Recovery Using Laplacian Sharpening

The motion-blurred image is processed using Laplacian sharpening.

Two Laplacian operators are implemented.

## 4-Neighbor Laplacian

```text
 0  -1   0
-1   4  -1
 0  -1   0
```

This operator mainly considers the horizontal and vertical neighboring pixels.

## 8-Neighbor Laplacian

```text
-1  -1  -1
-1   8  -1
-1  -1  -1
```

This operator also considers diagonal neighbors.

### Comparison

Both operators should be applied to the same motion-blurred image.

The output should include:

* Motion-blurred image
* 4-neighbor Laplacian response
* 8-neighbor Laplacian response
* Sharpened results

The two Laplacian variants should be compared based on their ability to recover obstacle boundaries.

---

# Task 3 — Unsharp Masking and High-Boost Filtering

Unsharp masking and high-boost filtering are applied to the noisy image with:

```text
σ = 10
```

The following boost factors are tested:

```text
k = 1
k = 1.5
k = 2
k = 3
```

### Purpose

The objective is to determine how increasing `k` affects:

* Image sharpness
* Edge strength
* Noise amplification
* Overall image quality

A very high boost factor may increase unwanted noise along with useful edges.

The experiment should identify the point beyond which increasing `k` no longer provides useful improvement.

---

# Task 4 — Objective Evaluation

Visual inspection alone is not sufficient.

Every processed output must be quantitatively evaluated.

Two metrics are used.

---

## 1. PSNR — Peak Signal-to-Noise Ratio

PSNR measures the similarity between the processed image and the original clean ground-truth image.

The mean squared error is:

```text
MSE = (1 / MN) ΣΣ [I(i,j) - K(i,j)]²
```

PSNR is calculated as:

```text
PSNR = 10 log10(MAX² / MSE)
```

For an 8-bit grayscale image:

```text
MAX = 255
```

Higher PSNR generally indicates greater similarity to the clean ground-truth image.

---

## 2. Sharpness / Edge-Strength Metric

A sharpness metric is also calculated.

One suitable metric is:

```text
Variance of Laplacian
```

The metric measures the strength of intensity changes in the image.

Higher values generally indicate stronger edges and greater high-frequency content.

However, excessive noise can also increase the sharpness metric, so it must be interpreted together with PSNR.

---

# Results Table

The final report should contain a table similar to:

| Task       | Configuration | PSNR | Sharpness |
| ---------- | ------------- | ---: | --------: |
| Averaging  | σ=10, 3×3     |    — |         — |
| Averaging  | σ=10, 5×5     |    — |         — |
| Averaging  | σ=10, 9×9     |    — |         — |
| Averaging  | σ=25, 3×3     |    — |         — |
| Averaging  | σ=25, 5×5     |    — |         — |
| Averaging  | σ=25, 9×9     |    — |         — |
| Laplacian  | 4-neighbor    |    — |         — |
| Laplacian  | 8-neighbor    |    — |         — |
| High-Boost | k=1           |    — |         — |
| High-Boost | k=1.5         |    — |         — |
| High-Boost | k=2           |    — |         — |
| High-Boost | k=3           |    — |         — |

The actual values must be generated by running the program.

---

# Task 5 — Final AGV Pipeline

Based on the numerical results from Task 4, a final two-stage preprocessing pipeline must be selected.

The recommended structure is:

```text
Input AGV Frame
      ↓
Denoising
      ↓
Sharpening
      ↓
Enhanced Image
      ↓
Obstacle Detection /
Terrain Classification
```

The final recommendation must specify:

* Noise condition
* Averaging filter size
* Laplacian variant or sharpening method
* Boost factor if applicable
* PSNR obtained
* Sharpness obtained
* Reason for selecting the configuration

The final configuration should be justified using the actual experimental results rather than visual judgment alone.

---

# Implementation Requirements

The major filtering operations must be implemented from scratch.

## Required

* Custom 2D correlation
* Custom 2D convolution
* Custom averaging filter
* Custom Laplacian filtering
* Custom sharpening
* Custom unsharp masking
* Custom high-boost filtering
* PSNR calculation
* Sharpness calculation

## Avoid Library Shortcuts

Do not use:

```python
cv2.filter2D()
scipy.signal.convolve2d()
```

or equivalent ready-made filtering functions for the core operations.

Libraries may be used for:

* Reading images
* Saving images
* Displaying images
* Plotting graphs
* Numerical calculations that do not replace the required filtering implementation

```
```

The core spatial-filtering operations should still be implemented manually.

---

# Expected Outputs

The program should generate:

### Input and Degraded Images

```text
1. Clean image
2. Gaussian noisy image — σ=10
3. Gaussian noisy image — σ=25
4. Motion-blurred image
```

### Task 1 Outputs

A labeled grid containing all six averaging-filter results.

```text
σ=10:
3×3
5×5
9×9

σ=25:
3×3
5×5
9×9
```

### Task 2 Outputs

Comparison of:

```text
4-neighbor Laplacian
8-neighbor Laplacian
```

### Task 3 Outputs

Results for:

```text
k=1
k=1.5
k=2
k=3
```

### Task 4 Outputs

* PSNR values
* Sharpness values
* Results table
* Sharpness vs. `k` graph

### Task 5 Output

Final recommended:

```text
Denoising → Sharpening
```

pipeline.

---


# Report Structure

The final report should be approximately **4–6 pages**.

Suggested structure:

```text
1. Introduction

2. Problem Statement

3. Objectives

4. Methodology
   4.1 Input Image
   4.2 Gaussian Noise
   4.3 Motion Blur
   4.4 Averaging Filter
   4.5 Laplacian Sharpening
   4.6 Unsharp Masking
   4.7 High-Boost Filtering

5. Experimental Results
   5.1 Task 1
   5.2 Task 2
   5.3 Task 3
   5.4 PSNR and Sharpness Results

6. Discussion

7. Final Pipeline Recommendation

8. Reflection Questions

9. Conclusion

10. References
```

---

# Conclusion

This laboratory demonstrates how spatial filtering can be used to prepare degraded camera images for an Autonomous Ground Vehicle.

The experiment investigates the complete process of:

```text
Image Degradation
       ↓
Noise Reduction
       ↓
Boundary Recovery
       ↓
Image Sharpening
       ↓
Objective Evaluation
       ↓
Pipeline Selection
```

The final preprocessing configuration should be selected using the measured **PSNR and sharpness values**, while also considering the preservation of navigation-relevant edges.

There is no single predetermined correct filter size or boost factor. The final choice must be supported by the experimental results obtained from the implemented system.

---

## 📌 Deliverables Checklist

* [ ] Commented Python source code
* [ ] From-scratch correlation implementation
* [ ] From-scratch convolution implementation
* [ ] Clean input image
* [ ] Gaussian noise σ=10 image
* [ ] Gaussian noise σ=25 image
* [ ] Motion-blurred image
* [ ] Task 1 output grid
* [ ] Task 2 Laplacian comparison
* [ ] Task 3 unsharp/high-boost results
* [ ] PSNR calculations
* [ ] Sharpness calculations
* [ ] Complete results table
* [ ] Sharpness vs. `k` graph
* [ ] Final pipeline recommendation
* [ ] Reflection question answers
* [ ] 4–6 page PDF report
* [ ] `requirements.txt`
* [ ] `README.md`

---

