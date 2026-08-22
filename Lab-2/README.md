# DIVP Lab 2 – Histogram Matching and CLAHE

## Digital Image and Video Processing Laboratory

This practical focuses on two important image enhancement techniques:

1. **Histogram Matching (Histogram Specification)**
2. **Contrast Limited Adaptive Histogram Equalization (CLAHE)**

Both methods are implemented in Python from first principles using NumPy and Matplotlib.

---

# 1. Aim

To implement and study **Histogram Matching (Histogram Specification)** and **Contrast Limited Adaptive Histogram Equalization (CLAHE)** for image enhancement, and to analyze their performance using visual and quantitative measures.

---

# 2. Objectives

The objectives of this practical are:

- To understand image histograms and cumulative distribution functions.
- To implement histogram calculation from first principles.
- To calculate the CDF of an image.
- To implement global histogram equalization.
- To implement histogram matching between a source image and a reference image.
- To study why histogram equalization is not suitable when a specific tonal appearance is required.
- To match differently exposed images with a common reference image.
- To compare the histograms and CDFs of source, reference, and matched images.
- To generate an analytically defined target histogram.
- To study the failure cases of histogram matching.
- To implement CLAHE from first principles.
- To compare global histogram equalization with CLAHE.
- To study the effect of clip limit and tile size.
- To calculate entropy and local contrast.
- To analyze the failure modes of different CLAHE parameter settings.

---

# 3. Problem Statement

## Part A – Histogram Matching

Consider a film restoration application where several frames of the same scene have different brightness and tonal characteristics because of changing lighting conditions, aging film, or different camera rolls.

A colorist selects one frame as the reference frame.

The objective is to transform other frames so that their intensity distributions follow the tonal distribution of the reference frame.

This problem is solved using **Histogram Matching**, also called **Histogram Specification**.

---

## Part B – CLAHE

Consider a medical X-ray image where important details occur in both dark and bright regions.

Global histogram equalization uses one histogram for the entire image. As a result, important local details may not receive sufficient enhancement.

The objective is to enhance local contrast independently in different regions while preventing excessive noise amplification.

This problem is solved using **Contrast Limited Adaptive Histogram Equalization (CLAHE)**.

---

# 4. Technologies Used

- Python 3
- NumPy
- Matplotlib
- pathlib

---

# 5. Installation

First, make sure Python 3 is installed.

Check the Python version:

```bash
python3 --version
