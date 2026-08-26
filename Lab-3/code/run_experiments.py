"""
run_experiments.py
-------------------
Runs the full AGV Spatial Filtering Lab pipeline:
  - builds the degraded test set (noise + motion blur)
  - Task 1: box filter denoising, 3 kernel sizes x 2 noise levels
  - Task 2: 4-neighbor vs 8-neighbor Laplacian sharpening of blurred image
  - Task 3: unsharp masking / high-boost sweep over k
  - Task 4: PSNR + sharpness-metric table for every configuration
  - Reflection experiments: order-swap (sharpen->denoise vs denoise->sharpen),
    correlation-vs-convolution check for the Laplacian, and a sigma sweep
    to find the point where sharpening stops recovering usable edges.

All ../output/figures/images/csv land in ../output/ and ../output/figures/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from filters import (
    box_filter, laplacian_response, laplacian_sharpen,
    unsharp_highboost, add_gaussian_noise, apply_motion_blur,
    motion_blur_kernel, psnr, sharpness_metric, clip_to_uint8,
    correlate2d, convolve2d, LAPLACIAN_4, LAPLACIAN_8
)

np.set_printoptions(suppress=True)

# ----------------------------------------------------------------------
# 0. Load clean ground truth
# ----------------------------------------------------------------------
clean = np.array(Image.open("../input/00_clean.png").convert("L")).astype(np.float64)

# ----------------------------------------------------------------------
# 1. Build degraded test data
# ----------------------------------------------------------------------
noisy_s10 = add_gaussian_noise(clean, sigma=10, seed=1)
noisy_s25 = add_gaussian_noise(clean, sigma=25, seed=2)
blurred = apply_motion_blur(clean, length=8, angle_deg=0)

Image.fromarray(clip_to_uint8(noisy_s10)).save("../output/01_noisy_sigma10.png")
Image.fromarray(clip_to_uint8(noisy_s25)).save("../output/02_noisy_sigma25.png")
Image.fromarray(clip_to_uint8(blurred)).save("../output/03_motion_blurred.png")

print("Degraded test data generated.")
print(f"  clean vs noisy(s10) PSNR = {psnr(clean, noisy_s10):.2f} dB")
print(f"  clean vs noisy(s25) PSNR = {psnr(clean, noisy_s25):.2f} dB")
print(f"  clean vs blurred    PSNR = {psnr(clean, blurred):.2f} dB")

results = []  # collects every (config, PSNR, sharpness) row for Task 4

# ----------------------------------------------------------------------
# TASK 1 — Averaging filter denoising, 3x3/5x5/9x9 on sigma=10 and sigma=25
# ----------------------------------------------------------------------
kernel_sizes = [3, 5, 9]
noisy_variants = {"sigma10": noisy_s10, "sigma25": noisy_s25}

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for row, (label, noisy_img) in enumerate(noisy_variants.items()):
    axes[row, 0].imshow(clip_to_uint8(noisy_img), cmap="gray", vmin=0, vmax=255)
    axes[row, 0].set_title(f"Noisy input ({label})\nPSNR={psnr(clean, noisy_img):.2f} dB")
    axes[row, 0].axis("off")

    for col, k in enumerate(kernel_sizes, start=1):
        filtered = box_filter(noisy_img, k)
        p = psnr(clean, filtered)
        s = sharpness_metric(filtered)
        results.append({
            "task": "Task1_Averaging", "noise_sigma": label.replace("sigma", ""),
            "kernel_or_k": f"{k}x{k}", "laplacian_variant": "-",
            "PSNR_dB": round(p, 3), "sharpness_varLap": round(s, 3)
        })
        axes[row, col].imshow(clip_to_uint8(filtered), cmap="gray", vmin=0, vmax=255)
        axes[row, col].set_title(f"{k}x{k} avg\nPSNR={p:.2f} dB, S={s:.0f}")
        axes[row, col].axis("off")

        Image.fromarray(clip_to_uint8(filtered)).save(
            f"../output/task1_{label}_avg{k}x{k}.png")

plt.suptitle("Task 1 — Box/Averaging Filter Denoising (rows: noise level, cols: kernel size)")
plt.tight_layout()
plt.savefig("../output/figures/task1_averaging_grid.png", dpi=140)
plt.close()
print("Task 1 done -> ../output/figures/task1_averaging_grid.png")

# ----------------------------------------------------------------------
# TASK 2 — Laplacian sharpening: 4-neighbor vs 8-neighbor on blurred image
# ----------------------------------------------------------------------
lap4_resp = laplacian_response(blurred, variant="4")
lap8_resp = laplacian_response(blurred, variant="8")
sharp4 = laplacian_sharpen(blurred, variant="4", c=-1.0)
sharp8 = laplacian_sharpen(blurred, variant="8", c=-1.0)

for name, arr in [("lap4_response", lap4_resp), ("lap8_response", lap8_resp),
                   ("sharpened_lap4", sharp4), ("sharpened_lap8", sharp8)]:
    # normalize response maps to 0-255 for display/saving (they're signed)
    disp = arr - arr.min()
    disp = disp / (disp.max() + 1e-9) * 255
    Image.fromarray(disp.astype(np.uint8)).save(f"../output/task2_{name}.png")

p4, p8 = psnr(clean, sharp4), psnr(clean, sharp8)
s4, s8 = sharpness_metric(sharp4), sharpness_metric(sharp8)
results.append({"task": "Task2_Laplacian", "noise_sigma": "-", "kernel_or_k": "3x3",
                 "laplacian_variant": "4-neighbor", "PSNR_dB": round(p4, 3),
                 "sharpness_varLap": round(s4, 3)})
results.append({"task": "Task2_Laplacian", "noise_sigma": "-", "kernel_or_k": "3x3",
                 "laplacian_variant": "8-neighbor", "PSNR_dB": round(p8, 3),
                 "sharpness_varLap": round(s8, 3)})

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
axes[0, 0].imshow(clip_to_uint8(blurred), cmap="gray"); axes[0, 0].set_title("Motion-blurred input"); axes[0, 0].axis("off")
axes[0, 1].imshow(lap4_resp, cmap="gray"); axes[0, 1].set_title("4-neighbor Laplacian response"); axes[0, 1].axis("off")
axes[0, 2].imshow(lap8_resp, cmap="gray"); axes[0, 2].set_title("8-neighbor Laplacian response"); axes[0, 2].axis("off")
axes[1, 0].imshow(clip_to_uint8(clean), cmap="gray"); axes[1, 0].set_title("Clean ground truth"); axes[1, 0].axis("off")
axes[1, 1].imshow(clip_to_uint8(sharp4), cmap="gray"); axes[1, 1].set_title(f"Sharpened (4-nbr)\nPSNR={p4:.2f} dB, S={s4:.0f}"); axes[1, 1].axis("off")
axes[1, 2].imshow(clip_to_uint8(sharp8), cmap="gray"); axes[1, 2].set_title(f"Sharpened (8-nbr)\nPSNR={p8:.2f} dB, S={s8:.0f}"); axes[1, 2].axis("off")
plt.suptitle("Task 2 — Laplacian Sharpening: 4-neighbor vs 8-neighbor")
plt.tight_layout()
plt.savefig("../output/figures/task2_laplacian_comparison.png", dpi=140)
plt.close()
print(f"Task 2 done -> 4-nbr PSNR={p4:.2f}/S={s4:.0f}  |  8-nbr PSNR={p8:.2f}/S={s8:.0f}")

# ----------------------------------------------------------------------
# TASK 3 — Unsharp masking (k=1) and high-boost (k=1.5,2,3) on sigma=10 image
# ----------------------------------------------------------------------
k_values = [1.0, 1.5, 2.0, 3.0]
k_results = []
fig, axes = plt.subplots(1, len(k_values), figsize=(4 * len(k_values), 4.5))
for i, k in enumerate(k_values):
    out = unsharp_highboost(noisy_s10, k=k, blur_size=5)
    p = psnr(clean, out)
    s = sharpness_metric(out)
    k_results.append({"k": k, "PSNR_dB": p, "sharpness_varLap": s})
    results.append({"task": "Task3_UnsharpHighBoost", "noise_sigma": "10",
                     "kernel_or_k": f"k={k}", "laplacian_variant": "-",
                     "PSNR_dB": round(p, 3), "sharpness_varLap": round(s, 3)})
    axes[i].imshow(clip_to_uint8(out), cmap="gray")
    label = "Unsharp mask (k=1)" if k == 1.0 else f"High-boost k={k}"
    axes[i].set_title(f"{label}\nPSNR={p:.2f} dB, S={s:.0f}")
    axes[i].axis("off")
    Image.fromarray(clip_to_uint8(out)).save(f"../output/task3_k{k}.png")

plt.suptitle("Task 3 — Unsharp Masking / High-Boost Sweep (input: sigma=10 noisy image)", y=1.04)
plt.tight_layout()
plt.savefig("../output/figures/task3_unsharp_grid.png", dpi=140, bbox_inches="tight")
plt.close()

# sharpness-vs-k plot
kdf = pd.DataFrame(k_results)
fig, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(kdf["k"], kdf["sharpness_varLap"], "o-", color="tab:blue", label="Sharpness (var of Laplacian)")
ax1.set_xlabel("Boost factor k")
ax1.set_ylabel("Sharpness metric (variance of Laplacian)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.plot(kdf["k"], kdf["PSNR_dB"], "s--", color="tab:red", label="PSNR vs clean")
ax2.set_ylabel("PSNR (dB) vs clean ground truth", color="tab:red")
ax2.tick_params(axis="y", labelcolor="tab:red")

plt.title("Task 3 — Sharpness and PSNR vs Boost Factor k")
fig.tight_layout()
plt.savefig("../output/figures/task3_sharpness_vs_k.png", dpi=140)
plt.close()
print("Task 3 done -> ../output/figures/task3_unsharp_grid.png, ../output/figures/task3_sharpness_vs_k.png")
print(kdf)

# ----------------------------------------------------------------------
# TASK 4 — full results table
# ----------------------------------------------------------------------
df = pd.DataFrame(results)
df.to_csv("../output/task4_results_table.csv", index=False)
print("\nTask 4 full results table:")
print(df.to_string(index=False))

# ----------------------------------------------------------------------
# REFLECTION EXPERIMENT 1 — order of operations: denoise->sharpen vs sharpen->denoise
# ----------------------------------------------------------------------
# Use sigma=10 noisy image, box(5x5) as the denoiser, unsharp k=1.5 as sharpener
denoise_then_sharpen = unsharp_highboost(box_filter(noisy_s10, 5), k=1.5, blur_size=5)
sharpen_then_denoise = box_filter(unsharp_highboost(noisy_s10, k=1.5, blur_size=5), 5)

p_ds = psnr(clean, denoise_then_sharpen)
p_sd = psnr(clean, sharpen_then_denoise)
s_ds = sharpness_metric(denoise_then_sharpen)
s_sd = sharpness_metric(sharpen_then_denoise)

order_df = pd.DataFrame([
    {"order": "denoise -> sharpen", "PSNR_dB": round(p_ds, 3), "sharpness_varLap": round(s_ds, 3)},
    {"order": "sharpen -> denoise", "PSNR_dB": round(p_sd, 3), "sharpness_varLap": round(s_sd, 3)},
])
order_df.to_csv("../output/reflection1_order_of_operations.csv", index=False)
print("\nReflection 1 (order of operations):")
print(order_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
axes[0].imshow(clip_to_uint8(denoise_then_sharpen), cmap="gray")
axes[0].set_title(f"Denoise->Sharpen\nPSNR={p_ds:.2f} dB, S={s_ds:.0f}")
axes[0].axis("off")
axes[1].imshow(clip_to_uint8(sharpen_then_denoise), cmap="gray")
axes[1].set_title(f"Sharpen->Denoise\nPSNR={p_sd:.2f} dB, S={s_sd:.0f}")
axes[1].axis("off")
plt.suptitle("Reflection 1 — Does operation order matter?")
plt.tight_layout()
plt.savefig("../output/figures/reflection1_order.png", dpi=140)
plt.close()

# ----------------------------------------------------------------------
# REFLECTION EXPERIMENT 2 — correlation vs true convolution for the Laplacian
# ----------------------------------------------------------------------
corr_result = correlate2d(blurred, LAPLACIAN_4, mode="same")
conv_result = convolve2d(blurred, LAPLACIAN_4, mode="same")
max_abs_diff = float(np.max(np.abs(corr_result - conv_result)))
print(f"\nReflection 2: max |correlation - convolution| for LAPLACIAN_4 = {max_abs_diff}")
print("(LAPLACIAN_4 is point-symmetric, i.e. kernel == kernel flipped 180deg,")
print(" so correlation and convolution are mathematically identical for it.)")

# ----------------------------------------------------------------------
# REFLECTION EXPERIMENT 3 — sigma sweep: when does sharpening stop helping?
# ----------------------------------------------------------------------
sigma_sweep = [10, 25, 40, 60, 80, 100]
sweep_rows = []
for sig in sigma_sweep:
    noisy = add_gaussian_noise(clean, sigma=sig, seed=42)
    best_k, best_s, best_p = None, -1, None
    for k in [1.0, 1.5, 2.0, 3.0, 4.0]:
        out = unsharp_highboost(noisy, k=k, blur_size=5)
        s = sharpness_metric(out)
        p = psnr(clean, out)
        if p > (best_p if best_p is not None else -999):
            pass
        sweep_rows.append({"sigma": sig, "k": k, "PSNR_dB": round(p, 3),
                            "sharpness_varLap": round(s, 3)})

sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv("../output/reflection4_sigma_sweep.csv", index=False)
print("\nReflection 4 (sigma sweep, PSNR degradation as sigma grows):")
print(sweep_df.pivot(index="sigma", columns="k", values="PSNR_dB"))

plt.figure(figsize=(7, 5))
for k in [1.0, 1.5, 2.0, 3.0, 4.0]:
    sub = sweep_df[sweep_df.k == k]
    plt.plot(sub.sigma, sub.PSNR_dB, "o-", label=f"k={k}")
plt.xlabel("Noise sigma")
plt.ylabel("PSNR (dB) vs clean ground truth")
plt.title("Reflection 4 — PSNR vs noise sigma for each boost factor k")
plt.legend()
plt.tight_layout()
plt.savefig("../output/figures/reflection4_sigma_sweep.png", dpi=140)
plt.close()

print("\nALL DONE. See ../output/ and ../output/figures/ directories.")
