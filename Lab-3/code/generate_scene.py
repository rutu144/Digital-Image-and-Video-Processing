"""
generate_scene.py
------------------
Produces a synthetic grayscale "AGV forward-camera frame": a path/road
receding to a horizon, flanked by ground texture, with a few obstacle
blocks that have clear, high-contrast boundaries (needed so Task 2/3
sharpening has something meaningful to recover).

NOTE FOR THE STUDENT: The problem statement explicitly allows "a personal
photo converted to grayscale". If you have your own AGV-like photo
(a path, corridor, or parking area), just drop it in as
../input/00_clean.png (grayscale, e.g. 512x512) and skip this script —
everything downstream (run_experiments.py) only assumes a grayscale
uint8 array of that name. This synthetic frame is provided so the full
pipeline runs end-to-end even without an uploaded photo.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_agv_frame(size=512, seed=7):
    rng = np.random.default_rng(seed)
    img = Image.new("L", (size, size), color=60)
    draw = ImageDraw.Draw(img)

    # Sky / far background gradient (top portion)
    horizon = int(size * 0.38)
    for y in range(horizon):
        shade = int(150 + 60 * (y / horizon))
        draw.line([(0, y), (size, y)], fill=shade)

    # Ground gradient below horizon (darker near vehicle -> lighter far away,
    # mimicking perspective/dust haze at distance)
    for y in range(horizon, size):
        t = (y - horizon) / (size - horizon)
        shade = int(70 + 40 * (1 - t))
        draw.line([(0, y), (size, y)], fill=shade)

    # Perspective path/road (trapezoid narrowing toward horizon)
    road_top_w = size * 0.10
    road_bot_w = size * 0.62
    cx = size / 2
    road_poly = [
        (cx - road_top_w / 2, horizon),
        (cx + road_top_w / 2, horizon),
        (cx + road_bot_w / 2, size),
        (cx - road_bot_w / 2, size),
    ]
    draw.polygon(road_poly, fill=110)

    # Lane-ish texture lines on the road for extra edge content
    for i in range(1, 6):
        t = i / 6
        y = int(horizon + t * (size - horizon))
        w_at_y = road_top_w + t * (road_bot_w - road_top_w)
        x0 = cx - w_at_y / 2 + w_at_y * 0.08
        x1 = cx - w_at_y / 2 + w_at_y * 0.18
        draw.line([(x0, y), (x1, y)], fill=150, width=max(1, int(2 + 4 * t)))

    # Obstacle blocks (crates / rocks) with sharp rectangular / polygon edges
    obstacles = [
        (0.30, 0.62, 0.14, 0.16, 40),   # (x_frac, y_frac, w_frac, h_frac, shade)
        (0.58, 0.70, 0.16, 0.14, 200),
        (0.44, 0.85, 0.10, 0.10, 30),
        (0.20, 0.90, 0.12, 0.09, 220),
    ]
    for xf, yf, wf, hf, shade in obstacles:
        x0, y0 = xf * size, yf * size
        w, h = wf * size, hf * size
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=shade, outline=max(0, shade - 60))
        # a slight cast shadow for realism (still crisp-edged)
        draw.rectangle([x0 + w * 0.05, y0 + h, x0 + w * 1.05, y0 + h * 1.15],
                        fill=max(20, shade - 90))

    # A distant "obstacle" near horizon (small, tests fine-edge recovery)
    draw.rectangle([cx - 8, horizon - 10, cx + 8, horizon + 6], fill=15)

    arr = np.array(img).astype(np.float64)

    # Very mild base texture noise so the "clean" image isn't a flat CG image
    arr += rng.normal(0, 2.0, arr.shape)
    arr = np.clip(arr, 0, 255)

    # A touch of blur to remove hard anti-aliasing staircase edges from
    # PIL's polygon rasterizer, so degradation steps applied later are the
    # dominant, controlled source of blur/noise (not rasterization artifacts)
    img_final = Image.fromarray(arr.astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    return np.array(img_final).astype(np.uint8)


if __name__ == "__main__":
    frame = generate_agv_frame()
    Image.fromarray(frame).save("../input/00_clean.png")
    print("Saved ../input/00_clean.png", frame.shape, frame.dtype)
