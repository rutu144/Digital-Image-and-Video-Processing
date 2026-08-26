from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
                                 Table, TableStyle, PageBreak)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import pandas as pd

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1c", parent=styles["Heading1"], spaceAfter=10))
styles.add(ParagraphStyle(name="H2c", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6))
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], spaceAfter=8, leading=14))
styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontSize=8.5,
                           textColor=colors.grey, spaceAfter=10, leading=11))

story = []

def h1(t): story.append(Paragraph(t, styles["H1c"]))
def h2(t): story.append(Paragraph(t, styles["H2c"]))
def body(t): story.append(Paragraph(t, styles["Body"]))
def caption(t): story.append(Paragraph(t, styles["Caption"]))
def img(path, width=14*cm, aspect=0.60):
    story.append(RLImage(path, width=width, height=width*aspect))
    story.append(Spacer(1, 3))

# ---------------- Title ----------------
story.append(Paragraph("Restoring and Enhancing a Degraded AGV Onboard Camera Feed",
                        styles["Title"]))
story.append(Paragraph("Spatial Filtering Lab Report — PCC-03 Digital Image and Video Processing",
                        styles["Heading3"]))
story.append(Spacer(1, 10))
body("This report implements and evaluates a from-scratch spatial-filtering pre-processing "
     "pipeline for a simulated Autonomous Ground Vehicle (AGV) monocular camera feed. All "
     "correlation, convolution, averaging, Laplacian, and unsharp/high-boost operations are "
     "implemented directly on top of numpy array arithmetic (see <b>filters.py</b>) — no "
     "cv2.filter2D or scipy.signal routines are used anywhere in the pipeline.")
body("<b>Test data:</b> the user's own photograph (a mandrill and infant) was converted to "
     "grayscale and resized to 401x512 to keep processing times reasonable — saved as "
     "<i>../input/00_clean.png</i>. This is a genuinely useful stress-test for spatial "
     "filtering: unlike a flat synthetic scene, the fur is itself a dense, fine-grained "
     "texture, which makes it visibly harder for a filter to tell 'real high-frequency "
     "detail' apart from 'added noise' — a realistic complication for any AGV camera looking "
     "at natural, textured terrain rather than clean geometric obstacles.")

h2("Degradation model applied to the clean frame")
body("Gaussian sensor noise was added at &sigma; = 10 and &sigma; = 25 (0–255 scale, seeded "
     "for reproducibility). A linear motion-blur kernel of length 8 px was built by "
     "rasterizing a line segment through the kernel center and normalizing it to unit sum "
     "(own implementation, no library motion-blur helper).")
img("../input/00_clean.png", width=6*cm, aspect=1.0)
caption("Figure 0. Clean grayscale ground-truth photograph (401x512).")

story.append(Spacer(1, 10))

# ---------------- Task 1 ----------------
h1("Task 1 — Noise Suppression via Averaging")
body("A box/averaging filter was implemented from scratch as a normalized k&times;k kernel "
     "passed through a custom <b>correlate2d()</b> routine (explicit shift-and-accumulate "
     "correlation sum with reflect-padding). It was applied to both noisy variants at "
     "3&times;3, 5&times;5 and 9&times;9.")
img("../output/figures/task1_averaging_grid.png", aspect=0.5)
caption("Figure 1. Box-filter denoising across kernel size (columns) and noise level (rows). "
        "PSNR is against the clean ground truth; S is the variance-of-Laplacian sharpness metric.")
body("<b>Trade-off observed:</b> as kernel size increases, noise is suppressed more "
     "aggressively — the sharpness metric falls sharply as kernel grows (S=204 &rarr; 45 &rarr; 8 "
     "at &sigma;=10) — but so does genuine detail: the fine fur texture, whisker-level "
     "strands, and the eye/nose ridge lines all visibly smear out, and by 9x9 the faces read "
     "as a soft blur with barely any of the original fur structure left (Figure 1). Unlike a "
     "flat synthetic scene, PSNR here falls monotonically as kernel size grows (24.38 &rarr; "
     "22.46 &rarr; 21.67 dB at &sigma;=10) rather than peaking at some middle size — because "
     "the fur itself is dense, fine-grained texture that a box filter cannot distinguish from "
     "noise, so every kernel enlargement keeps discarding real signal along with noise. This "
     "is an important, realistic lesson for an AGV: on naturally textured terrain (gravel, "
     "foliage, grass) rather than clean geometric obstacles, a bigger averaging kernel is "
     "<i>not free</i> — every increment costs real detail, and the smallest kernel that still "
     "adequately suppresses noise is the right default, not the largest one available.")

story.append(Spacer(1, 6))

# ---------------- Task 2 ----------------
h1("Task 2 — Boundary Recovery via Laplacian Sharpening")
body("The 4-neighbor and 8-neighbor Laplacian kernels were both implemented and applied to "
     "the motion-blurred frame, with sharpened output g = f + c&middot;&nabla;&sup2;f, c = -1.")
img("../output/figures/task2_laplacian_comparison.png", aspect=0.58)
caption("Figure 2. Top row: motion-blurred input and raw Laplacian response maps (signed, "
        "min-max normalized for display). Bottom row: clean reference and the two sharpened outputs.")
body("<b>4-neighbor vs 8-neighbor:</b> the 8-neighbor kernel is mathematically sensitive to "
     "intensity change along the diagonal directions in addition to the axis-aligned "
     "directions the 4-neighbor kernel sees, and its center weight (-8) is twice as strong "
     "relative to its neighbors. On this densely-textured photo that difference is dramatic: "
     "the 8-neighbor response reacts to nearly every individual fur strand as an edge "
     "(sharpness metric 31479 vs 5443 for 4-neighbor), and the resulting sharpened image "
     "(Figure 2, bottom-right) looks aggressively over-processed — gritty, high-contrast "
     "noise-like texture rather than cleanly recovered structure — which is reflected in a "
     "large PSNR penalty: 13.48 dB for 8-neighbor vs 19.94 dB for 4-neighbor. <b>The "
     "4-neighbor Laplacian recovers boundaries far more usefully here</b>: it still "
     "sharpens the key structural edges (eye outlines, nose ridge lines, ear boundary) "
     "without turning the entire fur field into spurious high-frequency noise, which matters "
     "because a downstream edge or obstacle detector would otherwise be flooded with false "
     "edges from texture alone.")

story.append(Spacer(1, 6))

# ---------------- Task 3 ----------------
h1("Task 3 — Unsharp Masking / High-Boost Design")
body("Unsharp masking (k=1) and high-boost filtering (k=1.5, 2, 3) were applied to the "
     "&sigma;=10 noisy image from Task 1, using g = f + k&middot;(f - blur(f)) with a 5x5 "
     "box blur as the low-pass stage.")
img("../output/figures/task3_unsharp_grid.png", aspect=0.32)
caption("Figure 3. High-boost sweep on the sigma=10 noisy input.")
img("../output/figures/task3_sharpness_vs_k.png", width=10*cm, aspect=0.72)
caption("Figure 4. Sharpness metric (rising) and PSNR (falling) as functions of boost factor k.")
body("<b>Where k stops helping:</b> exactly as with the earlier averaging test, PSNR falls "
     "monotonically as k rises (19.38 dB at k=1 down to 11.31 dB at k=3) while the sharpness "
     "metric climbs sharply (20996 &rarr; 84568) — but here that climb is almost entirely "
     "amplified sensor grain riding on top of already-dense fur texture, not newly recovered "
     "structure; visually the k=3 panel in Figure 3 looks gritty and speckled rather than "
     "'sharper' in any useful sense. There is no k above 1 in the tested range that improves "
     "PSNR at all, so on this noisy, texture-rich input <b>k should stay at 1</b>. This is an "
     "even stronger case for restraint than a simple geometric scene: on natural texture, "
     "the unsharp mask's high-frequency component (f - blur(f)) is dominated by noise the "
     "moment any noise is present, because real fur/foliage detail already occupies that same "
     "frequency band, leaving high-boost filtering very little genuine signal to amplify "
     "safely.")

story.append(Spacer(1, 6))

# ---------------- Task 4 ----------------
h1("Task 4 — Objective Evaluation")
body("PSNR (against the clean ground truth) and a sharpness metric — variance of the "
     "4-neighbor Laplacian response, computed with our own <b>laplacian_response()</b> — "
     "were tabulated for every configuration tested.")

df = pd.read_csv("../output/task4_results_table.csv")
table_data = [list(df.columns)] + df.astype(str).values.tolist()
t = Table(table_data, repeatRows=1, colWidths=[3.7*cm, 1.9*cm, 1.9*cm, 2.5*cm, 2.0*cm, 2.7*cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 7.6),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
]))
story.append(t)
story.append(Spacer(1, 10))
body("Reading the table: within Task 1, PSNR is highest at 3x3 for <i>both</i> noise levels "
     "(24.38 dB at &sigma;=10, 23.43 dB at &sigma;=25) and falls steadily as kernel size "
     "grows — the dense fur texture means larger kernels simply discard real detail faster "
     "than they remove noise. Within Task 2, the 4-neighbor Laplacian dominates the "
     "8-neighbor variant by roughly 6.5 dB. Within Task 3, PSNR is maximized at k=1 and "
     "falls for every larger k, again because this texture-rich input leaves little headroom "
     "for boosting without mostly boosting grain.")

story.append(Spacer(1, 6))

# ---------------- Task 5 ----------------
h1("Task 5 — Pipeline Recommendation")
body("<b>Recommended two-stage pipeline: 3x3 box-filter denoise &rarr; 4-neighbor Laplacian "
     "sharpen (c = -1), applied in that order.</b>")
body("<b>Stage 1 (denoise, 3x3 average):</b> chosen because, unlike a scene made of flat "
     "geometric regions, this photograph is dominated by fine natural texture (fur), where a "
     "box filter cannot separate 'real detail' from 'sensor noise' — every increase in "
     "kernel size costs real image content roughly as fast as it removes noise. The Task 4 "
     "data confirms this directly: PSNR is highest at 3x3 for both tested noise levels "
     "(24.38 dB / 23.43 dB) and drops steadily at 5x5 and 9x9, with 9x9 visibly erasing "
     "almost all fur structure (Figure 1). For this kind of input, the smallest kernel that "
     "still meaningfully reduces the noisy input's PSNR gap is the right choice, not a larger "
     "'safety margin' kernel.")
body("<b>Stage 2 (sharpen, 4-neighbor Laplacian):</b> chosen over the 8-neighbor variant "
     "because it recovered structural edges at 19.94 dB vs 13.48 dB (Task 2 table) without "
     "turning the fur field into a field of spurious high-frequency noise. Plain Laplacian "
     "sharpening (c=-1) was preferred over high-boost unsharp masking at k&gt;1 for the "
     "final stage because Task 3 showed every tested k&gt;1 reduced PSNR versus k=1 on this "
     "input — texture-rich AGV terrain leaves very little safe headroom for boosting, so the "
     "simplest, single-pass Laplacian sharpen is both the cheapest and the best-performing "
     "option the Task 4 data supports. A team wanting a tunable dial could still use unsharp "
     "masking at k=1 specifically (not higher) on the denoised output.")
body("<b>Deployed parameters:</b> box_filter(size=3) &rarr; laplacian_sharpen(variant='4', "
     "c=-1). Note this differs from a run against a cleaner, more geometric synthetic scene, "
     "where a larger 5x5 kernel was the more robust choice — the correct kernel size is not a "
     "universal constant, it depends on how much real high-frequency content the specific "
     "terrain/scene actually contains, which is exactly why Task 4's own measurements, not a "
     "fixed rule of thumb, should drive the final choice.")

story.append(PageBreak())

# ---------------- Reflection Questions ----------------
h1("Reflection Questions")

h2("1. Does swapping the pipeline order (sharpen-then-denoise) change Task 4's numbers?")
body("Empirically: no. Denoise&rarr;sharpen and sharpen&rarr;denoise (5x5 box, then unsharp "
     "k=1.5, on the &sigma;=10 image) produced <b>identical</b> PSNR (21.983 dB both ways) "
     "and identical sharpness (267.953 both ways) — see "
     "<i>../output/reflection1_order_of_operations.csv</i>. This is expected for linear, "
     "shift-invariant filters: both the box filter and the unsharp-mask operator "
     "(g = (1+k)f - k&middot;blur(f)) are linear combinations of convolutions, and linear "
     "time-invariant systems commute — convolving with kernel A then kernel B gives the "
     "same result as B then A, since convolution is associative and commutative. The order "
     "would matter the moment a nonlinearity enters the pipeline — e.g. clipping to [0,255] "
     "between the two stages, or a nonlinear denoiser like a median filter — because clipping "
     "and rank-order operations do not commute with convolution.")

h2("2. Does true convolution (flipped kernel) change the Task 2 output?")
body("No, not for the kernels used here. Measured directly: the maximum absolute pixel "
     "difference between correlate2d(blurred, LAPLACIAN_4) and convolve2d(blurred, "
     "LAPLACIAN_4) is exactly 0.0 (see run_log.txt). This is because LAPLACIAN_4 = "
     "[[0,1,0],[1,-4,1],[0,1,0]] is point-symmetric about its center — flipping it 180 "
     "degrees returns the identical kernel — and the same is true of LAPLACIAN_8. Since "
     "convolution is correlation with a flipped kernel, a symmetric kernel makes the two "
     "operations mathematically identical. The motion-blur kernel used to degrade the image, "
     "in contrast, is a directional line segment (not point-symmetric for most angles), so "
     "for that kernel correlation and convolution would generally differ.")

h2("3. Why might PSNR disagree with human/task-relevant judgment of the Task 1 outputs?")
body("PSNR is a single global number derived from the mean squared pixel-wise intensity "
     "error against ground truth — it treats every pixel equally regardless of whether that "
     "pixel sits on a flat background or on a boundary that the navigation stack actually "
     "depends on. On this photograph, PSNR happens to be highest at the smallest kernel "
     "(3x3) and falls steadily as kernel size grows — but that number alone does not tell "
     "the full story: the sharpness metric at 3x3 is also the highest of the three (S=204 at "
     "&sigma;=10), and on this densely-textured image a meaningful share of that surviving "
     "'sharpness' is still leftover noise grain sitting on top of the fur texture, not purely "
     "clean recovered detail — the two are genuinely hard to tell apart in a texture-rich "
     "scene, which is precisely why PSNR (a pixel-fidelity number) and the sharpness metric "
     "(an edge-energy number) can each look favorable for reasons that do not fully agree "
     "with what a human viewer — or a downstream obstacle/feature detector — would actually "
     "judge as 'clean.' A human scanning the 3x3 result in Figure 1 can still see visible "
     "speckle in the fur even though PSNR ranks it best; a human scanning the 9x9 result "
     "sees a much smoother image with lower PSNR. Neither metric alone should be trusted in "
     "isolation, which is why Task 4 tabulates both rather than optimizing PSNR alone.")

h2("4. At what noise level does sharpening stop recovering usable edges?")
body("A sigma sweep (&sigma; = 10, 25, 40, 60, 80, 100) with unsharp/high-boost k &isin; "
     "{1, 1.5, 2, 3, 4} was run (Figure 5; full numbers in "
     "<i>../output/reflection4_sigma_sweep.csv</i>). PSNR falls steeply and monotonically with "
     "&sigma; for every k, dropping below 10 dB by &sigma;&asymp;40 (k=1) and going negative "
     "for the higher-k configurations by &sigma;=60-80 — a negative-dB regime means the "
     "processed image is now farther from the ground truth than the noisy image was, i.e. the "
     "'sharpening' is amplifying noise faster than any genuine signal is left to recover. "
     "Because this photograph is already densely textured even in its clean state, this "
     "crossover happens at a comparable or slightly lower &sigma; than it would for a "
     "geometric scene — there is less 'flat, easy' background for averaging to fall back on. "
     "Beyond roughly &sigma;&asymp;40 on this image, higher k stops being a viable dial at "
     "all: every increase in k makes PSNR worse, not better, meaning spatial-domain "
     "sharpening has run out of true edge signal to amplify and is amplifying grain "
     "exclusively. This is the fundamental limit of spatial-domain pre-processing: it can "
     "only re-weight information already present in the frame, it cannot manufacture edge "
     "information that noise has genuinely destroyed. Beyond this point, the pipeline would "
     "need either (a) frequency-domain or edge-preserving denoising (e.g. bilateral/ "
     "non-local-means filtering, which is nonlinear and edge-aware) before any sharpening "
     "is attempted, or (b) a multi-frame approach (temporal averaging across consecutive AGV "
     "frames) to reduce noise below this threshold before a single-frame spatial filter is "
     "asked to do any sharpening at all.")
img("../output/figures/reflection4_sigma_sweep.png", width=10*cm, aspect=0.72)
caption("Figure 5. PSNR vs noise sigma for each boost factor k. Note the crossover to negative "
        "dB at high sigma / high k, indicating sharpening becomes net-harmful.")

doc = SimpleDocTemplate("../output/AGV_Spatial_Filtering_Lab_Report.pdf", pagesize=A4,
                         topMargin=1.6*cm, bottomMargin=1.6*cm,
                         leftMargin=1.8*cm, rightMargin=1.8*cm)
doc.build(story)
print("Report built.")
