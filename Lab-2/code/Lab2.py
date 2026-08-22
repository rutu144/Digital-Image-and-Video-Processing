import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# COMMON FUNCTIONS
# ============================================================

def calculate_histogram(image):
    image = np.asarray(image).astype(np.uint8)

    histogram = np.zeros(256, dtype=np.int64)

    for pixel in image.ravel():
        histogram[pixel] += 1

    return histogram


def calculate_cdf(histogram):
    cdf = np.cumsum(histogram)

    if cdf[-1] == 0:
        return np.zeros(256)

    cdf = cdf / cdf[-1]

    return cdf


# ============================================================
# HISTOGRAM EQUALIZATION
# ============================================================

def histogram_equalization(image):

    histogram = calculate_histogram(image)

    cdf = calculate_cdf(histogram)

    lookup_table = np.floor(
        255 * cdf
    ).astype(np.uint8)

    output = lookup_table[image]

    return output


# ============================================================
# HISTOGRAM MATCHING
# ============================================================

def histogram_matching(source, reference):

    source = np.asarray(source).astype(np.uint8)
    reference = np.asarray(reference).astype(np.uint8)

    source_histogram = calculate_histogram(source)

    reference_histogram = calculate_histogram(reference)

    source_cdf = calculate_cdf(
        source_histogram
    )

    reference_cdf = calculate_cdf(
        reference_histogram
    )

    lookup_table = np.zeros(
        256,
        dtype=np.uint8
    )

    for source_level in range(256):

        source_probability = (
            source_cdf[source_level]
        )

        difference = np.abs(
            reference_cdf -
            source_probability
        )

        reference_level = np.argmin(
            difference
        )

        lookup_table[source_level] = (
            reference_level
        )

    output = lookup_table[source]

    return output, lookup_table


# ============================================================
# ANALYTICAL TARGET HISTOGRAM
# ============================================================

def create_moody_target_histogram():

    x = np.arange(256)

    target = (
        0.78 *
        np.exp(
            -0.5 *
            ((x - 55) / 25) ** 2
        )
        +
        0.22 *
        np.exp(
            -0.5 *
            ((x - 145) / 32) ** 2
        )
    )

    target = target / np.sum(target)

    return target


def histogram_match_analytical_target(
        source,
        target_histogram):

    source_histogram = calculate_histogram(
        source
    )

    source_cdf = calculate_cdf(
        source_histogram
    )

    target_cdf = np.cumsum(
        target_histogram
    )

    target_cdf = (
        target_cdf /
        target_cdf[-1]
    )

    lookup_table = np.zeros(
        256,
        dtype=np.uint8
    )

    for source_level in range(256):

        source_probability = (
            source_cdf[source_level]
        )

        difference = np.abs(
            target_cdf -
            source_probability
        )

        target_level = np.argmin(
            difference
        )

        lookup_table[source_level] = (
            target_level
        )

    output = lookup_table[source]

    return output, lookup_table


# ============================================================
# FAILURE CASE
# ============================================================

def create_failure_case_images():

    H = 300
    W = 450

    source = np.zeros(
        (H, W),
        dtype=np.uint8
    )

    reference = np.zeros(
        (H, W),
        dtype=np.uint8
    )

    # Source image

    source[:150, :] = 40

    source[150:300, 100:350] = 210

    source[220:300, :] = 110

    # Reference image

    reference[:150, :] = 210

    reference[150:300, 100:350] = 40

    reference[220:300, :] = 110

    return source, reference


# ============================================================
# CLAHE
# ============================================================

def clahe_first_principles(
        image,
        tile_grid=(8, 8),
        clip_limit=2.0):

    image = np.asarray(
        image
    ).astype(np.uint8)

    H, W = image.shape

    number_of_rows = tile_grid[0]
    number_of_columns = tile_grid[1]

    tile_height = int(
        np.ceil(
            H / number_of_rows
        )
    )

    tile_width = int(
        np.ceil(
            W / number_of_columns
        )
    )

    mappings = np.zeros(
        (
            number_of_rows,
            number_of_columns,
            256
        ),
        dtype=float
    )

    # Calculate histogram for each tile

    for row in range(
        number_of_rows
    ):

        row_start = (
            row * tile_height
        )

        row_end = min(
            (row + 1) *
            tile_height,
            H
        )

        for column in range(
            number_of_columns
        ):

            column_start = (
                column * tile_width
            )

            column_end = min(
                (column + 1) *
                tile_width,
                W
            )

            tile = image[
                row_start:row_end,
                column_start:column_end
            ]

            histogram = calculate_histogram(
                tile
            )

            histogram = histogram.astype(
                float
            )

            # Contrast limiting

            number_of_pixels = tile.size

            average_bin_count = (
                number_of_pixels /
                256
            )

            maximum_bin_count = (
                clip_limit *
                average_bin_count
            )

            maximum_bin_count = max(
                maximum_bin_count,
                1
            )

            excess = 0

            for intensity in range(256):

                if (
                    histogram[intensity]
                    >
                    maximum_bin_count
                ):

                    excess += (
                        histogram[intensity]
                        -
                        maximum_bin_count
                    )

                    histogram[intensity] = (
                        maximum_bin_count
                    )

            # Redistribute excess

            redistribution = (
                excess / 256
            )

            for intensity in range(256):

                histogram[intensity] += (
                    redistribution
                )

            # Local CDF

            cdf = np.cumsum(
                histogram
            )

            nonzero_bins = np.where(
                histogram > 0
            )[0]

            if len(nonzero_bins) == 0:

                mappings[
                    row,
                    column
                ] = np.arange(256)

            else:

                cdf_min = cdf[
                    nonzero_bins[0]
                ]

                denominator = (
                    cdf[-1] -
                    cdf_min
                )

                if denominator <= 0:

                    mappings[
                        row,
                        column
                    ] = np.arange(256)

                else:

                    mapping = (
                        (cdf - cdf_min)
                        /
                        denominator
                        * 255
                    )

                    mapping = np.clip(
                        mapping,
                        0,
                        255
                    )

                    mappings[
                        row,
                        column
                    ] = mapping

    # Bilinear interpolation

    output = np.zeros(
        (H, W),
        dtype=float
    )

    tile_centers_y = (
        (np.arange(number_of_rows) + 0.5)
        * H
        / number_of_rows
    )

    tile_centers_x = (
        (np.arange(number_of_columns) + 0.5)
        * W
        / number_of_columns
    )

    for y in range(H):

        if number_of_rows > 1:

            fy = (
                (y - tile_centers_y[0])
                /
                (
                    tile_centers_y[1]
                    -
                    tile_centers_y[0]
                )
            )

            iy = int(
                np.floor(fy)
            )

            wy = fy - iy

            iy = np.clip(
                iy,
                0,
                number_of_rows - 2
            )

            wy = np.clip(
                wy,
                0,
                1
            )

        else:

            iy = 0
            wy = 0

        for x in range(W):

            if number_of_columns > 1:

                fx = (
                    (x - tile_centers_x[0])
                    /
                    (
                        tile_centers_x[1]
                        -
                        tile_centers_x[0]
                    )
                )

                ix = int(
                    np.floor(fx)
                )

                wx = fx - ix

                ix = np.clip(
                    ix,
                    0,
                    number_of_columns - 2
                )

                wx = np.clip(
                    wx,
                    0,
                    1
                )

            else:

                ix = 0
                wx = 0

            pixel_value = int(
                image[y, x]
            )

            if (
                number_of_rows == 1
                and
                number_of_columns == 1
            ):

                output[y, x] = mappings[
                    0,
                    0,
                    pixel_value
                ]

            elif number_of_rows == 1:

                output[y, x] = (
                    (1 - wx)
                    *
                    mappings[
                        0,
                        ix,
                        pixel_value
                    ]
                    +
                    wx
                    *
                    mappings[
                        0,
                        ix + 1,
                        pixel_value
                    ]
                )

            elif number_of_columns == 1:

                output[y, x] = (
                    (1 - wy)
                    *
                    mappings[
                        iy,
                        0,
                        pixel_value
                    ]
                    +
                    wy
                    *
                    mappings[
                        iy + 1,
                        0,
                        pixel_value
                    ]
                )

            else:

                top_left = mappings[
                    iy,
                    ix,
                    pixel_value
                ]

                top_right = mappings[
                    iy,
                    ix + 1,
                    pixel_value
                ]

                bottom_left = mappings[
                    iy + 1,
                    ix,
                    pixel_value
                ]

                bottom_right = mappings[
                    iy + 1,
                    ix + 1,
                    pixel_value
                ]

                output[y, x] = (

                    (1 - wy)
                    *
                    (1 - wx)
                    *
                    top_left

                    +

                    (1 - wy)
                    *
                    wx
                    *
                    top_right

                    +

                    wy
                    *
                    (1 - wx)
                    *
                    bottom_left

                    +

                    wy
                    *
                    wx
                    *
                    bottom_right
                )

    output = np.clip(
        output,
        0,
        255
    )

    return output.astype(
        np.uint8
    )


# ============================================================
# ENTROPY
# ============================================================

def calculate_entropy(image):

    histogram = calculate_histogram(
        image
    )

    probability = (
        histogram /
        histogram.sum()
    )

    probability = probability[
        probability > 0
    ]

    entropy = -np.sum(
        probability *
        np.log2(probability)
    )

    return entropy


# ============================================================
# LOCAL CONTRAST
# ============================================================

def calculate_local_contrast(
        image,
        window_size=15):

    image = image.astype(float)

    H, W = image.shape

    radius = window_size // 2

    padded = np.pad(
        image,
        radius,
        mode="reflect"
    )

    local_std_values = []

    step = 5

    for y in range(
        radius,
        H + radius,
        step
    ):

        for x in range(
            radius,
            W + radius,
            step
        ):

            window = padded[
                y - radius:y + radius + 1,
                x - radius:x + radius + 1
            ]

            std = np.std(window)

            local_std_values.append(
                std
            )

    return np.mean(
        local_std_values
    )


# ============================================================
# CREATE SYNTHETIC FILM FRAMES
# ============================================================

def create_film_frames():

    np.random.seed(10)

    H = 360
    W = 520

    Y, X = np.mgrid[
        0:H,
        0:W
    ]

    scene = (
        0.25

        +

        0.18 *
        np.exp(
            -(
                (X - 260) ** 2 /
                (2 * 170 ** 2)

                +

                (Y - 180) ** 2 /
                (2 * 120 ** 2)
            )
        )

        +

        0.20 *
        np.exp(
            -(
                (X - 130) ** 2 /
                (2 * 60 ** 2)

                +

                (Y - 230) ** 2 /
                (2 * 90 ** 2)
            )
        )

        +

        0.14 *
        np.exp(
            -(
                (X - 390) ** 2 /
                (2 * 70 ** 2)

                +

                (Y - 220) ** 2 /
                (2 * 100 ** 2)
            )
        )
    )

    texture = (
        0.035
        *
        np.sin(X / 7)
        *
        np.sin(Y / 11)
    )

    noise = (
        0.012 *
        np.random.normal(
            size=(H, W)
        )
    )

    scene = scene + texture + noise

    scene = np.clip(
        scene,
        0,
        1
    )

    reference = (
        0.90 *
        scene
        +
        0.05
    )

    dark = (
        0.68 *
        scene
        +
        0.02
    )

    bright = (
        1.22 *
        scene
        +
        0.08
    )

    reference = np.clip(
        reference,
        0,
        1
    )

    dark = np.clip(
        dark,
        0,
        1
    )

    bright = np.clip(
        bright,
        0,
        1
    )

    reference = (
        reference * 255
    ).astype(np.uint8)

    dark = (
        dark * 255
    ).astype(np.uint8)

    bright = (
        bright * 255
    ).astype(np.uint8)

    return reference, dark, bright


# ============================================================
# CREATE SYNTHETIC X-RAY
# ============================================================

def create_xray():

    np.random.seed(20)

    H = 360
    W = 520

    Y, X = np.mgrid[
        0:H,
        0:W
    ]

    center_x = W / 2
    center_y = H * 0.53

    body = np.exp(
        -(
            ((X - center_x)
             /
             (W * 0.43)) ** 2

            +

            ((Y - center_y)
             /
             (H * 0.50)) ** 2
        )
    )

    left_lung = np.exp(
        -(
            ((X - (center_x - 70))
             /
             (W * 0.20)) ** 2

            +

            ((Y - center_y)
             /
             (H * 0.35)) ** 2
        )
    )

    right_lung = np.exp(
        -(
            ((X - (center_x + 70))
             /
             (W * 0.20)) ** 2

            +

            ((Y - center_y)
             /
             (H * 0.35)) ** 2
        )
    )

    image = (
        0.035
        +
        0.55 *
        body
    )

    image -= (
        0.27 *
        (left_lung + right_lung)
    )

    image += (
        0.22 *
        np.exp(
            -(
                (X - center_x)
                /
                (W * 0.035)
            ) ** 2
        )
    )

    for k in range(-3, 4):

        curve = (
            center_y
            - 75
            + k * 38
            + 0.0009 *
            (X - center_x) ** 2
        )

        image += (
            0.045 *
            np.exp(
                -(
                    (Y - curve) / 3
                ) ** 2
            )
        )

    texture = (
        0.022 *
        np.sin(
            X / 3.3 + Y / 15
        )

        +

        0.018 *
        np.sin(
            X / 5.2 - Y / 8
        )
    )

    image += (
        texture *
        (left_lung + right_lung)
    )

    image += (
        0.012 *
        np.random.normal(
            size=(H, W)
        )
    )

    image = np.clip(
        image,
        0,
        1
    )

    image = (
        image * 255
    ).astype(np.uint8)

    return image


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

output_directory = Path(
    "histogram_clahe_results"
)

output_directory.mkdir(
    exist_ok=True
)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("HISTOGRAM MATCHING + CLAHE")

    # ========================================================
    # PART 1 - HISTOGRAM MATCHING
    # ========================================================

    reference, dark, bright = (
        create_film_frames()
    )

    dark_matched, dark_lut = (
        histogram_matching(
            dark,
            reference
        )
    )

    bright_matched, bright_lut = (
        histogram_matching(
            bright,
            reference
        )
    )

    images = [
        reference,
        dark,
        bright,
        dark_matched,
        bright_matched
    ]

    titles = [
        "Reference",
        "Dark Source",
        "Bright Source",
        "Dark Matched",
        "Bright Matched"
    ]

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(18, 7)
    )

    for i in range(5):

        axes[0, i].imshow(
            images[i],
            cmap="gray",
            vmin=0,
            vmax=255
        )

        axes[0, i].set_title(
            titles[i]
        )

        axes[0, i].axis("off")

        axes[1, i].hist(
            images[i].ravel(),
            bins=256,
            range=(0, 255)
        )

        axes[1, i].set_xlim(
            0,
            255
        )

        axes[1, i].set_xlabel(
            "Intensity"
        )

    fig.suptitle(
        "Histogram Matching"
    )

    plt.tight_layout()

    plt.savefig(
        output_directory /
        "01_histogram_matching.png",
        dpi=150
    )

    plt.show()

    # ========================================================
    # CDF COMPARISON
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    for image, label in [

        (reference, "Reference"),
        (dark, "Dark Source"),
        (bright, "Bright Source"),
        (dark_matched, "Dark Matched"),
        (bright_matched, "Bright Matched")

    ]:

        histogram = calculate_histogram(
            image
        )

        cdf = calculate_cdf(
            histogram
        )

        ax.plot(
            cdf,
            label=label
        )

    ax.set_xlabel(
        "Intensity"
    )

    ax.set_ylabel(
        "CDF"
    )

    ax.set_title(
        "CDF Comparison"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        output_directory /
        "02_histogram_matching_cdf.png",
        dpi=150
    )

    plt.show()

    # ========================================================
    # ANALYTICAL TARGET
    # ========================================================

    target = (
        create_moody_target_histogram()
    )

    analytical_output, analytical_lut = (
        histogram_match_analytical_target(
            reference,
            target
        )
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4)
    )

    axes[0].imshow(
        reference,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0].set_title(
        "Original Reference"
    )

    axes[0].axis("off")

    axes[1].imshow(
        analytical_output,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1].set_title(
        "Moody Target"
    )

    axes[1].axis("off")

    axes[2].plot(
        target,
        label="Target"
    )

    output_histogram = calculate_histogram(
        analytical_output
    )

    output_pdf = (
        output_histogram /
        output_histogram.sum()
    )

    axes[2].plot(
        output_pdf,
        label="Output"
    )

    axes[2].set_title(
        "Target vs Output"
    )

    axes[2].legend()

    plt.tight_layout()

    plt.savefig(
        output_directory /
        "03_analytical_target.png",
        dpi=150
    )

    plt.show()

    # ========================================================
    # FAILURE CASE
    # ========================================================

    failure_source, failure_reference = (
        create_failure_case_images()
    )

    failure_output, _ = (
        histogram_matching(
            failure_source,
            failure_reference
        )
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4)
    )

    axes[0].imshow(
        failure_source,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[0].set_title(
        "Source"
    )

    axes[0].axis("off")

    axes[1].imshow(
        failure_reference,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[1].set_title(
        "Reference"
    )

    axes[1].axis("off")

    axes[2].imshow(
        failure_output,
        cmap="gray",
        vmin=0,
        vmax=255
    )

    axes[2].set_title(
        "Histogram Matched"
    )

    axes[2].axis("off")

    plt.tight_layout()

    plt.savefig(
        output_directory /
        "04_histogram_matching_failure.png",
        dpi=150
    )

    plt.show()

    # ========================================================
    # PART 2 - CLAHE
    # ========================================================

    xray = create_xray()

    global_equalized = (
        histogram_equalization(
            xray
        )
    )

    clahe_8x8 = (
        clahe_first_principles(
            xray,
            tile_grid=(8, 8),
            clip_limit=2.0
        )
    )

    clahe_12x12 = (
        clahe_first_principles(
            xray,
            tile_grid=(12, 12),
            clip_limit=1.2
        )
    )

    # ========================================================
    # GLOBAL HE VS CLAHE
    # ========================================================

    images = [
        xray,
        global_equalized,
        clahe_8x8,
        clahe_12x12
    ]

    titles = [
        "Original X-ray",
        "Global HE",
        "CLAHE 8x8",
        "CLAHE 12x12"
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5)
    )

    for i in range(4):

        axes[i].imshow(
            images[i],
            cmap="gray",
            vmin=0,
            vmax=255
        )

        axes[i].set_title(
            titles[i]
        )

        axes[i].axis("off")

    plt.tight_layout()

    plt.savefig(
        output_directory /
        "05_global_vs_clahe.png",
        dpi=150
    )

    plt.show()

    # ========================================================
    # QUANTITATIVE RESULTS
    # ========================================================

    results = [
        ("Original", xray),
        ("Global HE", global_equalized),
        ("CLAHE 8x8", clahe_8x8),
        ("CLAHE 12x12", clahe_12x12)
    ]

    print("\nQuantitative Results")

    for name, image in results:

        entropy = calculate_entropy(
            image
        )

        contrast = calculate_local_contrast(
            image
        )

        print(
            name,
            "Entropy =",
            round(entropy, 3),
            "Local Contrast =",
            round(contrast, 3)
        )

    # ========================================================
    # CLIP LIMIT SWEEP
    # ========================================================

    clip_values = [
        0.5,
        1.0,
        2.0,
        4.0,
        8.0
    ]

    clip_entropies = []
    clip_contrasts = []

    for clip in clip_values:

        result = clahe_first_principles(
            xray,
            tile_grid=(8, 8),
            clip_limit=clip
        )

        ent = calculate_entropy(
            result
        )

        contrast = calculate_local_contrast(
            result
        )

        clip_entropies.append(
            ent
        )

        clip_contrasts.append(
            contrast
        )

        print(
            "Clip =",
            clip,
            "Entropy =",
            round(ent, 3),
            "Contrast =",
            round(contrast, 3)
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].plot(
        clip_values,
        clip_entropies,
        "o-"
    )

    axes[0].set_xlabel(
        "Clip Limit"
    )

    axes[0].set_ylabel(
        "Entropy"
    )

    axes[0].set_title(
        "Clip Limit vs Entropy"
    )

    axes[1].plot(
        clip_values,
        clip_contrasts,
        "o-"
    )

    axes[1].set_xlabel(
        "Clip Limit"
    )

    axes[1].set_ylabel(
        "Local Contrast"
    )

    axes[1].set_title(
        "Clip Limit vs Local Contrast"
    )

    plt.tight_layout()

    plt.savefig(
        output_directory /
        "06_clip_limit_sweep.png",
        dpi=150
    )

    plt.show()

    # ========================================================
    # TILE SIZE SWEEP
    # ========================================================

    tile_sizes = [
        (4, 4),
        (8, 8),
        (12, 12),
        (20, 20)
    ]

    tile_entropies = []
    tile_contrasts = []

    for tile in tile_sizes:

        result = clahe_first_principles(
            xray,
            tile_grid=tile,
            clip_limit=2.0
        )

        ent = calculate_entropy(
            result
        )

        contrast = calculate_local_contrast(
            result
        )

        tile_entropies.append(
            ent
        )

        tile_contrasts.append(
            contrast
        )

        print(
            "Tile =",
            tile,
            "Entropy =",
            round(ent, 3),
            "Contrast =",
            round(contrast, 3)
        )

    tile_labels = [
        "4x4",
        "8x8",
        "12x12",
        "20x20"
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    axes[0].plot(
        tile_labels,
        tile_entropies,
        "o-"
    )

    axes[0].set_xlabel(
        "Tile Grid"
    )

    axes[0].set_ylabel(
        "Entropy"
    )

    axes[0].set_title(
        "Tile Size vs Entropy"
    )

    axes[1].plot(
        tile_labels,
        tile_contrasts,
        "o-"
    )

    axes[1].set_xlabel(
        "Tile Grid"
    )

    axes[1].set_ylabel(
        "Local Contrast"
    )

    axes[1].set_title(
        "Tile Size vs Local Contrast"
    )

    plt.tight_layout()

    plt.savefig(
        output_directory /
        "07_tile_size_sweep.png",
        dpi=150
    )

    plt.show()

    # ========================================================
    # EXTREME PARAMETERS
    # ========================================================

    settings = [
        ((4, 4), 0.5),
        ((4, 4), 8.0),
        ((20, 20), 0.5),
        ((20, 20), 8.0),
        ((8, 8), 2.0)
    ]

    fig, axes = plt.subplots(
        1,
        len(settings),
        figsize=(20, 5)
    )

    for i, (tile, clip) in enumerate(
        settings
    ):

        result = clahe_first_principles(
            xray,
            tile_grid=tile,
            clip_limit=clip
        )

        axes[i].imshow(
            result,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        axes[i].set_title(
            f"{tile[0]}x{tile[1]}\n"
            f"Clip={clip}"
        )

        axes[i].axis("off")

    plt.tight_layout()

    plt.savefig(
        output_directory /
        "08_clahe_extremes.png",
        dpi=150
    )

    plt.show()

    # ========================================================
    # SAVE NUMERICAL DATA
    # ========================================================

    np.savetxt(
        output_directory /
        "clip_limit_results.csv",
        np.column_stack(
            (
                clip_values,
                clip_entropies,
                clip_contrasts
            )
        ),
        delimiter=",",
        header="clip_limit,entropy,local_contrast",
        comments=""
    )

    np.savetxt(
        output_directory /
        "tile_size_results.csv",
        np.column_stack(
            (
                np.arange(
                    len(tile_sizes)
                ),
                tile_entropies,
                tile_contrasts
            )
        ),
        delimiter=",",
        header="tile_index,entropy,local_contrast",
        comments=""
    )

    print("\nAll experiments completed.")

    print(
        "Results saved in:",
        output_directory.resolve()
    )

    print("\nGenerated files:")

    for file in sorted(
        output_directory.iterdir()
    ):

        print(
            file.name
        )
