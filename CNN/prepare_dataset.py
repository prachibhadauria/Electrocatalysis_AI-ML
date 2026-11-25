"""Prepare a tile dataset for CNN porosity regression from NFFA-Europe Porous_Sponge SEM images.

Usage (from this folder):
    .venv/bin/python prepare_dataset.py

Original images in ./Porous_Sponge/ are only read, never modified. Outputs written here:
    tiles/                  224x224 grayscale PNG tiles (every image, including excluded ones)
    images.csv              one row per source image (threshold, porosity, QC metrics, flags, split)
                            flagged images = bar not found, porosity <0.05 or >0.95, charging,
                            low contrast, non-bimodal histogram, noise-only frame, or exact duplicate
    tiles.csv               one row per tile (porosity label, split)
    train.csv, test.csv     the clean dataset: tiles of unflagged images, split by source image
    review/                 contact sheets: cropped image | Otsu mask (dark = pore)
    porosity_histogram.png  tile porosity distribution
    summary.txt             counts and settings
    porosity_dataset.zip    train/test CSVs + clean tiles, for upload to Google Drive (Colab)
"""
import csv
import os
import random
import shutil
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(HERE, "Porous_Sponge")
TILE_DIR = os.path.join(HERE, "tiles")
REVIEW_DIR = os.path.join(HERE, "review")

TILE = 256
OUT_SIZE = 224
SEED = 42
TEST_FRACTION = 0.2

# QC thresholds
POROSITY_MIN, POROSITY_MAX = 0.05, 0.95
BRIGHT_LEVEL, BRIGHT_MAX = 250, 0.05
CONTRAST_MIN = 15.0
# Otsu separability eta = between-class variance / total variance. Low eta means the
# histogram is not bimodal, so the threshold (and the porosity label) is arbitrary.
SEPARABILITY_MIN = 0.6
# Share of pixel std that survives a sigma=2 Gaussian blur. Pure-noise frames lose most of
# it (0.24-0.52 in this set); real structure keeps most (>= 0.66).
STRUCTURE_MIN = 0.6
STRUCTURE_SIGMA = 2.0
# Duplicate detection on 64x48 thumbnails of the top 600 rows (present in every crop).
DUP_CORR, DUP_MAD = 0.999, 1.0      # identical image: dropped
NEAR_DUP_CORR = 0.95                # same field, slightly shifted: kept, but in one split

# Info bar detection: the bar's top edge is a 1 px black line spanning nearly the
# full width, with the bar's white background just below it. Text colour is irrelevant.
BORDER_DARK_LEVEL = 80
BORDER_DARK_FRAC = 0.9
INSIDE_WHITE_LEVEL = 235
INSIDE_WHITE_FRAC = 0.6

ENTRIES_PER_SHEET = 20
UNFLAGGED_SAMPLE = 100


def find_info_bar_top(gray):
    """Return the row index of the info bar's top border, or None if not found."""
    h = gray.shape[0]
    dark = (gray <= BORDER_DARK_LEVEL).mean(axis=1)
    white = (gray >= INSIDE_WHITE_LEVEL).mean(axis=1)
    for r in range(h // 2, h - 3):
        if dark[r] >= BORDER_DARK_FRAC and white[r + 2] >= INSIDE_WHITE_FRAC:
            return r
    return None


def process_image(fname):
    gray = np.asarray(Image.open(os.path.join(SRC_DIR, fname)).convert("L"))
    bar_top = find_info_bar_top(gray)
    cropped = gray if bar_top is None else gray[:bar_top]

    # Dark class = pixels at or below the Otsu threshold (skimage's split convention).
    thresh = float(threshold_otsu(cropped))
    pores = cropped <= thresh
    stats = {
        "source_image": fname,
        "info_bar_row": "" if bar_top is None else bar_top,
        "otsu_threshold": thresh,
        "image_porosity": float(pores.mean()),
        "bright_fraction": float((cropped >= BRIGHT_LEVEL).mean()),
        "contrast": float(cropped.std()),
    }
    f = cropped.astype(float)
    w0 = pores.mean()
    eta = (w0 * (1 - w0) * (f[pores].mean() - f[~pores].mean()) ** 2 / f.var()
           if 0 < w0 < 1 else 0.0)
    stats["otsu_separability"] = float(eta)
    stats["structure_ratio"] = float(gaussian_filter(f, STRUCTURE_SIGMA).std() / f.std())

    reasons = []
    if bar_top is None:
        reasons.append("info bar not found")
    if stats["image_porosity"] < POROSITY_MIN:
        reasons.append("porosity < %.2f" % POROSITY_MIN)
    if stats["image_porosity"] > POROSITY_MAX:
        reasons.append("porosity > %.2f" % POROSITY_MAX)
    if stats["bright_fraction"] > BRIGHT_MAX:
        reasons.append("bright fraction > %.2f (charging?)" % BRIGHT_MAX)
    if stats["contrast"] < CONTRAST_MIN:
        reasons.append("contrast < %g" % CONTRAST_MIN)
    if stats["otsu_separability"] < SEPARABILITY_MIN:
        reasons.append("Otsu separability < %g" % SEPARABILITY_MIN)
    if stats["structure_ratio"] < STRUCTURE_MIN:
        reasons.append("noise-dominated")
    stats["reasons"] = reasons
    thumb = Image.fromarray(gray[:600]).resize((64, 48), Image.BILINEAR)
    stats["thumb"] = np.asarray(thumb, dtype=float)

    # Non-overlapping tiles; partial tiles at the right/bottom edges are discarded.
    stem = os.path.splitext(fname)[0]
    tiles = []
    for tr in range(cropped.shape[0] // TILE):
        for tc in range(cropped.shape[1] // TILE):
            ys, xs = tr * TILE, tc * TILE
            tile = cropped[ys:ys + TILE, xs:xs + TILE]
            # Label is computed on the full 256x256 region; resizing keeps the same region.
            porosity = float((tile <= thresh).mean())
            tile_name = "%s_r%d_c%d.png" % (stem, tr, tc)
            Image.fromarray(tile).resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS).save(
                os.path.join(TILE_DIR, tile_name))
            tiles.append({"tile_filename": tile_name, "source_image": fname,
                          "tile_row": tr, "tile_col": tc, "porosity": porosity})
    stats["n_tiles"] = len(tiles)
    return stats, tiles, cropped, pores


def find_duplicates(image_rows):
    """Flag exact duplicates (keeping the first by filename) and give near-duplicates a
    shared group id so they never straddle train and test."""
    parent = list(range(len(image_rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(image_rows)):
        a = image_rows[i]["thumb"]
        for j in range(i + 1, len(image_rows)):
            b = image_rows[j]["thumb"]
            corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
            if corr < NEAR_DUP_CORR:
                continue
            parent[root(j)] = root(i)
            if corr >= DUP_CORR and np.abs(a - b).mean() < DUP_MAD:
                image_rows[j]["reasons"].append(
                    "duplicate of %s" % image_rows[i]["source_image"])
    for i, row in enumerate(image_rows):
        row["group"] = image_rows[root(i)]["source_image"]
        row["flag"] = int(bool(row["reasons"]))
        row["flag_reason"] = "; ".join(row["reasons"])


def assign_splits(image_rows):
    """Split clean images into train/test by source image (near-duplicate groups kept
    together), stratified by info-bar layout (a proxy for instrument/session)."""
    rng = random.Random(SEED)
    strata = {}
    for row in image_rows:
        if row["flag"]:
            row["split"] = "excluded"
        else:
            strata.setdefault(row["info_bar_row"], {}).setdefault(row["group"], []).append(row)
    for key in sorted(strata, key=str):
        groups = [strata[key][g] for g in sorted(strata[key])]
        rng.shuffle(groups)
        n_test_target = round(sum(map(len, groups)) * TEST_FRACTION)
        n_test = 0
        for group in groups:
            split = "test" if n_test < n_test_target else "train"
            n_test += len(group) if split == "test" else 0
            for row in group:
                row["split"] = split


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("%.6f" % v if isinstance(v, float) else v)
                        for k, v in row.items()})


def make_review_sheets(entries, prefix):
    """entries: list of (stats, cropped, pores). 20 per sheet, 4 entries per row."""
    cols, rows = 4, 5
    for s in range(0, len(entries), ENTRIES_PER_SHEET):
        chunk = entries[s:s + ENTRIES_PER_SHEET]
        fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 2.6, rows * 2.4))
        for ax in axes.ravel():
            ax.axis("off")
        for i, (stats, cropped, pores) in enumerate(chunk):
            r, c = divmod(i, cols)
            ax_img, ax_mask = axes[r, 2 * c], axes[r, 2 * c + 1]
            ax_img.imshow(cropped, cmap="gray", vmin=0, vmax=255)
            ax_mask.imshow(~pores, cmap="gray")
            title = "%s\nporosity=%.3f" % (stats["source_image"][:15], stats["image_porosity"])
            ax_img.set_title(title, fontsize=7)
            if stats["flag_reason"]:
                ax_mask.set_title(stats["flag_reason"].replace("; ", "\n"), fontsize=6, color="red")
            else:
                ax_mask.set_title("Otsu mask (black = pore)", fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(REVIEW_DIR, "%s_%02d.png" % (prefix, s // ENTRIES_PER_SHEET + 1)),
                    dpi=110)
        plt.close(fig)


def main():
    for d in (TILE_DIR, REVIEW_DIR):
        shutil.rmtree(d, ignore_errors=True)  # clear outputs from previous runs
        os.makedirs(d)

    fnames = sorted(f for f in os.listdir(SRC_DIR)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")))
    image_rows, tile_rows, entries = [], [], []
    for fname in fnames:
        stats, tiles, cropped, pores = process_image(fname)
        image_rows.append(stats)
        tile_rows.extend(tiles)
        entries.append((stats, cropped, pores))

    find_duplicates(image_rows)
    flagged = [e for e in entries if e[0]["flag"]]
    unflagged = [e for e in entries if not e[0]["flag"]]
    assign_splits(image_rows)
    split_of = {r["source_image"]: r["split"] for r in image_rows}
    for t in tile_rows:
        t["split"] = split_of[t["source_image"]]

    write_csv(os.path.join(HERE, "images.csv"), image_rows,
              ["source_image", "otsu_threshold", "image_porosity", "bright_fraction",
               "contrast", "n_tiles", "flag", "flag_reason", "info_bar_row",
               "otsu_separability", "structure_ratio", "group", "split"])
    tile_cols = ["tile_filename", "source_image", "tile_row", "tile_col", "porosity"]
    write_csv(os.path.join(HERE, "tiles.csv"), tile_rows, tile_cols + ["split"])
    for split in ("train", "test"):
        write_csv(os.path.join(HERE, "%s.csv" % split),
                  [t for t in tile_rows if t["split"] == split], tile_cols)

    make_review_sheets(flagged, "flagged")
    rng = random.Random(SEED)
    sample = rng.sample(unflagged, min(UNFLAGGED_SAMPLE, len(unflagged)))
    make_review_sheets(sample, "unflagged_sample")

    porosities = np.array([t["porosity"] for t in tile_rows])
    clean = np.array([t["porosity"] for t in tile_rows if t["split"] != "excluded"])
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 41)
    ax.hist(porosities, bins=bins, color="#bbbbbb", label="all tiles (n=%d)" % len(porosities))
    ax.hist(clean, bins=bins, color="#3a6ea5", label="train+test tiles (n=%d)" % len(clean))
    ax.set_xlabel("tile porosity (fraction of pixels at/below image Otsu threshold)")
    ax.set_ylabel("tiles")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "porosity_histogram.png"), dpi=120)
    plt.close(fig)

    reason_counts = {}
    for r in image_rows:
        for reason in r["reasons"]:
            reason = "duplicate" if reason.startswith("duplicate of") else reason
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    n_split = {s: sum(r["split"] == s for r in image_rows) for s in ("train", "test")}
    t_split = {s: sum(t["split"] == s for t in tile_rows) for s in ("train", "test")}
    lines = [
        "Images processed:       %d" % len(image_rows),
        "Images flagged:         %d (excluded from train/test)" % len(flagged),
        "  by reason (an image can have several):",
    ] + ["    %-36s %d" % (k, v) for k, v in sorted(reason_counts.items())] + [
        "Total tiles:            %d" % len(tile_rows),
        "Train: %d images, %d tiles" % (n_split["train"], t_split["train"]),
        "Test:  %d images, %d tiles" % (n_split["test"], t_split["test"]),
        "Tile porosity (train+test): mean %.3f, std %.3f, min %.3f, max %.3f"
        % (clean.mean(), clean.std(), clean.min(), clean.max()) if len(clean) else "",
        "Split: by source image, %d%% test, stratified by info-bar layout, "
        "near-duplicates kept together, seed %d" % (TEST_FRACTION * 100, SEED),
    ]
    # Single-file bundle of the clean dataset for upload to Google Drive / Colab.
    # PNGs are already compressed, so store them as-is.
    with zipfile.ZipFile(os.path.join(HERE, "porosity_dataset.zip"), "w",
                         zipfile.ZIP_STORED) as zf:
        for name in ("train.csv", "test.csv", "images.csv"):
            zf.write(os.path.join(HERE, name), name)
        for t in tile_rows:
            if t["split"] != "excluded":
                zf.write(os.path.join(TILE_DIR, t["tile_filename"]),
                         "tiles/" + t["tile_filename"])
    lines.append("Colab bundle: porosity_dataset.zip (%.1f MB)"
                 % (os.path.getsize(os.path.join(HERE, "porosity_dataset.zip")) / 1e6))

    text = "\n".join(lines)
    with open(os.path.join(HERE, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
