"""Render an ablation overview table (baseline + variants) as a PNG figure."""

import os
import matplotlib.pyplot as plt


HEADER = ["Variant", "Backbone", "Upsampler", "Encoder", "Crop",
          "Clamp GT", "Pixel loss", "Composite (pix/grad/\nnorm/struct/smooth)"]

DASH = "—"

ROWS = [
    ["Freeze baseline",
     "DINOv2-S [5,7,9,11]", "Patch2Feature", "frozen", "560$\\times$560 sq.",
     "off", "L2", "0.3 / 1.0 / 1.0 / 0 / 0"],

    ["clamp_gt", DASH, DASH, DASH, DASH, "on", DASH, DASH],

    ["composite_mse_only", DASH, DASH, DASH, DASH, DASH, "MSE",
     "1.0 / 0 / 0 / 0 / 0"],

    ["composite_l1_only", DASH, DASH, DASH, DASH, DASH, "L1",
     "1.0 / 0 / 0 / 0 / 0"],

    ["crop_to_road", DASH, DASH, DASH, "994$\\times$504 road", DASH, DASH, DASH],

    ["da3", "DepthAnything3", DASH, DASH, DASH, DASH, DASH, DASH],

    ["efficientnet", "EfficientNet", "n/a", DASH, DASH, DASH, DASH, DASH],

    ["dinov2_intermediate_11s",
     "DINOv2-S [11,11,11,11]", DASH, DASH, DASH, DASH, DASH, DASH],

    ["dinov2_upsampler", DASH, "DinoUpsampler", DASH, DASH, DASH, DASH, DASH],

    ["dinov2_trainable", DASH, DASH, "fine-tuned\n(lr 1e-5)", DASH, DASH, DASH, DASH],
]


def render_png():
    fig, ax = plt.subplots(figsize=(17, 5.2))
    ax.axis("off")

    table = ax.table(
        cellText=ROWS,
        colLabels=HEADER,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    n_cols = len(HEADER)
    # Header row styling
    for c in range(n_cols):
        cell = table[(0, c)]
        cell.set_facecolor("#324a5e")
        cell.set_text_props(color="white", weight="bold")

    # Baseline row styling (row 1 in table coords)
    for c in range(n_cols):
        cell = table[(1, c)]
        cell.set_facecolor("#fff2cc")
        cell.set_text_props(weight="bold")

    # Highlight non-dash cells in ablation rows
    for r in range(2, len(ROWS) + 1):
        for c in range(1, n_cols):
            text = ROWS[r - 1][c]
            if text != DASH:
                table[(r, c)].set_facecolor("#d9ead3")

    out = "figures/ablation_table.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    render_png()
