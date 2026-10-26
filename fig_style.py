"""Shared style for the journal versions of the figures (IJC: A4, two columns).

Every figure is drawn at its printed size, so a font size here is the size on paper.
IJC asks for no text below 10 pt in figures; labels use 9 to 9.5 pt and nothing goes
below MIN_PT, which save() checks before writing a file.
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.text import Text

MM = 1 / 25.4            # inches per millimetre
SINGLE = 88 * MM         # one IJC column (88.3 mm)
DOUBLE = 180 * MM        # full IJC text width (183.8 mm) less a small margin
MIN_PT = 8.5             # smallest font allowed on paper

# Same hues as v4; teal is darker so that teal, grey and amber also separate in greyscale.
INK = "#1f2933"; TEAL = "#0b7a70"; AMBER = "#e0a100"; RED = "#c92a2a"; GREY = "#9aa1a9"
PURPLE = "#5b3cc4"; BLUE = "#1c6fbd"; GRID = "#e9ecef"; NOTE = "#59616b"


def apply():
    """Journal rcParams: Times New Roman like the body text, thin spines, light grid."""
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8.5, "legend.frameon": False,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 1.4, "lines.markersize": 4.5, "hatch.linewidth": 0.7,
        "savefig.dpi": 600, "svg.fonttype": "none",
    })


def figure(width, height_mm, **kw):
    """New figure at its printed size: width in inches (SINGLE or DOUBLE), height in mm."""
    return plt.subplots(figsize=(width, height_mm * MM), layout="constrained", **kw)


def smallest_font(fig):
    return min(t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip())


def save(fig, out, name):
    """Write PNG (600 dpi, for Word) and SVG at exactly the printed size; refuse text below MIN_PT."""
    small = smallest_font(fig)
    assert small >= MIN_PT, f"{name}: text at {small} pt is below the {MIN_PT} pt floor"
    for ext in ("png", "svg"):
        fig.savefig(f"{out}/{name}.{ext}", facecolor="white")
    w, h = fig.get_size_inches() / MM
    print(f"saved {name}: {w:.0f} x {h:.0f} mm, smallest text {small} pt")
    plt.close(fig)
