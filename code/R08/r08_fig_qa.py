"""QA pass for the R08 figures: rebuild them in-memory and report artist
bounding-box collisions (text vs text, text vs line/marker paths, legend
inside canvas) so clipping/overlap can be caught without eyeballing the PNG."""
import itertools
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r08_figures as F  # noqa: E402

FIGS = {}
F.save = lambda fig, name: FIGS.setdefault(name, fig)
F.budget()
F.repair()


def check(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    print(f"=== {name} ({W}x{H})")
    for ai, ax in enumerate(fig.axes):
        items = [(t.get_text()[:28], t.get_window_extent(r))
                 for t in ax.texts if t.get_text().strip()]
        for s, bb in items:
            if bb.x0 < -1 or bb.x1 > W + 1 or bb.y0 < -1 or bb.y1 > H + 1:
                print(f"  ax{ai} CLIP text '{s}' -> {bb}")
        for (s1, b1), (s2, b2) in itertools.combinations(items, 2):
            if b1.overlaps(b2):
                print(f"  ax{ai} OVERLAP '{s1}' x '{s2}'")
        # text vs plotted lines/markers (display-space path containment)
        for ln in ax.get_lines() + ax.patches:
            try:
                tr = ln.get_transform() + ax.transData.inverted() \
                    if ln.get_transform() != ax.transData else ln.get_transform()
                path = ln.get_path() if hasattr(ln, "get_path") else None
                if path is None:
                    continue
                p = path.transformed(ln.get_transform())
            except Exception:
                continue
            for s, bb in items:
                if p.intersects_bbox(bb, filled=False):
                    print(f"  ax{ai} text '{s}' crossed by line/patch")
        leg = ax.get_legend()
        if leg is not None:
            bb = leg.get_window_extent(r)
            if bb.x0 < -1 or bb.x1 > W + 1 or bb.y0 < -1 or bb.y1 > H + 1:
                print(f"  ax{ai} legend outside canvas: {bb}")
            else:
                print(f"  ax{ai} legend bbox ok ({bb.width:.0f}x{bb.height:.0f})")


for n, f in FIGS.items():
    check(f, n)
plt.close("all")
print("done")
