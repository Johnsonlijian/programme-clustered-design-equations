import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r08_figures as F

FIGS = {}
F.save = lambda fig, name: FIGS.setdefault(name, fig)
F.repair()
fig = FIGS["fig_repair"]
fig.canvas.draw()
r = fig.canvas.get_renderer()
ax = fig.axes[2]
print("ylim:", ax.get_ylim(), "xlim:", ax.get_xlim())
for t in ax.texts:
    if t.get_text().strip():
        bb = t.get_window_extent(r)
        print(repr(t.get_text()), "bbox", bb)
for j, ln in enumerate(ax.get_lines()):
    p = ln.get_path().transformed(ln.get_transform())
    ext = p.get_extents()
    print(j, ln.get_color(), ln.get_label(), "ext", ext)
# legend position
leg = fig.legends[0] if fig.legends else None
if leg:
    print("legend bbox:", leg.get_window_extent(r), "figsize:", fig.canvas.get_width_height())
plt.close("all")
