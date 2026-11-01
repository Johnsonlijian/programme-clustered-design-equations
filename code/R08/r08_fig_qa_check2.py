import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r08_figures as F

FIGS = {}
F.save = lambda fig, name: FIGS.setdefault(name, fig)
F.budget()
fig = FIGS["fig4_pred_vs_obs_r08"]
fig.canvas.draw()
r = fig.canvas.get_renderer()
ax = fig.axes[0]
inv = ax.transData.inverted()
for t in ax.texts:
    s = t.get_text()
    if not s.strip():
        continue
    bb = t.get_window_extent(r)
    p0 = inv.transform((bb.x0, bb.y0))
    p1 = inv.transform((bb.x1, bb.y1))
    print(f"{s[:24]:26s} bbox data x[{p0[0]:6.2f},{p1[0]:6.2f}] y[{p0[1]:6.2f},{p1[1]:6.2f}]")
plt.close("all")
