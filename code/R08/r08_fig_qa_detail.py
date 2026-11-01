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
F.repair()
for name, fig in FIGS.items():
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    print("=====", name)
    for ai, ax in enumerate(fig.axes):
        items = [(t.get_text()[:26], t.get_window_extent(r),
                  ax.transData.inverted().transform(
                      [(t.get_window_extent(r).x0 + t.get_window_extent(r).x1) / 2,
                       (t.get_window_extent(r).y0 + t.get_window_extent(r).y1) / 2]))
                 for t in ax.texts if t.get_text().strip()]
        arts = list(ax.get_lines()) + list(ax.patches)
        for s, bb, ctr in items:
            for j, a in enumerate(arts):
                try:
                    p = a.get_path().transformed(a.get_transform())
                except Exception:
                    continue
                if p.intersects_bbox(bb, filled=False):
                    ext = p.get_extents()
                    lo = ax.transData.inverted().transform(
                        [(ext.x0, ext.y0), (ext.x1, ext.y1)])
                    (x0, y0), (x1, y1) = lo
                    print(f'  ax{ai} "{s}" ctr=({ctr[0]:.1f},{ctr[1]:.1f}) '
                          f'<- artist{j} {type(a).__name__} '
                          f'data-extent x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}]')
plt.close("all")
