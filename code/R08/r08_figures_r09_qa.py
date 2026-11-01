"""QA: for every r09 figure, assert no two text window-extents intersect
and no text's centre sits inside another axes' patch. Prints a report."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r08_figures_r09 as F

FIGS = {}
F.save = lambda fig, name: FIGS.setdefault(name, fig)
for fn in (F.concept, F.evidence, F.grading, F.scarcity, F.predict, F.paka,
           F.leakage):
    fn()

for name, fig in FIGS.items():
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = []  # (label, bbox, owner_axes_or_None)
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text().strip() and t.get_visible():
                texts.append((t.get_text().replace("\n", " ")[:28],
                              t.get_window_extent(rend), ax))
        if ax.axison:
            for t in ax.get_xticklabels() + ax.get_yticklabels():
                if t.get_text().strip() and t.get_visible():
                    texts.append(("tick:" + t.get_text(),
                                  t.get_window_extent(rend), ax))
        for lab in (ax.xaxis.label, ax.yaxis.label, ax.title):
            if lab.get_text().strip():
                texts.append((lab.get_text().replace("\n", " ")[:28],
                              lab.get_window_extent(rend), ax))
    for leg in fig.legends + [ax.get_legend() for ax in fig.axes
                              if ax.get_legend()]:
        if leg is None:
            continue
        for t in leg.get_texts():
            if t.get_text().strip():
                texts.append(("leg:" + t.get_text()[:24],
                              t.get_window_extent(rend), None))
    problems = []
    # legend frames vs every text extent
    frames = []
    for leg in fig.legends + [ax.get_legend() for ax in fig.axes
                              if ax.get_legend()]:
        if leg is None:
            continue
        frames.append((leg.get_window_extent(rend).expanded(1.02, 1.02),
                       "legend-frame"))
    for fb, _ in frames:
        for txt, bb, owner in texts:
            if not txt.startswith("leg:") and fb.overlaps(bb):
                problems.append(f"  FRAME-TEXT: legend frame x '{txt}'")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if texts[i][1].overlaps(texts[j][1]):
                problems.append(f"  TEXT-TEXT: '{texts[i][0]}' x "
                                f"'{texts[j][0]}'")
    apatches = [(ax, ax.get_window_extent(rend)) for ax in fig.axes]
    for txt, bb, owner in texts:
        if owner is None:
            continue
        cx, cy = bb.x0 + bb.width / 2, bb.y0 + bb.height / 2
        for ax, ap in apatches:
            if ax is not owner and ap.contains(cx, cy):
                problems.append(f"  CROSS-AXES: '{txt}' inside other axes")
                break
    print(f"===== {name}: {len(texts)} texts checked")
    print("\n".join(sorted(set(problems))) if problems else "  clean")
plt.close("all")
