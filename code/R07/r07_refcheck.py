"""Verify candidate references via Crossref before they enter references.bib."""
import json
import sys
import urllib.parse
import urllib.request

QUERIES = [
    ("DerKiureghian2009", "Aleatory or epistemic? Does it matter? Der Kiureghian Ditlevsen"),
    ("KennedyOHagan2001", "Bayesian calibration of computer models Kennedy O'Hagan"),
    ("Wolpert1992", "Stacked generalization Wolpert"),
    ("Breiman1996", "Stacked regressions Breiman"),
    ("Yao2018", "Using stacking to average Bayesian predictive distributions Yao"),
    ("Koh2021", "WILDS: A benchmark of in-the-wild distribution shifts Koh"),
]

for key, q in QUERIES:
    url = ("https://api.crossref.org/works?rows=2&select=DOI,title,author,issued,container-title,type"
           "&query.bibliographic=" + urllib.parse.quote(q))
    req = urllib.request.Request(url, headers={"User-Agent": "IMUT-refcheck/1.0 (mailto:none@example.org)"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            data = json.load(r)
        items = data["message"]["items"]
        print(f"=== {key} ===")
        for it in items[:2]:
            title = (it.get("title") or ["?"])[0]
            authors = "; ".join(
                f"{a.get('family','')}, {a.get('given','')}" for a in it.get("author", [])[:4]
            )
            year = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
            cont = (it.get("container-title") or [""])[0]
            print(f"  DOI: {it.get('DOI')}")
            print(f"  {title}")
            print(f"  {authors} ({year}) {cont} [{it.get('type')}]")
    except Exception as e:  # noqa: BLE001
        print(f"=== {key} === FAILED: {e}", file=sys.stderr)
