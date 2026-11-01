"""R08 reference audit: resolve every DOI in references.bib on Crossref and compare titles."""
import difflib
import json
import re
import time
import urllib.request
from pathlib import Path

BIB = Path(__file__).resolve().parents[3] / "paper" / "manuscript" / "references.bib"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "doi_check.csv"


def entries(text):
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n?\}\s*(?=@|\Z)", text, re.S):
        body = m.group(3)
        t = re.search(r"title\s*=\s*\{(.+?)\}\s*,", body, re.S)
        d = re.search(r"doi\s*=\s*\{([^}]+)\}", body)
        y = re.search(r"year\s*=\s*\{(\d{4})\}", body)
        yield m.group(2).strip(), (t.group(1) if t else ""), (d.group(1).strip() if d else ""), (y.group(1) if y else "")


def norm(s):
    s = re.sub(r"[{}\\$]", "", s)
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


rows = ["key,doi,status,similarity,bib_year,cr_year,crossref_title"]
for key, title, doi, year in entries(BIB.read_text(encoding="utf-8")):
    if not doi:
        rows.append(f"{key},,no-doi,,{year},,")
        continue
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    status, sim, cy, ct = "", "", "", ""
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "kag-refcheck (mailto:renlijian@imut.edu.cn)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                msg = json.load(r)["message"]
            ct = (msg.get("title") or [""])[0]
            dp = (msg.get("issued") or {}).get("date-parts") or [[None]]
            cy = str(dp[0][0])
            sim = f"{difflib.SequenceMatcher(None, ' '.join(norm(title)), ' '.join(norm(ct))).ratio():.2f}"
            status = "ok"
            break
        except urllib.error.HTTPError as e:
            status = f"http{e.code}"
            if e.code == 404:
                break
            time.sleep(3 * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            status = f"err:{type(e).__name__}"
            time.sleep(3)
    rows.append(f'{key},{doi},{status},{sim},{year},{cy},"{ct.replace(chr(34), "")}"')
    print(rows[-1], flush=True)
    time.sleep(1.2)
OUT.write_text("\n".join(rows), encoding="utf-8")
