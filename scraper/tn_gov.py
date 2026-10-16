"""
STEP 1 - Scrape Tamil Nadu government schemes from https://www.tn.gov.in/schemes.php

Flow:
  1. Open the schemes page and find the department links
     (e.g. https://www.tn.gov.in/scheme/department_wise/2 = Agriculture - Farmers Welfare).
     Keeps agriculture-related departments by default (see AGRI_DEPT_KEYWORDS).
  2. Open each department page and collect every scheme link
     (handles "show all entries" tables and Next-page pagination).
  3. Open each scheme page and save:
       - fields from the details table (e.g. Description, Eligibility, How to avail ...)
       - the full page text
       - linked PDFs (downloaded to data/pdfs/)

Output: data/tn_gov_schemes.json   (saved after every scheme; a re-run resumes)
Debug : data/raw/  (page HTML, screenshots, report)

Run:
    python scraper/scrape_tngov.py --headed --limit 5   # quick test, watch it
    python scraper/scrape_tngov.py                      # all agriculture departments
    python scraper/scrape_tngov.py --all-depts          # every department
    python scraper/scrape_tngov.py --list-depts         # just print the departments found
"""

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
START_URL = "https://www.tn.gov.in/schemes.php"

# Department names containing any of these are kept (case-insensitive)
AGRI_DEPT_KEYWORDS = [
    "agri", "farmer", "horticult", "animal husbandry", "dairy", "fisher",
    "co-operation", "cooperation", "sericulture", "உழவர்", "வேளாண்",
]

DEPT_HREF = re.compile(r"department_wise|dept|department", re.I)
SKIP_HREF = re.compile(
    r"(mailto:|tel:|javascript:|facebook|twitter|x\.com|youtube|instagram|whatsapp|"
    r"\.(jpg|jpeg|png|gif|zip)$|/login|/contact|sitemap|feedback|screenreader)", re.I)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PDF_DIR = DATA_DIR / "pdfs"
OUT_FILE = DATA_DIR / "tn_gov_schemes.json"

DELAY = 1.5  # seconds between page loads - be gentle with the server

# --------------------------------------------------------------------------
# JavaScript run inside the page
# --------------------------------------------------------------------------
JS_HELPERS = """
window.__isVisible = el => !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
window.__inChrome = el => !!el.closest(
  'header, nav, footer, .navbar, .header, .footer, .top-bar, .topbar, .breadcrumb, ' +
  '#header, #footer, .menu, #menu, .sidebar-menu, .skip');
true;
"""

JS_LINKS = """
() => [...document.querySelectorAll('a[href]')]
  .filter(a => !__inChrome(a))
  .map(a => ({ text: (a.innerText || a.title || '').trim().replace(/\\s+/g, ' '),
               href: a.href, inTable: !!a.closest('table') }))
"""

JS_MAIN_TEXT = """
() => {
  const hide = [...document.querySelectorAll(
    'header, nav, footer, .navbar, .header, .footer, .top-bar, .topbar, #header, #footer, ' +
    '.breadcrumb, .menu, #menu, script, style, noscript, .skip')];
  const prev = hide.map(e => e.style.display);
  hide.forEach(e => e.style.display = 'none');
  const t = document.body.innerText;
  hide.forEach((e, i) => e.style.display = prev[i]);
  return t;
}
"""

# key/value pairs from 2-column tables, <th>/<td> rows and <dl> lists
JS_FIELDS = """
() => {
  const out = [];
  const txt = el => (el.innerText || '').trim().replace(/[ \\t]+/g, ' ');
  document.querySelectorAll('table tr').forEach(tr => {
    if (__inChrome(tr)) return;
    const cells = [...tr.children].filter(c => /^(TD|TH)$/.test(c.tagName));
    if (cells.length === 2) out.push([txt(cells[0]), txt(cells[1])]);
    else if (cells.length === 3 && /^\\d+\\.?$/.test(txt(cells[0])))   // "1 | Field | Value"
      out.push([txt(cells[1]), txt(cells[2])]);
  });
  document.querySelectorAll('dl').forEach(dl => {
    const dts = dl.querySelectorAll('dt');
    dts.forEach(dt => { const dd = dt.nextElementSibling;
      if (dd && dd.tagName === 'DD') out.push([txt(dt), txt(dd)]); });
  });
  return out.filter(([k, v]) => k && v && k.length < 120);
}
"""

JS_TITLE = """
() => {
  for (const sel of ['h1', 'h2', 'h3', '.page-title', '.title']) {
    const el = [...document.querySelectorAll(sel)].find(e => __isVisible(e) && !__inChrome(e));
    if (el && el.innerText.trim()) return el.innerText.trim();
  }
  return document.title || '';
}
"""

# DataTables etc: choose the biggest "show N entries" option so all rows appear
JS_SHOW_ALL = """
() => {
  let changed = false;
  document.querySelectorAll("select[name$='_length'], .dataTables_length select, select.page-size").forEach(s => {
    const opts = [...s.options];
    const best = opts.find(o => o.value === '-1') ||
                 opts.reduce((a, b) => (+b.value > +a.value ? b : a), opts[0]);
    if (best && s.value !== best.value) {
      s.value = best.value;
      s.dispatchEvent(new Event('change', {bubbles: true}));
      if (window.jQuery) jQuery(s).trigger('change');
      changed = true;
    }
  });
  return changed;
}
"""

JS_NEXT = """
() => {
  const cands = [...document.querySelectorAll(
     "a[rel='next'], .pagination a, .paginate_button, .pager a, a.next, li.next a, button.next")]
    .filter(e => __isVisible(e));
  const n = cands.find(e => /^(next|›|»|>|அடுத்த)/i.test((e.innerText || '').trim()) || e.rel === 'next');
  if (!n) return false;
  const li = n.closest('li');
  if (n.classList.contains('disabled') || (li && li.classList.contains('disabled'))) return false;
  n.click();
  return true;
}
"""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def clean(text):
    out, blank = [], False
    for l in (text or "").splitlines():
        l = re.sub(r"[ \t\u00a0]+", " ", l).strip()
        if l:
            out.append(l); blank = False
        elif not blank:
            out.append(""); blank = True
    return "\n".join(out).strip()


def rec_id(*parts):
    return hashlib.md5("||".join(map(str, parts)).encode("utf-8")).hexdigest()[:12]


def is_pdf(url):
    return ".pdf" in urlparse(url).path.lower()


def wait_settle(page, ms=800):
    try:
        page.wait_for_load_state("networkidle", timeout=15_000)
    except PWTimeout:
        pass
    page.wait_for_timeout(ms)


def run_js(page, script):
    page.evaluate(JS_HELPERS)
    return page.evaluate(script)


def goto(page, url, tries=3):
    for i in range(tries):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=90_000)
            wait_settle(page)
            return True
        except Exception as e:
            print(f"    retry {i + 1}: {str(e)[:100]}")
            time.sleep(3)
    return False


def same_site(url, base):
    return urlparse(url).netloc.split(":")[0].endswith(urlparse(base).netloc.split(":")[0].replace("www.", ""))


def download_pdf(context, url, title):
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    base = re.sub(r"[^\w\u0B80-\u0BFF\-]+", "_", title).strip("_")[:60] or "file"
    path = PDF_DIR / f"{base}_{rec_id(url)}.pdf"
    if path.exists():
        return path
    try:
        r = context.request.get(url, timeout=90_000)
        if r.ok and r.body()[:4] == b"%PDF":
            path.write_bytes(r.body())
            return path
    except Exception as e:
        print(f"    ! PDF failed: {url} ({str(e)[:80]})")
    return None


# --------------------------------------------------------------------------
# Step A: departments
# --------------------------------------------------------------------------
def find_departments(page, base):
    links = run_js(page, JS_LINKS)
    depts = {}
    for l in links:
        if l["text"] and DEPT_HREF.search(l["href"]) and same_site(l["href"], base) \
                and not SKIP_HREF.search(l["href"]):
            depts.setdefault(l["href"].split("#")[0], l["text"])
    return [{"name": n, "url": u} for u, n in depts.items()]


# --------------------------------------------------------------------------
# Step B: scheme links on a department page (all pages)
# --------------------------------------------------------------------------
def scheme_links_on_page(page, base, dept_urls):
    links = run_js(page, JS_LINKS)
    here = page.url.split("#")[0]
    good = []
    for l in links:
        h = l["href"].split("#")[0]
        if not l["text"] or h == here or h in dept_urls or SKIP_HREF.search(h):
            continue
        if not (same_site(h, base) or is_pdf(h)):
            continue
        good.append(l)
    # prefer links that look like scheme pages / sit inside the listing table
    scheme_like = [l for l in good if re.search(r"scheme|data_view|view|detail", l["href"], re.I)
                   and not DEPT_HREF.search(l["href"])]
    in_table = [l for l in good if l["inTable"]]
    return scheme_like or in_table or good


def collect_department(page, dept, base, dept_urls):
    if not goto(page, dept["url"]):
        return []
    if run_js(page, JS_SHOW_ALL):
        wait_settle(page, 1500)
    found, seen_first = {}, set()
    for page_no in range(1, 100):
        links = scheme_links_on_page(page, base, dept_urls)
        sig = tuple(l["href"] for l in links[:3])
        if sig in seen_first:  # pagination went nowhere
            break
        seen_first.add(sig)
        for l in links:
            found.setdefault(l["href"].split("#")[0], l["text"])
        if not run_js(page, JS_NEXT):
            break
        wait_settle(page, 1200)
    return [{"title": t, "url": u, "department": dept["name"]} for u, t in found.items()]


# --------------------------------------------------------------------------
# Step C: one scheme page
# --------------------------------------------------------------------------
def scrape_scheme(page, context, item):
    if is_pdf(item["url"]):
        p = download_pdf(context, item["url"], item["title"])
        return {
            "title": item["title"], "fields": {}, "content": "",
            "pdfs": [{"title": item["title"], "url": item["url"],
                      "path": str(p.relative_to(ROOT))}] if p else [],
        }
    if not goto(page, item["url"]):
        return None
    title = run_js(page, JS_TITLE).strip()
    fields = {}
    for k, v in run_js(page, JS_FIELDS):
        k = k.rstrip(":").strip()
        fields[k] = v if k not in fields else fields[k] + "\n" + v
    text = clean(run_js(page, JS_MAIN_TEXT))

    pdfs = []
    for l in run_js(page, JS_LINKS):
        if is_pdf(l["href"]):
            p = download_pdf(context, l["href"], l["text"] or item["title"])
            if p:
                pdfs.append({"title": l["text"], "url": l["href"], "path": str(p.relative_to(ROOT))})
    return {
        "title": item["title"] or title,
        "page_title": title,
        "fields": fields,
        "content": text,
        "pdfs": pdfs,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=START_URL)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--all-depts", action="store_true", help="scrape every department, not only agriculture")
    ap.add_argument("--dept", action="append", default=[],
                    help="only departments whose name contains this text (can repeat)")
    ap.add_argument("--list-depts", action="store_true", help="print departments and exit")
    ap.add_argument("--limit", type=int, default=0, help="max schemes to scrape (quick test)")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    records = {}
    if OUT_FILE.exists():
        try:
            records = {r["id"]: r for r in json.loads(OUT_FILE.read_text(encoding="utf-8"))}
            print(f"Resuming: {len(records)} schemes already saved")
        except Exception:
            pass

    def save():
        OUT_FILE.write_text(json.dumps(list(records.values()), ensure_ascii=False, indent=2),
                            encoding="utf-8")

    report = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        context = browser.new_context(
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
            viewport={"width": 1366, "height": 900},
            ignore_https_errors=True,
        )
        page = context.new_page()

        # ---- A. departments ----
        print(f"Opening {args.url}")
        if not goto(page, args.url):
            print("!! could not open the start page"); return
        base = page.url  # schemes.php may redirect
        print(f"  landed on {base}")
        (RAW_DIR / "tngov_start.html").write_text(page.content(), encoding="utf-8")
        page.screenshot(path=str(RAW_DIR / "tngov_start.png"), full_page=True)

        depts = find_departments(page, base)
        report.append(f"Start: {args.url} -> {base}")
        report.append(f"Departments found: {len(depts)}")
        report += [f"  {d['name']}  ->  {d['url']}" for d in depts]

        if args.dept:
            keys = [k.lower() for k in args.dept]
        elif args.all_depts:
            keys = []
        else:
            keys = AGRI_DEPT_KEYWORDS
        chosen = [d for d in depts if not keys or any(k in d["name"].lower() for k in keys)]

        print(f"Departments found: {len(depts)}, selected: {len(chosen)}")
        for d in chosen:
            print(f"  * {d['name']}")
        if args.list_depts:
            for d in depts:
                print(f"    {d['name']}  ->  {d['url']}")
            browser.close(); return

        if not depts:
            # no department links - treat the start page itself as the listing
            print("  (no department links - reading schemes from the start page)")
            chosen = [{"name": "All", "url": base}]
        dept_urls = {d["url"] for d in depts}

        # ---- B. scheme links ----
        items = []
        for d in chosen:
            got = collect_department(page, d, base, dept_urls)
            print(f"  {d['name']}: {len(got)} schemes")
            report.append(f"{d['name']}: {len(got)} schemes")
            items += got
            time.sleep(DELAY)
        if chosen:
            page.screenshot(path=str(RAW_DIR / "tngov_department.png"), full_page=True)
        items = list({i["url"]: i for i in items}.values())
        if args.limit:
            items = items[: args.limit]

        # ---- C. scheme pages ----
        print(f"\nScraping {len(items)} schemes\n")
        try:
            for n, item in enumerate(items, 1):
                rid = rec_id(item["url"])
                if rid in records:
                    continue
                print(f"[{n}/{len(items)}] {item['title'][:80]}")
                data = scrape_scheme(page, context, item)
                if not data:
                    print("    ! failed"); continue
                records[rid] = {
                    "id": rid,
                    "department": item["department"],
                    "source_url": item["url"],
                    **data,
                    "scraped_at": datetime.now().isoformat(timespec="seconds"),
                }
                save()
                print(f"    ✓ {len(data['fields'])} fields, {len(data['content'])} chars, {len(data['pdfs'])} pdfs")
                time.sleep(DELAY)
        except KeyboardInterrupt:
            print("\nStopped - progress saved.")
        finally:
            save()
            report.append(f"Schemes saved: {len(records)}")
            (RAW_DIR / "tngov_report.txt").write_text("\n".join(report), encoding="utf-8")
            browser.close()

    print(f"\nSaved {len(records)} schemes -> {OUT_FILE}")


if __name__ == "__main__":
    main()