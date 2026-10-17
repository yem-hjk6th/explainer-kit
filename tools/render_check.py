"""Render-check hand-written explainer HTML pages (MathJax + layout).

For each page: load in Chromium / Edge via Playwright at 1200 px and 390 px wide, wait for MathJax,
report formula count, MathJax errors (mjx-merror), broken <img>, horizontal overflow.
Run: python -X utf8 tools/render_check.py <page.html> [...]
Needs `pip install playwright` and `playwright install chromium` (or an installed Microsoft Edge).
"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

JS = """() => ({
  formulas: document.querySelectorAll('mjx-container').length,
  merrors: document.querySelectorAll('mjx-merror').length,
  badimgs: [...document.images].filter(i => !i.complete || i.naturalWidth === 0).map(i => i.getAttribute('src')),
  overflow: document.documentElement.scrollWidth - window.innerWidth,
  wide: [...document.querySelectorAll('main *')].filter(e => e.getBoundingClientRect().right > window.innerWidth + 1
          && !e.closest('.tblwrap, pre, mjx-container[display=true]') && !(e.closest('mjx-container') && e.tagName !== 'MJX-CONTAINER')).slice(0, 5).map(e => e.tagName + '.' + e.className + ' ' + (e.closest('p,div,li,td')||{}).textContent?.slice(0, 40))
})"""

with sync_playwright() as p:
    try:
        b = p.chromium.launch()
    except Exception:                      # no bundled chromium: fall back to Edge
        b = p.chromium.launch(channel="msedge")
    for f in sys.argv[1:]:
        url = Path(f).resolve().as_uri()
        for w in (1200, 390):
            pg = b.new_page(viewport={"width": w, "height": 900})
            pg.goto(url)
            pg.wait_for_function("() => window.MathJax && MathJax.startup && MathJax.startup.promise", timeout=30000)
            pg.evaluate("() => MathJax.startup.promise")
            pg.wait_for_timeout(500)
            r = pg.evaluate(JS)
            print(f"{Path(f).name} @{w}px: formulas {r['formulas']}, merror {r['merrors']}, bad img {r['badimgs']}, overflow {r['overflow']}px {r['wide']}")
            pg.close()
    b.close()
