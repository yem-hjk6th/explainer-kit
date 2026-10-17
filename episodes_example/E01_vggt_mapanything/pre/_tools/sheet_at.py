"""Contact sheet at given times (for inspecting pre shots; no video output).
  python -I -X utf8 sheet_at.py <shot.py> <out.jpg> <lang> t1 t2 ...   (3 columns, each cell 360x480)"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]          # repo root: animkit/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(sys.argv[1]).resolve().parent))   # the shot's folder, as tools/render_shot.py does
from PIL import Image, ImageDraw  # noqa: E402

from animkit import i18n  # noqa: E402
from animkit.text import font  # noqa: E402
from animkit.timeline import load_shot  # noqa: E402

script, out, lang = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]
ts = [float(x) for x in sys.argv[4:]]
i18n.setup(lang, script, None)
shot = load_shot(script)
cols, cw, ch = 3, 360, 480
rows = (len(ts) + cols - 1) // cols
sheet = Image.new("RGB", (cols * cw, rows * ch))
dr = ImageDraw.Draw(sheet)
f = font("light", 22)
for k, t in enumerate(ts):
    im = Image.fromarray(shot.render_frame(min(t, shot.dur - 1e-6))).resize((cw, ch), Image.LANCZOS)
    x, y = (k % cols) * cw, (k // cols) * ch
    sheet.paste(im, (x, y))
    dr.text((x + 6, y + 4), f"{t:.1f}s", font=f, fill=(255, 120, 120))
sheet.save(out, quality=88)
print(out, f"dur={shot.dur:.1f}")
