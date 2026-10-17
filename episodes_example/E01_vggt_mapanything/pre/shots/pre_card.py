"""Title cards of the E01 English pre version (English is written directly, no translation-table lookup). card(...) returns a Shot; a card script does `shot = card(...)`."""
from animkit import Shot, style
from animkit.ease import ss


def card(name, dur, kicker, title, sub_lines=(), items=(), accent=style.CYAN):
    shot = Shot(name, dur)

    @shot.check
    def check(ck):
        ck.true("card text has no CJK", all(ord(c) < 0x2E80 for c in kicker + title + "".join(sub_lines) + "".join(items)))

    @shot.frame
    def frame(cv, t):
        a = ss(t, 0.05, 0.45) * (1 - ss(t, dur - 0.3, dur))
        rise = 18 * (1 - ss(t, 0.05, 0.6))
        y0 = 560 if items else 620
        cv.text((540, y0 + rise), kicker, 34, "bold", a, accent, anchor="mm")
        cv.text((540, y0 + 80 + rise), title, 76, "bold", a, style.TEXT, anchor="mm")
        cv.s.line((340, y0 + 140), (740, y0 + 140), accent, a * ss(t, 0.2, 0.7), 2)
        for k, s in enumerate(sub_lines):
            cv.text((540, y0 + 190 + 46 * k), s, 32, "text", a, style.TEXT_DIM, anchor="mm")
        y1 = y0 + 190 + 46 * len(sub_lines) + 40
        for k, s in enumerate(items):
            ak = a * ss(t, 0.35 + 0.18 * k, 0.75 + 0.18 * k)
            cv.text((300, y1 + 64 * k), s, 42, "text", ak, style.TEXT, anchor="lm")

    return shot
