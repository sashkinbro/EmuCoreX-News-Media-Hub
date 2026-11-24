#!/usr/bin/env python3
"""Builds EmuCore theme packages and previews.

Input:  themes/src/<id>/theme.json      (the package's theme.json, format 1)
        themes/src/<id>/listing.json    (Hub listing: categories, tags, description, featured, publishedAt)
        themes/src/<id>/font.ttf|otf, background.webp|jpg|png   (optional package files)
Output: themes/dist/<id>/theme.ecxtheme (deterministic ZIP)
        themes/dist/<id>/preview.webp   (rendered mock of the app in the theme)
        themes/dist/<id>/screen-{hub,settings,game,ingame,store}.webp (gallery screens)

Run locally (needs Pillow) before `npm run build`; commit both src and dist.
The Node build hashes dist files into catalog/v1/themes.json; the validator checks that dist matches src.
"""
import hashlib
import io
import json
import math
import os
import random
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "themes" / "src"
DIST = ROOT / "themes" / "dist"
PACKAGE_FILES = ["theme.json", "font.ttf", "font.otf", "background.webp", "background.jpg", "background.png"]
ZIP_DATE = (2020, 1, 1, 0, 0, 0)
FONTS_DIR = ROOT / "themes" / "fonts"
_FONT_CACHE = {}
NAV_LABELS = ["Library", "Hub", "Store", "Settings"]
GAMES = [("Star Racer", "Racing"), ("Iron Fist", "Fighting"), ("Moon Kingdom", "RPG"), ("Pixel Quest", "Platformer"),
         ("Ocean Hero", "Adventure"), ("Neon Drive", "Racing"), ("Mecha Wars", "Action"), ("Sky Pirates", "Adventure"),
         ("Dark Manor", "Horror"), ("Soccer Pro", "Sports"), ("Rally X", "Racing"), ("Skate City", "Sports"),
         ("Dragon Isle", "RPG"), ("Cube Run", "Puzzle")]
PREVIEW_SIZE = (540, 960)
SCALE = 2  # supersampling for smooth edges


def hex_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def mix(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def source_digest(theme_dir):
    digest = hashlib.sha256()
    for name in PACKAGE_FILES + ["listing.json"]:
        path = theme_dir / name
        if path.is_file():
            data = path.read_bytes()
            if name.endswith(".json"):
                data = data.replace(b"\r\n", b"\n")
            digest.update(name.encode() + b"\0" + data + b"\0")
    digest.update(Path(__file__).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def build_package(theme_dir):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in PACKAGE_FILES:
            path = theme_dir / name
            if not path.is_file():
                continue
            data = path.read_bytes()
            if name.endswith(".json"):
                data = data.replace(b"\r\n", b"\n")
            info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return buffer.getvalue()


# ---------------------------------------------------------------- preview rendering

UI_STYLES = ("standard", "neon", "classic", "luna", "s60", "aero", "metro", "osd", "pixel", "terminal", "paper", "bubble")
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def sample_stops(stops, t):
    if t <= stops[0][0]:
        return stops[0][1]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            return mix(c0, c1, (t - t0) / (t1 - t0) if t1 > t0 else 0)
    return stops[-1][1]

class Painter:
    def __init__(self, theme, background_path, font_path=None):
        self.t = theme
        c = theme["colors"]
        self.c = {k: hex_rgb(v) for k, v in c.items()}
        self.bg = self.c["background"]
        self.surface = self.c["surface"]
        self.variant = self.c["surfaceVariant"]
        self.primary = self.c["primary"]
        self.on_primary = self.c["onPrimary"]
        self.secondary = self.c["secondary"]
        self.tertiary = self.c["tertiary"]
        self.on_bg = self.c["onBackground"]
        self.on_variant = self.c["onSurfaceVariant"]
        self.outline = self.c["outline"]
        accents = [hex_rgb(a) for a in theme.get("accents", [])]
        defaults = [self.primary, self.tertiary, self.secondary]
        self.accents = (accents + defaults[len(accents):])[:3]
        shape = theme.get("shape", {})
        self.style = shape.get("style", "rounded")
        self.scale = float(shape.get("scale", 1))
        effects = theme.get("effects", {})
        ui = effects.get("style", "standard")
        self.ui = ui if ui in UI_STYLES else "standard"
        self.neon = effects.get("decor") == "neon" or self.ui == "neon"
        self.scanlines = bool(effects.get("scanlines"))
        self.dim = float(theme.get("background", {}).get("dim", 0.35))
        self.background_path = background_path
        # Styles that force straight edges / notched pixel corners in the app.
        self.shape_style = "square" if self.ui in ("classic", "pixel", "terminal") else (
            "rounded" if self.ui == "bubble" else self.style)
        self.notch = 0
        self.force_rad = None
        self.solid_mode = False
        self.package_font = font_path
        self._init_structure(theme)
        w, h = PREVIEW_SIZE
        self.img = Image.new("RGB", (w * SCALE, h * SCALE), self.bg)
        self.d = ImageDraw.Draw(self.img)

    def px(self, v):
        return round(v * SCALE)

    # ---- original flat renderer (standard / neon)
    def box(self, x, y, w, h, radius, fill=None, outline=None, width=1):
        x0, y0, x1, y1 = self.px(x), self.px(y), self.px(x + w), self.px(y + h)
        r = self.px(self.force_rad if self.force_rad is not None else max(0.0, radius * self.scale))
        r = min(r, (x1 - x0) // 2, (y1 - y0) // 2)
        lw = self.px(width) if outline else 0
        if self.background_path and fill is not None:
            fill = tuple(fill) + (200,)
        if self.style == "square" or r <= 0:
            self.d.rectangle((x0, y0, x1, y1), fill=fill, outline=outline, width=lw)
        elif self.style == "rounded":
            self.d.rounded_rectangle((x0, y0, x1, y1), radius=r, fill=fill, outline=outline, width=lw)
        else:
            if self.style == "cut":
                pts = [(x0 + r, y0), (x1 - r, y0), (x1, y0 + r), (x1, y1 - r), (x1 - r, y1), (x0 + r, y1), (x0, y1 - r), (x0, y0 + r)]
            elif self.style == "tag":
                pts = [(x0, y0), (x1 - r, y0), (x1, y0 + r), (x1, y1 - r), (x1 - r, y1), (x0, y1)]
            else:  # cutDiagonal: top-end + bottom-start
                pts = [(x0, y0), (x1 - r, y0), (x1, y0 + r), (x1, y1), (x0 + r, y1), (x0, y1 - r)]
            self.d.polygon(pts, fill=fill, outline=outline, width=lw if outline else 0)

    def bar(self, x, y, w, h, color):
        self.d.rounded_rectangle((self.px(x), self.px(y), self.px(x + w), self.px(y + h)), radius=self.px(h / 2), fill=color)

    # ---- shape-clipped layer painting used by the UI style engine
    def _shape(self, d, a, b, c, e, r, fill):
        s = self.shape_style
        if self.notch:
            n = self.notch
            pts = [(a + n, b), (c - n, b), (c - n, b + n), (c, b + n), (c, e - n), (c - n, e - n), (c - n, e),
                   (a + n, e), (a + n, e - n), (a, e - n), (a, b + n), (a + n, b + n)]
            d.polygon(pts, fill=fill)
        elif s == "square" or r <= 0:
            d.rectangle((a, b, c, e), fill=fill)
        elif s == "rounded":
            d.rounded_rectangle((a, b, c, e), radius=r, fill=fill)
        elif s == "cut":
            d.polygon([(a + r, b), (c - r, b), (c, b + r), (c, e - r), (c - r, e), (a + r, e), (a, e - r), (a, b + r)], fill=fill)
        elif s == "tag":
            d.polygon([(a, b), (c - r, b), (c, b + r), (c, e - r), (c - r, e), (a, e)], fill=fill)
        else:
            d.polygon([(a, b), (c - r, b), (c, b + r), (c, e), (a + r, e), (a, e - r)], fill=fill)

    def _mask(self, el, inset=0):
        _, _, mw, mh, r = el
        m = Image.new("L", (mw, mh), 0)
        a, c, e = inset, mw - 1 - inset, mh - 1 - inset
        if c <= a or e <= a:
            return m
        r = max(0, min(r - inset, (c - a) // 2, (e - a) // 2))
        self._shape(ImageDraw.Draw(m), a, a, c, e, r, 255)
        return m

    def _paste(self, el, src, mask, alpha=1.0):
        x0, y0, mw, mh, _ = el
        if alpha < 1.0:
            mask = mask.point(lambda v: int(v * alpha))
        if isinstance(src, tuple):
            src = Image.new("RGB", (mw, mh), src)
        self.img.paste(src, (x0, y0, x0 + mw, y0 + mh), mask)

    def _fill(self, el, color, alpha=1.0):
        self._paste(el, color, self._mask(el), alpha)

    def _grad(self, el, stops, horizontal=False, alpha=1.0):
        mw, mh = el[2], el[3]
        g = Image.new("RGB", (mw, mh))
        gd = ImageDraw.Draw(g)
        n = mw if horizontal else mh
        for i in range(n):
            col = sample_stops(stops, i / max(1, n - 1))
            gd.line((i, 0, i, mh) if horizontal else (0, i, mw, i), fill=col)
        self._paste(el, g, self._mask(el), alpha)

    def _stroke(self, el, color, width, alpha=1.0, dashed=False):
        lw = max(1, self.px(width))
        ring = ImageChops.subtract(self._mask(el), self._mask(el, lw))
        if dashed:
            mw, mh = el[2], el[3]
            pat = Image.new("L", (mw, mh), 0)
            pd = ImageDraw.Draw(pat)
            period, on = self.px(12), self.px(7)
            for x in range(0, mw, period):
                pd.rectangle((x, 0, x + on, lw), fill=255)
                pd.rectangle((x, mh - lw - 1, x + on, mh), fill=255)
            for y in range(0, mh, period):
                pd.rectangle((0, y, lw, y + on), fill=255)
                pd.rectangle((mw - lw - 1, y, mw, y + on), fill=255)
            ring = ImageChops.multiply(ring, pat)
        self._paste(el, color, ring, alpha)

    def _gloss(self, el, alpha, until=0.5, fade=True, inset=0.0):
        """White highlight over the top part of the element, clipped to its shape."""
        mw, mh = el[2], el[3]
        grad = Image.new("L", (mw, mh), 0)
        gd = ImageDraw.Draw(grad)
        lim = max(1, int(mh * until))
        for y in range(lim):
            gd.line((0, y, mw, y), fill=int(255 * (1 - y / lim)) if fade else 255)
        inner = self._mask(el, self.px(inset)) if inset else self._mask(el)
        self._paste(el, WHITE, ImageChops.multiply(grad, inner), alpha)

    def _bevel(self, el, raised, t=1.5):
        x0, y0, mw, mh, _ = el
        w = self.px(t)
        light, dark, mid = WHITE, (64, 64, 64), (128, 128, 128)
        tl, br = (light, dark) if raised else (dark, light)
        x1, y1 = x0 + mw - 1, y0 + mh - 1
        d = self.d
        d.rectangle((x0, y1 - w + 1, x1, y1), fill=br)
        d.rectangle((x1 - w + 1, y0, x1, y1), fill=br)
        d.rectangle((x0, y0, x1 - w, y0 + w - 1), fill=tl)
        d.rectangle((x0, y0, x0 + w - 1, y1 - w), fill=tl)
        if raised:
            d.rectangle((x0 + w, y1 - 2 * w + 1, x1 - w, y1 - w), fill=mid)
            d.rectangle((x1 - 2 * w + 1, y0 + w, x1 - w, y1 - w), fill=mid)
        else:
            d.rectangle((x0 + w, y0 + w, x1 - w, y0 + 2 * w - 1), fill=mid)
            d.rectangle((x0 + w, y0 + w, x0 + 2 * w - 1, y1 - w), fill=mid)

    def _shadow(self, el, blur, alpha, dy):
        x0, y0, mw, mh, _ = el
        pad = self.px(blur * 3)
        m = Image.new("L", (mw + 2 * pad, mh + 2 * pad), 0)
        m.paste(self._mask(el), (pad, pad))
        m = m.filter(ImageFilter.GaussianBlur(self.px(blur))).point(lambda v: int(v * alpha))
        ox, oy = x0 - pad, y0 - pad + self.px(dy)
        self.img.paste(BLACK, (ox, oy, ox + m.width, oy + m.height), m)

    def _elem_style(self, kind, x, y, w, h, r, fill, outline=None, width=1, sel=False):
        """One UI element in the theme's UI style. kind: header, panel, cover, button, chip, plain."""
        st = self.ui
        if st in ("standard", "neon"):
            return self.box(x, y, w, h, r, fill=fill, outline=outline, width=width)
        mult = {"classic": 0, "pixel": 0, "terminal": 0, "metro": 0.25, "bubble": 1.7}.get(st, 1.0)
        rad = r * self.scale * mult
        if st == "bubble" and kind in ("button", "chip"):
            rad = h
        if self.force_rad is not None:
            rad = self.force_rad
        self.notch = 0
        fw = 3 if h >= 50 else 2
        if st == "pixel" and kind != "plain":
            self.notch = self.px(fw)
        px0, py0 = self.px(x), self.px(y)
        el = (px0, py0, self.px(x + w) - px0, self.px(y + h) - py0, self.px(rad))
        P, bg_a = self.primary, (200 / 255 if self.background_path else 1.0)
        f = fill
        card = kind in ("header", "panel", "cover")

        if kind == "plain":
            self._fill(el, f, bg_a)
        elif st == "classic":
            if kind == "header":
                self._grad(el, [(0, P), (1, mix(P, WHITE, 0.42))], horizontal=True)
                self._bevel(el, True)
            else:
                self._fill(el, f, bg_a)
                self._bevel(el, raised=not (sel or kind == "cover"))
        elif st == "luna":
            if kind == "header":
                self._grad(el, [(0, mix(P, WHITE, 0.45)), (0.18, mix(P, WHITE, 0.12)), (0.8, P), (1, mix(P, BLACK, 0.22))])
                self._stroke(el, mix(P, BLACK, 0.5), 1)
            elif kind == "button" or (kind == "chip" and sel):
                base = P if kind == "button" else f
                self._grad(el, [(0, mix(base, WHITE, 0.12)), (1, mix(base, BLACK, 0.06))])
                top = Image.new("L", (el[2], el[3]), 0)
                ImageDraw.Draw(top).rectangle((0, 0, el[2], el[3] // 2 - 1), fill=255)
                self._paste(el, WHITE, ImageChops.multiply(top, self._mask(el)), 0.32)
                self._stroke(el, mix(base, BLACK, 0.4), 1)
            else:
                self._grad(el, [(0, mix(f, WHITE, 0.14)), (1, f)], alpha=bg_a)
                self._stroke(el, mix(self.outline, P, 0.5), 1)
        elif st == "s60":
            if kind == "header":
                self._grad(el, [(0, mix(P, WHITE, 0.28)), (1, mix(P, BLACK, 0.18))])
                self._stroke(el, mix(P, BLACK, 0.35), 1)
            elif kind == "button" or (kind == "chip" and sel):
                base = P if kind == "button" else f
                self._grad(el, [(0, mix(base, WHITE, 0.32)), (1, base)])
                self._stroke(el, mix(base, BLACK, 0.3), 1)
            else:
                self._grad(el, [(0, mix(f, WHITE, 0.16)), (1, f)], alpha=bg_a)
                self._stroke(el, mix(self.outline, f, 0.35), 1)
        elif st == "aero":
            if kind == "button":
                self._fill(el, P, 0.88)
                self._gloss(el, 0.5, 0.55)
                self._stroke(el, WHITE, 1, 0.45)
            else:
                self._fill(el, f, 0.62 if kind != "cover" else 0.85)
                self._gloss(el, 0.38, 0.45)
                self._stroke(el, WHITE, 1, 0.3)
        elif st == "metro":
            solid = kind in ("header", "button")
            self._fill(el, P if solid else f, 1.0 if solid else bg_a)
        elif st == "osd":
            if kind == "header":
                self._grad(el, [(0, mix(P, BLACK, 0.5)), (1, mix(self.secondary, BLACK, 0.5))], horizontal=True, alpha=0.85)
                self._stroke(el, P, 1, 0.9)
            elif kind == "button":
                self._fill(el, P, 0.92)
                self._stroke(el, WHITE, 1, 0.35)
            else:
                self._fill(el, f, 0.66)
                self._stroke(el, P, 1, 0.8 if sel else 0.45)
        elif st == "pixel":
            if kind in ("header", "button"):
                self._fill(el, P)
                self._stroke(el, mix(P, BLACK, 0.45), fw)
            else:
                self._fill(el, f, bg_a)
                self._stroke(el, P if sel else self.outline, fw)
        elif st == "terminal":
            if kind == "button":
                self._fill(el, P)
            else:
                self._fill(el, f, bg_a)
                self._stroke(el, P, 1.6 if h >= 50 else 1.2, 0.7, dashed=True)
        elif st == "paper":
            if card:
                self._shadow(el, 3, 0.20, 2)
            elif kind in ("button", "chip"):
                self._shadow(el, 1.5, 0.14, 1)
            self._fill(el, P if kind == "button" else f)
        elif st == "bubble":
            if kind == "button":
                self._grad(el, [(0, mix(P, WHITE, 0.3)), (1, P)])
                self._gloss(el, 0.55, 0.5, inset=2)
                self._stroke(el, mix(P, WHITE, 0.45), 2, 0.6)
            else:
                self._grad(el, [(0, mix(f, WHITE, 0.55)), (1, f)], alpha=bg_a)
                if kind == "chip" and sel:
                    self._gloss(el, 0.45, 0.5, inset=2)
                self._stroke(el, P, 2, 0.85 if sel else 0.45)
        else:
            self._fill(el, f, bg_a)
        self.notch = 0

    def _init_structure(self, theme):
        def norm(v):
            return str(v or "default").lower().replace("-", "").replace("_", "")

        def pick(value, valid):
            n = norm(value)
            n = "double" if n == "doubleborder" else n
            n = "large-title" if n == "largetitle" else n
            return n if n in valid else None

        comp = theme.get("components", {})
        self.v_card = pick(comp.get("card"), {"elevated", "outlined", "flat", "glass", "inset", "notched", "double"})
        self.v_btn = pick(comp.get("button"), {"filled", "outline", "soft", "gel", "raised3d", "underline", "bracket"})
        self.v_chip = pick(comp.get("chip"), {"pill", "square", "tag"})
        self.v_sw = pick(comp.get("switch"), {"track", "checkbox", "led"})
        self.v_hdr = pick(comp.get("header"), {"solid", "gradient", "underline", "outlined", "glass", "large-title"})
        borders = theme.get("borders", {})
        bw = borders.get("width")
        self.bw = None if bw is None else max(0.0, min(3.0, float(bw)))
        edge = norm(borders.get("accentEdge"))
        self.edge = edge if edge in ("left", "top", "bottom") else "none"
        pat = norm(theme.get("background", {}).get("pattern"))
        self.pattern = pat if pat in ("dots", "grid", "diagonal", "waves", "noise", "circuit", "hex") else "none"
        typo = theme.get("typography", {})
        fk = norm(typo.get("font"))
        self.font_key = fk if fk in ("monospace", "serif", "pixel", "spacemono", "lora", "oswald", "exo2", "package") else "rubik"
        self.title_upper = norm(typo.get("titleCase")) == "upper"
        self.letter_spacing = max(-1.0, min(6.0, float(typo.get("letterSpacing", 0) or 0)))
        ic = norm(theme.get("icons", {}).get("container"))
        self.icons = ic if ic in ("none", "circle", "squircle", "square") else "default"
        seed = int(hashlib.md5(theme["id"].encode()).hexdigest()[:8], 16)
        self.nav_kind = "none"
        self.lib = "grid3"
        self.center = False
        self.nav_h = 0


    def _rad(self, r, h, kind):
        st = self.ui
        if st in ("standard", "neon"):
            return r * self.scale
        mult = {"classic": 0, "pixel": 0, "terminal": 0, "metro": 0.25, "bubble": 1.7}.get(st, 1.0)
        rad = r * self.scale * mult
        if st == "bubble" and kind in ("button", "chip"):
            rad = h
        return rad

    def _el(self, x, y, w, h, r, kind="cover"):
        px0, py0 = self.px(x), self.px(y)
        return (px0, py0, self.px(x + w) - px0, self.px(y + h) - py0, self.px(self._rad(r, h, kind)))

    def _inset(self, el, n):
        x0, y0, mw, mh, r = el
        return (x0 + n, y0 + n, mw - 2 * n, mh - 2 * n, max(0, r - n))

    # ---------------------------------------------------------------- structure-aware element painting
    def elem(self, kind, x, y, w, h, r, fill, outline=None, width=1, sel=False):
        """One UI element. Structural variants (components.*, borders.*) replace the style's painting."""
        if kind == "header" and self.v_hdr:
            return self._hdr_custom(x, y, w, h, r, fill)
        if kind == "button" and self.v_btn:
            return self._btn_custom(x, y, w, h, r, fill, sel)
        if kind in ("panel", "cover") and self.v_card:
            self._card_custom(kind, x, y, w, h, r, fill, outline, width, sel)
            return self._card_extras(kind, x, y, w, h, r)
        if kind == "chip" and self.v_chip:
            saved = (self.shape_style, self.style, self.force_rad)
            if self.v_chip == "pill":
                self.shape_style = self.style = "rounded"
                self.force_rad = h / 2
            elif self.v_chip == "square":
                self.shape_style = self.style = "square"
                self.force_rad = 0
            else:
                self.shape_style = self.style = "tag"
                self.force_rad = min(h * 0.4, 12)
            try:
                return self._elem_style(kind, x, y, w, h, r, fill, outline, width, sel)
            finally:
                self.shape_style, self.style, self.force_rad = saved
        if kind in ("panel", "cover") and self.bw is not None:
            outline = None if self.bw == 0 else outline
            width = self.bw if self.bw else width
        self._elem_style(kind, x, y, w, h, r, fill, outline, width, sel)
        if kind in ("panel", "cover"):
            if self.bw and self.ui not in ("standard", "neon"):
                self._stroke(self._el(x, y, w, h, r, kind), self.outline if not self.neon else self.accents[0], self.bw)
            self._card_extras(kind, x, y, w, h, r)

    def _big(self, w, h):
        return w >= 96 and h >= 40

    def _card_extras(self, kind, x, y, w, h, r):
        """accentEdge stripe (panels only) for card-sized elements."""
        if kind != "panel" or self.edge == "none" or not self._big(w, h):
            return
        el = self._el(x, y, w, h, r, kind)
        t = self.px(3)
        m = Image.new("L", (el[2], el[3]), 0)
        md = ImageDraw.Draw(m)
        if self.edge == "left":
            md.rectangle((0, 0, t, el[3]), fill=255)
        elif self.edge == "top":
            md.rectangle((0, 0, el[2], t), fill=255)
        else:
            md.rectangle((0, el[3] - t, el[2], el[3]), fill=255)
        self._paste(el, self.accents[0], ImageChops.multiply(m, self._mask(el)))

    def _band(self, el, side, thick, color, alpha):
        m = Image.new("L", (el[2], el[3]), 0)
        md = ImageDraw.Draw(m)
        t = self.px(thick)
        if side == "top":
            md.rectangle((0, 0, el[2], t), fill=255)
        else:
            md.rectangle((0, 0, t, el[3]), fill=255)
        self._paste(el, color, ImageChops.multiply(m, self._mask(el)), alpha)

    def _card_custom(self, kind, x, y, w, h, r, fill, outline, width, sel):
        v = self.v_card
        el = self._el(x, y, w, h, r, kind)
        bg_a = 200 / 255 if self.background_path else 1.0
        oc = outline or self.outline
        bw = self.bw if self.bw is not None else max(1.0, float(width or 1))
        if kind == "cover":
            self._fill(el, fill, bg_a)
            if v in ("outlined", "double", "glass", "notched", "inset") or sel:
                self._stroke(el, self.primary if sel else oc, bw if bw else 1)
            if v == "elevated":
                pass
            return
        if v == "elevated":
            self._shadow(el, 4, 0.30, 3)
            self._fill(el, fill)
            if self.bw:
                self._stroke(el, oc, self.bw, 0.6)
        elif v == "outlined":
            self._fill(el, fill, 0.95 if self.solid_mode else (0.38 if not self.background_path else 0.5))
            self._stroke(el, self.primary if sel else oc, max(bw, 1.2) if bw else 0)
        elif v == "flat":
            self._fill(el, fill, bg_a)
            if self.bw:
                self._stroke(el, oc, self.bw, 0.7)
        elif v == "glass":
            self._fill(el, fill, 0.92 if self.solid_mode else 0.58)
            self._gloss(el, 0.30, 0.42)
            self._stroke(el, WHITE, max(1.0, bw), 0.35)
        elif v == "inset":
            self._fill(el, mix(fill, BLACK, 0.16), bg_a)
            self._band(el, "top", 3, BLACK, 0.35)
            self._band(el, "left", 3, BLACK, 0.25)
            if self.bw:
                self._stroke(el, oc, self.bw, 0.5)
        elif v == "notched":
            self._fill(el, fill, bg_a)
            self._stroke(el, oc, max(bw, 1.0) if bw is not None else 1.2)
            if self._big(w, h):
                n = self.px(18)
                x1, y0 = el[0] + el[2], el[1]
                m = Image.new("L", (el[2], el[3]), 0)
                ImageDraw.Draw(m).polygon([(el[2] - n, 0), (el[2], 0), (el[2], n)], fill=255)
                self._paste(el, self.accents[0], ImageChops.multiply(m, self._mask(el)))
        else:  # double
            self._fill(el, fill, bg_a)
            self._stroke(el, oc, max(bw, 1.2) if bw is not None else 1.4)
            if self._big(w, h):
                inner = self._inset(el, self.px(4) + self.px(bw or 1))
                if inner[2] > 8 and inner[3] > 8:
                    self._stroke(inner, oc, 1.0, 0.8)

    def _btn_fg(self, base, filled):
        if filled:
            return self.on_primary if base == self.primary else self.c["onSurface"] if sum(base) > 380 else WHITE
        return base if base == self.primary else self.on_bg

    def btn_color(self, f):
        """Label colour for a button painted with fill f in the current button variant."""
        v = self.v_btn
        filled = v in ("filled", "gel", "raised3d")
        if not v:
            filled = True
        if f == self.primary or f == self.accents[0]:
            return self.on_primary if filled else f
        return self.c["onSurface"]

    def _btn_custom(self, x, y, w, h, r, f, sel):
        v = self.v_btn
        el = self._el(x, y, w, h, r, "button")
        base = f
        if f in (self.surface, self.variant):
            base = self.primary
        if v == "filled":
            self._fill(el, base)
        elif v == "outline":
            self._fill(el, self.surface, 0.25)
            self._stroke(el, base, 1.8)
        elif v == "soft":
            self._fill(el, mix(base, self.surface, 0.78))
        elif v == "gel":
            self._grad(el, [(0, mix(base, WHITE, 0.35)), (0.5, base), (1, mix(base, BLACK, 0.12))])
            self._gloss(el, 0.5, 0.5, inset=2)
            self._stroke(el, mix(base, BLACK, 0.35), 1)
        elif v == "raised3d":
            ledge = (el[0], el[1] + self.px(3), el[2], el[3], el[4])
            self._fill(ledge, mix(base, BLACK, 0.45))
            self._fill(el, base)
        elif v == "underline":
            m = (el[0], el[1] + el[3] - self.px(3), el[2], self.px(3), 0)
            self.d.rectangle((el[0], el[1] + el[3] - self.px(3), el[0] + el[2], el[1] + el[3]), fill=base)
        elif v == "bracket":
            x0, y0, x1, y1 = el[0], el[1], el[0] + el[2], el[1] + el[3]
            L, t = self.px(9), self.px(2)
            for (cx, cy, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
                self.d.rectangle((min(cx, cx + dx * L), cy if dy > 0 else cy - t, max(cx, cx + dx * L), cy + t if dy > 0 else cy), fill=base)
                self.d.rectangle((cx if dx > 0 else cx - t, min(cy, cy + dy * L), cx + t if dx > 0 else cx, max(cy, cy + dy * L)), fill=base)

    def _hdr_custom(self, x, y, w, h, r, f):
        v = self.v_hdr
        el = self._el(x, y, w, h, r, "header")
        if v == "solid":
            self._fill(el, self.primary)
        elif v == "gradient":
            self._grad(el, [(0, self.primary), (1, mix(self.primary, self.secondary, 0.55))], horizontal=True)
        elif v == "underline":
            self.d.rectangle((el[0], el[1] + el[3] - self.px(2.5), el[0] + el[2], el[1] + el[3]), fill=self.accents[0] if self.neon else self.primary)
        elif v == "outlined":
            self._fill(el, self.surface, 0.3)
            self._stroke(el, self.primary, 1.6)
        elif v == "glass":
            self._fill(el, self.surface, 0.5)
            self._gloss(el, 0.25, 0.5)
            self._stroke(el, WHITE, 1, 0.35)
        else:  # large-title: transparent bar, hairline below
            self.d.rectangle((el[0], el[1] + el[3] - self.px(1), el[0] + el[2], el[1] + el[3]), fill=mix(self.outline, self.bg, 0.5))

    # ---------------------------------------------------------------- text
    def _font(self, size, bold):
        key = (self.font_key, round(size * SCALE), bold)
        f = _FONT_CACHE.get(key)
        if f is None:
            path, kind = self.font_file(bold)
            px_size = max(6, round(size * SCALE * (0.72 if kind == "pixel" else 1.0)))
            try:
                f = ImageFont.truetype(str(path), px_size)
                if kind == "var":
                    try:
                        f.set_variation_by_axes([700 if bold else 430] if "oswald" not in str(path) else [600 if bold else 420])
                    except Exception:
                        pass
            except Exception:
                f = ImageFont.load_default()
            _FONT_CACHE[key] = f
        return f

    def font_file(self, bold):
        k = self.font_key
        if k == "pixel":
            return FONTS_DIR / "pressstart2p_regular.ttf", "pixel"
        if k == "spacemono":
            return FONTS_DIR / ("spacemono_bold.ttf" if bold else "spacemono_regular.ttf"), "static"
        if k == "monospace":
            for cand in ("C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
                if Path(cand).is_file():
                    return Path(cand), "static"
            return FONTS_DIR / ("spacemono_bold.ttf" if bold else "spacemono_regular.ttf"), "static"
        if k == "serif":
            for cand in ("C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",):
                if Path(cand).is_file():
                    return Path(cand), "static"
            return FONTS_DIR / "lora_variable.ttf", "var"
        if k == "package" and self.package_font:
            return self.package_font, "static"
        name = {"lora": "lora_variable.ttf", "oswald": "oswald_variable.ttf", "exo2": "exo2_variable.ttf"}.get(k, "rubik_variable.ttf")
        return FONTS_DIR / name, "var"

    def tw(self, s, size, bold=False, ls=0.0):
        return self._font(size, bold).getlength(s) / SCALE + ls * len(s)

    def tx(self, x, y, s, size, color, bold=False, align="l", maxw=None, ls=0.0, upper=False, title=False):
        """Draw text with its top at y (dp). Returns the drawn width in dp."""
        if upper or (title and self.title_upper):
            s = s.upper()
        if title:
            ls = ls + self.letter_spacing
        w = self.tw(s, size, bold, ls)
        if maxw and w > maxw:
            size = max(7.0, size * maxw / w)
            w = self.tw(s, size, bold, ls)
        x0 = x if align == "l" else (x - w / 2 if align == "c" else x - w)
        font = self._font(size, bold)
        if ls == 0:
            self.d.text((self.px(x0), self.px(y)), s, font=font, fill=color)
        else:
            cx = x0
            for ch in s:
                self.d.text((self.px(cx), self.px(y)), ch, font=font, fill=color)
                cx += font.getlength(ch) / SCALE + ls
        return w

    def label(self, x, y, w, h, s, size, color, bold=True, align="c", pad=10, ls=0.0, upper=False):
        font_h = size * 1.05
        tx = x + w / 2 if align == "c" else x + pad
        return self.tx(tx, y + (h - font_h) / 2 - size * 0.08, s, size, color, bold=bold, align=align, maxw=w - 2 * pad, ls=ls, upper=upper)

    # ---------------------------------------------------------------- glyphs and icons
    def glyph(self, cx, cy, r, kind, color):
        p = self.px
        cx, cy, r = p(cx), p(cy), p(r)
        d = self.d
        k = kind % 6
        if k == 0:
            d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=max(2, r // 4))
        elif k == 1:
            d.rounded_rectangle((cx - r, cy - r * 0.8, cx + r, cy + r * 0.8), radius=r // 4, fill=color)
        elif k == 2:
            d.polygon([(cx - r * 0.6, cy - r), (cx - r * 0.6, cy + r), (cx + r, cy)], fill=color)
        elif k == 3:
            d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=color)
        elif k == 4:
            t = max(2, r // 3)
            d.rectangle((cx - r, cy - t // 2, cx + r, cy + t // 2), fill=color)
            d.rectangle((cx - t // 2, cy - r, cx + t // 2, cy + r), fill=color)
        else:
            pts = []
            for i in range(10):
                a = -math.pi / 2 + i * math.pi / 5
                rr = r if i % 2 == 0 else r * 0.45
                pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
            d.polygon(pts, fill=color)

    def icon(self, x, y, s, color, kind=0, active=False):
        """Icon in the theme's icon container with a glyph inside. Top-left (x, y), size s."""
        cont = self.icons
        gcol = self.primary if cont in ("none",) else (self.on_primary if active else mix(color, BLACK, 0.45) if sum(color) > 380 else mix(color, WHITE, 0.6))
        if cont == "none":
            self.glyph(x + s / 2, y + s / 2, s * 0.30, kind, self.primary if not active else self.accents[0])
            return
        el = (self.px(x), self.px(y), self.px(x + s) - self.px(x), self.px(y + s) - self.px(y), 0)
        tile = color
        if cont == "circle":
            self.d.ellipse((el[0], el[1], el[0] + el[2], el[1] + el[3]), fill=tile)
        elif cont == "squircle":
            self.d.rounded_rectangle((el[0], el[1], el[0] + el[2], el[1] + el[3]), radius=self.px(s * 0.36), fill=tile)
        elif cont == "square":
            self.d.rectangle((el[0], el[1], el[0] + el[2], el[1] + el[3]), fill=tile)
        else:
            self.elem("plain", x, y, s, s, 10, tile)
        self.glyph(x + s / 2, y + s / 2, s * 0.26, kind, gcol)

    # ---------------------------------------------------------------- shared page pieces
    def _page(self):
        W, H = PREVIEW_SIZE
        if self.background_path:
            art = Image.open(self.background_path).convert("RGB")
            ratio = max(W * SCALE / art.width, H * SCALE / art.height)
            art = art.resize((math.ceil(art.width * ratio), math.ceil(art.height * ratio)), Image.LANCZOS)
            left = (art.width - W * SCALE) // 2
            top = (art.height - H * SCALE) // 2
            art = art.crop((left, top, left + W * SCALE, top + H * SCALE))
            overlay = Image.new("RGB", art.size, self.bg)
            self.img.paste(Image.blend(art, overlay, self.dim))
            self.d = ImageDraw.Draw(self.img, "RGBA")
        else:
            top = mix(self.bg, self.primary, 0.10)
            for y in range(H * SCALE):
                t = y / (H * SCALE)
                self.d.line((0, y, W * SCALE, y), fill=mix(top, self.bg, min(1, t * 1.6)))
        self._palette()
        self.bottom = H - 16 - self.nav_h - (10 if self.nav_h else 0)

    def _palette(self):
        st = self.ui
        coloured = st in ("classic", "luna", "s60", "metro", "pixel", "osd")
        if self.v_hdr:
            coloured = self.v_hdr in ("solid", "gradient")
        if self.v_hdr in ("solid", "gradient"):
            title_c = self.on_primary
        elif self.v_hdr:
            title_c = self.primary if st == "terminal" else self.on_bg
        elif st == "osd":
            title_c = (240, 240, 240)
        elif coloured:
            title_c = self.on_primary
        elif st == "terminal":
            title_c = self.primary
        else:
            title_c = self.on_bg
        sub_c = mix(title_c, self.primary, 0.35) if coloured else mix(self.on_bg, self.bg, 0.35)
        self.coloured = coloured
        self.title_c, self.sub_c = title_c, sub_c
        self.txt = self.c["onSurface"]
        self.txt2 = mix(self.c["onSurface"], self.surface, 0.38)
        self.tile = mix(self.primary, self.surface, 0.75)

    def _status(self):
        W, H = PREVIEW_SIZE
        col = mix(self.on_bg, self.bg, 0.25)
        self.tx(24, 10, "9:41", 11, col, bold=True)
        self.d.rectangle((self.px(W - 56), self.px(13), self.px(W - 24), self.px(23)), outline=col, width=self.px(1.4))
        self.d.rectangle((self.px(W - 54), self.px(15), self.px(W - 40), self.px(21)), fill=col)
        for i in range(3):
            self.d.rectangle((self.px(W - 90 + i * 7), self.px(21 - 3 - i * 3), self.px(W - 86 + i * 7), self.px(21)), fill=col)

    def _finish(self):
        if self.pattern != "none":
            self._pattern()
        if self.scanlines:
            layer = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
            ld = ImageDraw.Draw(layer)
            for y in range(0, self.img.height, self.px(4)):
                ld.line((0, y, self.img.width, y), fill=(255, 255, 255, 22), width=SCALE)
            self.img = Image.alpha_composite(self.img.convert("RGBA"), layer).convert("RGB")
        return self.img.resize(PREVIEW_SIZE, Image.LANCZOS)

    def _pattern(self):
        W, H = self.img.size
        m = Image.new("L", (W, H), 0)
        md = ImageDraw.Draw(m)
        p = self.px
        kind = self.pattern
        if kind == "dots":
            for yy in range(0, H, p(22)):
                for xx in range((yy // p(22) % 2) * p(11), W, p(22)):
                    md.ellipse((xx - p(1.6), yy - p(1.6), xx + p(1.6), yy + p(1.6)), fill=255)
        elif kind == "grid":
            for xx in range(0, W, p(28)):
                md.line((xx, 0, xx, H), fill=255, width=SCALE)
            for yy in range(0, H, p(28)):
                md.line((0, yy, W, yy), fill=255, width=SCALE)
        elif kind == "diagonal":
            for k in range(-H, W, p(20)):
                md.line((k, 0, k + H, H), fill=255, width=SCALE)
        elif kind == "waves":
            for yy in range(p(20), H, p(36)):
                md.line([(xx, yy + math.sin(xx / p(18)) * p(5)) for xx in range(0, W, p(6))], fill=255, width=SCALE)
        elif kind == "noise":
            rnd = random.Random(7)
            for _ in range(W * H // 90):
                xx, yy = rnd.randrange(W), rnd.randrange(H)
                md.point((xx, yy), fill=255)
        elif kind == "circuit":
            rnd = random.Random(11)
            for _ in range(70):
                xx, yy = rnd.randrange(0, W, p(20)), rnd.randrange(0, H, p(20))
                ln = rnd.randrange(2, 6) * p(20)
                if rnd.random() < 0.5:
                    md.line((xx, yy, xx + ln, yy), fill=255, width=SCALE)
                    md.line((xx + ln, yy, xx + ln + p(20), yy + p(20)), fill=255, width=SCALE)
                    nx_, ny_ = xx + ln + p(20), yy + p(20)
                else:
                    md.line((xx, yy, xx, yy + ln), fill=255, width=SCALE)
                    md.line((xx, yy + ln, xx + p(20), yy + ln + p(20)), fill=255, width=SCALE)
                    nx_, ny_ = xx + p(20), yy + ln + p(20)
                md.ellipse((nx_ - p(3), ny_ - p(3), nx_ + p(3), ny_ + p(3)), outline=255, width=SCALE)
        else:  # hex
            rr = p(22)
            hh = rr * math.sqrt(3)
            for row in range(-1, int(H / hh) + 2):
                for col in range(-1, int(W / (rr * 1.5)) + 2):
                    cx = col * rr * 1.5
                    cy = row * hh + (hh / 2 if col % 2 else 0)
                    pts = [(cx + rr * math.cos(math.pi / 3 * i), cy + rr * math.sin(math.pi / 3 * i)) for i in range(6)]
                    md.line(pts + [pts[0]], fill=255, width=SCALE)
        m = m.point(lambda v: int(v * 0.10))
        ink = self.on_bg
        self.img.paste(ink, (0, 0), m)

    # ---------------------------------------------------------------- composite pieces
    def _chip_row(self, x, y, labels, sel=0, h=32, gap=8, maxw=None):
        for i, lab in enumerate(labels):
            on = i == sel
            width = self.tw(lab, 12.5, on) + 30
            self.elem("chip", x, y, width, h, 10, mix(self.primary, self.surface, 0.55) if on else self.surface,
                      outline=self.primary if on else self.outline, width=1.2, sel=on)
            self.label(x, y, width, h, lab, 12.5, self.txt if on else self.txt2, bold=on, pad=8)
            x += width + gap

    def _button(self, x, y, w, h, text, primary=True, icon_play=False):
        f = self.primary if primary else self.surface
        self.elem("button", x, y, w, h, h / 2, f, outline=self.outline, width=1.2)
        col = self.btn_color(f) if primary else self.txt
        if not primary and self.v_btn in ("underline", "bracket", "outline", "soft", "gel"):
            col = self.txt
        if icon_play:
            tw_ = self.tw(text, 14, True, 0)
            sx = x + w / 2 - tw_ / 2 - 16
            self.d.polygon([(self.px(sx), self.px(y + h / 2 - 7)), (self.px(sx), self.px(y + h / 2 + 7)), (self.px(sx + 12), self.px(y + h / 2))], fill=col)
            self.tx(sx + 20, y + (h - 15) / 2 - 1, text, 14, col, bold=True, maxw=w - 60)
        else:
            self.label(x, y, w, h, text, 14, col, bold=True, upper=self.title_upper and self.ui in ("terminal", "pixel"))

    def _switch(self, x, y, on=True):
        """Switch with its top-left at (x, y), about 52 x 28."""
        v, st = self.v_sw, self.ui
        if v == "checkbox" or (not v and st == "classic") or (not v and st == "terminal" and False):
            self.elem("chip", x + 14, y + 3, 24, 24, 0, WHITE if st == "classic" else self.surface, outline=self.primary, width=2, sel=True)
            if on:
                self.d.line([(self.px(x + 19), self.px(y + 15)), (self.px(x + 24), self.px(y + 20)), (self.px(x + 33), self.px(y + 9))],
                            fill=BLACK if st == "classic" else self.primary, width=self.px(2.6), joint="curve")
        elif v == "led" or (not v and st == "terminal"):
            c = self.primary if on else mix(self.primary, self.bg, 0.65)
            if st == "terminal" and not v:
                w = self.px(2.6)
                for bx, dx in ((x + 2, 1), (x + 50, -1)):
                    self.d.line([(self.px(bx + 7 * dx), self.px(y + 2)), (self.px(bx), self.px(y + 2)), (self.px(bx), self.px(y + 26)),
                                 (self.px(bx + 7 * dx), self.px(y + 26))], fill=c, width=w)
                if on:
                    self.d.line([(self.px(x + 19), self.px(y + 8)), (self.px(x + 33), self.px(y + 20))], fill=c, width=w)
                    self.d.line([(self.px(x + 33), self.px(y + 8)), (self.px(x + 19), self.px(y + 20))], fill=c, width=w)
                else:
                    self.d.rectangle((self.px(x + 20), self.px(y + 13), self.px(x + 32), self.px(y + 15)), fill=c)
            else:
                cx, cy = x + 26, y + 14
                if on:
                    glow = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
                    ImageDraw.Draw(glow).ellipse((self.px(cx - 14), self.px(cy - 14), self.px(cx + 14), self.px(cy + 14)), fill=tuple(self.primary) + (110,))
                    self.img.paste(Image.alpha_composite(self.img.convert("RGBA"), glow.filter(ImageFilter.GaussianBlur(self.px(5)))).convert("RGB"))
                    self.d = ImageDraw.Draw(self.img, "RGBA") if self.background_path else ImageDraw.Draw(self.img)
                self.d.ellipse((self.px(cx - 9), self.px(cy - 9), self.px(cx + 9), self.px(cy + 9)), fill=c if on else None,
                               outline=mix(self.primary, self.bg, 0.2), width=self.px(1.8))
        elif v == "track" or not v and st not in ("pixel",):
            self.elem("chip", x, y, 52, 28, 14, self.primary if on else self.variant, outline=self.primary if on else self.outline, width=1.2, sel=on) \
                if not v else self._track(x, y, on)
            if not v:
                kx = x + 28 if on else x + 4
                self.elem("plain", kx, y + 4, 20, 20, 10, self.on_primary if on else mix(self.outline, self.on_variant, 0.3))
        else:  # pixel default
            self.elem("chip", x + 2, y + 2, 48, 24, 0, self.primary if on else self.variant, outline=self.outline, width=2, sel=on)
            kx = x + 28 if on else x + 6
            self.elem("plain", kx, y + 6, 16, 16, 0, self.on_primary if on else self.outline)

    def _track(self, x, y, on):
        r = self.px(14)
        box = (self.px(x), self.px(y), self.px(x + 52), self.px(y + 28))
        if on:
            self.d.rounded_rectangle(box, radius=r, fill=self.primary)
        else:
            self.d.rounded_rectangle(box, radius=r, fill=self.variant, outline=self.outline, width=self.px(1.4))
        kx = x + 28 if on else x + 5
        kr = (10 if on else 8)
        self.d.ellipse((self.px(kx + 10 - kr), self.px(y + 14 - kr), self.px(kx + 10 + kr), self.px(y + 14 + kr)),
                       fill=self.on_primary if on else mix(self.outline, self.on_variant, 0.3))

    def _rule(self, x, y, w):
        self.d.rectangle((self.px(x), self.px(y), self.px(x + w), self.px(y + 1)), fill=mix(self.outline, self.surface, 0.55))

    def _row(self, x, y, w, i, title, sub, sw=None, chev=False, value=None):
        self.icon(x + 14, y + 10, 40, mix((self.primary, self.secondary, self.tertiary, self.accents[0])[i % 4], self.surface, 0.55), i)
        if self.icons == "none":
            pass
        self.tx(x + 68, y + 11, title, 15, self.txt, bold=True, maxw=w - 160)
        self.tx(x + 68, y + 32, sub, 11.5, self.txt2, maxw=w - 160)
        if sw is not None:
            self._switch(x + w - 70, y + 16, sw)
        elif value:
            self.tx(x + w - 30, y + 20, value, 12.5, self.primary, bold=True, align="r")
        elif chev:
            self.d.line([(self.px(x + w - 30), self.px(y + 24)), (self.px(x + w - 22), self.px(y + 30)), (self.px(x + w - 30), self.px(y + 36))],
                        fill=self.txt2, width=self.px(2.5), joint="curve")

    def _section(self, x, y, text):
        self.tx(x, y, text, 12, self.primary if self.ui != "metro" else self.on_bg, bold=True, upper=True, ls=1.0)

    def _slider(self, x, y, w, frac):
        self.elem("plain", x, y + 8, w, 6, 3, self.variant)
        self.elem("plain", x, y + 8, w * frac, 6, 3, self.primary)
        self.elem("button", x + w * frac - 9, y + 3, 18, 18, 9, self.primary)

    # ---------------------------------------------------------------- screens
    def screen_ingame(self):
        W, H = PREVIEW_SIZE
        self._page()
        gh = 400
        g = Image.new("RGB", (W * SCALE, gh * SCALE))
        gd = ImageDraw.Draw(g)
        hor = 250
        sky_t = mix(self.bg, BLACK, 0.75)
        sky_b = mix(mix(self.primary, self.tertiary, 0.4), BLACK, 0.35)
        for yy in range(gh * SCALE):
            t = yy / SCALE
            if t < hor:
                col = mix(sky_t, sky_b, (t / hor) ** 1.4)
            else:
                col = mix(mix(sky_b, BLACK, 0.5), BLACK, min(1, (t - hor) / (gh - hor)) * 0.6)
            gd.line((0, yy, W * SCALE, yy), fill=col)

        def S(v):
            return round(v * SCALE)

        gd.ellipse((S(360), S(150), S(430), S(220)), fill=mix(self.accents[0], WHITE, 0.2))
        gd.rectangle((0, S(hor - 1), W * SCALE, S(hor + 1)), fill=mix(self.accents[1], WHITE, 0.4))
        gd.polygon([(0, S(hor)), (S(70), S(hor - 56)), (S(150), S(hor - 20)), (S(250), S(hor - 80)), (S(340), S(hor))], fill=mix(sky_t, BLACK, 0.4))
        gd.polygon([(S(300), S(hor)), (S(420), S(hor - 46)), (S(W), S(hor - 10)), (S(W), S(hor))], fill=mix(sky_t, BLACK, 0.5))
        for k in range(8):
            gd.line((S(W / 2), S(hor), S(W / 2 + (k - 3.5) * 120), S(gh)), fill=mix(sky_b, BLACK, 0.2), width=SCALE)
        gd.ellipse((S(250), S(330), S(290), S(370)), fill=mix(self.accents[2], WHITE, 0.1))
        gd.rounded_rectangle((S(20), S(52), S(180), S(64)), radius=S(6), fill=mix(self.secondary, BLACK, 0.2))
        gd.rounded_rectangle((S(20), S(52), S(120), S(64)), radius=S(6), fill=mix(self.secondary, WHITE, 0.3))
        gd.rounded_rectangle((S(W - 130), S(52), S(W - 20), S(64)), radius=S(6), fill=mix(WHITE, BLACK, 0.55))
        self.img.paste(g, (0, 0))
        self.d = ImageDraw.Draw(self.img, "RGBA") if self.background_path else ImageDraw.Draw(self.img)
        self.tx(24, 10, "60 FPS", 11, WHITE, bold=True)
        sy = 372
        self._solid()
        self.elem("panel", 16, sy, W - 32, H - sy - 16, 26, self.surface, outline=self.outline, width=1.2)
        self.elem("plain", W / 2 - 24, sy + 10, 48, 5, 2.5, mix(self.outline, self.surface, 0.3))
        self._button(32, sy + 30, W - 64, 52, "Resume", True, icon_play=True)
        cw = (W - 64 - 12) / 2
        labels = (("Save state", 0), ("Load state", 1), ("Settings", 4), ("Quit game", 3))
        for i, (lab, gk) in enumerate(labels):
            x = 32 + (i % 2) * (cw + 12)
            yy = sy + 96 + (i // 2) * 54
            self.elem("chip", x, yy, cw, 44, 14, self.variant, outline=self.outline, width=1.2)
            self.icon(x + 10, yy + 7, 30, mix(self.accents[i % 3], self.variant, 0.45), gk)
            self.tx(x + 50, yy + 13, lab, 14, self.on_variant if False else self.txt, bold=True, maxw=cw - 60)
        ly = sy + 214
        self.tx(34, ly, "Save slots", 14, self.txt, bold=True)
        self.tx(W - 34, ly + 1, "Slot 2 selected", 11.5, self.txt2, align="r")
        sw_ = (W - 64 - 3 * 10) / 4
        cols = (self.primary, self.secondary, self.tertiary, self.accents[0])
        for i in range(4):
            x = 32 + i * (sw_ + 10)
            on = i == 1
            self.elem("cover", x, ly + 26, sw_, 112, 12, self.surface, outline=self.primary if on else self.outline, width=1.5, sel=on)
            ts = self._inset(self._el(x, ly + 26, sw_, 112, 12), self.px(4))
            ts = (ts[0], ts[1], ts[2], round(ts[3] * 0.58), ts[4])
            self._grad(ts, [(0, mix(cols[i], WHITE, 0.2)), (1, mix(cols[i], BLACK, 0.5))])
            self.tx(x + 8, ly + 26 + 74, f"Slot {i + 1}", 12, self.txt, bold=True, maxw=sw_ - 12)
            self.tx(x + 8, ly + 26 + 92, ("12:04", "Today", "Mar 3", "Empty")[i], 10.5, self.txt2, maxw=sw_ - 12)
        vy = ly + 156
        self.glyph(45, vy + 11, 9, 2, self.txt2)
        self._slider(66, vy, W - 116, 0.7)
        return self._finish()

    def _ico(self, name, cx, cy, col, s=11):
        p, d = self.px, self.d
        w = p(2.2)
        if name == "menu":
            for k in (-1, 0, 1):
                d.rectangle((p(cx - s), p(cy + k * 6 - 1), p(cx + s), p(cy + k * 6 + 1)), fill=col)
        elif name == "back":
            d.line([(p(cx + s * 0.2), p(cy - s * 0.7)), (p(cx - s * 0.6), p(cy)), (p(cx + s * 0.2), p(cy + s * 0.7))], fill=col, width=w, joint="curve")
            d.line([(p(cx - s * 0.6), p(cy)), (p(cx + s), p(cy))], fill=col, width=w)
        elif name == "search":
            d.ellipse((p(cx - s * 0.8), p(cy - s * 0.8), p(cx + s * 0.3), p(cy + s * 0.3)), outline=col, width=w)
            d.line([(p(cx + s * 0.2), p(cy + s * 0.2)), (p(cx + s), p(cy + s))], fill=col, width=w)
        elif name == "bookmark":
            d.line([(p(cx - s * 0.6), p(cy - s)), (p(cx + s * 0.6), p(cy - s)), (p(cx + s * 0.6), p(cy + s)), (p(cx), p(cy + s * 0.5)),
                    (p(cx - s * 0.6), p(cy + s)), (p(cx - s * 0.6), p(cy - s))], fill=col, width=w, joint="curve")
        elif name == "refresh":
            d.arc((p(cx - s * 0.9), p(cy - s * 0.9), p(cx + s * 0.9), p(cy + s * 0.9)), 40, 330, fill=col, width=w)
            d.polygon([(p(cx + s * 0.9), p(cy - s * 0.9)), (p(cx + s * 0.9), p(cy - s * 0.1)), (p(cx + s * 0.1), p(cy - s * 0.1))], fill=col)
        elif name == "tune":
            for k, kx in ((-1, -0.3), (0, 0.4), (1, -0.1)):
                d.line([(p(cx - s), p(cy + k * 6)), (p(cx + s), p(cy + k * 6))], fill=col, width=p(1.6))
                d.ellipse((p(cx + s * kx - 3), p(cy + k * 6 - 3), p(cx + s * kx + 3), p(cy + k * 6 + 3)), fill=col)
        elif name == "pad":
            d.rounded_rectangle((p(cx - s), p(cy - s * 0.6), p(cx + s), p(cy + s * 0.6)), radius=p(5), fill=col)
        elif name == "list":
            d.rounded_rectangle((p(cx - s * 0.8), p(cy - s * 0.9), p(cx + s * 0.8), p(cy - s * 0.1)), radius=p(3), fill=col)
            d.rounded_rectangle((p(cx - s * 0.8), p(cy + s * 0.2), p(cx + s * 0.8), p(cy + s * 0.9)), radius=p(3), fill=col)
        elif name == "close":
            d.line([(p(cx - s * 0.6), p(cy - s * 0.6)), (p(cx + s * 0.6), p(cy + s * 0.6))], fill=col, width=w)
            d.line([(p(cx + s * 0.6), p(cy - s * 0.6)), (p(cx - s * 0.6), p(cy + s * 0.6))], fill=col, width=w)
        elif name == "star":
            self.glyph(cx, cy, s * 0.9, 5, col)
        elif name == "chev":
            d.line([(p(cx - s * 0.3), p(cy - s * 0.6)), (p(cx + s * 0.3), p(cy)), (p(cx - s * 0.3), p(cy + s * 0.6))], fill=col, width=w, joint="curve")
        else:
            self.glyph(cx, cy, s * 0.9, {"palette": 0, "image": 1, "sound": 3, "pad2": 4, "store": 1, "compass": 0}.get(name, 3), col)

    def _header(self, title="Library", sub=None, back=False, right=("tune", "pad", "list")):
        """Status bar + top app bar in the theme's header variant (hamburger or back button on the left)."""
        W, H = PREVIEW_SIZE
        self._status()
        big = self.v_hdr == "large-title"
        hh = 84 if big else 66
        y0 = 34
        self.elem("header", 16, y0, W - 32, hh, 16, self.surface, outline=self.outline if not self.neon else self.accents[0], width=1.5)
        tc, sc = self.title_c, self.sub_c
        by = y0 + (14 if big else hh / 2 - 22) if not big else y0 + 12
        self.elem("chip", 26, by, 44, 44, 12, mix(self.primary, self.surface, 0.55) if not self.coloured else mix(self.primary, WHITE, 0.2), outline=self.outline, width=1)
        self._ico("back" if back else "menu", 48, by + 22, tc if self.coloured else self.txt, 10)
        tx0 = 84
        tsize = 24 if big else 19
        ty = y0 + (12 if big else (hh - tsize - (15 if sub else 0)) / 2 + 1)
        prefix = "> " if self.ui == "terminal" and not self.v_hdr else ""
        w = self.tx(tx0, ty, prefix + title, tsize, tc, bold=True, title=True, maxw=W - tx0 - 20 - 38 * len(right))
        if prefix:
            self.d.rectangle((self.px(tx0 + w + 3), self.px(ty + 2), self.px(tx0 + w + 14), self.px(ty + tsize)), fill=self.primary)
        if sub:
            self.tx(tx0, ty + tsize + 3, sub, 11.5, sc, maxw=W - tx0 - 20 - 38 * len(right))
        for k, name in enumerate(reversed(right)):
            self._ico(name, W - 44 - k * 38, y0 + (hh / 2 if not big else 33), tc, 10)
        yb = y0 + hh
        if self.neon:
            self.d.rectangle((self.px(16), self.px(yb + 3), self.px(16 + (W - 32) * 0.3), self.px(yb + 5)), fill=self.accents[0])
            x2 = 16 + (W - 32) * 0.3 + 4
            self.d.rectangle((self.px(x2), self.px(yb + 3), self.px(x2 + (W - 32) * 0.46), self.px(yb + 5)), fill=self.accents[2])
            x3 = x2 + (W - 32) * 0.46 + 4
            self.d.rectangle((self.px(x3), self.px(yb + 3), self.px(W - 16), self.px(yb + 5)), fill=self.accents[1])
        return yb + 14

    def _seg(self, y, labels, sel, icons=None, h=52, kind=False):
        """Segmented switch (Discover | Store, or the store kind switch with icon above the label)."""
        W = PREVIEW_SIZE[0]
        n = len(labels)
        self.elem("panel", 16, y, W - 32, h, 18, self.surface, outline=self.outline, width=1.2)
        sw = (W - 32 - 12) / n
        for i, lab in enumerate(labels):
            x = 22 + i * sw
            on = i == sel
            if on:
                self.elem("chip", x, y + 6, sw, h - 12, 14, mix(self.primary, self.surface, 0.45), outline=self.primary, width=1.2, sel=True)
            col = self.txt if on else self.txt2
            if kind:
                self._ico(icons[i], x + sw / 2, y + 19, self.primary if on else self.txt2, 9)
                self.tx(x + sw / 2, y + h - 22, lab, 11.5, col, bold=on, align="c", maxw=sw - 6)
            else:
                self._ico(icons[i], x + sw / 2 - self.tw(lab, 14, on) / 2 - 16, y + h / 2, self.primary if on else self.txt2, 9)
                self.tx(x + sw / 2 + 4, y + (h - 16) / 2 - 1, lab, 14, col, bold=on, align="c", maxw=sw - 40)

    def _shelf(self, y, title, more="See all"):
        W = PREVIEW_SIZE[0]
        self.tx(20, y, title, 17, self.on_bg, bold=True, title=True)
        if more:
            self.tx(W - 40, y + 2, more, 13, self.primary, bold=True, align="r")
            self._ico("chev", W - 30, y + 11, self.primary, 7)

    def _nav(self, selected=0):
        return

    def _cover(self, x, y, w, h, i, title=None, sub=None, inside=False):
        pal = [self.primary, self.secondary, self.tertiary, self.accents[0], self.accents[1], self.accents[2]]
        pal = [mix(c_, self.primary, 0.5) if sum(c_) > 690 else c_ for c_ in pal]
        base = pal[i % 6]
        self.elem("cover", x, y, w, h, 10, mix(base, self.variant, 0.4), outline=self.outline, width=1)
        inner = self._inset(self._el(x, y, w, h, 10), self.px(2.5))
        a, b = base, pal[(i + 2) % 6]
        self._grad(inner, [(0, mix(a, WHITE, 0.3)), (0.6, a), (1, mix(b, BLACK, 0.55))])
        k = i % 5
        mx, my = x + w / 2, y + h * (0.55 if inside else 0.42)
        m_ = min(w, h) * 0.2
        if k == 0:
            self.d.ellipse((self.px(mx - m_), self.px(my - m_), self.px(mx + m_), self.px(my + m_)), fill=mix(WHITE, a, 0.2))
        elif k == 1:
            self.d.polygon([(self.px(x + 6), self.px(y + h * 0.85)), (self.px(mx - w * 0.1), self.px(y + h * 0.5)), (self.px(mx + w * 0.2), self.px(y + h * 0.72)),
                            (self.px(x + w - 6), self.px(y + h * 0.55)), (self.px(x + w - 6), self.px(y + h * 0.85))], fill=mix(b, BLACK, 0.5))
        elif k == 2:
            for j in range(3):
                self.d.rectangle((self.px(x + 8 + j * w * 0.28), self.px(y + h * 0.4), self.px(x + 8 + j * w * 0.28 + w * 0.16), self.px(y + h * 0.8)), fill=mix(WHITE, b, 0.4 + j * 0.15))
        elif k == 3:
            for j in range(3):
                rr = min(w, h) * (0.12 + j * 0.1)
                self.d.ellipse((self.px(mx - rr), self.px(my - rr), self.px(mx + rr), self.px(my + rr)), outline=mix(WHITE, a, 0.3), width=self.px(2))
        else:
            self.d.polygon([(self.px(mx), self.px(y + h * 0.3)), (self.px(mx + w * 0.22), self.px(my + 6)), (self.px(mx - w * 0.22), self.px(my + 6))], fill=mix(WHITE, a, 0.25))
        if inside and title:
            self.d.rectangle((self.px(x + 3), self.px(y + 3), self.px(x + 9), self.px(y + h - 3)), fill=mix(base, BLACK, 0.55))
            words = title.split(" ")
            for li, word in enumerate(words[:2]):
                self.tx(x + 16, y + 12 + li * 17, word.upper(), 14, WHITE, bold=True, maxw=w - 24)
            self.tx(x + 16, y + h - 20, "PS2", 9.5, mix(WHITE, base, 0.3), bold=True)
        elif title:
            self.tx(x + 5, y + h + 5, title, 12, self.on_bg, bold=True, maxw=w - 6)
            if sub:
                self.tx(x + 5, y + h + 21, sub, 10, mix(self.on_bg, self.bg, 0.4), maxw=w - 6)

    # ---------------------------------------------------------------- screens
    def render(self):
        """Library: top bar, Pro promo card, Recently opened, grid of tall covers."""
        W, H = PREVIEW_SIZE
        self._page()
        y = self._header("Library", "29 games", right=("tune", "pad", "list"))
        ph = 112
        self.elem("panel", 16, y, W - 32, ph, 18, mix(self.primary, self.surface, 0.82), outline=self.outline, width=1.2)
        self.d.ellipse((self.px(30), self.px(y + 14), self.px(74), self.px(y + 58)), fill=(18, 14, 10))
        self.glyph(52, y + 36, 11, 4, (232, 190, 90))
        self.tx(88, y + 26, "Make EmuCoreX yours", 17, self.txt, bold=True, maxw=W - 190)
        self._ico("close", W - 40, y + 34, self.txt2, 9)
        self.elem("chip", 30, y + 68, 150, 30, 15, (226, 176, 70), outline=(180, 130, 40), width=1)
        self.label(30, y + 68, 150, 30, "PRO  -  $4.99", 12.5, (50, 34, 6), pad=6)
        self._button(W - 172, y + 62, 142, 38, "Get Pro", True)
        y += ph + 16
        self.tx(20, y, "Recently opened", 17, self.on_bg, bold=True, title=True)
        wlab = self.tw("Recently opened", 17, True, self.letter_spacing if self.title_upper else 0)
        self.d.rectangle((self.px(30 + wlab), self.px(y + 11), self.px(W - 20), self.px(y + 12)), fill=mix(self.outline, self.bg, 0.4))
        y += 34
        cols, gap = 3, 12
        cw = (W - 32 - gap * (cols - 1)) / cols
        ch = cw * 1.2
        self._cover(16, y, cw, ch, 4, "Ocean Hero", inside=True)
        y += ch + 12
        self.d.rectangle((self.px(16), self.px(y), self.px(W - 16), self.px(y + 1)), fill=mix(self.outline, self.bg, 0.4))
        y += 14
        for i in range(6):
            cx = 16 + (i % cols) * (cw + gap)
            cy = y + (i // cols) * (ch + gap)
            if cy < H:
                self._cover(cx, cy, cw, ch, i, GAMES[i][0], inside=True)
        return self._finish()

    def screen_hub(self):
        W, H = PREVIEW_SIZE
        self._page()
        y = self._header("EmuCore - Hub", None, back=True, right=("search", "bookmark", "refresh"))
        self._seg(y, ["Discover", "Store"], 0, ["compass", "store"], h=52)
        y += 68
        hh = 224
        self._cover(16, y, W - 32, hh, 6)
        self.elem("chip", 28, y + 12, 80, 28, 8, (20, 20, 24), outline=WHITE, width=1)
        self.label(28, y + 12, 80, 28, "News", 12.5, WHITE, pad=4)
        self.d.rectangle((self.px(19), self.px(y + hh - 74), self.px(W - 19), self.px(y + hh - 3)), fill=(0, 0, 0, 120) if self.background_path else mix(BLACK, self.bg, 0.2))
        self.tx(30, y + hh - 70, "A pocket home for", 16, WHITE, bold=True, maxw=W - 70)
        self.tx(30, y + hh - 50, "handheld emulation on Android", 16, WHITE, bold=True, maxw=W - 70)
        self.tx(30, y + hh - 26, "Sep 19, 2026", 11.5, (210, 210, 220))
        y += hh + 26
        self._shelf(y, "News")
        y += 34
        cw = (W - 32 - 14) / 2
        for i in range(2):
            x = 16 + i * (cw + 14)
            self.elem("panel", x, y, cw, 214, 14, self.surface, outline=self.outline, width=1.2)
            self._cover(x + 4, y + 4, cw - 8, 116, i + 1)
            self.tx(x + 12, y + 128, ("Vulkan renderer gets", "Faster texture cache")[i], 13.5, self.txt, bold=True, maxw=cw - 22)
            self.tx(x + 12, y + 147, ("a big speed boost", "for older phones")[i], 13.5, self.txt, bold=True, maxw=cw - 22)
            self.tx(x + 12, y + 178, "Nov 24, 2026", 11, self.primary, bold=True)
        return self._finish()

    def screen_settings(self):
        W, H = PREVIEW_SIZE
        self._page()
        y = self._header("Settings", "General", back=True, right=())
        x = 16
        for i, (lab, ic) in enumerate((("General", "tune"), ("Video", "image"), ("Controls", "pad2"), ("Emulation", "sound"))):
            on = i == 0
            width = self.tw(lab, 13, on) + 52
            self.elem("chip", x, y, width, 40, 12, mix(self.primary, self.surface, 0.45) if on else self.surface,
                      outline=self.primary if on else self.outline, width=1.2, sel=on)
            self._ico(ic, x + 20, y + 20, self.primary, 8)
            self.tx(x + 36, y + 11, lab, 13, self.txt, bold=on, maxw=width - 40)
            x += width + 10
        y += 56
        self.tx(20, y, "Theme", 17, self.on_bg, bold=True, title=True)
        y += 30
        cx = 16
        for i, lab in enumerate(("System", "Light", "Dark", "Neon")):
            width = self.tw(lab, 13, i == 2) + 34
            on = i == 2
            self.elem("chip", cx, y, width, 38, 10, mix(self.primary, self.surface, 0.55) if on else self.surface,
                      outline=self.primary if on else self.outline, width=1.2, sel=on)
            self.label(cx, y, width, 38, lab, 13, self.txt, bold=on, pad=6)
            cx += width + 10
        y += 54
        sw_ = self.tw("Get more in Store", 14, True) + 70
        self.elem("button", 16, y, sw_, 44, 22, self.primary)
        f = self.primary
        self._ico("store", 38, y + 22, self.btn_color(f), 8)
        self.tx(54, y + 13, "Get more in Store", 14, self.btn_color(f), bold=True)
        y += 62
        rows = [("Theme manager", "Available with PRO", None), ("Keep screen on", "Prevent dimming while playing", True),
                ("Both orientations", "Let the screen rotate in games", False), ("Notch support", "Draw behind the camera cutout", False),
                ("Patch notices", "Show a short note when a patch applies", True)]
        rh = 70
        for i, (t, s, on) in enumerate(rows):
            if y + rh > H - 6:
                break
            self.elem("panel", 16, y, W - 32, rh - 8, 16, self.surface, outline=self.outline, width=1.2)
            self.icon(30, y + 9, 44, mix((self.primary, self.secondary, self.tertiary, self.accents[0])[i % 4], self.surface, 0.55), i)
            self.tx(88, y + 11, t, 15, self.txt, bold=True, maxw=W - 190)
            self.tx(88, y + 33, s, 11.5, self.txt2, maxw=W - 190)
            if on is not None:
                self._switch(W - 86, y + 17, on)
            y += rh
        return self._finish()

    def screen_game(self):
        """Game page behind the real confirmation dialog: eyebrow, icon tile, title, divider, body, right-aligned buttons."""
        W, H = PREVIEW_SIZE
        self._page()
        y = self._header("Library", "29 games", right=("tune", "pad", "list"))
        cols, gap = 3, 12
        cw = (W - 32 - gap * 2) / 3
        ch = cw * 1.2
        for i in range(9):
            cy = y + 4 + (i // 3) * (ch + gap)
            if cy < H:
                self._cover(16 + (i % 3) * (cw + gap), cy, cw, ch, i, GAMES[i][0], inside=True)
        self.img = self.img.filter(ImageFilter.GaussianBlur(self.px(4)))
        self.d = ImageDraw.Draw(self.img, "RGBA") if self.background_path else ImageDraw.Draw(self.img)
        self.img.paste(BLACK, (0, 0, self.img.width, self.img.height), Image.new("L", self.img.size, 150))
        self._solid()
        dw, dh = 452, 262
        dx, dy = (W - dw) / 2, 300
        self.elem("panel", dx, dy, dw, dh, 24, self.surface, outline=self.outline, width=1.5)
        self.tx(dx + 26, dy + 20, "EmuCoreX", 11.5, self.primary, bold=True, upper=True, ls=1.2)
        self.icon(dx + 26, dy + 46, 44, mix(self.tertiary, self.surface, 0.5), 4)
        self.tx(dx + 82, dy + 56, "Remove from library?", 19, self.txt, bold=True, maxw=dw - 110, title=True)
        self._rule(dx + 26, dy + 106, dw - 52)
        self.tx(dx + 26, dy + 124, '"Iron Fist" will be removed from the', 13, self.txt2, maxw=dw - 52)
        self.tx(dx + 26, dy + 144, "list. Your save data stays on the device.", 13, self.txt2, maxw=dw - 52)
        bw_ = 120
        self._button(dx + dw - 26 - bw_ * 2 - 12, dy + dh - 66, bw_, 44, "Cancel", False)
        self._button(dx + dw - 26 - bw_, dy + dh - 66, bw_, 44, "Remove", True)
        return self._finish()

    def screen_drawer(self):
        W, H = PREVIEW_SIZE
        self._page()
        y = self._header("Library", "29 games", right=("tune", "pad", "list"))
        cw = (W - 32 - 24) / 3
        for i in range(6):
            self._cover(16 + (i % 3) * (cw + 12), y + 130 + (i // 3) * (cw * 1.2 + 12), cw, cw * 1.2, i, GAMES[i][0], inside=True)
        if self.ui in ("aero", "osd") or self.v_card in ("glass", "outlined"):
            self.img = self.img.filter(ImageFilter.GaussianBlur(self.px(6)))
            self.d = ImageDraw.Draw(self.img, "RGBA") if self.background_path else ImageDraw.Draw(self.img)
        self.img.paste(BLACK, (0, 0, self.img.width, self.img.height), Image.new("L", self.img.size, 140))
        dw = 372
        self._solid()
        self.elem("panel", -40, -20, dw + 40, H + 40, 28, self.surface, outline=self.outline, width=1.2)
        if self.ui == "bubble":
            self._fill(self._inset(self._el(-40, -20, dw + 40, H + 40, 28, "panel"), self.px(3)), self.surface, 0.6)
        self.d.ellipse((self.px(26), self.px(52), self.px(80), self.px(106)), fill=(18, 14, 10))
        self.glyph(53, 79, 13, 4, (232, 190, 90))
        self.tx(94, 66, "EmuCoreX", 22, self.txt, bold=True, title=True, maxw=dw - 110)
        self.tx(94, 96, "PS2 emulator", 12, self.txt2)
        self._rule(24, 124, dw - 48)
        self.tx(30, 140, "Quick actions", 12, self.primary, bold=True, upper=True, ls=1.0)
        items = ["Library", "PS2 search", "EmuCore - Hub", "Achievements", "Profile", "Discord", "Settings", "About"]
        for i, lab in enumerate(items):
            yy = 166 + i * 58
            if i == 0:
                self.elem("chip", 14, yy, dw - 28, 50, 16, mix(self.primary, self.surface, 0.55), outline=self.primary, width=1.2, sel=True)
            self.icon(28, yy + 10, 30, self.primary if i == 0 else mix(self.txt2, self.surface, 0.6), i, active=i == 0)
            self.tx(72, yy + 15, lab, 15.5, self.txt, bold=i == 0, maxw=dw - 110)
        return self._finish()

    def screen_store(self):
        W, H = PREVIEW_SIZE
        self._page()
        y = self._header("EmuCore - Hub", None, back=True, right=("search", "bookmark", "refresh"))
        self._seg(y, ["Discover", "Store"], 1, ["compass", "store"], h=50)
        y += 62
        self._seg(y, ["Themes", "Wallpapers", "Sounds", "Controls"], 0, ["palette", "image", "sound", "pad2"], h=70, kind=True)
        y += 84
        self.tx(20, y, "Featured", 17, self.on_bg, bold=True, title=True)
        y += 30
        hh = 168
        self.elem("cover", 16, y, W - 32, hh, 20, self.primary, outline=self.outline, width=1)
        inner = self._inset(self._el(16, y, W - 32, hh, 20), self.px(3))
        self._grad(inner, [(0, mix(self.primary, BLACK, 0.55)), (0.6, mix(self.tertiary, BLACK, 0.3)), (1, mix(self.secondary, BLACK, 0.4))], horizontal=True)
        self.elem("plain", 40, y + 28, 150, 14, 7, mix(WHITE, self.primary, 0.1))
        self.elem("plain", 40, y + 50, 100, 8, 4, mix(WHITE, self.primary, 0.4))
        self.elem("panel", 40, y + 76, W - 96, 60, 12, mix(self.primary, BLACK, 0.5), outline=self.accents[0], width=1.2)
        self.tx(34, y + hh - 34, "Arena Fire", 19, WHITE, bold=True, title=True)
        wt = self.tw("Arena Fire", 19, True, self.letter_spacing if self.title_upper else 0)
        self.elem("chip", 46 + wt, y + hh - 36, 52, 26, 8, (20, 20, 24), outline=(226, 176, 70), width=1)
        self.label(46 + wt, y + hh - 36, 52, 26, "PRO", 12, (226, 176, 70), pad=2)
        for k in range(6):
            self.d.ellipse((self.px(W - 120 + k * 14 - (3 if k else 0)), self.px(y + hh - 22), self.px(W - 120 + k * 14 + (14 if k == 0 else 4)), self.px(y + hh - 16)), fill=WHITE if k == 0 else (150, 150, 160))
        y += hh + 24
        self._shelf(y, "New")
        y += 34
        tw_ = (W - 32 - 24) / 3
        names = (("Amber Shell", False), ("Aqua Gel", True), ("Arcade", True))
        for i in range(3):
            x = 16 + i * (tw_ + 12)
            ph = H - y - 10
            self.elem("panel", x, y, tw_, ph, 14, self.surface, outline=self.outline, width=1.2)
            self._mini(x + 4, y + 4, tw_ - 8, ph - 52, i)
            ny = y + ph - 38
            self.tx(x + 10, ny + 4, names[i][0], 12.5, self.txt, bold=True, maxw=tw_ - 20 - (46 if names[i][1] else 0))
            if names[i][1]:
                self.elem("chip", x + tw_ - 52, ny, 44, 24, 8, mix(self.accents[0], self.surface, 0.2), outline=self.accents[0], width=1, sel=True)
                self.label(x + tw_ - 52, ny, 44, 24, "PRO", 10.5, self.txt, pad=2)
        return self._finish()

    def _mini(self, x, y, w, h, i):
        """Tiny theme-preview thumbnail inside a store card."""
        cols = (self.primary, self.secondary, self.tertiary, self.accents[0], self.accents[1])
        base = mix(cols[i % 5], BLACK, 0.55)
        self.d.rounded_rectangle((self.px(x), self.px(y), self.px(x + w), self.px(y + h)), radius=self.px(8), fill=base)
        self.d.rounded_rectangle((self.px(x + 6), self.px(y + 8), self.px(x + w - 6), self.px(y + 30)), radius=self.px(5), fill=mix(cols[(i + 1) % 5], BLACK, 0.2))
        self.d.rounded_rectangle((self.px(x + 6), self.px(y + 36), self.px(x + w - 6), self.px(y + 62)), radius=self.px(5), fill=mix(cols[i % 5], WHITE, 0.15))
        gw = (w - 12 - 8) / 3
        for r in range(3):
            for c in range(3):
                gx = x + 6 + c * (gw + 4)
                gy = y + 70 + r * (gw * 1.2 + 4)
                if gy + gw * 1.2 < y + h - 4:
                    self.d.rounded_rectangle((self.px(gx), self.px(gy), self.px(gx + gw), self.px(gy + gw * 1.2)), radius=self.px(3), fill=mix(cols[(r + c + i) % 5], WHITE, 0.1))


    def _stars(self, x, y, filled=4, n=5, size=18):
        for i in range(n):
            cx, cy, s = x + i * (size + 5) + size / 2, y + size / 2, size / 2
            col = self.accents[0] if i < filled else mix(self.outline, self.surface, 0.4)
            self.glyph(cx, cy, s, 5, col)

    def _solid(self):
        self.background_path = None
        self.solid_mode = True


def find_background(theme_dir):
    return next((theme_dir / n for n in ("background.webp", "background.jpg", "background.png") if (theme_dir / n).is_file()), None)


def find_font(theme_dir):
    return next((theme_dir / n for n in ("font.ttf", "font.otf") if (theme_dir / n).is_file()), None)


def encode_webp(image):
    out = io.BytesIO()
    image.save(out, "WEBP", quality=86, method=6)
    return out.getvalue()


def render_preview(theme, theme_dir):
    return encode_webp(Painter(theme, find_background(theme_dir), find_font(theme_dir)).render())


# Gallery screens shown on the theme detail page: output file -> Painter method.
SCREENS = {
    "screen-hub.webp": "screen_hub",
    "screen-settings.webp": "screen_settings",
    "screen-game.webp": "screen_game",
    "screen-ingame.webp": "screen_ingame",
    "screen-store.webp": "screen_store",
}
MAX_SCREEN_BYTES = 512 * 1024


def render_screens(theme, theme_dir):
    background = find_background(theme_dir)
    result = {}
    for name, method in SCREENS.items():
        data = encode_webp(getattr(Painter(theme, background, find_font(theme_dir)), method)())
        if len(data) > MAX_SCREEN_BYTES:
            raise SystemExit(f"{theme['id']}: {name} is {len(data)} bytes, over the 512 KB limit")
        result[name] = data
    return result


def build_one(args):
    theme_dir, force = args
    theme_id = theme_dir.name
    theme = json.loads((theme_dir / "theme.json").read_text(encoding="utf-8"))
    if theme.get("id") != theme_id:
        raise SystemExit(f"{theme_dir}: theme.json id must equal the directory name")
    out_dir = DIST / theme_id
    stamp_path = out_dir / "build.json"
    digest = source_digest(theme_dir)
    if not force and stamp_path.is_file() and json.loads(stamp_path.read_text())["sourceSha256"] == digest:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    package = build_package(theme_dir)
    if len(package) > 4 * 1024 * 1024:
        raise SystemExit(f"{theme_id}: package exceeds 4 MB")
    (out_dir / "theme.ecxtheme").write_bytes(package)
    (out_dir / "preview.webp").write_bytes(render_preview(theme, theme_dir))
    for name, data in render_screens(theme, theme_dir).items():
        (out_dir / name).write_bytes(data)
    stamp_path.write_text(json.dumps({"sourceSha256": digest}, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 1


def main():
    from concurrent.futures import ProcessPoolExecutor
    only = set(sys.argv[1:])
    jobs = [(p, False) for p in sorted(p for p in SRC.iterdir() if p.is_dir()) if not only or p.name in only]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        built = sum(pool.map(build_one, jobs, chunksize=2))
    # Remove dist output whose source disappeared.
    if DIST.is_dir() and not only:
        for out_dir in DIST.iterdir():
            if out_dir.is_dir() and not (SRC / out_dir.name).is_dir():
                for f in out_dir.iterdir():
                    f.unlink()
                out_dir.rmdir()
    print(f"Built {built} theme package(s).")


if __name__ == "__main__":
    main()
