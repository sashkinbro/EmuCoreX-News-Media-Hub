#!/usr/bin/env python3
"""Builds EmuCore theme packages and previews.

Input:  themes/src/<id>/theme.json      (the package's theme.json, format 1)
        themes/src/<id>/listing.json    (Hub listing: categories, tags, description, featured, publishedAt)
        themes/src/<id>/font.ttf|otf, background.webp|jpg|png   (optional package files)
Output: themes/dist/<id>/theme.ecxtheme (deterministic ZIP)
        themes/dist/<id>/preview.webp   (rendered mock of the app in the theme)
        themes/dist/<id>/screen-{hub,settings,game,ingame,drawer,store}.webp (gallery screens)

Run locally (needs Pillow) before `npm run build`; commit both src and dist.
The Node build hashes dist files into catalog/v1/themes.json; the validator checks that dist matches src.
"""
import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "themes" / "src"
DIST = ROOT / "themes" / "dist"
PACKAGE_FILES = ["theme.json", "font.ttf", "font.otf", "background.webp", "background.jpg", "background.png"]
ZIP_DATE = (2020, 1, 1, 0, 0, 0)
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
    def __init__(self, theme, background_path):
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
        w, h = PREVIEW_SIZE
        self.img = Image.new("RGB", (w * SCALE, h * SCALE), self.bg)
        self.d = ImageDraw.Draw(self.img)

    def px(self, v):
        return round(v * SCALE)

    # ---- original flat renderer (standard / neon)
    def box(self, x, y, w, h, radius, fill=None, outline=None, width=1):
        x0, y0, x1, y1 = self.px(x), self.px(y), self.px(x + w), self.px(y + h)
        r = self.px(max(0.0, radius * self.scale))
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

    def elem(self, kind, x, y, w, h, r, fill, outline=None, width=1, sel=False):
        """One UI element in the theme's UI style. kind: header, panel, cover, button, chip, plain."""
        st = self.ui
        if st in ("standard", "neon"):
            return self.box(x, y, w, h, r, fill=fill, outline=outline, width=width)
        mult = {"classic": 0, "pixel": 0, "terminal": 0, "metro": 0.25, "bubble": 1.7}.get(st, 1.0)
        rad = r * self.scale * mult
        if st == "bubble" and kind in ("button", "chip"):
            rad = h
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

    # ---- shared page pieces (used by the catalog preview and the gallery screens)
    def _page(self):
        W, H = PREVIEW_SIZE
        # Page background: theme background image (dimmed) or a subtle vertical gradient.
        if self.background_path:
            art = Image.open(self.background_path).convert("RGB")
            ratio = max(W * SCALE / art.width, H * SCALE / art.height)
            art = art.resize((math.ceil(art.width * ratio), math.ceil(art.height * ratio)), Image.LANCZOS)
            left = (art.width - W * SCALE) // 2
            top = (art.height - H * SCALE) // 2
            art = art.crop((left, top, left + W * SCALE, top + H * SCALE))
            overlay = Image.new("RGB", art.size, self.bg)
            self.img.paste(Image.blend(art, overlay, self.dim))
            # Draw the UI blended so the theme art stays visible through the panels.
            self.d = ImageDraw.Draw(self.img, "RGBA")
        else:
            top = mix(self.bg, self.primary, 0.10)
            for y in range(H * SCALE):
                t = y / (H * SCALE)
                self.d.line((0, y, W * SCALE, y), fill=mix(top, self.bg, min(1, t * 1.6)))
        self._palette()

    def _palette(self):
        st = self.ui
        # Header text colours depend on whether the style paints a coloured header.
        coloured = st in ("classic", "luna", "s60", "metro", "pixel", "osd")
        if st == "osd":
            title_c = (240, 240, 240)
        elif coloured:
            title_c = self.on_primary
        elif st == "terminal":
            title_c = self.primary
        else:
            title_c = self.on_bg
        sub_c = mix(title_c, self.primary, 0.35) if coloured else self.on_variant
        if coloured and st != "osd":
            icon_c, right_c = mix(self.primary, WHITE, 0.3), mix(self.primary, WHITE, 0.2)
        else:
            icon_c = self.c.get("primaryContainer") or mix(self.primary, self.surface, 0.75)
            right_c = self.variant
        self.title_c, self.sub_c, self.icon_c, self.right_c = title_c, sub_c, icon_c, right_c

    def _header(self, back=False, title_w=170, sub_w=110):
        W, H = PREVIEW_SIZE
        st = self.ui
        title_c, sub_c, icon_c, right_c = self.title_c, self.sub_c, self.icon_c, self.right_c
        # Status bar + top app bar
        self.bar(24, 18, 46, 8, mix(self.on_bg, self.bg, 0.45))
        self.bar(W - 86, 18, 62, 8, mix(self.on_bg, self.bg, 0.45))
        self.elem("header", 16, 44, W - 32, 64, 18, self.surface, outline=self.outline if not self.neon else self.accents[0], width=1.5)
        if st == "terminal":
            # "> TITLE_" : chevron prompt, title bar, block cursor
            c = self.primary
            pts = [(self.px(32), self.px(64)), (self.px(46), self.px(76)), (self.px(32), self.px(88))]
            self.d.line(pts, fill=c, width=self.px(4), joint="curve")
            self.bar(58, 66, min(title_w, 150), 14, title_c)
            self.d.rectangle((self.px(214), self.px(64), self.px(226), self.px(82)), fill=c)
            self.bar(58, 86, sub_w, 9, sub_c)
        else:
            if back:
                self.d.line([(self.px(46), self.px(65)), (self.px(36), self.px(76)), (self.px(46), self.px(87))], fill=title_c, width=self.px(3.5), joint="curve")
            else:
                self.elem("plain", 30, 58, 36, 36, 10, icon_c)
            self.bar(80, 66, title_w, 14, title_c)
            self.bar(80, 86, sub_w, 9, sub_c)
        self.elem("plain", W - 72, 60, 40, 32, 10, right_c)
        if self.neon:
            self.d.rectangle((self.px(16), self.px(112), self.px(16 + (W - 32) * 0.3), self.px(114)), fill=self.accents[0])
            x2 = 16 + (W - 32) * 0.3 + 4
            self.d.rectangle((self.px(x2), self.px(112), self.px(x2 + (W - 32) * 0.46), self.px(114)), fill=self.accents[2])
            x3 = x2 + (W - 32) * 0.46 + 4
            self.d.rectangle((self.px(x3), self.px(112), self.px(W - 16), self.px(114)), fill=self.accents[1])

    def _nav(self, selected=0):
        W, H = PREVIEW_SIZE
        # Bottom navigation
        nav_y = H - 78
        self.elem("panel", 16, nav_y, W - 32, 62, 22, self.surface, outline=self.outline, width=1)
        for i in range(4):
            cx = 16 + (W - 32) * (i + 0.5) / 4
            if i == selected:
                self.elem("chip", cx - 30, nav_y + 12, 60, 38, 19, mix(self.primary, self.surface, 0.6), sel=True)
            self.elem("plain", cx - 11, nav_y + 20, 22, 22, 6, self.primary if i == selected else self.on_variant)

    def _finish(self):
        if self.scanlines:
            layer = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
            ld = ImageDraw.Draw(layer)
            for y in range(0, self.img.height, self.px(4)):
                ld.line((0, y, self.img.width, y), fill=(255, 255, 255, 22), width=SCALE)
            self.img = Image.alpha_composite(self.img.convert("RGBA"), layer).convert("RGB")

        return self.img.resize(PREVIEW_SIZE, Image.LANCZOS)

    # ---- helpers for the gallery screens
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

    def _art(self, x, y, w, h, r, stops, horizontal=False):
        """Framed picture block (cover style) with a gradient inside, clipped to the style's shape."""
        self.elem("cover", x, y, w, h, r, mix(self.primary, self.variant, 0.5), outline=self.outline, width=1)
        inner = self._inset(self._el(x, y, w, h, r), self.px(3))
        self._grad(inner, stops, horizontal=horizontal)

    def _solid(self):
        """Overlays (drawer, sheet) sit on top of other content: no see-through theme art for them."""
        self.background_path = None

    def _rule(self, x, y, w):
        self.d.rectangle((self.px(x), self.px(y), self.px(x + w), self.px(y + 1)), fill=mix(self.outline, self.surface, 0.55))

    def _chip_row(self, x, y, widths, sel=0, h=34, gap=10):
        for i, width in enumerate(widths):
            on = i == sel
            self.elem("chip", x, y, width, h, 10, mix(self.primary, self.surface, 0.55) if on else self.surface,
                      outline=self.primary if on else self.outline, width=1.2, sel=on)
            self.bar(x + 16, y + h / 2 - 4, width - 32, 8, self.on_bg if on else self.on_variant)
            x += width + gap

    def _stars(self, x, y, filled=4, n=5, size=20):
        for i in range(n):
            cx, cy, s = self.px(x + i * (size + 6) + size / 2), self.px(y + size / 2), self.px(size / 2)
            col = self.accents[0] if i < filled else mix(self.outline, self.surface, 0.4)
            self.d.polygon([(cx, cy - s), (cx + s * 0.8, cy), (cx, cy + s), (cx - s * 0.8, cy)], fill=col)

    def _switch(self, x, y, on=True):
        """Style-appropriate switch with its top-left at (x, y), roughly 52 x 28."""
        st = self.ui
        if st == "classic":
            self.elem("chip", x + 14, y + 3, 22, 22, 0, WHITE, sel=True)
            if on:
                self.d.line([(self.px(x + 19), self.px(y + 14)), (self.px(x + 23), self.px(y + 19)), (self.px(x + 31), self.px(y + 9))],
                            fill=BLACK, width=self.px(2.4), joint="curve")
        elif st == "terminal":
            c = self.primary if on else mix(self.primary, self.bg, 0.55)
            w = self.px(2.6)
            for bx, dx in ((x + 2, 1), (x + 50, -1)):
                self.d.line([(self.px(bx + 7 * dx), self.px(y + 2)), (self.px(bx), self.px(y + 2)), (self.px(bx), self.px(y + 26)),
                             (self.px(bx + 7 * dx), self.px(y + 26))], fill=c, width=w)
            if on:
                self.d.line([(self.px(x + 19), self.px(y + 8)), (self.px(x + 33), self.px(y + 20))], fill=c, width=w)
                self.d.line([(self.px(x + 33), self.px(y + 8)), (self.px(x + 19), self.px(y + 20))], fill=c, width=w)
            else:
                self.d.rectangle((self.px(x + 20), self.px(y + 13), self.px(x + 32), self.px(y + 15)), fill=c)
        elif st == "pixel":
            self.elem("chip", x + 2, y + 2, 48, 24, 0, self.primary if on else self.variant, outline=self.outline, width=2, sel=on)
            kx = x + 28 if on else x + 6
            self.elem("plain", kx, y + 6, 16, 16, 0, self.on_primary if on else self.outline)
        else:
            self.elem("chip", x, y, 52, 28, 14, self.primary if on else self.variant, outline=self.primary if on else self.outline, width=1.2, sel=on)
            kx = x + 28 if on else x + 4
            self.elem("plain", kx, y + 4, 20, 20, 10, self.on_primary if on else mix(self.outline, self.on_variant, 0.3))

    def _backdrop_library(self):
        """Library behind overlays: a header plus a cover grid."""
        W, H = PREVIEW_SIZE
        self._header()
        palette = [self.primary, self.secondary, self.tertiary, self.accents[0], self.accents[1], self.accents[2]]
        cols, gap = 3, 12
        cw = (W - 32 - gap * (cols - 1)) / cols
        ch = cw * 4 / 3
        for i in range(9):
            cx = 16 + (i % cols) * (cw + gap)
            cy = 130 + (i // cols) * (ch + 42)
            self.elem("cover", cx, cy, cw, ch, 14, mix(palette[i % 6], self.variant, 0.45), outline=self.outline, width=1)
            self.bar(cx + 2, cy + ch + 10, cw * 0.8, 10, self.on_bg)

    def _row(self, x, y, w, tile_c, a=170, b=110, sw=None, chev=False):
        """List row: icon tile, two text bars, optional switch / chevron on the right."""
        self.elem("plain", x + 14, y + 10, 40, 40, 11, tile_c)
        self.bar(x + 68, y + 15, a, 12, self.on_bg)
        self.bar(x + 68, y + 35, b, 8, self.on_variant)
        if sw is not None:
            self._switch(x + w - 70, y + 16, sw)
        elif chev:
            self.d.line([(self.px(x + w - 30), self.px(y + 24)), (self.px(x + w - 22), self.px(y + 30)), (self.px(x + w - 30), self.px(y + 36))],
                        fill=self.on_variant, width=self.px(2.5), joint="curve")

    # ---- gallery screens
    def screen_hub(self):
        W, H = PREVIEW_SIZE
        self._page()
        self._header(title_w=130, sub_w=90)
        self._chip_row(16, 126, (84, 108, 96, 92))
        pal = [(self.primary, self.tertiary), (self.secondary, self.primary)]
        for i, y in enumerate((178, 514)):
            self.elem("panel", 16, y, W - 32, 322, 22, self.surface, outline=self.outline, width=1.2)
            a, b = pal[i]
            self._art(28, y + 12, W - 56, 188, 14, [(0, mix(a, WHITE, 0.25)), (0.55, a), (1, mix(b, BLACK, 0.35))], horizontal=(i == 1))
            # picture decoration kept well inside the picture
            self.d.ellipse((self.px(W - 128), self.px(y + 36), self.px(W - 76), self.px(y + 88)), fill=mix(WHITE, a, 0.25))
            self.d.polygon([(self.px(44), self.px(y + 186)), (self.px(150), self.px(y + 120)), (self.px(240), self.px(y + 168)),
                            (self.px(330), self.px(y + 110)), (self.px(W - 44), self.px(y + 186))], fill=mix(b, BLACK, 0.55))
            self.elem("chip", 44, y + 24, 64, 24, 8, mix(self.primary, self.surface, 0.3), sel=True)
            self.bar(56, y + 33, 40, 6, self.on_bg)
            self.bar(32, y + 216, 360, 16, self.on_bg)
            self.bar(32, y + 242, 300, 16, self.on_bg)
            self.bar(32, y + 280, 230, 9, self.on_variant)
            self.bar(W - 130, y + 280, 82, 9, self.on_variant)
        self._nav(1)
        return self._finish()

    def screen_settings(self):
        W, H = PREVIEW_SIZE
        self._page()
        self._header(back=True, title_w=140, sub_w=80)
        self._chip_row(16, 126, (92, 96, 88, 104), sel=1)
        y0, rh = 178, 62
        self.elem("panel", 16, y0, W - 32, rh * 4 + 16, 22, self.surface, outline=self.outline, width=1.2)
        tiles = [self.primary, self.secondary, self.tertiary, self.accents[0]]
        for i, state in enumerate((True, True, False, True)):
            ry = y0 + 8 + i * rh
            self._row(16, ry, W - 32, mix(tiles[i], self.surface, 0.55), a=(180, 140, 200, 160)[i], b=(120, 150, 90, 130)[i], sw=state)
            if i < 3:
                self._rule(86, ry + rh - 1, W - 118)
        sy = y0 + rh * 4 + 38
        self.bar(20, sy, 120, 11, self.on_bg)
        self.bar(20, sy + 18, 200, 8, self.on_variant)
        cw = (W - 32 - 20) / 3
        for i in range(3):
            on = i == 1
            self.elem("chip", 16 + i * (cw + 10), sy + 40, cw, 42, 12, mix(self.primary, self.surface, 0.55) if on else self.surface,
                      outline=self.primary if on else self.outline, width=1.2, sel=on)
            self.bar(16 + i * (cw + 10) + cw / 2 - 30, sy + 56, 60, 9, self.on_bg if on else self.on_variant)
        py = sy + 106
        self.elem("panel", 16, py, W - 32, 3 * rh + 16, 22, self.surface, outline=self.outline, width=1.2)
        self._row(16, py + 8, W - 32, mix(self.secondary, self.surface, 0.55), a=150, b=100, chev=True)
        self._rule(86, py + 8 + rh - 1, W - 118)
        sy2 = py + 8 + rh
        self.elem("plain", 30, sy2 + 10, 40, 40, 11, mix(self.tertiary, self.surface, 0.55))
        self.bar(84, sy2 + 14, 130, 12, self.on_bg)
        tx0, tx1 = 84, W - 56
        self.elem("plain", tx0, sy2 + 38, tx1 - tx0, 8, 4, self.variant)
        self.elem("plain", tx0, sy2 + 38, (tx1 - tx0) * 0.62, 8, 4, self.primary)
        self.elem("button", tx0 + (tx1 - tx0) * 0.62 - 9, sy2 + 33, 18, 18, 9, self.primary)
        self._rule(86, sy2 + rh - 1, W - 118)
        self._row(16, sy2 + rh, W - 32, mix(self.accents[1], self.surface, 0.55), a=120, b=170, chev=True)
        self._nav(3)
        return self._finish()

    def screen_game(self):
        W, H = PREVIEW_SIZE
        self._page()
        self._header(back=True, title_w=120, sub_w=70)
        cy, chh = 126, 340
        a, b = self.primary, self.tertiary
        self._art(16, cy, W - 32, chh, 24, [(0, mix(a, WHITE, 0.2)), (0.6, mix(a, b, 0.5)), (1, mix(b, BLACK, 0.45))])
        # console silhouette made of rounded rects
        dark = mix(self.primary, BLACK, 0.72)
        body = mix(dark, WHITE, 0.08)
        mx, my, mw, mh = W / 2 - 150, cy + 82, 300, 150

        def rr(x0, y0, x1, y1, r, f):
            self.d.rounded_rectangle((self.px(x0), self.px(y0), self.px(x1), self.px(y1)), radius=self.px(r), fill=f)

        rr(mx, my, mx + mw, my + mh, 26, body)
        rr(mx + 52, my + 16, mx + mw - 52, my + mh - 18, 6, mix(dark, self.secondary, 0.35))
        rr(mx + 58, my + 22, mx + mw - 58, my + mh - 24, 3, mix(self.secondary, WHITE, 0.35))
        rr(mx + 17, my + 62, mx + 41, my + 70, 3, mix(body, WHITE, 0.3))
        rr(mx + 25, my + 54, mx + 33, my + 78, 3, mix(body, WHITE, 0.3))
        for bx, by in ((mx + mw - 30, my + 58), (mx + mw - 20, my + 70)):
            self.d.ellipse((self.px(bx - 6), self.px(by - 6), self.px(bx + 6), self.px(by + 6)), fill=mix(body, self.accents[0], 0.7))
        rr(mx + 20, my + mh + 16, mx + mw - 20, my + mh + 26, 5, mix(b, BLACK, 0.55))
        # title + rating
        self.bar(20, cy + chh + 22, 280, 20, self.on_bg)
        self.bar(20, cy + chh + 52, 170, 11, self.on_variant)
        self._stars(20, cy + chh + 78, 4)
        self.bar(20 + 5 * 26 + 12, cy + chh + 84, 70, 9, self.on_variant)
        # actions
        by = cy + chh + 120
        self.elem("button", 16, by, 224, 52, 26, self.primary)
        self.d.polygon([(self.px(58), self.px(by + 15)), (self.px(58), self.px(by + 37)), (self.px(76), self.px(by + 26))], fill=self.on_primary)
        self.bar(90, by + 21, 110, 10, self.on_primary)
        for x in (250, 392):
            self.elem("chip", x, by, 132, 52, 26, self.surface, outline=self.outline, width=1.2)
            self.elem("plain", x + 18, by + 17, 18, 18, 5, self.on_variant)
            self.bar(x + 46, by + 21, 66, 9, self.on_variant)
        # info chips
        iy = by + 72
        x = 16
        for width in (96, 112, 88, 118):
            self.elem("chip", x, iy, width, 34, 10, self.surface, outline=self.outline, width=1.2)
            self.elem("plain", x + 10, iy + 10, 14, 14, 4, self.accents[1])
            self.bar(x + 32, iy + 13, width - 46, 8, self.on_variant)
            x += width + 10
        dy = iy + 54
        self.elem("panel", 16, dy, W - 32, 108, 20, self.surface, outline=self.outline, width=1.2)
        self.bar(32, dy + 20, 140, 11, self.on_bg)
        for k, wd in enumerate((440, 420, 300)):
            self.bar(32, dy + 46 + k * 18, wd, 8, self.on_variant)
        self._nav(0)
        return self._finish()

    def screen_ingame(self):
        W, H = PREVIEW_SIZE
        self._page()
        gh = 470
        g = Image.new("RGB", (W * SCALE, gh * SCALE))
        gd = ImageDraw.Draw(g)
        hor = 300
        sky_t = mix(self.bg, BLACK, 0.75)
        sky_b = mix(mix(self.primary, self.tertiary, 0.4), BLACK, 0.35)
        for y in range(gh * SCALE):
            t = y / SCALE
            if t < hor:
                col = mix(sky_t, sky_b, (t / hor) ** 1.4)
            else:
                col = mix(mix(sky_b, BLACK, 0.5), BLACK, min(1, (t - hor) / (gh - hor)) * 0.6)
            gd.line((0, y, W * SCALE, y), fill=col)

        def S(v):
            return round(v * SCALE)

        gd.ellipse((S(360), S(190), S(430), S(260)), fill=mix(self.accents[0], WHITE, 0.2))
        gd.rectangle((0, S(hor - 1), W * SCALE, S(hor + 1)), fill=mix(self.accents[1], WHITE, 0.4))
        gd.polygon([(0, S(hor)), (S(70), S(hor - 56)), (S(150), S(hor - 20)), (S(250), S(hor - 80)), (S(340), S(hor))], fill=mix(sky_t, BLACK, 0.4))
        gd.polygon([(S(300), S(hor)), (S(420), S(hor - 46)), (S(W), S(hor - 10)), (S(W), S(hor))], fill=mix(sky_t, BLACK, 0.5))
        for k in range(8):
            gd.line((S(W / 2), S(hor), S(W / 2 + (k - 3.5) * 120), S(gh)), fill=mix(sky_b, BLACK, 0.2), width=SCALE)
        gd.ellipse((S(250), S(380), S(290), S(420)), fill=mix(self.accents[2], WHITE, 0.1))
        gd.rounded_rectangle((S(20), S(52), S(180), S(64)), radius=S(6), fill=mix(self.secondary, BLACK, 0.2))
        gd.rounded_rectangle((S(20), S(52), S(120), S(64)), radius=S(6), fill=mix(self.secondary, WHITE, 0.3))
        gd.rounded_rectangle((S(W - 130), S(52), S(W - 20), S(64)), radius=S(6), fill=mix(WHITE, BLACK, 0.55))
        self.img.paste(g, (0, 0))
        self.bar(24, 18, 46, 8, mix(WHITE, BLACK, 0.4))
        self.bar(W - 86, 18, 62, 8, mix(WHITE, BLACK, 0.4))
        # bottom sheet with the menu
        sy = 440
        self._solid()
        self.elem("panel", 16, sy, W - 32, H - sy - 16, 28, self.surface, outline=self.outline, width=1.2)
        self.elem("plain", W / 2 - 24, sy + 12, 48, 5, 2.5, mix(self.outline, self.surface, 0.3))
        self.elem("button", 32, sy + 34, W - 64, 54, 27, self.primary)
        self.d.polygon([(self.px(W / 2 - 62), self.px(sy + 50)), (self.px(W / 2 - 62), self.px(sy + 72)), (self.px(W / 2 - 44), self.px(sy + 61))], fill=self.on_primary)
        self.bar(W / 2 - 28, sy + 56, 96, 10, self.on_primary)
        cw = (W - 64 - 12) / 2
        icons = (self.accents[0], self.accents[1], self.secondary, self.tertiary)
        for i, lw in enumerate((110, 110, 90, 80)):
            x = 32 + (i % 2) * (cw + 12)
            y = sy + 102 + (i // 2) * 56
            self.elem("chip", x, y, cw, 46, 14, self.variant, outline=self.outline, width=1.2)
            self.elem("plain", x + 16, y + 13, 20, 20, 6, icons[i])
            self.bar(x + 48, y + 19, lw, 9, self.on_bg)
        ly = sy + 222
        self.bar(34, ly, 110, 11, self.on_bg)
        self.bar(W - 130, ly, 94, 9, self.on_variant)
        sw_ = (W - 64 - 3 * 10) / 4
        cols = (self.primary, self.secondary, self.tertiary, self.accents[0])
        for i in range(4):
            x = 32 + i * (sw_ + 10)
            on = i == 1
            self.elem("cover", x, ly + 24, sw_, 108, 14, self.surface, outline=self.primary if on else self.outline, width=1.5, sel=on)
            ts = self._inset(self._el(x, ly + 24, sw_, 108, 14), self.px(4))
            ts = (ts[0], ts[1], ts[2], round(ts[3] * 0.62), ts[4])
            self._grad(ts, [(0, mix(cols[i], WHITE, 0.2)), (1, mix(cols[i], BLACK, 0.5))])
            self.bar(x + 8, ly + 24 + 80, sw_ - 30, 7, self.on_bg)
            self.bar(x + 8, ly + 24 + 94, sw_ - 44, 6, self.on_variant)
        vy = ly + 154
        self.elem("plain", 34, vy, 22, 22, 6, self.on_variant)
        self.elem("plain", 72, vy + 8, W - 112, 6, 3, self.variant)
        self.elem("plain", 72, vy + 8, (W - 112) * 0.7, 6, 3, self.primary)
        self.elem("button", 72 + (W - 112) * 0.7 - 8, vy + 3, 16, 16, 8, self.primary)
        return self._finish()

    def screen_drawer(self):
        W, H = PREVIEW_SIZE
        self._page()
        self._backdrop_library()
        self._nav(0)
        if self.ui in ("aero", "osd"):  # translucent drawer: blur what shows through
            self.img = self.img.filter(ImageFilter.GaussianBlur(self.px(6)))
            self.d = ImageDraw.Draw(self.img, "RGBA") if self.background_path else ImageDraw.Draw(self.img)
        self.img.paste(BLACK, (0, 0, self.img.width, self.img.height), Image.new("L", self.img.size, 140))
        dw = 372
        self._solid()
        self.elem("panel", -40, -20, dw + 40, H + 40, 28, self.surface, outline=self.outline, width=1.2)
        if self.ui == "bubble":  # tall gradient panel gets pale at the top: calm it so the bars stay readable
            self._fill(self._inset(self._el(-40, -20, dw + 40, H + 40, 28, "panel"), self.px(3)), self.surface, 0.6)
        self.elem("plain", 28, 64, 64, 64, 32, self.primary)
        self.elem("plain", 44, 80, 32, 32, 16, self.on_primary)
        self.bar(108, 78, 130, 15, self.on_bg)
        self.bar(108, 102, 82, 9, self.on_variant)
        self._rule(24, 148, dw - 48)

        def item(y, w, sel):
            if sel:
                self.elem("chip", 16, y, dw - 32, 50, 16, mix(self.primary, self.surface, 0.55), outline=self.primary, width=1.2, sel=True)
            self.elem("plain", 32, y + 13, 24, 24, 7, self.primary if sel else self.on_variant)
            self.bar(72, y + 19, w, 11, self.on_bg if sel else self.on_variant)

        self.bar(30, 168, 84, 8, mix(self.on_variant, self.surface, 0.3))
        for i, w in enumerate((120, 96, 140, 110)):
            item(188 + i * 56, w, i == 0)
        self._rule(24, 424, dw - 48)
        self.bar(30, 446, 100, 8, mix(self.on_variant, self.surface, 0.3))
        for i, w in enumerate((104, 132, 90, 116)):
            item(466 + i * 56, w, False)
        self.bar(32, H - 62, 120, 9, mix(self.on_variant, self.surface, 0.3))
        return self._finish()

    def screen_store(self):
        W, H = PREVIEW_SIZE
        self._page()
        self._header(title_w=110, sub_w=150)
        by = 126
        self._art(16, by, W - 32, 176, 22, [(0, mix(self.primary, BLACK, 0.45)), (0.6, mix(self.tertiary, BLACK, 0.15)), (1, mix(self.secondary, BLACK, 0.25))], horizontal=True)
        self.bar(40, by + 28, 190, 18, WHITE)
        self.bar(40, by + 58, 140, 10, mix(WHITE, self.primary, 0.3))
        self.elem("button", 40, by + 108, 120, 40, 20, self.surface)
        self.bar(62, by + 124, 76, 9, self.on_bg)
        for k, dx in enumerate((0, 74)):
            px_, py_ = W - 190 + dx, by + 26 + k * 14
            self.d.rounded_rectangle((self.px(px_), self.px(py_), self.px(px_ + 62), self.px(py_ + 122)), radius=self.px(8),
                                     fill=mix(self.bg, self.primary, 0.15 * k), outline=WHITE, width=self.px(1.5))
            self.d.rounded_rectangle((self.px(px_ + 6), self.px(py_ + 8), self.px(px_ + 56), self.px(py_ + 24)), radius=self.px(4), fill=self.primary)
        ty = by + 198
        self.bar(20, ty, 110, 12, self.on_bg)
        self.bar(W - 90, ty + 1, 60, 9, self.primary)
        tw = (W - 32 - 24) / 3
        base = [self.primary, self.secondary, self.tertiary]
        for i in range(3):
            x = 16 + i * (tw + 12)
            self.elem("cover", x, ty + 26, tw, 132, 16, self.surface, outline=self.primary if i == 0 else self.outline, width=1.5, sel=i == 0)
            for j, col in enumerate((base[i], self.accents[(i + 1) % 3], self.variant)):
                self.d.rounded_rectangle((self.px(x + 14 + j * 30), self.px(ty + 26 + 14), self.px(x + 14 + j * 30 + 24), self.px(ty + 26 + 38)), radius=self.px(5), fill=col)
            self.d.rounded_rectangle((self.px(x + 14), self.px(ty + 26 + 48), self.px(x + tw - 14), self.px(ty + 26 + 60)), radius=self.px(6), fill=base[i])
            self.d.rounded_rectangle((self.px(x + 14), self.px(ty + 26 + 68), self.px(x + tw - 40), self.px(ty + 26 + 76)), radius=self.px(4), fill=mix(self.on_bg, self.surface, 0.5))
            self.bar(x + 14, ty + 26 + 132 - 24, tw * 0.55, 8, self.on_bg)
        fy = ty + 178
        x = 16
        for i in range(3):
            on = i == 0
            self.elem("chip", x, fy, 70, 34, 10, mix(self.primary, self.surface, 0.55) if on else self.surface,
                      outline=self.primary if on else self.outline, width=1.2, sel=on)
            self.bar(x + 16, fy + 13, 38, 8, self.on_bg if on else self.on_variant)
            x += 80
        gy = fy + 50
        cw = (W - 32 - 12) / 2
        chh = 124
        pal = [self.primary, self.secondary, self.tertiary, self.accents[0]]
        for i in range(4):
            x = 16 + (i % 2) * (cw + 12)
            y = gy + (i // 2) * (chh + 12)
            self.elem("panel", x, y, cw, chh, 18, self.surface, outline=self.outline, width=1.2)
            self.elem("plain", x + 12, y + 12, 56, 56, 12, mix(pal[i], self.surface, 0.35))
            self.d.ellipse((self.px(x + 28), self.px(y + 28), self.px(x + 52), self.px(y + 52)), fill=mix(pal[i], WHITE, 0.4))
            self.bar(x + 80, y + 18, cw - 112, 11, self.on_bg)
            self.bar(x + 80, y + 38, cw - 140, 8, self.on_variant)
            self.bar(x + 14, y + 90, cw - 100, 9, self.on_variant)
            if i in (1, 2):
                self.elem("chip", x + cw - 54, y + chh - 36, 42, 22, 8, mix(self.accents[0], self.surface, 0.2), outline=self.accents[0], width=1, sel=True)
                self.bar(x + cw - 46, y + chh - 28, 26, 6, self.on_bg)
        self._nav(2)
        return self._finish()

    def render(self):
        W, H = PREVIEW_SIZE
        self._page()
        self._header()

        # Hero card
        hero_y = 130
        self.elem("panel", 16, hero_y, W - 32, 190, 26, mix(self.primary, self.surface, 0.78), outline=self.primary, width=1.5)
        self.elem("plain", 32, hero_y + 18, 130, 156, 16, mix(self.secondary, self.surface, 0.35))
        self.bar(180, hero_y + 28, 220, 16, self.on_bg)
        self.bar(180, hero_y + 54, 170, 10, self.on_variant)
        self.bar(180, hero_y + 72, 190, 10, self.on_variant)
        self.elem("button", 180, hero_y + 122, 150, 42, 21, self.primary)
        self.bar(208, hero_y + 138, 94, 10, self.on_primary)
        if self.neon:
            r = self.px(12)
            x1, y1 = self.px(W - 16), self.px(hero_y + 190)
            self.d.polygon([(x1, y1), (x1 - r, y1), (x1, y1 - r)], fill=self.accents[0])

        # Chips
        chip_y = hero_y + 210
        x = 16
        for i, width in enumerate((92, 118, 84, 104)):
            selected = i == 0
            self.elem("chip", x, chip_y, width, 34, 10, mix(self.primary, self.surface, 0.55) if selected else self.surface,
                      outline=self.primary if selected else self.outline, width=1.2, sel=selected)
            self.bar(x + 16, chip_y + 13, width - 32, 8, self.on_bg if selected else self.on_variant)
            x += width + 10

        # Game cover grid (3 columns)
        grid_y = chip_y + 54
        cols, gap = 3, 12
        cw = (W - 32 - gap * (cols - 1)) / cols
        ch = cw * 4 / 3
        palette = [self.primary, self.secondary, self.tertiary, self.accents[0], self.accents[1], self.accents[2]]
        for i in range(6):
            cx = 16 + (i % cols) * (cw + gap)
            cy = grid_y + (i // cols) * (ch + 42)
            base = palette[i % len(palette)]
            self.elem("cover", cx, cy, cw, ch, 14, mix(base, self.variant, 0.45), outline=self.outline, width=1)
            self.bar(cx + 10, cy + ch - 26, cw * 0.55, 8, mix(self.on_bg, base, 0.1))
            self.bar(cx + 2, cy + ch + 10, cw * 0.8, 10, self.on_bg)
            self.bar(cx + 2, cy + ch + 26, cw * 0.5, 7, self.on_variant)

        self._nav()
        return self._finish()


def find_background(theme_dir):
    return next((theme_dir / n for n in ("background.webp", "background.jpg", "background.png") if (theme_dir / n).is_file()), None)


def encode_webp(image):
    out = io.BytesIO()
    image.save(out, "WEBP", quality=86, method=6)
    return out.getvalue()


def render_preview(theme, theme_dir):
    return encode_webp(Painter(theme, find_background(theme_dir)).render())


# Gallery screens shown on the theme detail page: output file -> Painter method.
SCREENS = {
    "screen-hub.webp": "screen_hub",
    "screen-settings.webp": "screen_settings",
    "screen-game.webp": "screen_game",
    "screen-ingame.webp": "screen_ingame",
    "screen-drawer.webp": "screen_drawer",
    "screen-store.webp": "screen_store",
}
MAX_SCREEN_BYTES = 512 * 1024


def render_screens(theme, theme_dir):
    background = find_background(theme_dir)
    result = {}
    for name, method in SCREENS.items():
        data = encode_webp(getattr(Painter(theme, background), method)())
        if len(data) > MAX_SCREEN_BYTES:
            raise SystemExit(f"{theme['id']}: {name} is {len(data)} bytes, over the 512 KB limit")
        result[name] = data
    return result


def main():
    only = set(sys.argv[1:])
    built = 0
    for theme_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
        theme_id = theme_dir.name
        if only and theme_id not in only:
            continue
        theme = json.loads((theme_dir / "theme.json").read_text(encoding="utf-8"))
        if theme.get("id") != theme_id:
            raise SystemExit(f"{theme_dir}: theme.json id must equal the directory name")
        out_dir = DIST / theme_id
        stamp_path = out_dir / "build.json"
        digest = source_digest(theme_dir)
        if stamp_path.is_file() and json.loads(stamp_path.read_text())["sourceSha256"] == digest:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        package = build_package(theme_dir)
        if len(package) > 4 * 1024 * 1024:
            raise SystemExit(f"{theme_id}: package exceeds 4 MB")
        (out_dir / "theme.ecxtheme").write_bytes(package)
        (out_dir / "preview.webp").write_bytes(render_preview(theme, theme_dir))
        for name, data in render_screens(theme, theme_dir).items():
            (out_dir / name).write_bytes(data)
        stamp_path.write_text(json.dumps({"sourceSha256": digest}, indent=2) + "\n", encoding="utf-8", newline="\n")
        built += 1
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
