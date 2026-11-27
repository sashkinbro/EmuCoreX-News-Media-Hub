#!/usr/bin/env python3
"""Builds the EmuCore Store on-screen controller skins (kind "controls").

Everything is procedural art (Python + Pillow only, no downloads, no fonts, no third-party artwork).
Symbols and labels are original vector strokes; face symbols are generic geometric shapes.

Output (the listing.json files and categories.json are generated from the data in this script):
  store/controls/categories.json
  store/controls/src/<id>/listing.json
  store/controls/dist/<id>/item.ecxcontrols   deterministic ZIP: controls.json + 35 PNG images (PlayStation symbols)
  store/controls/dist/<id>/neutral.ecxcontrols deterministic ZIP: label-free blank_<part> art + d-pad + stick and
                                              skin.labelColors, for Nintendo/Sega apps that draw their own labels
  store/controls/dist/<id>/cover.webp         640x360 card image
  store/controls/dist/<id>/ingame.webp        1280x720 gallery mock of the full on-screen layout
  store/controls/dist/<id>/build.json         {"sourceSha256": hash(listing + this script)}

Usage: python scripts/build-store-controls.py [skin-id ...] [--force] [--add-neutral] [--sheet contact.png]
  --add-neutral  write neutral.ecxcontrols for already built skins (no full re-render)
Needs Pillow only. Deterministic and idempotent (unchanged skins are skipped).
"""
import hashlib
import io
import json
import math
import random
import sys
import zipfile
import zlib
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent.parent
CTRL = ROOT / "store" / "controls"
SRC = CTRL / "src"
DIST = CTRL / "dist"
ZIP_DATE = (2020, 1, 1, 0, 0, 0)
S = 3  # supersampling for the vector skins
PACKAGE_VERSION = 3
INPUT_TOGGLE = "left_input_toggle"

FACE = ["cross", "circle", "square", "triangle"]
DIRS = ["up", "down", "left", "right"]
SHOULDERS = ["l1", "l2", "r1", "r2"]
WIDE_PARTS = SHOULDERS + ["start", "select"]
PARTS = FACE + ["dpad_" + d for d in DIRS] + SHOULDERS + ["l3", "r3", "start", "select", "analog_base", "analog_stick"]
# Parts that also ship as label-free "blank_<part>" art. Apps for other consoles (Nintendo, Sega) draw their own
# button labels (A/B/X/Y, L/ZL, 1/2, +/-...) on these, so one skin fits every app.
BLANK_PARTS = FACE + SHOULDERS + ["l3", "r3", "start", "select"]
BLANK = False  # True while rendering blank art: every symbol/label painter draws nothing
ROLE = {"l1": "shoulder", "r1": "shoulder", "l2": "shoulder2", "r2": "shoulder2", "l3": "l3", "r3": "l3",
        "start": "small", "select": "small", "analog_base": "base", "analog_stick": "stick"}
for _d in DIRS:
    ROLE["dpad_" + _d] = "dpad"
for _f in FACE:
    ROLE[_f] = _f

# ----------------------------------------------------------------------------------------------------------------
# colour + image toolkit
# ----------------------------------------------------------------------------------------------------------------


def mixc(a, b, t):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def lighten(c, t):
    return mixc(c[:3], (255, 255, 255), t)


def darkc(c, t):
    return mixc(c[:3], (0, 0, 0), t)


def hexc(c):
    return "#%02X%02X%02X" % tuple(c[:3])


def blank(c):
    return Image.new("RGBA", (c.w, c.h), (0, 0, 0, 0))


def over(a, b):
    return Image.alpha_composite(a, b)


def paint(mask, color, alpha=1.0):
    layer = Image.new("RGBA", mask.size, tuple(color[:3]) + (255,))
    a = mask if alpha >= 0.999 else mask.point(lambda v: int(v * alpha))
    if len(color) == 4 and color[3] < 255:
        a = a.point(lambda v: v * color[3] // 255)
    layer.putalpha(a)
    return layer


def clip(fill, mask):
    r = fill.copy()
    r.putalpha(ImageChops.multiply(fill.getchannel("A"), mask))
    return r


def tone(img, mul=1.0, add=0):
    r, g, b, a = img.split()
    f = lambda v: max(0, min(255, int(v * mul + add)))
    return Image.merge("RGBA", (r.point(f), g.point(f), b.point(f), a))


def blur(m, r):
    return m.filter(ImageFilter.GaussianBlur(max(0.01, r * S)))


def shift(m, dx, dy):
    return ImageChops.offset(m, int(round(dx * S)), int(round(dy * S)))


def steep(m, k=10):
    return m.point(lambda v: max(0, min(255, int((v - 128) * k + 128))))


def soften(m, r):
    """Rounds corners (blur + threshold)."""
    return steep(blur(m, r), 12)


def grow(m, r):
    if r <= 0:
        return m
    return blur(m, r / 1.5).point(lambda v: max(0, min(255, int((v - 17) * 12))))


def shrink(m, r):
    return ImageChops.invert(grow(ImageChops.invert(m), r))


def ring_of(m, r):
    return ImageChops.subtract(m, shrink(m, r))


def ramp(g, stops):
    stops = [(p, tuple(col) + ((255,) if len(col) == 3 else ())) for p, col in stops]
    luts = [[0] * 256 for _ in range(4)]
    for i in range(256):
        t = i / 255.0
        for k in range(len(stops) - 1):
            if t <= stops[k + 1][0] or k == len(stops) - 2:
                p0, c0 = stops[k]
                p1, c1 = stops[k + 1]
                u = (t - p0) / (p1 - p0) if p1 > p0 else 0
                u = min(1.0, max(0.0, u))
                for j in range(4):
                    luts[j][i] = int(round(c0[j] + (c1[j] - c0[j]) * u))
                break
    return Image.merge("RGBA", [g.point(luts[j]) for j in range(4)])


def lin_l(w, h, angle=0):
    n = int(math.hypot(w, h)) + 4
    g = Image.linear_gradient("L").resize((n, n), Image.BILINEAR)
    if angle:
        g = g.rotate(angle, resample=Image.BICUBIC)
    return g.crop(((n - w) // 2, (n - h) // 2, (n - w) // 2 + w, (n - h) // 2 + h))


def grad(w, h, stops, angle=0):
    return ramp(lin_l(w, h, angle), stops)


def rad_l(w, h, cx, cy, r):
    g = Image.radial_gradient("L").resize((max(2, int(2 * r)),) * 2, Image.BILINEAR)
    base = Image.new("L", (w, h), 255)
    base.paste(g, (int(cx - r), int(cy - r)))
    return base


def rgrad(w, h, cx, cy, r, stops):
    return ramp(rad_l(w, h, cx, cy, r), stops)


def rng_noise(w, h, seed):
    return Image.frombytes("L", (w, h), random.Random(seed).randbytes(w * h))


def smooth_noise(w, h, cell, seed):
    return rng_noise(max(2, w // cell + 2), max(2, h // cell + 2), seed).resize((w, h), Image.BICUBIC)


def streak_noise(w, h, xcell, seed):
    return rng_noise(max(2, w // xcell), h, seed).resize((w, h), Image.BICUBIC)


def modulate(fill, tex, amt):
    rgb = fill.convert("RGB")
    mixed = ImageChops.overlay(rgb, tex.convert("RGB"))
    out = Image.blend(rgb, mixed, amt).convert("RGBA")
    out.putalpha(fill.getchannel("A"))
    return out


def inner_shadow(mask, dx, dy, br, color, alpha):
    inv = ImageChops.invert(mask)
    cov = ImageChops.multiply(blur(shift(inv, dx, dy), br), mask)
    return paint(cov, color, alpha)


def bevel(mask, depth, hi_a, lo_a, hi=(255, 255, 255), lo=(0, 0, 0)):
    out = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    hc = blur(ImageChops.subtract(mask, shift(mask, depth, depth)), 0.7)
    lc = blur(ImageChops.subtract(mask, shift(mask, -depth, -depth)), 0.7)
    out = over(out, paint(ImageChops.multiply(hc, mask), hi, hi_a))
    out = over(out, paint(ImageChops.multiply(lc, mask), lo, lo_a))
    return out


# ----------------------------------------------------------------------------------------------------------------
# strokes, symbols, labels
# ----------------------------------------------------------------------------------------------------------------


def stroke(d, pts, t):
    d.line(pts, fill=255, width=int(t), joint="curve")
    r = t / 2.0
    for p in (pts[0], pts[-1]):
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255)


def arc_pts(cx, cy, rx, ry, a0, a1, n=18):
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


GLYPHS = {
    "L": [[(0.06, 0), (0.06, 1), (0.58, 1)]],
    "R": [[(0.06, 1), (0.06, 0), (0.32, 0)] + arc_pts(0.32, 0.26, 0.26, 0.26, -90, 90, 10) + [(0.06, 0.52)],
          [(0.30, 0.52), (0.6, 1)]],
    "1": [[(0.08, 0.24), (0.36, 0.0), (0.36, 1.0)]],
    "2": [arc_pts(0.31, 0.29, 0.26, 0.26, 190, 390, 16) + [(0.06, 1.0), (0.6, 1.0)]],
    "3": [arc_pts(0.30, 0.25, 0.25, 0.25, 210, 450, 16) + arc_pts(0.30, 0.75, 0.26, 0.25, 270, 510, 16)[1:]],
}


def label_mask(text, size, cx, cy, height, thick):
    w, h = size
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    gw, gap = 0.64, 0.2
    total = len(text) * gw + (len(text) - 1) * gap
    x0 = cx - total * height / 2.0
    for i, ch in enumerate(text):
        ox = x0 + i * (gw + gap) * height
        for path in GLYPHS[ch]:
            stroke(d, [(ox + px * height, cy - height / 2.0 + py * height) for px, py in path], thick)
    return m


def tri_pts(cx, cy, R):
    return [(cx, cy - R * 1.1), (cx - R * 1.06, cy + R * 0.8), (cx + R * 1.06, cy + R * 0.8)]


def symbol_mask(sym, size, cx, cy, R, t, filled):
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    if sym == "cross":
        k = R * 0.9
        stroke(d, [(cx - k, cy - k), (cx + k, cy + k)], t)
        stroke(d, [(cx - k, cy + k), (cx + k, cy - k)], t)
    elif sym == "circle":
        d.ellipse([cx - R, cy - R, cx + R, cy + R], fill=255)
        if not filled:
            d.ellipse([cx - R + t, cy - R + t, cx + R - t, cy + R - t], fill=0)
    elif sym == "square":
        k = R * 0.9
        d.rounded_rectangle([cx - k, cy - k, cx + k, cy + k], radius=t * 0.45, fill=255)
        if not filled:
            d.rounded_rectangle([cx - k + t, cy - k + t, cx + k - t, cy + k - t], radius=t * 0.12, fill=0)
    elif sym == "triangle":
        cy2 = cy - R * 0.12
        pts = tri_pts(cx, cy2, R)
        d.polygon(pts, fill=255)
        if not filled:
            gx = sum(p[0] for p in pts) / 3.0
            gy = sum(p[1] for p in pts) / 3.0
            k = max(0.05, (0.62 * R - t) / (0.62 * R))
            d.polygon([(gx + (p[0] - gx) * k, gy + (p[1] - gy) * k) for p in pts], fill=0)
        m = soften(m, t * 0.12 / S)
    return m


# ----------------------------------------------------------------------------------------------------------------
# per-part context (geometry in 1x tile units, rendered at S x)
# ----------------------------------------------------------------------------------------------------------------

ROUND_KINDS = ("face", "l3", "stick")


class Ctx:
    def __init__(self, skin, part, pressed):
        self.skin, self.part, self.p = skin, part, pressed
        self.role = ROLE.get(part, part)
        if part in FACE:
            self.kind = "face"
        elif part.startswith("dpad_"):
            self.kind = "dpad"
            self.dir = part[5:]
        elif part in ("l1", "r1"):
            self.kind = "sh1"
        elif part in ("l2", "r2"):
            self.kind = "sh2"
        elif part in ("l3", "r3"):
            self.kind = "l3"
        elif part in ("start", "select"):
            self.kind = "small"
        elif part == "analog_base":
            self.kind = "base"
        else:
            self.kind = "stick"
        self.tw, self.th = (256, 192) if self.kind in ("sh1", "sh2", "small") else (256, 256)
        self.w, self.h = self.tw * S, self.th * S
        self.cx, self.cy = self.w / 2.0, self.h / 2.0
        self._g = {}

    @property
    def round(self):
        return self.kind in ROUND_KINDS

    def new(self):
        return Image.new("L", (self.w, self.h), 0)

    def rect_box(self):
        # Match the roughly 2:1 and 5:3 viewports of the built-in wide controls.
        return {"sh1": (14, 39, 242, 153), "sh2": (14, 28, 242, 164), "small": (30, 47, 226, 145)}[self.kind]

    def geom(self, inset=0):
        if inset in self._g:
            return self._g[inset]
        i = inset * S
        m = self.new()
        d = ImageDraw.Draw(m)
        k = self.kind
        if k in ("face", "l3", "stick", "base"):
            r = {"face": 100, "l3": 100, "stick": 96, "base": 120}[k] * S - i
            shape = self.skin.face_shape if k in ("face", "l3") else "circle"
            if r <= 0:
                pass
            elif shape == "squircle":
                q = r * 0.93
                d.rounded_rectangle([self.cx - q, self.cy - q, self.cx + q, self.cy + q], radius=max(1, r * 0.42), fill=255)
            elif shape == "octagon":
                rr = r * 1.04
                d.polygon([(self.cx + rr * math.cos(math.radians(22.5 + 45 * j)),
                            self.cy + rr * math.sin(math.radians(22.5 + 45 * j))) for j in range(8)], fill=255)
            else:
                d.ellipse([self.cx - r, self.cy - r, self.cx + r, self.cy + r], fill=255)
        elif k == "dpad":
            d.polygon([(p[0] * S, p[1] * S) for p in ((18, 18), (238, 18), (238, 128), (128, 238), (18, 128))], fill=255)
            m = soften(m, 17)
            if inset:
                m = shrink(m, inset)
            m = self.rotate(m)
        else:
            x0, y0, x1, y1 = [v * S for v in self.rect_box()]
            r = (y1 - y0) / 2.0 - i if k != "sh2" else 56 * S - i
            if x1 - i > x0 + i and y1 - i > y0 + i:
                d.rounded_rectangle([x0 + i, y0 + i, x1 - i, y1 - i], radius=max(1, r), fill=255)
        self._g[inset] = m
        return m

    def rotate(self, m):
        op = {"up": None, "right": Image.Transpose.ROTATE_270, "down": Image.Transpose.ROTATE_180,
              "left": Image.Transpose.ROTATE_90}[self.dir]
        return m.transpose(op) if op is not None else m

    def mark_mask(self, thick=1.0, scale=1.0, filled=False):
        if BLANK and self.kind != "dpad":
            return self.new()
        sk = self.skin
        th = thick * sk.sym_thick
        sc = scale * sk.sym_scale
        k = self.kind
        if k == "face":
            return symbol_mask(self.part, (self.w, self.h), self.cx, self.cy, 46 * S * sc, 15 * S * th * (1.0 if not sk.sym_filled else 1), sk.sym_filled or filled)
        if k == "dpad":
            m = self.new()
            ImageDraw.Draw(m).polygon([(128 * S, 56 * S), (128 * S - 30 * S * sc, 108 * S), (128 * S + 30 * S * sc, 108 * S)], fill=255)
            return self.rotate(soften(m, 2.6))
        if k in ("sh1", "sh2"):
            x0, y0, x1, y1 = self.rect_box()
            hh = (y1 - y0) * (0.44 if k == "sh1" else 0.36) * sc
            return label_mask(self.part.upper(), (self.w, self.h), self.cx, (y0 + y1) / 2 * S, hh * S, 9.5 * S * th * (hh / 40.0) ** 0.5)
        if k == "l3":
            return label_mask(self.part.upper(), (self.w, self.h), self.cx, self.cy, 56 * S * sc, 11 * S * th)
        if k == "small":
            m = self.new()
            d = ImageDraw.Draw(m)
            s = 17 * S * sc
            if self.part == "start":
                d.polygon([(self.cx - s * 0.55, self.cy - s * 0.85), (self.cx + s * 0.85, self.cy), (self.cx - s * 0.55, self.cy + s * 0.85)], fill=255)
                return soften(m, 1.3)
            tt = 6.2 * S * th
            for dy in (-s * 0.75, 0, s * 0.75):
                stroke(d, [(self.cx - s * 0.95, self.cy + dy), (self.cx + s * 0.95, self.cy + dy)], tt)
            return m
        return None


def finish(layer, c):
    return layer.resize((c.tw, c.th), Image.LANCZOS)


# ----------------------------------------------------------------------------------------------------------------
# generic painters
# ----------------------------------------------------------------------------------------------------------------


def lit(c, hi, lo, spread=1.0):
    if c.round:
        return rgrad(c.w, c.h, c.w * 0.4, c.h * 0.3, c.w * 0.78 * spread, [(0, hi), (1, lo)])
    return grad(c.w, c.h, [(0.15, hi), (1, lo)])


def std(c, fill, shadow=(0, 6, 8, (0, 0, 0), 0.38), bev=(0.5, 0.35, 3.5), rim=None, gloss=0.0, glow=None, inner=0.55,
        grooves=True, dim=0.78):
    p, g = c.p, c.geom(0)
    out = blank(c)
    if glow:
        col, r, a = glow
        out = over(out, paint(blur(g, r), col, min(1.0, a * (1.6 if p else 1.0))))
    if shadow:
        dx, dy, br, col, a = shadow
        if p:
            dx, dy, a = dx * 0.35, dy * 0.35, a * 0.6
        out = over(out, paint(blur(shift(g, dx, dy), br), col, a))
    if p:
        fill = tone(fill, dim)
    out = over(out, clip(fill, g))
    if grooves and c.kind == "stick":
        for ins, a in ((16, 0.28), (34, 0.24), (52, 0.2)):
            rg = ImageChops.subtract(c.geom(ins), c.geom(ins + 4))
            out = over(out, paint(rg, (0, 0, 0), a))
            out = over(out, paint(shift(rg, 0, 1.6), (255, 255, 255), a * 0.7))
    if bev:
        hi, lo, d = bev
        if p:
            hi, lo = lo * 0.3, hi * 0.3
        out = over(out, bevel(g, d, hi, lo))
    if gloss:
        gl = ImageChops.multiply(c.geom(c.geom_gloss_inset()), glossmask(c))
        out = over(out, paint(gl, (255, 255, 255), gloss * (0.35 if p else 1.0)))
    if p and inner:
        out = over(out, inner_shadow(g, 0, 5, 6, (0, 0, 0), inner))
    if rim:
        col, wd = rim
        out = over(out, paint(ImageChops.subtract(g, c.geom(wd)), col))
    return out


def _geom_gloss_inset(self):
    return 5 if self.kind != "stick" else 8


Ctx.geom_gloss_inset = _geom_gloss_inset


def glossmask(c):
    """Top lens with a vertical fade: used for glossy candy / jelly highlights."""
    if c.kind == "dpad":
        return ImageChops.multiply(c.geom(0), lin_l(c.w, c.h).point(lambda v: max(0, 255 - int(v * 1.9))))
    m = c.new()
    d = ImageDraw.Draw(m)
    if c.round:
        r = c.w * 0.37
        d.ellipse([c.cx - r * 1.0, c.cy - r * 1.18, c.cx + r * 1.0, c.cy + r * 0.1], fill=255)
    else:
        x0, y0, x1, y1 = c.rect_box() if c.kind in ("sh1", "sh2", "small") else (60, 52, 196, 130)
        d.rounded_rectangle([x0 * S, y0 * S, x1 * S, (y0 + (y1 - y0) * 0.52) * S], radius=(y1 - y0) * 0.3 * S, fill=255)
    fade = lin_l(c.w, c.h).point(lambda v: max(0, 255 - int(v * 1.35)))
    if c.kind in ("sh1", "sh2", "small"):
        fade = lin_l(c.w, c.h).point(lambda v: max(0, 255 - int(v * 1.1)))
    return ImageChops.multiply(blur(m, 1.0), fade)


def mark_paint(c, m, col, mode="flat", outline=None, glow=None):
    out = blank(c)
    p = c.p
    if mode == "flat":
        out = paint(m, col)
    elif mode == "shadow":
        out = over(out, paint(blur(shift(m, 0, 2.5), 2.0), (0, 0, 0), 0.4))
        out = over(out, paint(m, col))
    elif mode == "engrave":
        out = over(out, paint(ImageChops.subtract(shift(m, 0, 1.8), m), (255, 255, 255), 0.5))
        out = over(out, paint(m, col))
        out = over(out, inner_shadow(m, 0, 2.4, 1.6, (0, 0, 0), 0.65))
    elif mode == "emboss":
        out = over(out, paint(blur(shift(m, 0, 2.6), 1.8), (0, 0, 0), 0.5))
        out = over(out, paint(m, col))
        out = over(out, paint(ImageChops.multiply(ImageChops.subtract(m, shift(m, 1.5, 2.2)), m), (255, 255, 255), 0.55))
    elif mode == "glow":
        for r, a in ((11, 0.5), (5, 0.8), (2.2, 0.9)):
            out = over(out, paint(blur(m, r), col, min(1.0, a * (1.35 if p else 1.0))))
        out = over(out, paint(m, lighten(col, 0.6)))
    if outline:
        ocol, ow = outline
        out = over(paint(grow(m, ow), ocol), out)
    return out


# ----------------------------------------------------------------------------------------------------------------
# skin base class
# ----------------------------------------------------------------------------------------------------------------

CLASSIC = {"cross": (80, 140, 240), "circle": (240, 80, 90), "square": (240, 120, 200), "triangle": (60, 200, 150)}


class Skin:
    id = name = desc = ""
    cats = []
    pro = False
    featured = False
    press = "grow"
    opacity = 0.85
    swatch = []
    tags = []
    backdrop = ((18, 22, 34), (34, 40, 58))
    scene = 0
    face_shape = "circle"
    sym_scale = 1.0
    sym_thick = 1.0
    sym_filled = False
    mark_mode = "flat"
    mark_outline = None
    sym_colors = CLASSIC
    mark_col = (90, 100, 120)
    pixel = False
    P = {}  # role -> (hi, lo)

    def __init__(self):
        self._cache = {}

    def cache(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def pal(self, c, role=None):
        role = role or c.role
        if role in self.P:
            return self.P[role]
        if role == "shoulder2" and "shoulder" in self.P:
            return self.P["shoulder"]
        return self.P["main"]

    def render(self, part, pressed):
        c = Ctx(self, part, pressed)
        layer = self.base(c) if c.kind == "base" else self.body(c)
        mk = self.mark(c)
        if mk is not None:
            layer = over(layer, mk)
        return finish(layer, c)

    def body(self, c):
        hi, lo = self.pal(c)
        return std(c, lit(c, hi, lo))

    def base(self, c):
        hi, lo = self.pal(c)
        g, well = c.geom(0), c.geom(30)
        out = blank(c)
        out = over(out, paint(blur(shift(g, 0, 4), 6), (0, 0, 0), 0.32))
        out = over(out, clip(grad(c.w, c.h, [(0, hi), (1, lo)], 0), g))
        out = over(out, bevel(g, 3, 0.45, 0.4))
        out = over(out, clip(grad(c.w, c.h, [(0, darkc(lo, 0.65)), (1, darkc(hi, 0.35))]), well))
        out = over(out, inner_shadow(well, 0, 7, 8, (0, 0, 0), 0.6))
        out = over(out, paint(ImageChops.subtract(shift(ring_of(well, 2), 0, 1.5), ring_of(well, 2)), (255, 255, 255), 0.25))
        return out

    def sym_color(self, c):
        if c.part in self.sym_colors:
            return self.sym_colors[c.part]
        return self.mark_col

    def mark(self, c):
        m = c.mark_mask()
        if m is None:
            return None
        return mark_paint(c, m, self.sym_color(c), self.mark_mode, self.mark_outline)


def ensure_dark(col):
    return col


SKINS = []


def register(cls):
    SKINS.append(cls())
    return cls


# ----------------------------------------------------------------------------------------------------------------
# skins: clean
# ----------------------------------------------------------------------------------------------------------------


@register
class CleanWhite(Skin):
    id, name = "clean-white", "Clean White"
    desc = "Flat white buttons with soft shadows and calm, classic symbol colors."
    cats, pro, featured, press = ["clean"], False, True, "grow"
    swatch = ["#F4F6FA", "#4C81DE", "#E95460", "#34B282"]
    backdrop = ((38, 48, 70), (62, 76, 104))
    P = {"main": ((255, 255, 255), (226, 230, 238)), "base": ((236, 240, 247), (205, 212, 224)),
         "stick": ((255, 255, 255), (214, 220, 232))}
    sym_colors = {"cross": (76, 129, 222), "circle": (233, 84, 96), "square": (222, 110, 170), "triangle": (52, 178, 130)}
    mark_col = (98, 108, 126)

    def body(self, c):
        hi, lo = self.pal(c)
        return std(c, lit(c, hi, lo), shadow=(0, 5, 8, (30, 40, 70), 0.34), bev=(0.7, 0.1, 2.5), rim=((196, 203, 216), 2.2))


@register
class DarkMinimal(Skin):
    id, name = "dark-minimal", "Dark Minimal"
    desc = "Smoked translucent buttons with a hairline outline and thin white symbols."
    cats, pro, featured, press = ["clean"], False, False, "shrink"
    swatch = ["#14161C", "#E6EAF2", "#8B93A7", "#3A4052"]
    backdrop = ((120, 132, 150), (70, 80, 100))
    sym_thick = 0.62
    sym_colors = {}
    mark_col = (236, 240, 248)
    P = {"main": ((26, 28, 36, 175), (14, 15, 20, 190))}
    scene = 1

    def body(self, c):
        g = c.geom(0)
        fill = lit(c, (40, 43, 54, 170), (14, 15, 20, 190))
        if c.p:
            fill = lit(c, (235, 240, 250, 150), (200, 208, 224, 150))
        out = std(c, fill, shadow=None, bev=None, inner=0, dim=1.0, grooves=False)
        out = over(out, paint(ImageChops.subtract(g, c.geom(3.4)), (232, 236, 246), 0.9))
        if c.kind == "stick":
            out = over(out, paint(ImageChops.subtract(c.geom(30), c.geom(33)), (232, 236, 246), 0.35))
        return out

    def base(self, c):
        g = c.geom(0)
        out = paint(g, (10, 11, 16), 0.45)
        out = over(out, paint(ImageChops.subtract(g, c.geom(3)), (232, 236, 246), 0.7))
        out = over(out, paint(ring_of(c.geom(30), 2.2), (232, 236, 246), 0.4))
        return out

    def mark(self, c):
        m = c.mark_mask()
        if m is None:
            return None
        col = (20, 22, 28) if c.p else self.mark_col
        return paint(m, col)


@register
class StealthMatte(Skin):
    id, name = "stealth-matte", "Stealth Matte"
    desc = "Blacked-out matte plastic with soft edges and quietly engraved symbols."
    cats, pro, featured, press = ["material", "clean"], False, False, "spring"
    swatch = ["#1B1C1F", "#2A2C31", "#55585F", "#0C0C0E"]
    backdrop = ((78, 82, 92), (46, 49, 58))
    mark_mode = "engrave"
    sym_colors = {}
    mark_col = (64, 67, 74)
    P = {"main": ((50, 52, 57), (24, 25, 28))}
    scene = 3

    def body(self, c):
        hi, lo = self.pal(c)
        tex = self.cache(("grain", c.w, c.h), lambda: rng_noise(c.w, c.h, 7))
        fill = modulate(lit(c, hi, lo), tex, 0.22)
        return std(c, fill, shadow=(0, 7, 9, (0, 0, 0), 0.5), bev=(0.2, 0.55, 3.5), rim=((10, 10, 12), 2.4))


@register
class GlassClear(Skin):
    id, name = "glass-clear", "Clear Glass"
    desc = "Translucent glass buttons with crisp specular edges that let the game show through."
    cats, pro, featured, press, opacity = ["clean"], False, True, "glow", 0.95
    swatch = ["#DCE8F5", "#FFFFFF", "#7FB4E8", "#B8CCE0"]
    backdrop = ((30, 52, 96), (70, 40, 110))
    sym_colors = {}
    mark_col = (255, 255, 255)
    scene = 2

    def body(self, c):
        g = c.geom(0)
        top, bot = ((255, 255, 255, 110), (210, 228, 250, 40)) if not c.p else ((255, 255, 255, 170), (190, 215, 245, 100))
        fill = grad(c.w, c.h, [(0.0, top), (1.0, bot)]) if not c.round else rgrad(c.w, c.h, c.w * 0.4, c.h * 0.28, c.w * 0.8, [(0, top), (1, bot)])
        out = blank(c)
        out = over(out, paint(blur(shift(g, 0, 5), 8), (10, 20, 50), 0.28 if not c.p else 0.15))
        out = over(out, clip(fill, g))
        out = over(out, inner_shadow(g, 0, -7, 7, (200, 225, 255), 0.55))
        out = over(out, bevel(g, 3, 0.85, 0.25, hi=(255, 255, 255), lo=(120, 160, 220)))
        out = over(out, paint(ImageChops.subtract(g, c.geom(2.8)), (255, 255, 255), 0.85))
        gl = ImageChops.multiply(c.geom(c.geom_gloss_inset() + 3), glossmask(c))
        out = over(out, paint(gl, (255, 255, 255), 0.30))
        if c.kind == "stick":
            out = over(out, paint(ring_of(c.geom(30), 2), (255, 255, 255), 0.55))
        return out

    def base(self, c):
        g = c.geom(0)
        out = paint(blur(shift(g, 0, 4), 6), (10, 20, 50), 0.25)
        out = over(out, clip(rgrad(c.w, c.h, c.cx, c.cy * 0.8, c.w * 0.7, [(0, (255, 255, 255, 25)), (1, (255, 255, 255, 80))]), g))
        out = over(out, paint(ImageChops.subtract(g, c.geom(2.8)), (255, 255, 255), 0.8))
        well = c.geom(28)
        out = over(out, paint(well, (20, 40, 90), 0.22))
        out = over(out, inner_shadow(well, 0, 6, 7, (10, 20, 60), 0.5))
        out = over(out, paint(ring_of(well, 2), (255, 255, 255), 0.6))
        return out

    def mark(self, c):
        m = c.mark_mask()
        if m is None:
            return None
        out = over(paint(blur(shift(m, 0, 2.2), 2), (10, 25, 70), 0.35), paint(blur(m, 5), (170, 210, 255), 0.5))
        out = over(out, paint(m, (255, 255, 255), 0.96))
        return out


# ----------------------------------------------------------------------------------------------------------------
# skins: playful
# ----------------------------------------------------------------------------------------------------------------


def dots_mask(c, spacing=30, r=5, ox=0, oy=0):
    m = c.new()
    d = ImageDraw.Draw(m)
    row = 0
    y = oy
    while y < c.th + spacing:
        x = ox + (spacing / 2 if row % 2 else 0)
        while x < c.tw + spacing:
            d.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=255)
            x += spacing
        y += spacing * 0.866
        row += 1
    return m


def star_mask(c, cx, cy, r, inner=0.28):
    m = c.new()
    pts = []
    for i in range(8):
        rr = r if i % 2 == 0 else r * inner
        a = math.radians(-90 + 45 * i)
        pts.append(((cx + rr * math.cos(a)) * S, (cy + rr * math.sin(a)) * S))
    ImageDraw.Draw(m).polygon(pts, fill=255)
    return soften(m, 0.6)


@register
class CandyGloss(Skin):
    id, name = "candy-gloss", "Candy Gloss"
    desc = "Juicy glossy candy buttons in bright colors with chunky white symbols."
    cats, pro, featured, press = ["playful"], False, True, "spring"
    swatch = ["#FF5A6E", "#4A8CFF", "#FF82D2", "#50E6A0"]
    backdrop = ((255, 214, 232), (200, 190, 250))
    scene = 0
    mark_mode = "shadow"
    sym_colors = {}
    mark_col = (255, 255, 255)
    P = {"main": ((190, 150, 255), (110, 70, 230)),
         "cross": ((110, 170, 255), (30, 90, 215)), "circle": ((255, 120, 135), (215, 30, 70)),
         "square": ((255, 150, 220), (210, 60, 175)), "triangle": ((110, 240, 175), (20, 170, 110)),
         "dpad": ((255, 200, 90), (240, 120, 20)), "stick": ((255, 140, 200), (225, 50, 140)),
         "base": ((255, 235, 250), (225, 190, 240)), "shoulder": ((190, 150, 255), (110, 70, 230))}

    def body(self, c):
        hi, lo = self.pal(c)
        return std(c, lit(c, hi, lo, 0.9), shadow=(0, 8, 7, darkc(lo, 0.55), 0.5), bev=(0.45, 0.35, 3), rim=(darkc(lo, 0.22), 2.4), gloss=0.85)


@register
class JellyPop(Skin):
    id, name = "jelly-pop", "Jelly Pop"
    desc = "Squishy translucent jelly in bright colors with glowing cores and wet highlights."
    cats, pro, featured, press, opacity = ["playful"], True, True, "spring", 0.92
    swatch = ["#4FA0FF", "#FF5C8A", "#7BEA7B", "#FFB03A"]
    backdrop = ((70, 40, 120), (30, 110, 150))
    scene = 2
    mark_col = (255, 255, 255)
    sym_colors = {}
    P = {"main": ((200, 170, 255), (130, 90, 240)),
         "cross": ((120, 190, 255), (40, 110, 235)), "circle": ((255, 140, 150), (235, 50, 90)),
         "square": ((255, 170, 235), (220, 80, 200)), "triangle": ((140, 255, 190), (30, 200, 120)),
         "dpad": ((190, 255, 120), (100, 200, 40)), "shoulder": ((255, 210, 120), (255, 140, 30)),
         "l3": ((160, 230, 255), (40, 170, 230)), "stick": ((255, 160, 200), (240, 70, 150)),
         "base": ((255, 255, 255), (170, 195, 235))}

    def body(self, c):
        hi, lo = self.pal(c)
        g = c.geom(0)
        p = c.p
        out = blank(c)
        out = over(out, paint(blur(shift(g, 0, 8 if not p else 3), 9), lo, 0.5 if not p else 0.3))
        fill = lit(c, tuple(hi) + (205,), tuple(lo) + (225,), 0.95)
        if p:
            fill = tone(fill, 0.8)
        out = over(out, clip(fill, g))
        out = over(out, inner_shadow(g, 0, -10, 9, lighten(hi, 0.55), 0.85))
        out = over(out, inner_shadow(g, 0, 7, 6, darkc(lo, 0.35), 0.45 if not p else 0.8))
        if c.kind == "stick":
            out = over(out, paint(ring_of(c.geom(26), 3), lighten(hi, 0.7), 0.5))
        gl = ImageChops.multiply(c.geom(7), glossmask(c))
        out = over(out, paint(gl, (255, 255, 255), 0.7 if not p else 0.3))
        if c.round:
            sp = c.new()
            ImageDraw.Draw(sp).ellipse([c.w * 0.27, c.h * 0.2, c.w * 0.37, c.h * 0.29], fill=255)
            out = over(out, paint(blur(sp, 0.8), (255, 255, 255), 0.95))
        out = over(out, paint(ring_of(g, 2.2), lighten(hi, 0.6), 0.55))
        return out

    def base(self, c):
        g, well = c.geom(0), c.geom(28)
        out = paint(blur(shift(g, 0, 5), 7), (60, 50, 120), 0.3)
        out = over(out, clip(rgrad(c.w, c.h, c.cx, c.cy * 0.8, c.w * 0.7, [(0, (255, 255, 255, 70)), (1, (190, 210, 255, 150))]), g))
        out = over(out, paint(ring_of(g, 2.6), (255, 255, 255), 0.7))
        out = over(out, paint(well, (90, 70, 170), 0.3))
        out = over(out, inner_shadow(well, 0, 7, 7, (40, 20, 100), 0.55))
        out = over(out, paint(ring_of(well, 2), (255, 255, 255), 0.5))
        return out

    def mark(self, c):
        m = c.mark_mask()
        if m is None:
            return None
        out = over(paint(blur(shift(m, 0, 2.6), 2.2), darkc(self.pal(c)[1], 0.5), 0.5), paint(m, (255, 255, 255), 0.97))
        return out


@register
class KawaiiPastel(Skin):
    id, name = "pad-kawaii-pastel", "Kawaii Pastel"
    desc = "Soft pastel buttons with polka dots, white piping, sparkles and outlined symbols."
    cats, pro, featured, press = ["playful"], False, False, "spring"
    swatch = ["#FFB8D0", "#B4D2FF", "#BFF0D8", "#E3C8FF"]
    backdrop = ((255, 232, 240), (222, 214, 255))
    scene = 0
    sym_colors = {}
    mark_col = (255, 255, 255)
    P = {"main": ((236, 214, 255), (206, 176, 245)),
         "cross": ((196, 222, 255), (150, 190, 245)), "circle": ((255, 200, 214), (250, 150, 180)),
         "square": ((255, 214, 240), (245, 170, 220)), "triangle": ((200, 245, 222), (140, 222, 184)),
         "dpad": ((255, 238, 190), (250, 214, 130)), "base": ((255, 244, 250), (240, 218, 240)),
         "stick": ((255, 205, 225), (248, 160, 195))}

    def body(self, c):
        hi, lo = self.pal(c)
        g = c.geom(0)
        out = std(c, lit(c, hi, lo), shadow=(0, 6, 7, darkc(lo, 0.45), 0.4), bev=(0.35, 0.15, 3), rim=((255, 255, 255), 5.5))
        dots = ImageChops.multiply(dots_mask(c, 30, 5, 10, 10), c.geom(10))
        out = over(out, paint(dots, (255, 255, 255), 0.5))
        out = over(out, paint(ring_of(c.geom(5.5), 1.4), darkc(lo, 0.06), 0.55))
        if c.kind in ("face", "l3", "stick"):
            out = over(out, paint(star_mask(c, c.tw * 0.76, c.th * 0.25, 15), (255, 255, 255), 0.95))
        elif c.kind in ("sh1", "sh2", "small"):
            out = over(out, paint(star_mask(c, 208, c.th * 0.5 - 22, 9), (255, 255, 255), 0.95))
        return out

    def base(self, c):
        hi, lo = self.pal(c)
        g, well = c.geom(0), c.geom(30)
        out = paint(blur(shift(g, 0, 4), 6), darkc(lo, 0.5), 0.3)
        out = over(out, clip(lit(c, hi, lo), g))
        out = over(out, paint(ring_of(g, 5), (255, 255, 255), 1))
        out = over(out, paint(well, darkc(lo, 0.14), 0.7))
        out = over(out, inner_shadow(well, 0, 6, 6, darkc(lo, 0.6), 0.6))
        out = over(out, paint(ring_of(well, 3), (255, 255, 255), 0.85))
        return out

    def mark(self, c):
        m = c.mark_mask()
        if m is None:
            return None
        edge = darkc(self.pal(c)[1], 0.18) if c.kind != "face" else darkc(self.pal(c)[1], 0.2)
        col = (255, 255, 255) if c.kind in ("face", "dpad") else darkc(self.pal(c)[1], 0.35)
        out = over(paint(grow(m, 3.6), edge), paint(m, col))
        return out


@register
class FrostedSakura(Skin):
    id, name = "frosted-sakura", "Frosted Sakura"
    desc = "Frosted-glass buttons in blossom pink, scattered with cherry petals and rose symbols."
    cats, pro, featured, press, opacity = ["playful", "clean"], True, True, "glow", 0.92
    swatch = ["#FFD9E6", "#F7A6C4", "#C2477A", "#FFFFFF"]
    backdrop = ((120, 70, 110), (200, 120, 150))
    scene = 0
    sym_colors = {}
    mark_col = (186, 58, 104)

    def petal(self, c, cx, cy, r, rot, alpha):
        m = c.new()
        d = ImageDraw.Draw(m)
        for i in range(5):
            a = math.radians(rot + 72 * i)
            px, py = cx + math.cos(a) * r * 0.62, cy + math.sin(a) * r * 0.62
            d.ellipse([(px - r * 0.52) * S, (py - r * 0.52) * S, (px + r * 0.52) * S, (py + r * 0.52) * S], fill=255)
        n = c.new()
        ImageDraw.Draw(n).ellipse([(cx - r * 0.2) * S, (cy - r * 0.2) * S, (cx + r * 0.2) * S, (cy + r * 0.2) * S], fill=255)
        out = over(paint(blur(m, 0.5), (255, 150, 190), alpha), paint(n, (255, 255, 255), alpha + 0.2))
        return out

    def body(self, c):
        g = c.geom(0)
        top, bot = ((255, 240, 246, 225), (250, 205, 225, 205)) if not c.p else ((255, 215, 232, 240), (240, 170, 200, 230))
        fill = lit(c, top, bot)
        grain = self.cache(("grain", c.w, c.h), lambda: rng_noise(c.w, c.h, 11))
        fill = modulate(fill, grain, 0.18)
        out = blank(c)
        out = over(out, paint(blur(shift(g, 0, 6), 8), (130, 40, 80), 0.32 if not c.p else 0.18))
        out = over(out, clip(fill, g))
        out = over(out, bevel(g, 3, 0.9, 0.18, lo=(200, 90, 140)))
        out = over(out, inner_shadow(g, 0, -6, 7, (255, 255, 255), 0.55))
        out = over(out, paint(ring_of(g, 2.4), (255, 255, 255), 0.9))
        if c.kind in ("face", "l3", "stick"):
            out = over(out, self.petal(c, c.tw * 0.28, c.th * 0.3, 20, 10, 0.55))
            out = over(out, self.petal(c, c.tw * 0.73, c.th * 0.74, 11, 40, 0.5))
        elif c.kind in ("sh1", "sh2", "small"):
            out = over(out, self.petal(c, 52, c.th * 0.5, 13, 20, 0.5))
            out = over(out, self.petal(c, 206, c.th * 0.5 + 6, 8, 55, 0.5))
        else:
            out = over(out, self.petal(c, 66, 168, 17, 15, 0.45))
        if c.kind == "stick":
            out = over(out, paint(ring_of(c.geom(30), 2.2), (255, 255, 255), 0.7))
        return out

    def base(self, c):
        g, well = c.geom(0), c.geom(28)
        out = paint(blur(shift(g, 0, 4), 6), (130, 40, 80), 0.28)
        out = over(out, clip(rgrad(c.w, c.h, c.cx, c.cy * 0.8, c.w * 0.7, [(0, (255, 235, 244, 150)), (1, (250, 190, 215, 190))]), g))
        out = over(out, paint(ring_of(g, 2.6), (255, 255, 255), 0.9))
        out = over(out, paint(well, (190, 70, 120), 0.2))
        out = over(out, inner_shadow(well, 0, 6, 7, (150, 40, 90), 0.5))
        out = over(out, paint(ring_of(well, 2), (255, 255, 255), 0.7))
        out = over(out, self.petal(c, c.tw * 0.5, c.th * 0.1 + 14, 9, 0, 0.55))
        return out

    def mark(self, c):
        m = c.mark_mask()
        if m is None:
            return None
        return over(paint(grow(m, 2.4), (255, 255, 255), 0.85), paint(m, self.mark_col))


@register
class CrystalIce(Skin):
    id, name = "crystal-ice", "Crystal Ice"
    desc = "Faceted ice-blue crystal with frosty edges, glints and deep sapphire symbols."
    cats, pro, featured, press, opacity = ["clean"], True, False, "glow", 0.95
    swatch = ["#D8F2FF", "#7CC3F0", "#2D78C8", "#FFFFFF"]
    backdrop = ((10, 28, 64), (30, 70, 120))
    scene = 2
    sym_colors = {}
    mark_col = (20, 76, 158)
    P = {"main": ((226, 246, 255), (120, 190, 240))}

    def facets(self, c, seed):
        rnd = random.Random(seed)
        im = Image.new("RGBA", (c.w, c.h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        n = 9
        ang = sorted(rnd.uniform(0, 360) for _ in range(n))
        cx, cy = c.w * rnd.uniform(0.42, 0.58), c.h * rnd.uniform(0.4, 0.56)
        R = max(c.w, c.h)
        pts = [(cx + R * math.cos(math.radians(a)), cy + R * math.sin(math.radians(a))) for a in ang]
        for i in range(n):
            a = rnd.uniform(-1, 1)
            col = (255, 255, 255, int(70 * a)) if a > 0 else (20, 80, 170, int(-80 * a))
            d.polygon([(cx, cy), pts[i], pts[(i + 1) % n]], fill=col)
        return im

    def body(self, c):
        hi, lo = self.pal(c)
        g = c.geom(0)
        fill = over(lit(c, hi, lo), self.facets(c, zlib.crc32(c.part.encode())))
        if c.p:
            fill = tone(fill, 0.85)
        out = blank(c)
        out = over(out, paint(blur(g, 7), (150, 210, 255), 0.55 if not c.p else 0.9))
        out = over(out, paint(blur(shift(g, 0, 6), 7), (0, 20, 70), 0.4))
        out = over(out, clip(fill, g))
        out = over(out, bevel(g, 3.5, 0.95, 0.4, lo=(20, 70, 150)))
        out = over(out, paint(ring_of(g, 2.4), (255, 255, 255), 0.85))
        out = over(out, paint(ImageChops.multiply(c.geom(7), glossmask(c)), (255, 255, 255), 0.35))
        gx, gy = (c.tw * 0.76, c.th * 0.24) if c.round else (c.tw * 0.8, c.th * 0.34)
        out = over(out, paint(star_mask(c, gx, gy, 17, 0.18), (255, 255, 255), 1.0))
        if c.kind == "stick":
            out = over(out, paint(ring_of(c.geom(30), 2.2), (255, 255, 255), 0.6))
        return out

    def base(self, c):
        g, well = c.geom(0), c.geom(30)
        out = paint(blur(g, 7), (150, 210, 255), 0.35)
        out = over(out, clip(over(lit(c, (226, 246, 255), (110, 180, 235)), self.facets(c, 5)), g))
        out = over(out, bevel(g, 3.5, 0.9, 0.4, lo=(20, 70, 150)))
        out = over(out, paint(ring_of(g, 2.4), (255, 255, 255), 0.8))
        out = over(out, paint(well, (10, 50, 120), 0.45))
        out = over(out, inner_shadow(well, 0, 7, 8, (0, 20, 70), 0.6))
        out = over(out, paint(ring_of(well, 2), (255, 255, 255), 0.7))
        return out

    def mark(self, c):
        m = c.mark_mask()
        if m is None:
            return None
        return over(paint(grow(m, 2.2), (255, 255, 255), 0.8), paint(m, self.mark_col))


# ----------------------------------------------------------------------------------------------------------------
# skins: neon
# ----------------------------------------------------------------------------------------------------------------


def neon_ring(out, c, col, width=4.5, inset=0, core=0.75, glow=1.0):
    g = c.geom(inset)
    ring = ring_of(g, width)
    p = c.p
    out = over(out, paint(blur(ring, 8), col, min(1.0, 0.85 * glow * (1.3 if p else 1.0))))
    out = over(out, paint(blur(ring, 2.6), col, 1.0))
    out = over(out, paint(ring_of(g, width * 0.5), lighten(col, core)))
    return out


@register
class NeonCyber(Skin):
    id, name = "neon-cyber", "Neon Cyber"
    desc = "Dark glass buttons wrapped in glowing cyan, magenta and lime neon tubes."
    cats, pro, featured, press, opacity = ["neon"], False, True, "glow", 0.92
    swatch = ["#00E1FF", "#FF3296", "#B464FF", "#50FF96"]
    backdrop = ((6, 8, 22), (20, 12, 46))
    scene = 2
    sym_colors = {}
    NC = {"main": (0, 225, 255), "cross": (0, 200, 255), "circle": (255, 50, 150), "square": (185, 100, 255),
          "triangle": (80, 255, 150), "stick": (255, 50, 150), "small": (255, 225, 60), "shoulder": (0, 225, 255),
          "shoulder2": (185, 100, 255), "l3": (255, 50, 150), "dpad": (0, 225, 255), "base": (0, 225, 255)}

    def nc(self, c):
        return self.NC.get(c.role, self.NC["main"])

    def body(self, c):
        col = self.nc(c)
        g = c.geom(0)
        fill = lit(c, (14, 16, 36, 225), (5, 6, 16, 238))
        if c.p:
            fill = over(fill, paint(g, col, 0.3))
        out = blank(c)
        out = over(out, clip(fill, g))
        out = neon_ring(out, c, col)
        if c.kind == "stick":
            out = neon_ring(out, c, col, 3.2, 30, 0.6, 0.6)
        if c.kind == "stick" and not c.p:
            out = over(out, paint(ImageChops.multiply(c.geom(8), glossmask(c)), col, 0.12))
        return out

    def base(self, c):
        col = self.nc(c)
        g, well = c.geom(0), c.geom(30)
        out = clip(Image.new("RGBA", (c.w, c.h), (6, 8, 20, 190)), g)
        out = neon_ring(out, c, col, 4, 0, 0.7, 0.8)
        out = over(out, paint(well, (0, 0, 8), 0.5))
        out = over(out, inner_shadow(well, 0, 5, 6, col, 0.35))
        out = neon_ring(out, c, col, 2.4, 30, 0.6, 0.4)
        return out

    def mark(self, c):
        m = c.mark_mask()
        if m is None:
            return None
        return mark_paint(c, m, self.nc(c), "glow")


@register
class Synthwave(Skin):
    id, name = "synthwave", "Synthwave Sunset"
    desc = "Retro-future sunset gradients with sliced sun stripes and hot cyan edges."
    cats, pro, featured, press, opacity = ["neon", "retro"], True, True, "glow", 0.92
    swatch = ["#FF50BE", "#8C32E6", "#28E6FF", "#FFE66E"]
    backdrop = ((22, 8, 56), (90, 20, 100))
    scene = 0
    sym_colors = {}
    mark_col = (255, 232, 120)

    def sunfill(self, c):
        hi, mid, lo = (255, 150, 60), (255, 60, 170), (110, 40, 210)
        fill = grad(c.w, c.h, [(0.0, hi), (0.38, mid), (1.0, lo)])
        cut = Image.new("L", (c.w, c.h), 255)
        d = ImageDraw.Draw(cut)
        for i in range(5):
            y = (0.5 + i * 0.1) * c.h
            d.rectangle([0, y, c.w, y + (i + 1.2) * 0.011 * c.h], fill=0)
        fill.putalpha(ImageChops.multiply(fill.getchannel("A"), cut))
        return fill

    def body(self, c):
        g = c.geom(0)
        fill = self.sunfill(c)
        if c.p:
            fill = tone(fill, 1.25, 10)
        out = blank(c)
        out = over(out, paint(blur(g, 8), (255, 60, 170), 0.25))
        back = clip(Image.new("RGBA", (c.w, c.h), (24, 8, 60, 235)), g)
        out = over(out, back)
        out = over(out, clip(fill, g))
        out = neon_ring(out, c, (40, 230, 255), 4.4, 0, 0.7)
        if c.kind == "stick":
            out = neon_ring(out, c, (255, 230, 120), 2.6, 32, 0.5, 0.4)
        return out

    def base(self, c):
        g, well = c.geom(0), c.geom(30)
        out = clip(Image.new("RGBA", (c.w, c.h), (28, 8, 64, 215)), g)
        out = neon_ring(out, c, (40, 230, 255), 4, 0, 0.7, 0.8)
        out = over(out, paint(well, (255, 60, 170), 0.15))
        out = over(out, inner_shadow(well, 0, 5, 6, (255, 60, 170), 0.5))
        out = neon_ring(out, c, (255, 60, 170), 2.4, 30, 0.6, 0.6)
        return out

    def mark(self, c):
        m = c.mark_mask()
        if m is None:
            return None
        return mark_paint(c, m, self.mark_col, "glow")


@register
class LavaGlow(Skin):
    id, name = "lava-glow", "Lava Glow"
    desc = "Cooled basalt buttons split by glowing lava cracks and ember-orange symbols."
    cats, pro, featured, press, opacity = ["neon", "material"], True, False, "glow", 0.95
    swatch = ["#2A1C1A", "#FF8C14", "#FFC850", "#C8321E"]
    backdrop = ((18, 8, 8), (70, 22, 10))
    scene = 3
    sym_colors = {}
    mark_col = (255, 176, 50)

    def cracks(self, c, seed):
        rnd = random.Random(seed)
        m = c.new()
        d = ImageDraw.Draw(m)
        for _ in range(6):
            a = rnd.uniform(0, 2 * math.pi)
            x, y = c.cx + math.cos(a) * c.w * 0.47, c.cy + math.sin(a) * c.h * 0.44
            pts = [(x, y)]
            ang = math.atan2(c.cy - y, c.cx - x)
            for _k in range(rnd.randint(4, 7)):
                ang += rnd.uniform(-0.8, 0.8)
                step = c.w * rnd.uniform(0.05, 0.11)
                x, y = x + math.cos(ang) * step, y + math.sin(ang) * step
                pts.append((x, y))
            d.line(pts, fill=255, width=int(rnd.uniform(2.0, 4.4) * S), joint="curve")
            if len(pts) > 3 and rnd.random() < 0.8:
                bx, by = pts[2]
                ba = ang + rnd.choice((-1, 1)) * rnd.uniform(0.7, 1.3)
                d.line([(bx, by), (bx + math.cos(ba) * c.w * 0.1, by + math.sin(ba) * c.w * 0.1),
                        (bx + math.cos(ba + 0.5) * c.w * 0.17, by + math.sin(ba + 0.5) * c.w * 0.17)], fill=255, width=int(1.8 * S), joint="curve")
        return m

    def rock(self, c):
        tex = self.cache(("rock", c.w, c.h), lambda: ImageChops.add(smooth_noise(c.w, c.h, 14, 3), smooth_noise(c.w, c.h, 40, 4), scale=2))
        return modulate(lit(c, (74, 52, 48), (26, 18, 18)), tex, 0.55)

    def body(self, c):
        g = c.geom(0)
        p = c.p
        out = blank(c)
        out = over(out, paint(blur(g, 8), (255, 90, 20), 0.28 if not p else 0.55))
        out = over(out, paint(blur(shift(g, 0, 6), 7), (0, 0, 0), 0.45))
        out = over(out, clip(self.rock(c), g))
        out = over(out, bevel(g, 3, 0.25, 0.6))
        cr = ImageChops.multiply(self.cracks(c, zlib.crc32(c.part.encode())), c.geom(5))
        out = over(out, paint(blur(cr, 5), (255, 90, 10), 0.85 if not p else 1.0))
        out = over(out, paint(blur(cr, 1.4), (255, 150, 30), 1.0))
        out = over(out, paint(cr, (255, 226, 130), 0.95 if not p else 1.0))
        out = over(out, inner_shadow(g, 0, 0, 9, (255, 110, 20), 0.5 if not p else 0.95))
        out = over(out, paint(ring_of(g, 2.4), (20, 10, 8), 0.9))
        if c.kind == "stick":
            out = over(out, paint(ring_of(c.geom(30), 2.2), (255, 120, 20), 0.6))
        return out

    def base(self, c):
        g, well = c.geom(0), c.geom(30)
        out = paint(blur(g, 8), (255, 90, 20), 0.2)
        out = over(out, clip(self.rock(c), g))
        out = over(out, bevel(g, 3, 0.25, 0.6))
        out = over(out, paint(well, (255, 100, 20), 0.2))
        out = over(out, inner_shadow(well, 0, 6, 8, (0, 0, 0), 0.85))
        out = over(out, paint(blur(ring_of(well, 3), 3), (255, 120, 20), 0.9))
        out = over(out, paint(ring_of(well, 1.6), (255, 220, 120), 0.9))
        return out

    def mark(self, c):
        m = c.mark_mask()
        if m is None:
            return None
        return mark_paint(c, m, self.mark_col, "glow")


@register
class Holographic(Skin):
    id, name = "holographic", "Holographic"
    desc = "Iridescent prismatic foil that shifts hue across every button, with split-color symbols."
    cats, pro, featured, press, opacity = ["neon", "playful"], True, False, "glow", 0.94
    swatch = ["#B4E6FF", "#FFB4F0", "#BEFFD2", "#FFF0A0"]
    backdrop = ((16, 14, 36), (44, 30, 78))
    scene = 2
    sym_colors = {}

    def foil(self, c, hue0):
        h = lin_l(c.w, c.h, 32).point(lambda v: (int(v * 1.5) + hue0) % 256)
        s = Image.new("L", (c.w, c.h), 120)
        v = Image.new("L", (c.w, c.h), 255)
        im = Image.merge("HSV", (h, s, v)).convert("RGBA")
        sheen = grad(c.w, c.h, [(0, (255, 255, 255, 0)), (0.38, (255, 255, 255, 0)), (0.5, (255, 255, 255, 150)),
                                (0.62, (255, 255, 255, 0)), (1, (255, 255, 255, 0))], 32)
        return over(im, sheen)

    def body(self, c):
        g = c.geom(0)
        hue0 = zlib.crc32(c.part.encode()) % 80
        fill = self.foil(c, hue0)
        if c.p:
            fill = tone(fill, 0.82)
        out = blank(c)
        out = over(out, paint(blur(g, 8), (190, 170, 255), 0.3))
        out = over(out, paint(blur(shift(g, 0, 6), 7), (10, 0, 50), 0.4))
        out = over(out, clip(fill, g))
        out = over(out, bevel(g, 3, 0.9, 0.3, lo=(100, 80, 170)))
        out = over(out, paint(ring_of(g, 2.6), (255, 255, 255), 0.85))
        if c.p:
            out = over(out, inner_shadow(g, 0, 5, 6, (60, 30, 120), 0.55))
        if c.kind == "stick":
            out = over(out, paint(ring_of(c.geom(30), 2.2), (255, 255, 255), 0.6))
        return out

    def base(self, c):
        g, well = c.geom(0), c.geom(30)
        out = paint(blur(shift(g, 0, 5), 7), (10, 0, 50), 0.3)
        out = over(out, clip(tone(self.foil(c, 30), 0.85), g))
        out = over(out, paint(ring_of(g, 2.6), (255, 255, 255), 0.85))
        out = over(out, paint(well, (30, 10, 80), 0.5))
        out = over(out, inner_shadow(well, 0, 6, 7, (0, 0, 40), 0.6))
        out = over(out, paint(ring_of(well, 2), (255, 255, 255), 0.6))
        return out

    def mark(self, c):
        m = c.mark_mask()
        if m is None:
            return None
        out = blank(c)
        out = over(out, paint(shift(m, -2, 0), (0, 230, 255), 0.95))
        out = over(out, paint(shift(m, 2, 0), (255, 40, 200), 0.95))
        out = over(out, paint(m, (255, 255, 255)))
        return out


# ----------------------------------------------------------------------------------------------------------------
# skins: material
# ----------------------------------------------------------------------------------------------------------------


def screw(c, cx, cy, r, base=(120, 120, 124)):
    m = c.new()
    ImageDraw.Draw(m).ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S], fill=255)
    out = blank(c)
    out = over(out, paint(shift(m, 0.8, 1.4), (0, 0, 0), 0.5))
    out = over(out, clip(rgrad(c.w, c.h, (cx - r * 0.3) * S, (cy - r * 0.4) * S, r * 1.8 * S, [(0, lighten(base, 0.6)), (1, darkc(base, 0.4))]), m))
    sl = c.new()
    stroke(ImageDraw.Draw(sl), [((cx - r * 0.6) * S, (cy + r * 0.25) * S), ((cx + r * 0.6) * S, (cy - r * 0.25) * S)], max(1, r * 0.34 * S))
    out = over(out, paint(sl, (30, 30, 34), 0.7))
    return out


def rivets(c, inset=14, r=4.2, base=(150, 150, 150), n=4):
    out = blank(c)
    if c.kind in ("face", "l3", "stick"):
        rad = (100 if c.kind != "stick" else 96) - inset
        for i in range(n):
            a = math.radians(45 + 360 / n * i)
            out = over(out, screw(c, c.tw / 2 + rad * math.cos(a), c.th / 2 + rad * math.sin(a), r, base))
    elif c.kind in ("sh1", "sh2", "small"):
        x0, y0, x1, y1 = c.rect_box()
        for x in (x0 + 24, x1 - 24):
            out = over(out, screw(c, x, (y0 + y1) / 2, r, base))
    return out


@register
class BrushedSteel(Skin):
    id, name = "brushed-steel", "Brushed Steel"
    desc = "Machined brushed-steel buttons with polished bevels, rivets and engraved symbols."
    cats, pro, featured, press, opacity = ["material"], True, True, "grow", 0.96
    swatch = ["#D2D6DE", "#8C919C", "#EEF0F4", "#4A4F5A"]
    backdrop = ((30, 34, 42), (64, 70, 82))
    scene = 1
    mark_mode = "engrave"
    sym_colors = {}
    mark_col = (58, 62, 72)

    def metal(self, c, base_ang=24):
        def make():
            m = grad(c.w, c.h, [(0, (214, 218, 226)), (0.3, (150, 156, 166)), (0.5, (240, 243, 248)), (0.72, (138, 144, 154)), (1, (196, 201, 210))], base_ang)
            st = streak_noise(c.w, c.h, max(8, c.w // 5), 21)
            fine = streak_noise(c.w, c.h, max(6, c.w // 40), 22)
            tex = ImageChops.add(st, fine, scale=2)
            return modulate(m, tex, 0.55)
        return self.cache(("metal", c.w, c.h, base_ang), make)

    def body(self, c):
        g = c.geom(0)
        fill = self.metal(c)
        out = std(c, fill, shadow=(0, 7, 7, (0, 0, 0), 0.5), bev=(0.85, 0.55, 3.8), rim=((52, 56, 66), 2.6))
        out = over(out, paint(ring_of(c.geom(5), 1.2), (255, 255, 255), 0.35))
        if c.kind in ("face", "l3", "sh1", "sh2", "small") and c.kind != "small":
            out = over(out, rivets(c))
        return out

    def base(self, c):
        g, well = c.geom(0), c.geom(30)
        out = paint(blur(shift(g, 0, 5), 6), (0, 0, 0), 0.4)
        out = over(out, clip(self.metal(c, 60), g))
        out = over(out, bevel(g, 3.5, 0.8, 0.5))
        out = over(out, paint(ring_of(g, 2.4), (52, 56, 66), 1))
        out = over(out, paint(well, (24, 26, 32), 0.85))
        out = over(out, inner_shadow(well, 0, 8, 8, (0, 0, 0), 0.9))
        out = over(out, paint(ImageChops.subtract(shift(ring_of(well, 2.4), 0, 1.6), ring_of(well, 2.4)), (255, 255, 255), 0.4))
        out = over(out, rivets(c, 10, 4.6, n=4))
        return out


def wood_tex(tw, th, seed, dark, light):
    warp = smooth_noise(tw, th, 40, seed)
    wp = warp.load()
    fine = streak_noise(tw, th, 2, seed + 1)
    fp = fine.load()
    out = Image.new("L", (tw, th))
    op = out.load()
    for y in range(th):
        for x in range(tw):
            t = x / 11.0 + wp[x, y] / 255.0 * 1.5 + 0.35 * math.sin(y / 70.0 + seed)
            v = 0.5 + 0.5 * math.sin(t * 6.2832)
            v = v ** 1.5
            op[x, y] = max(0, min(255, int(40 + 200 * v * 0.72 + fp[x, y] * 0.18)))
    return ImageOps.colorize(out, dark, light).convert("RGBA")


@register
class WoodGrain(Skin):
    id, name = "wood-grain", "Wood Grain"
    desc = "Warm carved wood with visible grain, dark piping and burned-in symbols."
    cats, pro, featured, press = ["material"], False, False, "shrink"
    swatch = ["#B97F45", "#E1A869", "#6E4222", "#3A2112"]
    backdrop = ((52, 40, 34), (92, 70, 56))
    scene = 3
    sym_colors = {}

    def tex(self, c):
        walnut = c.role in ("dpad", "shoulder", "shoulder2", "small", "base")
        seed = {"face": 3, "dpad": 8, "stick": 5}.get(c.kind, 12)
        if walnut:
            dark, light = (74, 42, 22), (140, 88, 48)
        else:
            dark, light = (164, 104, 52), (236, 178, 108)
        t = self.cache(("wood", c.tw, c.th, seed, walnut), lambda: wood_tex(c.tw, c.th, seed, dark, light))
        return t.resize((c.w, c.h), Image.BICUBIC)

    def body(self, c):
        shade = lit(c, (255, 250, 245), (150, 128, 112), 1.0)
        fill = ImageChops.multiply(self.tex(c).convert("RGB"), shade.convert("RGB")).convert("RGBA")
        return std(c, fill, shadow=(0, 6, 7, (20, 8, 0), 0.5), bev=(0.45, 0.5, 3.6), rim=((52, 28, 14), 2.8))

    def base(self, c):
        g, well = c.geom(0), c.geom(30)
        out = paint(blur(shift(g, 0, 5), 6), (20, 8, 0), 0.4)
        out = over(out, clip(self.tex(c), g))
        out = over(out, bevel(g, 3.5, 0.45, 0.5))
        out = over(out, paint(ring_of(g, 2.8), (52, 28, 14), 1))
        out = over(out, paint(well, (30, 14, 6), 0.75))
        out = over(out, inner_shadow(well, 0, 8, 8, (0, 0, 0), 0.9))
        out = over(out, paint(ImageChops.subtract(shift(ring_of(well, 2.4), 0, 1.6), ring_of(well, 2.4)), (255, 220, 170), 0.35))
        return out

    def mark(self, c):
        m = c.mark_mask()
        if m is None:
            return None
        if c.role in ("dpad", "shoulder", "shoulder2", "small", "base"):
            return over(paint(blur(shift(m, 0, 1.4), 1.2), (0, 0, 0), 0.5), paint(m, (244, 220, 176), 0.97))
        out = over(paint(blur(m, 1.6), (30, 12, 4), 0.55), paint(m, (64, 32, 14), 0.96))
        return over(paint(ImageChops.subtract(shift(m, 0, 1.4), m), (255, 220, 170), 0.35), out)


def carbon_tex(w, h, cell):
    prof = [int(36 + 116 * math.sin(math.pi * (i + 0.5) / cell)) for i in range(cell)]
    va = Image.new("L", (cell, cell))
    ha = Image.new("L", (cell, cell))
    vp, hp = va.load(), ha.load()
    for y in range(cell):
        for x in range(cell):
            vp[x, y] = prof[x]
            hp[x, y] = prof[y]
    block = Image.new("L", (cell * 2, cell * 2))
    block.paste(va, (0, 0))
    block.paste(ha, (cell, 0))
    block.paste(ha, (0, cell))
    block.paste(va, (cell, cell))
    out = Image.new("L", (w, h))
    for y in range(0, h, cell * 2):
        for x in range(0, w, cell * 2):
            out.paste(block, (x, y))
    return out


@register
class CarbonFiber(Skin):
    id, name = "carbon-fiber", "Carbon Fiber"
    desc = "Woven carbon-fiber buttons with a glossy clear coat, red piping and silver symbols."
    cats, pro, featured, press, opacity = ["material"], True, False, "grow", 0.96
    swatch = ["#15161A", "#3B3E46", "#E1202D", "#C8CCD6"]
    backdrop = ((30, 32, 38), (58, 62, 72))
    scene = 1
    mark_mode = "emboss"
    sym_colors = {}
    mark_col = (214, 218, 228)

    def weave(self, c):
        def make():
            tex = carbon_tex(c.w, c.h, 5 * S)
            col = ImageOps.colorize(tex, (6, 6, 8), (112, 116, 124)).convert("RGBA")
            sheen = grad(c.w, c.h, [(0, (255, 255, 255, 70)), (0.35, (255, 255, 255, 0)), (0.7, (255, 255, 255, 0)), (1, (255, 255, 255, 40))], 38)
            return over(col, sheen)
        return self.cache(("carbon", c.w, c.h), make)

    def body(self, c):
        g = c.geom(0)
        shade = lit(c, (255, 255, 255), (170, 170, 175), 1.0)
        fill = ImageChops.multiply(self.weave(c).convert("RGB"), shade.convert("RGB")).convert("RGBA")
        out = std(c, fill, shadow=(0, 7, 7, (0, 0, 0), 0.55), bev=(0.45, 0.5, 3.5), rim=((8, 8, 10), 2.4))
        out = over(out, paint(ring_of(c.geom(2.4), 1.8), (228, 32, 45), 0.95))
        gl = ImageChops.multiply(c.geom(6), glossmask(c))
        out = over(out, paint(gl, (255, 255, 255), 0.16 if not c.p else 0.06))
        return out

    def base(self, c):
        g, well = c.geom(0), c.geom(30)
        out = paint(blur(shift(g, 0, 5), 6), (0, 0, 0), 0.4)
        out = over(out, clip(self.weave(c), g))
        out = over(out, bevel(g, 3.5, 0.4, 0.5))
        out = over(out, paint(ring_of(c.geom(2.4), 1.8), (228, 32, 45), 0.95))
        out = over(out, paint(well, (4, 4, 6), 0.8))
        out = over(out, inner_shadow(well, 0, 8, 8, (0, 0, 0), 0.9))
        out = over(out, paint(ring_of(well, 1.8), (228, 32, 45), 0.8))
        return out


@register
class GoldLuxury(Skin):
    id, name = "gold-luxury", "Gold Luxury"
    desc = "Polished 24-karat gold with engraved borders and black enamel symbols."
    cats, pro, featured, press, opacity = ["material"], True, True, "glow", 0.96
    swatch = ["#FFE6A0", "#DEA837", "#9A6414", "#18120C"]
    backdrop = ((16, 14, 14), (52, 40, 22))
    scene = 3
    mark_mode = "engrave"
    sym_colors = {}
    mark_col = (26, 20, 14)

    def gold(self, c, ang=-30):
        stops = [(0, (255, 244, 176)), (0.26, (224, 172, 58)), (0.48, (255, 232, 140)), (0.74, (172, 114, 26)), (1, (240, 196, 88))]
        return self.cache(("gold", c.w, c.h, ang), lambda: grad(c.w, c.h, stops, ang))

    def body(self, c):
        g = c.geom(0)
        out = std(c, self.gold(c), shadow=(0, 7, 7, (30, 14, 0), 0.5), bev=(0.8, 0.5, 3.8), rim=((120, 76, 14), 2.6),
                  glow=((255, 200, 80), 7, 0.2))
        if c.kind != "dpad" or True:
            ring = ring_of(c.geom(9), 1.7)
            out = over(out, paint(shift(ring, 0, 1.2), (255, 244, 190), 0.6))
            out = over(out, paint(ring, (120, 76, 14), 0.8))
        return out

    def base(self, c):
        g, well = c.geom(0), c.geom(30)
        out = paint(blur(g, 7), (255, 200, 80), 0.18)
        out = over(out, paint(blur(shift(g, 0, 5), 6), (30, 14, 0), 0.4))
        out = over(out, clip(self.gold(c, -50), g))
        out = over(out, bevel(g, 3.5, 0.8, 0.5))
        out = over(out, paint(ring_of(g, 2.6), (120, 76, 14), 1))
        out = over(out, paint(well, (22, 16, 10), 0.92))
        out = over(out, inner_shadow(well, 0, 8, 8, (0, 0, 0), 0.9))
        out = over(out, paint(ImageChops.subtract(shift(ring_of(well, 2.6), 0, 1.6), ring_of(well, 2.6)), (255, 236, 160), 0.6))
        out = over(out, paint(ring_of(c.geom(8), 1.6), (120, 76, 14), 0.8))
        return out


@register
class MilitaryOlive(Skin):
    id, name = "military-olive", "Olive Drab"
    desc = "Rugged olive-drab field gear with octagonal keys, rivets and sand stencil symbols."
    cats, pro, featured, press = ["material"], False, False, "shrink"
    swatch = ["#68703E", "#464C28", "#E2D6A8", "#22260F"]
    backdrop = ((36, 40, 24), (74, 80, 50))
    scene = 1
    face_shape = "octagon"
    sym_colors = {}
    mark_col = (228, 216, 170)
    P = {"main": ((112, 120, 72), (70, 78, 42)), "dpad": ((96, 104, 62), (60, 66, 36)), "base": ((84, 92, 54), (52, 58, 32)),
         "stick": ((120, 128, 80), (74, 82, 46))}

    def body(self, c):
        hi, lo = self.pal(c)
        grain = self.cache(("mg", c.w, c.h), lambda: ImageChops.add(rng_noise(c.w, c.h, 31), smooth_noise(c.w, c.h, 30, 32), scale=2))
        fill = modulate(lit(c, hi, lo), grain, 0.22)
        out = std(c, fill, shadow=(0, 6, 6, (0, 0, 0), 0.5), bev=(0.32, 0.5, 3.2), rim=((26, 30, 14), 2.8))
        out = over(out, rivets(c, 14, 4.0, (170, 168, 140)))
        return out

    def base(self, c):
        hi, lo = self.pal(c)
        g, well = c.geom(0), c.geom(30)
        grain = self.cache(("mg", c.w, c.h), lambda: ImageChops.add(rng_noise(c.w, c.h, 31), smooth_noise(c.w, c.h, 30, 32), scale=2))
        out = paint(blur(shift(g, 0, 5), 6), (0, 0, 0), 0.4)
        out = over(out, clip(modulate(lit(c, hi, lo), grain, 0.22), g))
        out = over(out, bevel(g, 3.5, 0.3, 0.5))
        out = over(out, paint(ring_of(g, 2.6), (26, 30, 14), 1))
        out = over(out, paint(well, (22, 26, 12), 0.85))
        out = over(out, inner_shadow(well, 0, 8, 8, (0, 0, 0), 0.9))
        out = over(out, rivets(c, 10, 4.2, (170, 168, 140), 4))
        return out

    def mark(self, c):
        m = c.mark_mask()
        if m is None:
            return None
        if c.kind == "face":
            d = ImageDraw.Draw(m)
            d.rectangle([c.cx - 2.8 * S, 0, c.cx + 2.8 * S, c.h], fill=0)
        return over(paint(shift(m, 0, 1.3), (0, 0, 0), 0.5), paint(m, self.mark_col))


# ----------------------------------------------------------------------------------------------------------------
# skins: retro
# ----------------------------------------------------------------------------------------------------------------


@register
class RetroBeige(Skin):
    id, name = "retro-beige", "Retro Beige"
    desc = "Eighties beige plastic keycaps with dished tops, rust stripes and vintage print colors."
    cats, pro, featured, press = ["retro"], False, True, "shrink"
    swatch = ["#E2D6B8", "#C45A3C", "#3F7F95", "#54402F"]
    backdrop = ((58, 50, 42), (96, 86, 70))
    scene = 3
    face_shape = "squircle"
    sym_colors = {"cross": (52, 108, 150), "circle": (196, 64, 48), "square": (214, 122, 58), "triangle": (64, 142, 96)}
    mark_col = (84, 66, 52)
    P = {"main": ((234, 223, 194), (200, 186, 152)), "dpad": ((96, 84, 74), (56, 48, 42)),
         "stick": ((104, 90, 78), (58, 49, 42)), "base": ((222, 210, 178), (186, 172, 140))}

    def body(self, c):
        hi, lo = self.pal(c)
        g = c.geom(0)
        out = std(c, lit(c, hi, lo), shadow=(0, 6, 5, (30, 20, 10), 0.5), bev=(0.55, 0.4, 3.4), rim=((112, 98, 76) if c.role != "dpad" else (30, 24, 20), 2.6))
        if c.kind in ("face", "l3"):
            dish = c.geom(17)
            out = over(out, paint(dish, (0, 0, 0), 0.05))
            out = over(out, inner_shadow(dish, 0, 4, 4, (60, 40, 20), 0.35))
            out = over(out, paint(ImageChops.subtract(shift(ring_of(dish, 1.6), 0, 1.5), ring_of(dish, 1.6)), (255, 255, 240), 0.5))
        elif c.kind in ("sh1", "sh2", "small"):
            x0, y0, x1, y1 = c.rect_box()
            stripe = c.new()
            ImageDraw.Draw(stripe).rectangle([0, (y1 - 17) * S, c.w, (y1 - 9) * S], fill=255)
            out = over(out, paint(ImageChops.multiply(stripe, c.geom(3)), (196, 90, 60), 0.95))
        elif c.kind == "stick":
            out = over(out, paint(ring_of(c.geom(30), 3), (226, 214, 184), 0.8))
        return out

    def mark(self, c):
        m = c.mark_mask()
        if m is None:
            return None
        col = (236, 224, 196) if c.role == "dpad" else self.sym_color(c)
        if c.role == "dpad":
            return over(paint(shift(m, 0, 1.5), (0, 0, 0), 0.5), paint(m, col))
        return paint(m, col, 0.96)


@register
class Pixel8bit(Skin):
    id, name = "pixel-8bit", "Pixel 8-Bit"
    desc = "Chunky hand-placed pixel art buttons with dithered shading and a limited console palette."
    cats, pro, featured, press = ["retro"], False, True, "shrink"
    swatch = ["#3C6EE6", "#E6323C", "#E650AA", "#32C86E"]
    backdrop = ((32, 24, 64), (54, 40, 96))
    scene = 4
    pixel = True
    OUT = (20, 16, 34)
    PAL = {"cross": ((132, 178, 255), (60, 110, 230), (30, 50, 150)), "circle": ((255, 140, 140), (230, 50, 60), (130, 20, 50)),
           "square": ((255, 170, 226), (230, 80, 170), (126, 30, 100)), "triangle": ((140, 250, 176), (50, 200, 110), (20, 106, 70)),
           "main": ((236, 236, 248), (164, 168, 196), (92, 96, 134)), "dpad": ((120, 124, 160), (70, 74, 110), (38, 40, 70)),
           "stick": ((220, 220, 240), (150, 154, 190), (84, 88, 126))}
    BITS = {"L": ["X....", "X....", "X....", "X....", "X....", "X....", "XXXXX"],
            "R": ["XXXX.", "X...X", "X...X", "XXXX.", "X.X..", "X..X.", "X...X"],
            "1": ["..X..", ".XX..", "..X..", "..X..", "..X..", "..X..", ".XXX."],
            "2": [".XXX.", "X...X", "....X", "...X.", "..X..", ".X...", "XXXXX"],
            "3": [".XXX.", "X...X", "....X", "..XX.", "....X", "X...X", ".XXX."]}

    def text(self, d, s, cx, cy, scale, col):
        total = len(s) * 5 * scale + (len(s) - 1) * scale
        x0 = cx - total // 2
        for i, ch in enumerate(s):
            for ry, row in enumerate(self.BITS[ch]):
                for rx, v in enumerate(row):
                    if v == "X":
                        px = x0 + i * 6 * scale + rx * scale
                        py = cy - (7 * scale) // 2 + ry * scale
                        d.rectangle([px, py, px + scale - 1, py + scale - 1], fill=col)

    def render(self, part, pressed):
        c = Ctx(self, part, pressed)
        tw, th = c.tw // 8, c.th // 8
        im = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        OUT = self.OUT + (255,)
        hi, mid, lo = self.PAL.get(c.role, self.PAL["main"])
        mid_ = None
        if pressed:
            hi, mid, lo = mid, lo, darkc(lo, 0.35)
        mid_ = mid
        cx = tw // 2
        cy = th // 2
        txt = (40, 36, 70, 255)
        W = (252, 252, 255, 255)
        k = c.kind
        if k in ("face", "l3", "stick"):
            r = 12 if k != "stick" else 11
            d.ellipse([cx - r, cy - r, cx + r - 1, cy + r - 1], fill=OUT)
            d.ellipse([cx - r + 1, cy - r + 1, cx + r - 2, cy + r - 2], fill=lo + (255,))
            d.ellipse([cx - r + 1, cy - r + 1, cx + r - 4 + (1 if pressed else 0), cy + r - 4 + (1 if pressed else 0)], fill=mid + (255,))
            if not pressed:
                d.arc([cx - r + 3, cy - r + 3, cx + r - 6, cy + r - 6], 195, 262, fill=hi + (255,), width=1)
                d.point([(cx - r + 4, cy - r + 7)], fill=hi + (255,))
            o = 1 if pressed else 0
            if BLANK:
                pass
            elif k == "face":
                s = self.part_symbol(c.part)
                if s == "cross":
                    d.line([(cx - 5, cy - 5 + o), (cx + 4, cy + 4 + o)], fill=W, width=2)
                    d.line([(cx - 5, cy + 4 + o), (cx + 4, cy - 5 + o)], fill=W, width=2)
                elif s == "circle":
                    d.ellipse([cx - 6, cy - 6 + o, cx + 5, cy + 5 + o], outline=W, width=2)
                elif s == "square":
                    d.rectangle([cx - 6, cy - 6 + o, cx + 5, cy + 5 + o], outline=W, width=2)
                else:
                    d.polygon([(cx, cy - 7 + o), (cx - 7, cy + 5 + o), (cx + 7, cy + 5 + o)], fill=W)
                    d.polygon([(cx, cy - 3 + o), (cx - 3, cy + 3 + o), (cx + 3, cy + 3 + o)], fill=mid + (255,))
            elif k == "l3":
                self.text(d, c.part.upper(), cx, cy + o, 2, txt)
            else:
                d.ellipse([cx - 7, cy - 7, cx + 6, cy + 6], outline=lo + (255,), width=1)
                d.ellipse([cx - 3, cy - 3, cx + 2, cy + 2], outline=lo + (255,), width=1)
        elif k == "dpad":
            op = {"up": None, "right": Image.Transpose.ROTATE_270, "down": Image.Transpose.ROTATE_180, "left": Image.Transpose.ROTATE_90}[c.dir]
            sm = Image.new("L", (tw, th), 0)
            ImageDraw.Draw(sm).polygon([(3, 2), (28, 2), (28, 15), (15, 28), (3, 15)], fill=255)
            sm.putpixel((3, 2), 0)
            sm.putpixel((28, 2), 0)
            am = Image.new("L", (tw, th), 0)
            ImageDraw.Draw(am).polygon([(15, 8), (10, 14), (20, 14)], fill=255)
            if op is not None:
                sm, am = sm.transpose(op), am.transpose(op)
            inner = sm.filter(ImageFilter.MinFilter(3))
            mid = ImageChops.multiply(inner, ImageChops.offset(inner, -1, -1))
            mid = ImageChops.multiply(mid, ImageChops.offset(inner, -1, 0))
            hi_m = ImageChops.subtract(mid, ImageChops.offset(mid, 1, 1))
            for m_, col_ in ((sm, OUT), (inner, lo + (255,)), (mid, mid_ + (255,)), (hi_m if not pressed else Image.new("L", (tw, th), 0), hi + (255,)), (am, W)):
                m1 = m_.point(lambda v: 255 if v > 127 else 0)
                im.paste(Image.new("RGBA", (tw, th), col_), (0, 0), m1)
        elif k in ("sh1", "sh2", "small"):
            box = {"sh1": (2, 6, 29, 17), "sh2": (2, 3, 29, 20), "small": (4, 8, 27, 15)}[k]
            rad = {"sh1": 4, "sh2": 5, "small": 3}[k]
            d.rounded_rectangle(box, radius=rad, fill=OUT)
            d.rounded_rectangle([box[0] + 1, box[1] + 1, box[2] - 1, box[3] - 1], radius=max(1, rad - 1), fill=lo + (255,))
            d.rounded_rectangle([box[0] + 1, box[1] + 1, box[2] - 1, box[3] - 2 + (1 if pressed else 0)], radius=max(1, rad - 1), fill=mid + (255,))
            if not pressed:
                d.line([(box[0] + rad, box[1] + 1), (box[2] - rad, box[1] + 1)], fill=hi + (255,))
            my = (box[1] + box[3]) // 2 + (1 if pressed else 0)
            if BLANK:
                pass
            elif k == "small":
                if part == "start":
                    d.polygon([(cx - 1, my - 2), (cx + 2, my), (cx - 1, my + 2)], fill=txt)
                else:
                    for dy in (-2, 0, 2):
                        d.line([(cx - 3, my + dy), (cx + 3, my + dy)], fill=txt)
            else:
                self.text(d, part.upper(), cx, my, 1, txt)
        else:
            d.ellipse([1, 1, 30, 30], fill=OUT)
            d.ellipse([2, 2, 29, 29], fill=(74, 78, 116, 255))
            d.ellipse([6, 6, 25, 25], fill=OUT)
            d.ellipse([7, 7, 24, 24], fill=(30, 30, 54, 255))
            for y in range(32):
                for x in range(32):
                    dist = math.hypot(x - 15.5, y - 15.5)
                    if 9.4 < dist < 10.6 and (x + y) % 2 == 0:
                        d.point([(x, y)], fill=(104, 108, 150, 255))
                    elif 12.5 < dist < 14 and (x + y) % 2 == 0:
                        d.point([(x, y)], fill=(104, 108, 150, 255))
        if pressed and k in ("sh1", "sh2", "small"):
            pass
        return im.resize((c.tw, c.th), Image.NEAREST)

    @staticmethod
    def part_symbol(part):
        return part


@register
class LcdGreen(Skin):
    id, name = "lcd-green", "LCD Handheld"
    desc = "Monochrome green LCD plates with a visible pixel grid, like a classic handheld screen."
    cats, pro, featured, press = ["retro"], False, False, "shrink"
    swatch = ["#A7B780", "#94A66C", "#22301C", "#3B4D2B"]
    backdrop = ((40, 52, 36), (22, 30, 20))
    scene = 1
    face_shape = "squircle"
    sym_thick = 1.15
    sym_colors = {}
    mark_col = (30, 42, 26)
    P = {"main": ((172, 188, 132), (146, 164, 106))}

    def grid(self, c):
        def make():
            m = Image.new("L", (c.w, c.h), 0)
            d = ImageDraw.Draw(m)
            step = 4 * S
            for x in range(0, c.w, step):
                d.line([(x, 0), (x, c.h)], fill=255, width=max(1, S // 2 + 1))
            for y in range(0, c.h, step):
                d.line([(0, y), (c.w, y)], fill=255, width=max(1, S // 2 + 1))
            return m
        return self.cache(("grid", c.w, c.h), make)

    def body(self, c):
        hi, lo = self.pal(c)
        g = c.geom(0)
        p = c.p
        out = blank(c)
        out = over(out, paint(shift(g, 4 if not p else 1.5, 4 if not p else 1.5), (52, 66, 40), 0.9))
        fill = lit(c, hi, lo) if not p else lit(c, (50, 64, 42), (38, 50, 32))
        out = over(out, clip(fill, g))
        out = over(out, paint(ImageChops.multiply(self.grid(c), g), (30, 44, 26), 0.13 if not p else 0.25))
        out = over(out, paint(ring_of(g, 3.2), (30, 42, 26), 1))
        out = over(out, paint(ring_of(c.geom(6), 1.2), (30, 42, 26), 0.35))
        if c.kind == "stick":
            out = over(out, paint(ring_of(c.geom(26), 2.4), (30, 42, 26), 0.5 if not p else 0.9))
            out = over(out, paint(ring_of(c.geom(48), 2.4), (30, 42, 26), 0.4 if not p else 0.9))
        return out

    def base(self, c):
        g, well = c.geom(0), c.geom(30)
        out = paint(shift(g, 4, 4), (52, 66, 40), 0.9)
        out = over(out, clip(lit(c, (172, 188, 132), (146, 164, 106)), g))
        out = over(out, paint(ImageChops.multiply(self.grid(c), g), (30, 44, 26), 0.13))
        out = over(out, paint(ring_of(g, 3.2), (30, 42, 26), 1))
        out = over(out, paint(well, (30, 44, 26), 0.8))
        out = over(out, paint(ring_of(well, 1.6), (172, 188, 132), 0.7))
        return out

    def mark(self, c):
        m = c.mark_mask()
        if m is None:
            return None
        return paint(m, (176, 192, 136) if c.p else (30, 42, 26))


@register
class ArcadeCabinet(Skin):
    id, name = "pad-arcade-cabinet", "Arcade Cabinet"
    desc = "Concave arcade push-buttons in primary colors and a glossy ball-top stick on a black panel."
    cats, pro, featured, press, opacity = ["retro", "playful"], True, True, "spring", 0.96
    swatch = ["#E83240", "#FFC828", "#3C78F0", "#3CD25A"]
    backdrop = ((22, 22, 28), (50, 36, 64))
    scene = 2
    sym_colors = {}
    mark_col = (240, 240, 244)
    sym_scale = 0.78
    sym_thick = 0.8
    BTN = {"cross": (56, 120, 240), "circle": (236, 48, 60), "square": (255, 198, 40), "triangle": (60, 208, 92)}

    def body(self, c):
        p = c.p
        g = c.geom(0)
        out = blank(c)
        if c.kind == "face":
            col = self.BTN[c.part]
            inner = c.geom(12)
            out = over(out, paint(blur(shift(g, 0, 6), 7), (0, 0, 0), 0.5))
            out = over(out, clip(grad(c.w, c.h, [(0, (70, 72, 80)), (1, (14, 14, 18))]), g))
            out = over(out, paint(ring_of(g, 2.0), (150, 154, 166), 0.9))
            dish = rgrad(c.w, c.h, c.w * 0.58, c.h * 0.66, c.w * 0.62, [(0, lighten(col, 0.5)), (0.55, col), (1, darkc(col, 0.45))])
            if p:
                dish = tone(dish, 0.8)
            out = over(out, clip(dish, inner))
            out = over(out, inner_shadow(inner, 3, 8 if not p else 10, 6, darkc(col, 0.7), 0.7))
            out = over(out, paint(ring_of(inner, 1.6), lighten(col, 0.55), 0.5))
            out = over(out, paint(ImageChops.multiply(c.geom(20), glossmask(c)), (255, 255, 255), 0.18))
            return out
        if c.kind == "stick":
            col = (232, 36, 40)
            out = over(out, paint(blur(shift(g, 0, 8), 8), (0, 0, 0), 0.5 if not p else 0.3))
            fill = rgrad(c.w, c.h, c.w * 0.38, c.h * 0.3, c.w * 0.85, [(0, (255, 196, 186)), (0.18, (255, 92, 84)), (0.45, col), (1, (96, 4, 14))])
            if p:
                fill = tone(fill, 0.82)
            out = over(out, clip(fill, g))
            out = over(out, inner_shadow(g, 0, -5, 6, (255, 120, 110), 0.45))
            sp = c.new()
            ImageDraw.Draw(sp).ellipse([c.w * 0.24, c.h * 0.16, c.w * 0.44, c.h * 0.3], fill=255)
            out = over(out, paint(blur(sp, 2.5), (255, 255, 255), 0.8))
            out = over(out, paint(ring_of(g, 1.6), (70, 0, 10), 0.6))
            return out
        # d-pad, shoulders, small, l3: black plastic with a chrome edge
        fill = lit(c, (74, 76, 84), (22, 22, 26))
        out = std(c, fill, shadow=(0, 6, 6, (0, 0, 0), 0.55), bev=(0.35, 0.6, 3.2), rim=((156, 160, 172), 2.2))
        if c.kind == "l3":
            dish = c.geom(14)
            out = over(out, inner_shadow(dish, 2, 6, 5, (0, 0, 0), 0.7))
            out = over(out, paint(ring_of(dish, 1.5), (110, 114, 126), 0.7))
        return out

    def base(self, c):
        g, well = c.geom(0), c.geom(30)
        out = paint(blur(shift(g, 0, 6), 7), (0, 0, 0), 0.5)
        out = over(out, clip(grad(c.w, c.h, [(0, (246, 248, 252)), (0.5, (120, 124, 136)), (1, (210, 214, 222))], -30), g))
        out = over(out, paint(ring_of(g, 3), (50, 52, 60), 1))
        plate = c.geom(14)
        out = over(out, clip(lit(c, (60, 62, 70), (16, 16, 20)), plate))
        out = over(out, paint(ring_of(plate, 1.6), (0, 0, 0), 0.8))
        out = over(out, paint(well, (6, 6, 8), 0.9))
        out = over(out, inner_shadow(well, 0, 8, 8, (0, 0, 0), 0.9))
        for a in (45, 135, 225, 315):
            rad = 100
            out = over(out, screw(c, c.tw / 2 + rad * 0.7 * math.cos(math.radians(a)) * 1.0, c.th / 2 + rad * 0.7 * math.sin(math.radians(a)), 5, (170, 172, 182)))
        return out

    def mark(self, c):
        m = c.mark_mask()
        if m is None:
            return None
        return over(paint(blur(shift(m, 0, 2), 1.5), (0, 0, 0), 0.5), paint(m, self.mark_col, 0.95))


# ----------------------------------------------------------------------------------------------------------------
# skins: playful (comic) + accessibility
# ----------------------------------------------------------------------------------------------------------------


@register
class ComicInk(Skin):
    id, name = "comic-ink", "Comic Ink"
    desc = "Bold flat colors, halftone dots and thick black ink outlines with hard offset shadows."
    cats, pro, featured, press = ["playful"], False, False, "spring"
    swatch = ["#FFD60A", "#FF3C3C", "#2890FF", "#1A1A1A"]
    backdrop = ((255, 236, 150), (255, 196, 90))
    scene = 0
    sym_thick = 1.0
    sym_colors = {}
    mark_col = (255, 255, 255)
    INK = (16, 14, 20)
    P = {"main": ((255, 222, 40), (255, 190, 0)), "cross": ((70, 160, 255), (30, 120, 235)),
         "circle": ((255, 84, 84), (240, 40, 56)), "square": ((255, 130, 205), (235, 80, 170)),
         "triangle": ((90, 220, 120), (40, 190, 90)), "dpad": ((255, 222, 40), (255, 190, 0)),
         "shoulder": ((250, 250, 250), (220, 220, 226)), "stick": ((255, 84, 84), (240, 40, 56)),
         "base": ((255, 250, 235), (240, 226, 190))}

    def body(self, c):
        hi, lo = self.pal(c)
        g = c.geom(0)
        p = c.p
        out = blank(c)
        off = 7 if not p else 2.5
        out = over(out, paint(shift(g, off, off), self.INK))
        fill = Image.new("RGBA", (c.w, c.h), tuple(hi) + (255,))
        if p:
            fill = tone(fill, 0.85)
        out = over(out, clip(fill, g))
        # halftone shading in the lower right
        dots = dots_mask(c, 17, 3.6, 4, 4)
        zone = lin_l(c.w, c.h, 40).point(lambda v: 255 if v > 125 else 0)
        dm = ImageChops.multiply(ImageChops.multiply(dots, zone), c.geom(7))
        out = over(out, paint(dm, darkc(lo, 0.22), 0.55))
        if c.round:
            hl = c.new()
            r = c.w * 0.5 - 24 * S
            ImageDraw.Draw(hl).arc([c.cx - r, c.cy - r, c.cx + r, c.cy + r], 200, 255, fill=255, width=int(7 * S))
            out = over(out, paint(ImageChops.multiply(hl, g), (255, 255, 255), 0.9))
        out = over(out, paint(ring_of(g, 7), self.INK))
        if c.kind == "stick":
            out = over(out, paint(ring_of(c.geom(30), 4), self.INK, 0.9))
        return out

    def base(self, c):
        g, well = c.geom(0), c.geom(30)
        out = paint(shift(g, 7, 7), self.INK)
        out = over(out, clip(Image.new("RGBA", (c.w, c.h), (255, 250, 235, 255)), g))
        dots = ImageChops.multiply(dots_mask(c, 17, 3.2, 4, 4), c.geom(8))
        out = over(out, paint(ImageChops.multiply(dots, lin_l(c.w, c.h, 40).point(lambda v: 255 if v > 125 else 0)), (200, 180, 130), 0.6))
        out = over(out, paint(ring_of(g, 7), self.INK))
        out = over(out, paint(well, self.INK, 0.95))
        out = over(out, paint(ring_of(well, 3), (255, 222, 40), 0.9))
        return out

    def mark(self, c):
        m = c.mark_mask(1.1 if c.kind == "face" else 1.0)
        if m is None:
            return None
        col = (255, 255, 255) if c.role not in ("shoulder", "shoulder2", "base") else self.INK
        if c.role in ("dpad",):
            col = self.INK
        out = paint(grow(m, 4.4), self.INK) if col != self.INK else blank(c)
        return over(out, paint(m, col))


@register
class HighContrastYellow(Skin):
    id, name = "high-contrast-yellow", "High Contrast Yellow"
    desc = "Large black symbols on bright yellow with thick outlines. Pressed keys invert for instant feedback."
    cats, pro, featured, press, opacity = ["accessibility"], False, True, "grow", 1.0
    swatch = ["#FFE100", "#000000", "#FFFFFF", "#FFB400"]
    backdrop = ((52, 56, 66), (30, 32, 40))
    scene = 2
    sym_scale = 1.18
    sym_thick = 1.55
    sym_filled = True
    sym_colors = {}
    Y, K = (255, 226, 0), (0, 0, 0)

    def cols(self, c):
        fg, bg = self.K, self.Y
        if c.role in ("shoulder", "shoulder2", "small"):
            bg = (255, 255, 255)
        if c.p:
            fg, bg = bg, fg
        return fg, bg

    def body(self, c):
        fg, bg = self.cols(c)
        g = c.geom(0)
        ink = self.K if not c.p else (255, 226, 0)
        edge = self.K if not c.p else (255, 255, 255)
        out = Image.new("RGBA", (c.w, c.h), (0, 0, 0, 0))
        out = over(out, paint(g, edge))
        out = over(out, paint(c.geom(13), bg))
        out = over(out, paint(ring_of(g, 5.5), (255, 255, 255) if not c.p else (255, 226, 0)))
        out = over(out, paint(ring_of(g, 13), self.K if not c.p else (255, 255, 255)))
        out = over(out, paint(ImageChops.subtract(c.geom(5.5), c.geom(13)), self.K if not c.p else (255, 255, 255)))
        out = over(out, paint(c.geom(13), bg))
        if c.kind == "stick":
            out = over(out, paint(ring_of(c.geom(34), 6), fg))
        return out

    def base(self, c):
        g, well = c.geom(0), c.geom(24)
        out = paint(g, self.K)
        out = over(out, paint(c.geom(12), (255, 255, 255)))
        out = over(out, paint(well, self.K))
        out = over(out, paint(ring_of(well, 5), self.Y))
        return out

    def mark(self, c):
        m = c.mark_mask(1.0, 1.0 if c.kind == "face" else 1.12)
        if m is None:
            return None
        fg, _ = self.cols(c)
        return paint(m, fg)


@register
class HighContrastNight(HighContrastYellow):
    id, name = "high-contrast-night", "High Contrast Night"
    desc = "Pure black keys with heavy white outlines and large filled symbols for maximum legibility."
    cats, pro, featured, press, opacity = ["accessibility"], False, False, "grow", 1.0
    swatch = ["#000000", "#FFFFFF", "#00E5FF", "#FFEB3B"]
    backdrop = ((190, 196, 206), (120, 126, 140))
    scene = 1
    SYM = {"cross": (0, 229, 255), "circle": (255, 90, 90), "square": (255, 120, 240), "triangle": (120, 255, 120)}

    def cols(self, c):
        fg, bg = (255, 255, 255), (0, 0, 0)
        if c.part in self.SYM:
            fg = self.SYM[c.part]
        if c.p:
            fg, bg = bg, (255, 255, 255)
            if c.part in self.SYM:
                bg = self.SYM[c.part]
        return fg, bg

    def body(self, c):
        fg, bg = self.cols(c)
        g = c.geom(0)
        out = paint(g, (255, 255, 255))
        out = over(out, paint(c.geom(11), (0, 0, 0)))
        out = over(out, paint(c.geom(16), bg if c.p else (0, 0, 0)))
        if c.kind == "stick":
            out = over(out, paint(ring_of(c.geom(34), 6), (255, 255, 255) if not c.p else (0, 0, 0)))
        return out

    def base(self, c):
        g, well = c.geom(0), c.geom(24)
        out = paint(g, (255, 255, 255))
        out = over(out, paint(c.geom(12), (0, 0, 0)))
        out = over(out, paint(ring_of(well, 5), (255, 255, 255)))
        return out

    def mark(self, c):
        m = c.mark_mask(1.0, 1.0 if c.kind == "face" else 1.12)
        if m is None:
            return None
        fg, _ = self.cols(c)
        return paint(m, fg)


# ----------------------------------------------------------------------------------------------------------------
# form-driven skins: shapes x materials x glyph styles (assembled from small parts, see FormSkin below)
# ----------------------------------------------------------------------------------------------------------------


def _pseed(c, extra=0):
    return zlib.crc32(c.part.encode()) % 9973 + extra


def ngon(cx, cy, r, n, rot=-90):
    return [(cx + r * math.cos(math.radians(rot + 360.0 * i / n)), cy + r * math.sin(math.radians(rot + 360.0 * i / n)))
            for i in range(n)]


def poly_mask(c, pts, soft=0):
    m = c.new()
    ImageDraw.Draw(m).polygon([(x * S, y * S) for x, y in pts], fill=255)
    return soften(m, soft) if soft else m


def wobble(m, amp, cell, seed):
    """Smooth hand-drawn distortion of a mask (mesh warp with random vertex offsets)."""
    w, h = m.size
    cols = max(2, int(round(w / float(cell * S))))
    rows = max(2, int(round(h / float(cell * S))))
    rnd = random.Random(seed)
    pts = []
    for j in range(rows + 1):
        row = []
        for i in range(cols + 1):
            e = 0 if i in (0, cols) or j in (0, rows) else 1
            row.append((i * w / float(cols) + rnd.uniform(-amp, amp) * S * e, j * h / float(rows) + rnd.uniform(-amp, amp) * S * e))
        pts.append(row)
    mesh = []
    for j in range(rows):
        for i in range(cols):
            x0, y0 = int(round(i * w / float(cols))), int(round(j * h / float(rows)))
            x1, y1 = int(round((i + 1) * w / float(cols))), int(round((j + 1) * h / float(rows)))
            p00, p01, p11, p10 = pts[j][i], pts[j + 1][i], pts[j + 1][i + 1], pts[j][i + 1]
            mesh.append(((x0, y0, x1, y1), (p00[0], p00[1], p01[0], p01[1], p11[0], p11[1], p10[0], p10[1])))
    return m.transform((w, h), Image.Transform.MESH, mesh, Image.BILINEAR)


def stripe_mask(c, period, duty, angle=45, ofs=0):
    n = int(math.hypot(c.w, c.h)) + 8
    m = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(m)
    p = period * S
    x = (ofs * S) % p - p
    while x < n:
        d.rectangle([x, 0, x + p * duty, n], fill=255)
        x += p
    m = m.rotate(angle, resample=Image.BILINEAR)
    x0, y0 = (n - c.w) // 2, (n - c.h) // 2
    return m.crop((x0, y0, x0 + c.w, y0 + c.h))


def grid_mask(c, period, thick):
    m = c.new()
    d = ImageDraw.Draw(m)
    v = 0
    while v < 260:
        d.line([(v * S, 0), (v * S, c.h)], fill=255, width=max(1, int(thick * S)))
        d.line([(0, v * S), (c.w, v * S)], fill=255, width=max(1, int(thick * S)))
        v += period
    return m


def led_grid(c, spacing, r, lit_fn=None):
    m = c.new()
    d = ImageDraw.Draw(m)
    n = int(130 / spacing) + 1
    for j in range(-n, n + 1):
        for i in range(-n, n + 1):
            x, y = 128 + i * spacing, 128 + j * spacing
            if lit_fn is None or lit_fn(x * S, y * S):
                d.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=255)
    return m


def dropout(m, c, seed, thr, soft=0.8):
    n = blur(rng_noise(c.w, c.h, seed), soft).point(lambda v: 255 if v > thr else 0)
    return ImageChops.multiply(m, n)


# ----------------------------------------------------------------------------------------------------------------
# shape masks
# ----------------------------------------------------------------------------------------------------------------


def face_shape(c, name, R):
    m = c.new()
    d = ImageDraw.Draw(m)
    if R <= 0:
        return m
    cx = cy = 128.0
    if name == "circle":
        d.ellipse([(cx - R) * S, (cy - R) * S, (cx + R) * S, (cy + R) * S], fill=255)
    elif name == "squircle":
        q = R * 0.93
        d.rounded_rectangle([(cx - q) * S, (cy - q) * S, (cx + q) * S, (cy + q) * S], radius=max(1, R * 0.42 * S), fill=255)
    elif name in ("hex", "hexf"):
        m = poly_mask(c, ngon(cx, cy, R * 1.04, 6, -90 if name == "hex" else 0), 4)
    elif name == "diamond":
        m = poly_mask(c, ngon(cx, cy, min(R * 1.2, 123), 4, -90), 8)
    elif name == "octagon":
        m = poly_mask(c, ngon(cx, cy, R * 1.04, 8, 22.5), 2)
    elif name == "chamfer":
        k, cu = R * 0.9, R * 0.36
        pts = [(cx - k + cu, cy - k), (cx + k - cu, cy - k), (cx + k, cy - k + cu), (cx + k, cy + k - cu), (cx + k - cu, cy + k),
               (cx - k + cu, cy + k), (cx - k, cy + k - cu), (cx - k, cy - k + cu)]
        m = poly_mask(c, pts, 3)
    elif name == "shield":
        o = cy - R * 0.08
        pts = [(cx - R * 0.88, o - R * 0.82), (cx + R * 0.88, o - R * 0.82), (cx + R * 0.88, o + R * 0.12), (cx, o + R * 1.0),
               (cx - R * 0.88, o + R * 0.12)]
        m = poly_mask(c, pts, 9)
    elif name == "flower":
        for dx, dy in ((0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)):
            x, y, r = cx + dx * R, cy + dy * R, R * 0.56
            d.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=255)
        r = R * 0.62
        d.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S], fill=255)
    elif name == "cog":
        r = R * 0.84
        d.ellipse([(cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S], fill=255)
        for i in range(10):
            a = 360.0 * i / 10
            pts = []
            for rr, da in ((R * 0.78, -10), (R * 1.04, -6.5), (R * 1.04, 6.5), (R * 0.78, 10)):
                pts.append((cx + rr * math.cos(math.radians(a + da)), cy + rr * math.sin(math.radians(a + da))))
            d.polygon([(x * S, y * S) for x, y in pts], fill=255)
        m = soften(m, 1.4)
    elif name == "burst":
        pts = []
        for i in range(24):
            rr = R * 1.04 if i % 2 == 0 else R * 0.8
            a = math.radians(-90 + 15 * i)
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        m = poly_mask(c, pts, 2.2)
    return m


def dpad_shape(c, form, inset=0):
    m = c.new()
    d = ImageDraw.Draw(m)
    i = inset
    if form == "wedge":
        m = poly_mask(c, ((18, 18), (238, 18), (238, 128), (128, 238), (18, 128)), 17)
        if i:
            m = shrink(m, i)
    elif form == "arm":
        d.rounded_rectangle([(78 + i) * S, (18 + i) * S, (178 - i) * S, 330 * S], radius=max(1, (28 - i * 0.5) * S), fill=255)
    elif form == "key":
        d.rounded_rectangle([(66 + i) * S, (66 + i) * S, (190 - i) * S, (190 - i) * S], radius=max(1, (28 - i * 0.5) * S), fill=255)
    elif form == "sector":
        pts = []
        for a in range(-31, 32, 3):
            pts.append((128 + 236 * math.sin(math.radians(a)), 260 - 236 * math.cos(math.radians(a))))
        for a in range(31, -32, -3):
            pts.append((128 + 70 * math.sin(math.radians(a)), 260 - 70 * math.cos(math.radians(a))))
        m = poly_mask(c, pts, 6)
        if i:
            m = shrink(m, i)
    elif form == "tri":
        m = poly_mask(c, ((128, 20), (222, 206), (34, 206)), 12)
        if i:
            m = shrink(m, i)
    elif form == "capsule":
        d.rounded_rectangle([(88 + i) * S, (20 + i) * S, (168 - i) * S, (236 - i) * S], radius=max(1, (40 - i) * S), fill=255)
    elif form == "dot":
        r = 62 - i
        d.ellipse([(128 - r) * S, (128 - r) * S, (128 + r) * S, (128 + r) * S], fill=255)
    elif form == "dia":
        m = poly_mask(c, ngon(128, 128, 92, 4, -90), 8)
        if i:
            m = shrink(m, i)
    return c.rotate(m) if form != "chev" else m


def rect_shape(c, name, inset=0):
    x0, y0, x1, y1 = [v for v in c.rect_box()]
    h = y1 - y0
    right = c.part in ("r1", "r2")
    m = c.new()
    d = ImageDraw.Draw(m)
    if name == "pill":
        r = h / 2.0 if c.kind != "sh2" else 56
        d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=r * S, fill=255)
    elif name == "rect":
        d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=14 * S, fill=255)
    elif name == "chamfer":
        k = 0.34 * h
        m = poly_mask(c, [(x0 + k, y0), (x1 - k, y0), (x1, y0 + k), (x1, y1 - k), (x1 - k, y1), (x0 + k, y1), (x0, y1 - k), (x0, y0 + k)], 2)
    elif name == "slant":
        s = 0.42 * h
        pts = [(x0, y0), (x1 - s, y0), (x1, y1), (x0 + s, y1)] if right else [(x0 + s, y0), (x1, y0), (x1 - s, y1), (x0, y1)]
        m = poly_mask(c, pts, 5)
    elif name == "trap":
        s = 0.38 * h
        m = poly_mask(c, [(x0 + s, y0), (x1 - s, y0), (x1, y1), (x0, y1)], 6)
    elif name == "arch":
        d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=(h * 0.62) * S, fill=255, corners=(True, True, False, False))
        d.rectangle([x0 * S, (y1 - h * 0.3) * S, x1 * S, y1 * S], fill=255)
        m = soften(m, 2)
    elif name == "hexbar":
        k = h / 2.0
        m = poly_mask(c, [(x0 + k, y0), (x1 - k, y0), (x1, (y0 + y1) / 2), (x1 - k, y1), (x0 + k, y1), (x0, (y0 + y1) / 2)], 3)
    elif name == "circle":
        r = h / 2.0
        cx = (x0 + x1) / 2.0
        d.ellipse([(cx - r) * S, (y0 + h / 2 - r) * S, (cx + r) * S, (y0 + h / 2 + r) * S], fill=255)
    if inset:
        m = shrink(m, inset)
    return m


def dpad_arrow(sk, c):
    form = sk.dpad
    if form == "chev":
        m = c.new()
        t = 24 * S * sk.sym_thick
        sc = sk.sym_scale
        stroke(ImageDraw.Draw(m), [((128 - 46 * sc) * S, (158 + 14 * sc) * S), (128 * S, (108 + 4 * sc) * S), ((128 + 46 * sc) * S, (158 + 14 * sc) * S)], t)
        return c.rotate(m)
    a, b, w = ARROW[form]
    sc = sk.sym_scale
    mid = (a + b) / 2.0
    a, b, w = mid + (a - mid) * sc, mid + (b - mid) * sc, w * sc
    m = poly_mask(c, ((128, a), (128 - w, b), (128 + w, b)))
    return c.rotate(soften(m, 2.6))


ARROW = {"wedge": (56, 108, 30), "arm": (66, 126, 28), "key": (94, 152, 30), "sector": (70, 134, 30), "tri": (96, 160, 26),
         "capsule": (70, 126, 26), "dot": (94, 152, 28), "dia": (96, 152, 26)}


# ----------------------------------------------------------------------------------------------------------------
# glyph styles (applied to the symbol / label / arrow mask)
# ----------------------------------------------------------------------------------------------------------------


def g_solid(sk, c, m, col):
    return paint(m, col)


def g_shadow(sk, c, m, col):
    return mark_paint(c, m, col, "shadow")


def g_engrave(sk, c, m, col):
    return mark_paint(c, m, col, "engrave")


def g_emboss(sk, c, m, col):
    return mark_paint(c, m, col, "emboss")


def g_glow(sk, c, m, col):
    return mark_paint(c, m, col, "glow")


def g_hollow(sk, c, m, col):
    return paint(ring_of(m, 3.2), col)


def g_wobble(sk, c, m, col):
    return paint(wobble(m, 2.0, 16, _pseed(c, 5)), col)


def g_dots(sk, c, m, col):
    sp = sk.led
    mb = blur(m, sp * 0.28)
    lit_m = led_grid(c, sp, sp * 0.36, lambda x, y: mb.getpixel((min(c.w - 1, int(x)), min(c.h - 1, int(y)))) > 100)
    out = paint(blur(lit_m, 5), col, 0.65 if not c.p else 0.9)
    out = over(out, paint(lit_m, col))
    return over(out, paint(shrink(lit_m, 1.4), lighten(col, 0.7)))


def g_pixel(sk, c, m, col):
    g = 8
    sm = m.resize((c.w // (g * S), c.h // (g * S)), Image.BOX).point(lambda v: 255 if v > 90 else 0)
    big = sm.resize((c.w, c.h), Image.NEAREST)
    return over(paint(shift(big, 1.5, 2), darkc(col, 0.7), 0.8), paint(big, col))


def g_satin(sk, c, m, col):
    out = paint(grow(m, 1.0), darkc(col, 0.45), 0.9)
    st = ImageChops.multiply(m, stripe_mask(c, 4.4, 0.62, 58))
    return over(over(out, paint(m, darkc(col, 0.2))), paint(st, lighten(col, 0.25)))


def g_stitch(sk, c, m, col):
    ring = ring_of(m, 3.0)
    st = ImageChops.multiply(ring, stripe_mask(c, 8, 0.58, 45))
    return over(over(paint(m, col, 0.22), paint(shift(st, 0, 1.2), (0, 0, 0), 0.35)), paint(st, col))


def g_brush(sk, c, m, col):
    mw = wobble(grow(m, 1.2), 2.6, 22, _pseed(c, 11))
    streak = streak_noise(c.w, c.h, 14, _pseed(c, 3)).point(lambda v: 255 if v > 58 else 150)
    return paint(ImageChops.multiply(mw, streak), col, 0.96)


def g_stencil(sk, c, m, col):
    gaps = c.new()
    d = ImageDraw.Draw(gaps)
    d.rectangle([126 * S, 0, 130 * S, c.h], fill=255)
    d.rectangle([0, 126 * S, c.w, 130 * S], fill=255)
    return paint(ImageChops.subtract(m, gaps), col)


def g_halftone(sk, c, m, col):
    dm = dots_mask(c, 9, 2.6)
    return over(paint(ring_of(m, 2.4), col), paint(ImageChops.multiply(m, dm), col))


def g_pop(sk, c, m, col):
    out = paint(shift(grow(m, 1.6), 3.4, 3.4), (10, 10, 14), 1)
    out = over(out, paint(grow(m, 2.4), (10, 10, 14), 1))
    return over(out, paint(m, col))


def g_inlay(sk, c, m, col):
    return over(paint(m, darkc(col, 0.72), 0.96), paint(ring_of(m, 2.2), col))


def g_chalk(sk, c, m, col):
    mw = wobble(m, 1.6, 14, _pseed(c, 7))
    return over(paint(blur(mw, 2.4), col, 0.14), paint(dropout(mw, c, _pseed(c, 4), 70), col, 0.92))


def g_stamp(sk, c, m, col):
    return paint(dropout(wobble(m, 1.0, 12, _pseed(c, 9)), c, _pseed(c, 6), 56, 0.6), col, 0.94)


GLYPHS_STYLE = {"solid": g_solid, "shadow": g_shadow, "engrave": g_engrave, "emboss": g_emboss, "glow": g_glow, "hollow": g_hollow,
                "wobble": g_wobble, "dots": g_dots, "pixel": g_pixel, "satin": g_satin, "stitch": g_stitch, "brush": g_brush,
                "stencil": g_stencil, "halftone": g_halftone, "pop": g_pop, "inlay": g_inlay, "chalk": g_chalk, "stamp": g_stamp}


# ----------------------------------------------------------------------------------------------------------------
# materials: fn(sk, c, M, hi, lo) -> RGBA layer. M is the shape mask (supersampled), c.p = pressed
# ----------------------------------------------------------------------------------------------------------------


def D(c, col, t=0.2):
    return darkc(col, t) if c.p else col


def shadow_of(c, M, dy=5, br=7, a=0.36, dx=0, col=(0, 0, 0)):
    if c.p:
        dy, a = dy * 0.35, a * 0.6
    return paint(blur(shift(M, dx, dy), br), col, a)


def relief(m, d=1.5):
    return ImageChops.add(shift(m, -d, -d), ImageChops.invert(shift(m, d, d)), 2.0)


def top_fade(c, k=1.6):
    return lin_l(c.w, c.h).point(lambda v: max(0, 255 - int(v * k)))


def bbox_of(M):
    b = M.getbbox()
    return b if b else (0, 0, 1, 1)


def mat_none(sk, c, M, hi, lo):
    return blank(c)


def mat_flat(sk, c, M, hi, lo):
    out = shadow_of(c, M, 3, 5, 0.3)
    out = over(out, clip(lit(c, D(c, hi), D(c, lo)), M))
    pat = (sk.patterns or {}).get(c.part)
    if pat:
        dk = darkc(lo, 0.5)
        if pat == "stripes":
            pm = ImageChops.multiply(shrink(M, 4), stripe_mask(c, 15, 0.3, 45))
        elif pat == "dots":
            pm = ImageChops.multiply(shrink(M, 6), dots_mask(c, 24, 5.2))
        else:
            pm = ImageChops.multiply(shrink(M, 4), grid_mask(c, 22, 2.4))
        out = over(out, paint(pm, dk, 0.4))
    out = over(out, paint(ring_of(M, sk.line), sk.edge or darkc(lo, 0.4), 0.9))
    return out


def mat_outline(sk, c, M, hi, lo):
    ring = ring_of(M, sk.line)
    out = blank(c)
    fa = sk.fill_a + (0.3 if c.p else 0)
    if fa:
        out = over(out, paint(M, hi, fa))
    if sk.glow:
        out = over(out, paint(blur(ring, 7), hi, 0.5))
    if sk.rings == 2:
        out = over(out, paint(ring_of(shrink(M, sk.line * 3.4), sk.line * 0.55), hi, 0.85))
    return over(out, paint(ring, hi))


def mat_neon(sk, c, M, hi, lo):
    ring = ring_of(M, 5)
    out = paint(M, (0, 0, 0), 0.5)
    k = 1.35 if c.p else 1.0
    out = over(out, paint(blur(ring, 9), hi, min(1.0, 0.8 * k)))
    out = over(out, paint(blur(ring, 3), hi, 1.0))
    out = over(out, paint(ring_of(M, 2.4), lighten(hi, 0.75)))
    if c.p:
        out = over(out, paint(shrink(M, 6), hi, 0.3))
    return out


def mat_glass(sk, c, M, hi, lo):
    ga = sk.ga * (1.7 if c.p else 1.0)
    out = shadow_of(c, M, 4, 8, 0.22)
    fill = grad(c.w, c.h, [(0, (255, 255, 255, int(min(255, 255 * ga * 1.3)))), (1, tuple(hi) + (int(min(255, 255 * ga * 0.8)),))], 70)
    out = over(out, clip(fill, M))
    out = over(out, paint(ring_of(M, 2.6), (255, 255, 255), 0.85))
    out = over(out, paint(ring_of(shrink(M, 5), 1.3), (255, 255, 255), 0.3))
    out = over(out, paint(ImageChops.multiply(shrink(M, 8), top_fade(c, 1.7)), (255, 255, 255), 0.34))
    out = over(out, paint(ImageChops.subtract(ring_of(shrink(M, 3), 3), shift(ring_of(shrink(M, 3), 3), 0, -2)), lighten(hi, 0.4), 0.4))
    return out


def mat_brushed(sk, c, M, hi, lo):
    seed = _pseed(c)
    fill = grad(c.w, c.h, [(0, lighten(hi, 0.2)), (0.5, hi), (1, lo)], 35)
    tex = sk.cache(("streak", c.w, c.h, seed), lambda: streak_noise(c.w, c.h, 4, seed))
    fill = modulate(tone(fill, 0.8 if c.p else 1.0), tex, 0.6)
    out = shadow_of(c, M, 5, 7, 0.42)
    out = over(out, clip(fill, M))
    out = over(out, bevel(M, 3, 0.5 if not c.p else 0.15, 0.45))
    return over(out, paint(ring_of(M, 2.4), darkc(lo, 0.55), 0.85))


def mat_wood(sk, c, M, hi, lo):
    seed = 3 + _pseed(c) % 11
    dark, light = sk.wood
    t = sk.cache(("wood", c.tw, c.th, seed), lambda: wood_tex(c.tw, c.th, seed, dark, light))
    t = t.resize((c.w, c.h), Image.BICUBIC)
    if sk.bamboo:
        bands = c.new()
        bd = ImageDraw.Draw(bands)
        for y in (60, 190):
            bd.rectangle([0, (y - 3) * S, c.w, (y + 3) * S], fill=255)
        t = over(t, paint(blur(bands, 1.2), darkc(dark, 0.4), 0.85))
    shade = lit(c, (255, 250, 245), (150, 128, 112))
    fill = ImageChops.multiply(t.convert("RGB"), shade.convert("RGB")).convert("RGBA")
    out = shadow_of(c, M, 6, 7, 0.5, 0, (20, 8, 0))
    out = over(out, clip(tone(fill, 0.8 if c.p else 1.0), M))
    out = over(out, bevel(M, 3.4, 0.45, 0.5))
    return over(out, paint(ring_of(M, 2.6), darkc(dark, 0.4), 0.9))


def _textured(sk, c, M, hi, lo, cell, amt, speck=0.0, rel=0.0):
    seed = _pseed(c, 21)
    n = sk.cache(("tx", c.w, c.h, cell, seed), lambda: smooth_noise(c.w, c.h, cell, seed))
    fill = modulate(lit(c, D(c, hi), D(c, lo)), relief(n, rel) if rel else n, amt)
    out = shadow_of(c, M, 5, 7, 0.45)
    out = over(out, clip(fill, M))
    if speck:
        sp = rng_noise(c.w, c.h, seed + 1).point(lambda v: 255 if v > 250 - int(speck * 20) else 0)
        out = over(out, paint(ImageChops.multiply(blur(sp, 0.5), M), lighten(hi, 0.7), 0.55))
    return out


def mat_stone(sk, c, M, hi, lo):
    out = _textured(sk, c, M, hi, lo, 70, 0.32, 1.0)
    out = over(out, bevel(M, 4, 0.5, 0.55))
    return over(out, paint(ring_of(M, 2.6), darkc(lo, 0.55), 0.8))


def mat_hammered(sk, c, M, hi, lo):
    out = _textured(sk, c, M, hi, lo, 26, 0.9, 0, 1.8)
    out = over(out, bevel(M, 3.4, 0.55, 0.5))
    return over(out, paint(ring_of(M, 2.6), darkc(lo, 0.6), 0.85))


def mat_clay(sk, c, M, hi, lo):
    out = _textured(sk, c, M, hi, lo, 38, 0.55, 0, 3.0)
    out = over(out, paint(ImageChops.multiply(blur(ImageChops.subtract(M, shift(M, 3, 4)), 3), M), (255, 255, 255), 0.4 if not c.p else 0.1))
    return over(out, paint(ImageChops.multiply(blur(ImageChops.subtract(M, shift(M, -3, -4)), 3), M), darkc(lo, 0.5), 0.5))


def mat_leather(sk, c, M, hi, lo):
    seed = _pseed(c, 31)
    g = sk.cache(("lg", c.w, c.h, seed), lambda: smooth_noise(c.w, c.h, 5, seed))
    fill = modulate(lit(c, D(c, hi), D(c, lo)), relief(g, 1.0), 0.8)
    out = shadow_of(c, M, 5, 7, 0.45)
    out = over(out, clip(fill, M))
    out = over(out, bevel(M, 3.6, 0.32, 0.55))
    sc = sk.acc or lighten(hi, 0.6)
    st = ImageChops.multiply(ring_of(shrink(M, 15), 2.2), stripe_mask(c, 9, 0.58, 45))
    if sk.perf:
        dm = ImageChops.multiply(dots_mask(c, 17, 2.6), ImageChops.subtract(shrink(M, 6), shrink(M, 30)))
        out = over(out, paint(dm, (0, 0, 0), 0.55))
    out = over(out, paint(shift(st, 0, 1.4), (0, 0, 0), 0.4))
    out = over(out, paint(st, sc))
    return over(out, paint(ring_of(M, 2.2), darkc(lo, 0.6), 0.8))


def mat_inflate(sk, c, M, hi, lo):
    h = blur(M, sk.infl)
    lm = relief(h, 3.0)
    base = ramp(h.point(lambda v: max(0, min(255, int((v - 100) * 1.9)))), [(0, lo), (0.55, hi), (1, lighten(hi, 0.3))])
    base = modulate(tone(base, 0.82 if c.p else 1.0), lm, 0.65)
    base.putalpha(M.point(lambda v: int(v * sk.alpha)))
    out = shadow_of(c, M, 6, 9, 0.28, 0, darkc(lo, 0.6))
    if sk.alpha < 0.95:
        out = over(out, paint(ImageChops.multiply(blur(shift(shrink(M, 10), 0, 12), 8), M), lighten(hi, 0.5), 0.35))
    out = over(out, base)
    if sk.alpha < 0.95:
        out = over(out, paint(ring_of(M, 2.2), lighten(hi, 0.45), 0.55))
    spec = blur(ImageChops.multiply(shrink(M, 12), top_fade(c, 1.9)), 2.2)
    out = over(out, paint(spec, (255, 255, 255), sk.spec * (0.5 if c.p else 1.0)))
    return out


def mat_paper(sk, c, M, hi, lo):
    cols = [lo, mixc(lo, hi, 0.5), hi]
    out = blank(c)
    for ins, col in zip((0, 9, 18), cols):
        mk = shrink(M, ins) if ins else M
        out = over(out, paint(blur(shift(mk, 0.5, 3.2 if not c.p else 1.2), 2.6), (0, 0, 0), 0.4))
        out = over(out, paint(mk, D(c, col)))
        out = over(out, paint(ImageChops.multiply(ImageChops.subtract(mk, shift(mk, 0, 1.6)), mk), (255, 255, 255), 0.35))
    return out


def mat_ledmatrix(sk, c, M, hi, lo):
    out = shadow_of(c, M, 5, 6, 0.5)
    out = over(out, clip(grad(c.w, c.h, [(0, (40, 44, 52)), (1, (12, 14, 18))]), M))
    out = over(out, bevel(M, 2.6, 0.35, 0.5))
    out = over(out, paint(ring_of(M, 2.4), (96, 102, 116), 0.9))
    grid = ImageChops.multiply(led_grid(c, sk.led, sk.led * 0.3), shrink(M, 6))
    return over(out, paint(grid, (58, 64, 78), 0.95))


def mat_blueprint(sk, c, M, hi, lo):
    out = paint(M, lo, 0.6 + (0.2 if c.p else 0))
    out = over(out, paint(ImageChops.multiply(grid_mask(c, 32, 1.1), M), hi, 0.22))
    out = over(out, paint(ring_of(M, 2.4), hi))
    inner = ring_of(shrink(M, 11), 1.4)
    out = over(out, paint(ImageChops.multiply(inner, stripe_mask(c, 10, 0.6, 0)), hi, 0.8))
    x0, y0, x1, y1 = bbox_of(M)
    tk = c.new()
    d = ImageDraw.Draw(tk)
    mx, my = (x0 + x1) // 2, (y0 + y1) // 2
    L = 12 * S
    for (px, py, dx, dy) in ((mx, y0, 0, 1), (mx, y1, 0, -1), (x0, my, 1, 0), (x1, my, -1, 0)):
        d.line([(px - dx * 6 * S, py - dy * 6 * S), (px + dx * L, py + dy * L)], fill=255, width=int(1.6 * S))
    return over(out, paint(tk, hi, 0.95))


def mat_cap(sk, c, M, hi, lo):
    dish = shrink(M, 12)
    out = shadow_of(c, M, 6, 7, 0.5)
    out = over(out, clip(grad(c.w, c.h, [(0, (70, 72, 80)), (1, (14, 14, 18))]), M))
    out = over(out, paint(ring_of(M, 2.0), (150, 154, 166), 0.9))
    cap = rgrad(c.w, c.h, c.w * 0.58, c.h * 0.66, c.w * 0.62, [(0, lighten(hi, 0.5)), (0.55, hi), (1, darkc(hi, 0.5))])
    out = over(out, clip(tone(cap, 0.8 if c.p else 1.0), dish))
    out = over(out, inner_shadow(dish, 3, 8 if not c.p else 10, 6, darkc(hi, 0.7), 0.7))
    out = over(out, paint(ring_of(dish, 1.6), lighten(hi, 0.55), 0.5))
    return over(out, paint(ImageChops.multiply(shrink(M, 20), glossmask(c)), (255, 255, 255), 0.16))


def mat_halftone(sk, c, M, hi, lo):
    out = paint(shift(M, 6, 6) if not c.p else shift(M, 2.5, 2.5), (12, 12, 16), 1)
    out = over(out, clip(tone(lit(c, hi, hi), 0.85 if c.p else 1.0), M))
    dm = ImageChops.multiply(dots_mask(c, 15, 4.6), ImageChops.multiply(M, lin_l(c.w, c.h, 45).point(lambda v: min(255, int(v * 1.5)))))
    out = over(out, paint(dm, lo, 0.75))
    return over(out, paint(ring_of(M, 6.5), (12, 12, 16)))


def mat_felt(sk, c, M, hi, lo):
    seed = _pseed(c, 41)
    fib = rng_noise(c.w, c.h, seed)
    fill = modulate(lit(c, D(c, hi), D(c, lo)), fib, 0.4)
    out = shadow_of(c, M, 4, 6, 0.38)
    fz = ImageChops.multiply(blur(M, 1.1), blur(rng_noise(c.w, c.h, seed + 2), 0.5).point(lambda v: min(255, 150 + v // 2)))
    out = over(out, clip(fill, steep(fz, 3)))
    st = ImageChops.multiply(ring_of(shrink(M, 12), 3.0), stripe_mask(c, 10, 0.5, 35))
    out = over(out, paint(shift(st, 0, 1.2), (0, 0, 0), 0.3))
    return over(out, paint(st, sk.acc or lighten(hi, 0.7)))


def mat_crystal(sk, c, M, hi, lo):
    out = shadow_of(c, M, 6, 8, 0.4)
    out = over(out, clip(grad(c.w, c.h, [(0, hi), (1, lo)], 35), M))
    x0, y0, x1, y1 = bbox_of(M)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    R = max(x1 - x0, y1 - y0) * 0.7
    n = sk.facets
    rnd = random.Random(_pseed(c, 51))
    fm_hi, fm_lo = c.new(), c.new()
    dh, dl = ImageDraw.Draw(fm_hi), ImageDraw.Draw(fm_lo)
    for k in range(n):
        a0, a1 = 360.0 * k / n, 360.0 * (k + 1) / n
        mid = math.radians((a0 + a1) / 2)
        s = math.cos(mid + math.radians(40))
        pts = [(cx, cy), (cx + R * math.cos(math.radians(a0)), cy + R * math.sin(math.radians(a0))),
               (cx + R * math.cos(math.radians(a1)), cy + R * math.sin(math.radians(a1)))]
        v = int(255 * min(1.0, abs(s) * 0.8 + rnd.uniform(0, 0.15)))
        (dh if s > 0 else dl).polygon(pts, fill=v)
    out = over(out, paint(ImageChops.multiply(fm_hi, M), (255, 255, 255), sk.fac_a * (0.6 if c.p else 1)))
    out = over(out, paint(ImageChops.multiply(fm_lo, M), (0, 0, 0), sk.fac_a * 0.9))
    if sk.table:
        T = shrink(M, 24)
        out = over(out, clip(grad(c.w, c.h, [(0, lighten(hi, 0.55)), (1, hi)], 30), T))
        out = over(out, paint(ring_of(T, 1.8), (255, 255, 255), 0.7))
    out = over(out, paint(ring_of(M, 2.2), lighten(hi, 0.6), 0.9))
    return out


def mat_hud(sk, c, M, hi, lo):
    out = paint(M, lo, 0.35 + (0.25 if c.p else 0))
    mk = c.new()
    d = ImageDraw.Draw(mk)
    x0, y0, x1, y1 = bbox_of(M)
    w = 3.4 * S
    if c.kind in ("face", "l3", "stick", "base"):
        for a0 in (-80, 10, 100, 190):
            d.arc((x0, y0, x1, y1), a0, a0 + 70, fill=255, width=int(w))
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        r = (x1 - x0) / 2.0
        for a in (0, 90, 180, 270):
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            d.line([(cx + ca * (r - 13 * S), cy + sa * (r - 13 * S)), (cx + ca * (r + 2 * S), cy + sa * (r + 2 * S))], fill=255, width=int(w * 0.8))
        mk = ImageChops.lighter(mk, ImageChops.multiply(ring_of(shrink(M, 16), 1.4), stripe_mask(c, 9, 0.55, 20)))
    else:
        L = min(x1 - x0, y1 - y0) * 0.34
        for (px, py, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            d.line([(px, py + sy * L), (px, py), (px + sx * L, py)], fill=255, width=int(w), joint="curve")
        mk = ImageChops.lighter(mk, ring_of(M, 1.4).point(lambda v: v // 2))
    if sk.glow:
        out = over(out, paint(blur(mk, 6), hi, 0.6))
    return over(out, paint(mk, hi))


def mat_ghost(sk, c, M, hi, lo):
    out = paint(M, hi, 0.16 + (0.16 if c.p else 0))
    return over(out, paint(ring_of(M, 2.2), hi, 0.5 + (0.2 if c.p else 0)))


def mat_hc(sk, c, M, hi, lo):
    fill, ink = (lo, hi) if c.p else (hi, lo)
    out = paint(M, fill)
    out = over(out, paint(ring_of(M, 9), ink))
    return over(out, paint(ring_of(shrink(M, 13), 2.4), ink, 0.9))


def mat_sketch(sk, c, M, hi, lo):
    seed = _pseed(c, 61)
    Mw = wobble(M, 2.4, 26, seed)
    out = paint(blur(shift(Mw, 2, 3), 2.2), (0, 0, 0), 0.18)
    out = over(out, paint(Mw, hi, 0.94))
    ink = lo
    hatch = ImageChops.multiply(shrink(Mw, 4), stripe_mask(c, 8, 0.2, 40))
    hatch = ImageChops.multiply(hatch, lin_l(c.w, c.h, 45).point(lambda v: 255 if c.p else max(0, min(255, (v - 110) * 3))))
    out = over(out, paint(hatch, ink, 0.75))
    out = over(out, paint(ring_of(wobble(M, 2.8, 22, seed + 9), 1.8), ink, 0.65))
    return over(out, paint(ring_of(Mw, 3.0), ink, 0.95))


def mat_embroidery(sk, c, M, hi, lo):
    out = shadow_of(c, M, 3, 4, 0.3)
    fill = clip(tone(lit(c, hi, lo, 1.4), 0.8 if c.p else 1.0), M)
    wv = ImageChops.multiply(grid_mask(c, 4, 0.7), M)
    out = over(out, fill)
    out = over(out, paint(wv, (0, 0, 0), 0.12))
    sat = ImageChops.subtract(shrink(M, 6), shrink(M, 6 + 11))
    ac = sk.acc or lighten(hi, 0.7)
    out = over(out, paint(sat, darkc(ac, 0.45)))
    out = over(out, paint(ImageChops.multiply(sat, stripe_mask(c, 4.4, 0.62, 58)), ac))
    return over(out, paint(ring_of(M, 1.6), darkc(lo, 0.5), 0.7))


def mat_bracket(sk, c, M, hi, lo):
    out = paint(M, lo, 0.72)
    out = over(out, paint(ImageChops.multiply(stripe_mask(c, 4, 0.34, 0), M), (0, 0, 0), 0.22))
    x0, y0, x1, y1 = bbox_of(M)
    mk = c.new()
    d = ImageDraw.Draw(mk)
    h = y1 - y0
    t = int(5.5 * S)
    k = h * 0.2
    for (xa, sgn) in ((x0 + 3 * S, 1), (x1 - 3 * S, -1)):
        d.line([(xa + sgn * 14 * S, y0 + k), (xa, y0 + k), (xa, y1 - k), (xa + sgn * 14 * S, y1 - k)], fill=255, width=t, joint="curve")
    out = over(out, paint(blur(mk, 4), hi, 0.5))
    out = over(out, paint(mk, hi))
    if c.p:
        out = over(out, paint(shrink(M, 6), hi, 0.22))
    return out


def mat_chalk(sk, c, M, hi, lo):
    seed = _pseed(c, 71)
    Mw = wobble(M, 1.6, 22, seed)
    fill = modulate(lit(c, lo, lo), smooth_noise(c.w, c.h, 9, seed), 0.35)
    out = clip(fill, Mw)
    ring = ring_of(Mw, 4.4)
    out = over(out, paint(blur(ring, 5), hi, 0.14))
    out = over(out, paint(dropout(ring, c, seed + 3, 64), hi, 0.92))
    d2 = ImageChops.multiply(ring_of(shrink(Mw, 11), 1.8), stripe_mask(c, 11, 0.5, 30))
    out = over(out, paint(dropout(d2, c, seed + 5, 90), hi, 0.6))
    if c.p:
        out = over(out, paint(dropout(shrink(Mw, 4), c, seed + 8, 140), hi, 0.3))
    return out


def mat_watercolor(sk, c, M, hi, lo):
    seed = _pseed(c, 81)
    out = blank(c)
    for dx, dy, br, a, col in ((-3, -2, 4, 0.5, hi), (3, 2, 5, 0.42, mixc(hi, lo, 0.5)), (0, 0, 2.5, 0.5, lo)):
        out = over(out, paint(ImageChops.multiply(blur(shift(M, dx, dy), br), blur(M, 6).point(lambda v: min(255, v * 3))), D(c, col, 0.15), a))
    out = over(out, paint(blur(ring_of(M, 4.5), 1.3), darkc(hi, 0.4), 0.38))
    gran = smooth_noise(c.w, c.h, 3, seed).point(lambda v: 255 if v > 190 else 0)
    return over(out, paint(ImageChops.multiply(gran, M), (255, 255, 255), 0.28))


def mat_brush(sk, c, M, hi, lo):
    seed = _pseed(c, 91)
    Mw = wobble(M, 3.0, 24, seed)
    ring = ring_of(Mw, 11)
    streak = streak_noise(c.w, c.h, 11, seed).point(lambda v: 255 if v > 60 else 130)
    out = paint(blur(ring, 1.4), hi, 0.25)
    out = over(out, paint(ImageChops.multiply(ring, streak), hi, 0.95))
    if c.p:
        out = over(out, paint(shrink(Mw, 10), hi, 0.22))
    return out


def mat_toybrick(sk, c, M, hi, lo):
    dy = 4 if c.p else 8
    out = shadow_of(c, shift(M, 0, dy), 5, 7, 0.4)
    out = over(out, clip(grad(c.w, c.h, [(0, darkc(hi, 0.25)), (1, darkc(lo, 0.45))]), shift(M, 0, dy)))
    out = over(out, clip(lit(c, D(c, lighten(hi, 0.1)), D(c, lo)), M))
    out = over(out, bevel(M, 3, 0.55, 0.35))
    if c.kind in ("face", "l3", "stick"):
        studs = [(128, 120, 50)]
    elif c.kind == "dpad":
        studs = [(128, 128, 34)]
    elif c.kind == "base":
        studs = []
    else:
        x0, y0, x1, y1 = c.rect_box()
        n = 3 if c.kind != "small" else 2
        studs = [(x0 + (x1 - x0) * (i + 0.5) / n, (y0 + y1) / 2 - 3, 18 if c.kind != "small" else 14) for i in range(n)]
    for sx, sy, r in studs:
        sm = c.new()
        sd = ImageDraw.Draw(sm)
        sd.ellipse([(sx - r) * S, (sy - r + 7) * S, (sx + r) * S, (sy + r + 7) * S], fill=255)
        st = c.new()
        ImageDraw.Draw(st).ellipse([(sx - r) * S, (sy - r) * S, (sx + r) * S, (sy + r) * S], fill=255)
        out = over(out, paint(blur(shift(sm, 1, 3), 2.5), (0, 0, 0), 0.35))
        out = over(out, clip(grad(c.w, c.h, [(0, darkc(hi, 0.2)), (1, darkc(lo, 0.4))]), ImageChops.lighter(sm, st)))
        out = over(out, clip(lit(c, lighten(hi, 0.35), hi), st))
        out = over(out, paint(ring_of(st, 1.6), darkc(lo, 0.3), 0.6))
    return out


def mat_keycap(sk, c, M, hi, lo):
    off = -2.0 if c.p else -7.0
    top = shift(shrink(M, 11), 0, off)
    out = shadow_of(c, M, 6, 7, 0.45)
    out = over(out, clip(grad(c.w, c.h, [(0, darkc(hi, 0.2)), (1, darkc(lo, 0.55))]), M))
    out = over(out, paint(ring_of(M, 2.0), darkc(lo, 0.65), 0.8))
    out = over(out, clip(grad(c.w, c.h, [(0, lighten(hi, 0.22)), (1, D(c, hi, 0.08))]), top))
    out = over(out, inner_shadow(top, 0, 6, 8, (0, 0, 0), 0.32))
    out = over(out, paint(ImageChops.multiply(ImageChops.subtract(top, shift(top, 0, 2.4)), top), (255, 255, 255), 0.5))
    return out


def mat_enamel(sk, c, M, hi, lo):
    inner = shrink(M, 11)
    out = shadow_of(c, M, 6, 7, 0.5)
    chrome = grad(c.w, c.h, [(0, (250, 250, 252)), (0.35, (150, 154, 162)), (0.55, (236, 238, 242)), (1, (110, 114, 124))], -35)
    out = over(out, clip(chrome, M))
    out = over(out, paint(ring_of(M, 2.0), (60, 62, 70), 0.8))
    out = over(out, paint(ring_of(shrink(M, 9), 1.8), (30, 30, 36), 0.75))
    out = over(out, clip(lit(c, D(c, lighten(hi, 0.18)), D(c, lo)), inner))
    out = over(out, paint(ImageChops.multiply(shrink(M, 16), glossmask(c)), (255, 255, 255), 0.36 if not c.p else 0.12))
    return out


def mat_circuit(sk, c, M, hi, lo):
    out = shadow_of(c, M, 4, 6, 0.4)
    out = over(out, clip(tone(lit(c, hi, lo), 0.82 if c.p else 1.0), M))
    gold = sk.acc or (214, 176, 84)
    tr = c.new()
    d = ImageDraw.Draw(tr)
    rnd = random.Random(_pseed(c, 101))
    x0, y0, x1, y1 = bbox_of(M)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    for _ in range(7):
        a = rnd.uniform(0, 360)
        r0 = min(x1 - x0, y1 - y0) / 2.0 * rnd.uniform(0.62, 0.9)
        px, py = cx + r0 * math.cos(math.radians(a)), cy + r0 * math.sin(math.radians(a))
        pts = [(px, py)]
        for _s in range(3):
            if rnd.random() < 0.5:
                px += rnd.choice((-1, 1)) * rnd.uniform(14, 40) * S
            else:
                py += rnd.choice((-1, 1)) * rnd.uniform(14, 40) * S
            pts.append((px, py))
        d.line(pts, fill=255, width=int(3 * S), joint="curve")
        d.ellipse([pts[-1][0] - 5 * S, pts[-1][1] - 5 * S, pts[-1][0] + 5 * S, pts[-1][1] + 5 * S], fill=255)
    band = ImageChops.subtract(shrink(M, 5), shrink(M, 34)) if c.kind in ("face", "l3", "stick", "base") else shrink(M, 5)
    tr = ImageChops.multiply(tr, band)
    out = over(out, paint(shift(tr, 0, 1.2), (0, 0, 0), 0.4))
    out = over(out, paint(tr, gold, 0.92))
    out = over(out, paint(ring_of(M, 3.2), gold, 0.95))
    return over(out, paint(ring_of(M, 1.2), darkc(lo, 0.5), 0.8))


def mat_marble(sk, c, M, hi, lo):
    seed = _pseed(c, 111)
    n = sk.cache(("mb", c.w, c.h, seed), lambda: smooth_noise(c.w, c.h, 90, seed))
    n2 = smooth_noise(c.w, c.h, 36, seed + 1)
    v = ImageChops.add(n, n2, 2.0)
    veins = v.point(lambda x: max(0, 255 - abs(x - 128) * 22))
    out = shadow_of(c, M, 5, 7, 0.4)
    out = over(out, clip(lit(c, D(c, hi), D(c, mixc(hi, lo, 0.5))), M))
    out = over(out, paint(ImageChops.multiply(blur(veins, 0.7), M), lo, 0.55))
    out = over(out, paint(ImageChops.multiply(shrink(M, 6), top_fade(c, 1.8)), (255, 255, 255), 0.22))
    out = over(out, bevel(M, 3, 0.4, 0.3))
    return over(out, paint(ring_of(M, sk.line), sk.acc or lo, 0.95))


def mat_sticker(sk, c, M, hi, lo):
    W = grow(M, 7)
    out = shadow_of(c, W, 5, 6, 0.42)
    out = over(out, paint(W, (252, 252, 252)))
    out = over(out, paint(ring_of(W, 1.2), (190, 192, 200), 0.8))
    out = over(out, clip(tone(lit(c, hi, lo), 0.82 if c.p else 1.0), M))
    band = c.new()
    ImageDraw.Draw(band).polygon([(0, c.h * 0.18), (c.w, c.h * 0.02), (c.w, c.h * 0.2), (0, c.h * 0.4)], fill=255)
    out = over(out, paint(ImageChops.multiply(blur(band, 1.0), shrink(M, 4)), (255, 255, 255), 0.4))
    return out


def mat_riso(sk, c, M, hi, lo):
    seed = _pseed(c, 121)
    grain = rng_noise(c.w, c.h, seed).point(lambda v: 255 if v > 38 else 0)
    a = ImageChops.multiply(grain, M)
    b = ImageChops.multiply(grain, ring_of(shift(M, 4, 3), 7))
    out = over(blank(c), paint(a, hi, 0.9 if not c.p else 1.0))
    ov = ImageChops.multiply(a, shift(M, 4, 3))
    out = over(out, paint(b, lo, 0.92))
    out = over(out, paint(ImageChops.multiply(ov, dots_mask(c, 11, 2.4)), lo, 0.55))
    return out


def mat_stamp(sk, c, M, hi, lo):
    seed = _pseed(c, 131)
    ring = ImageChops.lighter(ring_of(M, 8), ring_of(shrink(M, 14), 2.4))
    out = paint(dropout(M, c, seed, 18), hi, 0.12 + (0.18 if c.p else 0))
    return over(out, paint(dropout(ring, c, seed + 1, 52), hi, 0.95))


def mat_ceramic(sk, c, M, hi, lo):
    seed = _pseed(c, 141)
    out = shadow_of(c, M, 6, 7, 0.4)
    out = over(out, clip(lit(c, D(c, hi), D(c, lo)), M))
    rnd = random.Random(seed)
    cr = c.new()
    d = ImageDraw.Draw(cr)
    for _ in range(3):
        a = rnd.uniform(0, 360)
        px, py = 128 + 100 * math.cos(math.radians(a)), 128 + 100 * math.sin(math.radians(a))
        pts = [(px * S, py * S)]
        for _s in range(7):
            a += rnd.uniform(-35, 35) + 180 * (1 if _s == 0 else 0)
            px += math.cos(math.radians(a)) * 15
            py += math.sin(math.radians(a)) * 15
            pts.append((px * S, py * S))
        d.line(pts, fill=255, width=int(2.6 * S), joint="curve")
    cr = ImageChops.multiply(cr, shrink(M, 3))
    gold = sk.acc or (214, 170, 70)
    out = over(out, paint(shift(cr, 0, 1.2), (0, 0, 0), 0.35))
    out = over(out, paint(cr, gold))
    out = over(out, paint(ImageChops.multiply(shrink(M, 8), top_fade(c, 1.6)), (255, 255, 255), 0.4))
    return over(out, paint(ring_of(M, 2.0), darkc(lo, 0.4), 0.8))


def mat_knit(sk, c, M, hi, lo):
    st = c.new()
    sd = ImageDraw.Draw(st)
    sh = c.new()
    shd = ImageDraw.Draw(sh)
    for j in range(-1, 30):
        for i in range(-1, 24):
            x, y = i * 11.0 + (5.5 if j % 2 else 0), j * 9.0
            sd.line([(x * S, y * S), ((x + 5.5) * S, (y + 8) * S), ((x + 11) * S, y * S)], fill=255, width=int(4.2 * S), joint="curve")
            shd.line([(x * S, (y + 2) * S), ((x + 5.5) * S, (y + 10) * S), ((x + 11) * S, (y + 2) * S)], fill=255, width=int(4.2 * S), joint="curve")
    base = clip(lit(c, D(c, darkc(hi, 0.3)), D(c, darkc(lo, 0.35))), M)
    out = shadow_of(c, M, 5, 6, 0.38)
    out = over(out, base)
    out = over(out, paint(ImageChops.multiply(sh, M), (0, 0, 0), 0.4))
    out = over(out, clip(lit(c, D(c, lighten(hi, 0.12)), D(c, lo)), ImageChops.multiply(st, M)))
    fz = ImageChops.multiply(ring_of(M, 3), rng_noise(c.w, c.h, _pseed(c, 151)).point(lambda v: 255 if v > 150 else 0))
    return over(out, paint(fz, lighten(hi, 0.25), 0.8))


def mat_deco(sk, c, M, hi, lo):
    gold = hi
    out = shadow_of(c, M, 5, 6, 0.5)
    out = over(out, clip(grad(c.w, c.h, [(0, (34, 34, 40)), (1, (6, 6, 9))]), M))
    out = over(out, paint(ring_of(M, 3.2), gold))
    out = over(out, paint(ring_of(shrink(M, 8), 1.4), gold, 0.9))
    inner = shrink(M, 12)
    if c.kind in ("face", "l3", "stick", "base"):
        rays = c.new()
        rd = ImageDraw.Draw(rays)
        r = 118
        for a in range(0, 360, 15):
            rd.line([(128 * S + 0.34 * r * S * math.cos(math.radians(a)), 128 * S + 0.34 * r * S * math.sin(math.radians(a))),
                     (128 * S + 0.8 * r * S * math.cos(math.radians(a)), 128 * S + 0.8 * r * S * math.sin(math.radians(a)))], fill=255, width=int(1.6 * S))
        out = over(out, paint(ImageChops.multiply(rays, inner), gold, 0.55))
    else:
        out = over(out, paint(ImageChops.multiply(stripe_mask(c, 12, 0.1, 45), inner), gold, 0.5))
    if c.p:
        out = over(out, paint(inner, gold, 0.22))
    return out


def mat_mesh(sk, c, M, hi, lo):
    out = shadow_of(c, M, 4, 6, 0.4)
    out = over(out, clip(grad(c.w, c.h, [(0, mixc(lo, (0, 0, 0), 0.5)), (1, (2, 8, 8))]), M))
    out = over(out, paint(ImageChops.multiply(grid_mask(c, 7, 0.8), shrink(M, 3)), hi, 0.2))
    rg = ring_of(M, 2.4)
    out = over(out, paint(blur(rg, 5), hi, 0.5))
    out = over(out, paint(rg, hi, 0.95))
    if c.p:
        out = over(out, paint(shrink(M, 5), hi, 0.22))
    return out


def mat_bubble(sk, c, M, hi, lo):
    ring = ring_of(M, 11)
    hue = grad(c.w, c.h, [(0, (255, 120, 200)), (0.33, (120, 230, 255)), (0.66, (255, 240, 130)), (1, (255, 120, 200))], 35)
    out = paint(M, (255, 255, 255), 0.07 + (0.12 if c.p else 0))
    hue.putalpha(ring.point(lambda v: int(v * 0.55)))
    out = over(out, hue)
    out = over(out, paint(ring_of(M, 1.8), (255, 255, 255), 0.7))
    x0, y0, x1, y1 = bbox_of(M)
    w, h = x1 - x0, y1 - y0
    sp = c.new()
    ImageDraw.Draw(sp).ellipse([x0 + w * 0.14, y0 + h * 0.1, x0 + w * 0.42, y0 + h * 0.3], fill=255)
    sp2 = c.new()
    ImageDraw.Draw(sp2).ellipse([x0 + w * 0.64, y0 + h * 0.76, x0 + w * 0.8, y0 + h * 0.86], fill=255)
    out = over(out, paint(blur(sp, 1.6), (255, 255, 255), 0.8))
    return over(out, paint(blur(sp2, 1.6), (255, 255, 255), 0.5))


def mat_carve(sk, c, M, hi, lo):
    out = shadow_of(c, M, 4, 5, 0.4)
    out = over(out, clip(lit(c, D(c, lo), D(c, lo)), M))
    band = ImageChops.subtract(shrink(M, 6), shrink(M, 30)) if c.kind in ("face", "l3", "stick", "base") else shrink(M, 6)
    out = over(out, paint(ImageChops.multiply(stripe_mask(c, 9, 0.12, 40), band), hi, 0.9))
    out = over(out, paint(ring_of(shrink(M, 5), 2.2), hi))
    out = over(out, paint(ring_of(M, 2.0), hi, 0.5))
    if c.p:
        out = over(out, paint(shrink(M, 6), hi, 0.2))
    return out


def mat_gloss(sk, c, M, hi, lo):
    out = shadow_of(c, M, 6, 8, 0.4)
    out = over(out, clip(tone(lit(c, hi, lo), 0.82 if c.p else 1.0), M))
    out = over(out, bevel(M, 3.2, 0.45 if not c.p else 0.12, 0.45))
    out = over(out, paint(ImageChops.multiply(shrink(M, 7), top_fade(c, 1.35)), (255, 255, 255), 0.4 if not c.p else 0.14))
    return over(out, paint(ring_of(M, 2.2), sk.edge or darkc(lo, 0.5), 0.8))


def mat_neumorph(sk, c, M, hi, lo):
    dk = darkc(lo, 0.45)
    if not c.p:
        out = paint(blur(shift(M, 6, 7), 9), dk, 0.5)
        out = over(out, paint(blur(shift(M, -6, -6), 8), (255, 255, 255), 0.9))
        out = over(out, clip(grad(c.w, c.h, [(0, lighten(hi, 0.25)), (1, lo)], 40), M))
    else:
        out = clip(grad(c.w, c.h, [(0, lo), (1, lighten(hi, 0.15))], 40), M)
        out = over(out, inner_shadow(M, 5, 6, 8, dk, 0.6))
        out = over(out, paint(ImageChops.multiply(blur(ImageChops.subtract(M, shift(M, 4, 4)), 3), M), (255, 255, 255), 0.55))
    return over(out, paint(ring_of(M, 1.2), (255, 255, 255), 0.35))


MATS = {k[4:]: v for k, v in dict(globals()).items() if k.startswith("mat_") and callable(v)}


# ----------------------------------------------------------------------------------------------------------------
# stick / base decoration and the FormSkin class
# ----------------------------------------------------------------------------------------------------------------


def stick_decor(sk, c, M, hi, lo):
    st = sk.stick_style
    out = blank(c)
    ac = sk.acc or lighten(hi, 0.5)
    if st == "grooves":
        for ins, a in ((16, 0.28), (34, 0.24), (52, 0.2)):
            rg = ImageChops.subtract(shrink(M, ins), shrink(M, ins + 4))
            out = over(out, paint(rg, (0, 0, 0), a))
            out = over(out, paint(shift(rg, 0, 1.6), (255, 255, 255), a * 0.7))
    elif st == "dome":
        edge = ImageChops.multiply(blur(ImageChops.subtract(M, shrink(M, 24)), 8), M)
        out = over(out, paint(edge, (0, 0, 0), 0.4))
        sp = c.new()
        ImageDraw.Draw(sp).ellipse([c.w * 0.26, c.h * 0.17, c.w * 0.48, c.h * 0.34], fill=255)
        out = over(out, paint(blur(sp, 3.5), (255, 255, 255), 0.7))
    elif st == "concave":
        W = shrink(M, 14)
        out = over(out, inner_shadow(W, 6, 8, 9, (0, 0, 0), 0.6))
        out = over(out, paint(ImageChops.subtract(shift(ring_of(W, 2.0), -1.6, -2.2), ring_of(W, 2.0)), (255, 255, 255), 0.4))
        out = over(out, paint(ring_of(W, 1.6), (0, 0, 0), 0.25))
    elif st == "ridges":
        rd = c.new()
        d = ImageDraw.Draw(rd)
        for a in range(0, 360, 12):
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            d.line([(128 * S + 70 * S * ca, 128 * S + 70 * S * sa), (128 * S + 90 * S * ca, 128 * S + 90 * S * sa)], fill=255, width=int(3.2 * S))
        out = over(out, paint(ImageChops.multiply(rd, M), (0, 0, 0), 0.4))
        out = over(out, paint(shift(ImageChops.multiply(rd, M), 0.8, 1.2), (255, 255, 255), 0.25))
    elif st == "dimples":
        dm = ImageChops.multiply(dots_mask(c, 19, 4.6), shrink(M, 12))
        out = over(out, paint(shift(dm, 0.8, 1.2), (255, 255, 255), 0.35))
        out = over(out, paint(dm, (0, 0, 0), 0.38))
    elif st == "reticle":
        rt = c.new()
        d = ImageDraw.Draw(rt)
        t = int(2.6 * S)
        d.line([(128 * S, 40 * S), (128 * S, 216 * S)], fill=255, width=t)
        d.line([(40 * S, 128 * S), (216 * S, 128 * S)], fill=255, width=t)
        d.ellipse([88 * S, 88 * S, 168 * S, 168 * S], outline=255, width=t)
        d.ellipse([122 * S, 122 * S, 134 * S, 134 * S], fill=255)
        out = over(out, paint(ImageChops.multiply(rt, shrink(M, 8)), ac, 0.9))
    elif st == "pointer":
        pt = c.new()
        d = ImageDraw.Draw(pt)
        d.line([(128 * S, 128 * S), (128 * S, 52 * S)], fill=255, width=int(10 * S))
        d.ellipse([(128 - 5) * S, (52 - 5) * S, (128 + 5) * S, (52 + 5) * S], fill=255)
        out = over(out, paint(shift(pt, 0, 1.5), (0, 0, 0), 0.4))
        out = over(out, paint(pt, ac))
        out = over(out, paint(ring_of(shrink(M, 20), 2), ac, 0.5))
    elif st == "ring":
        out = over(out, paint(ring_of(shrink(M, 18), 3.4), ac, 0.9))
        dot = c.new()
        ImageDraw.Draw(dot).ellipse([108 * S, 108 * S, 148 * S, 148 * S], fill=255)
        out = over(out, paint(dot, ac, 0.9))
    elif st == "cap":
        W = shrink(M, 22)
        out = over(out, paint(blur(shift(W, 0, 5), 5), (0, 0, 0), 0.3))
        out = over(out, clip(lit(c, lighten(hi, 0.3), hi), W))
        out = over(out, paint(ring_of(W, 1.6), darkc(lo, 0.4), 0.6))
    return out


def base_decor(sk, c, M, hi, lo):
    st = sk.base_style
    out = blank(c)
    ac = sk.acc or lighten(hi, 0.5)
    if st in ("well", "ticks", "cross", "dots"):
        well = shrink(M, 30)
        wa = sk.well_a
        if wa:
            out = over(out, paint(well, (0, 0, 0), wa))
            out = over(out, inner_shadow(well, 0, 7, 8, (0, 0, 0), min(1.0, wa + 0.1)))
            out = over(out, paint(ImageChops.subtract(shift(ring_of(well, 2), 0, 1.5), ring_of(well, 2)), (255, 255, 255), 0.22))
        else:
            out = over(out, paint(ring_of(well, 1.8), ac, 0.7))
    if st == "ticks":
        tk = c.new()
        d = ImageDraw.Draw(tk)
        for a in range(0, 360, 15):
            ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
            r0, r1 = (100, 112) if a % 90 == 0 else (104, 110)
            d.line([(128 * S + r0 * S * ca, 128 * S + r0 * S * sa), (128 * S + r1 * S * ca, 128 * S + r1 * S * sa)], fill=255, width=int(2.4 * S))
        out = over(out, paint(ImageChops.multiply(tk, M), ac, 0.9))
    elif st == "cross":
        cr = c.new()
        d = ImageDraw.Draw(cr)
        d.line([(128 * S, 28 * S), (128 * S, 228 * S)], fill=255, width=int(2.0 * S))
        d.line([(28 * S, 128 * S), (228 * S, 128 * S)], fill=255, width=int(2.0 * S))
        out = over(out, paint(ImageChops.multiply(cr, M), ac, 0.6))
    elif st == "dots":
        dt = c.new()
        d = ImageDraw.Draw(dt)
        for a in range(0, 360, 20):
            x, y = 128 + 104 * math.cos(math.radians(a)), 128 + 104 * math.sin(math.radians(a))
            d.ellipse([(x - 3) * S, (y - 3) * S, (x + 3) * S, (y + 3) * S], fill=255)
        out = over(out, paint(ImageChops.multiply(dt, M), ac, 0.9))
    return out


PIX_BITS = {"L": ["X....", "X....", "X....", "X....", "X....", "X....", "XXXXX"],
            "R": ["XXXX.", "X...X", "X...X", "XXXX.", "X.X..", "X..X.", "X...X"],
            "1": ["..X..", ".XX..", "..X..", "..X..", "..X..", "..X..", ".XXX."],
            "2": [".XXX.", "X...X", "....X", "...X.", "..X..", ".X...", "XXXXX"],
            "3": [".XXX.", "X...X", "....X", "..XX.", "....X", "X...X", ".XXX."]}


class FormSkin(Skin):
    mat, glyph = "flat", "solid"
    face, dpad, stick, sh, small = "circle", "wedge", "disc", "pill", "pill"
    base_shape, stick_shape = "circle", "circle"
    stick_style, base_style, well_a = "grooves", "well", 0.6
    body_dpad = True
    line, fill_a, glow, rings = 3.0, 0.0, False, 1
    edge = acc = None
    patterns = None
    wood, bamboo, perf = ((150, 96, 50), (232, 176, 108)), False, False
    ga, infl, alpha, spec = 0.3, 18, 1.0, 0.5
    facets, fac_a, table = 10, 0.34, True
    led = 14
    wob = 0.0
    mark_dy = 0.0
    pgrid = 0
    dither = False
    PAL = None
    sym_colors = {}
    mark_col = (240, 240, 244)
    base_mat = None
    ink_p = None
    OUT = (10, 10, 14)

    def __init__(self, **kw):
        super().__init__()
        for k, v in kw.items():
            setattr(self, k, v)
        self.P = dict(self.P) if self.P else {}

    # -- geometry -------------------------------------------------------------------------------------------
    def shape(self, c, inset=0):
        key = ("shape", c.part, inset)
        if key in c._g:
            return c._g[key]
        k = c.kind
        if k == "dpad":
            m = dpad_shape(c, self.dpad, inset)
        elif k in ("sh1", "sh2", "small"):
            m = rect_shape(c, self.sh if k != "small" else self.small, inset)
        else:
            R = {"face": 100, "l3": 100, "stick": 96, "base": 120}[k] - inset
            name = self.face if k in ("face", "l3") else (self.stick_shape if k == "stick" else self.base_shape)
            if isinstance(name, dict):
                name = name.get(c.part, name.get("*", "circle"))
            m = face_shape(c, name, R)
        if self.wob and not inset:
            m = wobble(m, self.wob, 36, _pseed(c, 201))
        c._g[key] = m
        return m

    # -- rendering ------------------------------------------------------------------------------------------
    def render(self, part, pressed):
        c = Ctx(self, part, pressed)
        if self.pgrid:
            return self.render_pixel(c)
        hi, lo = self.pal(c)
        M = self.shape(c)
        k = c.kind
        if k == "dpad" and not self.body_dpad:
            layer = blank(c)
        elif k == "base":
            layer = MATS[self.base_mat or self.mat](self, c, M, hi, lo)
            layer = over(layer, base_decor(self, c, M, hi, lo))
        else:
            layer = MATS[self.mat](self, c, M, hi, lo)
            if k == "stick":
                layer = over(layer, stick_decor(self, c, M, hi, lo))
        mk = self.glyph_layer(c)
        if mk is not None:
            layer = over(layer, mk)
        return finish(layer, c)

    def glyph_mask(self, c):
        if c.kind == "base":
            return None
        if c.kind == "stick":
            return None
        if c.kind == "dpad":
            return dpad_arrow(self, c)
        return c.mark_mask()

    def glyph_layer(self, c):
        m = self.glyph_mask(c)
        if m is None:
            return None
        dy = self.mark_dy * (0.3 if c.p else 1.0)
        if dy:
            m = shift(m, 0, dy)
        col = self.sym_color(c)
        if c.p and self.glyph in ("glow", "dots"):
            col = lighten(col, 0.25)
        return GLYPHS_STYLE[self.glyph](self, c, m, col)

    # -- pixel path -----------------------------------------------------------------------------------------
    def render_pixel(self, c):
        g = self.pgrid
        tw, th = c.tw // g, c.th // g

        def small(m):
            return m.resize((tw, th), Image.BOX).point(lambda v: 255 if v >= 128 else 0)

        hi, mid, lo = self.PAL.get(c.role, self.PAL["main"])
        OUT = self.OUT + (255,)
        if c.p:
            hi, mid, lo = mid, lo, darkc(lo, 0.35)
        Ms = small(self.shape(c))
        inner = Ms.filter(ImageFilter.MinFilter(3))
        lit_m = ImageChops.multiply(inner, ImageChops.multiply(ImageChops.offset(inner, -1, -1), ImageChops.offset(inner, -1, 0)))
        hl = ImageChops.subtract(lit_m, ImageChops.offset(lit_m, 1, 1))
        im = Image.new("RGBA", (tw, th), (0, 0, 0, 0))

        def lay(mask, col):
            im.paste(Image.new("RGBA", (tw, th), tuple(col[:3]) + (255,)), (0, 0), mask)
        lay(Ms, OUT)
        lay(inner, lo)
        if self.dither:
            band = ImageChops.subtract(inner, lit_m)
            chk = Image.new("L", (tw, th), 0)
            cp = chk.load()
            for y in range(th):
                for x in range(tw):
                    if (x + y) % 2 == 0:
                        cp[x, y] = 255
            lay(ImageChops.multiply(band, chk), mid)
        lay(lit_m, mid)
        if not c.p:
            lay(hl, hi)
        if c.kind == "stick" and self.stick_style != "none":
            ring = ImageChops.subtract(small(self.shape(c, 22)), small(self.shape(c, 30)))
            lay(ring, lo)
        ink = self.ink_p if (c.p and self.ink_p) else self.sym_color(c)
        if BLANK and c.kind in ("sh1", "sh2", "l3", "small", "face"):
            pass
        elif c.kind in ("sh1", "sh2", "l3"):
            fs = max(1, 8 // g) * (2 if c.kind == "l3" else 1)
            txt = c.part.upper()
            tm = Image.new("L", (tw, th), 0)
            td = ImageDraw.Draw(tm)
            x0 = tw // 2 - (len(txt) * 6 * fs - fs) // 2
            if c.kind == "l3":
                cy0 = th // 2
            else:
                bx = c.rect_box()
                cy0 = int((bx[1] + bx[3]) / 2 / g)
            cy0 += (fs if c.p else 0)
            for i, chh in enumerate(txt):
                for ry, row in enumerate(PIX_BITS[chh]):
                    for rx, v in enumerate(row):
                        if v == "X":
                            px, py = x0 + i * 6 * fs + rx * fs, cy0 - (7 * fs) // 2 + ry * fs
                            td.rectangle([px, py, px + fs - 1, py + fs - 1], fill=255)
            lay(tm, ink)
        else:
            m = self.glyph_mask(c)
            if m is not None:
                gm = small(m)
                if c.p:
                    gm = ImageChops.offset(gm, 0, max(1, 8 // g))
                lay(gm, ink)
        if c.kind == "base":
            lay(small(self.shape(c, 30)), darkc(lo, 0.6))
        return im.resize((c.tw, c.th), Image.NEAREST)


def form(**kw):
    s = FormSkin(**kw)
    if not s.swatch:
        src = s.PAL if s.pgrid else s.P
        cols = []
        for key in ("main", "cross", "circle", "square", "triangle", "dpad", "shoulder"):
            if key in src and len(cols) < 4:
                v = src[key]
                col = v[1] if s.pgrid else v[0]
                if hexc(col) not in cols:
                    cols.append(hexc(col))
        for key in ("cross", "circle", "square", "triangle"):
            col = s.sym_colors.get(key)
            if col and hexc(col) not in cols and len(cols) < 4:
                cols.append(hexc(col))
        if hexc(s.mark_col) not in cols and len(cols) < 4:
            cols.append(hexc(s.mark_col))
        while len(cols) < 4:
            cols.append(cols[-1])
        s.swatch = cols[:4]
    SKINS.append(s)
    return s


# ----------------------------------------------------------------------------------------------------------------
# the form skins (pad- prefix): every skin differs in at least two of button shape / material / glyph style
# ----------------------------------------------------------------------------------------------------------------


def pr(col, a=0.28, b=0.22):
    return (lighten(col, a), darkc(col, b))


def faces(cols, a=0.28, b=0.22, **extra):
    P = {}
    for part, col in zip(FACE, cols):
        P[part] = pr(col, a, b)
    P.update(extra)
    return P


def scol(cols):
    return dict(zip(FACE, cols))


W = (255, 255, 255)
BLUE, RED, PINK, GREEN = (64, 132, 240), (236, 70, 84), (240, 120, 200), (60, 196, 140)
NEON4 = ((60, 220, 255), (255, 70, 110), (255, 60, 220), (90, 255, 160))
PASTEL4 = ((130, 170, 230), (240, 140, 150), (240, 170, 214), (130, 210, 170))
VIVID4 = ((40, 130, 255), (255, 60, 70), (255, 206, 40), (50, 205, 110))

# ---- line, glyph-only and ghost ------------------------------------------------------------------------------

form(id="pad-mono-line", name="Mono Line", desc="Hairline outlines with no fill at all: thin monoline buttons that vanish into the game.",
     cats=["clean"], pro=False, featured=True, press="grow", opacity=0.9, backdrop=((30, 36, 52), (52, 62, 90)), scene=2,
     mat="outline", face="circle", dpad="arm", sh="rect", small="rect", stick_style="ring", base_style="none", glyph="solid",
     sym_thick=0.55, line=3.2, P={"main": ((236, 240, 248),) * 2}, mark_col=(236, 240, 248))

form(id="pad-hex-frame", name="Hex Frame", desc="Amber double-line hexagon frames with hollow outline symbols on a faint tinted glass.",
     cats=["clean", "retro"], pro=False, press="shrink", opacity=0.92, backdrop=((34, 30, 26), (60, 50, 38)), scene=3,
     mat="outline", face="hex", dpad="tri", sh="slant", small="slant", stick_shape="hex", stick_style="ring", base_shape="hex",
     base_style="ticks", well_a=0, glyph="hollow", sym_thick=1.25, line=4.6, fill_a=0.13, rings=2,
     P={"main": ((255, 188, 64),) * 2}, mark_col=(255, 200, 96))

form(id="pad-diamond-wire", name="Diamond Wire", desc="Glowing wireframe diamonds and sharp chamfered bars drawn as thin luminous lines.",
     cats=["neon"], pro=False, press="glow", opacity=0.92, backdrop=((8, 24, 34), (14, 52, 66)), scene=2,
     mat="outline", face="diamond", dpad="dia", sh="chamfer", small="chamfer", stick_shape="diamond", stick_style="reticle", base_shape="octagon",
     base_style="cross", well_a=0, glyph="glow", sym_thick=0.7, line=2.6, glow=True, rings=2,
     P=faces(NEON4, 0.0, 0.0, main=((90, 240, 255),) * 2), sym_colors=scol(NEON4), mark_col=(90, 240, 255))

form(id="pad-blueprint", name="Blueprint", desc="Technical drafting sheet: white-blue construction lines, grid and dimension ticks on cyanotype blue.",
     cats=["retro", "clean"], pro=False, press="shrink", opacity=0.92, backdrop=((10, 36, 90), (20, 66, 140)), scene=4,
     mat="blueprint", face="octagon", dpad="key", sh="hexbar", small="hexbar", stick_shape="octagon", stick_style="reticle", base_shape="octagon",
     base_style="ticks", well_a=0, glyph="solid", sym_thick=0.5, sym_scale=0.9,
     P={"main": ((190, 230, 255), (18, 70, 160))}, mark_col=(214, 240, 255))

form(id="pad-glyph-bold", name="Bold Glyphs", desc="No button bodies at all, just huge white symbols with a hard shadow and open arrow chevrons.",
     cats=["clean", "accessibility"], pro=False, press="grow", opacity=0.96, backdrop=((46, 54, 76), (84, 96, 128)), scene=1,
     mat="none", face="circle", dpad="chev", sh="rect", small="rect", stick_style="ring", base_style="dots", well_a=0, glyph="shadow",
     sym_thick=1.6, sym_scale=1.3, P={"main": ((255, 255, 255),) * 2}, mark_col=(255, 255, 255))

form(id="pad-ghost-soft", name="Ghost Soft", desc="Barely-there frosted squares for big screens: faint fills, thin rims, nothing in the way.",
     cats=["clean"], pro=False, press="grow", opacity=0.9, backdrop=((46, 70, 96), (90, 120, 150)), scene=4,
     mat="ghost", face="squircle", dpad="capsule", sh="pill", small="pill", stick_shape="squircle", stick_style="ring", base_shape="squircle",
     base_style="none", glyph="solid", sym_thick=0.6, P={"main": ((255, 255, 255),) * 2}, mark_col=(255, 255, 255))

form(id="pad-ghost-shield", name="Ghost Shield", desc="Translucent shield-shaped keys with outlined symbols, kept quiet and see-through.",
     cats=["clean"], pro=False, press="shrink", opacity=0.9, backdrop=((34, 44, 70), (66, 84, 120)), scene=0,
     mat="ghost", face="shield", dpad="tri", sh="trap", small="trap", stick_shape="hex", stick_style="none", base_shape="hex",
     base_style="none", glyph="hollow", sym_thick=1.1, P={"main": ((190, 224, 255),) * 2}, mark_col=(210, 234, 255))

form(id="pad-terminal", name="Terminal Brackets", desc="Green phosphor console look: scanlined plates framed by square [ ] brackets and glowing marks.",
     cats=["retro", "neon"], pro=False, press="shrink", opacity=0.94, backdrop=((2, 14, 8), (8, 36, 22)), scene=1,
     mat="bracket", face="chamfer", dpad="key", sh="rect", small="rect", stick_shape="chamfer", stick_style="reticle", base_shape="chamfer",
     base_style="cross", well_a=0.5, glyph="glow", sym_thick=0.7,
     P={"main": ((96, 255, 150), (2, 34, 14))}, mark_col=(120, 255, 170))

# ---- neon, HUD and displays ----------------------------------------------------------------------------------

form(id="pad-neon-tube", name="Neon Tube", desc="Hexagonal glass-tube signs: bent neon rings in pink, cyan and mint with a humming glow.",
     cats=["neon"], pro=True, press="glow", opacity=0.95, backdrop=((12, 8, 28), (36, 14, 60)), scene=2,
     mat="neon", face="hex", dpad="sector", sh="hexbar", small="hexbar", stick_shape="hex", stick_style="ring", base_shape="hex",
     base_style="well", well_a=0.5, glyph="glow", sym_thick=0.8,
     P=faces(NEON4, 0, 0, main=((255, 70, 190),) * 2), sym_colors=scol(NEON4), mark_col=(255, 90, 200))

form(id="pad-neon-diamond", name="Neon Diamond", desc="Slanted diamond neon plates in electric blue and amber, with tube-outlined symbols.",
     cats=["neon"], pro=True, press="glow", opacity=0.95, backdrop=((6, 12, 34), (14, 28, 70)), scene=2,
     mat="neon", face="diamond", dpad="dia", sh="slant", small="slant", stick_shape="diamond", stick_style="reticle", base_shape="diamond",
     base_style="none", well_a=0, glyph="hollow", sym_thick=1.2,
     P={"main": ((70, 160, 255),) * 2, "circle": ((255, 170, 40),) * 2, "triangle": ((255, 170, 40),) * 2}, mark_col=(110, 190, 255))

form(id="pad-hud-reticle", name="HUD Reticle", desc="Cyberpunk targeting rings with broken arcs, compass ticks and corner brackets in glowing cyan.",
     cats=["neon"], pro=True, featured=True, press="glow", opacity=0.94, backdrop=((4, 18, 26), (8, 46, 62)), scene=2,
     mat="hud", face="circle", dpad="capsule", sh="chamfer", small="chamfer", stick_style="reticle", base_style="ticks", well_a=0.35, glow=True,
     glyph="solid", sym_thick=0.8, P={"main": ((90, 232, 255), (0, 52, 74)), "triangle": ((120, 255, 190), (0, 70, 48))},
     sym_colors={"triangle": (150, 255, 210)}, mark_col=(190, 246, 255))

form(id="pad-hud-amber", name="HUD Amber", desc="Cockpit-style amber octagon brackets with LED-dot symbols, like a flight-deck display.",
     cats=["neon", "retro"], pro=True, press="glow", opacity=0.94, backdrop=((20, 12, 4), (52, 30, 8)), scene=3,
     mat="hud", face="octagon", dpad="key", sh="arch", small="arch", stick_shape="octagon", stick_style="ring", base_shape="octagon",
     base_style="ticks", well_a=0.3, glow=True, glyph="dots", sym_thick=1.35, led=13,
     P={"main": ((255, 184, 60), (70, 38, 0))}, mark_col=(255, 196, 80))

form(id="pad-vfd-mesh", name="VFD Display", desc="Vacuum-fluorescent teal glass behind a fine grid, with softly glowing segment-bright symbols.",
     cats=["retro", "neon"], pro=False, press="glow", opacity=0.94, backdrop=((4, 20, 24), (10, 52, 58)), scene=1,
     mat="mesh", face="squircle", dpad="dot", sh="rect", small="rect", stick_shape="squircle", stick_style="concave", base_shape="squircle",
     base_style="dots", well_a=0.5, glyph="glow", sym_thick=0.85, P={"main": ((96, 255, 232), (4, 70, 70))}, mark_col=(150, 255, 238))

form(id="pad-led-matrix", name="LED Matrix", desc="Black plates studded with a dot grid; every symbol is lit as amber LED pixels.",
     cats=["retro"], pro=False, press="glow", opacity=0.95, backdrop=((16, 16, 20), (38, 34, 30)), scene=3,
     mat="ledmatrix", face="circle", dpad="dot", sh="pill", small="pill", stick_style="ring", base_style="ticks", well_a=0.55,
     glyph="dots", sym_thick=1.4, led=14, P={"main": ((255, 176, 48), (60, 40, 10))}, mark_col=(255, 176, 48))

# ---- pixel art -----------------------------------------------------------------------------------------------

form(id="pad-pixel-hex", name="Pixel Hex", desc="Chunky hexagon pixel buttons with dithered shading in a cool teal 4-tone palette.",
     cats=["retro"], pro=False, press="shrink", opacity=0.96, backdrop=((8, 30, 36), (16, 56, 66)), scene=1, pixel=True, pgrid=8,
     face="hex", dpad="dia", sh="chamfer", small="chamfer", stick_shape="circle", stick_style="ring", dither=True, sym_thick=1.1, sym_scale=0.95,
     OUT=(8, 22, 26), PAL={"main": ((160, 252, 232), (64, 192, 182), (22, 100, 110))}, mark_col=(244, 255, 252))

form(id="pad-pixel-1bit", name="Pixel 1-Bit", desc="Strict black-and-white pixels with checkerboard dithering, like a monochrome handheld.",
     cats=["retro", "accessibility"], pro=False, featured=True, press="shrink", opacity=0.96, backdrop=((150, 154, 140), (92, 96, 88)), scene=1,
     pixel=True, pgrid=8, face="squircle", dpad="key", sh="rect", small="rect", stick_shape="squircle", stick_style="ring", dither=True,
     sym_thick=1.1, sym_scale=0.95, OUT=(0, 0, 0), PAL={"main": ((255, 255, 255), (255, 255, 255), (0, 0, 0))},
     mark_col=(0, 0, 0), ink_p=(255, 255, 255))

form(id="pad-pixel-16bit", name="Pixel 16-Bit", desc="Fine 16-bit sprites: bevelled chamfered keys with rich colour ramps and crisp highlights.",
     cats=["retro"], pro=True, press="spring", opacity=0.96, backdrop=((26, 20, 54), (60, 38, 96)), scene=4, pgrid=4,
     face="chamfer", dpad="arm", sh="trap", small="trap", stick_shape="octagon", stick_style="ring", dither=False, sym_thick=1.0,
     OUT=(14, 10, 30), PAL={"main": ((236, 236, 252), (150, 156, 196), (74, 78, 126)), "cross": ((150, 196, 255), (60, 120, 240), (30, 52, 150)),
                            "circle": ((255, 160, 150), (236, 60, 70), (130, 20, 50)), "square": ((255, 196, 236), (236, 100, 190), (126, 30, 100)),
                            "triangle": ((170, 255, 190), (60, 210, 120), (20, 106, 70)), "dpad": ((130, 136, 180), (80, 84, 126), (44, 46, 80))},
     mark_col=(255, 255, 255))

form(id="pad-pixel-diamond", name="Pixel Diamond", desc="Hot-orange pixel diamonds and slanted bars with a hard-edged retro console outline.",
     cats=["retro"], pro=False, press="spring", opacity=0.96, backdrop=((46, 14, 10), (96, 34, 18)), scene=3, pixel=True, pgrid=8,
     face="diamond", dpad="tri", sh="slant", small="slant", stick_shape="diamond", stick_style="ring", dither=False, sym_thick=1.1, sym_scale=0.9,
     OUT=(30, 8, 6), PAL={"main": ((255, 226, 150), (250, 150, 50), (160, 56, 24))}, mark_col=(60, 18, 8))

# ---- metal and mechanical ------------------------------------------------------------------------------------

form(id="pad-titanium-hex", name="Titanium Hex", desc="Cold brushed titanium hexagons and chamfered bars with laser-etched inlaid symbols.",
     cats=["material"], pro=True, press="shrink", opacity=0.95, backdrop=((40, 46, 58), (82, 92, 108)), scene=2,
     mat="brushed", face="hex", dpad="sector", sh="chamfer", small="chamfer", stick_shape="hex", stick_style="concave", base_shape="hex",
     base_style="ticks", well_a=0.55, glyph="inlay", sym_thick=0.9, P={"main": ((200, 208, 220), (92, 102, 118)), "base": ((150, 160, 176), (72, 80, 94))},
     acc=(150, 214, 255), mark_col=(236, 246, 255))

form(id="pad-copper-cog", name="Copper Cog", desc="Steampunk hammered-copper gears and rivets with verdigris-green embossed symbols.",
     cats=["material", "retro"], pro=True, featured=True, press="spring", opacity=0.95, backdrop=((44, 28, 18), (96, 62, 36)), scene=3,
     mat="hammered", face="cog", dpad="dot", sh="trap", small="trap", stick_shape="cog", stick_style="dome", base_shape="cog",
     base_style="dots", well_a=0.5, glyph="emboss", sym_thick=1.0,
     P={"main": ((236, 150, 96), (120, 56, 30)), "base": ((190, 112, 70), (90, 40, 22))}, acc=(250, 200, 140), mark_col=(96, 214, 178))

form(id="pad-gunmetal-stencil", name="Gunmetal Stencil", desc="Hazard-striped dark gunmetal plates with orange stencil marks and bridged cut-out letters.",
     cats=["material"], pro=False, press="spring", opacity=0.96, backdrop=((26, 28, 32), (56, 60, 68)), scene=3,
     mat="flat", face="octagon", dpad="tri", sh="slant", small="slant", stick_shape="octagon", stick_style="ridges", base_shape="octagon",
     base_style="ticks", well_a=0.5, glyph="stencil", sym_thick=1.25, line=3.4, edge=(12, 12, 14), patterns={p: "stripes" for p in PARTS},
     P={"main": ((92, 96, 106), (44, 46, 54))}, acc=(255, 150, 40), mark_col=(255, 150, 40))

form(id="pad-bakelite-radio", name="Bakelite Radio", desc="Cream enamel knobs ringed in chrome, with engraved maroon marks like a 1950s tabletop radio.",
     cats=["retro", "material"], pro=True, press="grow", opacity=0.96, backdrop=((54, 34, 26), (110, 76, 56)), scene=3,
     mat="enamel", face="circle", dpad="capsule", sh="arch", small="arch", stick_style="pointer", base_style="ticks", well_a=0.55,
     glyph="engrave", sym_thick=1.0, P={"main": ((244, 230, 198), (162, 128, 90))}, acc=(170, 38, 30), mark_col=(120, 40, 24))

form(id="pad-typewriter", name="Typewriter Keys", desc="Round cream keycaps with raised sides and embossed black glyphs, like an old typewriter.",
     cats=["retro"], pro=False, press="spring", opacity=0.96, backdrop=((38, 36, 34), (80, 76, 70)), scene=3,
     mat="keycap", face="circle", dpad="dot", sh="rect", small="rect", stick_style="concave", base_style="dots", well_a=0.55, glyph="emboss",
     mark_dy=-6, sym_thick=0.9, sym_scale=0.85, P={"main": ((250, 244, 226), (196, 186, 160)), "base": ((74, 72, 70), (36, 36, 36))},
     acc=(214, 206, 180), mark_col=(36, 34, 36))

form(id="pad-mech-keys", name="Mech Keys", desc="Graphite mechanical-keyboard keycaps with dished tops and bright RGB-lit legends.",
     cats=["material"], pro=False, press="spring", opacity=0.96, backdrop=((20, 22, 30), (46, 50, 66)), scene=2,
     mat="keycap", face="squircle", dpad="key", sh="rect", small="rect", stick_shape="squircle", stick_style="dimples", base_shape="squircle",
     base_style="ticks", well_a=0.5, glyph="glow", mark_dy=-6, sym_thick=0.75, sym_scale=0.8,
     P={"main": ((84, 88, 102), (38, 40, 52)), "base": ((50, 54, 66), (26, 28, 36))}, sym_colors=scol(NEON4), mark_col=(120, 230, 255), acc=(120, 230, 255))

form(id="pad-microswitch", name="Clear Switch", desc="Clear-lens arcade pushbuttons lit from below, with triangular rocker arrows and a red ball stick.",
     cats=["retro"], pro=True, press="spring", opacity=0.96, backdrop=((20, 20, 28), (46, 36, 62)), scene=2,
     mat="cap", face="circle", dpad="tri", sh="arch", small="arch", stick_style="dome", base_style="ticks", well_a=0.6, glyph="engrave", sym_scale=0.7, sym_thick=0.8,
     P={"main": ((190, 196, 210), (60, 64, 78)), "cross": ((60, 130, 255), (20, 60, 190)), "circle": ((255, 70, 80), (180, 20, 40)),
        "square": ((255, 214, 70), (200, 140, 10)), "triangle": ((80, 230, 120), (20, 140, 60)), "stick": ((255, 80, 84), (150, 10, 24))},
     mark_col=(250, 252, 255), sym_colors={})


# ---- natural materials and textiles --------------------------------------------------------------------------

form(id="pad-bamboo-brush", name="Bamboo Brush", desc="Green bamboo squircles with node rings and symbols painted in dry black brush strokes.",
     cats=["material"], pro=False, press="shrink", opacity=0.95, backdrop=((30, 44, 26), (74, 98, 52)), scene=1,
     mat="wood", bamboo=True, wood=((116, 140, 58), (214, 226, 128)), face="squircle", dpad="capsule", sh="pill", small="pill",
     stick_shape="squircle", stick_style="ridges", base_style="well", well_a=0.5, glyph="brush", sym_thick=1.1,
     P={"main": ((255, 255, 255), (160, 140, 120))}, mark_col=(34, 40, 18))

form(id="pad-slate-rune", name="Slate Rune", desc="Rough-hewn slate hexagons with chiselled symbols, speckled grain and chipped edges.",
     cats=["material"], pro=True, press="shrink", opacity=0.96, backdrop=((26, 30, 36), (58, 64, 74)), scene=0,
     mat="stone", face="hex", dpad="tri", sh="trap", small="trap", stick_shape="hex", stick_style="concave", base_shape="hex",
     base_style="well", well_a=0.55, glyph="engrave", sym_thick=1.1, P={"main": ((136, 144, 154), (62, 68, 78))}, mark_col=(30, 34, 42))

form(id="pad-marble-gold", name="Marble and Gold", desc="White veined marble octagons trimmed with gold lines and gold-inlaid symbols.",
     cats=["material"], pro=True, press="shrink", opacity=0.96, backdrop=((30, 28, 34), (70, 66, 76)), scene=0,
     mat="marble", face="octagon", dpad="sector", sh="chamfer", small="chamfer", stick_shape="octagon", stick_style="dome", base_shape="octagon",
     base_style="ticks", well_a=0.45, glyph="inlay", sym_thick=0.9, line=3.2,
     P={"main": ((250, 250, 252), (118, 126, 148))}, acc=(206, 160, 62), mark_col=(222, 178, 76))

form(id="pad-leather-stitch", name="Stitched Leather", desc="Saddle-brown pebbled leather squircles with cream hand stitching around every edge.",
     cats=["material"], pro=False, press="spring", opacity=0.96, backdrop=((40, 28, 22), (84, 58, 44)), scene=3,
     mat="leather", face="squircle", dpad="arm", sh="rect", small="rect", stick_shape="squircle", stick_style="dimples", base_shape="squircle",
     base_style="well", well_a=0.5, glyph="stitch", sym_thick=1.1, P={"main": ((158, 98, 60), (82, 44, 24))}, acc=(240, 210, 150), mark_col=(240, 210, 150))

form(id="pad-leather-racing", name="Racing Leather", desc="Perforated black leather chamfered keys with red contrast stitching and red inlaid symbols.",
     cats=["material"], pro=True, press="spring", opacity=0.96, backdrop=((14, 14, 18), (44, 22, 24)), scene=2,
     mat="leather", perf=True, face="chamfer", dpad="capsule", sh="arch", small="arch", stick_shape="chamfer", stick_style="ridges", base_shape="chamfer",
     base_style="ticks", well_a=0.5, glyph="inlay", sym_thick=1.0, P={"main": ((70, 70, 78), (24, 24, 28))}, acc=(236, 56, 52), mark_col=(244, 72, 66))

form(id="pad-felt-patch", name="Felt Patches", desc="Soft felt flowers with blanket-stitched edges and cross-stitched symbols in cheerful colors.",
     cats=["playful"], pro=False, press="spring", opacity=0.96, backdrop=((60, 92, 108), (104, 138, 150)), scene=1,
     mat="felt", face="flower", dpad="dot", sh="pill", small="pill", stick_shape="flower", stick_style="ring", base_style="dots", well_a=0.2,
     glyph="stitch", sym_thick=1.2, P=faces(((92, 140, 220), (230, 100, 110), (240, 160, 210), (110, 200, 150)), 0.1, 0.25, main=((238, 232, 212), (190, 180, 150))),
     acc=(255, 248, 232), mark_col=(255, 248, 232))

form(id="pad-embroidery", name="Embroidery", desc="Navy cloth shields with gold satin-stitched borders and thread-sewn colored symbols.",
     cats=["playful", "material"], pro=True, press="shrink", opacity=0.96, backdrop=((14, 20, 44), (34, 46, 88)), scene=4,
     mat="embroidery", face="shield", dpad="tri", sh="trap", small="trap", stick_shape="shield", stick_style="ring", base_shape="hex",
     base_style="ticks", well_a=0.5, glyph="satin", sym_thick=1.2, P={"main": ((58, 78, 134), (24, 34, 74))},
     acc=(250, 214, 100), sym_colors=scol(((120, 200, 255), (255, 120, 130), (255, 170, 230), (140, 230, 160))), mark_col=(250, 214, 100))

form(id="pad-knit-wool", name="Knit Wool", desc="Chunky knitted-wool squares with visible V stitches and fuzzy edges in warm red and cream.",
     cats=["playful", "material"], pro=True, press="spring", opacity=0.96, backdrop=((54, 40, 36), (110, 84, 72)), scene=3,
     mat="knit", face="squircle", dpad="dot", sh="pill", small="pill", stick_shape="circle", stick_style="dimples", base_shape="circle",
     base_style="dots", well_a=0.4, glyph="solid", sym_thick=1.25,
     P=faces(((70, 130, 200), (214, 72, 64), (232, 132, 168), (84, 170, 110)), 0.2, 0.2, main=((226, 214, 190), (160, 140, 110))), mark_col=(255, 246, 226))

# ---- soft and playful ----------------------------------------------------------------------------------------

form(id="pad-gummy-blob", name="Gummy Blobs", desc="Wobbly translucent gummy candies with a bright inner glow and glossy highlights.",
     cats=["playful"], pro=True, press="spring", opacity=0.96, backdrop=((44, 20, 60), (100, 44, 110)), scene=2,
     mat="inflate", infl=20, alpha=0.78, spec=0.8, wob=5.0, face="circle", dpad="capsule", sh="pill", small="pill", stick_style="dome", base_style="well", well_a=0.35,
     glyph="shadow", sym_thick=1.15,
     P=faces(((70, 150, 255), (255, 80, 110), (255, 130, 220), (70, 220, 140)), 0.35, 0.25, main=((200, 130, 255), (120, 50, 200))), mark_col=(255, 255, 255))

form(id="pad-marshmallow", name="Marshmallow", desc="Puffy pastel pillow keys with soft shading and no hard edges anywhere.",
     cats=["playful", "clean"], pro=False, press="grow", opacity=0.96, backdrop=((250, 226, 232), (232, 208, 244)), scene=0,
     mat="inflate", infl=24, alpha=1.0, spec=0.22, face="circle", dpad="key", sh="pill", small="pill", stick_shape="circle", stick_style="dome",
     base_shape="circle", base_style="none", well_a=0.2, glyph="solid", sym_thick=1.3,
     P={"main": ((255, 244, 244), (236, 208, 220)), "base": ((240, 226, 236), (214, 190, 210))}, sym_colors=scol(PASTEL4), mark_col=(214, 136, 160))

form(id="pad-clay-thumb", name="Clay Thumbprint", desc="Hand-modelled matte clay in earthy colors, lumpy surfaces and embossed symbols.",
     cats=["material", "playful"], pro=False, press="shrink", opacity=0.96, backdrop=((64, 48, 38), (112, 88, 70)), scene=3,
     mat="clay", wob=3.0, face="circle", dpad="capsule", sh="pill", small="pill", stick_style="dome", base_style="well", well_a=0.4, glyph="emboss", sym_thick=1.15,
     P=faces(((60, 150, 160), (214, 100, 70), (230, 180, 70), (120, 170, 110)), 0.2, 0.25, main=((224, 204, 174), (170, 144, 112))), mark_col=(250, 240, 220))

form(id="pad-sticker-pop", name="Sticker Pop", desc="Die-cut vinyl stickers with a thick white border, glossy sheen bands and bold colors.",
     cats=["playful"], pro=False, press="spring", opacity=0.97, backdrop=((38, 40, 84), (80, 60, 130)), scene=2,
     mat="sticker", face="circle", dpad="dot", sh="pill", small="pill", stick_style="dome", base_style="dots", well_a=0.3, glyph="pop", sym_thick=1.0, sym_scale=0.95,
     P=faces(VIVID4, 0.2, 0.2, main=((255, 120, 190), (220, 60, 140))), mark_col=(255, 255, 255))

form(id="pad-pop-burst", name="Pop Burst", desc="Comic starbursts with halftone shading, hard black outlines and shouting white symbols.",
     cats=["playful"], pro=True, press="spring", opacity=0.97, backdrop=((255, 226, 90), (255, 160, 60)), scene=0,
     mat="halftone", face="burst", dpad="tri", sh="slant", small="slant", stick_shape="burst", stick_style="ring", base_shape="octagon", base_style="none",
     well_a=0, glyph="pop", sym_thick=1.2, P=faces(VIVID4, 0.0, 0.4, main=((255, 214, 60), (200, 100, 20))), acc=(12, 12, 16), mark_col=(255, 255, 255))

form(id="pad-soap-bubble", name="Soap Bubble", desc="Transparent soap bubbles with shimmering rainbow rims and tiny light reflections.",
     cats=["playful"], pro=True, press="grow", opacity=0.95, backdrop=((14, 28, 58), (34, 70, 110)), scene=4,
     mat="bubble", face="circle", dpad="sector", sh="pill", small="pill", stick_style="ring", base_style="dots", well_a=0, glyph="hollow", sym_thick=1.1,
     P={"main": ((255, 255, 255),) * 2}, mark_col=(244, 252, 255))

form(id="pad-toy-bricks", name="Toy Bricks", desc="Glossy plastic building bricks with round studs on top in classic primary colors.",
     cats=["playful"], pro=False, press="spring", opacity=0.97, backdrop=((40, 54, 96), (90, 120, 170)), scene=4,
     mat="toybrick", face="squircle", dpad="key", sh="rect", small="rect", stick_shape="squircle", stick_style="none", base_shape="squircle",
     base_style="none", well_a=0, glyph="solid", mark_dy=-8, sym_thick=1.0, sym_scale=0.62,
     P=faces(VIVID4, 0.12, 0.25, main=((240, 240, 244), (170, 174, 188))), sym_colors={"cross": W, "circle": W, "square": (84, 56, 0), "triangle": W}, mark_col=(40, 44, 60))

# ---- paper, print and ink ------------------------------------------------------------------------------------

form(id="pad-paper-cut", name="Paper Cut", desc="Layered cut-paper hexagons stacked three deep, each casting a soft shadow on the one below.",
     cats=["material", "playful"], pro=True, press="shrink", opacity=0.96, backdrop=((236, 226, 210), (214, 198, 176)), scene=3,
     mat="paper", face="hexf", dpad="key", sh="trap", small="trap", stick_shape="hexf", stick_style="ring", base_shape="hexf", base_style="none", well_a=0,
     glyph="shadow", sym_thick=1.2, P=faces(((70, 150, 190), (236, 110, 100), (246, 190, 80), (110, 180, 130)), 0.4, 0.3, main=((250, 236, 210), (200, 130, 100))),
     mark_col=(255, 255, 255))

form(id="pad-origami", name="Origami", desc="Folded-paper gems made of flat triangular facets with crisp fold lines and thin white marks.",
     cats=["material", "clean"], pro=False, press="spring", opacity=0.96, backdrop=((242, 238, 232), (214, 224, 232)), scene=4,
     mat="crystal", facets=8, fac_a=0.26, table=False, face="diamond", dpad="tri", sh="trap", small="trap", stick_shape="diamond", stick_style="none",
     base_shape="diamond", base_style="none", well_a=0, glyph="hollow", sym_thick=0.9,
     P=faces(((70, 150, 220), (240, 90, 100), (250, 190, 70), (90, 190, 140)), 0.2, 0.15, main=((236, 238, 244), (170, 180, 200))), mark_col=(255, 255, 255))

form(id="pad-riso-print", name="Riso Print", desc="Two-ink risograph octagons: pink and blue layers slightly off-register with grainy halftone symbols.",
     cats=["playful"], pro=False, press="shrink", opacity=0.96, backdrop=((248, 240, 222), (234, 222, 196)), scene=3,
     mat="riso", face="octagon", dpad="capsule", sh="pill", small="pill", stick_shape="octagon", stick_style="ring", base_shape="octagon", base_style="none", well_a=0,
     glyph="halftone", sym_thick=1.5, P={"main": ((255, 72, 160), (0, 120, 200))}, acc=(0, 120, 200), mark_col=(24, 40, 140))

form(id="pad-woodcut", name="Woodcut", desc="Black linocut hexagons carved with parallel gouge lines and bold cream symbols.",
     cats=["retro"], pro=False, press="shrink", opacity=0.96, backdrop=((72, 58, 46), (130, 108, 84)), scene=3,
     mat="carve", face="hexf", dpad="arm", sh="rect", small="rect", stick_shape="hexf", stick_style="ring", base_shape="hexf", base_style="ticks", well_a=0.3,
     glyph="solid", sym_thick=1.15, P={"main": ((244, 236, 214), (22, 20, 20))}, mark_col=(244, 236, 214))

form(id="pad-enso-brush", name="Enso Brush", desc="Sumi ink circles brushed in a single dry stroke on rice paper, with one red seal.",
     cats=["clean", "playful"], pro=True, featured=True, press="grow", opacity=0.96, backdrop=((238, 230, 212), (220, 210, 188)), scene=4,
     mat="brush", face="circle", dpad="capsule", sh="rect", small="rect", stick_style="ring", base_style="none", well_a=0, glyph="brush", sym_thick=1.2,
     P={"main": ((24, 24, 30),) * 2}, sym_colors={"circle": (200, 40, 40)}, mark_col=(24, 24, 30))

form(id="pad-watercolor", name="Watercolor", desc="Loose watercolor washes with dark pooled edges and paper grain, symbols dabbed in ink.",
     cats=["playful"], pro=True, press="grow", opacity=0.95, backdrop=((244, 242, 236), (226, 228, 232)), scene=4,
     mat="watercolor", wob=4.0, face="circle", dpad="dot", sh="pill", small="pill", stick_style="ring", base_style="none", well_a=0, glyph="wobble", sym_thick=1.1,
     P=faces(((70, 140, 220), (240, 90, 110), (200, 110, 220), (80, 190, 130)), 0.1, 0.35, main=((250, 190, 90), (210, 110, 60))), mark_col=(40, 44, 80))

form(id="pad-chalkboard", name="Chalkboard", desc="Slate boards with rough chalk outlines, smudges and chalk-dust symbols in pastel colors.",
     cats=["playful", "retro"], pro=False, press="grow", opacity=0.95, backdrop=((26, 40, 36), (48, 68, 62)), scene=1,
     mat="chalk", face="circle", dpad="arm", sh="rect", small="rect", stick_style="ring", base_style="ticks", well_a=0, glyph="chalk", sym_thick=1.0,
     P={"main": ((244, 246, 240), (34, 46, 44))}, sym_colors=scol(((150, 214, 255), (255, 160, 170), (255, 210, 120), (170, 240, 170))), mark_col=(244, 246, 240))

form(id="pad-ink-sketch", name="Ink Sketch", desc="Hand-drawn pencil-and-ink buttons with wobbly outlines, cross-hatch shading and doodled symbols.",
     cats=["playful"], pro=True, featured=True, press="spring", opacity=0.96, backdrop=((214, 204, 184), (238, 230, 210)), scene=4,
     mat="sketch", face="circle", dpad="arm", sh="rect", small="rect", stick_style="none", base_style="none", well_a=0, glyph="wobble", sym_thick=0.95,
     P={"main": ((252, 248, 236), (40, 40, 52))}, mark_col=(40, 40, 52))

form(id="pad-rubber-stamp", name="Rubber Stamp", desc="Uneven red rubber-stamp ink rings pressed on paper, with worn and speckled marks.",
     cats=["playful", "retro"], pro=False, press="shrink", opacity=0.95, backdrop=((236, 226, 204), (216, 202, 174)), scene=3,
     mat="stamp", face="circle", dpad="key", sh="rect", small="rect", stick_style="ring", base_style="none", well_a=0, glyph="stamp", sym_thick=1.2,
     P={"main": ((196, 38, 52),) * 2, "cross": ((40, 70, 170),) * 2}, sym_colors={"cross": (40, 70, 170)}, mark_col=(196, 38, 52))

# ---- glass, crystal and ceramic ------------------------------------------------------------------------------

form(id="pad-glass-hex", name="Glass Hex", desc="Frosted blue glassmorphism hexagons with bright rims, inner highlights and fine white marks.",
     cats=["material", "clean"], pro=False, press="grow", opacity=0.92, backdrop=((28, 50, 96), (74, 110, 170)), scene=4,
     mat="glass", ga=0.28, face="hexf", dpad="capsule", sh="trap", small="trap", stick_shape="hexf", stick_style="concave", base_shape="hexf",
     base_style="ticks", well_a=0.25, glyph="solid", sym_thick=0.65, P={"main": ((170, 220, 255), (60, 110, 200))}, mark_col=(255, 255, 255))

form(id="pad-crystal-gem", name="Crystal Gems", desc="Cut gemstones with faceted sides and a bright table: sapphire, ruby, amethyst and emerald.",
     cats=["material"], pro=True, featured=True, press="spring", opacity=0.96, backdrop=((20, 16, 44), (54, 34, 90)), scene=2,
     mat="crystal", facets=12, fac_a=0.4, table=True, face="octagon", dpad="dia", sh="chamfer", small="chamfer", stick_shape="octagon", stick_style="dome",
     base_shape="octagon", base_style="ticks", well_a=0.5, glyph="solid", sym_thick=0.9, sym_scale=0.9,
     P=faces(((60, 110, 235), (230, 40, 80), (170, 80, 230), (30, 190, 120)), 0.3, 0.3, main=((196, 226, 250), (96, 136, 190))), mark_col=(255, 255, 255))

form(id="pad-kintsugi", name="Kintsugi", desc="Glazed white porcelain discs mended with golden crack lines and gold-inlaid symbols.",
     cats=["material"], pro=True, press="shrink", opacity=0.96, backdrop=((30, 36, 48), (70, 80, 98)), scene=0,
     mat="ceramic", face="circle", dpad="sector", sh="arch", small="arch", stick_style="dome", base_style="ticks", well_a=0.4, glyph="inlay", sym_thick=0.9,
     P={"main": ((252, 250, 246), (196, 192, 186))}, acc=(214, 170, 70), mark_col=(222, 178, 76))

# ---- accessibility -------------------------------------------------------------------------------------------

form(id="pad-hc-shapes", name="Shape Coded", desc="Every button has its own silhouette, thick black outlines and color-blind-safe fills.",
     cats=["accessibility"], pro=False, press="spring", opacity=1.0, backdrop=((236, 238, 244), (200, 206, 220)), scene=4,
     mat="hc", face={"cross": "hexf", "circle": "circle", "square": "chamfer", "triangle": "shield", "*": "circle"}, dpad="arm", sh="rect", small="rect",
     stick_shape="octagon", stick_style="ridges", base_style="ticks", well_a=0.2, glyph="solid", sym_thick=1.25,
     P={"cross": ((86, 180, 233), (10, 10, 14)), "circle": ((230, 159, 0), (10, 10, 14)), "square": ((240, 228, 66), (10, 10, 14)),
        "triangle": ((0, 158, 115), (10, 10, 14)), "main": ((250, 250, 250), (10, 10, 14))},
     sym_colors={"cross": (10, 10, 14), "circle": (10, 10, 14), "square": (10, 10, 14), "triangle": (10, 10, 14)}, mark_col=(10, 10, 14))

form(id="pad-colorblind-safe", name="Pattern Colors", desc="Okabe-Ito color-blind-safe palette where every button also carries its own pattern.",
     cats=["accessibility"], pro=False, press="spring", opacity=1.0, backdrop=((44, 48, 60), (84, 90, 108)), scene=0,
     mat="flat", face="squircle", dpad="arm", sh="pill", small="pill", stick_shape="squircle", stick_style="grooves", base_shape="squircle",
     base_style="well", well_a=0.5, glyph="solid", sym_thick=1.15, line=3.4, edge=(10, 10, 14), patterns={"cross": "stripes", "circle": "dots", "square": "grid"},
     P=faces(((0, 114, 178), (213, 94, 0), (240, 228, 66), (0, 158, 115)), 0.12, 0.2, main=((236, 238, 244), (170, 176, 192))),
     sym_colors={"cross": W, "circle": W, "square": (10, 10, 14), "triangle": W}, mark_col=(30, 34, 44))

form(id="pad-big-label", name="Big Labels", desc="Oversized symbols and shoulder labels with huge triangle arrows, made for small screens and either hand.",
     cats=["accessibility"], pro=False, press="grow", opacity=1.0, backdrop=((20, 24, 40), (40, 48, 76)), scene=2,
     mat="flat", face="circle", dpad="tri", sh="rect", small="rect", stick_style="ring", base_style="ticks", well_a=0.5, glyph="shadow",
     sym_thick=1.4, sym_scale=1.32, line=4.2, edge=(255, 208, 60), P={"main": ((44, 52, 84), (22, 26, 48))}, acc=(255, 208, 60), mark_col=(255, 255, 255))

# ---- misc ----------------------------------------------------------------------------------------------------

form(id="pad-circuit-board", name="Circuit Board", desc="Green printed-circuit octagons with gold traces, solder pads and white silkscreen symbols.",
     cats=["retro", "material"], pro=False, press="shrink", opacity=0.96, backdrop=((6, 30, 22), (14, 66, 46)), scene=1,
     mat="circuit", face="octagon", dpad="arm", sh="hexbar", small="hexbar", stick_shape="octagon", stick_style="ring", base_shape="octagon",
     base_style="dots", well_a=0.4, glyph="solid", sym_thick=0.9, P={"main": ((34, 140, 90), (8, 60, 40))}, acc=(222, 182, 88), mark_col=(244, 248, 240))

form(id="pad-soft-ui", name="Soft UI", desc="Neumorphic buttons that rise from a pale surface with twin soft shadows and sink in when pressed.",
     cats=["clean"], pro=False, press="shrink", opacity=0.95, backdrop=((208, 216, 230), (226, 232, 242)), scene=4,
     mat="neumorph", face="circle", dpad="sector", sh="pill", small="pill", stick_style="concave", base_style="none", well_a=0, glyph="solid", sym_thick=0.9,
     P={"main": ((234, 238, 246), (200, 208, 224))}, sym_colors=scol(((96, 130, 210), (214, 100, 120), (160, 110, 210), (70, 170, 140))), mark_col=(110, 122, 150))

form(id="pad-art-deco", name="Art Deco", desc="Black-lacquer octagons with fanned gold sunburst lines, double gold borders and inlaid symbols.",
     cats=["retro", "material"], pro=True, press="shrink", opacity=0.96, backdrop=((14, 14, 20), (36, 30, 26)), scene=2,
     mat="deco", face="octagon", dpad="arm", sh="chamfer", small="chamfer", stick_shape="octagon", stick_style="ring", base_shape="octagon",
     base_style="ticks", well_a=0, glyph="inlay", sym_thick=0.9, P={"main": ((226, 184, 96), (90, 66, 20))}, acc=(226, 184, 96), mark_col=(236, 196, 108))


# ----------------------------------------------------------------------------------------------------------------
# packaging: previews, cover, gallery, ZIP
# ----------------------------------------------------------------------------------------------------------------

# ================================================================================================================
# WAVE 2: new shapes, glyph styles and materials (numpy helpers). Everything below stays procedural.
# ================================================================================================================
import numpy as np


def A(m):
    return np.asarray(m, dtype=np.float32) / 255.0


def LI(a):
    return Image.fromarray(np.clip(np.asarray(a) * 255.0, 0, 255).astype(np.uint8), "L")


def _img(arr):
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")


_GRID = {}


def _grid(c):
    k = (c.w, c.h)
    if k not in _GRID:
        yy, xx = np.mgrid[0:c.h, 0:c.w].astype(np.float32)
        _GRID[k] = (yy, xx)
    return _GRID[k]


def _cmap(t, stops):
    t = np.clip(t, 0, 1)
    ps = [p for p, _ in stops]
    out = np.zeros(t.shape + (3,), np.float32)
    for k in range(3):
        out[..., k] = np.interp(t, ps, [col[k] for _, col in stops])
    return out


def _hsv(h, s, v):
    h, s, v = np.broadcast_arrays(np.asarray(h, np.float32), np.asarray(s, np.float32), np.asarray(v, np.float32))
    h6 = (h % 1.0) * 6
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], -1) * 255


def _n(sk, c, cell, seed, k=2.2):
    def mk():
        a = A(smooth_noise(c.w, c.h, max(2, int(cell * S)), seed))
        return np.clip((a - 0.5) * k + 0.5, 0, 1)
    return sk.cache(("n", c.w, c.h, cell, seed, k), mk)


def _bb(M):
    x0, y0, x1, y1 = bbox_of(M)
    return x0, y0, x1, y1, (x0 + x1) / 2.0, (y0 + y1) / 2.0


def _pts_in(M, n, seed):
    rnd = random.Random(seed)
    x0, y0, x1, y1 = bbox_of(M)
    mp = M.load()
    out, tries = [], 0
    while len(out) < n and tries < n * 40:
        tries += 1
        x, y = rnd.uniform(x0, x1), rnd.uniform(y0, y1)
        if mp[min(int(x), M.width - 1), min(int(y), M.height - 1)] > 200:
            out.append((x, y))
    return out


def _vor(c, pts):
    yy, xx = _grid(c)
    best = np.full(yy.shape, 1e18, np.float32)
    idx = np.zeros(yy.shape, np.int32)
    for i, (px, py) in enumerate(pts):
        d = (xx - px) ** 2 + (yy - py) ** 2
        m = d < best
        best = np.where(m, d, best)
        idx = np.where(m, i, idx)
    return idx, np.sqrt(best)


def _polar(c, M, circular=False):
    yy, xx = _grid(c)
    x0, y0, x1, y1, cx, cy = _bb(M)
    rx, ry = max(1.0, (x1 - x0) / 2.0), max(1.0, (y1 - y0) / 2.0)
    dx, dy = (max(rx, ry), max(rx, ry)) if circular else (rx, ry)
    u, v = (xx - cx) / dx, (yy - cy) / dy
    return np.sqrt(u * u + v * v), np.arctan2(v, u), (x0, y0, x1, y1, cx, cy, rx, ry)


def fin(sk, c, M, fill, lo, bev=(0.4, 0.45, 3.2), rim=0.8, sh=(5, 7, 0.42), pdark=0.8, shcol=(0, 0, 0)):
    out = shadow_of(c, M, sh[0], sh[1], sh[2], 0, shcol) if sh else blank(c)
    f = tone(fill, pdark) if c.p else fill
    out = over(out, clip(f, M))
    if bev:
        out = over(out, bevel(M, bev[2], bev[0] * (0.4 if c.p else 1), bev[1]))
    if rim:
        out = over(out, paint(ring_of(M, 2.2), darkc(lo, 0.55), rim))
    return out


def gloss_top(c, M, a=0.3, k=1.9, ins=12):
    return paint(blur(ImageChops.multiply(shrink(M, ins), top_fade(c, k)), 2.4), (255, 255, 255), a * (0.45 if c.p else 1.0))


def _dashes(c, ring, period=8, duty=0.58, angle=45):
    return ImageChops.multiply(ring, stripe_mask(c, period, duty, angle))


FormSkin.gem = None
FormSkin.duo = None

# ---- new shapes ---------------------------------------------------------------------------------------------

_NEW_FACE = ("star", "scallop", "arch", "gothic", "pent", "sparkle", "octstar", "drop", "kite", "tile")
_face_shape_v1 = face_shape


def face_shape(c, name, R):
    if name not in _NEW_FACE:
        return _face_shape_v1(c, name, R)
    m = c.new()
    if R <= 0:
        return m
    d = ImageDraw.Draw(m)
    cx = cy = 128.0
    if name == "star":
        pts = [(cx + (R * 1.14 if i % 2 == 0 else R * 0.6) * math.cos(math.radians(-90 + 36 * i)),
                cy + (R * 1.14 if i % 2 == 0 else R * 0.6) * math.sin(math.radians(-90 + 36 * i))) for i in range(10)]
        return poly_mask(c, pts, 6)
    if name == "scallop":
        pts = []
        for i in range(240):
            a = math.radians(i * 1.5)
            rr = R * (0.91 + 0.085 * math.cos(a * 12))
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        return poly_mask(c, pts, 1.2)
    if name == "arch":
        d.rounded_rectangle([(cx - R * 0.8) * S, (cy - R * 1.02) * S, (cx + R * 0.8) * S, (cy + R * 0.94) * S], radius=R * 0.8 * S,
                            fill=255, corners=(True, True, False, False))
        return soften(m, 3)
    if name == "gothic":
        hw = R * 0.82
        r = hw * 1.7
        ys = cy - R * 0.12
        a_end = math.degrees(math.acos(-(r - hw) / r))
        left = [(cx + (r - hw) + r * math.cos(math.radians(a)), ys + r * math.sin(math.radians(a))) for a in np.linspace(180, 360 - a_end, 14)]
        right = [(cx - (r - hw) + r * math.cos(math.radians(a)), ys + r * math.sin(math.radians(a))) for a in np.linspace(360 - (180 - a_end), 360, 14)]
        pts = [(cx - hw, cy + R * 0.96)] + left + right + [(cx + hw, cy + R * 0.96)]
        return poly_mask(c, pts, 5)
    if name == "pent":
        return poly_mask(c, ngon(cx, cy + R * 0.04, R * 1.1, 5, -90), 6)
    if name == "sparkle":
        pts = [(cx + (R * 1.16 if i % 2 == 0 else R * 0.5) * math.cos(math.radians(-90 + 45 * i)),
                cy + (R * 1.16 if i % 2 == 0 else R * 0.5) * math.sin(math.radians(-90 + 45 * i))) for i in range(8)]
        return poly_mask(c, pts, 4)
    if name == "octstar":
        pts = [(cx + (R * 1.12 if i % 2 == 0 else R * 0.8) * math.cos(math.radians(-90 + 22.5 * i)),
                cy + (R * 1.12 if i % 2 == 0 else R * 0.8) * math.sin(math.radians(-90 + 22.5 * i))) for i in range(16)]
        return poly_mask(c, pts, 2.2)
    if name == "drop":
        r = R * 0.74
        d.ellipse([(cx - r) * S, (cy + R * 0.2 - r) * S, (cx + r) * S, (cy + R * 0.2 + r) * S], fill=255)
        d.polygon([(cx * S, (cy - R * 1.14) * S), ((cx - r * 0.97) * S, (cy + R * 0.02) * S), ((cx + r * 0.97) * S, (cy + R * 0.02) * S)], fill=255)
        return soften(m, 5)
    if name == "kite":
        return poly_mask(c, [(cx, cy - R * 1.16), (cx + R * 0.8, cy - R * 0.12), (cx, cy + R * 1.0), (cx - R * 0.8, cy - R * 0.12)], 6)
    if name == "tile":
        q = R * 0.9
        d.rounded_rectangle([(cx - q) * S, (cy - q) * S, (cx + q) * S, (cy + q) * S], radius=max(1, R * 0.17 * S), fill=255)
        return m
    return m


_NEW_DPAD = ("petal", "bar", "house", "oct")
_dpad_shape_v1 = dpad_shape


def dpad_shape(c, form, inset=0):
    if form not in _NEW_DPAD:
        return _dpad_shape_v1(c, form, inset)
    m = c.new()
    d = ImageDraw.Draw(m)
    i = inset
    if form == "petal":
        d.ellipse([(128 - 70 + i) * S, (150 - 70 + i) * S, (128 + 70 - i) * S, (150 + 70 - i) * S], fill=255)
        d.polygon([(128 * S, (20 + i) * S), ((128 - 66) * S, 136 * S), ((128 + 66) * S, 136 * S)], fill=255)
        m = soften(m, 6)
    elif form == "bar":
        d.rounded_rectangle([(100 + i) * S, (24 + i) * S, (156 - i) * S, (210 - i) * S], radius=max(1, (24 - i * 0.5) * S), fill=255)
    elif form == "house":
        m = poly_mask(c, [(128, 16), (190, 78), (190, 214), (66, 214), (66, 78)], 9)
        if i:
            m = shrink(m, i)
    elif form == "oct":
        m = poly_mask(c, ngon(128, 128, 70, 8, 22.5), 3)
        if i:
            m = shrink(m, i)
    return c.rotate(m)


ARROW.update({"petal": (82, 142, 26), "bar": (56, 108, 22), "house": (88, 152, 26), "oct": (94, 154, 26)})

_rect_shape_v1 = rect_shape


def rect_shape(c, name, inset=0):
    if name not in ("ticket", "leaf", "tab"):
        return _rect_shape_v1(c, name, inset)
    x0, y0, x1, y1 = c.rect_box()
    h = y1 - y0
    m = c.new()
    d = ImageDraw.Draw(m)
    if name == "ticket":
        d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=12 * S, fill=255)
        r = h * 0.2
        for px in (x0, x1):
            d.ellipse([(px - r) * S, ((y0 + y1) / 2 - r) * S, (px + r) * S, ((y0 + y1) / 2 + r) * S], fill=0)
        m = soften(m, 1.4)
    elif name == "leaf":
        d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=h * 0.5 * S, fill=255, corners=(True, False, True, False))
        d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=6 * S, fill=255, corners=(False, True, False, True))
    elif name == "tab":
        d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=h * 0.5 * S, fill=255, corners=(True, True, False, False))
        d.rounded_rectangle([x0 * S, (y1 - h * 0.3) * S, x1 * S, y1 * S], radius=8 * S, fill=255)
    if inset:
        m = shrink(m, inset)
    return m


# ---- new glyph styles ---------------------------------------------------------------------------------------


def g_gilt(sk, c, m, col):
    gold = grad(c.w, c.h, [(0, lighten(col, 0.6)), (0.5, col), (1, darkc(col, 0.5))], 65)
    out = paint(grow(m, 1.2), darkc(col, 0.72), 0.9)
    out = over(out, clip(gold, m))
    return over(out, paint(ImageChops.multiply(ImageChops.subtract(m, shift(m, 1.2, 2.0)), m), (255, 255, 255), 0.6))


def g_duo(sk, c, m, col):
    dcol = getattr(sk, "duo", None) or (255, 70, 150)
    return over(paint(shift(m, 2.8, 2.8), dcol, 0.92), paint(m, col))


def g_sticker(sk, c, m, col):
    out = paint(blur(shift(grow(m, 3), 0, 2.6), 2.0), (0, 0, 0), 0.38)
    out = over(out, paint(grow(m, 3.4), (255, 255, 255)))
    return over(out, paint(m, col))


def g_cutout(sk, c, m, col):
    out = paint(ImageChops.subtract(shift(m, 0, 1.8), m), (255, 255, 255), 0.45)
    out = over(out, paint(m, darkc(col, 0.75), 0.9))
    out = over(out, inner_shadow(m, 0, 2.6, 1.8, (0, 0, 0), 0.85))
    return over(out, paint(m, col, 0.16))


def g_prism(sk, c, m, col):
    yy, xx = _grid(c)
    arr = _hsv((xx + yy) / (c.w * 1.6) + 0.05, 0.55, 1.0)
    out = clip(_img(arr), m)
    return over(paint(blur(m, 2.0), (255, 255, 255), 0.25), out)


def g_spray(sk, c, m, col):
    out = paint(blur(m, 2.4), col, 0.5)
    ring = ImageChops.multiply(blur(grow(m, 7), 3), ImageChops.invert(m))
    sp = rng_noise(c.w, c.h, _pseed(c, 77)).point(lambda v: 255 if v > 215 else 0)
    out = over(out, paint(ImageChops.multiply(sp, ring), col, 0.9))
    out = over(out, paint(m, col, 0.94))
    a = np.asarray(m) > 128
    dr = c.new()
    dd = ImageDraw.Draw(dr)
    xs = np.where(a.any(0))[0]
    if xs.size:
        for f, ln in ((0.3, 15), (0.72, 24)):
            x = int(xs[0] + (xs[-1] - xs[0]) * f)
            ys = np.where(a[:, x])[0]
            if ys.size:
                y = ys.max()
                dd.line([(x, y - 4 * S), (x, y + ln * S)], fill=255, width=int(3.0 * S))
                dd.ellipse([x - 3.2 * S, y + ln * S - 3 * S, x + 3.2 * S, y + ln * S + 4 * S], fill=255)
    return over(out, paint(dr, col, 0.94))


def g_dyslexic(sk, c, m, col):
    out = paint(grow(m, 1.6), darkc(col, 0.62), 0.9)
    out = over(out, paint(m, col))
    n = int(c.part[1]) if c.part[0] in "lr" and c.part[1:].isdigit() else 0
    if n and c.kind in ("sh1", "sh2", "l3") and not BLANK:
        pm = c.new()
        dd = ImageDraw.Draw(pm)
        y = (c.rect_box()[3] - 9) * S if c.kind != "l3" else (128 + 74) * S
        if c.kind == "sh2":
            y = (c.rect_box()[3] - 18) * S
        for i in range(n):
            x = c.cx + (i - (n - 1) / 2.0) * 19 * S
            dd.ellipse([x - 6 * S, y - 6 * S, x + 6 * S, y + 6 * S], fill=255)
        out = over(out, paint(pm, col))
    return out


GLYPHS_STYLE.update({"gilt": g_gilt, "duo": g_duo, "sticker": g_sticker, "cutout": g_cutout, "prism": g_prism, "spray": g_spray,
                     "dyslexic": g_dyslexic})


# ---- new materials (fn(sk, c, M, hi, lo) -> RGBA layer) ----------------------------------------------------


def mat_stained(sk, c, M, hi, lo):
    gem = sk.gem or [hi, lo, sk.acc or lighten(hi, 0.5)]
    n = 13 if c.round else 8

    def mk():
        pts = _pts_in(M, n, _pseed(c, 7))
        idx, dist = _vor(c, pts)
        edge = np.zeros(idx.shape, bool)
        edge[:-1, :] |= idx[:-1, :] != idx[1:, :]
        edge[:, :-1] |= idx[:, :-1] != idx[:, 1:]
        return idx, dist, LI(edge.astype(np.float32)), len(pts)
    idx, dist, edge, cnt = sk.cache(("vor", c.part, sk.id), mk)
    rnd = random.Random(_pseed(c, 9))
    order = [gem[(i + rnd.randrange(len(gem))) % len(gem)][:3] for i in range(cnt + 1)]
    cols = np.array(order, np.float32)
    arr = cols[idx]
    arr *= np.clip(1.28 - dist / (40.0 * S), 0.72, 1.28)[..., None]
    arr *= (1.15 if c.p else 1.0)
    fill = _img(arr)
    out = shadow_of(c, M, 5, 8, 0.4)
    out = over(out, paint(blur(M, 10), lighten(hi, 0.4), 0.25))
    out = over(out, clip(fill, M))
    out = over(out, paint(ImageChops.multiply(blur(M, 2), top_fade(c, 1.6)), (255, 255, 255), 0.16))
    lead = ImageChops.multiply(grow(edge, 2.0), M)
    out = over(out, paint(lead, (28, 26, 34), 0.95))
    out = over(out, paint(shift(lead, 0, -1), (255, 255, 255), 0.1))
    return over(out, paint(ring_of(M, 5.5), (30, 28, 38), 0.98))


def mat_cloison(sk, c, M, hi, lo):
    gold = sk.acc or (226, 178, 72)
    out = shadow_of(c, M, 5, 6, 0.45)
    metal = grad(c.w, c.h, [(0, lighten(gold, 0.6)), (0.45, gold), (1, darkc(gold, 0.45))], -40)
    out = over(out, clip(tone(metal, 0.85 if c.p else 1.0), M))
    inner = shrink(M, 12)
    out = over(out, clip(tone(lit(c, lighten(hi, 0.12), lo), 0.84 if c.p else 1.0), inner))
    out = over(out, inner_shadow(inner, 0, 3, 3.5, darkc(lo, 0.5), 0.5))
    cell = ImageChops.multiply(shrink(M, 22), ring_of(shrink(M, 22), 1.6))
    out = over(out, paint(cell, gold, 0.8))
    out = over(out, paint(ImageChops.multiply(shrink(M, 15), glossmask(c)), (255, 255, 255), 0.22 if not c.p else 0.08))
    return over(out, paint(ring_of(M, 1.6), darkc(gold, 0.6), 0.9))


def mat_jade(sk, c, M, hi, lo):
    n1, n2 = _n(sk, c, 16, _pseed(c, 3)), _n(sk, c, 5, _pseed(c, 4), 1.4)
    t = np.clip(0.4 + 0.5 * (n1 - 0.5) + 0.35 * (n2 - 0.5), 0, 1)
    fill = _img(_cmap(t, [(0, lo), (0.5, hi), (1, lighten(hi, 0.5))]))
    vein = A(streak_noise(c.w, c.h, 30, _pseed(c, 5)))
    fill = over(fill, paint(LI(np.clip((vein - 0.6) * 2.5, 0, 1) * 0.5), lighten(hi, 0.7), 0.5))
    out = fin(sk, c, M, fill, lo, bev=(0.3, 0.5, 3.6), rim=0.5, shcol=darkc(lo, 0.7))
    edge = ImageChops.multiply(blur(ImageChops.subtract(M, shrink(M, 12)), 5), M)
    out = over(out, paint(edge, lighten(hi, 0.55), 0.35))
    out = over(out, inner_shadow(M, 0, 5, 8, darkc(lo, 0.6), 0.45))
    out = over(out, gloss_top(c, M, 0.3))
    return over(out, paint(ring_of(M, 1.6), lighten(hi, 0.5), 0.5))


def mat_obsidian(sk, c, M, hi, lo):
    fill = grad(c.w, c.h, [(0, hi), (1, lo)], 40)
    out = fin(sk, c, M, fill, lo, bev=(0.5, 0.5, 3.0), rim=0.6, sh=(6, 8, 0.55))
    st = ImageChops.multiply(blur(stripe_mask(c, 38, 0.1, 28, _pseed(c) % 30), 2.0), M)
    out = over(out, paint(st, (190, 205, 255), 0.18))
    x0, y0, x1, y1, cx, cy = _bb(M)
    ar = c.new()
    d = ImageDraw.Draw(ar)
    for k in range(1, 6):
        r = k * 26 * S
        d.ellipse([x1 - r - 10 * S, y0 - r + 20 * S, x1 + r - 10 * S, y0 + r + 20 * S], outline=255, width=int(1.4 * S))
    out = over(out, paint(ImageChops.multiply(ar, shrink(M, 3)), (170, 190, 255), 0.2))
    out = over(out, gloss_top(c, M, 0.34, 2.2))
    return over(out, paint(ring_of(M, 2.0), lighten(hi, 2.2) if False else (150, 164, 205), 0.6))


def mat_amber(sk, c, M, hi, lo):
    px, py = c.cx, c.cy
    fill = rgrad(c.w, c.h, c.w * 0.45, c.h * 0.4, c.w * 0.75, [(0, lighten(hi, 0.35)), (0.55, hi), (1, lo)])
    out = fin(sk, c, M, fill, lo, bev=(0.4, 0.5, 3.4), rim=0.6, shcol=darkc(lo, 0.7))
    inc = c.new()
    d = ImageDraw.Draw(inc)
    rnd = random.Random(_pseed(c, 13))
    for (x, y) in _pts_in(M, 16, _pseed(c, 14)):
        r = rnd.uniform(1.2, 4.2) * S
        d.ellipse([x - r, y - r * rnd.uniform(0.5, 1), x + r, y + r], fill=255)
    for (x, y) in _pts_in(M, 3, _pseed(c, 15)):
        pts = [(x, y)]
        a = rnd.uniform(0, 6.28)
        for _ in range(5):
            a += rnd.uniform(-0.8, 0.8)
            x, y = x + math.cos(a) * 8 * S, y + math.sin(a) * 8 * S
            pts.append((x, y))
        d.line(pts, fill=255, width=int(2.2 * S), joint="curve")
    inc = ImageChops.multiply(inc, shrink(M, 6))
    out = over(out, paint(blur(inc, 1.0), darkc(lo, 0.55), 0.8))
    bub = c.new()
    d = ImageDraw.Draw(bub)
    for (x, y) in _pts_in(M, 7, _pseed(c, 16)):
        r = rnd.uniform(2.5, 6) * S
        d.ellipse([x - r, y - r, x + r, y + r], outline=255, width=int(1.1 * S))
    out = over(out, paint(ImageChops.multiply(bub, shrink(M, 5)), (255, 240, 200), 0.75))
    out = over(out, inner_shadow(M, 0, 4, 7, darkc(lo, 0.5), 0.4))
    return over(out, gloss_top(c, M, 0.36))


def mat_rust(sk, c, M, hi, lo):
    n1, n2 = _n(sk, c, 22, _pseed(c, 21)), _n(sk, c, 6, _pseed(c, 22))
    iron = grad(c.w, c.h, [(0, (112, 110, 112)), (1, (58, 58, 62))], 40)
    t = np.clip((n1 * 1.2 + n2 * 0.2 - 0.5), 0, 1)
    rust = _cmap(n2, [(0, lo), (0.55, hi), (1, lighten(hi, 0.35))])
    edge = A(blur(ImageChops.subtract(M, shrink(M, 20)), 8))
    t = np.clip(t + edge * 0.6, 0, 1)
    s = np.clip((t - 0.25) * 3.5, 0, 1)[..., None]
    arr = np.asarray(iron.convert("RGB"), np.float32) * (1 - s) + rust * s
    pits = A(blur(rng_noise(c.w, c.h, _pseed(c, 23)), 0.8))
    arr *= np.where(pits < 0.3, 0.55, 1.0)[..., None]
    out = fin(sk, c, M, _img(arr), lo, bev=(0.35, 0.55, 3.4), rim=0.7)
    sc = ImageChops.multiply(stripe_mask(c, 60, 0.02, 20 + _pseed(c) % 40), shrink(M, 4))
    out = over(out, paint(sc, (200, 190, 180), 0.22))
    return over(out, inner_shadow(M, 0, 4, 6, (0, 0, 0), 0.35))


def mat_chrome(sk, c, M, hi, lo):
    x0, y0, x1, y1, cx, cy = _bb(M)
    yy, xx = _grid(c)
    v = (yy - y0) / max(1.0, (y1 - y0))
    v = np.clip(v + 0.1 * np.sin((xx - cx) / max(1.0, (x1 - x0)) * 3.2) * (v - 0.5), 0, 1)
    arr = _cmap(v, [(0, (238, 246, 255)), (0.4, (118, 146, 190)), (0.49, (24, 34, 54)), (0.52, (222, 214, 190)), (0.8, (148, 128, 100)), (1, (70, 60, 56))])
    tint = np.array(hi[:3], np.float32)
    arr = arr * 0.78 + (arr.mean(-1, keepdims=True) * tint / 255.0) * 0.5
    out = fin(sk, c, M, _img(arr), lo, bev=(0.6, 0.5, 2.6), rim=0.85, sh=(6, 8, 0.5))
    sp = c.new()
    ImageDraw.Draw(sp).ellipse([x0 + (x1 - x0) * 0.18, y0 + (y1 - y0) * 0.08, x0 + (x1 - x0) * 0.5, y0 + (y1 - y0) * 0.26], fill=255)
    out = over(out, paint(blur(ImageChops.multiply(sp, M), 2.4), (255, 255, 255), 0.7 if not c.p else 0.4))
    return over(out, paint(ring_of(M, 1.4), (255, 255, 255), 0.7))


def mat_foil(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    n1, n2 = _n(sk, c, 12, _pseed(c, 31), 2.6), _n(sk, c, 3, _pseed(c, 32), 2.0)
    hue = ((xx * 0.8 + yy * 0.45) / (70.0 * S) + n1 * 0.22) % 1.0
    lines = 0.55 + 0.45 * np.sin((xx * 0.7 + yy) / (2.6 * S) * 6.28)
    base = np.asarray(lit(c, hi, lo).convert("RGB"), np.float32)
    rain = _hsv(hue, 0.7, 0.55 + 0.45 * lines * (0.6 + n2 * 0.8))
    a = np.clip(0.55 + (n1 - 0.5) * 0.3, 0.3, 0.8)[..., None]
    arr = base * (1 - a) + rain * a
    sp = A(rng_noise(c.w, c.h, _pseed(c, 33)))
    arr += np.where(sp > 0.985, 130, 0)[..., None]
    out = fin(sk, c, M, _img(arr), lo, bev=(0.5, 0.5, 2.8), rim=0.7)
    return over(out, paint(ring_of(M, 1.6), (255, 255, 255), 0.55))


def mat_velvet(sk, c, M, hi, lo):
    fib = A(rng_noise(c.w, c.h, _pseed(c, 41)))
    t = np.clip(0.5 + (fib - 0.5) * 0.35 + (A(lin_l(c.w, c.h, 30)) - 0.5) * 0.4, 0, 1)
    fill = _img(_cmap(t, [(0, lo), (0.6, hi), (1, lighten(hi, 0.22))]))
    out = fin(sk, c, M, fill, lo, bev=None, rim=0, sh=(5, 8, 0.5))
    edge = ImageChops.multiply(blur(ImageChops.subtract(M, shrink(M, 22)), 9), M)
    out = over(out, paint(edge, lighten(hi, 0.65), 0.7 if not c.p else 0.45))
    out = over(out, paint(ImageChops.multiply(blur(shrink(M, 26), 14), M), darkc(lo, 0.5), 0.35))
    return over(out, paint(ring_of(M, 1.6), lighten(hi, 0.4), 0.5))


def mat_denim(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    thread = 0.5 + 0.5 * np.sin((xx + yy) / (1.9 * S) * 3.14159)
    warp = A(rng_noise(c.w, c.h, _pseed(c, 51)))
    n = _n(sk, c, 30, _pseed(c, 52), 1.6)
    t = np.clip(0.32 + 0.3 * thread + 0.18 * (warp - 0.5) + 0.35 * (n - 0.5), 0, 1)
    fill = _img(_cmap(t, [(0, lo), (1, hi)]))
    out = fin(sk, c, M, fill, lo, bev=(0.2, 0.5, 3.2), rim=0.7)
    wear = ImageChops.multiply(blur(ImageChops.subtract(M, shrink(M, 14)), 6), M)
    out = over(out, paint(wear, lighten(hi, 0.5), 0.35))
    st = _dashes(c, ring_of(shrink(M, 13), 2.2), 9, 0.6)
    out = over(out, paint(shift(st, 0, 1.2), (0, 0, 0), 0.4))
    return over(out, paint(st, sk.acc or (236, 160, 70)))


def mat_cork(sk, c, M, hi, lo):
    big, sm = _n(sk, c, 5, _pseed(c, 61), 3.2), _n(sk, c, 2, _pseed(c, 62), 2.4)
    t = np.clip(0.5 + (big - 0.5) * 0.8 + (sm - 0.5) * 0.5, 0, 1)
    fill = _img(_cmap(t, [(0, lo), (0.45, hi), (1, lighten(hi, 0.3))]))
    pores = A(blur(rng_noise(c.w, c.h, _pseed(c, 63)), 0.7))
    arr = np.asarray(fill.convert("RGB"), np.float32) * np.where(pores < 0.22, 0.55, 1.0)[..., None]
    out = fin(sk, c, M, _img(arr), lo, bev=(0.3, 0.5, 3.0), rim=0.6, shcol=darkc(lo, 0.6))
    return over(out, inner_shadow(M, 0, 3, 4, darkc(lo, 0.6), 0.35))


def mat_terrazzo(sk, c, M, hi, lo):
    gem = sk.gem or [lo, sk.acc or (200, 90, 80), (90, 120, 110), (230, 190, 90)]
    fill = lit(c, hi, mixc(hi, lo, 0.2))
    d = ImageDraw.Draw(fill)
    rnd = random.Random(_pseed(c, 71))
    for _ in range(80):
        x, y = rnd.uniform(0, c.w), rnd.uniform(0, c.h)
        r = rnd.uniform(2.5, 10) * S
        n = rnd.randint(4, 6)
        a0 = rnd.uniform(0, 6.28)
        pts = [(x + r * rnd.uniform(0.6, 1.1) * math.cos(a0 + 6.283 * i / n), y + r * rnd.uniform(0.6, 1.1) * math.sin(a0 + 6.283 * i / n)) for i in range(n)]
        d.polygon(pts, fill=tuple(gem[rnd.randrange(len(gem))][:3]) + (255,))
    out = fin(sk, c, M, fill, lo, bev=(0.35, 0.4, 3.0), rim=0.6)
    return over(out, gloss_top(c, M, 0.18))


def mat_zellige(sk, c, M, hi, lo):
    gem = sk.gem or [hi, lo, sk.acc or lighten(hi, 0.5)]
    fill = Image.new("RGBA", (c.w, c.h), tuple(gem[0][:3]) + (255,))
    d = ImageDraw.Draw(fill)
    cs = 54.0 * S
    rnd = random.Random(_pseed(c, 81))
    ox, oy = c.cx, c.cy
    grout = (236, 232, 222)
    for j in range(-7, 8):
        for i in range(-7, 8):
            px, py = ox + i * cs, oy + j * cs
            r = cs * 0.5
            jit = rnd.uniform(0.88, 1.08)
            sq1 = [(px - r, py - r), (px + r, py - r), (px + r, py + r), (px - r, py + r)]
            d.polygon(sq1, fill=tuple(gem[0][:3]))
            d.polygon([(px + r * math.cos(a) * 1.0, py + r * math.sin(a) * 1.0) for a in [math.radians(45 + 90 * k) for k in range(4)]], fill=None)
            star = []
            for k in range(16):
                rr = r * (0.98 if k % 2 == 0 else 0.55)
                a = math.radians(22.5 * k)
                star.append((px + rr * math.cos(a), py + rr * math.sin(a)))
            col = tuple(int(v * jit) for v in gem[1][:3])
            d.polygon(star, fill=col)
            cx2, cy2 = px + r, py + r
            d.polygon([(cx2, cy2 - r * 0.42), (cx2 + r * 0.42, cy2), (cx2, cy2 + r * 0.42), (cx2 - r * 0.42, cy2)], fill=tuple(gem[2 % len(gem)][:3]))
            d.polygon(star, outline=grout)
    d2 = ImageDraw.Draw(fill)
    for k in range(-8, 9):
        d2.line([(ox + k * cs + cs * 0.5, 0), (ox + k * cs + cs * 0.5, c.h)], fill=grout, width=int(1.4 * S))
        d2.line([(0, oy + k * cs + cs * 0.5), (c.w, oy + k * cs + cs * 0.5)], fill=grout, width=int(1.4 * S))
    fill = modulate(fill, LI(_n(sk, c, 14, _pseed(c, 82))), 0.25)
    out = fin(sk, c, M, fill, lo, bev=(0.5, 0.45, 2.6), rim=0.8)
    return over(out, gloss_top(c, M, 0.22))


def mat_studs(sk, c, M, hi, lo):
    out = fin(sk, c, M, lit(c, hi, lo), lo, bev=(0.5, 0.5, 3.2), rim=0.7)
    dm = ImageChops.multiply(dots_mask(c, 34, 11, 6, 6), shrink(M, 9))
    out = over(out, paint(blur(shift(dm, 2.4, 3.4), 2.0), (0, 0, 0), 0.45))
    body = clip(tone(rgrad(c.w, c.h, c.w * 0.4, c.h * 0.3, c.w * 0.5, [(0, lighten(hi, 0.55)), (1, hi)]), 0.85 if c.p else 1.0), dm)
    out = over(out, body)
    out = over(out, paint(ImageChops.multiply(shrink(dm, 5), shift(shrink(dm, 5), 1.6, 2.4)), (255, 255, 255), 0.0))
    out = over(out, paint(shift(shrink(dm, 6), -1.6, -2.2), (255, 255, 255), 0.0))
    out = over(out, paint(ring_of(dm, 1.3), darkc(lo, 0.5), 0.55))
    return out


def mat_swirl(sk, c, M, hi, lo):
    # Draw the spiral with the same units on both axes, then clip it to the button.
    rr, th, g = _polar(c, M, circular=c.kind in ("sh1", "sh2", "small"))
    ph = (th / 6.2832) * 3.0 + rr * 2.6
    w = np.clip((np.sin(ph * 6.2832) * 6.0) * 0.5 + 0.5, 0, 1)[..., None]
    a = np.array(hi[:3], np.float32)
    b = np.array((255, 248, 242), np.float32)
    arr = a * w + b * (1 - w)
    arr *= np.clip(1.15 - 0.25 * rr, 0.8, 1.1)[..., None]
    out = fin(sk, c, M, _img(arr), lo, bev=(0.5, 0.4, 3.2), rim=0.6, shcol=darkc(lo, 0.6))
    out = over(out, paint(ring_of(M, 3.4), (255, 255, 255), 0.35))
    sp = c.new()
    ImageDraw.Draw(sp).ellipse([g[0] + (g[2] - g[0]) * 0.14, g[1] + (g[3] - g[1]) * 0.1, g[0] + (g[2] - g[0]) * 0.52, g[1] + (g[3] - g[1]) * 0.32], fill=255)
    out = over(out, paint(blur(ImageChops.multiply(sp, M), 3.0), (255, 255, 255), 0.65 if not c.p else 0.3))
    return out


def mat_citrus(sk, c, M, hi, lo):
    rr, th, g = _polar(c, M)
    seg = (th + 3.14159) / 6.2832 * 10.0
    dseg = np.abs((seg % 1.0) - 0.5)
    ves = A(blur(rng_noise(c.w, c.h, _pseed(c, 91)), 1.2))
    pulp = np.array(hi[:3], np.float32)[None, None, :] * (0.78 + 0.5 * ves[..., None])
    pith = np.array((252, 244, 222), np.float32)
    rind = np.array(lo[:3], np.float32)
    arr = np.where((rr > 0.89)[..., None], rind, np.where((rr > 0.82)[..., None], pith, pulp))
    memb = ((dseg > 0.46) | (rr < 0.09)) & (rr <= 0.82)
    arr = np.where(memb[..., None], pith, arr)
    out = fin(sk, c, M, _img(arr), lo, bev=(0.3, 0.4, 3.0), rim=0.6, pdark=0.85)
    out = over(out, gloss_top(c, M, 0.22, 2.0, 18))
    return over(out, paint(ring_of(M, 1.8), darkc(lo, 0.3), 0.7))


def mat_cookie(sk, c, M, hi, lo):
    n = _n(sk, c, 14, _pseed(c, 101), 1.6)
    fill = _img(_cmap(0.45 + (n - 0.5) * 0.7, [(0, lo), (0.5, hi), (1, lighten(hi, 0.25))]))
    out = fin(sk, c, M, fill, lo, bev=(0.3, 0.5, 3.4), rim=0.5, shcol=darkc(lo, 0.7))
    out = over(out, paint(ImageChops.multiply(blur(ImageChops.subtract(M, shrink(M, 24)), 9), M), darkc(lo, 0.4), 0.5))
    choc = sk.acc or (74, 44, 30)
    rnd = random.Random(_pseed(c, 102))
    ch = c.new()
    d = ImageDraw.Draw(ch)
    for (x, y) in _pts_in(shrink(M, 14), 7, _pseed(c, 103)):
        r = rnd.uniform(5, 9) * S
        d.polygon([(x + r * rnd.uniform(0.7, 1.1) * math.cos(a), y + r * rnd.uniform(0.7, 1.1) * math.sin(a)) for a in [6.283 * k / 6 + rnd.uniform(0, 1) for k in range(6)]], fill=255)
    ch = soften(ch, 1.0)
    out = over(out, paint(shift(ch, 0, 1.8), (0, 0, 0), 0.35))
    out = over(out, paint(ch, choc))
    out = over(out, paint(ImageChops.multiply(ImageChops.subtract(ch, shift(ch, 1.4, 2.0)), ch), (255, 255, 255), 0.4))
    dk = ImageChops.multiply(dots_mask(c, 26, 2.2, 5, 9), shrink(M, 12))
    return over(out, paint(dk, darkc(lo, 0.5), 0.5))


def mat_sprinkles(sk, c, M, hi, lo):
    gem = sk.gem or [(255, 90, 120), (80, 190, 255), (255, 220, 70), (120, 220, 150), (255, 255, 255)]
    out = fin(sk, c, M, lit(c, lighten(hi, 0.12), hi), lo, bev=(0.4, 0.3, 3.2), rim=0.4, shcol=darkc(lo, 0.5))
    out = over(out, paint(ring_of(M, 11), (255, 255, 255), 0.22))
    rnd = random.Random(_pseed(c, 111))
    pr = shrink(M, 10)
    for (x, y) in _pts_in(pr, 26, _pseed(c, 112)):
        a = rnd.uniform(0, 3.14)
        L = 7 * S
        lay = c.new()
        dd = ImageDraw.Draw(lay)
        dd.line([(x - L * math.cos(a), y - L * math.sin(a)), (x + L * math.cos(a), y + L * math.sin(a))], fill=255, width=int(3.6 * S))
        r = 1.8 * S
        for sx in (-1, 1):
            dd.ellipse([x + sx * L * math.cos(a) - r, y + sx * L * math.sin(a) - r, x + sx * L * math.cos(a) + r, y + sx * L * math.sin(a) + r], fill=255)
        out = over(out, paint(shift(lay, 0, 1.3), (0, 0, 0), 0.25))
        out = over(out, paint(lay, gem[rnd.randrange(len(gem))]))
    return over(out, gloss_top(c, M, 0.34, 1.8, 16))


def mat_maki(sk, c, M, hi, lo):
    nori, rice, fill_in = shrink(M, 0), shrink(M, 17), shrink(M, 46)
    out = shadow_of(c, M, 5, 7, 0.4)
    nz = A(blur(rng_noise(c.w, c.h, _pseed(c, 121)), 0.6))
    nr = np.clip(np.stack([22 + 14 * nz, 40 + 16 * nz, 32 + 12 * nz], -1), 0, 255)
    out = over(out, clip(_img(nr), nori))
    out = over(out, paint(ring_of(M, 2.0), (6, 14, 10), 0.8))
    out = over(out, clip(tone(lit(c, (255, 255, 252), (226, 226, 218)), 0.85 if c.p else 1.0), rice))
    rnd = random.Random(_pseed(c, 122))
    g = c.new()
    d = ImageDraw.Draw(g)
    for (x, y) in _pts_in(rice, 150, _pseed(c, 123)):
        r = rnd.uniform(2.4, 3.6) * S
        a = rnd.uniform(0, 3.14)
        d.ellipse([x - r, y - r * 0.55, x + r, y + r * 0.55], fill=255)
    g = ImageChops.multiply(g, shrink(rice, 3))
    out = over(out, paint(shift(g, 0.8, 1.2), (150, 150, 140), 0.5))
    out = over(out, paint(g, (255, 255, 255), 0.95))
    out = over(out, clip(tone(lit(c, lighten(hi, 0.25), lo), 0.85 if c.p else 1.0), fill_in))
    out = over(out, inner_shadow(fill_in, 0, 3, 3, darkc(lo, 0.5), 0.4))
    return over(out, paint(ring_of(fill_in, 1.4), lighten(hi, 0.5), 0.5))


def mat_crt(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    base = rgrad(c.w, c.h, c.cx, c.cy, c.w * 0.62, [(0, mixc(lo, hi, 0.28)), (1, darkc(lo, 0.7))])
    arr = np.asarray(base.convert("RGB"), np.float32)
    scan = 0.62 + 0.38 * (np.sin(yy / (1.0 * S) * 3.14159) * 0.5 + 0.5)
    tri = np.stack([np.where((xx // S) % 3 == k, 1.0, 0.82) for k in range(3)], -1)
    arr = arr * scan[..., None] * tri
    out = fin(sk, c, M, _img(arr), lo, bev=None, rim=0, sh=(5, 7, 0.5))
    out = over(out, paint(blur(ImageChops.subtract(M, shrink(M, 14)), 7), (0, 0, 0), 0.5))
    out = over(out, paint(blur(M, 9), hi, 0.12 + (0.2 if c.p else 0)))
    rf = c.new()
    ImageDraw.Draw(rf).ellipse([c.w * 0.18, c.h * 0.14, c.w * 0.5, c.h * 0.34], fill=255)
    out = over(out, paint(ImageChops.multiply(blur(rf, 4), M), (255, 255, 255), 0.13))
    bz = ring_of(M, 7)
    out = over(out, clip(grad(c.w, c.h, [(0, (96, 100, 108)), (1, (28, 30, 34))], 50), bz))
    return over(out, paint(ring_of(shrink(M, 7), 1.2), hi, 0.55))


def mat_scope(sk, c, M, hi, lo):
    out = shadow_of(c, M, 5, 7, 0.5)
    out = over(out, clip(rgrad(c.w, c.h, c.cx, c.cy, c.w * 0.62, [(0, mixc(lo, hi, 0.12)), (1, darkc(lo, 0.5))]), M))
    grid = ImageChops.multiply(grid_mask(c, 32, 0.9), shrink(M, 8))
    out = over(out, paint(grid, hi, 0.22))
    x0, y0, x1, y1, cx, cy = _bb(M)
    ax = c.new()
    d = ImageDraw.Draw(ax)
    d.line([(cx, y0 + 8 * S), (cx, y1 - 8 * S)], fill=255, width=int(1.4 * S))
    d.line([(x0 + 8 * S, cy), (x1 - 8 * S, cy)], fill=255, width=int(1.4 * S))
    out = over(out, paint(ImageChops.multiply(ax, M), hi, 0.5))
    tr = c.new()
    d = ImageDraw.Draw(tr)
    ph = (_pseed(c) % 7) / 7.0 * 3
    rx, ry = (x1 - x0) / 2.0 * 0.86, (y1 - y0) / 2.0 * 0.7
    pts = [(cx + rx * math.sin(3 * t / 40.0 * 6.283 + ph), cy + ry * math.sin(2 * t / 40.0 * 6.283)) for t in range(0, 41)]
    pts = [(cx + rx * math.sin(3 * (t / 300.0) * 6.283 + ph), cy + ry * math.sin(2 * (t / 300.0) * 6.283)) for t in range(0, 301)]
    d.line(pts, fill=255, width=int(2.2 * S), joint="curve")
    tr = ImageChops.multiply(tr, shrink(M, 8))
    out = over(out, paint(blur(tr, 4), hi, 0.3))
    out = over(out, paint(tr, hi, 0.4))
    out = over(out, clip(grad(c.w, c.h, [(0, (150, 156, 166)), (1, (46, 48, 56))], 45), ring_of(M, 6)))
    return over(out, paint(ring_of(shrink(M, 6), 1.2), (0, 0, 0), 0.6))


def mat_drafting(sk, c, M, hi, lo):
    out = shadow_of(c, M, 4, 6, 0.28)
    out = over(out, clip(lit(c, D(c, hi), D(c, mixc(hi, lo, 0.25))), M))
    x0, y0, x1, y1, cx, cy = _bb(M)
    w, h = x1 - x0, y1 - y0
    cl = c.new()
    d = ImageDraw.Draw(cl)
    ox, oy = cx - w * 0.16, cy + h * 0.14
    for k in (0.2, 0.36, 0.52, 0.7):
        r = k * w
        d.arc([ox - r, oy - r, ox + r, oy + r], 200, 330, fill=255, width=int(1.1 * S))
    d.line([(x0 + 10 * S, oy), (x1 - 10 * S, oy)], fill=255, width=int(0.9 * S))
    d.line([(ox, y0 + 10 * S), (ox, y1 - 10 * S)], fill=255, width=int(0.9 * S))
    out = over(out, paint(ImageChops.multiply(cl, shrink(M, 5)), lo, 0.65))
    dm = c.new()
    d = ImageDraw.Draw(dm)
    yb = y1 - h * 0.17
    d.line([(x0 + w * 0.16, yb), (x1 - w * 0.16, yb)], fill=255, width=int(1.4 * S))
    for xx in (x0 + w * 0.16, x1 - w * 0.16):
        d.line([(xx, yb - 7 * S), (xx, yb + 7 * S)], fill=255, width=int(1.4 * S))
        d.polygon([(xx, yb), (xx + (9 if xx < cx else -9) * S, yb - 3 * S), (xx + (9 if xx < cx else -9) * S, yb + 3 * S)], fill=255)
    out = over(out, paint(ImageChops.multiply(dm, shrink(M, 4)), (214, 72, 52), 0.85))
    out = over(out, paint(ring_of(M, 2.0), lo, 0.8))
    return over(out, paint(ring_of(shrink(M, 8), 1.0), lo, 0.4))


def mat_spray(sk, c, M, hi, lo):
    Mw = wobble(M, 1.6, 30, _pseed(c, 131))
    out = paint(blur(Mw, 7), hi, 0.38)
    spk = rng_noise(c.w, c.h, _pseed(c, 132)).point(lambda v: 255 if v > 226 else 0)
    out = over(out, paint(ImageChops.multiply(spk, blur(grow(Mw, 12), 6)), hi, 0.85))
    core = ImageChops.multiply(Mw, A_to_L(_n(sk, c, 3, _pseed(c, 133), 1.2), 0.55, 1.0))
    out = over(out, paint(core, D(c, hi), 0.97))
    out = over(out, paint(ImageChops.multiply(shrink(Mw, 10), blur(M, 5)), lighten(hi, 0.3), 0.3))
    out = over(out, paint(ring_of(shrink(Mw, 3), 1.6), darkc(hi, 0.4), 0.35))
    a = np.asarray(Mw) > 128
    dr = c.new()
    dd = ImageDraw.Draw(dr)
    xs = np.where(a.any(0))[0]
    if xs.size and c.kind not in ("dpad",):
        for f, ln in ((0.3, 16), (0.74, 26)):
            x = int(xs[0] + (xs[-1] - xs[0]) * f)
            ys = np.where(a[:, x])[0]
            if ys.size:
                y = ys.max()
                dd.line([(x, y - 6 * S), (x, y + ln * S)], fill=255, width=int(3.2 * S))
                dd.ellipse([x - 3.4 * S, y + ln * S - 3 * S, x + 3.4 * S, y + ln * S + 4 * S], fill=255)
    return over(out, paint(dr, D(c, hi), 0.95))


def A_to_L(arr, lo_v, hi_v):
    return LI(lo_v + (hi_v - lo_v) * arr)


def mat_tribal(sk, c, M, hi, lo):
    out = fin(sk, c, M, lit(c, mixc(lo, hi, 0.2), lo), lo, bev=(0.3, 0.6, 3.4), rim=0.8)
    band = ImageChops.subtract(shrink(M, 9), shrink(M, 31))
    pat = ImageChops.multiply(stripe_mask(c, 17, 0.5, 45), stripe_mask(c, 17, 0.5, -45))
    pat2 = ImageChops.invert(pat)
    out = over(out, paint(ImageChops.multiply(band, pat), D(c, hi)))
    out = over(out, paint(ImageChops.multiply(band, pat2), darkc(lo, 0.5), 0.8))
    out = over(out, paint(ring_of(shrink(M, 9), 2.2), D(c, hi)))
    out = over(out, paint(ring_of(shrink(M, 31), 2.2), D(c, hi)))
    x0, y0, x1, y1, cx, cy = _bb(M)
    dots = c.new()
    d = ImageDraw.Draw(dots)
    for k in range(8):
        a = 6.283 * k / 8
        r = min(x1 - x0, y1 - y0) * 0.22
        d.ellipse([cx + r * math.cos(a) - 4 * S, cy + r * math.sin(a) - 4 * S, cx + r * math.cos(a) + 4 * S, cy + r * math.sin(a) + 4 * S], fill=255)
    return over(out, paint(ImageChops.multiply(dots, shrink(M, 32)), D(c, hi), 0.7))


def mat_nouveau(sk, c, M, hi, lo):
    gold = sk.acc or (214, 172, 78)
    out = fin(sk, c, M, grad(c.w, c.h, [(0, lighten(hi, 0.2)), (1, lo)], 55), lo, bev=(0.3, 0.45, 3.0), rim=0.7)
    x0, y0, x1, y1, cx, cy = _bb(M)
    rx, ry = (x1 - x0) / 2.0, (y1 - y0) / 2.0
    ln = c.new()
    d = ImageDraw.Draw(ln)
    wd = int(2.4 * S)
    for k in range(4):
        a = math.radians(90 * k + 45)
        ca, sa = math.cos(a), math.sin(a)
        upper, lower = [], []
        for i in range(0, 21):
            t = i / 20.0
            r = 0.14 + 0.62 * t
            hw = 0.15 * math.sin(math.pi * t) * (1.0 - 0.2 * t)
            px, py = cx + ca * r * rx, cy + sa * r * ry
            nx, ny = -sa, ca
            upper.append((px + nx * hw * rx, py + ny * hw * ry))
            lower.append((px - nx * hw * rx, py - ny * hw * ry))
        d.line(upper + lower[::-1] + [upper[0]], fill=255, width=wd, joint="curve")
        d.line([(cx + ca * 0.14 * rx, cy + sa * 0.14 * ry), (cx + ca * 0.74 * rx, cy + sa * 0.74 * ry)], fill=255, width=int(1.4 * S))
        for sgn in (-1, 1):
            b = a + sgn * math.radians(45)
            d.ellipse([cx + math.cos(b) * 0.7 * rx - 3 * S, cy + math.sin(b) * 0.7 * ry - 3 * S, cx + math.cos(b) * 0.7 * rx + 3 * S, cy + math.sin(b) * 0.7 * ry + 3 * S], fill=255)
    out = over(out, paint(shift(ImageChops.multiply(ln, M), 0, 1.2), (0, 0, 0), 0.3))
    out = over(out, paint(ImageChops.multiply(ln, M), gold))
    out = over(out, paint(ring_of(shrink(M, 10), 1.8), gold, 0.95))
    return over(out, paint(ring_of(M, 1.6), gold, 0.8))


def mat_gothic(sk, c, M, hi, lo):
    n = _n(sk, c, 20, _pseed(c, 141), 1.5)
    fill = _img(_cmap(0.55 + (n - 0.5) * 0.35, [(0, lo), (0.5, hi), (1, lighten(hi, 0.2))]))
    out = fin(sk, c, M, fill, lo, bev=(0.6, 0.6, 4.0), rim=0.85)
    x0, y0, x1, y1, cx, cy = _bb(M)
    w = min(x1 - x0, y1 - y0)
    tr = c.new()
    d = ImageDraw.Draw(tr)
    for k in range(3):
        a = math.radians(-90 + 120 * k)
        px, py = cx + math.cos(a) * w * 0.17, cy + math.sin(a) * w * 0.17 + 4 * S
        r = w * 0.17
        d.ellipse([px - r, py - r, px + r, py + r], outline=255, width=int(4 * S))
    d.ellipse([cx - w * 0.05, cy + 4 * S - w * 0.05, cx + w * 0.05, cy + 4 * S + w * 0.05], outline=255, width=int(3 * S))
    tr = ImageChops.multiply(tr, shrink(M, 6))
    out = over(out, paint(shift(tr, 1, 1.6), (255, 255, 255), 0.2))
    out = over(out, paint(tr, darkc(lo, 0.55), 0.4))
    fr = ring_of(shrink(M, 14), 3.4)
    out = over(out, paint(shift(fr, 1, 1.6), (255, 255, 255), 0.3))
    out = over(out, paint(fr, darkc(lo, 0.55), 0.8))
    return over(out, inner_shadow(shrink(M, 14), 0, 4, 5, (0, 0, 0), 0.3))


def mat_gauge(sk, c, M, hi, lo):
    brass = sk.acc or (196, 148, 66)
    out = shadow_of(c, M, 6, 8, 0.5)
    out = over(out, clip(grad(c.w, c.h, [(0, lighten(brass, 0.6)), (0.45, brass), (1, darkc(brass, 0.55))], -40), M))
    dial = shrink(M, 17)
    out = over(out, clip(tone(rgrad(c.w, c.h, c.cx, c.cy, c.w * 0.6, [(0, lighten(hi, 0.1)), (1, mixc(hi, lo, 0.5))]), 0.85 if c.p else 1.0), dial))
    out = over(out, inner_shadow(dial, 0, 5, 5, (60, 40, 10), 0.5))
    rr, th, g = _polar(c, M)
    x0, y0, x1, y1, cx, cy, rx, ry = g
    tk = c.new()
    d = ImageDraw.Draw(tk)
    for i in range(41):
        a = math.radians(135 + 270 * i / 40.0)
        r0 = 0.64 if i % 5 == 0 else 0.72
        d.line([(cx + math.cos(a) * r0 * rx, cy + math.sin(a) * r0 * ry), (cx + math.cos(a) * 0.8 * rx, cy + math.sin(a) * 0.8 * ry)], fill=255, width=int((2.6 if i % 5 == 0 else 1.4) * S))
    out = over(out, paint(ImageChops.multiply(tk, dial), (46, 34, 22), 0.9))
    na = math.radians(135 + 270 * (0.22 + 0.5 * ((_pseed(c) % 9) / 9.0) + (0.25 if c.p else 0)))
    nd = c.new()
    d = ImageDraw.Draw(nd)
    d.polygon([(cx + math.cos(na) * 0.72 * rx, cy + math.sin(na) * 0.72 * ry), (cx + math.cos(na + 1.6) * 0.05 * rx, cy + math.sin(na + 1.6) * 0.05 * ry),
               (cx - math.cos(na) * 0.14 * rx, cy - math.sin(na) * 0.14 * ry), (cx + math.cos(na - 1.6) * 0.05 * rx, cy + math.sin(na - 1.6) * 0.05 * ry)], fill=255)
    out = over(out, paint(shift(nd, 1.5, 2.5), (0, 0, 0), 0.35))
    out = over(out, paint(nd, (190, 40, 36)))
    hub = c.new()
    ImageDraw.Draw(hub).ellipse([cx - 8 * S, cy - 8 * S, cx + 8 * S, cy + 8 * S], fill=255)
    out = over(out, clip(rgrad(c.w, c.h, cx - 3 * S, cy - 3 * S, 14 * S, [(0, lighten(brass, 0.7)), (1, darkc(brass, 0.4))]), hub))
    rv = c.new()
    d = ImageDraw.Draw(rv)
    for k in range(8):
        a = 6.283 * k / 8 + 0.4
        px, py = cx + math.cos(a) * (rx - 8.5 * S), cy + math.sin(a) * (ry - 8.5 * S)
        d.ellipse([px - 3 * S, py - 3 * S, px + 3 * S, py + 3 * S], fill=255)
    out = over(out, paint(shift(rv, 0.6, 1), (0, 0, 0), 0.4))
    out = over(out, paint(rv, lighten(brass, 0.5)))
    out = over(out, paint(ring_of(M, 1.8), darkc(brass, 0.6), 0.9))
    return over(out, gloss_top(c, dial, 0.25, 1.8, 6))


def mat_astro(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    wv = 0.96 + 0.04 * np.sin(xx / (1.3 * S)) * np.sin(yy / (1.3 * S))
    base = np.asarray(lit(c, hi, lo).convert("RGB"), np.float32) * wv[..., None]
    out = fin(sk, c, M, _img(base), lo, bev=(0.5, 0.35, 3.8), rim=0.6, shcol=darkc(lo, 0.5))
    st = _dashes(c, ring_of(shrink(M, 12), 1.8), 7, 0.55)
    out = over(out, paint(st, darkc(lo, 0.45), 0.8))
    x0, y0, x1, y1, cx, cy = _bb(M)
    band = c.new()
    ImageDraw.Draw(band).rectangle([x0, y0 + (y1 - y0) * 0.6, x1, y0 + (y1 - y0) * 0.74], fill=255)
    band = ImageChops.multiply(band, shrink(M, 4))
    ac = sk.acc or (255, 112, 32)
    out = over(out, paint(band, D(c, ac)))
    out = over(out, paint(ImageChops.subtract(shift(band, 0, 1.4), band), (255, 255, 255), 0.0))
    out = over(out, paint(ring_of(band, 1.1), darkc(ac, 0.5), 0.7))
    sc = c.new()
    d = ImageDraw.Draw(sc)
    yb = y0 + (y1 - y0) * 0.67
    for px in (x0 + (x1 - x0) * 0.26, x1 - (x1 - x0) * 0.26):
        d.ellipse([px - 4 * S, yb - 4 * S, px + 4 * S, yb + 4 * S], fill=255)
    out = over(out, paint(ImageChops.multiply(sc, M), (60, 64, 74), 0.9))
    return out


def mat_mecha(sk, c, M, hi, lo):
    out = fin(sk, c, M, grad(c.w, c.h, [(0, lighten(hi, 0.25)), (0.5, hi), (1, lo)], 70), lo, bev=(0.6, 0.6, 3.6), rim=0.9)
    x0, y0, x1, y1, cx, cy = _bb(M)
    w, h = x1 - x0, y1 - y0
    pl = c.new()
    d = ImageDraw.Draw(pl)
    d.line([(x0, y0 + h * 0.34), (x0 + w * 0.62, y0 + h * 0.34), (x0 + w * 0.82, y0 + h * 0.52), (x1, y0 + h * 0.52)], fill=255, width=int(2.4 * S), joint="curve")
    d.line([(x0 + w * 0.22, y0), (x0 + w * 0.22, y0 + h * 0.34)], fill=255, width=int(2.2 * S))
    pl = ImageChops.multiply(pl, shrink(M, 2))
    out = over(out, paint(shift(pl, 0, 1.8), (255, 255, 255), 0.35))
    out = over(out, paint(pl, darkc(lo, 0.6), 0.9))
    hz = c.new()
    ImageDraw.Draw(hz).rectangle([x0, y0 + h * 0.76, x1, y1], fill=255)
    hz = ImageChops.multiply(ImageChops.multiply(hz, shrink(M, 7)), stripe_mask(c, 14, 0.5, 45))
    out = over(out, paint(hz, (250, 206, 40), 0.95))
    sc = c.new()
    d = ImageDraw.Draw(sc)
    for (px, py) in ((x0 + w * 0.14, y0 + h * 0.18), (x1 - w * 0.14, y0 + h * 0.18)):
        d.ellipse([px - 3.4 * S, py - 3.4 * S, px + 3.4 * S, py + 3.4 * S], fill=255)
    return over(out, paint(ImageChops.multiply(sc, M), darkc(lo, 0.6), 0.85))


def mat_urushi(sk, c, M, hi, lo):
    out = fin(sk, c, M, grad(c.w, c.h, [(0, lighten(hi, 0.12)), (0.6, hi), (1, lo)], 50), lo, bev=(0.5, 0.5, 3.0), rim=0.8, sh=(6, 8, 0.55))
    gold = sk.acc or (232, 186, 84)
    n = _n(sk, c, 8, _pseed(c, 151), 2.0)
    edge = A(blur(ImageChops.subtract(M, shrink(M, 38)), 14))
    dust = A(rng_noise(c.w, c.h, _pseed(c, 152)))
    dm = LI(np.where(dust > (0.9 - 0.07 * edge - 0.03 * n), 1.0, 0.0))
    out = over(out, paint(ImageChops.multiply(blur(dm, 0.4), shrink(M, 4)), gold, 0.95))
    out = over(out, paint(ring_of(shrink(M, 11), 1.5), gold, 0.95))
    st = c.new()
    ImageDraw.Draw(st).line([(c.w * 0.1, c.h * 0.22), (c.w * 0.6, c.h * 0.1)], fill=255, width=int(6 * S))
    out = over(out, paint(ImageChops.multiply(blur(st, 3), shrink(M, 6)), (255, 255, 255), 0.22 if not c.p else 0.1))
    return over(out, gloss_top(c, M, 0.3, 2.1))


def mat_porcelain(sk, c, M, hi, lo):
    out = fin(sk, c, M, lit(c, (253, 254, 255), (214, 224, 240)), lo, bev=(0.35, 0.3, 3.2), rim=0.5, shcol=(30, 40, 70))
    rr, th, g = _polar(c, M)
    x0, y0, x1, y1, cx, cy, rx, ry = g
    cb = c.new()
    d = ImageDraw.Draw(cb)
    wd = int(2.0 * S)
    for k in range(8):
        a = 6.283 * k / 8
        ca, sa = math.cos(a), math.sin(a)
        up, lw = [], []
        for i in range(0, 15):
            t = i / 14.0
            r = 0.52 + 0.34 * t
            hw = 0.11 * math.sin(math.pi * t)
            nx, ny = -sa, ca
            up.append((cx + (ca * r + nx * hw) * rx, cy + (sa * r + ny * hw) * ry))
            lw.append((cx + (ca * r - nx * hw) * rx, cy + (sa * r - ny * hw) * ry))
        poly = up + lw[::-1]
        d.polygon(poly, fill=70)
        d.line(poly + [poly[0]], fill=255, width=wd, joint="curve")
    cb = blur(cb, 0.5)
    cb = ImageChops.lighter(cb, ring_of(shrink(M, 8), 3.0))
    cb = ImageChops.lighter(cb, ring_of(shrink(M, 15), 1.2))
    cb = ImageChops.multiply(cb, M)
    out = over(out, paint(cb, tuple(int(v * (0.8 if c.p else 1.0)) for v in hi[:3]), 0.95))
    return over(out, gloss_top(c, M, 0.4, 1.7, 8))


def mat_crossstitch(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    wv = 0.9 + 0.1 * (np.sin(xx / (1.1 * S) * 3.14) * np.sin(yy / (1.1 * S) * 3.14) > 0)
    base = np.asarray(lit(c, hi, mixc(hi, lo, 0.2)).convert("RGB"), np.float32) * wv[..., None]
    out = fin(sk, c, M, _img(base), lo, bev=(0.2, 0.4, 3.0), rim=0.5)
    cs = 7.0 * S
    inner = shrink(M, 12)
    band = ImageChops.subtract(shrink(M, 5), inner)
    ip, bp = inner.load(), band.load()
    red, blk = sk.acc or (190, 34, 40), (40, 30, 34)
    lay = c.new()
    sh = c.new()
    dl = ImageDraw.Draw(lay)
    dsh = ImageDraw.Draw(sh)
    cols = [(red, []), (blk, [])]
    n = int(c.w / cs) + 1
    i0 = n // 2
    for j in range(n):
        for i in range(n):
            x, y = (i + 0.5) * cs, (j + 0.5) * cs
            xi, yi = min(c.w - 1, int(x)), min(c.h - 1, int(y))
            col = None
            if bp[xi, yi] > 200:
                col = 0 if (i + j) % 2 == 0 else 1
            elif ip[xi, yi] > 200:
                dd = (abs(i - i0) + abs(j - i0)) % 8
                col = 0 if dd in (0, 4) else (1 if dd == 2 else None)
            if col is not None:
                cols[col][1].append((x, y))
    out_l = out
    for ccol, pts in cols:
        lm = c.new()
        d = ImageDraw.Draw(lm)
        r = cs * 0.4
        for (x, y) in pts:
            d.line([(x - r, y - r), (x + r, y + r)], fill=255, width=int(2.4 * S))
            d.line([(x - r, y + r), (x + r, y - r)], fill=255, width=int(2.4 * S))
        out_l = over(out_l, paint(shift(lm, 0, 0.8), (0, 0, 0), 0.3))
        out_l = over(out_l, paint(lm, tuple(int(v * (0.8 if c.p else 1.0)) for v in ccol)))
    return out_l


def mat_celtic(sk, c, M, hi, lo):
    out = fin(sk, c, M, lit(c, mixc(lo, hi, 0.1), darkc(lo, 0.4)), lo, bev=(0.4, 0.5, 3.2), rim=0.8)
    rr, th, g = _polar(c, M)
    x0, y0, x1, y1, cx, cy, rx, ry = g
    k = 8
    acc = sk.acc or lighten(hi, 0.4)

    def pts(ph, t0, t1, n=240):
        return [(cx + (0.72 + 0.11 * math.sin(k * t + ph)) * math.cos(t) * rx, cy + (0.72 + 0.11 * math.sin(k * t + ph)) * math.sin(t) * ry)
                for t in [t0 + (t1 - t0) * i / n for i in range(n + 1)]]
    lay = Image.new("RGBA", (c.w, c.h), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    cA, cB = tuple(int(v * (0.82 if c.p else 1)) for v in hi[:3]), tuple(int(v * (0.82 if c.p else 1)) for v in acc[:3])
    ow, iw = int(11 * S), int(7 * S)

    def draw(ph, col, t0=0.0, t1=6.2832, n=300):
        p = pts(ph, t0, t1, n)
        d.line(p, fill=(10, 14, 10, 255), width=ow, joint="curve")
        d.line(p, fill=col + (255,), width=iw, joint="curve")
        d.line(p, fill=lighten(col, 0.4) + (255,), width=int(1.6 * S), joint="curve")
    draw(0.0, cA)
    draw(math.pi, cB)
    for q in range(2 * k):
        if q % 2 == 1:
            t = math.pi * q / k
            draw(math.pi, cB, t - 0.2, t + 0.2, 20)
    out = over(out, clip(lay, shrink(M, 4)))
    out = over(out, paint(ring_of(shrink(M, 9), 1.6), (10, 14, 10), 0.6))
    return over(out, paint(ring_of(M, 1.8), acc, 0.7))


def mat_pinpoint(sk, c, M, hi, lo):
    x0, y0, x1, y1, cx, cy = _bb(M)
    r = (28 if c.kind in ("face", "l3", "stick", "dpad") else 18) * S * (1.2 if c.p else 1.0)
    dot = c.new()
    if c.kind in ("sh1", "sh2", "small"):
        ImageDraw.Draw(dot).rounded_rectangle([cx - 46 * S, cy - 13 * S, cx + 46 * S, cy + 13 * S], radius=13 * S, fill=255)
    else:
        ImageDraw.Draw(dot).ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    out = paint(blur(dot, 7), hi, 0.5)
    out = over(out, paint(dot, hi, 0.95))
    out = over(out, paint(ring_of(shrink(dot, 3), 1.6), lighten(hi, 0.6), 0.55))
    out = over(out, paint(ring_of(M, 1.4), hi, 0.22))
    return out


def mat_corners(sk, c, M, hi, lo):
    x0, y0, x1, y1, cx, cy = _bb(M)
    L = min(x1 - x0, y1 - y0) * 0.3
    mk = c.new()
    d = ImageDraw.Draw(mk)
    w = int((2.4 if c.p else 1.8) * S * sk.line / 3.0)
    for (px, py, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        d.line([(px, py + sy * L), (px, py), (px + sx * L, py)], fill=255, width=w, joint="curve")
    return paint(mk, hi, 0.95)


def mat_target(sk, c, M, hi, lo):
    out = shadow_of(c, M, 4, 6, 0.35)
    for k, ins in enumerate((0, 16, 34)):
        mk = shrink(M, ins) if ins else M
        out = over(out, paint(mk, D(c, hi if k % 2 == 0 else lo)))
    out = over(out, paint(ring_of(M, 2.0), darkc(lo, 0.5), 0.8))
    out = over(out, paint(ImageChops.multiply(shrink(M, 8), top_fade(c, 2.0)), (255, 255, 255), 0.12))
    return out


def mat_crayon(sk, c, M, hi, lo):
    Mw = wobble(M, 2.4, 26, _pseed(c, 161))
    yy, xx = _grid(c)
    st = ImageChops.multiply(stripe_mask(c, 3.4, 0.5, 35 + _pseed(c) % 30), Mw)
    n = _n(sk, c, 3, _pseed(c, 162), 2.0)
    pap = np.where(A(blur(rng_noise(c.w, c.h, _pseed(c, 163)), 0.9)) < 0.36, 1.0, 0.0)
    out = shadow_of(c, Mw, 4, 5, 0.25)
    out = over(out, paint(Mw, D(c, hi)))
    out = over(out, paint(st, lighten(hi, 0.22), 0.55))
    out = over(out, paint(ImageChops.multiply(LI(pap), Mw), (255, 250, 236), 0.35))
    ring = ImageChops.multiply(ring_of(Mw, 6.0), A_to_L(n, 0.6, 1.0))
    out = over(out, paint(ring, darkc(hi, 0.55), 0.95))
    return over(out, paint(ring_of(shrink(Mw, 8), 1.4), lighten(hi, 0.4), 0.5))


def mat_hivis(sk, c, M, hi, lo):
    fill, ink = (lo, hi) if c.p else (hi, lo)
    out = blank(c) if c.p else paint(shift(M, 5, 5), ink, 1.0)
    out = over(out, paint(M, ink))
    out = over(out, paint(shrink(M, 12), fill))
    out = over(out, paint(ring_of(shrink(M, 22), 3.4), ink))
    return out


def mat_nacre(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    n = _n(sk, c, 22, _pseed(c, 171), 1.8)
    ph = 0.8 * np.sin(xx / (46.0 * S) * 6.28) + 0.6 * np.sin(yy / (38.0 * S) * 6.28 + xx / (90.0 * S)) + n * 1.6
    rgb = np.stack([0.84 + 0.14 * np.cos(6.283 * (ph + o)) for o in (0.0, 0.33, 0.67)], -1) * 255
    tint = np.array(hi[:3], np.float32) / 255.0
    arr = rgb * (0.65 + 0.35 * tint)
    out = fin(sk, c, M, _img(arr), lo, bev=(0.5, 0.4, 3.0), rim=0.5, shcol=darkc(lo, 0.5))
    return over(out, gloss_top(c, M, 0.45, 1.7))


def mat_meteorite(sk, c, M, hi, lo):
    n = _n(sk, c, 24, _pseed(c, 181), 2.4)
    fill = lit(c, mixc(lo, hi, 0.5), lo)
    reg = np.clip((n * 3).astype(int), 0, 2)
    lines = c.new()
    for k, ang in enumerate((20, 80, 140)):
        sm = stripe_mask(c, 7, 0.16, ang, _pseed(c) % 5)
        lines = ImageChops.lighter(lines, ImageChops.multiply(sm, LI(np.where(reg == k, 1.0, 0.0))))
    out = fin(sk, c, M, fill, lo, bev=(0.5, 0.55, 3.2), rim=0.85)
    out = over(out, paint(ImageChops.multiply(lines, shrink(M, 4)), (206, 210, 220), 0.4))
    pit = c.new()
    d = ImageDraw.Draw(pit)
    rnd = random.Random(_pseed(c, 182))
    for (x, y) in _pts_in(shrink(M, 10), 6, _pseed(c, 183)):
        r = rnd.uniform(6, 13) * S
        d.ellipse([x - r, y - r * 0.8, x + r, y + r * 0.8], fill=255)
    out = over(out, paint(shift(pit, 1.4, 2.0), (255, 255, 255), 0.0))
    out = over(out, inner_shadow(pit, 1.2, 1.8, 2.2, (0, 0, 0), 0.0))
    out = over(out, paint(blur(pit, 1.2), (0, 0, 0), 0.5))
    out = over(out, paint(ImageChops.subtract(shift(pit, 1.2, 2.0), pit), (255, 255, 255), 0.25))
    return over(out, paint(ImageChops.multiply(blur(ImageChops.subtract(M, shrink(M, 9)), 3), M), (0, 0, 0), 0.45))


def mat_aurora(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    x0, y0, x1, y1, cx, cy = _bb(M)
    v = (yy - y0) / max(1.0, (y1 - y0))
    n = _n(sk, c, 30, _pseed(c, 191), 1.6)
    wave = 0.12 * np.sin((xx - x0) / (60.0 * S) * 6.28 + n * 4.0)
    arr = np.zeros((c.h, c.w, 3), np.float32) + np.array(lo[:3], np.float32)
    for pos, wd, col in ((0.38, 0.13, (60, 255, 170)), (0.6, 0.12, hi[:3]), (0.78, 0.1, (110, 170, 255))):
        b = np.exp(-(((v - pos - wave) / wd) ** 2))
        stri = 0.7 + 0.3 * np.sin((xx / S) / 5.0 + n * 6)
        arr += b[..., None] * stri[..., None] * np.array(col, np.float32) * (0.95 if c.p else 0.75)
    sp = A(rng_noise(c.w, c.h, _pseed(c, 192)))
    arr += np.where(sp > 0.9985, 210, 0)[..., None]
    out = fin(sk, c, M, _img(arr), lo, bev=(0.4, 0.5, 3.0), rim=0.7)
    return over(out, paint(ring_of(M, 1.8), (200, 255, 240), 0.5))


def mat_galaxy(sk, c, M, hi, lo):
    n1, n2 = _n(sk, c, 34, _pseed(c, 201), 2.0), _n(sk, c, 14, _pseed(c, 202), 2.0)
    arr = np.zeros((c.h, c.w, 3), np.float32) + np.array(lo[:3], np.float32)
    arr += (np.clip(n1 * 1.2 - 0.3, 0, 1) ** 1.5)[..., None] * np.array(hi[:3], np.float32) * (0.9 if not c.p else 1.2)
    arr += (np.clip(n2 * 1.2 - 0.45, 0, 1) ** 2)[..., None] * np.array(sk.acc or (255, 120, 190), np.float32)
    stars = A(rng_noise(c.w, c.h, _pseed(c, 203)))
    arr += np.where(stars > 0.9965, 255, 0)[..., None]
    out = fin(sk, c, M, _img(arr), lo, bev=(0.3, 0.5, 3.0), rim=0.7)
    fl = c.new()
    d = ImageDraw.Draw(fl)
    for (x, y) in _pts_in(M, 3, _pseed(c, 204)):
        d.line([(x - 8 * S, y), (x + 8 * S, y)], fill=255, width=int(1.1 * S))
        d.line([(x, y - 8 * S), (x, y + 8 * S)], fill=255, width=int(1.1 * S))
    out = over(out, paint(blur(fl, 0.8), (255, 255, 255), 0.95))
    return over(out, paint(ring_of(M, 1.6), lighten(hi, 0.5), 0.55))


def mat_waxseal(sk, c, M, hi, lo):
    Mw = wobble(M, 3.2, 36, _pseed(c, 211))
    out = fin(sk, c, Mw, lit(c, lighten(hi, 0.12), lo), lo, bev=(0.55, 0.55, 3.6), rim=0.5, shcol=darkc(lo, 0.7))
    inner = shrink(Mw, 19)
    out = over(out, inner_shadow(inner, 0, 4, 4, darkc(lo, 0.6), 0.65))
    out = over(out, clip(tone(lit(c, hi, darkc(lo, 0.1)), 0.9 if c.p else 1.0), inner))
    out = over(out, paint(ImageChops.subtract(shift(ring_of(inner, 2.2), 0, 1.5), ring_of(inner, 2.2)), (255, 255, 255), 0.35))
    out = over(out, paint(ring_of(shrink(Mw, 30), 2.2), darkc(lo, 0.5), 0.6))
    return over(out, gloss_top(c, Mw, 0.42, 1.7, 14))


def mat_melon(sk, c, M, hi, lo):
    rr, th, g = _polar(c, M)
    x0, y0, x1, y1, cx, cy, rx, ry = g
    stripe = 0.5 + 0.5 * np.sin(th * 9 + rr * 2.0)
    rind = np.stack([30 + 20 * stripe, 110 + 40 * stripe, 50 + 20 * stripe], -1)
    white = np.array((222, 240, 200), np.float32)
    flesh = _cmap(rr / 0.8, [(0, lighten(hi, 0.18)), (1, hi)])
    arr = np.where((rr > 0.9)[..., None], rind, np.where((rr > 0.82)[..., None], white, flesh))
    out = fin(sk, c, M, _img(arr), lo, bev=(0.3, 0.4, 3.0), rim=0.6, pdark=0.85)
    sd = c.new()
    d = ImageDraw.Draw(sd)
    for k in range(9):
        a = 6.283 * k / 9 + 0.3
        for rf in ((0.35, 0.58) if k % 2 == 0 else (0.5,)):
            px, py = cx + math.cos(a) * rf * rx, cy + math.sin(a) * rf * ry
            d.ellipse([px - 2.4 * S, py - 4.4 * S, px + 2.4 * S, py + 4.4 * S], fill=255)
    sd = sd.rotate(25, center=(cx, cy))
    out = over(out, paint(sd, (24, 18, 18), 0.95))
    return over(out, gloss_top(c, M, 0.22, 2.0, 18))


def mat_honey(sk, c, M, hi, lo):
    fill = Image.new("RGBA", (c.w, c.h), tuple(hi[:3]) + (255,))
    d = ImageDraw.Draw(fill)
    s = 22.0 * S
    wall = lighten(hi, 0.5)
    rnd = random.Random(_pseed(c, 221))
    for r in range(-9, 10):
        for q in range(-9, 10):
            px = c.cx + s * 1.732 * (q + r / 2.0)
            py = c.cy + s * 1.5 * r
            pts = [(px + s * math.cos(math.radians(30 + 60 * i)), py + s * math.sin(math.radians(30 + 60 * i))) for i in range(6)]
            col = mixc(lo, hi, rnd.uniform(0.35, 1.0))
            d.polygon(pts, fill=col + (255,))
            d.polygon([(px + s * 0.78 * math.cos(math.radians(30 + 60 * i)), py + s * 0.78 * math.sin(math.radians(30 + 60 * i)) + 2) for i in range(6)], fill=lighten(col, 0.2) + (255,))
            d.line(pts + [pts[0]], fill=tuple(wall) + (255,), width=int(2.6 * S))
    out = fin(sk, c, M, fill, lo, bev=(0.4, 0.45, 3.0), rim=0.7, shcol=darkc(lo, 0.7))
    out = over(out, paint(ImageChops.multiply(blur(ImageChops.subtract(M, shrink(M, 10)), 4), M), lighten(hi, 0.6), 0.35))
    return over(out, gloss_top(c, M, 0.26, 1.8))


def mat_treering(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    rr, th, g = _polar(c, M)
    x0, y0, x1, y1, cx, cy, rx, ry = g
    n = _n(sk, c, 28, _pseed(c, 231), 1.6)
    d = np.sqrt(((xx - cx - 0.12 * rx) / rx) ** 2 + ((yy - cy + 0.08 * ry) / ry) ** 2)
    ring = 0.5 + 0.5 * np.sin(6.283 * (d * 7.0 + (n - 0.5) * 1.6))
    t = np.clip(ring, 0, 1) ** 1.6
    fill = _img(_cmap(t, [(0, hi), (1, lo)]))
    out = fin(sk, c, M, fill, lo, bev=(0.35, 0.5, 3.2), rim=0.8)
    bark = ImageChops.subtract(M, shrink(M, 11))
    bn = A(streak_noise(c.w, c.h, 8, _pseed(c, 232)))
    out = over(out, paint(ImageChops.multiply(bark, A_to_L(bn, 0.6, 1.0)), darkc(lo, 0.55), 0.95))
    ck = c.new()
    ImageDraw.Draw(ck).line([(cx + 0.12 * rx, cy - 0.08 * ry), (cx + rx * 0.9, cy - ry * 0.55)], fill=255, width=int(1.8 * S))
    out = over(out, paint(ImageChops.multiply(ck, shrink(M, 10)), darkc(lo, 0.6), 0.8))
    return out


def mat_concrete(sk, c, M, hi, lo):
    big, sm = _n(sk, c, 30, _pseed(c, 241), 1.6), _n(sk, c, 3, _pseed(c, 242), 1.8)
    t = np.clip(0.5 + (big - 0.5) * 0.5 + (sm - 0.5) * 0.35, 0, 1)
    arr = _cmap(t, [(0, lo), (1, lighten(hi, 0.2))])
    pores = A(blur(rng_noise(c.w, c.h, _pseed(c, 243)), 0.7))
    arr *= np.where(pores < 0.2, 0.45, 1.0)[..., None]
    out = fin(sk, c, M, _img(arr), lo, bev=(0.5, 0.55, 3.4), rim=0.7)
    x0, y0, x1, y1, cx, cy = _bb(M)
    h = c.new()
    d = ImageDraw.Draw(h)
    for px in (x0 + (x1 - x0) * 0.16, x1 - (x1 - x0) * 0.16):
        d.ellipse([px - 6 * S, y0 + (y1 - y0) * 0.5 - 6 * S, px + 6 * S, y0 + (y1 - y0) * 0.5 + 6 * S], fill=255)
    h = ImageChops.multiply(h, M)
    out = over(out, paint(h, (30, 30, 34), 0.85))
    return over(out, paint(ImageChops.subtract(shift(h, 0, 1.3), h), (255, 255, 255), 0.3))


def mat_meshgrad(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    x0, y0, x1, y1, cx, cy = _bb(M)
    w, h = x1 - x0, y1 - y0
    pal = sk.gem or [hi, lo, lighten(hi, 0.4), sk.acc or lighten(lo, 0.4)]
    rnd = random.Random(_pseed(c, 251))
    num = np.zeros((c.h, c.w, 3), np.float32)
    den = np.full((c.h, c.w), 1e-4, np.float32)
    for i, col in enumerate(pal):
        px, py = x0 + w * rnd.uniform(0.1, 0.9), y0 + h * rnd.uniform(0.1, 0.9)
        wgt = np.exp(-(((xx - px) ** 2 + (yy - py) ** 2) / (max(w, h) * 0.45) ** 2))
        num += wgt[..., None] * np.array(col[:3], np.float32)
        den += wgt
    arr = num / den[..., None]
    out = shadow_of(c, M, 6, 10, 0.3, 0, darkc(lo, 0.6))
    out = over(out, clip(tone(_img(arr), 0.84 if c.p else 1.0), M))
    out = over(out, paint(ring_of(M, 2.6), (255, 255, 255), 0.55))
    out = over(out, gloss_top(c, M, 0.3, 2.0, 10))
    return over(out, inner_shadow(M, 0, 6, 10, darkc(lo, 0.5), 0.25))


MATS.update({k[4:]: v for k, v in list(globals().items()) if k.startswith("mat_") and callable(v)})


# ---- wave 2 skins ------------------------------------------------------------------------------------------

def mono(hi, lo):
    return {"main": (hi, lo)}


def fcols(first, mainhi, a=0.2, b=0.25):
    return {"main": pr(mainhi, a, b)}


CRAFT = ["clean", "material"]

# jewels, glass and stone
form(id="pad-stained-glass", name="Stained Glass", desc="Jewel-tone leaded panes in pointed-arch windows, lit from behind, with gilded symbols.",
     cats=["ornate", "material"], pro=True, featured=True, press="glow", opacity=0.94, backdrop=((20, 16, 40), (52, 36, 84)), scene=2,
     mat="stained", gem=[(206, 52, 60), (52, 110, 200), (236, 186, 52), (48, 150, 96), (140, 70, 170)], face="arch", dpad="petal", sh="tab",
     small="tab", stick_shape="arch", stick_style="ring", base_shape="octagon", base_style="ticks", well_a=0.3, glyph="gilt", sym_thick=1.1,
     P=mono((230, 210, 160), (90, 70, 60)), mark_col=(246, 206, 96), tags=["glass", "cathedral"])

form(id="pad-rose-window", name="Rose Window", desc="Eight-point star panes in ruby, sapphire and gold, outlined like a cathedral rose window.",
     cats=["ornate"], pro=False, press="glow", opacity=0.94, backdrop=((14, 18, 44), (40, 30, 80)), scene=2,
     mat="stained", gem=[(60, 100, 210), (222, 60, 76), (244, 204, 70), (70, 170, 120)], face="octstar", dpad="house", sh="hexbar", small="hexbar",
     stick_shape="octstar", stick_style="reticle", base_shape="octagon", base_style="cross", well_a=0.3, glyph="hollow", sym_thick=1.2,
     P=mono((200, 180, 150), (60, 50, 50)), mark_col=(255, 250, 235), tags=["glass", "star"])

form(id="pad-enamel-pin", name="Enamel Pin", desc="Glossy hard-enamel pentagon badges set in raised golden metal, with bold sticker-style symbols.",
     cats=["playful", "material"], pro=False, press="spring", opacity=0.96, backdrop=((30, 34, 52), (62, 70, 98)), scene=0,
     mat="cloison", face="pent", dpad="arm", sh="ticket", small="ticket", stick_shape="pent", stick_style="dome", base_shape="hex", base_style="dots",
     well_a=0.4, glyph="pop", sym_thick=1.1, acc=(228, 180, 74),
     P=fcols((70, 130, 240), (66, 66, 80)) | {"circle": pr((236, 70, 84), 0.1, 0.25), "square": pr((240, 130, 200), 0.1, 0.25),
                                              "triangle": pr((60, 200, 140), 0.1, 0.25), "cross": pr((70, 130, 240), 0.1, 0.25)},
     mark_col=(255, 255, 255), tags=["badge", "gold"])

form(id="pad-jade-disc", name="Jade Disc", desc="Polished translucent jade with soft cloudy veins and engraved symbols, calm and cool to the touch.",
     cats=["material", "ornate"], pro=True, press="shrink", opacity=0.95, backdrop=((8, 30, 28), (20, 66, 58)), scene=1,
     mat="jade", face="circle", dpad="bar", sh="pill", small="pill", stick_shape="circle", stick_style="concave", base_shape="circle",
     base_style="well", well_a=0.45, glyph="engrave", sym_thick=1.0, P=mono((120, 214, 160), (20, 88, 68)), mark_col=(214, 250, 230),
     tags=["stone", "green"])

form(id="pad-obsidian-shard", name="Obsidian Shard", desc="Razor-edged volcanic glass kites with sharp conchoidal reflections and cold blue glowing symbols.",
     cats=["material", "neon"], pro=True, press="glow", opacity=0.95, backdrop=((8, 8, 18), (26, 24, 50)), scene=2,
     mat="obsidian", face="kite", dpad="house", sh="slant", small="slant", stick_shape="kite", stick_style="ring", base_shape="octagon",
     base_style="ticks", well_a=0.4, glyph="glow", sym_thick=0.9, P=mono((64, 62, 84), (6, 6, 12)), mark_col=(150, 196, 255),
     tags=["glass", "dark"])

form(id="pad-amber-fossil", name="Amber Fossil", desc="Honey-gold teardrops of ancient resin with trapped bubbles, specks and plant fragments.",
     cats=["material"], pro=True, press="grow", opacity=0.95, backdrop=((34, 18, 6), (84, 46, 14)), scene=3,
     mat="amber", face="drop", dpad="petal", sh="trap", small="trap", stick_shape="drop", stick_style="dome", base_shape="circle", base_style="dots",
     well_a=0.4, glyph="satin", sym_thick=1.0, P=mono((255, 196, 70), (140, 62, 6)), mark_col=(96, 42, 8), tags=["resin", "warm"])

form(id="pad-rusted-iron", name="Rusted Iron", desc="Weathered cog plates with orange rust blooms, pitted metal and stencilled bone-white symbols.",
     cats=["material", "retro"], pro=False, press="shrink", opacity=0.95, backdrop=((26, 22, 20), (60, 48, 40)), scene=3,
     mat="rust", face="cog", dpad="key", sh="rect", small="rect", stick_shape="cog", stick_style="grooves", base_shape="octagon", base_style="ticks",
     well_a=0.5, glyph="stencil", sym_thick=1.1, P=mono((220, 116, 50), (74, 36, 22)), mark_col=(236, 228, 212), tags=["metal", "grunge"])

form(id="pad-chrome-mirror", name="Chrome Mirror", desc="Show-car chrome with a bright horizon reflection, octagonal d-pad and embossed dark symbols.",
     cats=["material", "retro"], pro=True, press="grow", opacity=0.96, backdrop=((26, 30, 44), (70, 82, 112)), scene=4,
     mat="chrome", face="circle", dpad="oct", sh="trap", small="trap", stick_shape="circle", stick_style="dome", base_shape="circle", base_style="ticks",
     well_a=0.4, glyph="emboss", sym_thick=1.0, P=mono((200, 216, 240), (40, 50, 72)), mark_col=(28, 34, 50), tags=["metal", "shiny"])

form(id="pad-holo-foil", name="Holo Foil", desc="Crinkled prismatic foil with fine diffraction lines and glitter, like a collectible trading card.",
     cats=["playful", "neon"], pro=True, press="glow", opacity=0.95, backdrop=((14, 12, 34), (46, 30, 82)), scene=2,
     mat="foil", face="hexf", dpad="house", sh="ticket", small="ticket", stick_shape="hexf", stick_style="ridges", base_shape="hex", base_style="dots",
     well_a=0.4, glyph="shadow", sym_thick=1.0, P=mono((140, 96, 220), (22, 16, 46)), mark_col=(255, 255, 255), tags=["rainbow", "foil"])

form(id="pad-velvet-night", name="Velvet Night", desc="Deep plush velvet cushions with light-catching edges and gilded symbols, rich and soft.",
     cats=["ornate", "material"], pro=True, press="shrink", opacity=0.95, backdrop=((24, 8, 22), (64, 20, 52)), scene=0,
     mat="velvet", face="tile", dpad="petal", sh="pill", small="pill", stick_shape="tile", stick_style="dimples", base_shape="tile", base_style="none",
     well_a=0, glyph="gilt", sym_thick=1.05, P=mono((168, 42, 100), (40, 8, 30)), mark_col=(240, 196, 98), tags=["fabric", "luxury"])

form(id="pad-denim-stitch", name="Denim Stitch", desc="Faded indigo twill squares with worn edges and bright orange topstitching, like favourite jeans.",
     cats=["material", "playful"], pro=False, press="shrink", opacity=0.95, backdrop=((20, 28, 48), (44, 62, 100)), scene=4,
     mat="denim", face="tile", dpad="key", sh="rect", small="rect", stick_shape="tile", stick_style="ring", base_shape="tile", base_style="dots",
     well_a=0.35, glyph="stitch", sym_thick=1.0, acc=(238, 160, 70), P=mono((88, 128, 188), (22, 42, 88)), mark_col=(244, 170, 80),
     tags=["fabric", "casual"])

form(id="pad-cork-board", name="Cork Board", desc="Speckled natural cork coasters with rubber-stamped dark symbols, warm and earthy.",
     cats=["material"], pro=False, press="shrink", opacity=0.95, backdrop=((38, 26, 16), (84, 60, 38)), scene=3,
     mat="cork", face="circle", dpad="oct", sh="leaf", small="leaf", stick_shape="circle", stick_style="dimples", base_shape="circle", base_style="ticks",
     well_a=0.4, glyph="stamp", sym_thick=1.15, P=mono((212, 166, 110), (118, 78, 42)), mark_col=(60, 34, 18), tags=["natural", "earthy"])

form(id="pad-terrazzo", name="Terrazzo", desc="Polished speckled terrazzo with coloured stone chips and playful two-tone offset symbols.",
     cats=["material", "playful"], pro=False, press="grow", opacity=0.96, backdrop=((36, 38, 52), (84, 88, 110)), scene=4,
     mat="terrazzo", face="squircle", dpad="dot", sh="pill", small="pill", stick_shape="squircle", stick_style="ring", base_shape="squircle",
     base_style="none", well_a=0, glyph="duo", sym_thick=1.1, duo=(232, 96, 78), gem=[(214, 90, 76), (70, 122, 112), (230, 188, 80), (60, 60, 74)],
     P=mono((238, 232, 222), (198, 188, 172)), mark_col=(40, 42, 56), tags=["stone", "speckle"])

form(id="pad-zellige-star", name="Zellige Star", desc="Hand-cut glazed tile mosaic in deep blue and gold, with eight-point stars and fine white grout.",
     cats=["ornate"], pro=True, press="shrink", opacity=0.96, backdrop=((8, 24, 44), (20, 56, 92)), scene=1,
     mat="zellige", face="octagon", dpad="dia", sh="hexbar", small="hexbar", stick_shape="octagon", stick_style="reticle", base_shape="octagon",
     base_style="ticks", well_a=0.3, glyph="emboss", sym_thick=1.0, gem=[(28, 92, 144), (236, 200, 96), (36, 150, 124)],
     P=mono((28, 92, 144), (10, 40, 72)), mark_col=(252, 250, 240), tags=["mosaic", "morocco"])

form(id="pad-studded-plate", name="Studded Plate", desc="Chunky plates covered with round domed studs in signal orange, grippy and toy-like.",
     cats=["playful", "material"], pro=False, press="spring", opacity=0.96, backdrop=((20, 24, 38), (44, 52, 78)), scene=0,
     mat="studs", face="chamfer", dpad="arm", sh="rect", small="rect", stick_shape="chamfer", stick_style="cap", base_shape="chamfer", base_style="well",
     well_a=0.4, glyph="shadow", sym_thick=1.2, P=mono((240, 126, 56), (150, 60, 22)), mark_col=(255, 255, 255), tags=["toy", "orange"])

# sweets and food
form(id="pad-hard-candy", name="Hard Candy", desc="Glassy swirled lollipop discs in red, blue, orange and green, with white sticker-outlined symbols.",
     cats=["sweet", "playful"], pro=False, featured=True, press="spring", opacity=0.96, backdrop=((54, 26, 58), (120, 60, 120)), scene=0,
     mat="swirl", face="circle", dpad="dot", sh="pill", small="pill", stick_shape="circle", stick_style="dome", base_shape="circle", base_style="dots",
     well_a=0.35, glyph="sticker", sym_thick=1.1,
     P=fcols((64, 120, 232), (236, 64, 90)) | {"circle": pr((236, 64, 90), 0.1, 0.25), "square": pr((250, 160, 60), 0.1, 0.25),
                                               "triangle": pr((60, 190, 120), 0.1, 0.25), "cross": pr((64, 120, 232), 0.1, 0.25)},
     sym_colors={"cross": (28, 70, 170), "circle": (170, 24, 56), "square": (200, 100, 10), "triangle": (20, 120, 70)},
     mark_col=(170, 24, 56), tags=["candy", "swirl"])

form(id="pad-citrus-slice", name="Citrus Slice", desc="Juicy fruit-slice discs with rind, pith and glistening segments in lime, orange, lemon and grapefruit.",
     cats=["sweet", "playful"], pro=False, press="grow", opacity=0.96, backdrop=((24, 44, 36), (60, 110, 84)), scene=1,
     mat="citrus", face="circle", dpad="sector", sh="pill", small="pill", stick_shape="circle", stick_style="ring", base_shape="circle", base_style="ticks",
     well_a=0.3, glyph="shadow", sym_thick=1.15,
     P=fcols((150, 206, 60), (255, 150, 40), 0.0, 0.25) | {"circle": pr((255, 150, 40), 0.0, 0.3), "square": pr((250, 214, 70), 0.0, 0.3),
                                                           "triangle": pr((244, 100, 100), 0.0, 0.3), "cross": pr((150, 206, 60), 0.0, 0.3)},
     mark_col=(255, 255, 255), tags=["fruit", "fresh"])

form(id="pad-cookie-crumb", name="Cookie Crumb", desc="Golden-baked scalloped cookies with chocolate chunks, docking dots and engraved crumbs-brown symbols.",
     cats=["sweet", "playful"], pro=False, press="grow", opacity=0.96, backdrop=((42, 26, 16), (96, 62, 38)), scene=3,
     mat="cookie", face="scallop", dpad="oct", sh="leaf", small="leaf", stick_shape="scallop", stick_style="dimples", base_shape="circle",
     base_style="dots", well_a=0.4, glyph="engrave", sym_thick=1.1, acc=(74, 44, 30), P=mono((238, 190, 106), (150, 90, 40)), mark_col=(120, 70, 30),
     tags=["biscuit", "baked"])

form(id="pad-ice-cream", name="Ice Cream", desc="Pastel frosted scoops dusted with rainbow sprinkles and bold comic-outlined symbols.",
     cats=["sweet", "playful"], pro=True, press="spring", opacity=0.96, backdrop=((60, 38, 66), (130, 92, 130)), scene=0,
     mat="sprinkles", face="squircle", dpad="petal", sh="pill", small="pill", stick_shape="squircle", stick_style="dome", base_shape="squircle",
     base_style="dots", well_a=0.3, glyph="pop", sym_thick=1.15,
     P=fcols((190, 232, 255), (255, 176, 206), 0.0, 0.2) | {"circle": pr((255, 160, 190), 0.0, 0.2), "square": pr((255, 238, 160), 0.0, 0.2),
                                                           "triangle": pr((180, 238, 200), 0.0, 0.2), "cross": pr((178, 218, 255), 0.0, 0.2)},
     mark_col=(255, 255, 255), tags=["sprinkles", "pastel"])

form(id="pad-maki-roll", name="Maki Roll", desc="Abstract sushi rolls: dark seaweed rims, glistening rice grains and coloured fillings.",
     cats=["sweet", "playful"], pro=True, press="grow", opacity=0.96, backdrop=((30, 30, 36), (70, 66, 74)), scene=4,
     mat="maki", face="circle", dpad="bar", sh="ticket", small="ticket", stick_shape="circle", stick_style="ring", base_shape="circle", base_style="none",
     well_a=0, glyph="solid", sym_thick=1.1, sym_scale=0.9,
     P=fcols((140, 200, 90), (240, 110, 90), 0.0, 0.2) | {"circle": pr((230, 70, 76), 0.0, 0.2), "square": pr((250, 200, 90), 0.0, 0.2),
                                                         "triangle": pr((240, 130, 100), 0.0, 0.2), "cross": pr((140, 200, 90), 0.0, 0.2)},
     mark_col=(28, 48, 34), tags=["food", "japan"])

form(id="pad-watermelon", name="Watermelon", desc="Round fruit slices with green rinds, red flesh and black seeds that scatter across every button.",
     cats=["sweet", "playful"], pro=False, press="grow", opacity=0.96, backdrop=((20, 46, 34), (44, 100, 70)), scene=1,
     mat="melon", face="octagon", dpad="petal", sh="trap", small="trap", stick_shape="circle", stick_style="dimples", base_shape="circle",
     base_style="ticks", well_a=0.4, glyph="satin", sym_thick=1.1, P=mono((244, 74, 96), (26, 110, 56)), mark_col=(255, 244, 240),
     tags=["fruit", "summer"])

form(id="pad-honeycomb", name="Honeycomb", desc="Translucent amber honeycomb cells with waxy walls and glowing honey-gold satin symbols.",
     cats=["material", "sweet"], pro=False, press="grow", opacity=0.96, backdrop=((38, 22, 4), (96, 58, 8)), scene=3,
     mat="honey", face="hexf", dpad="house", sh="hexbar", small="hexbar", stick_shape="hexf", stick_style="ring", base_shape="hex", base_style="ticks",
     well_a=0.35, glyph="satin", sym_thick=1.1, P=mono((255, 188, 44), (142, 74, 0)), mark_col=(112, 54, 4), tags=["hex", "gold"])

# retro displays, drafting, street
form(id="pad-gb-2bit", name="Two-Bit Handheld", desc="Four olive-green shades, chunky pixels and checker dithering, like an old pocket handheld.",
     cats=["retro"], pro=False, press="shrink", opacity=0.97, backdrop=((100, 120, 76), (60, 80, 52)), scene=1, pixel=True, pgrid=8,
     face="tile", dpad="arm", sh="rect", small="rect", stick_shape="tile", stick_style="ring", dither=True, sym_thick=1.4, sym_scale=1.05,
     OUT=(16, 36, 24), PAL={"main": ((200, 224, 150), (126, 160, 90), (56, 88, 52))}, mark_col=(20, 44, 28), tags=["handheld", "pixel"])

form(id="pad-crt-phosphor", name="CRT Phosphor", desc="Curved tube glass with scanlines, RGB triads, a warm phosphor bloom and glowing marks.",
     cats=["retro", "neon"], pro=True, press="glow", opacity=0.95, backdrop=((6, 12, 12), (18, 40, 34)), scene=1,
     mat="crt", face="squircle", dpad="oct", sh="rect", small="rect", stick_shape="squircle", stick_style="concave", base_shape="squircle",
     base_style="cross", well_a=0.45, glyph="glow", sym_thick=0.85, P=mono((120, 255, 170), (6, 40, 24)), mark_col=(190, 255, 220),
     tags=["crt", "scanlines"])

form(id="pad-oscilloscope", name="Oscilloscope", desc="Lab-bench scope screens with graticule grids, metal bezels and a glowing Lissajous trace.",
     cats=["retro", "neon"], pro=True, press="glow", opacity=0.95, backdrop=((6, 14, 12), (16, 38, 30)), scene=1,
     mat="scope", face="octagon", dpad="house", sh="chamfer", small="chamfer", stick_shape="octagon", stick_style="reticle", base_shape="octagon",
     base_style="ticks", well_a=0.4, glyph="glow", sym_thick=0.8, P=mono((100, 255, 160), (2, 26, 18)), mark_col=(210, 255, 230),
     tags=["lab", "vector"])

form(id="pad-vellum-draft", name="Vellum Draft", desc="Cream drafting vellum with compass arcs, centre lines and red dimension arrows; ink-blue outlined marks.",
     cats=["retro", "clean"], pro=False, press="shrink", opacity=0.96, backdrop=((28, 44, 68), (58, 84, 120)), scene=4,
     mat="drafting", face="hex", dpad="tri", sh="hexbar", small="hexbar", stick_shape="hex", stick_style="reticle", base_shape="hex",
     base_style="cross", well_a=0, glyph="hollow", sym_thick=1.0, line=2.4, P=mono((240, 240, 228), (52, 86, 150)), mark_col=(40, 76, 150),
     tags=["blueprint", "paper"])

form(id="pad-spray-tag", name="Spray Tag", desc="Airbrushed neon paint with overspray speckle, soft stencil edges and dripping spray-can symbols.",
     cats=["playful", "neon"], pro=True, press="grow", opacity=0.96, backdrop=((22, 22, 30), (52, 50, 68)), scene=2,
     mat="spray", face="circle", dpad="arm", sh="pill", small="pill", stick_shape="circle", stick_style="ring", base_shape="circle", base_style="dots",
     well_a=0.3, glyph="spray", sym_thick=1.2, P=fcols((60, 200, 255), (255, 70, 150)) | {"circle": pr((255, 70, 120), 0.0, 0.3), "square": pr((255, 220, 60), 0.0, 0.3),
                                                                                         "triangle": pr((110, 255, 120), 0.0, 0.3), "cross": pr((60, 200, 255), 0.0, 0.3)},
     mark_col=(255, 255, 255), tags=["graffiti", "street"])

# heritage and ornament
form(id="pad-tribal-totem", name="Tribal Totem", desc="Carved dark plates framed by zig-zag diamond bands in sunset ochre, with engraved symbols.",
     cats=["ornate"], pro=False, press="shrink", opacity=0.96, backdrop=((30, 18, 12), (72, 44, 26)), scene=3,
     mat="tribal", face="shield", dpad="wedge", sh="chamfer", small="chamfer", stick_shape="shield", stick_style="ring", base_shape="octagon",
     base_style="dots", well_a=0.4, glyph="engrave", sym_thick=1.1, P=mono((236, 172, 84), (48, 28, 20)), mark_col=(240, 190, 110),
     tags=["ornament", "line-art"])

form(id="pad-art-nouveau", name="Art Nouveau", desc="Sage-green enamel with whiplash leaf curves, golden outlines and gilded symbols from the belle epoque.",
     cats=["ornate"], pro=True, press="grow", opacity=0.96, backdrop=((20, 34, 30), (52, 80, 66)), scene=1,
     mat="nouveau", face="flower", dpad="tri", sh="arch", small="arch", stick_shape="flower", stick_style="ring", base_shape="circle", base_style="ticks",
     well_a=0.3, glyph="gilt", sym_thick=1.0, acc=(214, 172, 78), P=mono((190, 208, 156), (62, 94, 74)), mark_col=(214, 172, 78),
     tags=["floral", "gold"])

form(id="pad-gothic-cathedral", name="Gothic Cathedral", desc="Pointed-arch stone buttons with carved trefoil tracery and weathered grey masonry.",
     cats=["ornate", "material"], pro=True, press="shrink", opacity=0.96, backdrop=((18, 20, 30), (50, 54, 74)), scene=2,
     mat="gothic", face="gothic", dpad="house", sh="tab", small="tab", stick_shape="circle", stick_style="grooves", base_shape="octagon", base_style="well",
     well_a=0.5, glyph="engrave", sym_thick=1.1, P=mono((170, 172, 182), (66, 68, 82)), mark_col=(44, 46, 60), tags=["stone", "medieval"])

form(id="pad-brass-gauge", name="Brass Gauge", desc="Steampunk pressure dials: riveted brass bezels, cream faces, tick scales and red needles.",
     cats=["retro", "ornate"], pro=True, featured=True, press="spring", opacity=0.96, backdrop=((30, 20, 12), (74, 50, 28)), scene=3,
     mat="gauge", face="circle", dpad="dot", sh="hexbar", small="hexbar", stick_shape="circle", stick_style="cap", base_shape="circle", base_style="ticks",
     well_a=0.4, glyph="stamp", sym_thick=1.15, acc=(196, 148, 66), P=mono((242, 234, 208), (148, 126, 96)), mark_col=(50, 34, 20),
     tags=["steampunk", "brass"])

form(id="pad-astronaut", name="Astronaut Panel", desc="White space-suit fabric with stitched seams, an orange stripe and mission-patch screws.",
     cats=["clean", "retro"], pro=False, press="shrink", opacity=0.96, backdrop=((10, 14, 30), (30, 40, 76)), scene=2,
     mat="astro", face="squircle", dpad="key", sh="rect", small="rect", stick_shape="squircle", stick_style="ring", base_shape="squircle",
     base_style="ticks", well_a=0.4, glyph="shadow", sym_thick=1.1, acc=(255, 112, 32), P=mono((242, 244, 248), (166, 174, 190)),
     mark_col=(30, 44, 80), tags=["space", "white"])

form(id="pad-mecha-armor", name="Mecha Armor", desc="Layered robot armour plates with panel seams, hazard-striped edges and stencilled markings.",
     cats=["playful", "material"], pro=True, press="spring", opacity=0.96, backdrop=((14, 18, 30), (36, 44, 70)), scene=0,
     mat="mecha", face="hexf", dpad="wedge", sh="slant", small="slant", stick_shape="hexf", stick_style="ridges", base_shape="hex", base_style="ticks",
     well_a=0.4, glyph="stencil", sym_thick=1.2,
     P=fcols((60, 124, 232), (232, 232, 238), 0.2, 0.3) | {"circle": pr((228, 60, 60), 0.2, 0.3), "square": pr((250, 198, 50), 0.2, 0.3),
                                                           "triangle": pr((60, 180, 110), 0.2, 0.3), "cross": pr((60, 124, 232), 0.2, 0.3)},
     sym_colors={"cross": (255, 255, 255), "circle": (255, 255, 255), "square": (40, 40, 50), "triangle": (255, 255, 255)},
     mark_col=(34, 40, 58), tags=["robot", "armor"])

form(id="pad-urushi-lacquer", name="Urushi Lacquer", desc="Deep red and black lacquer with a mirror gloss, gold maki-e dust and gilded symbols.",
     cats=["ornate", "material"], pro=True, featured=True, press="grow", opacity=0.96, backdrop=((16, 6, 8), (50, 18, 22)), scene=0,
     mat="urushi", face="squircle", dpad="capsule", sh="pill", small="pill", stick_shape="squircle", stick_style="ring", base_shape="squircle",
     base_style="ticks", well_a=0.35, glyph="gilt", sym_thick=1.05, acc=(232, 186, 84), P=mono((158, 26, 32), (30, 4, 8)), mark_col=(236, 190, 88),
     tags=["japan", "gold"])

form(id="pad-blue-willow", name="Blue Willow", desc="Glazed white porcelain with hand-painted cobalt petals, scroll rings and brush-stroke symbols.",
     cats=["ornate"], pro=False, press="grow", opacity=0.96, backdrop=((14, 28, 62), (36, 64, 120)), scene=2,
     mat="porcelain", face="circle", dpad="petal", sh="leaf", small="leaf", stick_shape="circle", stick_style="ring", base_shape="circle", base_style="dots",
     well_a=0.3, glyph="brush", sym_thick=1.1, P=mono((28, 78, 170), (220, 228, 244)), mark_col=(28, 70, 160), tags=["china", "cobalt"])

form(id="pad-vyshyvanka", name="Vyshyvanka", desc="Cross-stitch embroidery in red and black on linen, inspired by traditional Ukrainian geometric rhombus patterns.",
     cats=["ornate"], pro=False, press="shrink", opacity=0.96, backdrop=((38, 30, 26), (86, 68, 58)), scene=3,
     mat="crossstitch", face="tile", dpad="key", sh="rect", small="rect", stick_shape="tile", stick_style="ring", base_shape="tile", base_style="none",
     well_a=0, glyph="shadow", sym_thick=1.15, acc=(190, 32, 44), P=mono((246, 240, 226), (200, 188, 164)), mark_col=(120, 18, 28),
     tags=["embroidery", "folk"])

form(id="pad-celtic-knot", name="Celtic Knot", desc="Interwoven gold and green ribbons on a dark disc, over and under in an endless Celtic braid.",
     cats=["ornate"], pro=True, press="grow", opacity=0.96, backdrop=((10, 26, 20), (30, 62, 44)), scene=1,
     mat="celtic", face="circle", dpad="dot", sh="hexbar", small="hexbar", stick_shape="circle", stick_style="concave", base_shape="circle",
     base_style="dots", well_a=0.35, glyph="inlay", sym_thick=1.05, acc=(216, 178, 84), P=mono((92, 170, 102), (16, 48, 30)), mark_col=(228, 192, 100),
     tags=["knot", "braid"])

# minimal and accessibility
form(id="pad-dot-touch", name="Dot Touch", desc="Tiny glowing dots instead of buttons: a nearly invisible layout for big tablet screens.",
     cats=["minimal", "clean"], pro=False, featured=True, press="grow", opacity=0.92, backdrop=((26, 34, 54), (58, 74, 108)), scene=4,
     mat="pinpoint", face="circle", dpad="dot", sh="pill", small="pill", stick_shape="circle", stick_style="none", base_shape="circle", base_style="none",
     well_a=0, glyph="solid", sym_thick=0.7, sym_scale=0.5,
     P=fcols((90, 160, 255), (236, 240, 248), 0.15, 0.2) | {"circle": pr((255, 96, 110), 0.15, 0.2), "square": pr((255, 150, 220), 0.15, 0.2),
                                                           "triangle": pr((90, 220, 170), 0.15, 0.2), "cross": pr((90, 160, 255), 0.15, 0.2)},
     mark_col=(14, 20, 34), tags=["tablet", "tiny"])

form(id="pad-corner-ticks", name="Corner Ticks", desc="Only four hairline corner marks per button: an ultra-minimal outline for tablets and big screens.",
     cats=["minimal", "clean"], pro=False, press="grow", opacity=0.9, backdrop=((40, 54, 76), (84, 104, 134)), scene=4,
     mat="corners", face="tile", dpad="key", sh="rect", small="rect", stick_shape="tile", stick_style="reticle", base_shape="tile", base_style="none",
     well_a=0, glyph="solid", sym_thick=0.45, line=3.0, P=mono((240, 244, 252), (240, 244, 252)), mark_col=(240, 244, 252), tags=["tablet", "outline"])

form(id="pad-mirror-twin", name="Mirror Twin", desc="Perfectly symmetric concentric rings and neutral colours, equally comfortable for left and right hands.",
     cats=["accessibility", "clean"], pro=False, press="shrink", opacity=0.95, backdrop=((30, 34, 52), (62, 70, 100)), scene=0,
     mat="target", face="circle", dpad="capsule", sh="pill", small="pill", stick_shape="circle", stick_style="ring", base_shape="circle", base_style="ticks",
     well_a=0.4, glyph="solid", sym_thick=1.0, P=mono((236, 240, 248), (70, 94, 150)), mark_col=(22, 34, 78), tags=["left-handed", "symmetric"])

form(id="pad-kids-crayon", name="Kids Crayon", desc="Giant colourful crayon-scribbled stars and big wobbly symbols that small hands can hit every time.",
     cats=["accessibility", "playful"], pro=False, press="spring", opacity=0.96, backdrop=((250, 244, 224), (232, 218, 180)), scene=0,
     mat="crayon", face="star", dpad="petal", sh="pill", small="pill", stick_shape="star", stick_style="ring", base_shape="circle", base_style="dots",
     well_a=0.2, glyph="wobble", sym_thick=1.5, sym_scale=1.2,
     P=fcols((50, 130, 255), (255, 120, 60)) | {"circle": pr((255, 70, 70), 0.0, 0.2), "square": pr((255, 200, 40), 0.0, 0.2),
                                                "triangle": pr((60, 200, 90), 0.0, 0.2), "cross": pr((50, 130, 255), 0.0, 0.2)},
     mark_col=(40, 30, 60), tags=["kids", "big"])

form(id="pad-sunlight-bold", name="Sunlight Bold", desc="Thick black frames, heavy symbols and hard shadows on white, readable even in direct sunlight.",
     cats=["accessibility"], pro=False, featured=True, press="shrink", opacity=1.0, backdrop=((196, 214, 232), (146, 170, 198)), scene=4,
     mat="hivis", face="octagon", dpad="key", sh="rect", small="rect", stick_shape="octagon", stick_style="ring", base_shape="octagon", base_style="well",
     well_a=0.2, glyph="solid", sym_thick=1.7, sym_scale=1.15, P=mono((255, 255, 255), (10, 10, 12)), mark_col=(10, 10, 12), tags=["contrast", "outdoor"])

form(id="pad-dyslexia-friendly", name="Easy Read", desc="Cream buttons with heavy, evenly weighted symbols; L buttons in blue, R in orange, with dots counting 1, 2 or 3.",
     cats=["accessibility"], pro=False, press="grow", opacity=0.97, backdrop=((44, 50, 70), (78, 86, 112)), scene=0,
     mat="flat", face="squircle", dpad="capsule", sh="tab", small="tab", stick_shape="squircle", stick_style="ring", base_shape="squircle", base_style="well",
     well_a=0.2, glyph="dyslexic", sym_thick=1.5, sym_scale=1.05, line=3.5, P=mono((252, 246, 214), (214, 202, 150)), mark_col=(24, 24, 30),
     sym_colors={"l1": (24, 72, 190), "l2": (24, 72, 190), "l3": (24, 72, 190), "r1": (200, 84, 8), "r2": (200, 84, 8), "r3": (200, 84, 8)},
     tags=["dyslexia", "readable"])

# space, nature, light
form(id="pad-pearl-nacre", name="Pearl Nacre", desc="Mother-of-pearl pentagons with shifting pastel iridescence and softly embossed symbols.",
     cats=["material", "clean"], pro=True, press="grow", opacity=0.96, backdrop=((34, 32, 58), (84, 80, 124)), scene=2,
     mat="nacre", face="pent", dpad="bar", sh="leaf", small="leaf", stick_shape="pent", stick_style="dome", base_shape="circle", base_style="dots",
     well_a=0.3, glyph="emboss", sym_thick=1.05, P=mono((238, 228, 242), (120, 108, 140)), mark_col=(112, 88, 150), tags=["pearl", "iridescent"])

form(id="pad-meteorite", name="Meteorite", desc="Space iron with crossing crystal lines, thumbprint pits and a dark fusion crust.",
     cats=["material"], pro=True, press="shrink", opacity=0.96, backdrop=((12, 14, 24), (36, 40, 62)), scene=2,
     mat="meteorite", face="hexf", dpad="oct", sh="trap", small="trap", stick_shape="hexf", stick_style="dimples", base_shape="hex", base_style="well",
     well_a=0.5, glyph="engrave", sym_thick=1.05, P=mono((156, 158, 170), (34, 34, 42)), mark_col=(226, 230, 238), tags=["space", "iron"])

form(id="pad-aurora-glass", name="Aurora Glass", desc="Dark glass with green, violet and blue northern-light curtains rippling across and glowing marks.",
     cats=["neon", "material"], pro=True, press="glow", opacity=0.95, backdrop=((6, 12, 28), (14, 40, 60)), scene=2,
     mat="aurora", face="squircle", dpad="bar", sh="pill", small="pill", stick_shape="squircle", stick_style="ring", base_shape="squircle", base_style="cross",
     well_a=0.3, glyph="glow", sym_thick=0.9, P=mono((150, 90, 255), (6, 14, 36)), mark_col=(206, 255, 240), tags=["sky", "glow"])

form(id="pad-galaxy-dust", name="Galaxy Dust", desc="Starfield sparkle buttons filled with pink and violet nebula clouds, with two-tone offset symbols.",
     cats=["neon", "playful"], pro=False, press="glow", opacity=0.95, backdrop=((6, 6, 22), (26, 16, 56)), scene=2,
     mat="galaxy", face="sparkle", dpad="dia", sh="chamfer", small="chamfer", stick_shape="sparkle", stick_style="reticle", base_shape="octagon",
     base_style="dots", well_a=0.3, glyph="duo", sym_thick=1.0, duo=(110, 150, 255), acc=(255, 120, 190), P=mono((96, 62, 210), (6, 8, 26)),
     mark_col=(255, 255, 255), tags=["space", "nebula"])

form(id="pad-wax-seal", name="Wax Seal", desc="Irregular blobs of red sealing wax with a pressed rim, a mirror gloss and cut-in symbols.",
     cats=["ornate", "material"], pro=True, press="shrink", opacity=0.96, backdrop=((30, 18, 14), (74, 46, 38)), scene=3,
     mat="waxseal", face="hexf", dpad="dot", sh="pill", small="pill", stick_shape="hexf", stick_style="concave", base_shape="hex", base_style="none",
     well_a=0, glyph="cutout", sym_thick=1.1, P=mono((204, 38, 48), (84, 10, 18)), mark_col=(250, 200, 196), tags=["letter", "red"])

form(id="pad-tree-rings", name="Tree Rings", desc="Sliced log discs with growth rings, a rough bark rim and a crack, stamped with dark symbols.",
     cats=["material"], pro=False, press="shrink", opacity=0.96, backdrop=((26, 30, 20), (62, 70, 44)), scene=1,
     mat="treering", face="circle", dpad="bar", sh="leaf", small="leaf", stick_shape="circle", stick_style="grooves", base_shape="circle", base_style="ticks",
     well_a=0.3, glyph="pop", sym_thick=1.1, P=mono((226, 178, 116), (106, 64, 32)), mark_col=(255, 250, 240), tags=["wood", "nature"])

form(id="pad-concrete-slab", name="Concrete Slab", desc="Brutalist concrete tiles with pores, form-tie holes and stencilled white symbols.",
     cats=["material", "retro"], pro=False, press="shrink", opacity=0.96, backdrop=((30, 32, 36), (74, 78, 86)), scene=0,
     mat="concrete", face="tile", dpad="bar", sh="rect", small="rect", stick_shape="tile", stick_style="grooves", base_shape="tile", base_style="ticks",
     well_a=0.4, glyph="stencil", sym_thick=1.15, P=mono((172, 174, 176), (80, 82, 88)), mark_col=(250, 250, 250), tags=["grey", "industrial"])

form(id="pad-gradient-orb", name="Gradient Orb", desc="Soft glassy orbs filled with blurred pastel colour fields and thin white symbols.",
     cats=["clean", "playful"], pro=False, press="grow", opacity=0.95, backdrop=((30, 30, 66), (86, 66, 130)), scene=2,
     mat="meshgrad", face="circle", dpad="petal", sh="pill", small="pill", stick_shape="circle", stick_style="ring", base_shape="circle", base_style="none",
     well_a=0, glyph="solid", sym_thick=0.75, gem=[(255, 126, 166), (126, 146, 255), (126, 232, 212), (255, 204, 126)],
     P=mono((255, 140, 170), (92, 62, 182)), mark_col=(255, 255, 255), tags=["gradient", "soft"])

# extra colourways on existing materials with new shapes and glyph styles
form(id="pad-ruby-cut", name="Ruby Cut", desc="Faceted ruby stars with a bright table cut and gilded symbols, like gems on velvet.",
     cats=["material", "ornate"], pro=True, press="glow", opacity=0.95, backdrop=((26, 8, 20), (66, 18, 44)), scene=0,
     mat="crystal", face="star", dpad="oct", sh="slant", small="slant", stick_shape="star", stick_style="dome", base_shape="octagon", base_style="ticks",
     well_a=0.35, glyph="gilt", sym_thick=1.0, facets=12, P=mono((236, 40, 84), (90, 6, 32)), mark_col=(250, 206, 110), tags=["gem", "red"])

form(id="pad-gold-leaf", name="Gold Leaf", desc="Hand-beaten gold leaf scallops with hammer dimples and two-tone offset dark symbols.",
     cats=["material", "ornate"], pro=True, press="grow", opacity=0.96, backdrop=((22, 16, 10), (58, 42, 24)), scene=3,
     mat="hammered", face="scallop", dpad="petal", sh="tab", small="tab", stick_shape="scallop", stick_style="cap", base_shape="circle", base_style="well",
     well_a=0.4, glyph="duo", sym_thick=1.05, duo=(120, 70, 10), P=mono((246, 208, 96), (150, 100, 20)), mark_col=(60, 34, 6), tags=["gold", "metal"])

form(id="pad-lapis-lazuli", name="Lapis Lazuli", desc="Deep blue lapis stone arches flecked with gold pyrite, with gilded engraved symbols.",
     cats=["material", "ornate"], pro=False, press="shrink", opacity=0.96, backdrop=((8, 14, 44), (22, 36, 100)), scene=2,
     mat="stone", face="arch", dpad="house", sh="trap", small="trap", stick_shape="arch", stick_style="grooves", base_shape="octagon", base_style="well",
     well_a=0.4, glyph="gilt", sym_thick=1.1, P=mono((52, 84, 190), (10, 20, 80)), mark_col=(240, 200, 100), tags=["stone", "blue"])

form(id="pad-pearl-glaze", name="Pearl Glaze", desc="Pentagonal glazed ivory ceramic with golden crack lines and brush-painted teal symbols.",
     cats=["material", "clean"], pro=False, press="grow", opacity=0.96, backdrop=((14, 36, 40), (34, 80, 86)), scene=1,
     mat="ceramic", face="pent", dpad="bar", sh="trap", small="trap", stick_shape="pent", stick_style="ring", base_shape="circle", base_style="dots",
     well_a=0.3, glyph="brush", sym_thick=1.1, P=mono((242, 234, 214), (186, 164, 124)), mark_col=(24, 84, 92), tags=["ceramic", "teal"])

form(id="pad-dusk-neumorph", name="Dusk Neumorph", desc="Soft extruded plates in dusk violet with punched-out symbols and no hard outlines.",
     cats=["clean", "minimal"], pro=False, press="shrink", opacity=0.95, backdrop=((60, 54, 96), (96, 88, 140)), scene=2,
     mat="neumorph", face="tile", dpad="oct", sh="leaf", small="leaf", stick_shape="tile", stick_style="ring", base_shape="tile", base_style="well",
     well_a=0.2, glyph="cutout", sym_thick=1.0, P=mono((166, 156, 226), (118, 106, 190)), mark_col=(246, 242, 255), tags=["soft", "violet"])

form(id="pad-vapor-gloss", name="Vapor Gloss", desc="Glossy pastel star lollipops in cyan and pink with white sticker symbols, vaporwave style.",
     cats=["playful", "neon"], pro=False, press="spring", opacity=0.96, backdrop=((46, 26, 92), (130, 70, 150)), scene=2,
     mat="gloss", face="star", dpad="petal", sh="pill", small="pill", stick_shape="star", stick_style="dome", base_shape="circle", base_style="dots",
     well_a=0.3, glyph="sticker", sym_thick=1.1,
     P=fcols((90, 220, 240), (255, 130, 200)) | {"circle": pr((255, 120, 200), 0.1, 0.2), "square": pr((190, 150, 255), 0.1, 0.2),
                                                "triangle": pr((120, 240, 200), 0.1, 0.2), "cross": pr((90, 220, 240), 0.1, 0.2)},
     sym_colors={"cross": (0, 110, 140), "circle": (190, 30, 130), "square": (100, 60, 190), "triangle": (10, 130, 90)}, mark_col=(190, 30, 130),
     tags=["vaporwave", "pastel"])

form(id="pad-mint-clay", name="Mint Clay", desc="Hand-pressed mint clay teardrops with soft thumbprints and cut-in symbols.",
     cats=["material", "playful"], pro=False, press="grow", opacity=0.96, backdrop=((20, 44, 44), (60, 100, 96)), scene=1,
     mat="clay", face="drop", dpad="bar", sh="leaf", small="leaf", stick_shape="drop", stick_style="dimples", base_shape="circle", base_style="none",
     well_a=0, glyph="cutout", sym_thick=1.1, P=mono((176, 232, 206), (84, 160, 134)), mark_col=(40, 110, 90), tags=["clay", "mint"])

form(id="pad-ember-glass", name="Ember Glass", desc="Smoked amber glass kites with an inner ember glow and glowing orange symbols.",
     cats=["neon", "material"], pro=True, press="glow", opacity=0.95, backdrop=((26, 8, 4), (74, 24, 10)), scene=3,
     mat="glass", face="kite", dpad="petal", sh="chamfer", small="chamfer", stick_shape="drop", stick_style="ring", base_shape="circle", base_style="ticks",
     well_a=0.3, glyph="glow", sym_thick=0.9, ga=0.34, P=mono((255, 130, 40), (120, 30, 6)), mark_col=(255, 168, 80), tags=["glass", "fire"])

form(id="pad-night-paper", name="Night Paper", desc="Layered midnight-blue paper arches with gentle shadows between each cut sheet and satin symbols.",
     cats=["material", "clean"], pro=False, press="shrink", opacity=0.96, backdrop=((14, 18, 44), (36, 48, 100)), scene=2,
     mat="paper", face="gothic", dpad="petal", sh="tab", small="tab", stick_shape="gothic", stick_style="ring", base_shape="circle", base_style="dots",
     well_a=0.3, glyph="satin", sym_thick=1.05, P=mono((120, 150, 236), (28, 40, 110)), mark_col=(236, 244, 255), tags=["paper", "blue"])

form(id="pad-coral-pop", name="Coral Pop", desc="Puffy inflated coral scallops with glossy highlights and bold white sticker symbols.",
     cats=["playful"], pro=False, press="spring", opacity=0.96, backdrop=((50, 22, 36), (130, 62, 80)), scene=0,
     mat="inflate", face="scallop", dpad="petal", sh="pill", small="pill", stick_shape="scallop", stick_style="dome", base_shape="circle",
     base_style="dots", well_a=0.3, glyph="sticker", sym_thick=1.1, P=mono((255, 124, 112), (190, 50, 70)), mark_col=(210, 50, 70), tags=["puffy", "coral"])

form(id="pad-silk-thread", name="Silk Thread", desc="Satin-stitched embroidery on a linen square, in rose and gold thread with gilded symbols.",
     cats=["ornate", "material"], pro=True, press="shrink", opacity=0.96, backdrop=((38, 24, 36), (92, 60, 82)), scene=0,
     mat="embroidery", face="tile", dpad="bar", sh="ticket", small="ticket", stick_shape="tile", stick_style="ring", base_shape="tile", base_style="none",
     well_a=0, glyph="gilt", sym_thick=1.0, acc=(244, 196, 100), P=mono((226, 120, 150), (110, 34, 66)), mark_col=(246, 206, 112), tags=["thread", "silk"])

form(id="pad-neon-star", name="Neon Star", desc="Star-shaped glass tubes in electric pink and cyan with offset two-tone neon symbols.",
     cats=["neon"], pro=True, press="glow", opacity=0.95, backdrop=((10, 6, 30), (38, 14, 66)), scene=2,
     mat="neon", face="star", dpad="house", sh="ticket", small="ticket", stick_shape="star", stick_style="ring", base_shape="octagon", base_style="cross",
     well_a=0.3, glyph="duo", sym_thick=0.85, duo=(40, 220, 255), P=mono((255, 70, 190), (255, 70, 190)), mark_col=(255, 120, 210), tags=["neon", "star"])


SCENES = [
    ((22, 18, 60), (255, 120, 90), (255, 220, 150), 1),
    ((14, 30, 40), (60, 150, 130), (230, 240, 190), 2),
    ((10, 12, 44), (70, 90, 200), (200, 220, 255), 3),
    ((40, 22, 18), (220, 130, 60), (255, 230, 170), 4),
    ((16, 30, 62), (60, 160, 210), (250, 245, 220), 5),
]


def game_scene(idx):
    top, mid, sun, seed = SCENES[idx % len(SCENES)]
    W, H = 1280, 720
    rnd = random.Random(seed * 17)
    im = grad(W, H, [(0, top), (0.62, mid), (1, darkc(mid, 0.5))]).convert("RGB")
    d = ImageDraw.Draw(im, "RGBA")
    sx, sy = int(W * (0.3 + 0.4 * rnd.random())), int(H * 0.42)
    for r, a in ((190, 30), (120, 55), (70, 255)):
        d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=sun + (a,))
    for layer in range(3):
        base = H * (0.52 + 0.1 * layer)
        pts = [(0, H)]
        y = base
        for x in range(0, W + 41, 40):
            y += rnd.uniform(-34, 34)
            y = min(base + 40, max(base - 90, y))
            pts.append((x, y))
        pts.append((W, H))
        d.polygon(pts, fill=mixc(top, (0, 0, 0), 0.25 + layer * 0.22) + (255,))
    for _ in range(26):
        bx, bw = rnd.randrange(0, W, 20), rnd.choice([30, 40, 60])
        bh = rnd.randrange(40, 150)
        d.rectangle([bx, H * 0.78 - bh, bx + bw, H], fill=darkc(top, 0.55) + (255,))
    im = im.filter(ImageFilter.GaussianBlur(7))
    return Image.blend(im, Image.new("RGB", im.size, (0, 0, 0)), 0.18)


_scene_cache = {}


def scene_for(idx):
    if idx % len(SCENES) not in _scene_cache:
        _scene_cache[idx % len(SCENES)] = game_scene(idx)
    return _scene_cache[idx % len(SCENES)].copy()


def put(canvas, tile, cx, cy, scale, opacity=1.0, pixel=False):
    if pixel:
        scale = max(0.125, round(scale * 8) / 8.0)
    w, h = max(1, int(round(tile.width * scale))), max(1, int(round(tile.height * scale)))
    t = tile.resize((w, h), Image.NEAREST if pixel else Image.LANCZOS)
    if opacity < 1.0:
        t.putalpha(t.getchannel("A").point(lambda v: int(v * opacity)))
    canvas.alpha_composite(t, (int(round(cx - w / 2.0)), int(round(cy - h / 2.0))))


def dpad_put(canvas, T, cx, cy, k, opacity, pixel, pressed=()):
    off = 132 * k
    for dname, (dx, dy) in {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}.items():
        key = "dpad_" + dname + ("_pressed" if dname in pressed else "")
        put(canvas, T[key], cx + dx * off, cy + dy * off, k, opacity, pixel)


def make_cover(skin, T):
    W, H = 640, 360
    bg = grad(W, H, [(0, skin.backdrop[0]), (1, skin.backdrop[1])]).convert("RGBA")
    glow = rgrad(W, H, W / 2, H * 0.45, W * 0.55, [(0, (255, 255, 255, 34)), (1, (255, 255, 255, 0))])
    bg = over(bg, glow)
    px = skin.pixel
    dpad_put(bg, T, 146, 182, 0.42, 1.0, px)
    fk, fo = 0.42, 64
    for name, dx, dy in (("triangle", 0, -1), ("square", -1, 0), ("circle", 1, 0), ("cross", 0, 1)):
        put(bg, T[name], 496 + dx * fo, 182 + dy * fo, fk, 1.0, px)
    put(bg, T["analog_base"], 320, 236, 0.5, 1.0, px)
    put(bg, T["analog_stick"], 320, 236, 0.5, 1.0, px)
    return bg.convert("RGB")


def make_ingame(skin, T):
    im = scene_for(skin.scene).convert("RGBA")
    op, px = skin.opacity, skin.pixel
    put(im, T["l2"], 150, 82, 0.78, op, px)
    put(im, T["l1"], 150, 190, 0.78, op, px)
    put(im, T["r2"], 1130, 82, 0.78, op, px)
    put(im, T["r1"], 1130, 190, 0.78, op, px)
    dpad_put(im, T, 175, 392, 0.5, op, px)
    fo = 82
    for name, dx, dy in (("triangle", 0, -1), ("square", -1, 0), ("circle", 1, 0), ("cross", 0, 1)):
        put(im, T[name + ("_pressed" if name == "cross" else "")], 1110 + dx * fo, 392 + dy * fo, 0.52, op, px)
    for cx in (330, 950):
        put(im, T["analog_base"], cx, 598, 0.78, op, px)
        put(im, T["analog_stick"], cx, 598, 0.78, op, px)
    put(im, T["select"], 520, 660, 0.62, op, px)
    put(im, T[INPUT_TOGGLE], 640, 660, 0.24, op, px)
    put(im, T["start"], 760, 660, 0.62, op, px)
    return im.convert("RGB")


def png_bytes(im):
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    return b.getvalue()


def webp_bytes(im):
    b = io.BytesIO()
    im.save(b, "WEBP", quality=88, method=6)
    return b.getvalue()


def label_colors(skin):
    """Colour each app should use for its own label on blank_<part> art (the skin's own symbol/label ink)."""
    return {part: hexc(skin.sym_color(Ctx(skin, part, False))) for part in BLANK_PARTS}


def build_zip(skin, images, label_ink=False):
    spec = {"images": {k: k + ".png" for k in sorted(images)}, "tint": None, "opacity": skin.opacity,
            "pressEffect": skin.press}
    if label_ink:
        spec["labelColors"] = label_colors(skin)
    manifest = {"formatVersion": 1, "id": skin.id, "version": PACKAGE_VERSION, "skin": spec}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        def add(name, data):
            zi = zipfile.ZipInfo(name, ZIP_DATE)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, data, compresslevel=9)
        add("controls.json", json.dumps(manifest, indent=2).encode() + b"\n")
        for k in sorted(images):
            add(k + ".png", images[k])
    return buf.getvalue()


CATEGORIES = [("clean", "Clean"), ("retro", "Retro"), ("neon", "Neon"), ("material", "Material"),
              ("playful", "Playful"), ("accessibility", "Accessibility"),
              ("ornate", "Ornate"), ("minimal", "Minimal"), ("sweet", "Sweets")]


def keep_locales(lst, path):
    """Carry non-English name/description translations over from the existing listing.json."""
    try:
        old = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return lst
    for key in ("name", "description"):
        for loc, txt in (old.get(key) or {}).items():
            if loc != "en":
                lst[key].setdefault(loc, txt)
    return lst


def keep_label_locales(cats, path):
    """Carry non-English category labels over from the existing categories.json."""
    try:
        old = {c["id"]: c.get("labels", {}) for c in json.loads(Path(path).read_text(encoding="utf-8"))}
    except (OSError, ValueError, KeyError, TypeError):
        return cats
    for c in cats:
        for loc, txt in old.get(c["id"], {}).items():
            if loc != "en":
                c["labels"].setdefault(loc, txt)
    return cats


def listing_of(skin):
    return {"version": PACKAGE_VERSION, "categories": skin.cats, "tags": skin.tags, "featured": skin.featured,
            "publishedAt": "2026-11-24T12:00:00Z", "requiresPro": skin.pro, "minAppVersionCode": 0,
            "swatch": skin.swatch, "name": {"en": skin.name}, "description": {"en": skin.desc}}


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == text.encode("utf-8"):
        return
    path.write_bytes(text.encode("utf-8"))


def digest_of(listing_text):
    h = hashlib.sha256()
    h.update(b"listing.json\0" + listing_text.replace("\r\n", "\n").encode("utf-8") + b"\0")
    h.update(Path(__file__).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def render_blanks(skin, parts=BLANK_PARTS):
    global BLANK
    T = {}
    BLANK = True
    try:
        for part in parts:
            T["blank_" + part] = skin.render(part, False)
            T["blank_" + part + "_pressed"] = skin.render(part, True)
    finally:
        BLANK = False
    return T


def input_toggle_art(skin, stick, pressed):
    c = Ctx(skin, "analog_stick", pressed)
    mask = c.new()
    draw = ImageDraw.Draw(mask)
    radius = 46 * S
    draw.ellipse((c.cx - radius, c.cy - radius, c.cx + radius, c.cy + radius), outline=255, width=10 * S)
    radius = 14 * S
    draw.ellipse((c.cx - radius, c.cy - radius, c.cx + radius, c.cy + radius), fill=255)
    ink = skin.sym_color(c)
    outline = (255, 255, 255) if sum(ink) < 384 else (0, 0, 0)
    mark = finish(mark_paint(c, mask, ink, outline=(outline, 3)), c)
    tile = over(stick, mark)
    return ImageEnhance.Brightness(tile).enhance(0.84) if pressed else tile


def render_all(skin):
    T = {}
    for part in PARTS:
        T[part] = skin.render(part, False)
        if part != "analog_base":
            T[part + "_pressed"] = skin.render(part, True)
    for suffix in ("", "_pressed"):
        T[INPUT_TOGGLE + suffix] = input_toggle_art(skin, T["analog_stick" + suffix], bool(suffix))
    return T


# The neutral package: the same skin for apps whose buttons are not PlayStation symbols (Nintendo, Sega). It holds
# the label-free art plus the parts that carry no symbol (d-pad, stick). Kept apart from item.ecxcontrols so the
# PlayStation package stays exactly as released apps expect it (they accept at most 48 skin images).
NEUTRAL_SHARED = ["dpad_" + d for d in DIRS] + ["analog_base", "analog_stick", INPUT_TOGGLE]


def neutral_images(skin, shared_png):
    images = {k: v for k, v in shared_png.items() if k.removesuffix("_pressed") in NEUTRAL_SHARED}
    images.update({k: png_bytes(v) for k, v in render_blanks(skin).items()})
    return images


def add_neutral(skin):
    """Writes neutral.ecxcontrols for an already built skin, reusing its d-pad and stick art (no full re-render)."""
    out_dir = DIST / skin.id
    listing_text = json.dumps(keep_locales(listing_of(skin), SRC / skin.id / "listing.json"), indent=2, ensure_ascii=False) + "\n"
    with zipfile.ZipFile(out_dir / "item.ecxcontrols") as z:
        shared = {n[:-4]: z.read(n) for n in z.namelist() if n.endswith(".png")}
    (out_dir / "neutral.ecxcontrols").write_bytes(build_zip(skin, neutral_images(skin, shared), label_ink=True))
    (out_dir / "build.json").write_text(json.dumps({"sourceSha256": digest_of(listing_text)}, indent=2) + "\n",
                                        encoding="utf-8", newline="\n")


def refresh_wide_controls(skin):
    """Rebuilds shoulder, Start and Select tiles without rerendering the other published artwork."""
    out_dir = DIST / skin.id
    package_path = out_dir / "item.ecxcontrols"
    neutral_path = out_dir / "neutral.ecxcontrols"

    def read_images(path):
        with zipfile.ZipFile(path) as archive:
            return {name[:-4]: archive.read(name) for name in archive.namelist() if name.endswith(".png")}

    images = read_images(package_path)
    neutral = read_images(neutral_path)
    for part in WIDE_PARTS:
        images[part] = png_bytes(skin.render(part, False))
        images[part + "_pressed"] = png_bytes(skin.render(part, True))
    neutral.update({key: png_bytes(tile) for key, tile in render_blanks(skin, WIDE_PARTS).items()})
    tiles = {key: Image.open(io.BytesIO(data)).convert("RGBA") for key, data in images.items()}

    listing_path = SRC / skin.id / "listing.json"
    listing_text = json.dumps(keep_locales(listing_of(skin), listing_path), indent=2, ensure_ascii=False) + "\n"
    write_text(listing_path, listing_text)
    package_path.write_bytes(build_zip(skin, images))
    neutral_path.write_bytes(build_zip(skin, neutral, label_ink=True))
    (out_dir / "ingame.webp").write_bytes(webp_bytes(make_ingame(skin, tiles)))
    stamp = json.dumps({"sourceSha256": digest_of(listing_text)}, indent=2) + "\n"
    write_text(out_dir / "build.json", stamp)


def build_skin(skin, force=False, tiles_dir=None):
    out_dir = DIST / skin.id
    listing_text = json.dumps(keep_locales(listing_of(skin), SRC / skin.id / "listing.json"), indent=2, ensure_ascii=False) + "\n"
    write_text(SRC / skin.id / "listing.json", listing_text)
    digest = digest_of(listing_text)
    stamp = out_dir / "build.json"
    need = ["item.ecxcontrols", "neutral.ecxcontrols", "cover.webp", "ingame.webp", "build.json"]
    if not force and all((out_dir / n).is_file() for n in need) and json.loads(stamp.read_text())["sourceSha256"] == digest:
        return False
    T = render_all(skin)
    images = {k: png_bytes(v) for k, v in T.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "item.ecxcontrols").write_bytes(build_zip(skin, images))
    (out_dir / "neutral.ecxcontrols").write_bytes(build_zip(skin, neutral_images(skin, images), label_ink=True))
    (out_dir / "cover.webp").write_bytes(webp_bytes(make_cover(skin, T)))
    (out_dir / "ingame.webp").write_bytes(webp_bytes(make_ingame(skin, T)))
    stamp.write_text(json.dumps({"sourceSha256": digest}, indent=2) + "\n", encoding="utf-8", newline="\n")
    return True


def contact_sheet(skins, path):
    cw, chh, cols = 320, 180, 6
    rows = (len(skins) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw, rows * chh), (0, 0, 0))
    for i, s in enumerate(skins):
        im = Image.open(DIST / s.id / "cover.webp").convert("RGB").resize((cw, chh), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 8 + 6 * len(s.id), 15], fill=(0, 0, 0))
        d.text((4, 2), s.id, fill=(255, 255, 0))
        sheet.paste(im, ((i % cols) * cw, (i // cols) * chh))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


#### ---- KITS BEGIN ----
# ====================================================================================================================
# Game-genre KITS (wave 3): 26 original controller skins (one free + one PRO per genre). They capture the mood of a
# genre only (colour, material, texture); there are no logos, names, characters, costumes or franchise symbols.
# This block is a pure insertion: digest_of() below hashes the file WITHOUT it, so older skins never rebuild when it
# changes; kit skins also hash this block.
# ====================================================================================================================

KIT_CAT_LABELS = {
    "kits": {"en": "Kits", "ar": "حقائب الأنماط", "cs": "Sady", "de": "Kits", "es": "Kits", "fa": "کیت‌ها", "fr": "Kits", "hi": "किट",
             "id": "Kit", "it": "Kit", "ja": "キット", "ko": "키트", "pl": "Zestawy", "pt": "Kits", "ru": "Наборы", "tr": "Setler",
             "uk": "Набори", "zh": "主题套装"},
}
if not any(k == "kits" for k, _ in CATEGORIES):
    CATEGORIES.append(("kits", "Kits"))

_keep_label_locales_v1 = keep_label_locales


def keep_label_locales(cats, path):
    cats = _keep_label_locales_v1(cats, path)
    for cat in cats:
        for loc, txt in KIT_CAT_LABELS.get(cat["id"], {}).items():
            cat["labels"].setdefault(loc, txt)
    return cats


_KB, _KE = "#### ---- KITS " + "BEGIN ----", "#### ---- KITS " + "END ----"


def _kits_split():
    raw = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    a = raw.find(_KB.encode())
    b = raw.find(_KE.encode())
    if a < 0 or b < 0:
        return raw, b""
    b = raw.find(b"\n", b) + 1
    return raw[:a] + raw[b:], raw[a:b]


def digest_of(listing_text):
    base, block = _kits_split()
    h = hashlib.sha256()
    h.update(b"listing.json\0" + listing_text.replace("\r\n", "\n").encode("utf-8") + b"\0")
    h.update(base)
    h.update(GLYPH_FIX)
    if '"kit:' in listing_text:
        h.update(block)
    return h.hexdigest()


def kit(genre, id, name, desc, cats2, pro, **kw):
    tags = ["kit:" + genre] + list(kw.pop("tags", []))
    return form(id=id, name=name, desc=desc, cats=["kits"] + cats2, pro=pro, tags=tags, **kw)


# ---- new shapes -----------------------------------------------------------------------------------------------------
_KIT_FACE = ("slash", "claw", "ragged", "plate", "crest", "dome")
_face_shape_v2 = face_shape


def face_shape(c, name, R):
    if name not in _KIT_FACE:
        return _face_shape_v2(c, name, R)
    m = c.new()
    if R <= 0:
        return m
    d = ImageDraw.Draw(m)
    cx = cy = 128.0
    if name == "slash":
        k, s = R * 0.88, R * 0.34
        return poly_mask(c, [(cx - k + s, cy - k), (cx + k + s, cy - k), (cx + k - s, cy + k), (cx - k - s, cy + k)], 6)
    if name == "claw":
        d.ellipse([(cx - R) * S, (cy - R) * S, (cx + R) * S, (cy + R) * S], fill=255)
        cut = c.new()
        cd = ImageDraw.Draw(cut)
        for k in (-1, 0, 1):
            x0 = cx + k * R * 0.34
            cd.line([((x0 + R * 0.5) * S, (cy - R * 0.62) * S), ((x0 - R * 0.5) * S, (cy + R * 0.62) * S)], fill=255, width=int(R * 0.075 * S))
        return ImageChops.subtract(m, soften(cut, 1.0))
    if name == "ragged":
        rnd = random.Random(41)
        pts = []
        for i in range(44):
            a = math.radians(i * 360.0 / 44)
            rr = R * (1.02 if i % 2 == 0 else 0.84) * (0.94 + 0.08 * rnd.random())
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        return poly_mask(c, pts, 1.6)
    if name == "plate":
        k, b, s = R * 0.92, R * 0.46, R * 0.14
        return poly_mask(c, [(cx - k + b, cy - k), (cx + k - s, cy - k), (cx + k, cy - k + s), (cx + k, cy + k - b),
                             (cx + k - b, cy + k), (cx - k + s, cy + k), (cx - k, cy + k - s), (cx - k, cy - k + b)], 3)
    if name == "crest":
        return poly_mask(c, [(cx, cy - R * 1.12), (cx + R * 0.96, cy - R * 0.4), (cx + R * 0.8, cy + R * 0.9), (cx - R * 0.8, cy + R * 0.9),
                             (cx - R * 0.96, cy - R * 0.4)], 8)
    pts = [(cx + R * 1.04 * math.cos(math.radians(a)), cy + R * 0.2 + R * 1.0 * math.sin(math.radians(a))) for a in range(180, 361, 6)]
    pts += [(cx + R * 1.04, cy + R * 0.9), (cx - R * 1.04, cy + R * 0.9)]
    return poly_mask(c, pts, 5)


_KIT_DPAD = ("blade", "fang", "clip", "spike")
_dpad_shape_v2 = dpad_shape


def dpad_shape(c, form, inset=0):
    if form not in _KIT_DPAD:
        return _dpad_shape_v2(c, form, inset)
    pts = {"blade": [(128, 12), (176, 62), (162, 216), (94, 216), (80, 62)],
           "fang": [(128, 18), (222, 214), (128, 172), (34, 214)],
           "clip": [(88, 40), (168, 40), (192, 66), (192, 214), (64, 214), (64, 66)],
           "spike": [(128, 12), (194, 100), (160, 100), (160, 214), (96, 214), (96, 100), (62, 100)]}[form]
    m = poly_mask(c, pts, 7)
    if inset:
        m = shrink(m, inset)
    return c.rotate(m)


ARROW.update({"blade": (70, 134, 24), "fang": (100, 162, 26), "clip": (84, 146, 26), "spike": (60, 120, 24)})

_rect_shape_v2 = rect_shape


def rect_shape(c, name, inset=0):
    if name not in ("diag", "notched", "ears"):
        return _rect_shape_v2(c, name, inset)
    x0, y0, x1, y1 = c.rect_box()
    h = y1 - y0
    k = 0.36 * h
    if name == "diag":
        m = poly_mask(c, [(x0 + k, y0), (x1, y0), (x1, y1 - k), (x1 - k, y1), (x0, y1), (x0, y0 + k)], 3)
    elif name == "notched":
        m = poly_mask(c, [(x0, y0), (x1, y0), (x1, y1), ((x0 + x1) / 2 + k * 0.9, y1), ((x0 + x1) / 2, y1 - k * 0.75),
                          ((x0 + x1) / 2 - k * 0.9, y1), (x0, y1)], 4)
    else:
        e = k * 0.6
        m = poly_mask(c, [(x0, y0 - e * 0.0), (x0 + k * 1.2, y0), (x0 + k * 1.2 + e, y0 + e), (x1 - k * 1.2 - e, y0 + e), (x1 - k * 1.2, y0),
                          (x1, y0), (x1, y1), (x0, y1)], 4)
    return shrink(m, inset) if inset else m


# ---- glyph centring, tile-fit and edge fade (all skins; GLYPH_FIX salts every digest) ---------------------------------
GLYPH_FIX = b"glyphfix1"
FormSkin.center_glyph = True


def _centroid(m):
    a = np.asarray(m.point(lambda v: 255 if v > 100 else 0), np.bool_)
    ys, xs = np.nonzero(a)
    if xs.size == 0:
        return None
    return xs.mean(), ys.mean()


def _bbox_c(m):
    b = m.point(lambda v: 255 if v > 100 else 0).getbbox()
    return None if not b else ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _recenter(self, c, sk, m, shape):
    gc, sc = _bbox_c(m), _centroid(shape)
    if gc is None or sc is None:
        return m
    if c.kind == "dpad":
        dx, dy = sc[0] - gc[0], sc[1] - gc[1]
    else:  # keep the symbol's own optical offset (triangle); only follow the button shape's centre of mass
        dx, dy = sc[0] - c.cx, sc[1] - c.cy
    if abs(dx) < 1 and abs(dy) < 1:
        return m
    return ImageChops.offset(m, int(round(dx)), int(round(dy)))


_glyph_mask_v1 = FormSkin.glyph_mask


def _glyph_mask_centered(self, c):
    m = _glyph_mask_v1(self, c)
    if m is None or not self.center_glyph or c.kind not in ("dpad", "face", "l3"):
        return m
    return _recenter(self, c, self, m, self.shape(c))


FormSkin.glyph_mask = _glyph_mask_centered
_ctx_mark_v1 = Ctx.mark_mask


def _ctx_mark_centered(self, thick=1.0, scale=1.0, filled=False):
    m = _ctx_mark_v1(self, thick, scale, filled)
    if m is None or isinstance(self.skin, FormSkin) or self.kind not in ("dpad", "face", "l3"):
        return m
    return _recenter(self, self, self.skin, m, self.geom(0))


Ctx.mark_mask = _ctx_mark_centered

_face_fit = {}
_face_shape_v3 = face_shape


def face_shape(c, name, R):
    m = _face_shape_v3(c, name, R)
    if R <= 0:
        return m
    key = (name, R, c.w)
    if key not in _face_fit:
        b = m.point(lambda v: 255 if v > 100 else 0).getbbox()
        k = 1.0
        if b:
            ex, ey = max(c.cx - b[0], b[2] - c.cx), max(c.cy - b[1], b[3] - c.cy)
            k = min(1.0, (c.w / 2.0 - 4 * S) / max(1.0, ex), (c.h / 2.0 - 4 * S) / max(1.0, ey))
        _face_fit[key] = k
    k = _face_fit[key]
    if k >= 0.999:
        return m
    return _face_shape_v3(c, name, R * k)


_finish_v1 = finish


def finish(layer, c):
    im = _finish_v1(layer, c)
    if c.kind == "dpad":
        return im
    a = np.asarray(im.getchannel("A"), np.float32)
    h, w = a.shape
    if max(a[0].max(), a[-1].max(), a[:, 0].max(), a[:, -1].max()) < 4:
        return im
    f = 7
    ramp_ = np.minimum(np.minimum(np.arange(w), np.arange(w)[::-1]), f)[None, :] / float(f)
    ramp_ = np.minimum(ramp_, (np.minimum(np.minimum(np.arange(h), np.arange(h)[::-1]), f)[:, None] / float(f)))
    im.putalpha(Image.fromarray(np.clip(a * ramp_, 0, 255).astype(np.uint8), "L"))
    return im


# ---- new glyph styles ------------------------------------------------------------------------------------------------


def g_glitch(sk, c, m, col):
    out = paint(shift(m, -2.6, 0), (0, 232, 255), 0.85)
    out = over(out, paint(shift(m, 2.6, 0.6), (255, 40, 160), 0.85))
    out = over(out, paint(m, col))
    cut = c.new()
    d = ImageDraw.Draw(cut)
    for fy in (0.42, 0.6):
        d.rectangle([0, fy * c.h - 1.4 * S, c.w, fy * c.h + 1.4 * S], fill=255)
    out.putalpha(ImageChops.subtract(out.getchannel("A"), cut))
    return out


def g_slashed(sk, c, m, col):
    cut = c.new()
    d = ImageDraw.Draw(cut)
    d.line([(c.w * 0.36, c.h * 0.9), (c.w * 0.66, c.h * 0.1)], fill=255, width=int(3.4 * S))
    out = paint(blur(m, 3.0), col, 0.35)
    return over(out, paint(ImageChops.subtract(m, cut), col))


def g_heat(sk, c, m, col):
    fill = grad(c.w, c.h, [(0, (255, 238, 150)), (0.5, col), (1, darkc(col, 0.5))], 90)
    out = paint(blur(m, 5), col, 0.55)
    return over(out, clip(fill, m))


GLYPHS_STYLE.update({"glitch": g_glitch, "slashed": g_slashed, "heat": g_heat})

# ---- helpers ---------------------------------------------------------------------------------------------------------


def _vg(c, M, stops, ang=0.0):
    """Array gradient across the shape's bounding box; ang 0 = top to bottom, 90 = left to right."""
    yy, xx = _grid(c)
    x0, y0, x1, y1, cx, cy = _bb(M)
    a = math.radians(ang)
    ext = abs((x1 - x0) * math.sin(a)) + abs((y1 - y0) * math.cos(a))
    t = ((xx - cx) * math.sin(a) + (yy - cy) * math.cos(a)) / max(1.0, ext) + 0.5
    return _cmap(t, stops)


def _L(mask):
    return A(mask)


def _acc(sk, dflt):
    return sk.acc or dflt


def _edges(idx, k=3):
    return ((idx != np.roll(idx, k, 0)) | (idx != np.roll(idx, k, 1))).astype(np.float32)


def _poly_line(d, p0, p1, wmax, steps=14, pw=0.7):
    pts_l, pts_r = [], []
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    ln = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / ln, dx / ln
    for i in range(steps + 1):
        t = i / steps
        w = wmax * (math.sin(math.pi * t) ** pw) / 2.0
        x, y = p0[0] + dx * t, p0[1] + dy * t
        pts_l.append((x + nx * w, y + ny * w))
        pts_r.append((x - nx * w, y - ny * w))
    d.polygon(pts_l + pts_r[::-1], fill=255)


# ---- new materials ---------------------------------------------------------------------------------------------------


def mat_sunset(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    x0, y0, x1, y1, cx, cy = _bb(M)
    v = np.clip((yy - y0) / max(1.0, y1 - y0), 0, 1)
    arr = _cmap(v, [(0, lighten(hi, 0.4)), (0.4, hi), (0.64, mixc(hi, lo, 0.55)), (1, lo)])
    gap = np.zeros_like(v)
    for k in range(5):
        gap = np.maximum(gap, (np.abs(v - (0.5 + k * 0.095)) < 0.007 + k * 0.01).astype(np.float32))
    gap = A(blur(LI(gap), 0.5))[..., None]
    arr = arr * (1 - gap * 0.72) + np.array(darkc(lo, 0.7), np.float32) * gap * 0.72
    out = fin(sk, c, M, _img(arr), lo, bev=(0.4, 0.5, 3.0), rim=0.8)
    out = over(out, paint(ring_of(shrink(M, 6), 1.8), _acc(sk, (40, 230, 210)), 0.95))
    return over(out, gloss_top(c, M, 0.2))


def mat_palmfrond(sk, c, M, hi, lo):
    x0, y0, x1, y1, cx, cy = _bb(M)
    w, h = x1 - x0, y1 - y0
    arr = _vg(c, M, [(0, lighten(hi, 0.3)), (0.55, hi), (1, lo)], 25)
    out = fin(sk, c, M, _img(arr), lo, bev=(0.4, 0.5, 3.0), rim=0.8)
    sun = c.new()
    r = min(w, h) * 0.17
    ImageDraw.Draw(sun).ellipse([x0 + w * 0.32 - r, y0 + h * 0.36 - r, x0 + w * 0.32 + r, y0 + h * 0.36 + r], fill=255)
    out = over(out, paint(blur(ImageChops.multiply(sun, M), 6), (255, 150, 90), 0.5))
    out = over(out, paint(ImageChops.multiply(blur(sun, 0.8), M), (255, 214, 150), 0.95))
    fr = c.new()
    d = ImageDraw.Draw(fr)
    ox, oy = x1 - w * 0.04, y1 + h * 0.04
    rnd = random.Random(_pseed(c, 31))
    for i in range(4):
        a = math.radians(196 + i * 24)
        L = min(w, h) * (0.86 + 0.14 * rnd.random())
        pts = [(ox + math.cos(a + 0.45 * (t / 12.0) ** 2) * L * t / 12.0, oy + math.sin(a + 0.45 * (t / 12.0) ** 2) * L * t / 12.0) for t in range(13)]
        d.line(pts, fill=255, width=int(3.0 * S))
        for j in range(3, 13):
            ang = a + 0.45 * (j / 12.0) ** 2
            ll = L * 0.3 * math.sin(math.pi * (j - 2) / 12.0) ** 0.7 + L * 0.04
            for sg in (-1, 1):
                la = ang + sg * 1.05
                px, py = pts[j]
                tip = (px + math.cos(la) * ll, py + math.sin(la) * ll + ll * 0.35)
                nx, ny = -math.sin(la), math.cos(la)
                d.polygon([(px + nx * 3.4 * S, py + ny * 3.4 * S), tip, (px - nx * 3.4 * S, py - ny * 3.4 * S)], fill=255)
    out = over(out, paint(ImageChops.multiply(fr, shrink(M, 3)), darkc(lo, 0.6), 0.88))
    out = over(out, paint(ring_of(shrink(M, 6), 1.6), _acc(sk, (255, 120, 190)), 0.9))
    return over(out, gloss_top(c, M, 0.18))


def mat_comicsky(sk, c, M, hi, lo):
    x0, y0, x1, y1, cx, cy = _bb(M)
    w, h = x1 - x0, y1 - y0
    out = paint(shift(M, 0 if c.p else 5, 0 if c.p else 5), (8, 8, 16), 1.0)
    out = over(out, clip(_img(_vg(c, M, [(0, hi), (1, lo)], 0)), M))
    rays = c.new()
    d = ImageDraw.Draw(rays)
    rnd = random.Random(_pseed(c, 51))
    R = max(w, h)
    for i in range(54):
        a = math.radians(i * 360 / 54 + rnd.uniform(-2, 2))
        wd = math.radians(1.0 + 2.6 * rnd.random())
        r0 = R * (0.16 + 0.18 * rnd.random())
        d.polygon([(cx + math.cos(a) * r0, cy + math.sin(a) * r0), (cx + math.cos(a - wd) * R, cy + math.sin(a - wd) * R),
                   (cx + math.cos(a + wd) * R, cy + math.sin(a + wd) * R)], fill=255)
    out = over(out, paint(ImageChops.multiply(rays, M), lighten(hi, 0.5), 0.3))
    dm = ImageChops.multiply(ImageChops.multiply(dots_mask(c, 11, 3.2), M), lin_l(c.w, c.h).point(lambda v: min(255, int(v * 1.5))))
    out = over(out, paint(dm, lighten(hi, 0.6), 0.75))
    out = over(out, paint(ring_of(M, 5.0), (8, 8, 16), 1.0))
    return out


def mat_ripstop(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    x0, y0, x1, y1, cx, cy = _bb(M)
    arr = _vg(c, M, [(0, lighten(hi, 0.12)), (1, lo)], 30)
    th = np.maximum(((xx / S) % 20 < 1.7), ((yy / S) % 20 < 1.7)).astype(np.float32)
    weave = ((np.floor(xx / S / 2.2) + np.floor(yy / S / 2.2)) % 2).astype(np.float32)
    arr = arr * (1 + 0.2 * th[..., None] - 0.05 * weave[..., None]) * (0.94 + 0.1 * _n(sk, c, 2, _pseed(c, 61), 1.5))[..., None]
    sheen = np.exp(-((((xx - x0) + (yy - y0)) / S - 0.7 * ((x1 - x0) + (y1 - y0)) / S * 0.5) / 22.0) ** 2)
    arr = arr + sheen[..., None] * 34
    out = fin(sk, c, M, _img(arr), lo, bev=(0.35, 0.5, 3.0), rim=0.8)
    pipe = ring_of(shrink(M, 9), 2.3)
    out = over(out, paint(shift(pipe, 0, 1.5), (0, 0, 0), 0.4))
    return over(out, paint(pipe, _acc(sk, (206, 216, 232)), 0.95))


def mat_gouged(sk, c, M, hi, lo):
    x0, y0, x1, y1, cx, cy = _bb(M)
    w, h = x1 - x0, y1 - y0
    st = A(streak_noise(c.w, c.h, 2, _pseed(c, 71)))
    arr = _cmap(0.25 + st * 0.55, [(0, lo), (1, hi)]) * (0.9 + 0.2 * _n(sk, c, 18, _pseed(c, 72), 1.4))[..., None]
    out = fin(sk, c, M, _img(arr), lo, bev=(0.45, 0.55, 3.2), rim=0.85)
    gm = c.new()
    d = ImageDraw.Draw(gm)
    for k in range(3):
        o = (k - 1) * 0.24
        _poly_line(d, (cx + w * (0.3 + o), y0 + h * 0.06), (cx + w * (-0.28 + o), y1 - h * 0.06), 9 * S, 16, 0.6)
    gm = ImageChops.multiply(gm, shrink(M, 3))
    rage = _acc(sk, (236, 40, 36))
    if c.p:
        rage = lighten(rage, 0.3)
    out = over(out, paint(blur(grow(gm, 3), 4), rage, 0.55))
    out = over(out, paint(gm, darkc(rage, 0.8), 1.0))
    out = over(out, paint(shrink(gm, 2.0), rage, 0.9))
    out = over(out, paint(ImageChops.subtract(shift(gm, -1.4, -1.4), gm), (255, 255, 255), 0.55))
    return over(out, paint(ImageChops.subtract(shift(gm, 1.4, 1.4), gm), (0, 0, 0), 0.5))


def mat_ashen(sk, c, M, hi, lo):
    pts = _pts_in(M, 15, _pseed(c, 81))
    idx, dist = _vor(c, pts)
    crack = A(blur(LI(_edges(idx, 4)), 1.1)).clip(0, 1) ** 0.8
    n = _n(sk, c, 10, _pseed(c, 82), 1.8)
    arr = _cmap(n, [(0, lo), (1, hi)]) * (0.8 + 0.3 * np.clip(dist / 150.0, 0, 1))[..., None]
    glowa = A(blur(LI(crack), 4))
    ember = np.array(_acc(sk, (255, 120, 36)), np.float32)
    arr = arr + glowa[..., None] * ember * 0.55 + crack[..., None] * np.array(lighten(ember, 0.45), np.float32) * 0.9
    sp = (rng_noise(c.w, c.h, _pseed(c, 83)).point(lambda v: 255 if v > 252 else 0))
    out = fin(sk, c, M, _img(arr), lo, bev=(0.3, 0.6, 3.2), rim=0.85)
    out = over(out, paint(ImageChops.multiply(blur(sp, 0.8), M), lighten(ember, 0.5), 0.9))
    return over(out, inner_shadow(M, 0, 5, 8, (0, 0, 0), 0.4))


def mat_runic(sk, c, M, hi, lo):
    x0, y0, x1, y1, cx, cy = _bb(M)
    w, h = x1 - x0, y1 - y0
    n = _n(sk, c, 8, _pseed(c, 91), 1.7)
    arr = _vg(c, M, [(0, lighten(hi, 0.18)), (1, lo)], 40) * (0.82 + 0.3 * n)[..., None]
    arr[..., 1:] *= (0.93 + 0.1 * _n(sk, c, 30, _pseed(c, 92), 1.3))[..., None]
    out = fin(sk, c, M, _img(arr), lo, bev=(0.5, 0.55, 3.2), rim=0.85)
    rm = c.new()
    d = ImageDraw.Draw(rm)
    rnd = random.Random(_pseed(c, 93))
    wd = int(2.8 * S)
    if c.round:
        r = min(w, h) / 2.0 * 0.7
        for i in range(14):
            a = math.radians(i * 360 / 14)
            ca, sa = math.cos(a), math.sin(a)
            p0, p1 = (cx + ca * r * 0.88, cy + sa * r * 0.88), (cx + ca * r * 1.12, cy + sa * r * 1.12)
            d.line([p0, p1], fill=255, width=wd)
            tx, ty = -sa, ca
            for f in (0.35, 0.8):
                if rnd.random() < 0.7:
                    q = (p0[0] + (p1[0] - p0[0]) * f, p0[1] + (p1[1] - p0[1]) * f)
                    sg = rnd.choice((-1, 1))
                    d.line([q, (q[0] + (tx * sg + ca * 0.7) * r * 0.12, q[1] + (ty * sg + sa * 0.7) * r * 0.12)], fill=255, width=wd)
    else:
        for i in range(6):
            x = x0 + w * (0.17 + i * 0.13)
            d.line([(x, cy - h * 0.2), (x, cy + h * 0.2)], fill=255, width=wd)
            if rnd.random() < 0.8:
                d.line([(x, cy - h * 0.05 * rnd.choice((-1, 1))), (x + w * 0.05, cy - h * 0.2 * rnd.choice((-1, 1)))], fill=255, width=wd)
    rm = ImageChops.multiply(rm, shrink(M, 4))
    amber = _acc(sk, (255, 150, 48))
    if c.p:
        amber = lighten(amber, 0.25)
    out = over(out, paint(blur(rm, 5), amber, 0.7))
    out = over(out, paint(rm, lighten(amber, 0.35), 1.0))
    return over(out, paint(ring_of(shrink(M, 12), 1.4), darkc(lo, 0.5), 0.8))


def mat_hazardtape(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    x0, y0, x1, y1, cx, cy = _bb(M)
    w, h = x1 - x0, y1 - y0
    n = _n(sk, c, 9, _pseed(c, 101), 1.2)
    arr = _cmap(0.35 + n * 0.5, [(0, lo), (1, hi)]) * (0.93 + 0.1 * _n(sk, c, 2, _pseed(c, 105), 1.5))[..., None]
    out = fin(sk, c, M, _img(arr), lo, bev=(0.3, 0.55, 3.2), rim=0.8)
    a = math.radians(-24)
    dist = (xx - cx) * math.sin(a) + (yy - cy) * math.cos(a)
    band = (np.abs(dist - h * 0.12) < max(26 * S, h * 0.17)).astype(np.float32)
    bm = ImageChops.multiply(blur(LI(band), 0.6), M)
    tape = sk.acc or (240, 200, 22)
    stripes = A(stripe_mask(c, 20, 0.5, 45))
    tarr = np.where(stripes[..., None] > 0.5, np.array(tape, np.float32), np.array((28, 26, 22), np.float32))
    tarr = tarr * (0.86 + 0.2 * _n(sk, c, 4, _pseed(c, 102), 1.6))[..., None]
    out = over(out, paint(shift(bm, 0, 2.2), (0, 0, 0), 0.4))
    out = over(out, clip(_img(tarr), bm))
    rnd = random.Random(_pseed(c, 103))
    sp = c.new()
    d = ImageDraw.Draw(sp)
    for _ in range(26):
        px, py, r = rnd.uniform(x0, x1), rnd.uniform(y0, y1), rnd.uniform(1.2, 5.2) * S
        d.ellipse([px - r, py - r, px + r, py + r], fill=255)
    out = over(out, paint(ImageChops.multiply(blur(sp, 0.5), M), (110, 14, 18), 0.8))
    gr = A_to_L(_n(sk, c, 5, _pseed(c, 104), 2.4), 0.55, 1.0)
    return over(out, paint(ImageChops.multiply(gr, M), (14, 12, 8), 0.3))


def mat_chainlink(sk, c, M, hi, lo):
    x0, y0, x1, y1, cx, cy = _bb(M)
    back = _vg(c, M, [(0, mixc(sk.acc or (150, 40, 30), (255, 150, 60), 0.35)), (1, darkc(sk.acc or (150, 40, 30), 0.55))], 20)
    back = back * (0.8 + 0.3 * _n(sk, c, 12, _pseed(c, 111), 1.6))[..., None]
    out = fin(sk, c, M, _img(back), lo, bev=(0.3, 0.55, 3.2), rim=0.85)
    wire = ImageChops.lighter(stripe_mask(c, 17, 0.17, 45), stripe_mask(c, 17, 0.17, -45))
    wire = ImageChops.multiply(wire, shrink(M, 2))
    rustn = A_to_L(_n(sk, c, 6, _pseed(c, 112), 2.0), 0.0, 1.0)
    out = over(out, paint(shift(wire, 1.6, 2.4), (0, 0, 0), 0.55))
    out = over(out, paint(wire, D(c, hi)))
    out = over(out, paint(ImageChops.multiply(wire, rustn), (150, 78, 38), 0.55))
    out = over(out, paint(ImageChops.multiply(wire, ImageChops.invert(top_fade(c, 1.2))), (255, 255, 255), 0.16))
    return over(out, paint(ring_of(M, 6.0), D(c, lo), 0.95))


def mat_rarity(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    x0, y0, x1, y1, cx, cy = _bb(M)
    arr = _vg(c, M, [(0, darkc(lo, 0.55)), (0.45, lo), (1, hi)], 0)
    sweep = np.exp(-((((xx - x0) * 0.8 - (yy - y0)) / S - 0.15 * (x1 - x0) / S) / 20.0) ** 2) * 0.18
    arr = arr + sweep[..., None] * 255
    out = fin(sk, c, M, _img(arr), lo, bev=(0.5, 0.5, 3.0), rim=0.85)
    out = over(out, paint(ImageChops.multiply(blur(ring_of(shrink(M, 3), 6), 3), M), lighten(hi, 0.25), 0.55))
    L = min(x1 - x0, y1 - y0) * 0.28
    mk = c.new()
    d = ImageDraw.Draw(mk)
    ix = 12 * S
    for (px, py, sx, sy) in ((x0 + ix, y0 + ix, 1, 1), (x1 - ix, y0 + ix, -1, 1), (x0 + ix, y1 - ix, 1, -1), (x1 - ix, y1 - ix, -1, -1)):
        d.line([(px, py + sy * L), (px, py), (px + sx * L, py)], fill=255, width=int(2.4 * S), joint="curve")
    out = over(out, paint(ImageChops.multiply(mk, M), lighten(hi, 0.6), 0.9))
    return over(out, gloss_top(c, M, 0.22, 2.2))


def mat_stormring(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    x0, y0, x1, y1, cx, cy = _bb(M)
    w = x1 - x0
    t = _n(sk, c, 34, _pseed(c, 121), 1.0)
    arr = _cmap(t, [(0, lo), (1, hi)])
    cl = (np.abs(np.sin(t * 34.0)) < 0.13).astype(np.float32)
    arr = arr * (1 - 0.35 * cl[..., None]) + cl[..., None] * np.array(lighten(hi, 0.4), np.float32) * 0.35
    zc = (cx, cy)
    rr = np.hypot(xx - zc[0], yy - zc[1])
    zr = min(w, y1 - y0) * 0.42 if c.round else 1e5
    outside = (rr > zr).astype(np.float32)
    storm = np.array(_acc(sk, (150, 70, 255)), np.float32)
    arr = arr * (1 - 0.55 * outside[..., None]) + storm * 0.55 * outside[..., None]
    out = fin(sk, c, M, _img(arr), lo, bev=(0.4, 0.55, 3.0), rim=0.85)
    zring = c.new()
    if c.round:
        ImageDraw.Draw(zring).ellipse([zc[0] - zr, zc[1] - zr, zc[0] + zr, zc[1] + zr], outline=255, width=int(3.0 * S))
    zring = ImageChops.multiply(zring, shrink(M, 3))
    out = over(out, paint(blur(zring, 4), tuple(int(v) for v in storm), 0.8))
    return over(out, paint(zring, (240, 232, 255), 0.95))


def mat_sumi(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    x0, y0, x1, y1, cx, cy = _bb(M)
    w, h = x1 - x0, y1 - y0
    paper = np.array(hi, np.float32) * (0.95 + 0.07 * _n(sk, c, 3, _pseed(c, 131), 1.3))[..., None]
    st = A(streak_noise(c.w, c.h, 12, _pseed(c, 132)))
    yc = cy + (xx - cx) * 0.2
    dy = np.abs(yy - yc) / max(1.0, h * 0.5)
    taper = 1.05 - np.clip((xx - x0) / max(1.0, w), 0, 1) * 0.75
    ink = np.clip((0.46 * taper - dy + (st - 0.5) * 0.55) * 9.0, 0, 1) * (0.78 + 0.3 * st)
    ink = np.clip(ink, 0, 1)[..., None]
    inkc = np.array((22, 22, 28), np.float32)
    arr = paper * (1 - ink * 0.96) + inkc * ink * 0.96
    out = fin(sk, c, M, _img(arr), lo, bev=(0.25, 0.4, 2.6), rim=0.8)
    hk = c.new()
    ImageDraw.Draw(hk).rounded_rectangle([x1 - w * 0.3, y1 - h * 0.34, x1 - w * 0.14, y1 - h * 0.14], radius=int(2.5 * S), fill=255)
    out = over(out, paint(ImageChops.multiply(hk, shrink(M, 4)), _acc(sk, (190, 34, 40)), 0.92))
    return over(out, paint(ring_of(shrink(M, 4), 1.4), darkc(lo, 0.5), 0.8))


def mat_damascus(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    nw = _n(sk, c, 46, _pseed(c, 141), 1.0)
    band = 0.5 + 0.5 * np.sin(((xx * 0.35 + yy) / S) * 0.2 + nw * 12.0)
    band = np.clip(band + (0.5 + 0.5 * np.sin(((yy - xx * 0.2) / S) * 0.42 + nw * 5.0)) * 0.25, 0, 1)
    arr = _cmap(0.25 + band * 0.55, [(0, darkc(lo, 0.1)), (0.5, mixc(lo, hi, 0.6)), (1, lighten(hi, 0.2))])
    arr = arr * (0.94 + 0.1 * _n(sk, c, 2, _pseed(c, 142), 1.5))[..., None]
    out = fin(sk, c, M, _img(arr), lo, bev=(0.7, 0.6, 2.8), rim=0.9, sh=(6, 8, 0.5))
    out = over(out, paint(ring_of(shrink(M, 8), 1.7), _acc(sk, (214, 168, 70)), 0.95))
    return over(out, gloss_top(c, M, 0.2, 2.0))


def mat_scrap(sk, c, M, hi, lo):
    pts = _pts_in(M, 7, _pseed(c, 151))
    idx, dist = _vor(c, pts)
    pal = np.array([hi, lo, (158, 98, 60), (176, 156, 112), (96, 104, 88), (134, 76, 48), mixc(hi, lo, 0.5)], np.float32)
    arr = pal[idx % len(pal)] * (0.82 + 0.3 * _n(sk, c, 5, _pseed(c, 152), 1.8))[..., None]
    edge = A(blur(LI(_edges(idx, 3)), 0.7))
    arr = arr * (1 - 0.7 * edge[..., None])
    out = fin(sk, c, M, _img(arr), lo, bev=(0.4, 0.55, 3.2), rim=0.85)
    ld = c.new()
    d = ImageDraw.Draw(ld)
    for (px, py) in pts:
        d.ellipse([px - 4.4 * S, py - 4.4 * S, px + 4.4 * S, py + 4.4 * S], fill=255)
    ld = ImageChops.multiply(ld, shrink(M, 3))
    out = over(out, paint(ld, (40, 34, 30), 0.9))
    out = over(out, paint(ImageChops.subtract(shift(ld, -0.9, -1.2), ld), (255, 255, 255), 0.4))
    weld = ImageChops.multiply(ImageChops.multiply(blur(LI(_edges(idx, 6)), 0.6), stripe_mask(c, 6, 0.5, 45)), shrink(M, 3))
    out = over(out, paint(weld, (210, 200, 180), 0.5))
    return over(out, paint(ImageChops.multiply(M, A_to_L(_n(sk, c, 14, _pseed(c, 153), 2.0), 0.0, 0.7)), (214, 190, 140), 0.18))


def mat_sandripple(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    nw = _n(sk, c, 28, _pseed(c, 161), 1.0)
    rp = 0.5 + 0.5 * np.sin(((yy + xx * 0.25) / S) * 0.32 + nw * 9.0)
    arr = _cmap(rp, [(0, lo), (0.5, mixc(lo, hi, 0.6)), (1, lighten(hi, 0.22))])
    arr = arr * (0.93 + 0.12 * _n(sk, c, 2, _pseed(c, 162), 1.6))[..., None]
    out = fin(sk, c, M, _img(arr), lo, bev=(0.35, 0.5, 3.0), rim=0.8)
    st = ImageChops.multiply(A_to_L(A(streak_noise(c.w, c.h, 30, _pseed(c, 163))), 0.6, 1.0).point(lambda v: 255 if v > 205 else 0), blur(M, 1.0))
    out = over(out, paint(blur(st, 1.6), (255, 236, 196), 0.28))
    x0, y0, x1, y1, cx, cy = _bb(M)
    rv = c.new()
    d = ImageDraw.Draw(rv)
    for px in (x0 + (x1 - x0) * 0.14, x1 - (x1 - x0) * 0.14):
        d.ellipse([px - 5 * S, cy - 5 * S, px + 5 * S, cy + 5 * S], fill=255)
    rv = ImageChops.multiply(rv, M)
    out = over(out, paint(rv, _acc(sk, (120, 66, 38)), 0.9))
    return over(out, paint(ImageChops.subtract(shift(rv, -0.8, -1.1), rv), (255, 255, 255), 0.35))


def mat_armorplate(sk, c, M, hi, lo):
    x0, y0, x1, y1, cx, cy = _bb(M)
    w, h = x1 - x0, y1 - y0
    arr = _vg(c, M, [(0, lighten(hi, 0.22)), (0.5, hi), (1, lo)], 35)
    out = fin(sk, c, M, _img(arr), lo, bev=(0.6, 0.6, 3.6), rim=0.9)
    chip = ImageChops.multiply(ring_of(M, 12), A_to_L(_n(sk, c, 4, _pseed(c, 171), 3.0), 0.0, 1.0).point(lambda v: 255 if v > 170 else 0))
    out = over(out, paint(chip, (62, 66, 74), 0.95))
    pan = ring_of(shrink(M, 17), 2.0)
    out = over(out, paint(pan, darkc(lo, 0.65), 0.9))
    out = over(out, paint(ImageChops.subtract(shift(pan, 0.8, 1.6), pan), (255, 255, 255), 0.35))
    bars = c.new()
    d = ImageDraw.Draw(bars)
    by = y1 - h * 0.24
    x = x0 + w * 0.3
    for i, bw in enumerate((12, 5, 9, 5, 14, 5, 8)):
        d.rectangle([x, by, x + bw * S * 0.7, by + h * 0.07], fill=255)
        x += (bw + 5) * S * 0.7
    out = over(out, paint(ImageChops.multiply(bars, shrink(M, 6)), _acc(sk, (255, 122, 24)), 0.92))
    return out


def mat_reentry(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    x0, y0, x1, y1, cx, cy = _bb(M)
    v = np.clip((yy - y0) / max(1.0, y1 - y0), 0, 1)
    tile = np.floor(xx / S / 22.0) * 7 + np.floor(yy / S / 22.0) * 13
    tv = (np.sin(tile * 12.9898) * 43758.5453) % 1.0
    base = _cmap(tv, [(0, (24, 22, 24)), (1, (62, 54, 52))])
    heat = np.clip((v - 0.35) / 0.65, 0, 1) ** 1.4
    hot = np.array(hi, np.float32)
    arr = base * (1 - heat[..., None] * 0.8) + (hot * (0.5 + 0.6 * tv[..., None])) * heat[..., None]
    gm = grid_mask(c, 22, 2.6)
    gridl = A(gm)[..., None]
    arr = arr * (1 - gridl * 0.78) + np.array(lo, np.float32) * gridl * 0.55 * heat[..., None]
    out = fin(sk, c, M, _img(arr), lo, bev=(0.35, 0.55, 3.0), rim=0.85)
    sk_ = c.new()
    ImageDraw.Draw(sk_).polygon([(x0 + (x1 - x0) * 0.1, y1), (x0 + (x1 - x0) * 0.3, y1), (x0 + (x1 - x0) * 0.62, y0), (x0 + (x1 - x0) * 0.44, y0)], fill=255)
    out = over(out, paint(ImageChops.multiply(blur(sk_, 7), M), lighten(hi, 0.5), 0.34))
    return over(out, inner_shadow(M, 0, 5, 8, (0, 0, 0), 0.35))


def mat_tread(sk, c, M, hi, lo):
    x0, y0, x1, y1, cx, cy = _bb(M)
    base = _vg(c, M, [(0, hi), (1, lo)], 0) * (0.85 + 0.25 * _n(sk, c, 2, _pseed(c, 181), 1.8))[..., None]
    out = fin(sk, c, M, _img(base), lo, bev=(0.3, 0.5, 3.0), rim=0.85)
    blk = ImageChops.multiply(stripe_mask(c, 24, 0.62, 58), stripe_mask(c, 30, 0.8, 0))
    inner = shrink(M, 9)
    blk = ImageChops.multiply(blk, inner)
    out = over(out, paint(shift(blk, 0, 2.0), (0, 0, 0), 0.5))
    out = over(out, paint(blk, lighten(hi, 0.16), 1.0))
    out = over(out, paint(ImageChops.subtract(blk, shift(blk, 1.2, 1.8)), (255, 255, 255), 0.22))
    out = over(out, paint(blur(ring_of(shrink(M, 5), 2.8), 3), _acc(sk, (40, 150, 255)), 0.55))
    return over(out, paint(ring_of(shrink(M, 5), 2.0), lighten(_acc(sk, (40, 150, 255)), 0.45), 0.95))


def mat_carbonblue(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    tex = carbon_tex(c.tw, c.th, 6).resize((c.w, c.h), Image.BILINEAR)
    arr = _cmap(A(tex), [(0, (6, 6, 10)), (1, (78, 82, 96))])
    x0, y0, x1, y1, cx, cy = _bb(M)
    sw = np.exp(-((((xx - x0) - (yy - y0) * 0.9) / S - 0.35 * (x1 - x0) / S) / 26.0) ** 2) * 0.3
    arr = arr + sw[..., None] * 255
    out = fin(sk, c, M, _img(arr), lo, bev=(0.4, 0.55, 3.0), rim=0.85, sh=(6, 8, 0.5))
    blue = _acc(sk, (40, 160, 255))
    if c.p:
        blue = lighten(blue, 0.3)
    r = ring_of(shrink(M, 8), 2.2)
    out = over(out, paint(blur(r, 4), blue, 0.7))
    out = over(out, paint(r, lighten(blue, 0.5)))
    return over(out, gloss_top(c, M, 0.32, 1.8))


def mat_holoplate(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    arr = _vg(c, M, [(0, hi), (1, lo)], 35)
    arr = arr * (1 - 0.18 * (((yy / S) % 6) < 1.6)[..., None])
    rnd = random.Random(_pseed(c, 191))
    x0, y0, x1, y1, cx, cy = _bb(M)
    for _ in range(3):
        sy = rnd.uniform(y0, y1 - 20 * S)
        sh_ = rnd.uniform(5, 13) * S
        sel = (yy > sy) & (yy < sy + sh_)
        arr[sel] = arr[sel] * 1.0 + 38
    out = fin(sk, c, M, _img(arr), lo, bev=(0.3, 0.5, 3.0), rim=0.0, sh=(5, 7, 0.45))
    rg = ring_of(M, 2.4)
    out = over(out, paint(shift(rg, -1.8, 0), (0, 230, 255), 0.9))
    out = over(out, paint(shift(rg, 1.8, 0), (255, 40, 170), 0.9))
    L = min(x1 - x0, y1 - y0) * 0.22
    mk = c.new()
    d = ImageDraw.Draw(mk)
    ix = 12 * S
    for (px, py, sx, sy) in ((x0 + ix, y0 + ix, 1, 1), (x1 - ix, y1 - ix, -1, -1)):
        d.line([(px, py + sy * L), (px, py), (px + sx * L, py)], fill=255, width=int(2.2 * S), joint="curve")
    return over(out, paint(ImageChops.multiply(mk, M), (200, 250, 255), 0.85))


def mat_mercchrome(sk, c, M, hi, lo):
    x0, y0, x1, y1, cx, cy = _bb(M)
    yy, xx = _grid(c)
    v = (yy - y0) / max(1.0, y1 - y0)
    v = np.clip(v + 0.1 * np.sin((xx - cx) / max(1.0, x1 - x0) * 3.4) * (v - 0.5), 0, 1)
    arr = _cmap(v, [(0, (236, 244, 255)), (0.36, hi), (0.49, (14, 10, 30)), (0.52, (255, 112, 212)), (0.8, (60, 222, 255)), (1, lo)])
    out = fin(sk, c, M, _img(arr), lo, bev=(0.6, 0.5, 2.6), rim=0.85, sh=(6, 8, 0.5))
    ck = c.new()
    d = ImageDraw.Draw(ck)
    rnd = random.Random(_pseed(c, 201))
    for _ in range(5):
        px, py = rnd.uniform(x0 + 20, x1 - 20), rnd.uniform(cy, y1 - 20)
        d.line([(px, py), (px + rnd.uniform(14, 40) * S, py), (px + rnd.uniform(40, 60) * S, py + rnd.uniform(-14, 14) * S)], fill=255, width=int(1.8 * S))
    out = over(out, paint(ImageChops.multiply(ck, shrink(M, 5)), (20, 14, 40), 0.7))
    sp = c.new()
    ImageDraw.Draw(sp).ellipse([x0 + (x1 - x0) * 0.16, y0 + (y1 - y0) * 0.08, x0 + (x1 - x0) * 0.48, y0 + (y1 - y0) * 0.24], fill=255)
    out = over(out, paint(blur(ImageChops.multiply(sp, M), 2.2), (255, 255, 255), 0.6 if not c.p else 0.35))
    return over(out, paint(ring_of(M, 1.4), (255, 255, 255), 0.65))


def mat_ropewood(sk, c, M, hi, lo):
    seed = 5 + _pseed(c) % 9
    dark, light = sk.wood
    t = sk.cache(("kwood", c.tw, c.th, seed), lambda: wood_tex(c.tw, c.th, seed, dark, light)).resize((c.w, c.h), Image.BICUBIC)
    shade = lit(c, (255, 250, 245), (150, 128, 112))
    fill = ImageChops.multiply(t.convert("RGB"), shade.convert("RGB")).convert("RGBA")
    out = shadow_of(c, M, 6, 7, 0.5, 0, (20, 8, 0))
    out = over(out, clip(tone(fill, 0.8 if c.p else 1.0), M))
    out = over(out, bevel(M, 3.2, 0.4, 0.5))
    rope = ring_of(M, 15)
    tw = ImageChops.multiply(rope, stripe_mask(c, 9, 0.56, 62))
    out = over(out, paint(rope, (96, 74, 44), 1.0))
    out = over(out, paint(tw, (214, 186, 130), 1.0))
    out = over(out, paint(ImageChops.subtract(shift(tw, -0.8, -1.2), tw), (255, 244, 214), 0.45))
    out = over(out, paint(ImageChops.multiply(blur(rope, 3), ImageChops.invert(blur(shrink(rope, 3), 2))), (0, 0, 0), 0.25))
    out = over(out, paint(ring_of(shrink(M, 15), 2.4), (0, 0, 0), 0.5))
    return over(out, paint(ring_of(M, 1.6), (40, 26, 12), 0.8))


def mat_doubloon(sk, c, M, hi, lo):
    yy, xx = _grid(c)
    arr = _vg(c, M, [(0, lighten(hi, 0.5)), (0.45, hi), (1, lo)], 45)
    arr = arr * (0.93 + 0.12 * _n(sk, c, 7, _pseed(c, 211), 1.4))[..., None]
    out = fin(sk, c, M, _img(arr), lo, bev=(0.7, 0.55, 2.8), rim=0.9, sh=(6, 8, 0.5))
    rad, ang, bb = _polar(c, M)
    tick = (np.sin(ang * 46.0) > 0.0).astype(np.float32)
    zone = ImageChops.multiply(ring_of(M, 11), M)
    tk = ImageChops.multiply(zone, LI(tick))
    out = over(out, paint(tk, darkc(lo, 0.5), 0.8))
    inner = ring_of(shrink(M, 15), 1.8)
    out = over(out, paint(inner, darkc(lo, 0.55), 0.9))
    out = over(out, paint(ImageChops.subtract(shift(inner, 0.8, 1.4), inner), (255, 246, 190), 0.55))
    x0, y0, x1, y1, cx, cy = bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]
    sp = c.new()
    ImageDraw.Draw(sp).ellipse([x0 + (x1 - x0) * 0.15, y0 + (y1 - y0) * 0.1, x0 + (x1 - x0) * 0.46, y0 + (y1 - y0) * 0.3], fill=255)
    return over(out, paint(blur(ImageChops.multiply(sp, M), 3), (255, 255, 235), 0.5 if not c.p else 0.3))


def mat_tooled(sk, c, M, hi, lo):
    x0, y0, x1, y1, cx, cy = _bb(M)
    w, h = x1 - x0, y1 - y0
    n = _n(sk, c, 6, _pseed(c, 221), 1.1)
    arr = _cmap(0.3 + n * 0.6, [(0, lo), (1, hi)]) * (0.95 + 0.07 * _n(sk, c, 1.6, _pseed(c, 222), 1.4))[..., None]
    out = fin(sk, c, M, _img(arr), lo, bev=(0.35, 0.55, 3.0), rim=0.85)
    sp = c.new()
    d = ImageDraw.Draw(sp)
    for sg in (-1, 1):
        sx = cx + sg * w * 0.2
        pts = []
        for i in range(40):
            th = i * 0.17
            rr = (1.5 + th * 3.4) * S * min(w, h) / 150.0
            pts.append((sx + sg * math.cos(th * sg * 1.0 + (0 if sg < 0 else math.pi)) * rr * 1.0, cy + math.sin(th + (0 if sg < 0 else math.pi)) * rr * 1.0))
        d.line(pts, fill=255, width=int(3.6 * S), joint="curve")
    d.line([(x0 + w * 0.2, cy + h * 0.2), (cx, cy + h * 0.26), (x1 - w * 0.2, cy + h * 0.2)], fill=255, width=int(3.4 * S), joint="curve")
    sp = ImageChops.multiply(sp, shrink(M, 15))
    out = over(out, paint(ImageChops.subtract(shift(sp, 0.9, 1.4), sp), (255, 230, 190), 0.45))
    out = over(out, paint(sp, darkc(lo, 0.5), 0.85))
    out = over(out, paint(ring_of(M, 5), darkc(lo, 0.45), 0.55))
    stc = ImageChops.multiply(ring_of(shrink(M, 7), 1.5), stripe_mask(c, 7, 0.55, 45))
    out = over(out, paint(shift(stc, 0, 1.2), (0, 0, 0), 0.35))
    return over(out, paint(stc, _acc(sk, (236, 216, 172)), 0.95))


def mat_claycrack(sk, c, M, hi, lo):
    pts = _pts_in(M, 20, _pseed(c, 231))
    idx, dist = _vor(c, pts)
    base = _vg(c, M, [(0, lighten(hi, 0.18)), (0.55, hi), (1, lo)], 0)
    n = _n(sk, c, 5, _pseed(c, 232), 1.8)
    arr = base * (0.86 + 0.26 * n)[..., None] * (0.88 + 0.16 * np.clip(dist / 70.0, 0, 1))[..., None]
    crack = A(blur(LI(_edges(idx, 3)), 0.6))
    arr = arr * (1 - 0.78 * crack[..., None])
    out = fin(sk, c, M, _img(arr), lo, bev=(0.4, 0.55, 3.0), rim=0.85)
    cr = ImageChops.multiply(LI(crack), shrink(M, 2))
    out = over(out, paint(ImageChops.subtract(shift(cr, 0.8, 1.2), cr), lighten(hi, 0.45), 0.4))
    return over(out, paint(ImageChops.multiply(M, A_to_L(_n(sk, c, 10, _pseed(c, 233), 2.0), 0.0, 0.7)), (255, 226, 176), 0.14))


MATS.update({k[4:]: v for k, v in list(globals().items()) if k.startswith("mat_") and callable(v)})

# ---- the 26 skins ----------------------------------------------------------------------------------------------------
# `P` holds (highlight, shadow) colour pairs per role (face buttons by name, everything else under "main").


def _all(hi, lo, **extra):
    d = {"main": (hi, lo)}
    d.update(extra)
    return d


def _fc(cols, **kw):
    return faces(cols, 0.0, 0.0, **kw)


# 1 neon-vice ------------------------------------------------------------------------------------------------------
kit("neon-vice", "pad-neon-vice", "Vice Sunset", "Striped retro sunsets in pink and orange with teal neon rims: an 80s coast-city mood.", ["neon", "retro"], False,
    featured=True, press="glow", opacity=0.94, backdrop=((46, 12, 78), (255, 96, 120)), scene=0,
    mat="sunset", face="dome", dpad="clip", sh="tab", small="tab", stick_shape="dome", stick_style="ring", base_shape="circle", base_style="ticks",
    well_a=0.35, glyph="glow", sym_thick=0.9, acc=(40, 232, 214), P=_all((255, 120, 168), (96, 28, 140)),
    sym_colors=scol(((60, 240, 226), (255, 236, 200), (255, 190, 90), (255, 255, 255))), mark_col=(80, 244, 230))

kit("neon-vice", "pad-neon-vice-pro", "Palm Boulevard", "Teal-to-indigo glass with sun-lit palm fronds and two-tone neon symbols, like an ocean-drive night.", ["neon"], True,
    press="glow", opacity=0.94, backdrop=((8, 22, 54), (20, 150, 160)), scene=1,
    mat="palmfrond", face="crest", dpad="oct", sh="ears", small="ears", stick_shape="crest", stick_style="dome", base_shape="octagon", base_style="cross",
    well_a=0.3, glyph="duo", sym_thick=0.95, duo=(255, 70, 170), acc=(255, 120, 190), P=_all((54, 214, 200), (16, 44, 118)), mark_col=(255, 244, 230))

# 2 night-hero -----------------------------------------------------------------------------------------------------
kit("night-hero", "pad-night-hero", "Rooftop Halftone", "Comic-book night sky: deep blue panels, halftone dots, speed lines and thick ink outlines.", ["playful"], False,
    featured=True, press="spring", opacity=0.96, backdrop=((10, 14, 44), (40, 70, 160)), scene=2,
    mat="comicsky", face="diamond", dpad="blade", sh="diag", small="diag", stick_shape="diamond", stick_style="ridges", base_shape="octagon", base_style="ticks",
    well_a=0.4, glyph="pop", sym_thick=1.15, P=_all((92, 140, 250), (18, 28, 98)), mark_col=(255, 255, 255))

kit("night-hero", "pad-night-hero-pro", "Night Shift", "Black ripstop fabric with a faint reflective grid, silver piping and sharp cut-out symbols.", ["material"], True,
    press="shrink", opacity=0.95, backdrop=((6, 8, 18), (28, 36, 64)), scene=2,
    mat="ripstop", face="hexf", dpad="fang", sh="trap", small="trap", stick_shape="hexf", stick_style="concave", base_shape="hexf", base_style="ticks",
    well_a=0.5, glyph="shadow", sym_thick=1.0, acc=(196, 210, 232), P=_all((78, 92, 134), (14, 18, 36)), mark_col=(226, 236, 252))

# 3 steel-claws ----------------------------------------------------------------------------------------------------
kit("steel-claws", "pad-steel-claws", "Berserker Steel", "Raw brushed steel gouged by three deep slashes that glow rage-red from inside.", ["material"], False,
    featured=True, press="grow", opacity=0.95, backdrop=((14, 14, 18), (60, 62, 74)), scene=3,
    mat="gouged", face="claw", dpad="spike", sh="slant", small="slant", stick_shape="slash", stick_style="ridges", base_shape="hex", base_style="dots",
    well_a=0.55, glyph="slashed", sym_thick=1.05, acc=(238, 44, 40), P=_all((190, 196, 208), (62, 66, 78)), mark_col=(244, 60, 52))

kit("steel-claws", "pad-steel-claws-pro", "Rage Mode", "Blood-dark anodised metal ripped open by bright red gashes, with white-hot heat-gradient symbols.", ["neon"], True,
    press="glow", opacity=0.95, backdrop=((20, 2, 4), (110, 14, 18)), scene=3,
    mat="gouged", face="ragged", dpad="fang", sh="chamfer", small="chamfer", stick_shape="ragged", stick_style="grooves", base_shape="ragged", base_style="ticks",
    well_a=0.6, glyph="heat", sym_thick=1.15, acc=(255, 70, 44), P=_all((168, 34, 38), (30, 4, 8)), mark_col=(255, 150, 70))

# 4 souls-dark -----------------------------------------------------------------------------------------------------
kit("souls-dark", "pad-souls-dark", "Ashen Bonfire", "Charred stone buttons with glowing ember cracks and carved, burning symbols.", ["ornate", "material"], False,
    featured=True, press="glow", opacity=0.94, backdrop=((8, 6, 10), (60, 24, 12)), scene=3,
    mat="ashen", face="gothic", dpad="bar", sh="arch", small="arch", stick_shape="gothic", stick_style="concave", base_shape="octagon", base_style="well",
    well_a=0.6, glyph="glow", sym_thick=1.0, acc=(255, 122, 36), P=_all((84, 78, 84), (16, 14, 18)), mark_col=(255, 158, 70))

kit("souls-dark", "pad-souls-dark-pro", "Runebound Silver", "Oxidised dark silver ringed with glowing amber runes and gilded engraved symbols.", ["ornate"], True,
    press="glow", opacity=0.95, backdrop=((10, 10, 16), (56, 46, 40)), scene=3,
    mat="runic", face="sparkle", dpad="house", sh="hexbar", small="hexbar", stick_shape="sparkle", stick_style="ring", base_shape="octagon", base_style="ticks",
    well_a=0.55, glyph="gilt", sym_thick=1.0, acc=(255, 152, 50), P=_all((150, 154, 162), (28, 28, 38)), mark_col=(238, 200, 120))

# 5 zombie-outbreak ------------------------------------------------------------------------------------------------
kit("zombie-outbreak", "pad-zombie-outbreak", "Quarantine Tape", "Grimy olive plates banded with yellow-black hazard tape and dark red splatters.", ["material"], False,
    featured=True, press="shrink", opacity=0.95, backdrop=((14, 18, 12), (58, 66, 40)), scene=1,
    mat="hazardtape", face="tile", dpad="key", sh="notched", small="notched", stick_shape="tile", stick_style="grooves", base_shape="chamfer", base_style="cross",
    well_a=0.55, glyph="stamp", sym_thick=1.1, acc=(244, 202, 22), P=_all((132, 142, 104), (38, 44, 32)), mark_col=(236, 232, 208))

kit("zombie-outbreak", "pad-zombie-outbreak-pro", "Chainlink Siege", "Rusty chain-link fencing over a blood-lit wall, with sprayed-on warning symbols.", ["material"], True,
    press="shrink", opacity=0.95, backdrop=((20, 8, 6), (90, 32, 22)), scene=3,
    mat="chainlink", face="chamfer", dpad="arm", sh="rect", small="rect", stick_shape="chamfer", stick_style="reticle", base_shape="chamfer", base_style="dots",
    well_a=0.5, glyph="spray", sym_thick=1.1, acc=(176, 44, 30), P=_all((176, 182, 184), (60, 64, 68)), mark_col=(255, 226, 150))

# 6 battle-royale --------------------------------------------------------------------------------------------------
kit("battle-royale", "pad-battle-royale", "Loot Rarity", "Glossy loot cells that glow up from the bottom in blue, gold, purple and green rarity colours.", ["playful", "neon"], False,
    featured=True, press="spring", opacity=0.94, backdrop=((14, 20, 40), (44, 72, 120)), scene=4,
    mat="rarity", face="tile", dpad="bar", sh="chamfer", small="chamfer", stick_shape="tile", stick_style="cap", base_shape="chamfer", base_style="ticks",
    well_a=0.4, glyph="sticker", sym_thick=1.05,
    P=_fc(((70, 150, 255), (255, 170, 40), (176, 88, 245), (88, 214, 104)), main=((92, 108, 138), (22, 28, 46))), sym_colors=scol(((255,) * 3,) * 4), mark_col=(255, 255, 255))

kit("battle-royale", "pad-battle-royale-pro", "Storm Circle", "Topographic island map buttons with a glowing violet storm ring closing in from the edge.", ["neon", "material"], True,
    press="glow", opacity=0.94, backdrop=((14, 26, 22), (70, 40, 120)), scene=1,
    mat="stormring", face="octagon", dpad="capsule", sh="hexbar", small="hexbar", stick_shape="octagon", stick_style="reticle", base_shape="circle", base_style="cross",
    well_a=0.45, glyph="pop", sym_thick=1.1, sym_scale=1.05, acc=(156, 76, 255), P=_all((120, 160, 112), (22, 40, 38)), mark_col=(255, 255, 255))

# 7 ronin ----------------------------------------------------------------------------------------------------------
kit("ronin", "pad-ronin", "Sumi Ink", "Rice-paper discs swept by one bold ink stroke, with a small red seal and brush-painted symbols.", ["minimal", "ornate"], False,
    featured=True, press="shrink", opacity=0.95, backdrop=((26, 26, 32), (126, 120, 108)), scene=2,
    mat="sumi", face="circle", dpad="capsule", sh="leaf", small="leaf", stick_shape="circle", stick_style="ring", base_shape="circle", base_style="well",
    well_a=0.3, glyph="brush", sym_thick=1.15, acc=(190, 34, 40), P=_all((242, 236, 220), (80, 74, 70)), mark_col=(214, 44, 48))

kit("ronin", "pad-ronin-pro", "Folded Steel", "Wavy folded-steel grain polished like a sword blade, edged with a thin line of gold.", ["material", "ornate"], True,
    press="shrink", opacity=0.95, backdrop=((14, 16, 22), (70, 78, 94)), scene=2,
    mat="damascus", face="slash", dpad="blade", sh="slant", small="slant", stick_shape="slash", stick_style="dome", base_shape="octagon", base_style="ticks",
    well_a=0.45, glyph="inlay", sym_thick=1.0, acc=(214, 168, 70), P=_all((206, 214, 226), (50, 56, 70)), mark_col=(238, 206, 128))

# 8 wasteland ------------------------------------------------------------------------------------------------------
kit("wasteland", "pad-wasteland", "Scrap Patchwork", "Mismatched salvaged plates welded together, with rivets, rust and stencilled faded symbols.", ["material"], False,
    featured=True, press="grow", opacity=0.95, backdrop=((38, 24, 14), (150, 96, 50)), scene=3,
    mat="scrap", face="plate", dpad="key", sh="diag", small="diag", stick_shape="plate", stick_style="grooves", base_shape="plate", base_style="dots",
    well_a=0.5, glyph="stencil", sym_thick=1.15, P=_all((176, 134, 84), (82, 56, 40)), mark_col=(236, 224, 196))

kit("wasteland", "pad-wasteland-pro", "Dust Storm", "Wind-rippled sand plates with blowing dust, rusty rivets and chalky hand-scrawled symbols.", ["material"], True,
    press="grow", opacity=0.94, backdrop=((70, 44, 22), (214, 150, 84)), scene=3,
    mat="sandripple", face="hex", dpad="oct", sh="ears", small="ears", stick_shape="hex", stick_style="ridges", base_shape="hex", base_style="well",
    well_a=0.4, glyph="chalk", sym_thick=1.2, acc=(128, 70, 40), P=_all((226, 186, 124), (126, 84, 50)), mark_col=(72, 44, 28))

# 9 space-marine ---------------------------------------------------------------------------------------------------
kit("space-marine", "pad-space-marine", "Power Armor", "Layered gunmetal-blue armour plates with chipped paint, recessed panel seams and orange stencil bars.", ["material"], False,
    featured=True, press="shrink", opacity=0.96, backdrop=((10, 14, 24), (60, 78, 110)), scene=2,
    mat="armorplate", face="octagon", dpad="oct", sh="chamfer", small="hexbar", stick_shape="octagon", stick_style="cap", base_shape="octagon", base_style="cross",
    well_a=0.5, glyph="engrave", sym_thick=1.1, acc=(255, 122, 24), P=_all((118, 142, 178), (30, 40, 62)), mark_col=(255, 150, 60))

kit("space-marine", "pad-space-marine-pro", "Orbital Re-entry", "Charred heat-shield tiles glowing orange where the atmosphere burns, with hot cut-in symbols.", ["neon", "material"], True,
    press="glow", opacity=0.95, backdrop=((8, 6, 8), (150, 54, 20)), scene=3,
    mat="reentry", face="hexf", dpad="tri", sh="trap", small="trap", stick_shape="hexf", stick_style="concave", base_shape="hexf", base_style="ticks",
    well_a=0.5, glyph="emboss", sym_thick=1.0, P=_all((255, 140, 50), (92, 24, 14)), mark_col=(255, 230, 190))

# 10 street-racer --------------------------------------------------------------------------------------------------
kit("street-racer", "pad-street-racer", "Tire Tread", "Black rubber with a chunky V-block tread and a bright nitrous-blue pinstripe around each button.", ["material"], False,
    featured=True, press="grow", opacity=0.95, backdrop=((8, 10, 18), (26, 44, 88)), scene=2,
    mat="tread", face="squircle", dpad="bar", sh="slant", small="slant", stick_shape="squircle", stick_style="dimples", base_shape="circle", base_style="ticks",
    well_a=0.5, glyph="shadow", sym_thick=1.1, acc=(44, 156, 255), P=_all((70, 72, 80), (16, 16, 22)), mark_col=(236, 242, 255))

kit("street-racer", "pad-street-racer-pro", "Carbon Nitrous", "Clear-coated carbon weave with a sweeping reflection and a glowing nitrous-blue ring.", ["neon", "material"], True,
    press="glow", opacity=0.95, backdrop=((4, 6, 14), (18, 40, 96)), scene=2,
    mat="carbonblue", face="kite", dpad="spike", sh="chamfer", small="slant", stick_shape="kite", stick_style="ring", base_shape="hex", base_style="cross",
    well_a=0.5, glyph="glow", sym_thick=0.95, acc=(54, 170, 255), P=_all((60, 64, 76), (8, 8, 12)), mark_col=(120, 200, 255))

# 11 cyber-merc ----------------------------------------------------------------------------------------------------
kit("cyber-merc", "pad-cyber-merc", "Glitch Visor", "Dark smoked glass with scanlines, glitched slices and cyan-magenta split edges.", ["neon"], False,
    featured=True, press="glow", opacity=0.93, backdrop=((8, 4, 28), (60, 10, 90)), scene=2,
    mat="holoplate", face="slash", dpad="clip", sh="diag", small="diag", stick_shape="slash", stick_style="reticle", base_shape="chamfer", base_style="ticks",
    well_a=0.5, glyph="glitch", sym_thick=0.95, P=_all((38, 22, 78), (8, 6, 24)), mark_col=(244, 244, 255))

kit("cyber-merc", "pad-cyber-merc-pro", "Chrome Contract", "Mirror chrome that reflects a magenta and cyan neon skyline, edged with fine circuit scratches.", ["material", "neon"], True,
    press="glow", opacity=0.95, backdrop=((14, 4, 36), (92, 16, 120)), scene=2,
    mat="mercchrome", face="diamond", dpad="oct", sh="hexbar", small="hexbar", stick_shape="diamond", stick_style="dome", base_shape="diamond", base_style="cross",
    well_a=0.45, glyph="pop", sym_thick=1.1, P=_all((150, 96, 220), (20, 24, 70)), mark_col=(255, 255, 255))

# 12 pirate-cove ---------------------------------------------------------------------------------------------------
kit("pirate-cove", "pad-pirate-cove", "Rope and Plank", "Weathered ship-deck planks framed by twisted hemp rope, with branded-in symbols.", ["material"], False,
    featured=True, press="shrink", opacity=0.95, backdrop=((16, 30, 44), (60, 84, 92)), scene=1,
    mat="ropewood", face="circle", dpad="key", sh="pill", small="pill", stick_shape="circle", stick_style="grooves", base_shape="circle", base_style="well",
    well_a=0.55, glyph="engrave", sym_thick=1.1, wood=((92, 58, 30), (190, 138, 82)), P=_all((190, 138, 82), (92, 58, 30)), mark_col=(246, 232, 200))

kit("pirate-cove", "pad-pirate-cove-pro", "Doubloon Gold", "Milled-edge gold coins with worn faces, a warm glint and gilded engraved symbols.", ["ornate", "material"], True,
    press="spring", opacity=0.96, backdrop=((12, 24, 38), (42, 66, 74)), scene=1,
    mat="doubloon", face="scallop", dpad="dot", sh="ticket", small="ticket", stick_shape="circle", stick_style="cap", base_shape="scallop", base_style="dots",
    well_a=0.4, glyph="gilt", sym_thick=1.0, P=_all((240, 198, 84), (122, 80, 24)), mark_col=(250, 220, 130))

# 13 frontier ------------------------------------------------------------------------------------------------------
kit("frontier", "pad-frontier", "Saddle Leather", "Tooled saddle leather with carved scrolls, burnished edges and cream stitching, like a worn riding tack set.", ["material"], False,
    featured=True, press="shrink", opacity=0.95, backdrop=((38, 20, 10), (130, 78, 40)), scene=3,
    mat="tooled", face="scallop", dpad="arm", sh="tab", small="tab", stick_shape="scallop", stick_style="ridges", base_shape="scallop", base_style="well",
    well_a=0.5, glyph="stitch", sym_thick=1.15, acc=(236, 216, 172), P=_all((176, 112, 62), (66, 34, 16)), mark_col=(246, 228, 190))

kit("frontier", "pad-frontier-pro", "Dry Gulch Clay", "Sun-baked cracked clay glowing in desert sunset red, with deep-pressed symbols.", ["material"], True,
    press="grow", opacity=0.95, backdrop=((60, 24, 20), (226, 126, 60)), scene=3,
    mat="claycrack", face="arch", dpad="house", sh="arch", small="arch", stick_shape="arch", stick_style="dimples", base_shape="octagon", base_style="dots",
    well_a=0.45, glyph="stamp", sym_thick=1.15, P=_all((232, 142, 78), (110, 40, 28)), mark_col=(66, 24, 16))

# ---- legibility fixes for older skins (applied after registration; the GLYPH_FIX salt rebuilds them) -------------------
_BY_ID = {s_.id: s_ for s_ in SKINS}


def _tune(sid, **kw):
    for k, v in kw.items():
        setattr(_BY_ID[sid], k, v)


_tune("stealth-matte", mark_col=(178, 184, 198))
_tune("pad-leather-racing", mark_col=(255, 112, 92), sym_thick=1.4)
_tune("pad-denim-stitch", mark_col=(238, 234, 224))
_tune("pad-crystal-gem", sym_thick=1.45)
_tune("pad-felt-patch", mark_col=(38, 36, 54))
_tune("pad-wax-seal", glyph="gilt", mark_col=(255, 224, 160))
_tune("pad-origami", mark_col=(34, 52, 96))
_tune("pad-paper-cut", mark_col=(46, 52, 80))
_tune("pad-toy-bricks", sym_scale=0.92)
_tune("pad-microswitch", sym_scale=0.95, sym_thick=1.0)
_tune("pad-mech-keys", sym_scale=0.95, sym_thick=0.9)
_tune("pad-arcade-cabinet", sym_scale=0.95, sym_thick=1.0)
#### ---- KITS END ----
def main(argv):
    force = "--force" in argv
    sheet = None
    ids = []
    it = iter(argv)
    for a in it:
        if a == "--sheet":
            sheet = next(it)
        elif not a.startswith("--"):
            ids.append(a)
    CTRL.mkdir(parents=True, exist_ok=True)
    cats = keep_label_locales([{"id": i, "labels": {"en": n}} for i, n in CATEGORIES], CTRL / "categories.json")
    write_text(CTRL / "categories.json", json.dumps(cats, indent=2, ensure_ascii=False) + "\n")
    chosen = [s for s in SKINS if not ids or s.id in ids]
    if "--refresh-wide-controls" in argv:
        for s in chosen:
            refresh_wide_controls(s)
            print("wide controls " + s.id, flush=True)
        return
    if "--add-neutral" in argv:
        for s in chosen:
            add_neutral(s)
            print("neutral " + s.id, flush=True)
        return
    for s in chosen:
        built = build_skin(s, force)
        print(("built   " if built else "skipped ") + s.id, flush=True)
    if sheet:
        contact_sheet(chosen, sheet)


if __name__ == "__main__":
    main(sys.argv[1:])
