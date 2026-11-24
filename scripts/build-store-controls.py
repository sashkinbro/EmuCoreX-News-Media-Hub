#!/usr/bin/env python3
"""Builds the EmuCore Store on-screen controller skins (kind "controls").

Everything is procedural art (Python + Pillow only, no downloads, no fonts, no third-party artwork).
Symbols and labels are original vector strokes; face symbols are generic geometric shapes.

Output (the listing.json files and categories.json are generated from the data in this script):
  store/controls/categories.json
  store/controls/src/<id>/listing.json
  store/controls/dist/<id>/item.ecxcontrols   deterministic ZIP: controls.json + 35 PNG images
  store/controls/dist/<id>/cover.webp         640x360 card image
  store/controls/dist/<id>/ingame.webp        1280x720 gallery mock of the full on-screen layout
  store/controls/dist/<id>/build.json         {"sourceSha256": hash(listing + this script)}

Usage: python scripts/build-store-controls.py [skin-id ...] [--force] [--sheet contact.png]
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

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent.parent
CTRL = ROOT / "store" / "controls"
SRC = CTRL / "src"
DIST = CTRL / "dist"
ZIP_DATE = (2020, 1, 1, 0, 0, 0)
S = 3  # supersampling for the vector skins

FACE = ["cross", "circle", "square", "triangle"]
DIRS = ["up", "down", "left", "right"]
SHOULDERS = ["l1", "l2", "r1", "r2"]
PARTS = FACE + ["dpad_" + d for d in DIRS] + SHOULDERS + ["l3", "r3", "start", "select", "analog_base", "analog_stick"]
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
        return {"sh1": (14, 52, 242, 140), "sh2": (14, 30, 242, 162), "small": (30, 62, 226, 130)}[self.kind]

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
    id, name = "kawaii-pastel", "Kawaii Pastel"
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
            if k == "face":
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
            if k == "small":
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
    id, name = "arcade-cabinet", "Arcade Cabinet"
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
# packaging: previews, cover, gallery, ZIP
# ----------------------------------------------------------------------------------------------------------------

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


def build_zip(skin, images):
    manifest = {"formatVersion": 1, "id": skin.id, "version": 1,
                "skin": {"images": {k: k + ".png" for k in sorted(images)}, "tint": None, "opacity": skin.opacity,
                         "pressEffect": skin.press}}
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
              ("playful", "Playful"), ("accessibility", "Accessibility")]


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
    return {"version": 1, "categories": skin.cats, "tags": skin.tags, "featured": skin.featured,
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


def render_all(skin):
    T = {}
    for part in PARTS:
        T[part] = skin.render(part, False)
        if part != "analog_base":
            T[part + "_pressed"] = skin.render(part, True)
    return T


def build_skin(skin, force=False, tiles_dir=None):
    out_dir = DIST / skin.id
    listing_text = json.dumps(keep_locales(listing_of(skin), SRC / skin.id / "listing.json"), indent=2, ensure_ascii=False) + "\n"
    write_text(SRC / skin.id / "listing.json", listing_text)
    digest = digest_of(listing_text)
    stamp = out_dir / "build.json"
    need = ["item.ecxcontrols", "cover.webp", "ingame.webp", "build.json"]
    if not force and all((out_dir / n).is_file() for n in need) and json.loads(stamp.read_text())["sourceSha256"] == digest:
        return False
    T = render_all(skin)
    images = {k: png_bytes(v) for k, v in T.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "item.ecxcontrols").write_bytes(build_zip(skin, images))
    (out_dir / "cover.webp").write_bytes(webp_bytes(make_cover(skin, T)))
    (out_dir / "ingame.webp").write_bytes(webp_bytes(make_ingame(skin, T)))
    stamp.write_text(json.dumps({"sourceSha256": digest}, indent=2) + "\n", encoding="utf-8", newline="\n")
    return True


def contact_sheet(skins, path):
    cw, chh, cols = 320, 180, 5
    rows = (len(skins) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw, rows * chh), (0, 0, 0))
    for i, s in enumerate(skins):
        im = Image.open(DIST / s.id / "cover.webp").convert("RGB").resize((cw, chh), Image.LANCZOS)
        sheet.paste(im, ((i % cols) * cw, (i // cols) * chh))
    sheet.save(path)


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
    for s in chosen:
        built = build_skin(s, force)
        print(("built   " if built else "skipped ") + s.id, flush=True)
    if sheet:
        contact_sheet(chosen, sheet)


if __name__ == "__main__":
    main(sys.argv[1:])
