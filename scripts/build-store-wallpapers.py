#!/usr/bin/env python3
"""Procedural wallpaper generator for the EmuCore Store (everything is original, CC0).

Writes store/wallpaper/{categories.json, src/<id>/listing.json, dist/<id>/{item.ecxwall,cover.webp,phone.webp,build.json}}.
Pure numpy + Pillow (cv2 is used for blurs when available, with a Pillow fallback).
Deterministic per id (seeded), idempotent (skips items whose build.json hash matches).

  python scripts/build-store-wallpapers.py            # build everything that changed
  python scripts/build-store-wallpapers.py --force    # rebuild all
  python scripts/build-store-wallpapers.py --only a,b # selected ids
  python scripts/build-store-wallpapers.py --sheet out.png   # also write a contact sheet of covers
"""
import argparse
import hashlib
import io
import json
import math
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

HUB = Path(__file__).resolve().parents[1]
ROOT = HUB / "store" / "wallpaper"
W, H = 1440, 2560
PUBLISHED = "2026-11-24T12:00:00Z"
CREATOR_URL = "https://github.com/sashkinbro/EmuCoreX-News-Media-Hub"

YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


# ====================================================================== basics
def col(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0


def smooth(e0, e1, x):
    t = np.clip((np.asarray(x, np.float32) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def ramp(t, stops):
    pos = [s[0] for s in stops]
    cs = [col(s[1]) for s in stops]
    t = np.clip(t, 0, 1)
    out = np.empty(t.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(t, pos, [x[c] for x in cs])
    return out


def vgrad(stops):
    line = ramp(np.linspace(0, 1, H, dtype=np.float32), stops)
    return np.ascontiguousarray(np.broadcast_to(line[:, None, :], (H, W, 3)))


def blur(a, s):
    if s <= 0:
        return a
    if cv2 is not None:
        return cv2.GaussianBlur(np.ascontiguousarray(a, dtype=np.float32), (0, 0), s)
    f = max(1, int(s / 1.6))
    h, w = a.shape[:2]
    chans = a.shape[2] if a.ndim == 3 else 1
    res = []
    for c in range(chans):
        ch = a[..., c] if a.ndim == 3 else a
        im = Image.fromarray(np.ascontiguousarray(ch, dtype=np.float32), "F")
        im = im.resize((max(1, w // f), max(1, h // f)), Image.BILINEAR).resize((w, h), Image.BICUBIC)
        res.append(np.asarray(im))
    return np.stack(res, -1) if a.ndim == 3 else res[0]


def noise(seed, cx, cy=None, h=H, w=W):
    r = np.random.default_rng(seed)
    cy = cy if cy is not None else cx * h / w
    gw, gh = int(cx) + 4, int(cy) + 4
    g = r.random((gh, gw)).astype(np.float32)
    im = Image.fromarray(g, "F").resize((int(gw * w / cx), int(gh * h / cy)), Image.BICUBIC)
    return np.asarray(im)[:h, :w]


def fbm(seed, cx=3.0, oct=5, pers=0.5, cy=None, h=H, w=W, norm=True):
    tot = 0
    amp = 1.0
    for o in range(oct):
        tot = tot + noise(seed * 131 + o, cx * 2 ** o, None if cy is None else cy * 2 ** o, h, w) * amp
        amp *= pers
    if norm:
        s = tot[::7, ::7]
        lo, hi = np.percentile(s, [1, 99])
        tot = np.clip((tot - lo) / (hi - lo + 1e-6), 0, 1)
    return tot.astype(np.float32)


def warp_sample(a, wx, wy, amt):
    xs = np.clip(XX[: a.shape[0], : a.shape[1]] + (wx - 0.5) * amt, 0, a.shape[1] - 1).astype(np.int32)
    ys = np.clip(YY[: a.shape[0], : a.shape[1]] + (wy - 0.5) * amt, 0, a.shape[0] - 1).astype(np.int32)
    return a[ys, xs]


def comp(img, mask, color, k=1.0):
    m = (mask * k)[..., None] if mask.ndim == 2 else mask * k
    img *= 1 - m
    img += m * color
    return img


def draw_mask(fn, ss=2):
    im = Image.new("L", (W * ss, H * ss), 0)
    fn(ImageDraw.Draw(im), ss)
    return np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255.0


def rgb_canvas(ss=2, bg=(0, 0, 0)):
    return Image.new("RGB", (W * ss, H * ss), tuple(int(c * 255) for c in bg))


def from_pil(im):
    if im.size != (W, H):
        im = im.resize((W, H), Image.BOX)
    return np.asarray(im, np.float32) / 255.0


def glow(img, cx, cy, r, color, k=1.0, kind="exp", reach=4.5):
    x0, x1 = int(max(0, cx - r * reach)), int(min(W, cx + r * reach))
    y0, y1 = int(max(0, cy - r * reach)), int(min(H, cy + r * reach))
    if x1 <= x0 or y1 <= y0:
        return
    d = np.sqrt((XX[y0:y1, x0:x1] - cx) ** 2 + (YY[y0:y1, x0:x1] - cy) ** 2) / r
    if kind == "exp":
        g = np.exp(-d * 2.4)
    elif kind == "gauss":
        g = np.exp(-d * d * 1.6)
    else:
        g = 1.0 / (1.0 + d * d * 5)
    img[y0:y1, x0:x1] += g[..., None] * np.asarray(color, np.float32) * k


def shoulder(x, knee=0.82):
    return np.where(x < knee, x, knee + (1 - knee) * np.tanh((x - knee) / (1 - knee)))


def vignette(img, k=0.35, power=2.2, cy=0.5):
    d = np.sqrt(((XX / W - 0.5) * 1.15) ** 2 + ((YY / H - cy) * 0.8) ** 2)
    img *= (1 - k * np.clip(d * 1.6, 0, 1) ** power)[..., None]


def noise1(r, k, x):
    v = r.random(int(k) + 4).astype(np.float32)
    i = np.floor(x).astype(int)
    f = x - i
    f = f * f * (3 - 2 * f)
    return v[i] * (1 - f) + v[i + 1] * f


def profile(seed, base, amp, freq=1.6, oct=6, pers=0.5, sharp=0.0, n=W, skew=0.0):
    r = np.random.default_rng(seed)
    x = np.linspace(0, 1, n, dtype=np.float32)
    tot = np.zeros(n, np.float32)
    a = 1.0
    norm = 0
    for o in range(oct):
        v = noise1(r, freq * 2 ** o, x * freq * 2 ** o)
        if sharp:
            v = (1 - sharp) * v + sharp * (1 - np.abs(2 * v - 1))
        tot += v * a
        norm += a
        a *= pers
    tot /= norm
    tot = (tot - tot.min()) / (tot.max() - tot.min() + 1e-6)
    return base - amp * tot + skew * (x - 0.5)


def ridge(img, prof, top, bot, fade_y=None, lit=None, shade=None, lit_k=0.5, lit_fade=260.0, tex=0.0, seed=0,
          dirn=1.0, slope_scale=1.4):
    ymin = float(prof.min())
    t = np.clip((YY[:, :1] - ymin) / (fade_y if fade_y else max(1.0, H - ymin)), 0, 1)
    base = ramp(t, [(0, top), (1, bot)]) if isinstance(top, str) else None
    base = np.broadcast_to(base, (H, W, 3)).copy()
    xs = np.arange(len(prof))
    pts = [(float(x) * W / len(prof), float(p)) for x, p in zip(xs[::2], prof[::2])] + [(W, prof[-1]), (W, H + 4), (0, H + 4)]

    def f(d, ss):
        d.polygon([(x * ss, y * ss) for x, y in pts], fill=255)

    m = draw_mask(f)
    if lit is not None:
        sl = np.gradient(prof) * dirn
        sl = np.convolve(sl, np.ones(31) / 31, mode='same')
        sl = np.interp(np.arange(W), np.linspace(0, W - 1, len(prof)), sl)
        lf = smooth(-slope_scale, slope_scale, sl * 1.0)[None, :, None]
        pr = np.interp(np.arange(W), np.linspace(0, W - 1, len(prof)), prof)[None, :]
        decay = np.exp(-np.clip(YY - pr, 0, None) / lit_fade)[..., None]
        lc = col(lit)
        sc = col(shade if shade else top)
        base = base + (lc - base) * lf * decay * lit_k + (sc - base) * (1 - lf) * decay * lit_k
    if tex:
        n = fbm(seed + 5, 6, 5)
        base *= (1 + (n[..., None] - 0.5) * tex)
    comp(img, m, base)
    return m


def fog(img, y, h, color, k=0.5, seed=None, cover=0.6):
    band = np.exp(-(((YY - y) / h) ** 2))
    if seed is not None:
        n = fbm(seed, 2.0, 4, cy=6)
        band = band * (1 - cover + cover * n)
    img += (col(color) - img) * (band * k)[..., None] if False else 0
    c = col(color)
    m = np.clip(band * k, 0, 1)[..., None]
    img *= 1 - m
    img += c * m


def stars(img, seed, n, ymin=0, ymax=H, amp=(0.25, 1.0), size=0.8, tints=("#FFFFFF", "#BFD7FF", "#FFE6C4"), big=0, fade=None):
    r = np.random.default_rng(seed)
    xs = r.random(n) * (W - 4) + 2
    ys = ymin + r.random(n) * (ymax - ymin - 4) + 2
    b = amp[0] + r.random(n) ** 3 * (amp[1] - amp[0])
    if fade is not None:
        b = b * fade(ys)
    tc = np.array([col(t) for t in tints])[r.integers(0, len(tints), n)]
    layer = np.zeros((H, W, 3), np.float32)
    ix, iy = xs.astype(int), ys.astype(int)
    fx, fy = (xs - ix).astype(np.float32), (ys - iy).astype(np.float32)
    for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        np.add.at(layer, (iy + dy, ix + dx), tc * (b * wt)[:, None])
    layer = blur(layer, size) * 2.2 * size + layer * 0.6
    img += layer
    for i in range(big):
        x, y = r.random() * W, ymin + r.random() * (ymax - ymin)
        c = col(tints[int(r.integers(0, len(tints)))])
        glow(img, x, y, 10 + r.random() * 10, c, 0.55, "exp")
        glow(img, x, y, 3.2, c, 1.2, "exp")
        # flare cross
        L = 30 + r.random() * 36
        for ax in (0, 1):
            if ax == 0:
                xa, xb = int(max(0, x - L)), int(min(W, x + L))
                yy = int(y)
                if 0 <= yy < H and xb > xa:
                    fall = np.exp(-np.abs(np.arange(xa, xb) - x) / (L * 0.28))
                    img[yy, xa:xb] += c * fall[:, None] * 0.6
            else:
                ya, yb = int(max(0, y - L)), int(min(H, y + L))
                xx = int(x)
                if 0 <= xx < W and yb > ya:
                    fall = np.exp(-np.abs(np.arange(ya, yb) - y) / (L * 0.28))
                    img[ya:yb, xx] += c * fall[:, None] * 0.6


def bokeh(img, seed, n, colors, rmin, rmax, alpha=(0.08, 0.35), ring=0.5, region=(0, 0, W, H), soft=0.12, hard=0.0):
    r = np.random.default_rng(seed)
    for _ in range(n):
        rad = rmin + (rmax - rmin) * r.random() ** 1.5
        cx = region[0] + r.random() * (region[2] - region[0])
        cy = region[1] + r.random() * (region[3] - region[1])
        c = col(colors[int(r.integers(0, len(colors)))])
        a = alpha[0] + (alpha[1] - alpha[0]) * r.random()
        x0, x1 = int(max(0, cx - rad - 2)), int(min(W, cx + rad + 2))
        y0, y1 = int(max(0, cy - rad - 2)), int(min(H, cy + rad + 2))
        if x1 <= x0 or y1 <= y0:
            continue
        d = np.sqrt((XX[y0:y1, x0:x1] - cx) ** 2 + (YY[y0:y1, x0:x1] - cy) ** 2) / rad
        disc = 1 - smooth(1 - soft, 1.0, d)
        rim = np.exp(-(((d - 0.94) / 0.07) ** 2)) * ring
        v = (disc * (0.55 + 0.25 * (1 - d)) + rim * disc.clip(0.3, 1)) * a
        img[y0:y1, x0:x1] += v[..., None] * c


def dots(img, seed, n, color, rmin, rmax, blur_s, k=0.9, region=(0, 0, W, H), tints=None):
    r = np.random.default_rng(seed)
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for _ in range(n):
        x = region[0] + r.random() * (region[2] - region[0])
        y = region[1] + r.random() * (region[3] - region[1])
        rad = rmin + (rmax - rmin) * r.random() ** 2
        d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=int(120 + 135 * r.random()))
    a = np.asarray(m, np.float32) / 255.0
    a = blur(a, blur_s) * (1 + blur_s * 0.15)
    c = col(color)
    img += a[..., None] * c * k


def neon(img, mask, color, core=0.65, glows=((3.0, 1.0), (12.0, 0.8), (45.0, 0.55)), k=1.0):
    c = col(color)
    cc = c + (1 - c) * core
    img += mask[..., None] * cc * k
    for s, g in glows:
        img += blur(mask, s)[..., None] * c * g * k * (1 + s * 0.04)


def finish(img, seed, grain=1.6, vig=0.0):
    img = shoulder(img)
    if vig:
        vignette(img, vig)
    r = np.random.default_rng(seed + 99)
    g = r.standard_normal((H, W, 1), dtype=np.float32) * grain
    g2 = r.standard_normal((H, W, 3), dtype=np.float32) * (grain * 0.35)
    out = np.clip(img * 255.0 + g + g2 + r.random((H, W, 3), dtype=np.float32) - 0.5, 0, 255)
    return Image.fromarray(out.astype(np.uint8), "RGB")


# ====================================================================== components
def nebula(img, seed, stops, cx=2.0, warp=0.55, power=1.5, strength=1.0, dust=0.5, oct=6, mask=None):
    n = fbm(seed, cx, oct)
    w1 = fbm(seed + 3, cx * 1.4, 4)
    w2 = fbm(seed + 4, cx * 1.4, 4)
    n = warp_sample(n, w1, w2, warp * W * 0.45)
    n = warp_sample(n, w2, w1, warp * W * 0.2)
    t = n ** power
    c = ramp(t, stops) * (t[..., None] ** 0.7) * strength
    if mask is not None:
        c = c * mask[..., None]
    img += c
    if dust:
        d = fbm(seed + 9, cx * 2.2, 5)
        img *= 1 - dust * smooth(0.5, 0.85, d)[..., None]


def sphere_light(xx, yy, z, light):
    L = np.array(light, np.float32)
    L /= np.linalg.norm(L)
    return xx * L[0] + yy * L[1] + z * L[2], L


def planet(img, cx, cy, r, pal, light=(-0.6, -0.45, 0.65), seed=1, bands=0.0, cells=3.0, atmo=None, atmo_k=1.0,
           ambient=0.035, halo=1.0, bandwarp=0.5, tilt=0.0, spec=0.0, crater=0.0):
    pad = r * 0.9
    x0, x1 = int(max(0, cx - r - pad)), int(min(W, cx + r + pad))
    y0, y1 = int(max(0, cy - r - pad)), int(min(H, cy + r + pad))
    ph, pw = y1 - y0, x1 - x0
    if ph <= 2 or pw <= 2:
        return
    xx = (XX[y0:y1, x0:x1] - cx) / r
    yy = (YY[y0:y1, x0:x1] - cy) / r
    d2 = xx * xx + yy * yy
    d = np.sqrt(d2)
    z = np.sqrt(np.clip(1 - d2, 0, 1))
    lam, L = sphere_light(xx, yy, z, light)
    n = fbm(seed, cells, 6, 0.55, h=ph, w=pw)
    if bands:
        yb = yy * np.cos(tilt) + xx * np.sin(tilt)
        t = 0.5 + 0.5 * np.sin(yb * bands * math.pi + (n - 0.5) * bandwarp * 9 + np.sin(yb * bands * 2.7) * 0.6)
        t = 0.65 * t + 0.35 * n
    else:
        t = n
    base = ramp(t, pal)
    if crater:
        cn = fbm(seed + 11, cells * 3, 4, 0.6, h=ph, w=pw)
        base *= (1 - crater * smooth(0.55, 0.85, cn))[..., None]
    lit = smooth(-0.1, 0.6, lam)
    colr = base * (ambient + (1 - ambient) * lit)[..., None] * (0.6 + 0.4 * z)[..., None] * 1.15
    if spec:
        hv = np.array([L[0], L[1], L[2] + 1.0], np.float32)
        hv /= np.linalg.norm(hv)
        sp = np.clip(xx * hv[0] + yy * hv[1] + z * hv[2], 0, 1) ** 60
        colr += sp[..., None] * spec
    if atmo is not None:
        ac = col(atmo)
        rim = (1 - z) ** 2.6 * smooth(-0.35, 0.5, lam)
        colr += ac * (rim * atmo_k)[..., None]
        edge = np.where(d > 1, 1.0, 0.0)
        ldir = (xx * L[0] + yy * L[1]) / np.maximum(d, 1e-3)
        hl = np.exp(-(d - 1) * 11) * edge * smooth(-0.55, 0.6, ldir) * 0.55 * halo
        hl2 = np.exp(-(d - 1) * 3.2) * edge * smooth(-0.3, 0.7, ldir) * 0.12 * halo
        win = np.clip((1 + pad / r - d) / 0.35, 0, 1)
        img[y0:y1, x0:x1] += ((hl + hl2)[..., None] * ac * win[..., None])
    a = np.clip((1 - d) * r + 0.5, 0, 1)[..., None]
    sub = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = sub * (1 - a) + colr * a


def ring_system(img, cx, cy, r, inner, outer, tilt, angle, pal, front, seed=3, alpha=0.9, light_k=1.0):
    pad = r * (outer + 0.2)
    x0, x1 = int(max(0, cx - pad)), int(min(W, cx + pad))
    y0, y1 = int(max(0, cy - pad)), int(min(H, cy + pad))
    xx = (XX[y0:y1, x0:x1] - cx) / r
    yy = (YY[y0:y1, x0:x1] - cy) / r
    ca, sa = math.cos(angle), math.sin(angle)
    xr = xx * ca + yy * sa
    yr = -xx * sa + yy * ca
    rho = np.sqrt(xr * xr + (yr / tilt) ** 2)
    u = (rho - inner) / (outer - inner)
    inside = smooth(0, 0.01, u) * (1 - smooth(0.985, 1.0, u))
    rn = np.random.default_rng(seed)
    dens = np.zeros_like(rho)
    for k in range(5):
        f = 18 + 37 * k
        dens += (0.5 + 0.5 * np.sin(rho * f + rn.random() * 6.28)) * (0.5 ** k)
    dens = dens / 1.9
    gap = 1 - 0.85 * np.exp(-(((u - 0.62) / 0.025) ** 2))
    gap *= 1 - 0.6 * np.exp(-(((u - 0.33) / 0.012) ** 2))
    a = inside * (0.25 + 0.75 * dens) * gap * alpha
    a *= smooth(0.0, 0.1, u) * 0.6 + 0.4
    side = (yr > 0) if front else (yr <= 0)
    a = a * side
    c = ramp(u + (dens - 0.5) * 0.1, pal) * light_k
    sub = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = sub * (1 - a[..., None]) + c * a[..., None]


def glass_orb(img, cx, cy, r, tint, light=(-0.5, -0.6, 0.65), seed=1):
    pad = int(r * 0.4)
    x0, x1 = int(max(0, cx - r - pad)), int(min(W, cx + r + pad))
    y0, y1 = int(max(0, cy - r - pad)), int(min(H, cy + r + pad))
    xx = (XX[y0:y1, x0:x1] - cx) / r
    yy = (YY[y0:y1, x0:x1] - cy) / r
    d2 = xx * xx + yy * yy
    d = np.sqrt(d2)
    z = np.sqrt(np.clip(1 - d2, 0, 1))
    lam, L = sphere_light(xx, yy, z, light)
    tc = col(tint)
    fres = (1 - z) ** 2.4
    body = 0.16 + 0.55 * fres
    inner = np.clip(-lam, 0, 1) ** 1.5 * z  # caustic glow opposite to light
    colr = tc * (0.55 + 0.7 * np.clip(lam, 0, 1))[..., None] + (tc * 0.5 + 0.5)[None, None, :] * inner[..., None] * 0.9
    hv = np.array([L[0], L[1], L[2] + 1.0], np.float32)
    hv /= np.linalg.norm(hv)
    sp = np.clip(xx * hv[0] + yy * hv[1] + z * hv[2], 0, 1) ** 90 * 1.4
    # window highlight
    hl = np.exp(-(((xx + 0.36) / 0.20) ** 2 + ((yy + 0.42) / 0.09) ** 2)) * 0.55
    a = np.clip((1 - d) * r + 0.5, 0, 1)
    alpha = (body + inner * 0.45)[..., None] * a[..., None]
    sub = img[y0:y1, x0:x1]
    sub = sub * (1 - alpha) + colr * alpha
    sub += ((sp + hl) * a)[..., None] * (z[..., None] > 0)
    img[y0:y1, x0:x1] = sub
    # soft outer halo
    h = np.exp(-(d - 1) * 6) * (d > 1) * 0.18
    img[y0:y1, x0:x1] += h[..., None] * tc


def pine(d, x, base, h, w, ss, tiers=6, jag=0.18):
    pts = [(x, base - h)]
    right, left = [], []
    for i in range(1, tiers + 1):
        t = i / tiers
        yy = base - h + h * t
        ww = w * (0.18 + 0.82 * t)
        right.append((x + ww, yy))
        left.append((x - ww, yy))
        if i < tiers:
            right.append((x + ww * (1 - jag), yy))
            left.append((x - ww * (1 - jag), yy))
    # zigzag silhouette
    zz = [(x, base - h)]
    for i in range(1, tiers + 1):
        t = i / tiers
        t0 = (i - 1) / tiers
        ww = w * (0.12 + 0.88 * t)
        yy = base - h + h * t
        zz.append((x + ww, yy))
        zz.append((x + ww * (1 - jag * 1.8), yy - h / tiers * 0.28))
    zz.append((x + w * 0.06, base))
    zz.append((x - w * 0.06, base))
    for i in range(tiers, 0, -1):
        t = i / tiers
        ww = w * (0.12 + 0.88 * t)
        yy = base - h + h * t
        zz.append((x - ww * (1 - jag * 1.8), yy - h / tiers * 0.28))
        zz.append((x - ww, yy))
    d.polygon([(a * ss, b * ss) for a, b in zz], fill=255)


def forest_mask(seed, base_fn, n, hrange, wr=0.34, x0=-40, x1=W + 40, tiers=(5, 8)):
    r = np.random.default_rng(seed)

    def f(d, ss):
        for _ in range(n):
            x = x0 + r.random() * (x1 - x0)
            h = hrange[0] + (hrange[1] - hrange[0]) * r.random()
            pine(d, x, base_fn(x) + h * 0.08, h, h * wr * (0.8 + 0.4 * r.random()), ss, int(r.integers(tiers[0], tiers[1] + 1)))

    return draw_mask(f)


def spiral_points(seed, N, R, tilt, rot, arms=2, wind=5.0):
    r = np.random.default_rng(seed)
    rr = R * r.random(N) ** 1.35
    arm = r.integers(0, arms, N)
    th = arm * (2 * math.pi / arms) + wind * np.log1p(rr / (R * 0.12)) + r.standard_normal(N) * (0.5 / (1 + rr / (R * 0.5)))
    sc = 4 + rr * 0.05
    x = rr * np.cos(th) + r.standard_normal(N) * sc
    y = rr * np.sin(th) + r.standard_normal(N) * sc
    return x, y, rr / R


# ====================================================================== space
def g_nebula(seed, bg, stops, stops2=None, cx=1.8, warp=0.6, stars_n=2600):
    img = vgrad([(0, bg[0]), (1, bg[1])])
    nebula(img, seed, stops, cx, warp, 1.5, 1.15, 0.45)
    if stops2:
        nebula(img, seed + 20, stops2, cx * 1.2, warp, 1.8, 0.8, 0.0)
    stars(img, seed + 1, stars_n, size=0.7, big=14)
    stars(img, seed + 2, 500, amp=(0.5, 1.0), size=1.1)
    return img, 1.5, 0.15


def g_planet_ring(seed):
    img = vgrad([(0, "#02030A"), (0.5, "#05071A"), (1, "#0B0A24")])
    nebula(img, seed, [(0, "#000000"), (0.5, "#1B1650"), (0.8, "#6B3FA0"), (1, "#E7A7D8")], 1.6, 0.6, 1.9, 0.55, 0.3)
    stars(img, seed + 1, 2200, size=0.7, big=8)
    cx, cy, r = 720, 1290, 400
    pal = [(0, "#5B3A22"), (0.35, "#B7814A"), (0.6, "#E3BE86"), (0.85, "#C58F57"), (1, "#F3DDB4")]
    rp = [(0, "#6E5A44"), (0.25, "#CBB08A"), (0.5, "#F1DDB8"), (0.75, "#B79E7A"), (1, "#7B6650")]
    ang = -0.32
    ring_system(img, cx, cy, r, 1.35, 2.35, 0.2, ang, rp, False, 3, 0.92)
    planet(img, cx, cy, r, pal, (-0.7, -0.35, 0.6), seed + 3, bands=7.0, cells=3, atmo="#F4C98E", atmo_k=0.5, halo=0.7, bandwarp=0.5, tilt=ang)
    ring_system(img, cx, cy, r, 1.35, 2.35, 0.2, ang, rp, True, 3, 0.92)
    # small moon
    planet(img, 1130, 520, 62, [(0, "#4A4A52"), (0.5, "#9C9AA2"), (1, "#E6E3E6")], (-0.7, -0.3, 0.6), seed + 8, cells=5, crater=0.5)
    planet(img, 260, 2180, 34, [(0, "#3C2B35"), (0.5, "#9E6F6F"), (1, "#E0B7A8")], (-0.7, -0.3, 0.6), seed + 9, cells=4, crater=0.4)
    return img, 1.4, 0.2


def g_eclipse(seed):
    img = vgrad([(0, "#010209"), (0.45, "#040617"), (1, "#050314")])
    nebula(img, seed, [(0, "#000000"), (0.6, "#10163A"), (1, "#3B3E92")], 1.5, 0.5, 2.0, 0.28, 0.3)
    stars(img, seed + 1, 2400, size=0.7, big=6)
    cx, cy, R = 720, 1180, 330
    dx, dy = XX - cx, YY - cy
    rho = np.sqrt(dx * dx + dy * dy) / R
    th = np.arctan2(dy, dx)
    r = np.random.default_rng(seed)
    thx = np.linspace(0, 2 * math.pi, 2048, endpoint=False)

    def ang_noise(k, ph):
        v = np.zeros_like(thx)
        for o in range(4):
            v += np.sin(thx * k * 2 ** o + r.random() * 6.28) * (0.5 ** o)
        return v

    rays = 0.5 + 0.5 * (ang_noise(3, 0) * 0.5 + ang_noise(7, 0) * 0.35 + ang_noise(23, 0) * 0.25) / 1.1
    rays = np.clip(rays, 0, 1) ** 1.3
    rays_i = np.interp(th % (2 * math.pi), np.append(thx, 2 * math.pi), np.append(rays, rays[0]))
    rr = np.maximum(rho - 1, 0)
    cor = np.exp(-rr * (4.0 + 6.0 * (1 - rays_i))) * 1.25
    cor += 0.55 * np.exp(-rr * (1.2 + 1.8 * (1 - rays_i))) * rays_i ** 1.2
    cor += 0.30 * np.exp(-rr * 0.55)
    cor *= (rho >= 0.99)
    cc = ramp(np.clip(cor / 1.6, 0, 1), [(0, "#000000"), (0.12, "#25327A"), (0.3, "#6C86E0"), (0.55, "#CFE0FF"), (1, "#FFFFFF")])
    img += cc * np.clip(cor, 0, 1.4)[..., None] * 0.95
    # chromosphere
    ch = np.exp(-(((rho - 1.012) / 0.012) ** 2)) * 0.9
    img += ch[..., None] * col("#FF7FA8") * (0.3 + 0.7 * smooth(0.45, 0.8, rays_i))[..., None]
    # disc
    a = np.clip((rho - 1) * R + 0.5, 0, 1)
    img *= a[..., None] * 0.97 + 0.03
    # diamond ring bead
    ang = -0.8
    bx, by = cx + R * math.cos(ang), cy + R * math.sin(ang)
    cres = np.exp(-(((th - ang + math.pi) % (2 * math.pi) - math.pi) / 0.32) ** 2) * np.exp(-np.abs(rho - 1) * 70) * (rho > 0.99)
    img += cres[..., None] * col("#FFF6E8") * 2.0
    glow(img, bx, by, 26, col("#FFE9C8"), 1.6, "exp")
    glow(img, bx, by, 110, col("#9DB8FF"), 0.5, "exp")
    glow(img, bx, by, 6, col("#FFFFFF"), 2.5, "exp")
    return img, 1.2, 0.1


def g_galaxy(seed):
    img = vgrad([(0, "#010208"), (1, "#040616")])
    nebula(img, seed, [(0, "#000000"), (0.6, "#0E1236"), (1, "#2E2A6B")], 1.3, 0.5, 2.0, 0.28, 0.2)
    N = 1100000
    cx, cy, R = 720, 1240, 900
    x, y, t = spiral_points(seed + 1, N, R, 0.5, 0, 2, 4.0)
    rot = -0.9
    xr = x * math.cos(rot) - y * math.sin(rot)
    yr = (x * math.sin(rot) + y * math.cos(rot)) * 0.5
    px = np.clip((xr + cx).astype(int), 0, W - 1)
    py = np.clip((yr + cy).astype(int), 0, H - 1)
    pal = [(0, "#FFE7B0"), (0.18, "#FFD08A"), (0.4, "#BFD0FF"), (0.7, "#7EA2FF"), (1, "#4D62E0")]
    cols = ramp(t, pal) * np.clip(t * 5 + 0.12, 0, 1)[:, None]
    layer = np.zeros((H, W, 3), np.float32)
    idx = py * W + px
    for c in range(3):
        layer[..., c] = np.bincount(idx, weights=cols[:, c], minlength=H * W).reshape(H, W)
    S_ = blur(layer, 0.9) * 0.05 + blur(layer, 6) * 0.07 + blur(layer, 26) * 0.09
    lm = S_.max(-1, keepdims=True) + 1e-6
    img += S_ * ((1 - np.exp(-lm * 1.4)) / lm) * 1.1
    # faint disk
    rd = np.random.default_rng(seed + 4)
    dr_ = rd.exponential(R * 0.32, 500000)
    da = rd.random(500000) * 2 * math.pi
    dx_, dy_ = dr_ * np.cos(da), dr_ * np.sin(da)
    xr = dx_ * math.cos(rot) - dy_ * math.sin(rot)
    yr = (dx_ * math.sin(rot) + dy_ * math.cos(rot)) * 0.5
    idx = np.clip((yr + cy).astype(int), 0, H - 1) * W + np.clip((xr + cx).astype(int), 0, W - 1)
    dl = np.bincount(idx, minlength=H * W).reshape(H, W).astype(np.float32)
    img += blur(dl, 14)[..., None] * col("#7C93E0") * 0.08
    # bulge
    r = np.random.default_rng(seed + 5)
    bx = r.standard_normal(160000) * 70
    by = r.standard_normal(160000) * 52
    bxr = bx * math.cos(rot) - by * math.sin(rot) * 1.0
    byr = (bx * math.sin(rot) + by * math.cos(rot))
    idx = np.clip((byr + cy).astype(int), 0, H - 1) * W + np.clip((bxr + cx).astype(int), 0, W - 1)
    bl = np.bincount(idx, minlength=H * W).reshape(H, W).astype(np.float32)
    img += (1 - np.exp(-blur(bl, 5) * 0.012))[..., None] * col("#FFE2B0") * 0.8 + (1 - np.exp(-blur(bl, 24) * 0.04))[..., None] * col("#FFC27A") * 0.5
    glow(img, cx, cy, 90, col("#FFE9C4"), 0.35, "exp")
    # hii regions
    x2, y2, t2 = spiral_points(seed + 7, 700, R, 0.5, 0, 2, 4.6)
    xr = x2 * math.cos(rot) - y2 * math.sin(rot)
    yr = (x2 * math.sin(rot) + y2 * math.cos(rot)) * 0.5
    for a, b, tt in zip(xr, yr, t2):
        if tt > 0.25:
            glow(img, a + cx, b + cy, 6 + r.random() * 6, col("#FF6FA8"), 0.35, "exp", 3)
    # dust lanes
    d = fbm(seed + 9, 6, 5)
    dm = smooth(0.55, 0.8, d)[..., None] * np.clip(blur(layer, 20).mean(-1, keepdims=True) * 12, 0, 1)
    img *= 1 - 0.45 * dm
    stars(img, seed + 2, 2800, size=0.7, big=10)
    return img, 1.4, 0.0


def g_warp(seed):
    img = vgrad([(0, "#01030D"), (1, "#01030D")])
    cx, cy = 720, 1230
    rho = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2) / 1500
    img += ramp(np.clip(1 - rho, 0, 1) ** 1.6, [(0, "#000000"), (0.5, "#06194A"), (1, "#1D5FC9")]) * 0.9
    nb = fbm(seed, 2, 5)
    img += col("#2A6BFF") * (nb ** 2 * np.clip(1 - rho * 1.3, 0, 1) * 0.35)[..., None]
    r = np.random.default_rng(seed)
    layers = [rgb_canvas(2)]
    d = ImageDraw.Draw(layers[0])
    for i in range(1500):
        th = r.random() * 2 * math.pi
        r0 = math.exp(r.uniform(math.log(60), math.log(1650)))
        ln = r0 * (0.12 + 0.55 * r.random())
        wd = 1 + 5 * (r0 / 1650) ** 1.3
        base = 0.35 + 0.65 * r.random()
        tint = np.array([0.7, 0.85, 1.0]) if r.random() < 0.8 else np.array([1.0, 0.85, 0.7])
        segs = 6
        for s in range(segs):
            a0 = r0 + ln * s / segs
            a1 = r0 + ln * (s + 1) / segs
            f = ((s + 1) / segs) ** 1.5 * base
            c = tuple(int(255 * min(1, v * f * 1.4)) for v in tint)
            d.line([((cx + a0 * math.cos(th)) * 2, (cy + a0 * math.sin(th)) * 2), ((cx + a1 * math.cos(th)) * 2, (cy + a1 * math.sin(th)) * 2)],
                   fill=c, width=int(max(1, wd * (0.5 + 0.5 * (s + 1) / segs)) * 2))
    L = from_pil(layers[0])
    img += L * 1.1 + blur(L, 5) * 1.4 + blur(L, 22) * 1.0
    glow(img, cx, cy, 120, col("#7FB2FF"), 0.7, "exp")
    glow(img, cx, cy, 40, col("#FFFFFF"), 0.9, "exp")
    stars(img, seed + 2, 700, size=0.7)
    return img, 1.2, 0.2


def g_orbital_sunrise(seed):
    img = vgrad([(0, "#010207"), (0.6, "#030A1E"), (1, "#0A2548")])
    stars(img, seed + 1, 2600, ymax=1500, size=0.7, big=6, fade=lambda y: np.clip(1.25 - y / 1500, 0.1, 1))
    cx, cy, r = 720, 3500, 2250
    planet(img, cx, cy, r, [(0, "#07234F"), (0.3, "#0C3E78"), (0.46, "#14569A"), (0.497, "#2A86C0"), (0.503, "#CDBF90"), (0.54, "#5E9650"), (0.75, "#3B7440"), (0.9, "#8A8468"), (1, "#EEF4F8")],
           (0.0, -1.0, 0.25), seed + 3, cells=3.2, atmo="#6EC1FF", atmo_k=1.6, halo=1.4, ambient=0.02)
    # clouds
    n = fbm(seed + 4, 9, 7, 0.55, cy=9)
    top = cy - r
    dd = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2) / r
    cl = smooth(0.55, 0.68, n) * (dd < 1)[...] * np.clip((1 - dd) * 14, 0, 1)
    img += (cl * 0.5 * np.clip((YY - top) / 700, 0, 1) * np.clip(1.2 - dd * 0.4, 0, 1))[..., None] * col("#CFE7FF")
    # city lights on night side? sun rising behind the limb
    sx, sy = 720, top + 22
    glow(img, sx, sy, 900, col("#FFB36B"), 0.28, "exp")
    glow(img, sx, sy, 300, col("#FFD7A0"), 0.75, "exp")
    glow(img, sx, sy, 90, col("#FFF2D8"), 1.2, "exp")
    glow(img, sx, sy, 22, col("#FFFFFF"), 2.0, "exp")
    # limb flare streak
    streak = np.exp(-np.abs(YY - sy) / 5.0) * np.exp(-np.abs(XX - sx) / 600)
    img += streak[..., None] * col("#FFE0B0") * 0.9
    # satellite glints
    glow(img, 1120, 760, 12, col("#FFFFFF"), 0.8, "exp")
    return img, 1.2, 0.0


def g_gas_giant(seed):
    img = vgrad([(0, "#03040E"), (1, "#0A0720")])
    nebula(img, seed, [(0, "#000000"), (0.6, "#1A1043"), (1, "#5A2D7E")], 1.4, 0.5, 2.0, 0.45, 0.2)
    stars(img, seed + 1, 2400, size=0.7, big=8)
    pal = [(0, "#5A3A32"), (0.22, "#B98456"), (0.45, "#EBD7B4"), (0.62, "#C27A4E"), (0.82, "#E9C79A"), (1, "#7A4E3E")]
    planet(img, 330, 1320, 1180, pal, (0.85, -0.35, 0.5), seed + 3, bands=11.0, cells=3.5, atmo="#FFC48E", atmo_k=0.5, halo=0.6, bandwarp=0.32, tilt=0.18)
    # storm
    planet(img, 1210, 640, 70, [(0, "#4B4B55"), (0.5, "#A5A2A8"), (1, "#F0ECE8")], (0.85, -0.35, 0.5), seed + 8, cells=5, crater=0.5)
    planet(img, 1030, 2010, 34, [(0, "#8B4B22"), (0.5, "#D78B4B"), (1, "#FFD9A8")], (0.85, -0.35, 0.5), seed + 9, cells=4, crater=0.2)
    return img, 1.4, 0.15


# ====================================================================== landscapes
def sun_glow(img, x, y, r, c1, c2, k=1.0):
    glow(img, x, y, r * 6, col(c1), 0.35 * k, "exp")
    glow(img, x, y, r * 2.2, col(c1), 0.7 * k, "exp")
    glow(img, x, y, r * 0.8, col(c2), 1.0 * k, "exp")


def disc(img, cx, cy, r, c_top, c_bot=None, soft=1.2):
    d = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    a = 1 - smooth(r - soft, r + soft, d)
    if c_bot is None:
        c = col(c_top)
    else:
        c = ramp(np.clip((YY - (cy - r)) / (2 * r), 0, 1), [(0, c_top), (1, c_bot)])
    return comp(img, a, c), a


def g_alpine_dawn(seed):
    img = vgrad([(0, "#2B2E6B"), (0.25, "#7A5BA0"), (0.42, "#E78FA0"), (0.55, "#FFC48A"), (0.7, "#FFE2B8"), (1, "#FFE2B8")])
    stars(img, seed, 220, ymax=800, size=0.7, fade=lambda y: np.clip(1 - y / 800, 0, 1) * 0.8)
    sx, sy = 980, 1380
    glow(img, sx, sy, 700, col("#FFB07A"), 0.4, "exp")
    glow(img, sx, sy, 160, col("#FFF0CF"), 0.9, "exp")
    n = fbm(seed + 3, 3, 5, cy=14)
    img += (smooth(0.55, 0.85, n) * np.clip(1 - np.abs(YY - 900) / 700, 0, 1))[..., None] * col("#FFB89B") * 0.45
    layers = [
        (1360, 330, "#B58AA9", "#E9B4A4", 0.30, 1.8, 0.0, 0.7),
        (1560, 420, "#8C6A92", "#CF9AA0", 0.45, 2.1, 0.0, 0.7),
        (1760, 520, "#5E4F86", "#A97FA0", 0.6, 2.4, 0.0, 0.8),
        (1960, 560, "#3C3B6D", "#755A8C", 0.8, 2.7, 0.0, 0.9),
        (2250, 520, "#1D2250", "#3C3A6F", 1.0, 3.2, 0.0, 0.9),
    ]
    for i, (b, a, t, bt, hz, fq, sk, sh) in enumerate(layers):
        p = profile(seed + 10 + i, b, a, fq, 5, 0.5, sharp=0.8 if i < 4 else 0.5)
        ridge(img, p, t, bt, lit="#FFD9B0", shade="#2A2257", lit_k=0.8, lit_fade=170, tex=0.2, seed=seed + i, dirn=-1.0, slope_scale=1.1)
        fog(img, b + a * 0.6, 220, "#FFD0B4" if i < 3 else "#8C79B0", 0.22 + 0.04 * i, seed + 30 + i)
    return img, 1.4, 0.0


def g_misty_pines(seed):
    img = vgrad([(0, "#0F2B3A"), (0.3, "#2F6A73"), (0.52, "#9CC9BB"), (0.7, "#E9E6C9"), (1, "#E9E6C9")])
    sx, sy = 520, 1020
    glow(img, sx, sy, 600, col("#F9E9B5"), 0.45, "exp")
    disc(img, sx, sy, 130, "#FFF6D6", "#FFE9AE")
    glow(img, sx, sy, 150, col("#FFF3C8"), 0.9, "exp")
    cfg = [(1500, 360, "#7FA9A2", 60, (230, 330), 0.25), (1740, 400, "#5F8C88", 46, (300, 420), 0.35),
           (2020, 430, "#3F6C6C", 36, (380, 520), 0.5), (2340, 460, "#234B52", 24, (520, 720), 0.8), (2700, 480, "#10303A", 12, (780, 980), 1.0)]
    for i, (b, a, c, n, hr, dk) in enumerate(cfg):
        p = profile(seed + i, b, a * 0.4, 1.4, 5, 0.5)
        m_ground = ridge(img, p, c, c, tex=0.0)
        base_fn = lambda x, p=p: float(p[int(np.clip(x, 0, W - 1))])
        fm = forest_mask(seed + 40 + i, base_fn, n * 2, hr, 0.3)
        img *= 1 - fm[..., None] * 0
        comp(img, fm, col(c) * 0.92)
        fog(img, b + 40, 170 + 20 * i, "#E7EBD2", 0.4 - 0.05 * i, seed + 60 + i, 0.5)
    return img, 1.4, 0.0


def g_dune_sea(seed):
    img = vgrad([(0, "#29306B"), (0.18, "#8B5C97"), (0.32, "#F08F78"), (0.42, "#FFD08A"), (1, "#FFD08A")])
    sx, sy = 1080, 880
    disc(img, sx, sy, 90, "#FFF4D0", "#FFE0A0")
    glow(img, sx, sy, 340, col("#FFD08A"), 0.9, "exp")
    glow(img, sx, sy, 900, col("#FF9F6B"), 0.3, "exp")
    stars(img, seed, 160, ymax=420, size=0.7, fade=lambda y: np.clip(1 - y / 420, 0, 1) * 0.8)
    r = np.random.default_rng(seed)
    base = 1180
    for i in range(9):
        b = base + i * i * 11 + i * 105
        a = 50 + i * 26
        x = np.linspace(0, 1, W, dtype=np.float32)
        ph = r.random() * 6.28
        warp = noise1(r, 3, x * 3) * 0.5
        fq = 1.1 + 0.2 * i
        th_ = x * fq * 2 * math.pi + ph + warp * 5
        s0 = np.sin(th_ + 0.85 * np.sin(th_))
        prof = b - a * (0.5 + 0.5 * s0) - a * 0.12 * np.sin(x * 9.1 * math.pi + ph * 1.7 + warp * 3)
        t = i / 8
        lit = np.array([1.0, 0.72 - 0.15 * t, 0.43 - 0.1 * t])
        sh = np.array([0.52 - 0.12 * t, 0.26 - 0.06 * t, 0.36 - 0.1 * t])
        sl = np.convolve(np.gradient(prof), np.ones(7) / 7, mode='same')
        f = smooth(-0.12, 0.12, -sl)[None, :, None]
        depth = (np.clip((YY - prof[None, :]) / 520, 0, 1))[..., None]
        near = np.exp(-np.clip(YY - prof[None, :], 0, None) / 150.0)[..., None]
        mid = (lit + sh) * 0.5
        c = ((lit * f + sh * (1 - f)) * near + mid * (1 - near)) * (1 - 0.35 * depth)
        haze = 0.5 * (1 - t) ** 1.5
        c = c * (1 - haze) + col("#F4A67E") * haze
        pts = [(float(xx), float(yy)) for xx, yy in zip(range(0, W, 2), prof[::2])] + [(W, prof[-1]), (W, H + 3), (0, H + 3)]
        m = draw_mask(lambda d, ss: d.polygon([(a_ * ss, b_ * ss) for a_, b_ in pts], fill=255))
        # ripples on lit side
        rp = 0.5 + 0.5 * np.sin((YY - prof[None, :]) * 0.55 + noise(seed + i, 14, 40)[:, :W] * 14)
        c = c * (1 + (0.06 * rp * f[..., 0])[..., None])
        comp(img, m, c)
    fog(img, 1180, 140, "#FFB98E", 0.4, seed + 3, 0.5)
    return img, 1.6, 0.0


def g_ocean_sunset(seed):
    hy = 1380
    img = vgrad([(0, "#1B1B5A"), (0.25, "#6A3E8E"), (0.42, "#E0627F"), (0.5, "#FF9A5E"), (0.54, "#FFD27F"), (0.5400001, "#4A2F60"), (1, "#101030")])
    sx = 720
    n = fbm(seed + 1, 3, 5, cx=3, cy=16) if False else fbm(seed + 1, 2.5, 5, cy=14)
    cl = smooth(0.5, 0.9, n) * smooth(0, 700, hy - YY) * (YY < hy)
    cl = cl * np.clip(1 - (hy - YY) / 1200, 0, 1)
    img += cl[..., None] * ramp(np.clip((YY - 500) / 900, 0, 1)[..., None][..., 0], [(0, "#8A4B9C"), (0.5, "#FF7F7A"), (1, "#FFC88A")]) * 0.55
    glow(img, sx, hy, 650, col("#FF9A5E"), 0.55, "exp")
    # sun
    d = np.sqrt((XX - sx) ** 2 + (YY - (hy - 20)) ** 2)
    a = (1 - smooth(228, 232, d)) * (YY < hy)
    comp(img, a, ramp(np.clip((YY - (hy - 250)) / 250, 0, 1), [(0, "#FFF0C4"), (1, "#FFB35E")]))
    glow(img, sx, hy - 20, 250, col("#FFD27F"), 0.8, "exp")
    # water
    wm = YY >= hy
    z = np.maximum(YY - hy, 1.0)
    wnoise = fbm(seed + 2, 4, 4, cy=3)
    stripes = np.sin(2400 / np.sqrt(z + 4) * 3 + wnoise * 14 + (XX - sx) * 0.002 * 0) * 0.5 + 0.5
    water = ramp(np.clip((YY - hy) / (H - hy), 0, 1), [(0, "#FF9A6A"), (0.12, "#B2507B"), (0.45, "#3B2A66"), (1, "#0A0F30")])
    refl = np.exp(-((XX - sx) / (120 + z * 0.35)) ** 2) * np.exp(-z / 650)
    sparkle = smooth(0.62, 0.95, stripes * (0.6 + 0.8 * wnoise))
    water = water + col("#FFC77F") * (refl * (0.35 + 1.2 * sparkle))[..., None] + col("#FF8A8A") * (np.exp(-z / 120) * 0.25)[..., None]
    water = water * (1 + 0.08 * (stripes - 0.5))[..., None]
    img = np.where(wm[..., None], water, img)
    # horizon haze
    hz = np.exp(-(((YY - hy) / 14) ** 2))
    img += hz[..., None] * col("#FFD9A0") * 0.5
    return img, 1.4, 0.1


def aurora_curtain(img, seed, y0, amp, ray_hi, length, stops, k=1.0, x_scale=2.0):
    r = np.random.default_rng(seed)
    x = np.linspace(0, 1, W, dtype=np.float32)
    yb = y0 + amp * (noise1(r, x_scale, x * x_scale) * 2 - 1) + amp * 0.5 * (noise1(r, x_scale * 3, x * x_scale * 3) * 2 - 1)
    L = length * (0.6 + 0.8 * noise1(r, 4, x * 4))
    rays = 0.45 + 0.55 * noise1(r, ray_hi, x * ray_hi) ** 1.2
    rays2 = 0.6 + 0.4 * noise1(r, ray_hi * 2.7, x * ray_hi * 2.7)
    dy = yb[None, :] - YY
    prof = np.where(dy > 0, (1 - np.exp(-dy / 10.0)) * np.exp(-dy / L[None, :]), np.exp(dy / 22.0) * 0.0)
    inten = prof * (rays * rays2)[None, :]
    c = ramp(np.clip(dy / (L[None, :] * 2.2), 0, 1), stops)
    img += c * (inten * k)[..., None]
    return inten


def g_aurora_lake(seed):
    hy = 1560
    img = vgrad([(0, "#02060F"), (0.4, "#061427"), (0.6, "#0B2A3D"), (1, "#0B2A3D")])
    stars(img, seed, 1700, ymax=hy, size=0.7, big=4)
    stops = [(0, "#2BFFA0"), (0.45, "#1CD7B0"), (0.75, "#3A7CE0"), (1, "#9B4DE0")]
    layer = np.zeros_like(img)
    for i, (y0, amp, L, k, xs) in enumerate([(1000, 160, 520, 1.0, 2.0), (780, 220, 460, 0.8, 1.6), (1250, 110, 360, 0.7, 2.6)]):
        aurora_curtain(layer, seed + i, y0, amp, 60 + 20 * i, L, stops if i < 2 else [(0, "#7BFFC4"), (1, "#3AE0B8")], k, xs)
    img += blur(layer, 1.5) * 1.3 + blur(layer, 30) * 0.9
    # mountains
    for i, (b, a, c, bt) in enumerate([(hy - 60, 360, "#0E2A3C", "#0B1F30"), (hy - 5, 250, "#07141F", "#050C14")]):
        p = profile(seed + 10 + i, b, a, 2.0 + i, 7, 0.52, sharp=0.8)
        p = np.minimum(p, hy + 2)
        m = ridge(img, p, c, bt, lit="#2E6C84", shade=c, lit_k=0.5, lit_fade=120, dirn=-1.0, slope_scale=0.9)
    # forest line
    base_fn = lambda x: hy + 4
    fm = forest_mask(seed + 20, base_fn, 90, (60, 170), 0.3)
    comp(img, fm, col("#030A10"))
    # lake: mirror
    idx = np.clip(2 * hy - np.arange(H), 0, H - 1)
    refl = img[idx][hy:]
    wob = fbm(seed + 3, 5, 4, cy=70)[hy:H]
    xs = np.clip(np.arange(W)[None, :] + ((wob - 0.5) * 36).astype(int), 0, W - 1)
    refl = np.take_along_axis(refl, xs[..., None].repeat(3, 2), 1)
    refl = blur(refl, 1.4) * np.clip(0.55 - (np.arange(H - hy)[:, None, None] / (H - hy)) * 0.2, 0, 1)
    refl = refl * col("#B8D8E4")
    sub = img[hy:]
    water = np.clip(refl + sub * 0.0, 0, None)
    img[hy:] = water
    ripple = (np.sin(YY[hy:] * 0.9 + wob * 18) * 0.5 + 0.5)
    img[hy:] *= (0.92 + 0.12 * ripple)[..., None]
    img[hy:] = img[hy:] * (1 - 0.6 * np.clip((YY[hy:] - hy) / (H - hy), 0, 1) ** 1.5)[..., None]
    # near shore rocks
    sp = profile(seed + 30, H - 120, 160, 3, 6, 0.5, sharp=0.5)
    ridge(img, sp, "#050B10", "#03070B")
    img[hy - 3:hy + 3] += 0.04
    return img, 1.2, 0.15


def g_rolling_hills(seed):
    img = vgrad([(0, "#4C9BEA"), (0.3, "#8CC8F2"), (0.55, "#DFF1F4"), (1, "#F4F6DE")])
    n = fbm(seed, 2.2, 6, cy=9)
    cl = smooth(0.52, 0.8, n) * np.clip(1 - YY / 1500, 0, 1)
    sh = np.clip(smooth(0.52, 0.8, warp_sample(n, n, n, 40)) - cl * 0.5, 0, 1)
    img = img * (1 - cl[..., None] * 0.85) + col("#FFFFFF") * (cl * 0.85)[..., None]
    img -= (cl * smooth(0.7, 1.0, n) * 0.12)[..., None] * col("#7C9BB8")
    sx, sy = 1060, 560
    glow(img, sx, sy, 520, col("#FFF4C8"), 0.35, "exp")
    glow(img, sx, sy, 90, col("#FFFFFF"), 1.1, "exp")
    cfg = [(1500, 140, "#9ACB86", "#7FB26F"), (1700, 190, "#7DBB63", "#5EA04F"), (1950, 230, "#5FA84C", "#3F8C3F"), (2200, 250, "#44963F", "#2E7432"), (2450, 230, "#2F7D35", "#1F5E2A")]
    for i, (b, a, c, cb) in enumerate(cfg):
        p = profile(seed + 10 + i, b, a, 1.2 + 0.2 * i, 4, 0.45, skew=(-60 if i % 2 else 60))
        ridge(img, p, c, cb, lit="#E9F5A0", shade=cb, lit_k=0.5, lit_fade=240 + 40 * i, dirn=-1.0, slope_scale=0.35, tex=0.18, seed=seed + i)
        st = noise(seed + 50 + i, 8, 160)[:H, :W]
        if i > 1:
            pass
        fog(img, b + 60, 190, "#E8F4E8", 0.28 - 0.04 * i, seed + 70 + i, 0.4)
    # flowers
    r = np.random.default_rng(seed + 8)
    fl = np.zeros((H, W, 3), np.float32)
    for c, cnt in (("#FFF3A6", 380), ("#FFFFFF", 260), ("#FF9EC4", 160)):
        pts = np.stack([r.random(cnt) * W, 2000 + (r.random(cnt) ** 0.7) * 540], 1)
        im = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(im)
        for x, y in pts:
            s = 2.0 + 5 * ((y - 2000) / 540)
            d.ellipse([x - s, y - s * 0.7, x + s, y + s * 0.7], fill=230)
        fl += (np.asarray(im, np.float32) / 255)[..., None] * col(c)
    mask = np.clip(fl.sum(-1, keepdims=True), 0, 1)
    img = img * (1 - mask * 0.9) + fl * 0.9
    return img, 1.4, 0.0


def g_lowpoly(seed, pal_top, sky, rows=34, cols=16, sun=None):
    img = vgrad(sky)
    if sun:
        glow(img, sun[0], sun[1], 520, col(sun[2]), 0.55, "exp")
        disc(img, sun[0], sun[1], 120, "#FFFFFF", sun[2])
    r = np.random.default_rng(seed)
    hm = fbm(seed + 1, 3.0, 5, 0.52, cy=6)
    ss = 2
    im = Image.new("RGB", (W * ss, H * ss), (0, 0, 0))
    d = ImageDraw.Draw(im)
    hy = 1150
    imgs0 = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((W * ss, H * ss))
    im = imgs0
    d = ImageDraw.Draw(im)
    verts = {}
    for rr in range(rows + 2):
        t = rr / rows
        y0 = hy + (H + 220 - hy) * t ** 1.75
        spread = W * (0.55 + 1.35 * t)
        amp = 60 + 640 * t ** 1.25
        for c in range(cols + 3):
            u = (c - 1) / cols
            x = W / 2 + (u - 0.5) * spread * 1.4 + (r.random() - 0.5) * spread / cols * 0.7
            yj = (r.random() - 0.5) * (H - hy) / rows * 0.45 * (0.3 + t)
            hv = hm[min(H - 1, int(np.clip(rr * 15 + 3, 0, H - 1))), min(W - 1, int(np.clip(u * W, 0, W - 1)))]
            hh = (hm[int(np.clip(rr / rows * H * 0.75, 0, H - 1)), int(np.clip(c * W / cols, 0, W - 1))] - 0.35) * amp * 1.6
            verts[(rr, c)] = (x, y0 + yj - hh, hh / (amp + 1))
    L = np.array([-0.6, -0.55, 0.58])
    L /= np.linalg.norm(L)
    for rr in range(rows + 1):
        t = rr / rows
        for c in range(cols + 2):
            a, b, cc, dd = verts[(rr, c)], verts[(rr, c + 1)], verts[(rr + 1, c)], verts[(rr + 1, c + 1)]
            for tri in ((a, b, cc), (b, dd, cc)):
                p0, p1, p2 = [np.array([v[0], v[1], v[2] * 280]) for v in tri]
                nrm = np.cross(p1 - p0, p2 - p0)
                nrm = nrm / (np.linalg.norm(nrm) + 1e-6)
                if nrm[2] < 0:
                    nrm = -nrm
                lam = float(np.clip(np.dot(nrm, L), 0, 1))
                hgt = np.mean([v[2] for v in tri])
                tt = float(np.clip(0.5 + hgt * 1.4, 0, 1))
                base = ramp(np.array([tt]), pal_top)[0]
                shade = 0.35 + 0.95 * lam ** 0.9
                cl = base * shade
                # distance haze
                hz = float(np.clip(1 - t * 1.3, 0, 1)) ** 1.6
                cl = cl * (1 - hz * 0.4) + col(sky[3][1]) * hz * 0.4
                fill = tuple(int(np.clip(v, 0, 1) * 255) for v in cl)
                pts = [(v[0] * ss, v[1] * ss) for v in tri]
                d.polygon(pts, fill=fill, outline=fill)
    img = from_pil(im)
    fog(img, hy + 140, 130, sky[len(sky) // 2][1], 0.35, seed + 4, 0.5)
    return img, 1.4, 0.0


def g_canyon(seed):
    img = vgrad([(0, "#3A2B66"), (0.2, "#B2567A"), (0.34, "#FF9A66"), (0.46, "#FFD39A"), (1, "#FFD39A")])
    sx, sy = 560, 1010
    glow(img, sx, sy, 700, col("#FFB06A"), 0.5, "exp")
    disc(img, sx, sy, 110, "#FFF3D2", "#FFD69A")
    glow(img, sx, sy, 140, col("#FFE6B5"), 1.0, "exp")
    # sun shafts
    ang = np.arctan2(YY - sy, XX - sx)
    shaft = (0.5 + 0.5 * np.sin(ang * 22 + 1.0)) ** 3 * np.exp(-np.sqrt((XX - sx) ** 2 + (YY - sy) ** 2) / 1100)
    img += shaft[..., None] * col("#FFD39A") * 0.18
    r = np.random.default_rng(seed)
    cfg = [(1280, 230, "#C47A74", "#E39A82"), (1500, 300, "#A9586A", "#CF7C7C"), (1760, 360, "#8A4160", "#B05F6F"),
           (2050, 420, "#5E2D58", "#8B4260"), (2380, 440, "#33183F", "#5A2A55")]
    for i, (b, a, c, cb) in enumerate(cfg):
        p = profile(seed + 10 + i, b, a, 0.9 + 0.35 * i, 3, 0.5)
        # mesa quantisation
        levels = np.round((p - p.min()) / (a / 3.0)) * (a / 3.0) + p.min()
        k = int(W / 60)
        q = np.convolve(np.pad(levels, 12, mode='edge'), np.ones(25) / 25, mode='valid')[:W]
        p2 = np.minimum(p, q) if False else q
        ridge(img, p2, c, cb, lit="#FFD9A8", shade=cb, lit_k=0.5, lit_fade=260, tex=0.3, seed=seed + i, dirn=-1.0, slope_scale=0.3)
        # strata
        strata = 0.5 + 0.5 * np.sin(YY * (0.07 + 0.01 * i) + noise(seed + i, 6, 30)[:H, :W] * 9)
        pr = np.interp(np.arange(W), np.linspace(0, W - 1, len(p2)), p2)[None, :]
        inside = (YY > pr) * np.exp(-np.clip(YY - pr, 0, None) / 700)
        img *= 1 - ((strata - 0.5) * 0.12 * inside * (YY > pr))[..., None] * 0.0 + (1 - (strata - 0.5) * 0.1 * inside)[..., None] * 0.0 + 1.0 * 0
        img *= (1 + (strata - 0.5) * 0.14 * inside)[..., None]
        fog(img, b + 30, 200, "#FFC9A0" if i < 3 else "#7A3F6A", 0.25, seed + 50 + i)
    bokeh(img, seed + 7, 26, ["#FFD9A0", "#FFB67A"], 10, 40, (0.05, 0.16), 0.3, (0, 700, W, 2400))
    return img, 1.5, 0.0


# ====================================================================== retro
def g_synth_sun(seed):
    hy = 1470
    img = vgrad([(0, "#0A0420"), (0.25, "#2A0B52"), (0.45, "#7A1A78"), (0.58, "#FF3E8E"), (0.6, "#FF8A4A"), (0.6000001, "#14052E"), (1, "#090218")])
    stars(img, seed, 700, ymax=1000, size=0.7, big=4, fade=lambda y: np.clip(1.1 - y / 1000, 0, 1))
    sx, sy, R = 720, hy - 260, 330
    glow(img, sx, sy, 760, col("#FF4D9A"), 0.4, "exp")
    d = np.sqrt((XX - sx) ** 2 + (YY - sy) ** 2)
    a = 1 - smooth(R - 1, R + 1, d)
    bands = np.ones_like(YY)
    for i in range(9):
        yb = sy + 40 + i * 28
        wd = 4 + i * 4.5
        bands *= 1 - smooth(yb - 0.5, yb + 0.5, YY) * (1 - smooth(yb + wd - 0.5, yb + wd + 0.5, YY))
    a *= bands * (YY < hy)
    comp(img, a, ramp(np.clip((YY - (sy - R)) / (2 * R), 0, 1), [(0, "#FFF176"), (0.45, "#FFB03A"), (0.72, "#FF3E8E"), (1, "#C21FA5")]))
    glow(img, sx, sy, 200, col("#FFB03A"), 0.45, "exp")
    # distant mountains wire
    p = profile(seed + 1, hy, 160, 3.2, 5, 0.5, sharp=0.9)
    p = np.minimum(p, hy)
    m = ridge(img, p, "#1A0636", "#1A0636")
    edge = np.clip(blur(m, 1.2) - m * 0.0, 0, 1)
    ed = np.abs(np.gradient(blur(m, 1.5), axis=0))
    img += (ed * 2.2)[..., None] * col("#FF4FD8")
    # floor grid
    vx = W / 2

    def grid(d, ss):
        for i in range(-34, 35):
            x_b = vx + i * 260
            d.line([(vx * ss, hy * ss), (x_b * ss, (H + 40) * ss)], fill=255, width=3 * ss)
        z = 0.0
        for k in range(1, 26):
            t = k / 25.0
            y = hy + (H - hy) * (t ** 2.35)
            d.line([(0, y * ss), (W * ss, y * ss)], fill=255, width=int((2 + 4 * t) * ss))

    gm = draw_mask(grid)
    gm *= smooth(hy, hy + 80, YY)
    fadeg = np.clip((YY - hy) / 260, 0, 1) ** 0.6
    img += gm[..., None] * col("#FF5CE1") * fadeg[..., None] * 1.2
    img += blur(gm, 6)[..., None] * col("#FF2DB5") * 0.9 * fadeg[..., None]
    img += blur(gm, 22)[..., None] * col("#8A2DFF") * 1.2 * fadeg[..., None]
    glow(img, vx, hy, 520, col("#FF4D9A"), 0.35, "exp")
    hz = np.exp(-(((YY - hy) / 8) ** 2))
    img += hz[..., None] * col("#FFC0F0") * 0.8
    return img, 1.4, 0.0


def g_grid_tunnel(seed):
    img = vgrad([(0, "#06031A"), (1, "#06031A")])
    cx, cy = 720, 1180
    rho = np.sqrt(((XX - cx) / 1.0) ** 2 + ((YY - cy) / 1.0) ** 2)
    img += ramp(np.clip(1 - rho / 1500, 0, 1) ** 2, [(0, "#000000"), (0.5, "#240A4F"), (1, "#6E1CA8")]) * 0.8

    def tun(d, ss):
        for k in range(1, 40):
            t = k / 40.0
            s = (t ** 2.2)
            hw, hh = 40 + 1100 * s, 62 + 1700 * s
            d.rectangle([(cx - hw) * ss, (cy - hh) * ss, (cx + hw) * ss, (cy + hh) * ss], outline=255, width=int((1 + 5 * s) * ss))
        for ang in range(0, 360, 15):
            a = math.radians(ang)
            dx, dy = math.cos(a), math.sin(a)
            sx = 1 / max(abs(dx) / 1250, abs(dy) / 1950, 1e-6)
            d.line([(cx + dx * 20) * ss, (cy + dy * 30) * ss, (cx + dx * sx) * ss, (cy + dy * sx) * ss], fill=255, width=2 * ss)

    m = draw_mask(tun)
    t = np.clip(1 - np.sqrt(((XX - cx) / 1250) ** 2 + ((YY - cy) / 1950) ** 2), 0, 1)
    c = ramp(t, [(0, "#FF2DB5"), (0.5, "#8F5CFF"), (1, "#27E6FF")])
    neon(img, m, "#B04DFF", 0.35, ((2.0, 1.0), (9.0, 0.9), (35.0, 0.7)), 0.9)
    img += (m[..., None] * c * 0.8) + blur(m, 8)[..., None] * c * 0.8
    glow(img, cx, cy, 230, col("#7CF3FF"), 0.9, "exp")
    glow(img, cx, cy, 60, col("#FFFFFF"), 1.3, "exp")
    stars(img, seed, 300, size=0.7, amp=(0.2, 0.6))
    return img, 1.4, 0.2


def g_phosphor(seed):
    img = vgrad([(0, "#010A04"), (1, "#010A04")])
    cx, cy = 720, 1280
    rho = np.sqrt(((XX - cx) / 760) ** 2 + ((YY - cy) / 1300) ** 2)
    img += col("#0C3A1C") * (np.clip(1 - rho, 0, 1) ** 1.5 * 0.9)[..., None]

    def f(d, ss):
        for x in range(0, W + 1, 120):
            d.line([(x * ss, 0), (x * ss, H * ss)], fill=70, width=ss)
        for y in range(0, H + 1, 120):
            d.line([(0, y * ss), (W * ss, y * ss)], fill=70, width=ss)
        for yy in range(0, H, 24):
            d.line([(W / 2 * ss - 18 * ss, yy * ss), (W / 2 * ss + 18 * ss, yy * ss)], fill=140, width=ss)
        for xx in range(0, W, 24):
            d.line([(xx * ss, H / 2 * ss - 18 * ss), (xx * ss, H / 2 * ss + 18 * ss)], fill=140, width=ss)

    g = draw_mask(f)
    neon(img, g * 0.55, "#1CFF6A", 0.3, ((1.5, 0.5), (8.0, 0.2)), 0.6)
    xs = np.arange(0, W, 2)
    r = np.random.default_rng(seed)
    waves = []
    for i, (y0, a, f1, f2, wd) in enumerate([(780, 190, 5.2, 13.0, 5), (1320, 300, 3.1, 9.0, 6), (1850, 160, 7.4, 21.0, 4)]):
        n1 = noise1(r, 6, xs / W * 6)
        y = y0 + a * (np.sin(xs / W * f1 * math.pi * 2 + r.random() * 6) * (0.6 + 0.4 * np.sin(xs / W * 2.3 * math.pi + i)) + 0.35 * np.sin(xs / W * f2 * math.pi * 2 + 1 + i) * n1)
        waves.append((y, wd))

    def wf(d, ss):
        for (y, wd) in waves:
            d.line([(float(x) * ss, float(yy) * ss) for x, yy in zip(xs, y)], fill=255, width=wd * ss, joint="curve")

    wm = draw_mask(wf)
    neon(img, wm, "#2DFF7A", 0.5, ((2.5, 1.0), (10.0, 0.9), (40.0, 0.6)), 1.0)
    # blips
    for _ in range(36):
        x, y = r.random() * W, r.random() * H
        glow(img, x, y, 7, col("#4DFF8C"), 0.5, "exp")
    # scanlines & curvature
    scan = 0.82 + 0.18 * np.sin(YY * math.pi * 2 / 5.0)
    img *= scan[..., None]
    rr = np.sqrt(((XX / W - 0.5) * 2) ** 2 + ((YY / H - 0.5) * 2) ** 2)
    img *= (1 - 0.55 * np.clip(rr - 0.45, 0, 1) ** 1.6)[..., None]
    img[..., 0] *= 0.9
    return img, 1.6, 0.2


def g_pixel_sunset(seed):
    w, h = 180, 320
    r = np.random.default_rng(seed)
    bay = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0
    yy, xx = np.mgrid[0:h, 0:w]
    stops = [(0.0, "#1A1043"), (0.22, "#4A1B73"), (0.4, "#B02D7B"), (0.52, "#F0546A"), (0.62, "#FF9A4A"), (0.7, "#FFD36E")]
    pal = [col(s[1]) for s in stops]
    pos = np.array([s[0] for s in stops])
    t = np.clip(yy / 222.0, 0, 0.7)
    # nearest two palette entries, ordered dither
    k = np.searchsorted(pos, t, side="right") - 1
    k = np.clip(k, 0, len(stops) - 2)
    f = (t - pos[k]) / (pos[k + 1] - pos[k])
    use = (f > bay[yy % 4, xx % 4]).astype(int)
    px = np.array(pal, np.float32)[k + use].copy()

    def put(mask, c):
        px[mask] = col(c) if isinstance(c, str) else c

    # stars
    for _ in range(46):
        x, y = int(r.integers(0, w)), int(r.integers(0, 100))
        put(np.array([[yy_ == y and xx_ == x for xx_ in range(w)] for yy_ in range(h)]) if False else (yy == y) & (xx == x), "#FFE9F6")
    # sun
    sx, sy, sr = 104, 190, 46
    sun = (xx - sx) ** 2 + (yy - sy) ** 2 < sr * sr
    put(sun, "#FFE27A")
    put(sun & (yy > sy - 8), "#FFC14A")
    put(sun & (yy > sy + 12), "#FF8F4F")
    for i, yb in enumerate(range(sy + 6, sy + sr, 7)):
        put(sun & (yy >= yb) & (yy < yb + 1 + i // 2), px[0, 0] * 0 + col("#B02D7B"))
    # clouds
    for cy_, cx_, wd in [(70, 40, 40), (96, 140, 34), (128, 20, 28), (52, 120, 26)]:
        put((np.abs(yy - cy_) < 2) & (np.abs(xx - cx_) < wd // 2), "#FF8FB0")
        put((np.abs(yy - cy_ - 2) < 1) & (np.abs(xx - cx_ - 4) < wd // 3), "#C74B93")
    # mountains
    def mt(base, amp, freq, col_a, col_b, ph):
        x = np.arange(w)
        pr = base - amp * (np.abs(np.sin(x * freq + ph)) * 0.6 + 0.4 * np.abs(np.sin(x * freq * 2.3 + ph * 2))) - 0
        pr = np.round(pr).astype(int)
        for xi in range(w):
            put((xx == xi) & (yy >= pr[xi]), col_a)
            put((xx == xi) & (yy >= pr[xi]) & (yy < pr[xi] + 2), col_b)
    mt(238, 46, 0.045, "#5B1E7D", "#A6358C", 0.6)
    mt(258, 38, 0.06, "#3A1658", "#6F2A83", 2.1)
    # pines
    def pinepx(x0, yb, hh, c1, c2):
        for j in range(hh):
            half = 1 + j // 3
            row = yb - hh + j
            put((yy == row) & (np.abs(xx - x0) <= half), c1)
            put((yy == row) & (xx == x0 - half) & (j % 3 == 0), c2) if False else None
        put((yy >= yb) & (yy < yb + 3) & (xx == x0), "#2B1030")
    base = 292
    for x0 in range(-2, w + 6, 11):
        hh = int(14 + r.integers(0, 12))
        pinepx(x0 + int(r.integers(-2, 3)), base + int(r.integers(-4, 4)), hh, "#1A0B2C", "#2B1030")
    put(yy >= base + 2, "#120620")
    for x0 in range(5, w, 17):
        hh = int(22 + r.integers(0, 16))
        pinepx(x0 + int(r.integers(-3, 3)), 312, hh, "#0C0418", "#0C0418")
    put(yy >= 313, "#0C0418")
    arr = (np.clip(px, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(arr, "RGB").resize((W, H), Image.NEAREST)
    out = np.asarray(im, np.float32) / 255
    scan = 1 - 0.07 * ((YY % 8) < 1.5)
    return out * scan[..., None], 0.7, 0.2


def g_vaporwave(seed):
    hy = 1380
    img = vgrad([(0, "#2B1B6E"), (0.3, "#8A4BC8"), (0.5, "#FF7AC6"), (0.54, "#FFC6E6"), (0.5400001, "#1D1050"), (1, "#0E0838")])
    stars(img, seed, 500, ymax=900, size=0.7, tints=("#FFFFFF", "#FFD6F0", "#C6F0FF"), fade=lambda y: np.clip(1.2 - y / 900, 0, 1))
    sx, sy, R = 720, hy - 340, 360
    glow(img, sx, sy, 700, col("#FF8AD8"), 0.4, "exp")
    d = np.sqrt((XX - sx) ** 2 + (YY - sy) ** 2)
    a = (1 - smooth(R - 1, R + 1, d)) * (YY < hy)
    bands = np.ones_like(YY)
    for i in range(7):
        yb = sy + 90 + i * 38
        bands *= 1 - smooth(yb - .5, yb + .5, YY) * (1 - smooth(yb + 6 + i * 3 - .5, yb + 6 + i * 3 + .5, YY))
    comp(img, a * bands, ramp(np.clip((YY - (sy - R)) / (2 * R), 0, 1), [(0, "#FFF3A8"), (0.5, "#FF9ED6"), (1, "#9B6BFF")]))
    # pyramids / wire shapes
    def shapes(dr, ss):
        for (x, y, s) in [(260, hy - 20, 180), (1190, hy - 10, 140)]:
            dr.polygon([(x * ss, (y - s) * ss), ((x - s) * ss, y * ss), ((x + s) * ss, y * ss)], outline=255, width=4 * ss)
            dr.line([(x * ss, (y - s) * ss), (x * ss, y * ss)], fill=255, width=3 * ss)
        dr.ellipse([(1100 - 70) * ss, (700 - 70) * ss, (1100 + 70) * ss, (700 + 70) * ss], outline=255, width=4 * ss)
        dr.ellipse([(1100 - 70) * ss, (700 - 22) * ss, (1100 + 70) * ss, (700 + 22) * ss], outline=255, width=3 * ss)
        dr.polygon([(280 * ss, 520 * ss), (340 * ss, 600 * ss), (280 * ss, 680 * ss), (220 * ss, 600 * ss)], outline=255, width=4 * ss)

    sm = draw_mask(shapes)
    neon(img, sm, "#8FF3FF", 0.45, ((2.0, 0.8), (10.0, 0.6)), 0.9)
    # checker floor
    dy_ = np.maximum(YY - hy, 1.0)
    X = (XX - W / 2) / dy_ * 7.9 * 1.18
    Z = 12650.0 / dy_
    chk = (np.floor(X) + np.floor(Z)) % 2
    # anti-aliased fade toward the horizon where checks get sub-pixel
    sub = smooth(0, 1, np.clip((dy_ - 12) / 140, 0, 1))
    A = col("#FF6FC8") * 0.62
    B = col("#17B8F2") * 0.55
    horizon = col("#C58AE8") * 0.7
    floor = np.where(chk[..., None] > 0.5, A, B)
    floor = floor * sub[..., None] + horizon * (1 - sub[..., None])
    floor *= (0.45 + 0.55 * np.clip((YY - hy) / 700, 0, 1) ** 0.6)[..., None]
    img = np.where((YY > hy)[..., None], floor, img)
    img += (np.exp(-np.abs(YY - hy) / 6) * 0.6)[..., None] * col("#FFDCF4")
    return img, 1.3, 0.1


def g_seventies(seed):
    img = vgrad([(0, "#FFEBC9"), (0.55, "#FFD3A0"), (1, "#F9B58A")])
    cx, cy = 720, 2000
    cols = ["#FFC94A", "#F9A03A", "#EE6E2E", "#D6403F", "#A82E5C", "#6F2A72", "#43276F", "#2A1B5E", "#1B1248"]
    d = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    bw = 132
    for i, c in enumerate(cols):
        R = 1500 - i * bw
        a = 1 - smooth(R - 1.2, R + 1.2, d)
        comp(img, a, col(c))
    a = 1 - smooth(168, 171, d)
    comp(img, a, col("#FFF1D0"))
    glow(img, cx, cy, 420, col("#FFE9B0"), 0.25, "exp")
    # dots and soft sun rays on the sky
    img *= (0.94 + 0.06 * (1 - np.clip(d / 1700, 0, 1)))[..., None]
    return img, 1.8, 0.0


# ====================================================================== abstract
def g_mesh(seed, pts, warp=0.35):
    r = np.random.default_rng(seed)
    wx, wy = fbm(seed + 1, 1.3, 3), fbm(seed + 2, 1.3, 3)
    X = XX + (wx - 0.5) * W * warp
    Y = YY + (wy - 0.5) * W * warp
    acc = np.zeros((H, W, 3), np.float32)
    ws = np.zeros((H, W), np.float32)
    for x, y, c, s in pts:
        d2 = ((X - x * W) ** 2 + (Y - y * H) ** 2) / (W * W)
        wgt = 1.0 / (d2 / (s * s) + 0.015) ** 1.6
        acc += wgt[..., None] * col(c)
        ws += wgt
    img = acc / ws[..., None]
    # soft light bloom
    img *= 1 + 0.12 * (fbm(seed + 3, 2.2, 4)[..., None] - 0.5)
    return img, 1.2, 0.0


def g_silk(seed, stops, light=(-0.5, -0.6, 0.6), cx=1.1, amp=1.0, hue2=None):
    n = fbm(seed, cx, 4, 0.5)
    w1, w2 = fbm(seed + 1, cx, 3), fbm(seed + 2, cx, 3)
    n = warp_sample(n, w1, w2, W * 0.55)
    n = warp_sample(n, w2, w1, W * 0.2)
    n = blur(n, 16)
    hgt = n * 220 * amp
    gy, gx = np.gradient(hgt)
    nz = 1.0
    nl = np.sqrt(gx * gx + gy * gy + nz * nz)
    L = np.array(light, np.float32)
    L /= np.linalg.norm(L)
    lam = np.clip((-gx * L[0] - gy * L[1] + nz * L[2]) / nl, 0, 1)
    base = ramp(n, stops)
    img = base * (0.35 + 0.9 * lam ** 1.4)[..., None]
    hv = np.array([L[0], L[1], L[2] + 1], np.float32)
    hv /= np.linalg.norm(hv)
    sp = np.clip((-gx * hv[0] - gy * hv[1] + nz * hv[2]) / nl, 0, 1) ** 40
    img += sp[..., None] * 0.8 * (0.5 + 0.5 * base)
    return img, 1.2, 0.1


def g_bauhaus(seed):
    r = np.random.default_rng(seed)
    cream, red, blue, yel, blk = "#EDE3CD", "#D8381F", "#1F4E9C", "#F0B232", "#1B1B1B"
    palette = [red, blue, yel, blk, cream]
    ss = 2
    im = Image.new("RGB", (W * ss, H * ss), cream)
    d = ImageDraw.Draw(im)
    cols_, rows_ = 3, 5
    tw, th = W / cols_, H / rows_
    prev = [None] * cols_
    for ri in range(rows_):
        for ci in range(cols_):
            x0, y0 = ci * tw, ri * th
            bgc = palette[int(r.integers(0, 5))]
            while bgc == prev[ci]:
                bgc = palette[int(r.integers(0, 5))]
            prev[ci] = bgc
            fg = [c for c in palette if c != bgc]
            fc = fg[int(r.integers(0, len(fg)))]
            fc2 = [c for c in palette if c not in (bgc, fc)][int(r.integers(0, 3))]
            d.rectangle([x0 * ss, y0 * ss, (x0 + tw) * ss, (y0 + th) * ss], fill=bgc)
            s = int(r.integers(0, 8))
            cx, cy = x0 + tw / 2, y0 + th / 2
            R = tw / 2
            bx = lambda a, b, c_, d_: [a * ss, b * ss, c_ * ss, d_ * ss]
            if s == 0:
                d.ellipse(bx(cx - R * .8, cy - R * .8, cx + R * .8, cy + R * .8), fill=fc)
                d.ellipse(bx(cx - R * .4, cy - R * .4, cx + R * .4, cy + R * .4), fill=fc2)
            elif s == 1:
                d.pieslice(bx(x0 - tw, y0 - tw, x0 + tw, y0 + tw), 0, 90, fill=fc)
                d.pieslice(bx(x0 + tw - tw, y0 + th - tw, x0 + tw + tw, y0 + th + tw), 180, 270, fill=fc2)
            elif s == 2:
                d.pieslice(bx(cx - R, y0 + th - R * 1.0 - R, cx + R, y0 + th - R * 1.0 + R), 180, 360, fill=fc)
                d.rectangle(bx(x0, y0, x0 + tw, y0 + th * 0.18), fill=fc2)
            elif s == 3:
                n = 6
                for k in range(n):
                    d.rectangle(bx(x0, y0 + k * th / n, x0 + tw, y0 + k * th / n + th / n / 2), fill=fc)
                d.ellipse(bx(cx - R * .55, cy - R * .55, cx + R * .55, cy + R * .55), fill=fc2)
            elif s == 4:
                d.polygon([(x0 * ss, (y0 + th) * ss), ((x0 + tw) * ss, (y0 + th) * ss), (x0 * ss, y0 * ss)], fill=fc)
                d.ellipse(bx(x0 + tw * .55, y0 + th * .12, x0 + tw * .9, y0 + th * .12 + tw * .35), fill=fc2)
            elif s == 5:
                for k in range(4):
                    rr = R * (1 - k * 0.25)
                    d.ellipse(bx(cx - rr, cy - rr, cx + rr, cy + rr), fill=(fc, fc2)[k % 2])
            elif s == 6:
                d.pieslice(bx(x0 - R, y0 - R, x0 + R, y0 + R) if False else bx(x0, y0, x0 + tw * 2, y0 + tw * 2), 180, 270, fill=fc)
                d.rectangle(bx(x0 + tw * .55, y0 + th * .55, x0 + tw * .9, y0 + th * .9), fill=fc2)
            else:
                d.rectangle(bx(x0 + tw * .15, y0 + th * .15, x0 + tw * .85, y0 + th * .85), fill=fc)
                d.ellipse(bx(cx - R * .5, cy - R * .5, cx + R * .5, cy + R * .5), fill=bgc)
    # keep tile clipping: re-paint outside-of-tile overflow by overdrawing neighbours is complex; we overpaint in order and accept overlap
    img = from_pil(im)
    n = fbm(seed + 3, 40, 3)
    img *= (0.94 + 0.12 * n)[..., None]
    fib = fbm(seed + 4, 220, 2, cy=280)
    img *= (0.97 + 0.06 * fib)[..., None]
    return img, 1.0, 0.0


def g_deco(seed):
    ss = 2
    R = 130
    im = rgb_canvas(ss, (0.02, 0.07, 0.08))
    d = ImageDraw.Draw(im)
    rows = int(H / (R * 0.5)) + 3
    for ri in range(rows):
        y = ri * R * 0.5 - R * 0.5
        off = (ri % 2) * R
        for ci in range(-1, int(W / (2 * R)) + 2):
            x = ci * 2 * R + off
            t = np.clip(y / H, 0, 1)
            k = 0.5 + 0.5 * math.sin(t * math.pi)
            for j in range(5, 0, -1):
                rr = R * j / 5
                f = (0.07 + 0.05 * (5 - j) * 0.0)
                fillc = (0.02 + 0.03 * k * (1 - j / 5.5), 0.17 + 0.12 * k * (1 - j / 6), 0.18 + 0.1 * k * (1 - j / 6))
                d.ellipse([(x - rr) * ss, (y - rr) * ss, (x + rr) * ss, (y + rr) * ss], fill=tuple(int(255 * c) for c in fillc),
                          outline=tuple(int(255 * c) for c in (0.86 * (0.55 + 0.45 * k), 0.66 * (0.55 + 0.45 * k), 0.28 * (0.55 + 0.45 * k))), width=3 * ss)
    img = from_pil(im)
    # light sweep
    sweep = np.exp(-(((XX - 0.7 * W) + (YY - 1000) * 0.55) / 380) ** 2) * 0.35
    img *= 1 + sweep[..., None] * 1.2
    img += sweep[..., None] * col("#FFD98A") * 0.18
    img *= (0.35 + 0.65 * np.exp(-((YY - 1300) / 1500) ** 2))[..., None]
    n = fbm(seed, 6, 3)
    img *= (0.9 + 0.2 * n)[..., None]
    return img, 1.3, 0.1


def g_marble(seed, stops_base, vein, vein2, gloss=0.2, scale=(0.0013, 0.0020), k=2.2):
    n = fbm(seed, 1.1, 3, 0.45)
    w = fbm(seed + 1, 1.8, 3, 0.45)
    base_n = fbm(seed + 2, 3.0, 5)
    img = ramp(base_n, stops_base)
    t = XX * scale[0] + YY * scale[1] + k * (n - 0.5) + 0.9 * (w - 0.5)
    v = 1 - np.abs(np.sin(t * math.pi * 3.0))
    mod = smooth(0.25, 0.8, fbm(seed + 5, 2.2, 3))
    v1 = (v ** 12) * (0.4 + 0.9 * mod)
    t2 = XX * scale[1] * 1.1 - YY * scale[0] * 1.6 + 3.6 * (w - 0.5) + 1.7 * (n - 0.5)
    v2 = (1 - np.abs(np.sin(t2 * math.pi * 3.0))) ** 38 * 0.6
    img += col(vein) * v1[..., None] * 1.15 + col(vein2) * v2[..., None]
    img += col(vein) * (blur(v1, 8) * 0.35)[..., None]
    sheen = np.exp(-(((XX * 0.8 + YY * 0.6) - 1500) / 520) ** 2) * gloss
    img += sheen[..., None] * 0.6
    return img, 1.2, 0.0


def g_topo(seed, stops_bg, line, line2, K=26):
    h = fbm(seed, 1.5, 6, 0.5)
    h = blur(h, 3)
    img = ramp(h, stops_bg)
    f = h * K
    gy, gx = np.gradient(f)
    gm = np.sqrt(gx * gx + gy * gy) + 1e-4
    dist = np.abs(((f + 0.5) % 1) - 0.5) / gm
    flat = smooth(0.004, 0.02, gm)
    ln = (1 - smooth(0.4, 1.5, dist)) * flat
    idx = np.abs((((f / 5) + 0.5) % 1) - 0.5) / (gm / 5)
    ln5 = (1 - smooth(0.8, 2.6, idx)) * flat
    img += col(line)[None, None, :] * (ln * 0.55)[..., None] * (0.5 + 0.8 * h)[..., None]
    img += col(line2)[None, None, :] * (ln5 * 0.9)[..., None]
    hi = np.exp(-(((XX - 0.35 * W) / 380) ** 2 + ((YY - 1000) / 520) ** 2))
    img += hi[..., None] * col(line2) * 0.12
    return img, 1.3, 0.0


def g_iso(seed, pal, bg, s=92):
    r = np.random.default_rng(seed)
    ss = 2
    im = rgb_canvas(ss, tuple(col(bg)))
    d = ImageDraw.Draw(im)
    n = fbm(seed, 2.0, 3)
    dxs, dys = s * 1.7320508, s * 1.5
    rows = int(H / dys) + 4
    colsn = int(W / dxs) + 3
    H3 = s
    for j in range(-1, rows):
        for i in range(-1, colsn):
            cx = i * dxs + (dxs / 2 if j % 2 else 0)
            cy = j * dys + s
            u = float(n[int(np.clip(cy, 0, H - 1)), int(np.clip(cx, 0, W - 1))])
            hgt = int(np.clip(round(u * 4.5 - 0.6 + r.normal() * 0.7), 0, 4))
            if r.random() < 0.07:
                hgt = 0
            base = ramp(np.array([float(np.clip(u + (r.random() - 0.5) * 0.12, 0, 1))]), pal)[0]
            for k in range(max(1, hgt)):
                yy = cy - k * H3
                if hgt == 0:
                    continue
                v = lambda a, b: (cx + a * s * 0.8660254, yy + b * s)
                top = [v(0, -1), v(1, -.5), v(0, 0), v(-1, -.5)]
                left = [v(-1, -.5), v(0, 0), v(0, 1), v(-1, .5)]
                right = [v(0, 0), v(1, -.5), v(1, .5), v(0, 1)]
                kk = 1.0 - 0.06 * (hgt - k - 1)
                for poly, f in ((top, 1.22), (left, 0.9), (right, 0.62)):
                    c = tuple(int(np.clip(x * f * kk, 0, 1) * 255) for x in base)
                    ln = tuple(int(np.clip(x * f * kk * 0.7, 0, 1) * 255) for x in base)
                    d.polygon([(a * ss, b * ss) for a, b in poly], fill=c, outline=ln)
    img = from_pil(im)
    lightp = np.exp(-(((XX - 400) / 900) ** 2 + ((YY - 700) / 1100) ** 2))
    img *= (0.55 + 0.7 * lightp)[..., None]
    return img, 1.2, 0.0


def g_halftone(seed, bg, c1, c2, c3):
    img = vgrad([(0, bg[0]), (1, bg[1])])
    gx, gy = 1050, 1850
    field = np.zeros((H, W), np.float32)
    f1 = fbm(seed, 1.2, 3)

    def layer(angle, pitch, color, fieldfn, k):
        a = math.radians(angle)
        ca, sa = math.cos(a), math.sin(a)
        ss = 2
        im = Image.new("L", (W * ss, H * ss), 0)
        d = ImageDraw.Draw(im)
        span = int((W + H) / pitch) + 4
        for i in range(-span, span):
            for j in range(-span, span):
                px, py = i * pitch, j * pitch
                x = px * ca - py * sa + W / 2
                y = px * sa + py * ca + H / 2
                if x < -pitch or x > W + pitch or y < -pitch or y > H + pitch:
                    continue
                v = fieldfn(min(max(x, 0), W - 1), min(max(y, 0), H - 1))
                rad = pitch * 0.62 * v
                if rad < 0.6:
                    continue
                d.ellipse([(x - rad) * ss, (y - rad) * ss, (x + rad) * ss, (y + rad) * ss], fill=255)
        m = np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255
        return m

    def fa(x, y):
        d_ = math.hypot(x - gx, y - gy) / 1900
        return float(np.clip(1 - d_, 0, 1) ** 1.1 * (0.6 + 0.5 * f1[int(y), int(x)]))

    def fb(x, y):
        d_ = math.hypot(x - 300, y - 500) / 1500
        return float(np.clip(1 - d_, 0, 1) ** 1.3 * (0.5 + 0.7 * f1[int(y), int(x)]))

    m1 = layer(15, 40, c1, fa, 1)
    m2 = layer(75, 40, c2, fb, 1)
    img += m1[..., None] * col(c1) * 0.9 + m2[..., None] * col(c2) * 0.9
    ov = m1 * m2
    img += ov[..., None] * col(c3) * 0.6
    glow(img, gx, gy, 600, col(c1), 0.18, "exp")
    glow(img, 300, 500, 500, col(c2), 0.15, "exp")
    return img, 1.2, 0.0


def g_waves(seed, bg, c_a, c_b, n_lines=62):
    r = np.random.default_rng(seed)
    ss = 2
    im = rgb_canvas(ss, tuple(col(bg)))
    d = ImageDraw.Draw(im)
    xs = np.linspace(0, W, 360)
    ph = [r.random() * 6.28 for _ in range(6)]
    for i in range(n_lines):
        t = i / (n_lines - 1)
        y0 = 360 + t * (H - 560)
        amp = (40 + 330 * math.sin(t * math.pi) ** 1.2)
        u = xs / W
        env = np.exp(-((u - 0.5 + 0.18 * math.sin(t * 5)) / 0.28) ** 2)
        y = y0 - amp * env * (0.6 + 0.4 * np.sin(u * 9 + ph[0] + t * 8)) * (0.5 + 0.5 * np.sin(u * 17 + ph[1] - t * 6)) ** 1.0 \
            - amp * 0.25 * env * np.sin(u * 31 + ph[2] + t * 10)
        pts = [(float(x) * ss, float(yy) * ss) for x, yy in zip(xs, y)]
        poly = pts + [(W * ss, (H + 20) * ss), (0, (H + 20) * ss)]
        d.polygon(poly, fill=tuple(int(c * 255) for c in col(bg)))
        c = col(c_a) * (1 - t) + col(c_b) * t
        c = c * (0.35 + 0.65 * (0.4 + 0.6 * t))
        d.line(pts, fill=tuple(int(np.clip(x, 0, 1) * 255) for x in c), width=int((2.2 + 2.0 * t) * ss), joint="curve")
    img = from_pil(im)
    L = np.clip(img - col(bg), 0, None)
    img += blur(L, 7) * 1.1 + blur(L, 26) * 1.2
    glow(img, 720, 1500, 800, col(c_b), 0.12, "exp")
    return img, 1.2, 0.0


def g_orbs(seed, sky, orbs):
    img = vgrad(sky)
    n = fbm(seed, 1.6, 4)
    img += (n[..., None] - 0.5) * 0.06
    for x, y, rad, c in orbs:
        sh = np.exp(-(((XX - (x + rad * 0.35)) / (rad * 1.0)) ** 2 + ((YY - (y + rad * 1.18)) / (rad * 0.16)) ** 2))
        img *= 1 - sh[..., None] * 0.25
    for x, y, rad, c in sorted(orbs, key=lambda o: o[2]):
        glass_orb(img, x, y, rad, c, seed=seed)
    stars(img, seed + 2, 120, size=0.7, amp=(0.1, 0.35))
    return img, 1.2, 0.0


# ====================================================================== seasonal
def g_snowy_pines(seed):
    img = vgrad([(0, "#071433"), (0.3, "#12335E"), (0.55, "#2E6C92"), (0.8, "#8FBBD0"), (1, "#C8E2EA")])
    stars(img, seed, 260, ymax=1100, size=0.7, big=2, fade=lambda y: np.clip(1.1 - y / 1100, 0, 1))
    mx, my = 400, 620
    glow(img, mx, my, 560, col("#9FD0FF"), 0.45, "exp")
    disc(img, mx, my, 150, "#F4F8FF", "#DCE8F8")
    glow(img, mx, my, 180, col("#EAF4FF"), 0.7, "exp")
    n = fbm(seed + 2, 2.0, 5, cy=12)
    img += (smooth(0.55, 0.85, n) * np.clip(1 - np.abs(YY - 900) / 500, 0, 1))[..., None] * col("#6D9CC8") * 0.35
    cfg = [(1560, 200, "#4A7DA3", 70, (220, 320)), (1800, 240, "#325F8C", 52, (300, 420)), (2060, 260, "#1F4776", 38, (420, 560)),
           (2380, 260, "#12315A", 26, (580, 760)), (2700, 240, "#08213F", 15, (820, 1050))]
    for i, (b, a, c, n_, hr) in enumerate(cfg):
        p = profile(seed + i, b, a * 0.5, 1.3, 4, 0.5)
        ridge(img, p, c, c)
        fm = forest_mask(seed + 30 + i, lambda x, p=p: float(p[int(np.clip(x, 0, W - 1))]), n_, hr, 0.3)
        comp(img, fm, col(c))
        fog(img, b + 50, 220, "#9DC7D8" if i < 3 else "#3C7AA0", 0.30, seed + 60 + i, 0.5)
    dots(img, seed + 5, 520, "#FFFFFF", 1.2, 3.0, 0.8, 0.9)
    dots(img, seed + 6, 130, "#FFFFFF", 3.0, 6.0, 2.4, 1.0)
    dots(img, seed + 7, 14, "#FFFFFF", 12, 20, 8.0, 1.6)
    return img, 1.2, 0.0


def branch(seed, d, ss, x, y, ang, length, width, depth, twigs, r):
    segs = max(4, int(length / 24))
    pts = [(x, y)]
    for s in range(segs):
        ang += r.normal() * 0.14
        x += math.cos(ang) * length / segs
        y += math.sin(ang) * length / segs
        pts.append((x, y))
        w = width * (1 - s / segs * 0.6)
        d.line([pts[-2][0] * ss, pts[-2][1] * ss, x * ss, y * ss], fill=255, width=max(1, int(w * ss)))
        d.ellipse([(x - w / 2) * ss, (y - w / 2) * ss, (x + w / 2) * ss, (y + w / 2) * ss], fill=255)
        if depth > 0 and r.random() < 0.34:
            branch(seed, d, ss, x, y, ang + (r.random() - 0.5) * 2.0 + (0.5 if r.random() < 0.5 else -0.5), length * 0.55, width * 0.55, depth - 1, twigs, r)
    twigs.append((x, y))
    twigs.append(pts[len(pts) // 2])


def g_sakura(seed):
    img = vgrad([(0, "#5B4B9A"), (0.28, "#C486B6"), (0.5, "#FFB7B8"), (0.7, "#FFD9C2"), (1, "#FFE8D4")])
    glow(img, 360, 1250, 600, col("#FFE2B8"), 0.5, "exp")
    disc(img, 360, 1250, 140, "#FFF6E6", "#FFE3C2")
    n = fbm(seed, 2, 5, cy=12)
    img += (smooth(0.55, 0.85, n) * np.clip(1 - np.abs(YY - 1500) / 800, 0, 1))[..., None] * col("#FFE0D0") * 0.4
    # distant blurred blossom trees
    for i, (y, c) in enumerate([(1900, "#E79BB4"), (2150, "#D17AA0")]):
        p = profile(seed + i, y, 220, 2.5, 5, 0.6)
        ridge(img, p, c, "#B65C86", fade_y=600)
    bokeh(img, seed + 4, 28, ["#FFC7DA", "#FFFFFF", "#FFB0CB"], 40, 130, (0.08, 0.2), 0.35)
    # branches
    r = np.random.default_rng(seed + 9)
    twigs = []
    ss = 2

    def f(dr, s2):
        branch(seed, dr, s2, W + 80, 360, math.pi * 1.02 + 0.12, 980, 46, 4, twigs, r)
        branch(seed, dr, s2, W + 60, 980, math.pi * 0.98 - 0.12, 620, 26, 3, twigs, r)
        branch(seed, dr, s2, -80, 1980, -0.06, 760, 34, 3, twigs, r)

    bm = draw_mask(f)
    comp(img, bm, col("#3A1F33"))
    img += (np.clip(bm - np.roll(bm, 3, axis=1), 0, 1) * 0.3)[..., None] * col("#C06A8A")
    # blossoms
    fl = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    def flower(sz):
        im = Image.new("RGBA", (int(sz * 4), int(sz * 4)), (0, 0, 0, 0))
        dd = ImageDraw.Draw(im)
        c = sz * 2
        for k in range(5):
            a = k * 2 * math.pi / 5 + r.random() * 0.3
            px, py = c + math.cos(a) * sz * 0.62, c + math.sin(a) * sz * 0.62
            dd.ellipse([px - sz * 0.5, py - sz * 0.5, px + sz * 0.5, py + sz * 0.5], fill=(255, 196 + int(r.integers(0, 30)), 212 + int(r.integers(0, 24)), 255))
        dd.ellipse([c - sz * 0.22, c - sz * 0.22, c + sz * 0.22, c + sz * 0.22], fill=(255, 130, 160, 255))
        return im

    for (x, y) in twigs:
        for _ in range(int(r.integers(1, 3))):
            sz = 12 + r.random() * 16
            f_ = flower(sz)
            fl.alpha_composite(f_, (int(x + r.normal() * 30 - sz * 2), int(y + r.normal() * 30 - sz * 2)))
    a = np.asarray(fl, np.float32) / 255
    img = img * (1 - a[..., 3:4]) + a[..., :3] * a[..., 3:4]
    img += blur(a[..., 3:4].repeat(3, 2) * col("#FFB3CC"), 14) * 0.18
    # falling petals
    pl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for i in range(190):
        sz = 8 + r.random() * 20
        pt = Image.new("RGBA", (int(sz * 3), int(sz * 3)), (0, 0, 0, 0))
        dd = ImageDraw.Draw(pt)
        dd.ellipse([sz * .6, sz * .9, sz * 2.2, sz * 2.0], fill=(255, 190 + int(r.integers(0, 40)), 208 + int(r.integers(0, 30)), 235))
        pt = pt.rotate(r.random() * 360, resample=Image.BICUBIC)
        if sz > 22 or sz < 11:
            pt = pt.filter(ImageFilter.GaussianBlur(2.2))
        pl.alpha_composite(pt, (int(r.random() * W), int(r.random() * H)))
    a = np.asarray(pl, np.float32) / 255
    img = img * (1 - a[..., 3:4]) + a[..., :3] * a[..., 3:4]
    return img, 1.2, 0.1


def leaf_patch(r, sz, c):
    im = Image.new("RGBA", (int(sz * 3), int(sz * 3)), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    ox, oy = sz * 1.5, sz * 1.5
    pts = []
    for t in np.linspace(0, 1, 18):
        wd = math.sin(t * math.pi) ** 0.8 * sz * 0.46 * (1 + 0.3 * math.sin(t * 8 + r.random()))
        pts.append((ox - sz * 0.9 + t * sz * 1.8, oy - wd))
    for t in np.linspace(1, 0, 18):
        wd = math.sin(t * math.pi) ** 0.8 * sz * 0.46
        pts.append((ox - sz * 0.9 + t * sz * 1.8, oy + wd))
    d.polygon(pts, fill=c)
    d.line([ox - sz * 0.9, oy, ox + sz * 0.9, oy], fill=(max(c[0] - 70, 0), max(c[1] - 60, 0), max(c[2] - 30, 0), 255), width=max(1, int(sz * 0.07)))
    return im.rotate(r.random() * 360, resample=Image.BICUBIC)


def g_autumn(seed):
    img = vgrad([(0, "#5B2A3B"), (0.25, "#C5543A"), (0.5, "#F59E4E"), (0.75, "#FFD089"), (1, "#F9C877")])
    sx, sy = 980, 700
    glow(img, sx, sy, 900, col("#FFC27A"), 0.55, "exp")
    glow(img, sx, sy, 160, col("#FFF2CF"), 0.9, "exp")
    ang = np.arctan2(YY - sy, XX - sx)
    shaft = (0.5 + 0.5 * np.sin(ang * 14 + 0.5)) ** 4 * np.exp(-np.sqrt((XX - sx) ** 2 + (YY - sy) ** 2) / 1500)
    img += shaft[..., None] * col("#FFE2A8") * 0.22
    r = np.random.default_rng(seed)
    # canopy bokeh
    bokeh(img, seed + 1, 70, ["#FFB347", "#FF8A3D", "#E8553A", "#FFD36B"], 35, 150, (0.08, 0.28), 0.4, (0, 0, W, 1500))
    # tree trunks layers
    for i, (cnt, wd, c, hz) in enumerate([(7, 38, "#A55B42", 0.55), (6, 64, "#6E3A3A", 0.38), (4, 104, "#3E2030", 0.18), (3, 150, "#1C0F1E", 0.0)]):
        def f(dr, ss):
            for _ in range(cnt):
                x = r.random() * W
                w = wd * (0.7 + 0.6 * r.random())
                lean = (r.random() - 0.5) * 60
                dr.polygon([((x - w / 2) * ss, H * ss), ((x + w / 2) * ss, H * ss), ((x + w / 2 * 0.7 + lean) * ss, 0), ((x - w / 2 * 0.7 + lean) * ss, 0)], fill=255)
        m = draw_mask(f)
        comp(img, m, col(c) * (1 - hz) + col("#F6A867") * hz)
        fog(img, 1800, 700, "#FFB872", 0.18 * (1 - i * 0.2), seed + 5 + i, 0.5)
    # ground leaves
    gp = profile(seed + 3, 2330, 130, 2.2, 5, 0.5)
    ridge(img, gp, "#8A3A22", "#3A1620", fade_y=300, tex=0.6, seed=seed)
    # falling leaves
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cols_ = [(214, 62, 38), (240, 130, 40), (250, 190, 60), (176, 50, 40), (226, 100, 36)]
    for i in range(150):
        sz = 14 + r.random() * 40
        c = cols_[int(r.integers(0, len(cols_)))]
        pt = leaf_patch(r, sz, c + (255,))
        if sz > 42 or sz < 17:
            pt = pt.filter(ImageFilter.GaussianBlur(2.5 if sz > 42 else 1.2))
        layer.alpha_composite(pt, (int(r.random() * W - sz), int(r.random() * H - sz)))
    a = np.asarray(layer, np.float32) / 255
    img = img * (1 - a[..., 3:4]) + a[..., :3] * a[..., 3:4]
    return img, 1.4, 0.15


def g_fireworks(seed):
    img = vgrad([(0, "#02030E"), (0.5, "#0A0E2E"), (0.72, "#1B1C4A"), (0.73, "#050716"), (1, "#02030A")])
    stars(img, seed, 700, ymax=1700, size=0.7, big=0)
    hy = 1850
    layer = rgb_canvas(2)
    dr = ImageDraw.Draw(layer)
    r = np.random.default_rng(seed + 1)
    bursts = [(430, 640, 360, "#FFD36B", 1), (1030, 460, 300, "#FF5E8E", 1), (760, 1110, 330, "#6BE8FF", 1), (240, 1380, 200, "#B38CFF", 1), (1190, 1330, 220, "#7CFFB0", 1), (700, 300, 160, "#FFFFFF", 1)]
    for (cx, cy, R, c, _) in bursts:
        cc = col(c)
        n = int(R * 0.9)
        for k in range(n):
            a = r.random() * 2 * math.pi
            sp = R * (0.55 + 0.45 * r.random() ** 0.6)
            steps = 9
            for s in range(steps):
                t0, t1 = s / steps, (s + 1) / steps
                def pos(t):
                    return (cx + math.cos(a) * sp * (1 - (1 - t) ** 2), cy + math.sin(a) * sp * (1 - (1 - t) ** 2) + (R * 0.22) * t * t)
                (xa, ya), (xb, yb) = pos(t0), pos(t1)
                f = (t1 ** 1.6) * (0.35 + 0.65 * r.random() * 0 + 0.65)
                colr = tuple(int(255 * min(1, v * (1.0 - t1 * 0.25 * 0 ) * f + (1 - 0.0) * 0.0 + (0.55 * f if t1 > 0.9 else 0))) for v in cc)
                dr.line([xa * 2, ya * 2, xb * 2, yb * 2], fill=colr, width=int(2 + 3 * t1) )
            ex, ey = pos(1)
            dr.ellipse([ex * 2 - 5, ey * 2 - 5, ex * 2 + 5, ey * 2 + 5], fill=tuple(int(255 * min(1, v + 0.4)) for v in cc))
    L = from_pil(layer)
    img += L * 1.05 + blur(L, 6) * 1.3 + blur(L, 26) * 1.4
    for (cx, cy, R, c, _) in bursts:
        glow(img, cx, cy, R * 1.3, col(c), 0.26, "exp")
        glow(img, cx, cy, 22, col("#FFFFFF"), 0.9, "exp")
    # water + reflection
    idx = np.clip(2 * hy - np.arange(H), 0, H - 1)
    refl = img[idx][hy:]
    wob = fbm(seed + 3, 5, 4, cy=90)[hy:H]
    xs = np.clip(np.arange(W)[None, :] + ((wob - 0.5) * 60).astype(int), 0, W - 1)
    refl = np.take_along_axis(refl, xs[..., None].repeat(3, 2), 1)
    refl = blur(refl, 2.0) * 0.55
    fadey = np.clip(1 - (np.arange(H - hy)[:, None, None] / (H - hy)) * 1.3, 0, 1)
    img[hy:] = col("#05071A") * (1 - fadey * 0.3) + refl * fadey
    # skyline
    base = 1860
    sk = draw_mask(lambda d, ss: [d.rectangle([x * ss, (base - h) * ss, (x + w) * ss, (base + 4) * ss], fill=255) for x, w, h in
                                   [(int(a), int(b), int(c)) for a, b, c in zip(np.cumsum(np.r_[0, r.integers(30, 110, 22)])[:22], r.integers(30, 110, 22), r.integers(40, 260, 22))]])
    comp(img, sk, col("#04050F"))
    for _ in range(80):
        x, y = r.random() * W, base - r.random() * 130
        if sk[int(y), int(x)] > 0.5:
            glow(img, x, y, 3.5, col("#FFD27A"), 0.7, "exp")
    return img, 1.3, 0.1


def g_lanterns(seed):
    img = vgrad([(0, "#0A0F36"), (0.4, "#1B1F5E"), (0.7, "#4B2A6F"), (1, "#92457A")])
    stars(img, seed, 650, ymax=1500, size=0.7, big=3, fade=lambda y: np.clip(1.1 - y / 1500, 0, 1))
    glow(img, 1040, 480, 420, col("#C7D4FF"), 0.35, "exp")
    disc(img, 1040, 480, 85, "#F7F5FF", "#D8DEFA")
    for i, (b, a, c) in enumerate([(2100, 260, "#30245E"), (2300, 260, "#1C1448"), (2480, 200, "#0C0830")]):
        p = profile(seed + i, b, a, 1.6, 5, 0.5)
        ridge(img, p, c, c)
        fog(img, b, 120, "#8A4A86", 0.25, seed + 5 + i)
    r = np.random.default_rng(seed + 7)
    items = [(r.random() * W, r.random() * 2300 + 60, 14 + (r.random() ** 1.5) * 70) for _ in range(60)]
    items.sort(key=lambda t: t[2])
    for x, y, s in items:
        pad = int(s * 4)
        x0, x1 = int(max(0, x - pad)), int(min(W, x + pad))
        y0, y1 = int(max(0, y - pad)), int(min(H, y + pad))
        if x1 <= x0 or y1 <= y0:
            continue
        glow(img, x, y, s * 2.2, col("#FF9A3D"), 0.28, "exp", 3.5)
        glow(img, x, y, s * 0.9, col("#FFC46B"), 0.35, "exp", 3.5)
        d = ((XX[y0:y1, x0:x1] - x) / (s * 0.62)) ** 6 + ((YY[y0:y1, x0:x1] - y) / (s * 0.82)) ** 6
        a = 1 - smooth(0.92, 1.0, d)
        vy = (YY[y0:y1, x0:x1] - (y - s * 0.82)) / (s * 1.64)
        body = ramp(np.clip(vy, 0, 1), [(0, "#FFB347"), (0.5, "#FF8A2D"), (1, "#E0501E")])
        rib = 0.9 + 0.1 * np.cos((XX[y0:y1, x0:x1] - x) / (s * 0.62) * math.pi * 3)
        inner = np.exp(-(((XX[y0:y1, x0:x1] - x) / (s * 0.5)) ** 2 + ((YY[y0:y1, x0:x1] - y) / (s * 0.6)) ** 2))
        c = body * rib[..., None] * 1.2 + inner[..., None] * col("#FFE9B0") * 0.9
        if s < 30:
            sc = 1
        sub = img[y0:y1, x0:x1]
        img[y0:y1, x0:x1] = sub * (1 - a[..., None]) + c * a[..., None]
    return img, 1.3, 0.1


# ====================================================================== gaming
def g_arcade(seed):
    img = vgrad([(0, "#0A0420"), (0.5, "#17063A"), (1, "#06020F")])
    cx, cy = 720, 1130
    glow(img, cx, cy, 900, col("#6A1BFF"), 0.3, "exp")
    # floor grid
    hy = 1700

    def fl(d, ss):
        for i in range(-30, 31):
            d.line([(720 * ss, hy * ss), ((720 + i * 230) * ss, (H + 20) * ss)], fill=255, width=2 * ss)
        for k in range(1, 18):
            t = k / 17
            y = hy + (H - hy) * t ** 2.2
            d.line([(0, y * ss), (W * ss, y * ss)], fill=255, width=2 * ss)

    gm = draw_mask(fl) * smooth(hy, hy + 40, YY)
    img += gm[..., None] * col("#27E6FF") * 0.55 * np.clip((YY - hy) / 300, 0, 1)[..., None]
    img += blur(gm, 8)[..., None] * col("#27E6FF") * 0.5 * np.clip((YY - hy) / 300, 0, 1)[..., None]
    shapes = [("#FF2DB5", lambda d, ss: [d.rounded_rectangle([(cx - 520 + i * 70) * ss, (cy - 700 + i * 70) * ss, (cx + 520 - i * 70) * ss, (cy + 700 - i * 70) * ss], radius=(120 - i * 12) * ss, outline=255, width=9 * ss) for i in range(0, 1)]),
              ("#27E6FF", lambda d, ss: d.rounded_rectangle([(cx - 400) * ss, (cy - 580) * ss, (cx + 400) * ss, (cy + 580) * ss], radius=80 * ss, outline=255, width=9 * ss)),
              ("#FFD23F", lambda d, ss: d.rounded_rectangle([(cx - 280) * ss, (cy - 460) * ss, (cx + 280) * ss, (cy + 460) * ss], radius=50 * ss, outline=255, width=9 * ss))]
    for c, f in shapes:
        m = draw_mask(f)
        # cut gaps
        gap = ((XX > cx - 90) & (XX < cx + 90) & (YY < cy - 400)).astype(np.float32)
        neon(img, m * (1 - gap * (c == "#27E6FF")), c, 0.55, ((3.0, 1.0), (12.0, 0.8), (45.0, 0.5)), 0.9)

    def sym(d, ss):
        d.ellipse([(cx - 130) * ss, (cy - 130) * ss, (cx + 130) * ss, (cy + 130) * ss], outline=255, width=9 * ss)
        d.polygon([(cx * ss, (cy - 60) * ss), ((cx + 52) * ss, (cy + 40) * ss), ((cx - 52) * ss, (cy + 40) * ss)], outline=255, width=9 * ss)
        for k in range(4):
            a = math.radians(45 + 90 * k)
            d.line([(cx + math.cos(a) * 170) * ss, (cy + math.sin(a) * 170) * ss, (cx + math.cos(a) * 230) * ss, (cy + math.sin(a) * 230) * ss], fill=255, width=9 * ss)
    neon(img, draw_mask(sym), "#FF2DB5", 0.6, ((3.0, 1.0), (12.0, 0.8), (45.0, 0.5)), 1.0)

    def plusses(d, ss):
        r = np.random.default_rng(seed)
        for _ in range(26):
            x, y = r.random() * W, r.random() * 1600
            s = 10 + r.random() * 14
            d.line([(x - s) * ss, y * ss, (x + s) * ss, y * ss], fill=255, width=4 * ss)
            d.line([x * ss, (y - s) * ss, x * ss, (y + s) * ss], fill=255, width=4 * ss)
    neon(img, draw_mask(plusses), "#8F5CFF", 0.6, ((2, 1.0), (10, 0.6)), 0.8)
    stars(img, seed, 400, ymax=1650, size=0.7, amp=(0.15, 0.5))
    # scanlines
    img *= (0.9 + 0.1 * np.sin(YY * math.pi * 2 / 4))[..., None]
    return img, 1.2, 0.15


def torch_flame(img, x, y, s, seed):
    r = np.random.default_rng(seed)
    glow(img, x, y - s * 0.4, s * 5.5, col("#FF7A1F"), 0.5, "exp")
    glow(img, x, y - s * 0.4, s * 2.2, col("#FFA84A"), 0.7, "exp")
    dx = (XX - x) / (s * 0.9)
    dy = (YY - (y - s * 1.2)) / (s * 1.7)
    sway = 0.22 * np.sin(dy * 3 + seed)
    shape = np.clip(1 - (dx - sway * (1 - dy) * 0.5) ** 2 / np.clip(0.35 * (1 - np.clip(dy, -1, 1)) + 0.02, 0.02, 1) - np.clip(dy, -1, 1) ** 2, 0, 1)
    shape = shape[..., None] if False else shape
    x0, x1 = int(max(0, x - s * 3)), int(min(W, x + s * 3))
    y0, y1 = int(max(0, y - s * 4)), int(min(H, y + s * 1.2))
    sub = shape[y0:y1, x0:x1]
    c = ramp(np.clip(sub, 0, 1) ** 0.6, [(0, "#FF4B10"), (0.5, "#FFA02A"), (1, "#FFF1B8")])
    a = np.clip(sub * 2.0, 0, 1)[..., None]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + c * a


def g_dungeon(seed):
    r = np.random.default_rng(seed)
    ss = 1
    im = Image.new("RGB", (W, H), (14, 10, 12))
    d = ImageDraw.Draw(im)
    bh = 92
    y = 0
    ri = 0
    while y < H:
        x = -int(r.integers(0, 150))
        while x < W:
            bw = int(r.integers(150, 260))
            v = 0.55 + 0.45 * r.random()
            c = (int(80 * v + r.integers(-6, 7)), int(70 * v + r.integers(-6, 7)), int(76 * v + r.integers(-6, 7)))
            d.rectangle([x + 3, y + 3, x + bw - 3, y + bh - 3], fill=c)
            d.line([x + 3, y + 3, x + bw - 3, y + 3], fill=tuple(min(255, int(q * 1.35)) for q in c), width=3)
            d.line([x + 3, y + 3, x + 3, y + bh - 3], fill=tuple(min(255, int(q * 1.2)) for q in c), width=3)
            d.line([x + 3, y + bh - 4, x + bw - 3, y + bh - 4], fill=tuple(int(q * 0.6) for q in c), width=3)
            x += bw
        y += bh
        ri += 1
    base = from_pil(im)
    tx = fbm(seed + 1, 18, 5)
    big = fbm(seed + 2, 3, 4)
    base *= (0.65 + 0.7 * tx)[..., None] * (0.7 + 0.5 * big)[..., None]
    base = base * (1 + 0.0)
    torches = [(300, 900), (1140, 1500), (330, 2080)]
    light = np.zeros((H, W), np.float32)
    flick = fbm(seed + 3, 6, 3)
    for tx_, ty_ in torches:
        d2 = ((XX - tx_) ** 2 + (YY - ty_) ** 2) / 700.0 ** 2
        light += 1.0 / (1 + d2 * 7) ** 1.3
    light *= 0.75 + 0.5 * flick
    img = base * (0.05 + 2.7 * light[..., None]) * np.array([1.0, 0.62, 0.32], np.float32)
    img += col("#2A1E44") * (0.05 + 0.05 * big)[..., None]
    for i, (tx_, ty_) in enumerate(torches):
        # bracket
        m = draw_mask(lambda dd, s2: (dd.rectangle([(tx_ - 12) * s2, ty_ * s2, (tx_ + 12) * s2, (ty_ + 150) * s2], fill=255), dd.polygon([((tx_ - 46) * s2, (ty_ - 10) * s2), ((tx_ + 46) * s2, (ty_ - 10) * s2), ((tx_ + 24) * s2, (ty_ + 38) * s2), ((tx_ - 24) * s2, (ty_ + 38) * s2)], fill=255)))
        comp(img, m, col("#17100E"))
        torch_flame(img, tx_, ty_ - 8, 66, seed + i)
    # embers
    dots(img, seed + 5, 120, "#FFB13A", 1.2, 3.2, 1.0, 1.4)
    vignette(img, 0.55)
    return img, 1.6, 0.0


def g_mecha(seed):
    r = np.random.default_rng(seed)
    img = vgrad([(0, "#1B2128"), (0.5, "#262D36"), (1, "#14181E")])
    ss = 2
    im = Image.new("RGB", (W * ss, H * ss), (0, 0, 0))
    d = ImageDraw.Draw(im)
    # plates
    ys = [0]
    while ys[-1] < H:
        ys.append(ys[-1] + int(r.integers(220, 480)))
    for i in range(len(ys) - 1):
        xs = [0]
        while xs[-1] < W:
            xs.append(xs[-1] + int(r.integers(300, 760)))
        for j in range(len(xs) - 1):
            v = 0.7 + 0.5 * r.random()
            c = (int(52 * v), int(60 * v), int(70 * v))
            x0, y0, x1, y1 = xs[j], ys[i], xs[j + 1], ys[i + 1]
            d.rectangle([x0 * ss, y0 * ss, x1 * ss, y1 * ss], fill=c)
            d.line([x0 * ss, y0 * ss, x1 * ss, y0 * ss], fill=tuple(min(255, int(q * 1.5)) for q in c), width=3 * ss)
            d.line([x0 * ss, y0 * ss, x0 * ss, y1 * ss], fill=tuple(min(255, int(q * 1.3)) for q in c), width=3 * ss)
            d.line([x0 * ss, y1 * ss, x1 * ss, y1 * ss], fill=(6, 8, 10), width=5 * ss)
            d.line([x1 * ss, y0 * ss, x1 * ss, y1 * ss], fill=(6, 8, 10), width=5 * ss)
            for bx, by in ((x0 + 26, y0 + 26), (x1 - 26, y0 + 26), (x0 + 26, y1 - 26), (x1 - 26, y1 - 26)):
                d.ellipse([(bx - 9) * ss, (by - 9) * ss, (bx + 9) * ss, (by + 9) * ss], fill=(26, 31, 36), outline=(96, 106, 118), width=2 * ss)
    base = from_pil(im)
    brushed = fbm(seed + 2, 4, 3, cy=300)
    base *= (0.8 + 0.4 * brushed)[..., None]
    sc = fbm(seed + 3, 30, 3)
    base *= (0.9 + 0.2 * sc)[..., None]
    img = base
    # hazard band (diagonal)
    ang = math.radians(-22)
    u = (XX - 720) * math.sin(ang) * -1 + (YY - 1500) * math.cos(ang)
    band = (np.abs(u) < 220).astype(np.float32)
    bandm = blur(band, 0.8)
    stripe = ((( (XX - 720) * math.cos(ang) + (YY - 1500) * math.sin(ang)) / 70.0) % 2) < 1
    hz = np.where(stripe[..., None], col("#F6C21B"), col("#14161A"))
    hz *= (0.85 + 0.15 * fbm(seed + 8, 12, 3))[..., None]
    edge = np.clip(1 - np.abs(np.abs(u) - 220) / 6, 0, 1)
    comp(img, bandm, hz)
    img *= (1 - (np.exp(-np.clip(np.abs(u) - 220, 0, None) / 40) * (np.abs(u) >= 220)) * 0.0)[..., None]
    img += (np.exp(-np.clip(np.abs(u) - 220, 0, None) / 90) * (np.abs(u) > 218) * 0.14)[..., None] * col("#FF8A1F")
    # reactor rings
    def ring_f(dr, s2):
        for R, wd in ((300, 14), (250, 5), (200, 22), (120, 6)):
            dr.ellipse([(720 - R) * s2, (620 - R) * s2, (720 + R) * s2, (620 + R) * s2], outline=255, width=wd * s2)
        for k in range(12):
            a_ = math.radians(k * 30)
            dr.line([(720 + math.cos(a_) * 120) * s2, (620 + math.sin(a_) * 120) * s2, (720 + math.cos(a_) * 300) * s2, (620 + math.sin(a_) * 300) * s2], fill=255, width=4 * s2)
    rm = draw_mask(ring_f)
    dark = np.exp(-(((np.sqrt((XX - 720) ** 2 + (YY - 620) ** 2)) / 330) ** 6))
    img *= (1 - dark * 0.7)[..., None]
    neon(img, rm, "#FF7A1F", 0.45, ((2.0, 0.7), (10.0, 0.7), (40.0, 0.6)), 0.8)
    glow(img, 720, 620, 160, col("#FFB36B"), 0.9, "exp")
    # warn lights
    for x, y in ((140, 520), (1300, 2250), (1180, 760)):
        glow(img, x, y, 100, col("#FF4D1F"), 0.5, "exp")
        glow(img, x, y, 22, col("#FFB07A"), 1.0, "exp")
    # hex vent
    def hexv(dr, s2):
        for i in range(5):
            for j in range(3):
                cx_, cy_ = 150 + j * 70 + (i % 2) * 35, 1950 + i * 56
                pts = [(cx_ + 28 * math.cos(math.radians(60 * k)), cy_ + 28 * math.sin(math.radians(60 * k))) for k in range(6)]
                dr.polygon([(a * s2, b * s2) for a, b in pts], fill=255)
    hv = draw_mask(hexv)
    comp(img, hv, col("#07090C"))
    img += (np.clip(blur(hv, 3) - hv, 0, 1) * 0.0)[..., None]
    vignette(img, 0.45)
    return img, 1.5, 0.0


def g_flames(seed):
    img = vgrad([(0, "#07010A"), (0.55, "#1D040C"), (1, "#3A0A0A")])
    glow(img, 720, 2560, 1500, col("#FF4B14"), 0.55, "exp")
    yn = (H - YY) / H
    wx = fbm(seed + 1, 5, 3, cy=2)
    n = fbm(seed, 6, 6, 0.55, cy=1.8)
    n2 = warp_sample(n, wx, fbm(seed + 2, 4, 3), 420)
    yw = np.clip(1 - yn * 1.0, 0, 1)
    v = n2 * 1.25 + (1 - yn) * 0.0 - yn * 1.15 + 0.5
    # tongues: vertical stretch
    tong = fbm(seed + 4, 7, 4, cy=0.9)
    tong = warp_sample(tong, wx, wx, 260)
    t = np.clip((tong * 0.9 + n2 * 0.5) * 1.15 - yn * 1.25 + 0.38, 0, 1)
    f = ramp(t ** 0.8, [(0, "#000000"), (0.18, "#8A0E0E"), (0.45, "#E8331C"), (0.7, "#FF8A1E"), (0.9, "#FFC03A"), (1, "#FFE08A")])
    img += f * (t ** 0.6)[..., None] * 1.0
    img += blur(f, 18) * 0.25
    # sparks
    r = np.random.default_rng(seed)
    dots(img, seed + 7, 160, "#FFB13A", 1.2, 3.0, 0.9, 1.6, region=(0, 400, W, 2400))
    dots(img, seed + 8, 40, "#FF7A22", 4, 9, 4.0, 1.3, region=(0, 600, W, 2400))
    # spotlight cone
    cone = np.clip(1 - np.abs((XX - 720) / (200 + YY * 0.45)), 0, 1) ** 2 * np.clip(1 - YY / 2200, 0, 1)
    img += cone[..., None] * col("#FFEBD0") * 0.1
    vignette(img, 0.35)
    return img, 1.6, 0.0


def g_anime_city(seed):
    r = np.random.default_rng(seed)
    hy = 1620
    img = vgrad([(0, "#1A1055"), (0.2, "#4C2A92"), (0.38, "#B24AA8"), (0.5, "#FF7B8E"), (0.6, "#FFB07A"), (1, "#FFB07A")])
    n = fbm(seed, 2, 5, cy=14)
    cl = smooth(0.5, 0.82, n) * np.clip(1 - np.abs(YY - 900) / 800, 0, 1)
    img += cl[..., None] * ramp(np.clip(YY / hy, 0, 1), [(0, "#7A55D0"), (0.6, "#FF6FA0"), (1, "#FFC48A")]) * 0.45
    stars(img, seed + 1, 300, ymax=700, size=0.7, fade=lambda y: np.clip(1 - y / 700, 0, 1))
    glow(img, 920, 1430, 700, col("#FF9A6B"), 0.4, "exp")
    disc(img, 920, 1380, 150, "#FFF0D8", "#FFD4B4")
    # skyline layers
    cfg = [(hy, 0.7, "#6A3A8E", 0.35, 70, (160, 600)), (hy + 60, 0.5, "#4A2574", 0.5, 90, (200, 760)), (hy + 120, 0.3, "#2A1555", 0.75, 110, (260, 980)), (hy + 200, 0.15, "#140A30", 1.0, 150, (320, 1250))]
    for i, (b, lw, c, dk, wmean, hr) in enumerate(cfg):
        def f(dr, ss, wmean=wmean, hr=hr, b=b):
            x = -40
            while x < W + 40:
                w = wmean * (0.6 + 0.9 * r.random())
                h = hr[0] + (hr[1] - hr[0]) * r.random() ** 1.7
                dr.rectangle([x * ss, (b - h) * ss, (x + w) * ss, (H) * ss], fill=255)
                if r.random() < 0.3:
                    dr.rectangle([(x + w * 0.15) * ss, (b - h - 40) * ss, (x + w * 0.85) * ss, (b - h) * ss], fill=255)
                    dr.line([(x + w * 0.5) * ss, (b - h - 90) * ss, (x + w * 0.5) * ss, (b - h - 40) * ss], fill=255, width=2 * ss)
                x += w + r.random() * 10
        m = draw_mask(f)
        comp(img, m, col(c))
        # windows
        win = np.zeros((H, W), np.float32)
        ps = int(18 + i * 7)
        gx = (XX % (ps * 1.5)) < ps * 0.6
        gy = (YY % (ps * 2.0)) < ps * 0.9
        rnd = noise(seed + 30 + i, W / (ps * 1.5), H / (ps * 2.0))
        lit = (rnd > 0.62 + 0.03 * i)
        wm = (gx & gy & lit).astype(np.float32) * m * (YY < b + 20)
        wm = blur(wm, 0.6)
        wc = ramp(noise(seed + 50 + i, 10, 25), [(0, "#FFD27A"), (0.5, "#FFE9B0"), (1, "#8FE7FF")])
        img += wm[..., None] * wc * (0.6 + 0.4 * i / 3)
        img += blur(wm, 8)[..., None] * wc * 0.4
        fog(img, b - 80, 180, "#FF9A8A", 0.28 - i * 0.05, seed + 70 + i, 0.5)
    # power lines
    def pl(dr, ss):
        for k, (yy, sag) in enumerate([(300, 120), (360, 140), (420, 160)]):
            pts = [(x * ss, (yy + sag * (1 - ((x - W / 2) / (W / 2)) ** 2) * -1 + sag) * ss) for x in range(0, W + 1, 20)]
            dr.line(pts, fill=255, width=3 * ss, joint="curve")
        dr.rectangle([(W * 0.0 + 60) * ss, 100 * ss, (W * 0.0 + 82) * ss, H * ss], fill=255)
        dr.rectangle([(W - 100) * ss, 100 * ss, (W - 78) * ss, H * ss], fill=255)
        dr.rectangle([30 * ss, 220 * ss, 150 * ss, 238 * ss], fill=255)
        dr.rectangle([(W - 170) * ss, 220 * ss, (W - 50) * ss, 238 * ss], fill=255)
    comp(img, draw_mask(pl), col("#0B0620"), 0.95)
    # foreground bokeh
    bokeh(img, seed + 9, 38, ["#FF8FD0", "#7FE6FF", "#FFD27A", "#FF6F9A"], 45, 170, (0.12, 0.42), 0.55, (0, 1100, W, 2500), 0.1)
    bokeh(img, seed + 10, 34, ["#FFB86B", "#FF8FD0", "#B58CFF"], 12, 38, (0.18, 0.5), 0.5, (0, 1000, W, 2500), 0.1)
    return img, 1.2, 0.05


def g_radar(seed):
    img = vgrad([(0, "#031018"), (1, "#020A12")])
    cx, cy = 720, 1260
    rho = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    img += col("#073B49") * (np.clip(1 - rho / 1300, 0, 1) ** 1.5 * 0.9)[..., None]
    # hex faint
    def f(d, ss):
        for R in (180, 330, 480, 630):
            d.ellipse([(cx - R) * ss, (cy - R) * ss, (cx + R) * ss, (cy + R) * ss], outline=255, width=(3 if R != 630 else 6) * ss)
        for a in range(0, 360, 10):
            ar = math.radians(a)
            L = 18 if a % 30 else 36
            d.line([(cx + math.cos(ar) * (680 - L)) * ss, (cy + math.sin(ar) * (680 - L)) * ss, (cx + math.cos(ar) * 680) * ss, (cy + math.sin(ar) * 680) * ss], fill=255, width=3 * ss)
        d.line([(cx - 720) * ss, cy * ss, (cx + 720) * ss, cy * ss], fill=160, width=2 * ss)
        d.line([cx * ss, (cy - 720) * ss, cx * ss, (cy + 720) * ss], fill=160, width=2 * ss)
        d.arc([(cx - 790) * ss, (cy - 790) * ss, (cx + 790) * ss, (cy + 790) * ss], 200, 340, fill=255, width=10 * ss)
        d.arc([(cx - 790) * ss, (cy - 790) * ss, (cx + 790) * ss, (cy + 790) * ss], 20, 160, fill=255, width=10 * ss)
        d.arc([(cx - 850) * ss, (cy - 850) * ss, (cx + 850) * ss, (cy + 850) * ss], 250, 290, fill=255, width=5 * ss)
        d.arc([(cx - 850) * ss, (cy - 850) * ss, (cx + 850) * ss, (cy + 850) * ss], 70, 110, fill=255, width=5 * ss)
        for k in range(3):
            y = 270 + k * 24
            d.rectangle([(cx - 220 + k * 40) * ss, y * ss, (cx + 220 - k * 40) * ss, (y + 6) * ss], fill=255)
            y2 = 2290 + k * 24
            d.rectangle([(cx - 220 + k * 40) * ss, y2 * ss, (cx + 220 - k * 40) * ss, (y2 + 6) * ss], fill=255)
    m = draw_mask(f)
    neon(img, m, "#27F0D2", 0.4, ((2.0, 0.9), (9.0, 0.6), (36.0, 0.35)), 0.85)
    # sweep
    th = np.arctan2(YY - cy, XX - cx)
    sw0 = -0.6
    dlt = (sw0 - th) % (2 * math.pi)
    sweep = np.clip(1 - dlt / 1.7, 0, 1) ** 2.2 * (rho < 630)
    img += sweep[..., None] * col("#27F0D2") * 0.5
    edge = np.exp(-(dlt / 0.01) ** 2) * (rho < 630)
    img += edge[..., None] * col("#BFFFF2") * 0.9
    r = np.random.default_rng(seed)
    for i in range(14):
        a = r.random() * 2 * math.pi
        R = 80 + r.random() * 520
        x, y = cx + math.cos(a) * R, cy + math.sin(a) * R
        dd = (sw0 - a) % (2 * math.pi)
        k = 0.4 + 0.8 * np.clip(1 - dd / 3.0, 0, 1)
        glow(img, x, y, 14, col("#B6FFF0"), 0.8 * k, "exp")
        glow(img, x, y, 4, col("#FFFFFF"), 1.2 * k, "exp")
    glow(img, cx, cy, 14, col("#FFFFFF"), 1.0, "exp")
    img *= (0.92 + 0.08 * np.sin(YY * math.pi * 2 / 4))[..., None]
    return img, 1.3, 0.1


def crystal(d, x, y, ang, L, wd, ss, jit):
    ca, sa = math.cos(ang), math.sin(ang)
    px, py = -sa, ca
    tipx, tipy = x + ca * L, y + sa * L
    shoulder = 0.78
    sx, sy = x + ca * L * shoulder, y + sa * L * shoulder
    bl = (x + px * wd / 2, y + py * wd / 2)
    br = (x - px * wd / 2, y - py * wd / 2)
    sl = (sx + px * wd / 2 * 0.95, sy + py * wd / 2 * 0.95)
    sr = (sx - px * wd / 2 * 0.95, sy - py * wd / 2 * 0.95)
    mid_b = (x + px * wd * jit, y + py * wd * jit)
    mid_s = (sx + px * wd * jit, sy + py * wd * jit)
    return [("L", [bl, sl, mid_s, mid_b]), ("R", [mid_b, mid_s, sr, br]), ("TL", [sl, (tipx, tipy), mid_s]), ("TR", [mid_s, (tipx, tipy), sr])]


def g_crystal(seed):
    r = np.random.default_rng(seed)
    img = vgrad([(0, "#090418"), (0.5, "#140B33"), (1, "#0A0620")])
    n = fbm(seed, 2.5, 5)
    img += col("#2B1A66") * (smooth(0.35, 0.9, n) * 0.5)[..., None]
    glow(img, 720, 1500, 900, col("#5B2EFF"), 0.25, "exp")
    ss = 2
    pal = [("#22E7FF", "#0E8FD0", "#0A4C9C"), ("#FF4DD2", "#C21E9C", "#7A1275"), ("#9B6BFF", "#5E37D6", "#3A1E94"), ("#4DFFB0", "#14B97E", "#0A7A58")]
    items = []
    for i in range(40):
        side = r.random()
        if side < 0.62:
            x = r.random() * W
            ang = -math.pi / 2 + (r.random() - 0.5) * (0.9 + 0.5 * abs(x - W / 2) / (W / 2)) * (1 if x > W / 2 else 1) + (x - W / 2) / W * -0.5
            L = 360 + r.random() ** 1.4 * 900
            y = H + 20 - r.random() * 160
        elif side < 0.8:
            x = r.random() * W
            ang = math.pi / 2 + (r.random() - 0.5) * 0.9
            L = 180 + r.random() ** 1.6 * 520
            y = -20 + r.random() * 60
        else:
            x = -20 if r.random() < 0.5 else W + 20
            y = 900 + r.random() * 1200
            ang = (0.0 if x < 0 else math.pi) + (r.random() - 0.5) * 1.1
            L = 200 + r.random() ** 1.5 * 520
        items.append((x, y, ang, L, 46 + L * 0.10 * (0.7 + 0.6 * r.random()), pal[int(r.integers(0, 4))], r.random() * 0.3 - 0.15))
    items.sort(key=lambda t: t[3])
    emis = np.zeros((H, W, 3), np.float32)
    for (x, y, ang, L, wd, p, jit) in items:
        im = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        faces = crystal(d, x, y, ang, L, wd, ss, jit)
        cols_ = {"L": p[1], "R": p[2], "TL": "#FFFFFF", "TR": p[0]}
        for name, poly in faces:
            cc = col(cols_[name])
            if name == "TL":
                cc = cc * 0.2 + col(p[0]) * 0.8
            fill = tuple(int(np.clip(v, 0, 1) * 255) for v in cc) + (255,)
            d.polygon([(a * ss, b * ss) for a, b in poly], fill=fill)
        pic = im.resize((W, H), Image.BOX)
        a = np.asarray(pic, np.float32) / 255
        # internal gradient glow toward base
        al = a[..., 3:4]
        # core brightness
        dist_from_base = np.clip(((XX - x) * math.cos(ang) + (YY - y) * math.sin(ang)) / L, 0, 1)[..., None]
        c = a[..., :3] * (0.6 + 0.5 * dist_from_base)
        # approximate un-premultiplied
        c = np.where(al > 0.001, c / np.maximum(al, 1e-3), 0) * 1.0
        img = img * (1 - al) + np.clip(c, 0, 1.3) * al
        emis = np.maximum(emis, al * col(p[0]) * 0.9)
    img += blur(emis, 16) * 0.35 + blur(emis, 60) * 0.6
    fm = fbm(seed + 3, 2, 5, cy=10)
    img += (smooth(0.5, 0.9, fm) * np.clip((YY - 1500) / 1000, 0, 1) * 0.25)[..., None] * col("#7B4BFF")
    dots(img, seed + 6, 140, "#9DF4FF", 1.2, 3.2, 1.0, 1.1)
    vignette(img, 0.4)
    return img, 1.3, 0.1


def hex_cells(scale, ox=0.0, oy=0.0):
    x = (XX + ox) / scale
    y = (YY + oy) / scale
    a = np.stack([(x % 1.0) - 0.5, (y % 1.7320508) - 0.8660254], -1)
    b = np.stack([((x - 0.5) % 1.0) - 0.5, ((y - 0.8660254) % 1.7320508) - 0.8660254], -1)
    da = (a * a).sum(-1)
    db = (b * b).sum(-1)
    use_a = da < db
    gv = np.where(use_a[..., None], a, b)
    idc = np.stack([x - gv[..., 0], y - gv[..., 1]], -1)
    ax, ay = np.abs(gv[..., 0]), np.abs(gv[..., 1])
    edge = 0.5 - np.maximum(ay * 0.8660254 + ax * 0.5, ax)
    return edge, idc


def g_hexshield(seed):
    img = vgrad([(0, "#030818"), (1, "#050B22")])
    edge, idc = hex_cells(78)
    h = np.sin(idc[..., 0] * 127.1 + idc[..., 1] * 311.7) * 43758.5453
    rnd = (h - np.floor(h)).astype(np.float32)
    cx, cy = 720, 1240
    rho = np.sqrt(((XX - cx) / 720) ** 2 + ((YY - cy) / 1180) ** 2)
    dome = np.clip(1 - rho, 0, 1)
    pulse = 0.5 + 0.5 * np.sin(rho * 9 - 0.0 + rnd * 1.2)
    inten = (0.15 + 0.85 * rnd ** 3) * (0.35 + 0.65 * dome ** 0.6)
    ring = np.exp(-(((rho - 0.78) / 0.05) ** 2))
    ln = 1 - smooth(0.0, 0.03, edge)
    cells = col("#0F6FFF")
    img += cells * (inten * 0.55 * (edge > 0.0))[..., None] * 0.55
    img += (col("#27D8FF") * (ln * (0.25 + 0.9 * dome + 0.9 * ring))[..., None]) * (0.4 + 0.6 * rnd[..., None])
    img += col("#BFF4FF") * (ln * ring * 1.4)[..., None]
    # impact flash
    for (ix, iy, R) in [(520, 900, 350), (980, 1650, 260)]:
        d = np.sqrt((XX - ix) ** 2 + (YY - iy) ** 2) / R
        img += col("#FFFFFF") * (np.exp(-d * 4) * 0.6 * ln + np.exp(-d * d * 6) * 0.15)[..., None]
        img += col("#27D8FF") * (np.exp(-(((d - 0.9) / 0.1) ** 2)) * 0.5)[..., None]
    glow(img, cx, cy, 900, col("#1357D6"), 0.25, "exp")
    rr = np.sqrt((XX - cx) ** 2 + (YY - 2400) ** 2)
    img += col("#27D8FF") * (np.exp(-((YY - 2330) / 40) ** 2) * 0.25 * np.clip(1 - np.abs(XX - cx) / 900, 0, 1))[..., None]
    vignette(img, 0.35)
    return img, 1.3, 0.1


# ====================================================================== catalogue
SP, LS, RT, AB, SE, GM = "space", "landscapes", "retro", "abstract", "seasonal", "gaming"
CATEGORIES = [("space", "Space"), ("landscapes", "Landscapes"), ("retro", "Retro"), ("abstract", "Abstract"), ("seasonal", "Seasonal"), ("gaming", "Gaming")]

E = []


def item(id, name, desc, cat, pro, feat, swatch, dim, fn, **params):
    E.append(dict(id=id, name=name, desc=desc, cat=cat, pro=pro, feat=feat, swatch=swatch, dim=dim, fn=fn, params=params))


# ---- space
item("nebula-drift", "Nebula Drift", "Slow clouds of blue and violet gas drift through a deep starfield.", SP, False, True, ["#02030A", "#1B2A8A", "#7B4BD0", "#B7A1F2"], 0.15,
     "nebula", bg=("#02030A", "#070820"), stops=[(0, "#000000"), (0.4, "#0B1C6B"), (0.7, "#4C3BC0"), (0.9, "#9B7BF0"), (1, "#E6DAFF")],
     stops2=[(0, "#000000"), (0.6, "#06324D"), (1, "#2BB7D6")], cx=1.8, warp=0.6)
item("crimson-nebula", "Crimson Nebula", "A glowing red and amber stellar nursery with dark dust lanes.", SP, True, False, ["#0A0206", "#7A0F26", "#E0502F", "#FFC27A"], 0.15,
     "nebula", bg=("#070106", "#14040C"), stops=[(0, "#000000"), (0.35, "#4A0A22"), (0.62, "#B3202E"), (0.85, "#F2762C"), (1, "#FFE0A0")],
     stops2=[(0, "#000000"), (0.6, "#2A0A3E"), (1, "#7A2DA0")], cx=1.5, warp=0.7, seed_offset=7)
item("ringed-planet", "Ringed Giant", "A golden ringed gas giant glows against a violet nebula and tiny moons.", SP, True, True, ["#05071A", "#6B3FA0", "#B7814A", "#F1DDB8"], 0.12, "planet_ring")
item("gas-giant-dawn", "Gas Giant Dawn", "A huge banded planet rises over a cold violet sky with a distant moon.", SP, False, False, ["#0A0720", "#3B66B8", "#C97D58", "#E2C9A0"], 0.12, "gas_giant")
item("total-eclipse", "Total Eclipse", "A black moon blocks the sun while the pale corona and a diamond ring blaze.", SP, True, False, ["#010209", "#25327A", "#6C86E0", "#FFFFFF"], 0.1, "eclipse")
item("spiral-galaxy", "Spiral Galaxy", "A tilted spiral galaxy with a golden core, blue arms and pink star nurseries.", SP, True, True, ["#010208", "#FFD08A", "#7EA2FF", "#4D62E0"], 0.05, "galaxy")
item("warp-speed", "Warp Speed", "Light streaks rush out of a glowing blue core as the ship jumps to hyperspace.", SP, False, False, ["#01030D", "#06194A", "#1D5FC9", "#B6D6FF"], 0.2, "warp")
item("orbital-sunrise", "Orbital Sunrise", "Dawn breaks over the limb of a blue world, seen from low orbit.", SP, True, False, ["#010207", "#0A2548", "#3A87B6", "#FFD7A0"], 0.1, "orbital")
# ---- landscapes
item("alpine-dawn", "Alpine Dawn", "Layered purple ridges catch the first pink and gold light of morning.", LS, True, False, ["#2B2E6B", "#8C6A92", "#E78FA0", "#FFE2B8"], 0.1, "alpine")
item("misty-pines", "Misty Pines", "Evergreen layers fade into pale fog beneath a low morning sun.", LS, True, False, ["#10303A", "#3F6C6C", "#9CC9BB", "#F9E9B5"], 0.15, "misty")
item("dune-sea", "Dune Sea", "Wind-carved golden dunes under a hazy desert sunset.", LS, False, False, ["#29306B", "#F08F78", "#FFD08A", "#8E4B6E"], 0.1, "dunes")
item("ocean-sunset", "Ocean Sunset", "A molten sun sets over a calm sea, with a glittering path across the water.", LS, False, False, ["#1B1B5A", "#E0627F", "#FFD27F", "#101030"], 0.15, "ocean")
item("aurora-lake", "Aurora Lake", "Green and violet northern lights mirrored in a still arctic lake.", LS, True, True, ["#02060F", "#1CD7B0", "#3A7CE0", "#9B4DE0"], 0.15, "aurora")
item("rolling-hills", "Rolling Hills", "Soft green hills, wildflowers and fluffy clouds on a bright spring day.", LS, False, False, ["#4C9BEA", "#DFF1F4", "#7DBB63", "#2F7D35"], 0.25, "hills")
item("lowpoly-valley", "Low-Poly Valley", "A faceted crystal valley in cool teal and mint with a pale setting sun.", LS, True, False, ["#0E3B4A", "#2EA39A", "#9BE3C8", "#FFE9B8"], 0.15, "lowpoly",
     pal_top=[(0, "#0B3547"), (0.35, "#1E7F86"), (0.65, "#46BFAF"), (1, "#C9F4D8")], sky=[(0, "#18325E"), (0.25, "#4C6FAE"), (0.45, "#F2A4A0"), (0.55, "#FFE9B8"), (1, "#FFE9B8")], sun=(520, 880, "#FFE9B8"))
item("canyon-glow", "Canyon Glow", "Terraced red-rock mesas glow in warm dusk haze and sun rays.", LS, False, False, ["#3A2B66", "#B2567A", "#FF9A66", "#33183F"], 0.15, "canyon")
# ---- retro
item("synthwave-sun", "Synthwave Sun", "A striped neon sun sinks behind wireframe mountains and a glowing grid.", RT, True, True, ["#0A0420", "#7A1A78", "#FF3E8E", "#FFB03A"], 0.1, "synth")
item("neon-tunnel", "Neon Tunnel", "An endless gridded corridor in magenta and cyan racing to a bright vanishing point.", RT, False, False, ["#06031A", "#6E1CA8", "#FF2DB5", "#27E6FF"], 0.2, "tunnel")
item("phosphor-scope", "Phosphor Scope", "Green oscilloscope waves on a curved CRT with scanlines and a graticule.", RT, False, False, ["#010A04", "#0C3A1C", "#1CFF6A", "#2DFF7A"], 0.2, "phosphor")
item("pixel-sunset", "Pixel Sunset", "A dithered 16-bit sunset behind pixel mountains and pines.", RT, True, False, ["#1A1043", "#B02D7B", "#FF9A4A", "#FFE27A"], 0.2, "pixel")
item("vaporwave-dream", "Vaporwave Dream", "Pastel pink sun, floating wireframes and a checkered floor in dreamy haze.", RT, False, False, ["#2B1B6E", "#FF7AC6", "#8FF3FF", "#17B8F2"], 0.15, "vapor")
item("seventies-arcs", "Seventies Arcs", "Warm concentric retro rainbow arcs rising from a cream sun.", RT, False, False, ["#F6D8A8", "#E9632D", "#8E2B63", "#241454"], 0.3, "seventies")
# ---- abstract
item("gradient-mesh", "Gradient Mesh", "A smooth colour mesh blending coral, violet, teal and gold.", AB, False, False, ["#FF6F91", "#7B5CFF", "#22C1C3", "#FDBB2D"], 0.25, "mesh",
     pts=[(0.05, 0.05, "#FF6F91", 0.55), (0.95, 0.12, "#7B5CFF", 0.5), (0.2, 0.42, "#22C1C3", 0.55), (0.85, 0.5, "#FDBB2D", 0.5), (0.1, 0.8, "#4C2FD0", 0.5), (0.9, 0.88, "#FF8A5B", 0.55), (0.5, 0.64, "#B79BFF", 0.28)])
item("liquid-silk", "Liquid Silk", "Glossy satin folds in deep blue, magenta and peach with soft highlights.", AB, True, True, ["#101A5C", "#7A2E9E", "#E0508A", "#FFB38A"], 0.1, "silk",
     stops=[(0, "#0B1250"), (0.3, "#2B3BB8"), (0.55, "#9B3EBE"), (0.78, "#F0508C"), (1, "#FFB68A")])
item("bauhaus-blocks", "Bauhaus Blocks", "Bold primary shapes on a grid, inspired by early modernist posters.", AB, False, False, ["#D8381F", "#1F4E9C", "#F0B232", "#EDE3CD"], 0.35, "bauhaus")
item("art-deco-fans", "Art Deco Fans", "Gold-lined scalloped fans on deep emerald, lit by a slow golden sheen.", AB, True, False, ["#05191C", "#0F3A3C", "#C9A04A", "#F2D48A"], 0.15, "deco")
item("marble-onyx", "Onyx Marble", "Black stone shot through with fine gold veins and a polished sheen.", AB, True, False, ["#0A0A0C", "#2A2A30", "#C9A04A", "#F2D48A"], 0.1, "marble",
     stops_base=[(0, "#050506"), (0.5, "#16161B"), (1, "#2C2C34")], vein="#E5B95A", vein2="#8A6A2C", gloss=0.18)
item("marble-azure", "Azure Marble", "Pale blue-grey marble with silver and navy veining.", AB, False, False, ["#D8E3EE", "#9BB3CC", "#32507A", "#EAF1F8"], 0.25, "marble", seed_offset=3,
     stops_base=[(0, "#8CA3BF"), (0.5, "#B8CBDF"), (1, "#E1EAF4")], vein="#1E3A66", vein2="#587AA6", gloss=0.25)
item("topographic", "Topographic", "Fine contour lines trace a glowing mint landscape over deep teal.", AB, False, False, ["#042A32", "#0B5A5E", "#6FE0C4", "#C9FFF0"], 0.15, "topo",
     stops_bg=[(0, "#031C24"), (0.5, "#06454B"), (1, "#0F7A72")], line="#6FE0C4", line2="#D6FFF2")
item("isometric-cubes", "Isometric Cubes", "A tumbling field of isometric blocks shaded in violet, pink and amber.", AB, True, False, ["#1A1148", "#7B3FD0", "#FF5FA2", "#FFB347"], 0.15, "iso",
     pal=[(0, "#2D1B7A"), (0.35, "#7B3FD0"), (0.65, "#FF5FA2"), (1, "#FFB347")], bg="#0C0726")
item("halftone-dots", "Halftone Dots", "Two overlapping halftone screens in hot pink and amber on midnight blue.", AB, False, False, ["#0B0A33", "#FF3D8B", "#FFB347", "#7A2DFF"], 0.2, "halftone",
     bg=("#0B0A33", "#150B45"), c1="#FF3D8B", c2="#FFB347", c3="#FFF1D0")
item("wave-ridges", "Wave Ridges", "Stacked flowing lines form glowing ridges from violet to aqua.", AB, False, False, ["#0A0720", "#7A3CFF", "#27E6FF", "#FF4FD8"], 0.15, "waves",
     bg="#0A0720", c_a="#7A3CFF", c_b="#27E6FF")
item("glass-orbs", "Glass Orbs", "Translucent glossy spheres floating over a soft gradient.", AB, False, False, ["#1A1A6E", "#7A5CFF", "#FF8AD6", "#8AF0FF"], 0.2, "orbs",
     sky=[(0, "#161A66"), (0.5, "#5A4BC0"), (1, "#E88AC8")], orbs=[(470, 880, 300, "#8AF0FF"), (1000, 1480, 380, "#FF8AD6"), (360, 1900, 230, "#FFE08A"), (1150, 520, 140, "#C6A8FF"), (820, 2300, 150, "#8AFFC8")])
# ---- seasonal
item("snowy-pines", "Snowy Pines", "A moonlit winter forest with drifting snowflakes in layers of blue.", SE, False, False, ["#071433", "#2E6C92", "#9DC7D8", "#F4F8FF"], 0.15, "snowy")
item("sakura-dusk", "Sakura Dusk", "Cherry blossom branches against a peach and lilac sky as petals fall.", SE, True, True, ["#5B4B9A", "#FFB7B8", "#FFD9C2", "#E99AB8"], 0.15, "sakura")
item("autumn-glow", "Autumn Glow", "Golden-hour light through a red and amber forest with falling leaves.", SE, False, False, ["#5B2A3B", "#C5543A", "#F59E4E", "#FFD089"], 0.15, "autumn")
item("fireworks-night", "Fireworks Night", "Colourful fireworks burst over a harbour skyline and mirror in the water.", SE, True, False, ["#02030E", "#FFD36B", "#FF5E8E", "#6BE8FF"], 0.1, "fireworks")
item("paper-lanterns", "Paper Lanterns", "Glowing paper lanterns float up into a starry indigo night.", SE, False, False, ["#0A0F36", "#4B2A6F", "#FF8A2D", "#FFC46B"], 0.15, "lanterns")
# ---- gaming
item("arcade-neon", "Arcade Neon", "Nested neon frames and a glowing floor grid, like a cabinet marquee at night.", GM, False, False, ["#0A0420", "#FF2DB5", "#27E6FF", "#FFD23F"], 0.15, "arcade")
item("dungeon-torch", "Dungeon Torchlight", "Ancient stone walls lit by flickering torches, with embers in the air.", GM, True, False, ["#0E0A0C", "#4A3A3A", "#FF7A1F", "#FFC46B"], 0.1, "dungeon")
item("mecha-hazard", "Mecha Hazard", "Riveted armour plates cut by a yellow and black hazard band under warning lights.", GM, False, False, ["#1B2128", "#3A4450", "#F6C21B", "#FF4D1F"], 0.2, "mecha")
item("arena-flames", "Arena Flames", "Roaring flames and rising sparks fill a dark fighting arena.", GM, False, False, ["#07010A", "#8A0E0E", "#FF9A22", "#FFE27A"], 0.2, "flames")
item("anime-night-city", "Twilight City", "A dusk skyline of lit windows and glowing bokeh, in a stylised anime-film mood.", GM, True, True, ["#1A1055", "#B24AA8", "#FF7B8E", "#FFD27A"], 0.1, "anime")
item("radar-hud", "Radar HUD", "A sci-fi tactical radar with a sweeping beam and glowing teal rings.", GM, False, False, ["#031018", "#073B49", "#27F0D2", "#BFFFF2"], 0.15, "radar")
item("crystal-cave", "Crystal Cave", "Glowing cyan, magenta and violet crystal clusters in a dark cavern.", GM, True, False, ["#090418", "#22E7FF", "#FF4DD2", "#9B6BFF"], 0.1, "crystal")
item("hex-shield", "Hex Shield", "A hexagonal energy barrier pulses with blue light and impact flashes.", GM, False, False, ["#030818", "#0F6FFF", "#27D8FF", "#BFF4FF"], 0.15, "hexshield")

GENS = {
    "nebula": lambda s, p: g_nebula(s + p.pop("seed_offset", 0), **p),
    "planet_ring": lambda s, p: g_planet_ring(s),
    "gas_giant": lambda s, p: g_gas_giant(s),
    "eclipse": lambda s, p: g_eclipse(s),
    "galaxy": lambda s, p: g_galaxy(s),
    "warp": lambda s, p: g_warp(s),
    "orbital": lambda s, p: g_orbital_sunrise(s),
    "alpine": lambda s, p: g_alpine_dawn(s),
    "misty": lambda s, p: g_misty_pines(s),
    "dunes": lambda s, p: g_dune_sea(s),
    "ocean": lambda s, p: g_ocean_sunset(s),
    "aurora": lambda s, p: g_aurora_lake(s),
    "hills": lambda s, p: g_rolling_hills(s),
    "lowpoly": lambda s, p: g_lowpoly(s, **p),
    "canyon": lambda s, p: g_canyon(s),
    "synth": lambda s, p: g_synth_sun(s),
    "tunnel": lambda s, p: g_grid_tunnel(s),
    "phosphor": lambda s, p: g_phosphor(s),
    "pixel": lambda s, p: g_pixel_sunset(s),
    "vapor": lambda s, p: g_vaporwave(s),
    "seventies": lambda s, p: g_seventies(s),
    "mesh": lambda s, p: g_mesh(s, **p),
    "silk": lambda s, p: g_silk(s, **p),
    "bauhaus": lambda s, p: g_bauhaus(s),
    "deco": lambda s, p: g_deco(s),
    "marble": lambda s, p: g_marble(s + p.pop("seed_offset", 0), **p),
    "topo": lambda s, p: g_topo(s, **p),
    "iso": lambda s, p: g_iso(s, **p),
    "halftone": lambda s, p: g_halftone(s, **p),
    "waves": lambda s, p: g_waves(s, **p),
    "orbs": lambda s, p: g_orbs(s, **p),
    "snowy": lambda s, p: g_snowy_pines(s),
    "sakura": lambda s, p: g_sakura(s),
    "autumn": lambda s, p: g_autumn(s),
    "fireworks": lambda s, p: g_fireworks(s),
    "lanterns": lambda s, p: g_lanterns(s),
    "arcade": lambda s, p: g_arcade(s),
    "dungeon": lambda s, p: g_dungeon(s),
    "mecha": lambda s, p: g_mecha(s),
    "flames": lambda s, p: g_flames(s),
    "anime": lambda s, p: g_anime_city(s),
    "radar": lambda s, p: g_radar(s),
    "crystal": lambda s, p: g_crystal(s),
    "hexshield": lambda s, p: g_hexshield(s),
}


# ====================================================================== packaging
def seed_of(id):
    return int(hashlib.sha256(id.encode()).hexdigest()[:8], 16) % 100000


def render(e):
    import copy
    p = copy.deepcopy(e["params"])
    img, grain, vig = GENS[e["fn"]](seed_of(e["id"]), p)
    return finish(img, seed_of(e["id"]), grain, vig)


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


def listing(e):
    return {
        "version": 1, "categories": [e["cat"]], "tags": [], "featured": bool(e["feat"]), "publishedAt": PUBLISHED,
        "requiresPro": bool(e["pro"]), "minAppVersionCode": 0, "swatch": e["swatch"][:4],
        "name": {"en": e["name"]}, "description": {"en": e["desc"]},
    }


def wall_json(e):
    return {
        "formatVersion": 1, "id": e["id"], "version": 1, "type": "image", "dim": e["dim"], "focus": "center",
        "rights": {"creator": "EmuCore", "sourceUrl": CREATOR_URL, "licenseId": "CC0-1.0",
                   "licenseUrl": "https://creativecommons.org/publicdomain/zero/1.0/", "attribution": "Original artwork generated for EmuCore"},
    }


def webp_bytes(im, q=88, method=4):
    b = io.BytesIO()
    im.save(b, "WEBP", quality=q, method=method)
    return b.getvalue()


def det_zip(files):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for name, data, comp_ in files:
            zi = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            zi.compress_type = comp_
            zi.external_attr = 0o644 << 16
            zi.create_system = 3
            z.writestr(zi, data)
    return b.getvalue()


def phone_mock(wall):
    S = 2
    cw, ch = 720 * S, 1280 * S
    bg = wall.resize((cw // 8, ch // 8), Image.BILINEAR).filter(ImageFilter.GaussianBlur(5)).resize((cw, ch), Image.BICUBIC)
    bgarr = np.asarray(bg, np.float32) * 0.28 + np.array([6, 7, 12], np.float32)
    canvas = Image.fromarray(np.clip(bgarr, 0, 255).astype(np.uint8), "RGB").convert("RGBA")
    pw, ph = 560 * S, 1160 * S
    px, py = (cw - pw) // 2, (ch - ph) // 2
    # body
    shadow = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle([px, py + 24 * S, px + pw, py + ph + 24 * S], radius=70 * S, fill=(0, 0, 0, 150))
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(26 * S)))
    d = ImageDraw.Draw(canvas)
    d.rounded_rectangle([px, py, px + pw, py + ph], radius=70 * S, fill=(20, 22, 28, 255), outline=(70, 74, 86, 255), width=2 * S)
    bz = 12 * S
    sx, sy, sw, sh = px + bz, py + bz, pw - 2 * bz, ph - 2 * bz
    scr = wall.resize((sw, sh), Image.LANCZOS).convert("RGBA")
    # dim like the app does and add faint schematic UI
    ov = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    lum = float(np.asarray(wall.resize((32, 56)).convert("L"), np.float32).mean() / 255)
    ink = (255, 255, 255) if lum < 0.62 else (20, 22, 30)
    def A(a): return ink + (int(a * 255),)
    u = sw / 560.0
    r = lambda v: v * u
    od.rounded_rectangle([r(130), r(18), r(430), r(46)], radius=r(14), fill=(0, 0, 0, 110))
    od.rounded_rectangle([r(22), r(70), r(538), r(140)], radius=r(22), fill=A(0.14), outline=A(0.22), width=max(1, int(r(1.5))))
    od.ellipse([r(38), r(84), r(82), r(128)], fill=A(0.28))
    od.rounded_rectangle([r(98), r(90), r(260), r(104)], radius=r(7), fill=A(0.36))
    od.rounded_rectangle([r(98), r(112), r(200), r(122)], radius=r(5), fill=A(0.2))
    for i in range(4):
        x = r(32 + i * 128)
        od.rounded_rectangle([x, r(176), x + r(108), r(284)], radius=r(20), fill=A(0.12), outline=A(0.2), width=max(1, int(r(1.5))))
        od.ellipse([x + r(34), r(196), x + r(74), r(236)], fill=A(0.26))
        od.rounded_rectangle([x + r(22), r(250), x + r(86), r(260)], radius=r(5), fill=A(0.3))
    for i in range(3):
        y = r(330 + i * 92)
        od.rounded_rectangle([r(22), y, r(538), y + r(76)], radius=r(20), fill=A(0.11), outline=A(0.18), width=max(1, int(r(1.5))))
        od.rounded_rectangle([r(40), y + r(18), r(88), y + r(58)], radius=r(12), fill=A(0.26))
        od.rounded_rectangle([r(106), y + r(22), r(300), y + r(34)], radius=r(6), fill=A(0.34))
        od.rounded_rectangle([r(106), y + r(46), r(220), y + r(56)], radius=r(5), fill=A(0.18))
    od.rounded_rectangle([r(40), r(970), r(520), r(1048)], radius=r(26), fill=A(0.16), outline=A(0.26), width=max(1, int(r(1.5))))
    for i in range(4):
        cxn = r(40 + 60 + i * 120)
        od.ellipse([cxn - r(14), r(1009) - r(14), cxn + r(14), r(1009) + r(14)], fill=A(0.34 if i == 0 else 0.18))
    od.rounded_rectangle([r(200), r(1118), r(360), r(1124)], radius=r(3), fill=A(0.5))
    scr = Image.alpha_composite(Image.alpha_composite(scr, Image.new("RGBA", (sw, sh), (0, 0, 0, 38))), ov)
    mask = Image.new("L", (sw, sh), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, sw - 1, sh - 1], radius=58 * S, fill=255)
    canvas.paste(scr, (sx, sy), mask)
    d = ImageDraw.Draw(canvas)
    d.ellipse([px + pw / 2 - 11 * S, sy + 14 * S, px + pw / 2 + 11 * S, sy + 14 * S + 22 * S], fill=(10, 11, 14, 255))
    # subtle glass sheen
    sh_ = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    ImageDraw.Draw(sh_).polygon([(sx, sy), (sx + sw * 0.55, sy), (sx, sy + sh * 0.35)], fill=(255, 255, 255, 14))
    sm = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(sm).rounded_rectangle([sx, sy, sx + sw, sy + sh], radius=58 * S, fill=255)
    canvas.alpha_composite(Image.composite(sh_, Image.new("RGBA", (cw, ch), (0, 0, 0, 0)), sm))
    return canvas.convert("RGB").resize((720, 1280), Image.LANCZOS)


def build_one(args):
    try:
        return build_one_(args)
    except Exception:
        import traceback
        traceback.print_exc()
        return args[0]["id"], "FAIL", 0


def build_one_(args):
    e, script_hash, force = args
    lst = listing(e)
    ldir = ROOT / "src" / e["id"]
    lst = keep_locales(lst, ldir / "listing.json")
    ddir = ROOT / "dist" / e["id"]
    ldir.mkdir(parents=True, exist_ok=True)
    ddir.mkdir(parents=True, exist_ok=True)
    (ldir / "listing.json").write_text(json.dumps(lst, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    h = hashlib.sha256()
    h.update(json.dumps(lst, sort_keys=True).encode())
    h.update(script_hash.encode())
    h.update(json.dumps({"fn": e["fn"], "dim": e["dim"], "params": e["params"]}, sort_keys=True, default=str).encode())
    sha = h.hexdigest()
    bj = ddir / "build.json"
    if not force and bj.exists() and (ddir / "item.ecxwall").exists() and (ddir / "cover.webp").exists() and (ddir / "phone.webp").exists():
        try:
            if json.loads(bj.read_text())["sourceSha256"] == sha:
                return e["id"], "skip", (ddir / "item.ecxwall").stat().st_size
        except Exception:
            pass
    im = render(e)
    wp = webp_bytes(im, 88)
    q = 88
    while len(wp) > 3_800_000 and q > 60:
        q -= 6
        wp = webp_bytes(im, q)
    pkg = det_zip([("wallpaper.json", (json.dumps(wall_json(e), indent=2) + "\n").encode(), zipfile.ZIP_DEFLATED), ("wallpaper.webp", wp, zipfile.ZIP_STORED)])
    (ddir / "item.ecxwall").write_bytes(pkg)
    cover = im.resize((540, 960), Image.LANCZOS)
    cq = 84
    cb = webp_bytes(cover, cq)
    while len(cb) > 580_000 and cq > 50:
        cq -= 6
        cb = webp_bytes(cover, cq)
    (ddir / "cover.webp").write_bytes(cb)
    pm = phone_mock(cover)
    pq = 84
    pb = webp_bytes(pm, pq)
    while len(pb) > 580_000 and pq > 50:
        pq -= 6
        pb = webp_bytes(pm, pq)
    (ddir / "phone.webp").write_bytes(pb)
    bj.write_text(json.dumps({"sourceSha256": sha}, indent=2) + "\n", encoding="utf-8", newline="\n")
    return e["id"], "built", len(pkg)


def contact_sheet(path, ids, cols=8):
    cw, chh = 270, 480
    rows = (len(ids) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cw + 8) + 8, rows * (chh + 8) + 8), (24, 24, 28))
    for i, id in enumerate(ids):
        f = ROOT / "dist" / id / "cover.webp"
        if f.exists():
            im = Image.open(f).convert("RGB").resize((cw, chh), Image.LANCZOS)
            sheet.paste(im, (8 + (i % cols) * (cw + 8), 8 + (i // cols) * (chh + 8)))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--sheet", default="")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "categories.json").write_text(json.dumps(keep_label_locales([{"id": i, "labels": {"en": n}} for i, n in CATEGORIES], ROOT / "categories.json"), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    only = set(x for x in a.only.split(",") if x)
    todo = [(e, script_hash, a.force) for e in E if not only or e["id"] in only]
    assert len({e["id"] for e in E}) == len(E)
    for e in E:
        assert len(e["name"]) <= 40 and len(e["desc"]) <= 160, e["id"]
    if a.jobs > 1 and len(todo) > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            for id, st, sz in ex.map(build_one, todo):
                print(f"{st:6} {id} {sz // 1024} KB", flush=True)
    else:
        for t in todo:
            id, st, sz = build_one(t)
            print(f"{st:6} {id} {sz // 1024} KB", flush=True)
    if a.sheet:
        contact_sheet(a.sheet, [e["id"] for e in E])


if __name__ == "__main__":
    main()
