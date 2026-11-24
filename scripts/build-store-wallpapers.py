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


# ====================================================================== catalogue 2: procedural families (patterns, techniques)
PT, UR = "patterns", "urban"
CATEGORIES += [("patterns", "Patterns"), ("urban", "City")]
NEW_CAT_LABELS = {
    "patterns": {"ar": "الأنماط", "cs": "Vzory", "de": "Muster", "es": "Patrones", "fa": "الگوها", "fr": "Motifs", "hi": "पैटर्न", "id": "Pola", "it": "Motivi", "ja": "パターン", "ko": "패턴", "pl": "Wzory", "pt": "Padrões", "ru": "Узоры", "tr": "Desenler", "uk": "Візерунки", "zh": "图案"},
    "urban": {"ar": "المدينة", "cs": "Město", "de": "Stadt", "es": "Ciudad", "fa": "شهر", "fr": "Ville", "hi": "शहर", "id": "Kota", "it": "Città", "ja": "都市", "ko": "도시", "pl": "Miasto", "pt": "Cidade", "ru": "Город", "tr": "Şehir", "uk": "Місто", "zh": "城市"},
}

#### ---- batch A ----
def rgb8(c):
    c = col(c) if isinstance(c, str) else np.asarray(c, np.float32)
    return tuple(int(np.clip(x, 0, 1) * 255) for x in c)


def voronoi(px, py, ds=2):
    h, w = H // ds, W // ds
    X = XX[:h * ds:ds, :w * ds:ds]
    Y = YY[:h * ds:ds, :w * ds:ds]
    d1 = np.full((h, w), 1e9, np.float32)
    d2 = d1.copy()
    idx = np.zeros((h, w), np.int32)
    for i, (x, y) in enumerate(zip(px, py)):
        d = (X - x) ** 2 + (Y - y) ** 2
        m1 = d < d1
        d2 = np.where(m1, d1, np.minimum(d2, d))
        idx = np.where(m1, i, idx)
        d1 = np.where(m1, d, d1)
    up = lambda a: cv2.resize(np.sqrt(a), (W, H), interpolation=cv2.INTER_LINEAR)
    return up(d1), up(d2), cv2.resize(idx.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int32)


def g_vglass(seed, mode, pal, lead="#0A0810", n=150):
    r = np.random.default_rng(seed)
    if mode == "arch":
        step = 118
        gx = np.arange(0, W + step, step)
        gy = np.arange(0, H + step, step)
        P = [(x + (r.random() - 0.5) * step * 0.8, y + (r.random() - 0.5) * step * 0.8) for x in gx for y in gy]
    elif mode == "shards":
        P = []
        for _ in range(n):
            a = r.random() * 6.283
            rad = (r.random() ** 1.8) * 1700
            P.append((720 + math.cos(a) * rad * 0.8, 1180 + math.sin(a) * rad))
        P += [(r.random() * W, r.random() * H) for _ in range(40)]
    else:
        step = 150
        P = []
        for j, y in enumerate(np.arange(-step, H + step, step * 0.86)):
            for x in np.arange(-step, W + step, step):
                P.append((x + (step / 2 if j % 2 else 0) + (r.random() - 0.5) * step * 0.6, y + (r.random() - 0.5) * step * 0.5))
    px = [p[0] for p in P]
    py = [p[1] for p in P]
    d1, d2, idx = voronoi(px, py)
    e = d2 - d1
    rv = r.random(len(P)).astype(np.float32)
    cv = rv[idx]
    if mode == "arch":
        t = np.clip(0.55 * cv + 0.45 * fbm(seed, 3, 3), 0, 1)
        glass = ramp(t, pal)
        dd = np.sqrt((XX - 720) ** 2 + (YY - 1000) ** 2)
        lightk = 0.55 + 0.9 * np.exp(-(dd / 700) ** 2)
        glass = glass * lightk[..., None] * (0.8 + 0.4 * cv)[..., None]
        arch = (np.sqrt((XX - 720) ** 2 + (YY - 860) ** 2) < 500) | ((np.abs(XX - 720) < 500) & (YY >= 860) & (YY < 2330))
        am = smooth(0, 1, 0.5 - np.clip(0, 0, 0)) * 0 + arch.astype(np.float32)
        am = blur(am, 1.2)
        stone = fbm(seed + 2, 40, 4, cy=60)
        stone_c = (np.array([0.07, 0.065, 0.085], np.float32) + 0.05 * stone[..., None]) * (1 - 0.7 * smooth(0, 1400, np.sqrt((XX - 720) ** 2 + (YY - 1200) ** 2)))[..., None]
        ring = np.abs(np.sqrt((XX - 720) ** 2 + (YY - 1000) ** 2) - 210)
        ring2 = np.abs(np.sqrt((XX - 720) ** 2 + (YY - 1000) ** 2) - 90)
        mull = np.abs(XX - 720) < 6
        trans = np.abs(YY - 1520) < 8
        ln = 1 - smooth(5, 9, e)
        ln = np.maximum.reduce([ln, 1 - smooth(5, 9, ring), 1 - smooth(5, 9, ring2), mull.astype(np.float32) * (YY > 1210), trans.astype(np.float32)])
        ln = blur(ln.astype(np.float32), 0.7)
        img = glass * (1 - ln[..., None]) + col(lead) * ln[..., None] * 0.9
        img = img * am[..., None] + stone_c * (1 - am[..., None])
        # frame lead on arch edge
        edge = np.abs(np.where(arch, 1, 0).astype(np.float32) - 0.5)
        shaft = np.exp(-((XX - 720) / 330) ** 2) * np.clip((YY - 1400) / 1100, 0, 1) * 0.16
        img += shaft[..., None] * col(pal[2][1]) * (1 - am[..., None] * 0.2)
        return img, 1.2, 0.3
    if mode == "shards":
        yy = YY / H
        base = ramp(np.clip(yy + (cv - 0.5) * 0.45, 0, 1), pal)
        facet = (0.65 + 0.7 * cv)[..., None]
        img = base * facet
        crack = (1 - smooth(1.2, 3.2, e))
        glowc = (1 - smooth(2, 16, e)) * 0.18
        img = img * (1 - crack[..., None] * 0.9) + crack[..., None] * col("#FFF6E8") * 0.85 + glowc[..., None] * col(pal[-1][1])
        d = np.sqrt((XX - 720) ** 2 + (YY - 1180) ** 2)
        img *= (0.55 + 0.6 * np.exp(-(d / 1100) ** 2))[..., None]
        return img, 1.0, 0.25
    # ice facets
    t = fbm(seed + 8, 2.4, 4) * 0.7 + cv * 0.3
    base = ramp(t, pal)
    bev = smooth(0, 70, e)
    dirv = ((XX - np.array(px, np.float32)[idx]) * 0.6 + (YY - np.array(py, np.float32)[idx]) * 0.8) / 150.0
    light = 0.55 + 0.45 * bev + 0.28 * dirv * (1 - bev)
    img = base * light[..., None] * (0.82 + 0.3 * cv)[..., None]
    lines = 1 - smooth(1.0, 3.0, e)
    img = img * (1 - lines[..., None] * 0.7) + lines[..., None] * col("#E9FBFF") * 0.75
    sp = fbm(seed + 3, 30, 3)
    img += (smooth(0.8, 0.95, sp) * 0.25)[..., None] * col("#FFFFFF")
    return img, 1.0, 0.25


GENS["vglass"] = lambda s, p: g_vglass(s, **p)


def trace_flow(field_fn, starts, steps, step_len):
    pos = starts.astype(np.float32).copy()
    out = [pos.copy()]
    for _ in range(steps):
        a = field_fn(pos[:, 0], pos[:, 1])
        pos = pos + np.stack([np.cos(a), np.sin(a)], 1) * step_len
        out.append(pos.copy())
    return np.stack(out, 1)  # N, S, 2


def g_flow(seed, mode, bg, c1, c2, n=3500, steps=120, step_len=7.0, width=1, accent=None, cx=2.2, twist=1.0, sun=None):
    r = np.random.default_rng(seed)
    f = fbm(seed, cx, 3, h=H // 4, w=W // 4)
    f = blur(f, 2)
    sh, sw = f.shape

    def samp(x, y):
        xi = np.clip((x / 4).astype(np.int32), 0, sw - 1)
        yi = np.clip((y / 4).astype(np.int32), 0, sh - 1)
        return f[yi, xi]

    if mode == "swirl":
        cs = [(0.3 * W, 0.35 * H), (0.72 * W, 0.68 * H)]

        def fld(x, y):
            a = samp(x, y) * 2.0
            tx = 0
            ty = 0
            for k, (cx_, cy_) in enumerate(cs):
                dx, dy = x - cx_, y - cy_
                dd = np.sqrt(dx * dx + dy * dy) + 1
                w_ = np.exp(-(dd / 900) ** 2) * (1 if k == 0 else -1)
                tx = tx + (-dy / dd) * w_
                ty = ty + (dx / dd) * w_
            return np.arctan2(ty + 0.35 * np.sin(a * 3.14), tx + 0.35 * np.cos(a * 3.14))
    else:
        def fld(x, y):
            return samp(x, y) * 6.283 * twist
    starts = np.stack([r.random(n) * (W + 200) - 100, r.random(n) * (H + 200) - 100], 1)
    P = trace_flow(fld, starts, steps, step_len)
    base = np.zeros((H, W, 3), np.uint8)
    base[:] = rgb8(bg)
    if mode == "ink":
        paper = fbm(seed + 3, 120, 3, cy=170)
        pa = col(bg)[None, None, :] * (0.93 + 0.1 * paper[..., None])
        base = np.clip(pa * 255, 0, 255).astype(np.uint8)
    canvas = np.ascontiguousarray(base)
    if sun:
        cv2.circle(canvas, (int(sun[0]), int(sun[1])), int(sun[2]), rgb8(sun[3]), -1, cv2.LINE_AA)
    t = fbm(seed + 5, 1.6, 3, h=H // 4, w=W // 4)
    ac = set(r.choice(n, int(n * 0.07), replace=False).tolist()) if accent else set()
    for i in range(n):
        pts = P[i]
        ok = (pts[:, 0] > -50) & (pts[:, 0] < W + 50) & (pts[:, 1] > -50) & (pts[:, 1] < H + 50)
        if ok.sum() < 6:
            continue
        pp = (pts[ok] * 8).astype(np.int32)
        k = float(t[min(sh - 1, max(0, int(pts[ok][0, 1] / 4))), min(sw - 1, max(0, int(pts[ok][0, 0] / 4)))])
        c = col(c1) * (1 - k) + col(c2) * k
        if i in ac:
            c = col(accent)
        c = c * (0.45 + 0.55 * r.random()) if mode != "ink" else c
        cv2.polylines(canvas, [pp.reshape(-1, 1, 2)], False, rgb8(c), width + (1 if i in ac else 0), cv2.LINE_AA, 3)
    img = canvas.astype(np.float32) / 255.0
    if mode != "ink":
        L = np.clip(img - col(bg), 0, None)
        img += blur(L, 6) * 0.9 + blur(L, 22) * 0.7
        vignette(img, 0.4)
    return img, 1.0 if mode != "ink" else 0.7, 0.0


GENS["flow"] = lambda s, p: g_flow(s, **p)


def g_gscott(seed, f, k, steps, pal, mode="plain", glowc=None, bg=None, nseed=40, hh=512, ww=288):
    r = np.random.default_rng(seed)
    U = np.ones((hh, ww), np.float32)
    V = np.zeros((hh, ww), np.float32)
    for _ in range(nseed):
        y, x = int(r.integers(10, hh - 10)), int(r.integers(10, ww - 10))
        s_ = int(r.integers(3, 8))
        U[y:y + s_, x:x + s_] = 0.5
        V[y:y + s_, x:x + s_] = 0.25
    V += r.random((hh, ww)).astype(np.float32) * 0.02
    kern = np.array([[.05, .2, .05], [.2, -1, .2], [.05, .2, .05]], np.float32)
    for _ in range(steps):
        lu = cv2.filter2D(U, -1, kern)
        lv = cv2.filter2D(V, -1, kern)
        uvv = U * V * V
        U += lu - uvv + f * (1 - U)
        V += 0.5 * lv + uvv - (f + k) * V
    v = V
    lo, hi = np.percentile(v, [2, 99.5])
    v = np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)
    big = cv2.resize(v, (W, H), interpolation=cv2.INTER_CUBIC)
    big = np.clip(big, 0, 1)
    img = ramp(big, pal)
    gy, gx = np.gradient(blur(big, 2.5))
    lit = np.clip(-(gx * 0.6 + gy * 0.8) * 22, -1, 1)
    if mode == "emboss":
        img = img * (0.6 + 0.5 * lit[..., None] + 0.2 * big[..., None])
        spec = np.clip(lit - 0.5, 0, 1) ** 2
        img += spec[..., None] * 0.6
    elif mode == "glow":
        img += blur(img, 9) * 0.9 + blur(img, 30) * 0.8
    else:
        img = img * (0.8 + 0.3 * lit[..., None])
    return img, 1.1, 0.25


GENS["gscott"] = lambda s, p: g_gscott(s, **p)


def halftone_layer(val, cell, angle, ox=0.0, oy=0.0, gain=1.0):
    ca, sa = math.cos(angle), math.sin(angle)
    u = (XX - ox) * ca + (YY - oy) * sa
    v = -(XX - ox) * sa + (YY - oy) * ca
    fu = (u / cell) - np.floor(u / cell) - 0.5
    fv = (v / cell) - np.floor(v / cell) - 0.5
    d = np.sqrt(fu * fu + fv * fv)
    rad = np.sqrt(np.clip(val * gain, 0, 1)) * 0.72
    return 1 - smooth(rad - 0.04, rad + 0.04, d)


def g_popart(seed, mode, pal):
    r = np.random.default_rng(seed)
    ss = 2
    paper = col(pal[0])
    if mode == "burst":
        yel, red, blue, pink = pal[1], pal[2], pal[3], pal[4]
        cx, cy = 720, 1000
        ang = np.arctan2(YY - cy, XX - cx)
        rays = (np.floor((ang + 3.14159) / 6.28318 * 28) % 2).astype(np.float32)
        img = np.where(rays[..., None] > 0.5, col(yel), col(red)).astype(np.float32)
        dist = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
        dots = halftone_layer(smooth(200, 1700, dist), 34, 0.785)
        img = img * (1 - dots[..., None] * 0.9) + dots[..., None] * paper * 0.9
        im = Image.new("L", (W * ss, H * ss), 0)
        d = ImageDraw.Draw(im)
        pts = []
        for i in range(34):
            a = i / 34 * 6.283
            rr = 520 if i % 2 == 0 else 330
            pts.append(((cx + math.cos(a) * rr * (1 + 0.05 * r.random())) * ss, (cy + math.sin(a) * rr) * ss))
        d.polygon(pts, fill=255)
        star = np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255.0
        im2 = Image.new("L", (W * ss, H * ss), 0)
        ImageDraw.Draw(im2).polygon(pts, fill=255, outline=255, width=26 * ss)
        outl = np.asarray(im2.resize((W, H), Image.BOX), np.float32) / 255.0
        img = img * (1 - outl[..., None]) + outl[..., None] * col("#101018")
        inner = halftone_layer(smooth(cy - 520, cy + 520, YY), 26, 0.52)
        fill = paper * (1 - inner[..., None]) + col(blue) * inner[..., None]
        img = img * (1 - star[..., None]) + fill * star[..., None]
        # speech bubble
        im3 = Image.new("L", (W * ss, H * ss), 0)
        d3 = ImageDraw.Draw(im3)
        bx0, by0, bx1, by1 = 250, 1750, 1190, 2250
        d3.ellipse([bx0 * ss, by0 * ss, bx1 * ss, by1 * ss], fill=255)
        d3.polygon([(520 * ss, 2150 * ss), (400 * ss, 2400 * ss), (700 * ss, 2200 * ss)], fill=255)
        bub = np.asarray(im3.resize((W, H), Image.BOX), np.float32) / 255.0
        ol = blur(bub, 1) 
        edge = np.clip(blur(bub, 9) * 2.0, 0, 1)
        outline = np.clip(bub - (1 - np.clip(blur(bub, 12) * 1.9 - 0.45, 0, 1)) * 0, 0, 1)
        ero = 1 - np.clip(blur(1 - bub, 10) * 2.4 - 0.2, 0, 1)
        ero = np.minimum(ero, bub)
        img = img * (1 - bub[..., None]) + bub[..., None] * col("#101018")
        img = img * (1 - ero[..., None]) + ero[..., None] * paper
        ht = halftone_layer(smooth(1750, 2300, YY) * 0.8 + 0.1, 22, 0.26) * ero
        img = img * (1 - ht[..., None] * 0.7) + ht[..., None] * col(pink) * 0.7
        return img, 0.8, 0.0
    if mode == "panels":
        img = np.zeros((H, W, 3), np.float32)
        img[:] = col("#101018")
        polys = [
            [(40, 40), (1400, 40), (1400, 820), (40, 1000)],
            [(40, 1040), (1400, 860), (1400, 1760), (40, 1620)],
            [(40, 1660), (1400, 1800), (1400, 2520), (40, 2520)],
        ]
        fills = []
        # panel 1: sun + yellow dots
        for pi, poly in enumerate(polys):
            im = Image.new("L", (W * ss, H * ss), 0)
            ImageDraw.Draw(im).polygon([(x * ss, y * ss) for x, y in poly], fill=255)
            m = np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255.0
            if pi == 0:
                bgc = col(pal[1]) * np.ones((H, W, 3), np.float32)
                dd = np.sqrt((XX - 1000) ** 2 + (YY - 400) ** 2)
                dots = halftone_layer(1 - smooth(0, 800, dd), 30, 0.78)
                bgc = bgc * (1 - dots[..., None]) + dots[..., None] * col(pal[2])
                sunm = 1 - smooth(210, 216, dd)
                bgc = bgc * (1 - sunm[..., None]) + sunm[..., None] * paper
            elif pi == 1:
                bgc = col(pal[3]) * np.ones((H, W, 3), np.float32)
                ln = ((XX * 0.6 + YY * 0.8) % 46) < 14
                bgc = np.where(ln[..., None], col(pal[4]), bgc)
                bolt = Image.new("L", (W * ss, H * ss), 0)
                bp = [(760, 960), (900, 960), (800, 1280), (980, 1280), (640, 1700), (720, 1360), (580, 1360)]
                ImageDraw.Draw(bolt).polygon([(x * ss, y * ss) for x, y in bp], fill=255, outline=255)
                bm = np.asarray(bolt.resize((W, H), Image.BOX), np.float32) / 255.0
                bo = np.clip(blur(bm, 8) * 2.5, 0, 1)
                bgc = bgc * (1 - bo[..., None]) + bo[..., None] * col("#101018")
                bgc = bgc * (1 - bm[..., None]) + bm[..., None] * col(pal[1])
            else:
                dd = np.sqrt((XX - 300) ** 2 + (YY - 2300) ** 2)
                dots = halftone_layer(smooth(0, 1500, dd), 38, 0.3)
                bgc = col(pal[4]) * (1 - dots[..., None]) + dots[..., None] * col(pal[2])
                ring = np.abs(dd - 520) < 38
                bgc = np.where(ring[..., None], paper, bgc)
                ring2 = np.abs(dd - 520) < 16
                bgc = np.where(ring2[..., None], col(pal[3]), bgc)
            inner = 1 - np.clip(blur(1 - m, 7) * 2.3 - 0.05, 0, 1)
            inner = np.minimum(inner, m)
            img = img * (1 - inner[..., None]) + bgc * inner[..., None]
        return img, 0.8, 0.0
    # cmyk: halftone of a soft colour image
    base = np.zeros((H, W, 3), np.float32)
    base[:] = 1.0
    layers = []
    shapes = [((520, 800, 520), pal[1]), ((960, 1300, 470), pal[2]), ((640, 1800, 560), pal[3]), ((1080, 2150, 330), pal[1])]
    c_ = np.zeros((H, W), np.float32)
    m_ = c_.copy()
    y_ = c_.copy()
    for (x, y, rad), cc in shapes:
        d = np.sqrt((XX - x) ** 2 + (YY - y) ** 2)
        s = 1 - smooth(rad * 0.2, rad, d)
        cc = col(cc)
        c_ = np.maximum(c_, s * (1 - cc[0]))
        m_ = np.maximum(m_, s * (1 - cc[1]))
        y_ = np.maximum(y_, s * (1 - cc[2]))
    tint = fbm(seed, 1.4, 3)
    c_ = np.clip(c_ + 0.0, 0, 1)
    paper_img = np.ones((H, W, 3), np.float32) * col(pal[0])
    for chan, ang, ox in ((c_, 0.26, 0), (m_, 1.31, 6), (y_, 0.0, -5)):
        h = halftone_layer(chan, 30, ang, ox, ox)
        ink = {0: col("#00A6D6"), 1: col("#E8318A"), 2: col("#FFE000")}
        pass
    inks = [col("#12A8DC"), col("#EC2A86"), col("#FFE31A")]
    out = paper_img.copy()
    for chan, ang, ox, ink in ((y_, 0.0, -5, inks[2]), (c_, 0.26, 0, inks[0]), (m_, 1.31, 6, inks[1])):
        h = halftone_layer(chan, 30, ang, ox, ox, 1.1)
        out = out * (1 - h[..., None]) + (out * ink) * h[..., None]
    return out, 0.8, 0.0


GENS["popart"] = lambda s, p: g_popart(s, **p)


def g_seigaiha(seed, R, cols_, bgc, mode="full", sun=None, fade=None):
    ss = 2
    im = Image.new("RGB", (W * ss, H * ss), rgb8(bgc))
    d = ImageDraw.Draw(im)
    rows = int(H / (R / 2)) + 4
    top = 0 if mode == "full" else int(H * 0.42)
    if sun:
        sx, sy, sr, sc = sun
        d.ellipse([(sx - sr) * ss, (sy - sr) * ss, (sx + sr) * ss, (sy + sr) * ss], fill=rgb8(sc))
    for j in range(rows):
        y = j * R / 2
        if y + R < top:
            continue
        for i in range(-1, int(W / (2 * R)) + 3):
            x = i * 2 * R + (R if j % 2 else 0)
            t = np.clip(y / H, 0, 1)
            nrings = len(cols_)
            for k in range(nrings):
                rr = R * (1 - k / nrings)
                c = col(cols_[k % len(cols_)])
                if fade:
                    c = c * (fade[0] + (fade[1] - fade[0]) * t)
                if fade and len(fade) > 2:
                    c = c * (1 - fade[2] * t) + col(fade[3]) * fade[2] * t
                d.ellipse([(x - rr) * ss, (y - rr) * ss, (x + rr) * ss, (y + rr) * ss], fill=rgb8(c), outline=rgb8(col(bgc) * 0.5), width=ss * 1)
    img = from_pil(im)
    return img, 1.0, 0.0


GENS["seigaiha"] = lambda s, p: g_seigaiha(s, **p)


def g_constel(seed, bgstops, line, star, chart="blue", n=95, ring_c=None, nodes_c=None):
    r = np.random.default_rng(seed)
    img = vgrad(bgstops)
    ss = 2
    im = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(im)
    cx, cy = 720, 1280
    # chart rings and ticks
    if chart == "blue":
        for rr in (330, 520, 640, 660):
            d.ellipse([(cx - rr) * ss, (cy - rr) * ss, (cx + rr) * ss, (cy + rr) * ss], outline=70, width=ss * (3 if rr == 660 else 1))
        for a in range(0, 360, 5):
            ra = math.radians(a)
            l0, l1 = (660, 700) if a % 15 == 0 else (660, 680)
            d.line([((cx + math.cos(ra) * l0) * ss, (cy + math.sin(ra) * l0) * ss), ((cx + math.cos(ra) * l1) * ss, (cy + math.sin(ra) * l1) * ss)], fill=120, width=ss * 2)
        for a in range(0, 180, 30):
            ra = math.radians(a)
            d.line([((cx - math.cos(ra) * 640) * ss, (cy - math.sin(ra) * 640) * ss), ((cx + math.cos(ra) * 640) * ss, (cy + math.sin(ra) * 640) * ss)], fill=40, width=ss)
        rng = (500, 600)
        pts = []
        while len(pts) < n:
            a = r.random() * 6.283
            rr = math.sqrt(r.random()) * 600
            pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
    elif chart == "gold":
        for rr in (180, 300, 420, 540, 660):
            d.ellipse([(cx - rr) * ss, (cy - rr) * ss, (cx + rr) * ss, (cy + rr) * ss], outline=90, width=ss * (3 if rr in (180, 660) else 1))
        for a in range(0, 360, 30):
            ra = math.radians(a)
            d.line([((cx + math.cos(ra) * 180) * ss, (cy + math.sin(ra) * 180) * ss), ((cx + math.cos(ra) * 660) * ss, (cy + math.sin(ra) * 660) * ss)], fill=55, width=ss)
        pts = []
        while len(pts) < n:
            a = r.random() * 6.283
            rr = 180 + math.sqrt(r.random()) * 480
            pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
    else:
        pts = [(r.random() * (W - 160) + 80, r.random() * (H - 300) + 150) for _ in range(n)]
        for gx in range(0, W + 1, 120):
            d.line([(gx * ss, 0), (gx * ss, H * ss)], fill=18, width=ss)
        for gy in range(0, H + 1, 120):
            d.line([(0, gy * ss), (W * ss, gy * ss)], fill=18, width=ss)
    P = np.array(pts)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    lim = 230 if chart != "net" else 280
    link = set()
    for i in range(len(P)):
        order = np.argsort(D[i])[1:4]
        for j in order[:2 if chart != "net" else 3]:
            if D[i, j] < lim:
                link.add((min(i, int(j)), max(i, int(j))))
    for i, j in link:
        d.line([(P[i][0] * ss, P[i][1] * ss), (P[j][0] * ss, P[j][1] * ss)], fill=255, width=ss * 3)
    chart_m = np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255.0
    img += chart_m[..., None] * col(line) * 0.9
    img += blur(chart_m, 4)[..., None] * col(line) * 0.5
    lay = np.zeros((H, W, 3), np.float32)
    for i, (x, y) in enumerate(P):
        mag = r.random() ** 2
        glow(img, x, y, 10 + 40 * mag, col(star), 0.8, "exp")
        glow(img, x, y, 4 + 6 * mag, col(star), 1.6, "exp")
        if mag > 0.45:
            L = 50 + 90 * mag
            img[int(y), int(max(0, x - L)):int(min(W, x + L))] += col(star) * 0.5
            img[int(max(0, y - L)):int(min(H, y + L)), int(x)] += col(star) * 0.5
    stars(img, seed + 7, 700, amp=(0.1, 0.5), size=0.7)
    return img, 1.3, 0.3


GENS["constel"] = lambda s, p: g_constel(s, **p)


#### ---- batch B ----
BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0 - 0.5


def dither_pal(t, pal, h, w):
    """t in 0..1 (h,w); pal list of hex; ordered Bayer dither between palette steps."""
    P = np.array([col(c) for c in pal], np.float32)
    n = len(P)
    bay = np.tile(BAYER4, (h // 4 + 1, w // 4 + 1))[:h, :w]
    v = np.clip(t, 0, 1) * (n - 1) + bay * 0.9
    i = np.clip(np.round(v), 0, n - 1).astype(np.int32)
    return P[i]


def g_pixel(seed, scene):
    r = np.random.default_rng(seed)
    h, w = 320, 180
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)

    def rect(x0, y0, x1, y1, c):
        img[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = col(c)

    def disc_(cx, cy, rr, c, shade=None):
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr * rr
        img[m] = col(c)
        if shade:
            m2 = m & ((xx - cx) * 0.7 + (yy - cy) * 0.7 > rr * 0.35)
            img[m2] = col(shade)
        return m

    if scene == "city":
        hor = 250
        t = 1 - yy / hor
        img[:] = dither_pal(np.clip(1 - t * 1.0, 0, 1) ** 1.2 * 0 + np.clip(yy / hor, 0, 1) ** 1.5, ["#120A30", "#27114F", "#5A1E6E", "#A93A7E", "#F2686A", "#FFB06A", "#FFD99A"], h, w)
        for _ in range(110):
            x, y = int(r.integers(0, w)), int(r.integers(0, 140))
            img[y, x] = col("#FFF1D8") if r.random() < 0.4 else col("#B9A8E8")
        disc_(130, 62, 17, "#F7ECD2", "#D8C6AE")
        for cx_, cy_, rr in ((124, 56, 3), (136, 68, 2), (128, 70, 2)):
            disc_(cx_, cy_, rr, "#D2C0A6")
        layers = [("#5A2670", 150, 215, 6, 16), ("#341650", 120, 190, 8, 20), ("#150A28", 90, 150, 10, 24)]
        for li, (c, hmin, hmax, wmin, wmax) in enumerate(layers):
            x = -4
            ground = 300 - li * 12
            while x < w:
                bw = int(r.integers(wmin, wmax))
                bh = int(r.integers(hmin // 2 + 10, hmax // 2 + 30)) if li < 2 else int(r.integers(40, 120 + li * 10))
                top = ground - bh
                rect(x, top, x + bw, 320, c)
                if r.random() < 0.25 and li > 0 and x + bw < w:
                    rect(x + bw // 2, top - int(r.integers(6, 16)), x + bw // 2 + 1, top, c)
                    if li == 2:
                        img[top - 8, x + bw // 2] = col("#FF3A4A")
                if li == 2:
                    for wy in range(top + 4, 296, 5):
                        for wx in range(x + 2, x + bw - 2, 4):
                            if r.random() < 0.38:
                                img[wy, wx:wx + 2] = col("#FFD27A") if r.random() < 0.85 else col("#7AD8FF")
                x += bw + int(r.integers(0, 3))
        rect(0, 300, w, 320, "#07040F")
        rect(0, 300, w, 301, "#2A1850")
        for x in range(4, w, 18):
            rect(x, 309, x + 8, 310, "#5A4A28")
    elif scene == "harbour":
        hor = 190
        t = np.where(yy < hor, 1 - yy / hor, 0)
        img[:] = dither_pal(t ** 1.4, ["#0B2A52", "#1F5C86", "#4C97A8", "#F2A890", "#FFD39A", "#FFEEC0"][::-1][::-1], h, w)
        img[:hor] = dither_pal((1 - yy[:hor] / hor) ** 1.3, ["#0B2A52", "#1F5C86", "#5A8FB0", "#F2A890", "#FFD39A", "#FFEEC0"][::1], hor, w)[:, :] if False else img[:hor]
        sky = dither_pal(np.clip(yy[:hor] / hor, 0, 1) ** 1.3, ["#103A6E", "#2D6B98", "#6AA3B8", "#F2A890", "#FFD39A", "#FFEEC0"], hor, w)
        img[:hor] = sky
        disc_(88, hor - 6, 26, "#FFF3C8")
        img[hor:] = 0
        sea = dither_pal(np.clip((yy[hor:] - hor) / (h - hor), 0, 1) ** 0.8, ["#F0A488", "#9A6E8E", "#3D5C86", "#17345E", "#0A1C3C"], h - hor, w)
        img[hor:] = sea
        # sun reflection
        for y in range(hor, 300):
            k = (y - hor) / (300 - hor)
            half = int(30 * (1 - k * 0.6))
            if (y % 3) != 0 or r.random() < 0.15 + 0.5 * k:
                continue
            ox = int(math.sin(y * 0.7) * 3)
            img[y, 88 - half + ox:88 + half + ox] = col("#FFE6A8")
        for _ in range(180):
            y = int(r.integers(hor + 2, 316))
            x = int(r.integers(0, w))
            L = int(r.integers(3, 9 + (y - hor) // 8))
            img[y, x:x + L] = img[y, x:x + L] * 0.7 + col("#A8C8E0") * 0.3
        # lighthouse
        rect(128, 120, 168, 196, "#2A2A4A") if False else None
        rect(120, 196, 180, 206, "#1A1730")
        rect(128, 188, 172, 196, "#241E3C")
        rect(144, 120, 156, 190, "#F4EEE6")
        for y in range(128, 186, 14):
            rect(144, y, 156, y + 7, "#C83A3A")
        rect(142, 114, 158, 120, "#1A1730")
        rect(145, 107, 155, 114, "#FFE27A")
        rect(147, 100, 153, 107, "#C83A3A")
        rect(149, 96, 151, 100, "#1A1730")
        for k in range(0, 60, 1):
            xx_ = 145 - k
            img[110 - k // 14:112 + k // 14 + 1, xx_] = img[110 - k // 14:112 + k // 14 + 1, xx_] * 0.8 + col("#FFE27A") * 0.2 if xx_ >= 0 else 0
        # boat
        rect(40, 232, 62, 236, "#3A1E2E")
        rect(48, 214, 49, 232, "#2A1A2A")
        for k in range(16):
            rect(50, 216 + k, 50 + max(1, 12 - k * 0 - k // 2), 217 + k, "#F7EBD8") if False else None
        for k in range(16):
            rect(50, 217 + k, 50 + 12 - int(k * 0.5), 218 + k, "#F7EBD8")
        for bx, by in ((30, 60), (52, 48), (64, 70)):
            img[by, bx - 2] = img[by + 1, bx - 1] = img[by + 1, bx + 1] = img[by, bx + 2] = col("#1A1730")
    else:  # desert
        hor = 235
        t = np.clip(yy / hor, 0, 1)
        img[:] = dither_pal(t ** 1.7, ["#070B24", "#141A4A", "#32246A", "#7A3A7A", "#D9647A", "#FFA878"], h, w)
        for _ in range(130):
            x, y = int(r.integers(0, w)), int(r.integers(0, 170))
            img[y, x] = col("#FFF1D8") if r.random() < 0.35 else col("#9FB4F2")
        disc_(118, 70, 29, "#FBE8C8", "#E3C7A0")
        for cx_, cy_, rr in ((108, 60, 5), (126, 80, 4), (112, 82, 3), (130, 62, 2)):
            disc_(cx_, cy_, rr, "#E8D2B0")
        # mesas
        def mesa(x0, x1, top, c, step=2):
            for xx_ in range(x0, x1):
                k = min(xx_ - x0, x1 - xx_)
                y = top + max(0, 10 - k) * step // 2 * 0 + (0 if k > 6 else (6 - k) * 2)
                rect(xx_, y, xx_ + 1, 320, c)
        mesa(-6, 60, 175, "#4C2A64")
        mesa(90, 190, 190, "#4C2A64")
        mesa(20, 120, 205, "#341A4A")
        mesa(130, 200, 215, "#341A4A")
        # dunes
        prof = 255 + 14 * np.sin(np.arange(w) * 0.05 + 1.0) + 7 * np.sin(np.arange(w) * 0.13)
        for x in range(w):
            rect(x, int(prof[x]), x + 1, 320, "#1C0F2E")
        prof2 = 285 + 10 * np.sin(np.arange(w) * 0.04 + 3) + 5 * np.sin(np.arange(w) * 0.11)
        for x in range(w):
            rect(x, int(prof2[x]), x + 1, 320, "#0C0716")

        def cactus(x, base, hgt):
            c = "#05030C"
            rect(x, base - hgt, x + 5, base, c)
            rect(x + 1, base - hgt - 1, x + 4, base - hgt, c)
            ah = hgt // 3
            rect(x - 7, base - hgt + ah, x - 2, base - hgt + ah + 4, c)
            rect(x - 7, base - hgt + ah - 12, x - 4, base - hgt + ah + 4, c)
            rect(x + 8, base - hgt + ah * 2 - 6, x + 13, base - hgt + ah * 2 - 2, c)
            rect(x + 10, base - hgt + ah * 2 - 18, x + 13, base - hgt + ah * 2 - 2, c)
            rect(x - 2, base - hgt + ah, x, base - hgt + ah + 4, c)
            rect(x + 5, base - hgt + ah * 2 - 6, x + 8, base - hgt + ah * 2 - 2, c)
        cactus(40, 300, 70)
        cactus(130, 312, 46)
        cactus(90, 290, 28)
    big = np.repeat(np.repeat(img, 8, 0), 8, 1)
    # subtle CRT-ish pixel grid
    return big, 0.5, 0.0


GENS["pixscene"] = lambda s, p: g_pixel(s, **p)


def g_circuit(seed, mode, bg, trace, pad, chip="#10161C", glowc=None, pitch=34, tw=7):
    r = np.random.default_rng(seed)
    ss = 2
    L = Image.new("L", (W * ss, H * ss), 0)
    dL = ImageDraw.Draw(L)
    P = Image.new("L", (W * ss, H * ss), 0)
    dP = ImageDraw.Draw(P)
    C = Image.new("L", (W * ss, H * ss), 0)
    dC = ImageDraw.Draw(C)
    DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

    def walk(x, y, d, segs, width=tw, pad_end=True):
        pts = [(x, y)]
        for _ in range(segs):
            ln = pitch * int(r.integers(2, 9))
            dx, dy = DIRS[d]
            n = math.hypot(dx, dy)
            x += dx / n * ln if dx == 0 or dy == 0 else dx * ln * 0.7071 * 1.0
            y += dy / n * ln if dx == 0 or dy == 0 else dy * ln * 0.7071 * 1.0
            pts.append((x, y))
            d = (d + int(r.choice([-1, 1, 0, 0]))) % 8
            if d % 2 == 1 and r.random() < 0.5:
                d = (d + 1) % 8
        sp = [(a * ss, b * ss) for a, b in pts]
        dL.line(sp, fill=255, width=width * ss, joint="curve")
        for a, b in sp:
            dL.ellipse([a - width * ss / 2, b - width * ss / 2, a + width * ss / 2, b + width * ss / 2], fill=255)
        if pad_end:
            a, b = sp[-1]
            rr = width * 1.6 * ss
            dP.ellipse([a - rr, b - rr, a + rr, b + rr], fill=255)
            dP.ellipse([a - rr * 0.45, b - rr * 0.45, a + rr * 0.45, b + rr * 0.45], fill=0)

    def chipbox(x0, y0, x1, y1, pins=True):
        dC.rectangle([x0 * ss, y0 * ss, x1 * ss, y1 * ss], fill=255)
        if pins:
            for x in range(int(x0) + 14, int(x1) - 8, 22):
                dP.rectangle([x * ss, (y0 - 14) * ss, (x + 10) * ss, y0 * ss], fill=255)
                dP.rectangle([x * ss, y1 * ss, (x + 10) * ss, (y1 + 14) * ss], fill=255)
            for y in range(int(y0) + 14, int(y1) - 8, 22):
                dP.rectangle([(x0 - 14) * ss, y * ss, x0 * ss, (y + 10) * ss], fill=255)
                dP.rectangle([x1 * ss, y * ss, (x1 + 14) * ss, (y + 10) * ss], fill=255)

    if mode == "bus":
        for b in range(7):
            y0 = 140 + b * 340 + float(r.integers(-30, 30))
            segs = [(int(r.choice([0, 1, 7, 0, 0])), pitch * int(r.integers(3, 9))) for _ in range(5)]
            for k in range(9):
                x, y = 0.0, y0 + k * 26
                pts = [(x, y)]
                for d, ln in segs:
                    dx, dy = DIRS[d]
                    k_ = 0.7071 if dx and dy else 1.0
                    x += dx * ln * k_ * (1.0 if dx else 0)
                    y += dy * ln * k_ * 1.0
                    if not dx:
                        x += 0
                    pts.append((x, y))
                x += 40
                pts.append((W + 20, y))
                dL.line([(a * ss, b_ * ss) for a, b_ in pts], fill=255, width=tw * ss, joint="curve")
                a, b_ = pts[1][0], pts[1][1]
                dP.ellipse([(a - 9) * ss, (b_ - 9) * ss, (a + 9) * ss, (b_ + 9) * ss], fill=255)
                dP.ellipse([(a - 4) * ss, (b_ - 4) * ss, (a + 4) * ss, (b_ + 4) * ss], fill=0)
        for cx_, cy_ in ((1100, 700), (320, 1500), (1000, 2150)):
            chipbox(cx_ - 110, cy_ - 110, cx_ + 110, cy_ + 110)
    elif mode == "radial":
        cx_, cy_ = 720, 1280
        chipbox(cx_ - 190, cy_ - 190, cx_ + 190, cy_ + 190)
        for side in range(4):
            for k in range(14):
                off = -170 + k * 25
                dxy = [(1, 0), (0, 1), (-1, 0), (0, -1)][side]
                px_ = cx_ + dxy[0] * 215 + (off if dxy[0] == 0 else 0)
                py_ = cy_ + dxy[1] * 215 + (off if dxy[1] == 0 else 0)
                d0 = DIRS.index((dxy[0], dxy[1]))
                walk(px_, py_, d0, int(r.integers(3, 8)), 6)
    else:
        for _ in range(70):
            x = float(r.integers(0, W // pitch)) * pitch
            y = float(r.integers(0, H // pitch)) * pitch
            walk(x, y, int(r.choice([0, 2, 4, 6])), int(r.integers(3, 8)))
        for _ in range(7):
            x0 = float(r.integers(2, 9) * 150)
            y0 = float(r.integers(2, 15) * 150)
            sz = float(r.integers(70, 150))
            chipbox(x0, y0, x0 + sz * 1.3, y0 + sz)
    lm = np.asarray(L.resize((W, H), Image.BOX), np.float32) / 255.0
    pm = np.asarray(P.resize((W, H), Image.BOX), np.float32) / 255.0
    cm = np.asarray(C.resize((W, H), Image.BOX), np.float32) / 255.0
    base = fbm(seed + 3, 8, 4)
    img = np.ones((H, W, 3), np.float32) * col(bg) * (0.9 + 0.2 * base[..., None])
    sh = blur(lm, 4)
    img *= (1 - 0.4 * np.roll(sh, 5, 0)[..., None])
    img = img * (1 - lm[..., None]) + lm[..., None] * col(trace)
    gx, gy = np.gradient(blur(lm, 1.6))
    img += np.clip(-(gx * 0.7 + gy * 0.7), 0, 1)[..., None] * 1.5 * col(pad) * 0.5
    img = img * (1 - pm[..., None]) + pm[..., None] * col(pad)
    cs = blur(cm, 5)
    img *= (1 - 0.45 * np.roll(cs, 8, 0)[..., None])
    img = img * (1 - cm[..., None]) + cm[..., None] * col(chip)
    ei = np.clip(cm - blur(cm, 3), 0, 1)
    img += ei[..., None] * col(trace) * 0.4
    if glowc:
        g_ = np.clip(lm + pm, 0, 1)
        img += blur(g_, 5)[..., None] * col(glowc) * 1.0 + blur(g_, 22)[..., None] * col(glowc) * 0.6
    return img, 1.2, 0.3


GENS["circuit"] = lambda s, p: g_circuit(s, **p)


def paper_layer(img, mask, color, shadow_k=0.55, seed=0):
    sh = blur(np.roll(mask, -14, 0) * 1.0, 11)
    img *= (1 - shadow_k * sh * (1 - mask))[..., None]
    sh2 = blur(np.roll(mask, -3, 0) * 1.0, 2.5)
    img *= (1 - 0.5 * sh2 * (1 - mask))[..., None]
    fib = fbm(seed, 160, 2, cy=210)
    c = np.asarray(color, np.float32) if not isinstance(color, str) else col(color)
    layer = c * (0.94 + 0.12 * fib[..., None]) if c.ndim == 1 else c * (0.94 + 0.12 * fib[..., None])
    m = mask[..., None]
    img *= (1 - m)
    img += layer * m
    # light top edge
    e = np.clip(mask - np.roll(mask, 3, 0), 0, 1)
    img += e[..., None] * 0.12


def poly_below(y, ss=2):
    pts = [(x * ss, float(v) * ss) for x, v in zip(np.linspace(0, W, len(y)), y)] + [(W * ss, (H + 8) * ss), (0, (H + 8) * ss)]
    im = Image.new("L", (W * ss, H * ss), 0)
    ImageDraw.Draw(im).polygon(pts, fill=255)
    return np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255.0


def g_papercut(seed, mode, sky, layers, sun=None, n=7):
    r = np.random.default_rng(seed)
    img = vgrad(sky)
    if sun:
        d = np.sqrt((XX - sun[0]) ** 2 + (YY - sun[1]) ** 2)
        m = 1 - smooth(sun[2] - 1.5, sun[2] + 1.5, d)
        paper_layer(img, m, sun[3], 0.35, seed + 50)
    xs = np.linspace(0, W, 720)
    for i in range(n):
        t = i / (n - 1)
        c = ramp(np.array([t], np.float32), layers)[0]
        if mode == "mountains":
            y = profile(seed + i * 7, 760 + i * 250, 300 - i * 18, 1.4 + i * 0.35, 5, 0.5, 0.35, 720)
        elif mode == "waves":
            ph = r.random() * 6.28
            y = 900 + i * 230 + (70 + i * 14) * np.sin(xs / (260 - i * 12) + ph) + 28 * np.sin(xs / 90 + ph * 2)
        else:
            base = 1060 + i * 205
            y = np.full(720, float(base))
            x = -40.0
            while x < W + 40:
                wd = float(r.integers(40, 90)) * (1.35 - 0.12 * i)
                ht = float(r.integers(110, 250)) * (1.25 - 0.1 * i)
                tri = ht * np.clip(1 - np.abs(xs - x) / wd, 0, 1)
                steps_ = np.floor(tri / (ht / 6)) * (ht / 6)
                y = np.minimum(y, base - np.maximum(tri * 0.0 + steps_ * 0 + tri, 0) + 0)
                x += wd * float(r.uniform(0.7, 1.3))
            y = np.minimum(y, base)
        m = poly_below(y)
        cc = c * 1.0
        mist = (1 - t) * 0.0
        paper_layer(img, m, cc, 0.55, seed + 90 + i)
        if i < n - 1:
            mm = np.clip(blur(m, 20), 0, 1) * 0
    return img, 1.0, 0.25


GENS["papercut"] = lambda s, p: g_papercut(s, **p)


def g_watercolor(seed, mode, paper="#F6F0E2", washes=None, spl=None):
    r = np.random.default_rng(seed)
    pap = fbm(seed + 1, 150, 3, cy=210)
    gr = fbm(seed + 2, 500, 2, cy=700)
    img = np.ones((H, W, 3), np.float32) * col(paper) * (0.93 + 0.1 * pap[..., None])

    def put(m, color, alpha=0.8, edge=0.9):
        gran = 0.75 + 0.5 * gr
        e = np.clip(m - blur(m, 11), 0, 1)
        dens = m * 0.55 * gran + e * edge
        c = col(color)
        img[...] *= 1 - np.clip(dens, 0, 1)[..., None] * alpha * (1 - c)[None, None, :] * 1.3

    if mode == "sky":
        n1 = fbm(seed + 3, 2.2, 4)
        field = YY / H + (n1 - 0.5) * 0.25
        bands = washes
        for i, (lo, hi, color, a) in enumerate(bands):
            m = smooth(lo - 0.02, lo + 0.02, field) * (1 - smooth(hi - 0.02, hi + 0.02, field))
            m *= 0.8 + 0.2 * fbm(seed + 10 + i, 5, 3)
            put(m, color, a)
        for k in range(4):
            cx, cy = r.random() * W, r.random() * 900
            dd = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2) / (120 + r.random() * 140)
            put((1 - smooth(0.7, 1.0, dd + 0.1 * (fbm(seed + 20 + k, 4, 3) - 0.5))), washes[k % len(washes)][2], 0.5)
        # hills at the bottom
        for k in range(3):
            prof = profile(seed + 40 + k, 1700 + k * 280, 260, 1.2 + k * 0.4, 4, 0.5, 0.0, 720)
            m = poly_below(prof)
            m *= 0.9 + 0.1 * fbm(seed + 60 + k, 4, 3)
            put(m, ["#35597A", "#2C6E6A", "#1E3F5C"][k], 0.6 + 0.12 * k)
    elif mode == "blooms":
        for i, (cc, a) in enumerate(washes):
            cx, cy = r.random() * W, 200 + r.random() * (H - 400)
            rad = 260 + r.random() * 260
            n_ = fbm(seed + 20 + i, 5, 4)
            dd = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2) / rad + (n_ - 0.5) * 0.7
            m = 1 - smooth(0.82, 1.0, dd)
            put(m, cc, a)
        for i in range(70):
            cx, cy = r.random() * W, r.random() * H
            rad = 3 + r.random() ** 2 * 26
            dd = np.sqrt((XX[max(0, int(cy) - 40):int(cy) + 40, max(0, int(cx) - 40):int(cx) + 40] - cx) ** 2 + (YY[max(0, int(cy) - 40):int(cy) + 40, max(0, int(cx) - 40):int(cx) + 40] - cy) ** 2)
            m = np.zeros((H, W), np.float32)
            m[max(0, int(cy) - 40):int(cy) + 40, max(0, int(cx) - 40):int(cx) + 40][...] = (1 - smooth(rad - 1, rad + 1, dd))[: m[max(0, int(cy) - 40):int(cy) + 40, max(0, int(cx) - 40):int(cx) + 40].shape[0], : m[max(0, int(cy) - 40):int(cy) + 40, max(0, int(cx) - 40):int(cx) + 40].shape[1]]
            put(m, washes[int(r.integers(0, len(washes)))][0], 0.9)
    else:
        sx = fbm(seed + 3, 16, 3, cy=1.6)
        sx2 = fbm(seed + 4, 7, 3, cy=1.2)
        for i, (cc, a) in enumerate(washes):
            f = fbm(seed + 20 + i, 11, 3, cy=1.4)
            thr = 0.5 + 0.05 * i
            m = smooth(thr - 0.04, thr + 0.04, f) * smooth(0.25, 0.6, sx2) * (0.5 + 0.5 * smooth(0.0, 1.0, 1 - YY / H * 0.7))
            put(m, cc, a)
        sal = fbm(seed + 77, 260, 1, cy=360)
        img *= 1 + 0.0 * sal[..., None]
        sp = smooth(0.88, 0.95, fbm(seed + 78, 330, 2, cy=470))
        img = img * (1 - sp[..., None] * 0.65) + sp[..., None] * col(paper) * 0.65
    return img, 0.8, 0.12


GENS["watercolor"] = lambda s, p: g_watercolor(s, **p)


def crt_post(img, curve=0.07, scan=0.22, split=2.0, seedn=0, bleed=0.6):
    h, w = img.shape[:2]
    # barrel
    u = (XX / W - 0.5) * 2
    v = (YY / H - 0.5) * 2
    r2 = u * u * 0.6 + v * v * 0.3
    k = 1 + curve * r2
    mx = ((u * k) * 0.5 + 0.5) * W
    my = ((v * k) * 0.5 + 0.5) * H
    out = np.empty_like(img)
    for c, off in enumerate((-split, 0.0, split)):
        out[..., c] = cv2.remap(np.ascontiguousarray(img[..., c]), (mx + off).astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out = out + blur(out, 6) * bleed * 0.5
    sl = 1 - scan * (0.5 + 0.5 * np.cos(YY * (2 * math.pi / 4.0)))
    out *= sl[..., None]
    edge = np.clip(1 - np.maximum(np.abs(u * k), np.abs(v * k)), 0, 1)
    out *= (0.35 + 0.65 * smooth(0, 0.12, edge))[..., None]
    return out


def g_testcard(seed, mode, tint="#27FF9A", bg="#02100A"):
    r = np.random.default_rng(seed)
    if mode == "bars":
        img = np.zeros((H, W, 3), np.float32)
        cols_ = ["#E8E8E8", "#E8E81C", "#1CE8E8", "#1CE81C", "#E81CE8", "#E81C1C", "#1C1CE8"]
        y1 = int(H * 0.62)
        bw = W / 7
        for i, c in enumerate(cols_):
            img[:y1, int(i * bw):int((i + 1) * bw + 1)] = col(c)
        rev = ["#1C1CE8", "#101010", "#E81CE8", "#101010", "#1CE8E8", "#101010", "#E8E8E8"]
        y2 = int(H * 0.72)
        for i, c in enumerate(rev):
            img[y1:y2, int(i * bw):int((i + 1) * bw + 1)] = col(c)
        pl = ["#0A2A50", "#F4F4F4", "#3A0A6A", "#101010", "#000000", "#101010", "#202020"]
        for i, c in enumerate(pl):
            img[y2:int(H * 0.82), int(i * bw * 1.0):int((i + 1) * bw + 1)] = col(c)
        img[int(H * 0.82):] = col("#16161E")
        g = np.clip((XX / W), 0, 1)
        img[int(H * 0.82):int(H * 0.9)] = (g[int(H * 0.82):int(H * 0.9), :, None] * 0.9)
        img *= 0.9
        out = crt_post(img, 0.09, 0.28, 3.0, seed)
        n = np.random.default_rng(seed).random((H, W, 1), dtype=np.float32) * 0.06
        return out + n, 1.0, 0.2
    if mode == "snow":
        r2 = np.random.default_rng(seed)
        n = r2.random((H // 2, W // 2), dtype=np.float32)
        n = cv2.resize(n, (W, H), interpolation=cv2.INTER_NEAREST)
        n2 = blur(n, 1.2)
        g = (n * 0.6 + n2 * 0.6)
        rowj = np.repeat(np.random.default_rng(seed + 1).random(H // 3 + 1, dtype=np.float32), 3)[:H]
        band = np.exp(-(((YY - 1500) / 260) ** 2)) * 0.5 + np.exp(-(((YY - 420) / 30) ** 2)) * 0.35
        lum = np.clip(g * (0.65 + 0.5 * rowj[:, None]) + band, 0, 1) ** 1.3
        img = lum[..., None] * (col(tint) * 0.7 + 0.3)
        sh = np.floor(YY / 80)
        out = crt_post(img.astype(np.float32), 0.11, 0.3, 2.5, seed, 0.3)
        return out, 0.6, 0.15
    # amber monitor geometry pattern
    img = np.zeros((H, W, 3), np.float32)
    img[:] = col(bg)
    ss = 2
    im = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(im)
    for x in range(0, W + 1, 120):
        d.line([(x * ss, 0), (x * ss, H * ss)], fill=110, width=ss * 2)
    for y in range(0, H + 1, 120):
        d.line([(0, y * ss), (W * ss, y * ss)], fill=110, width=ss * 2)
    cx, cy = 720, 1280
    for rr in (120, 240, 360, 480, 600, 700):
        d.ellipse([(cx - rr) * ss, (cy - rr) * ss, (cx + rr) * ss, (cy + rr) * ss], outline=220, width=ss * 3)
    d.line([(0, 0), (W * ss, H * ss)], fill=170, width=ss * 2)
    d.line([(W * ss, 0), (0, H * ss)], fill=170, width=ss * 2)
    d.rectangle([60 * ss, 60 * ss, (W - 60) * ss, (H - 60) * ss], outline=255, width=ss * 6)
    for k in range(12):
        d.rectangle([(150 + k * 100) * ss, 120 * ss, (150 + k * 100 + (k + 1) * 4) * ss, 260 * ss], fill=230)
        d.rectangle([(150 + k * 100) * ss, (H - 260) * ss, (150 + k * 100 + (k + 1) * 4) * ss, (H - 120) * ss], fill=230)
    m = np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255.0
    img += m[..., None] * col(tint) + blur(m, 6)[..., None] * col(tint) * 0.8
    out = crt_post(img, 0.12, 0.35, 1.5, seed, 0.9)
    return out, 1.0, 0.2


GENS["testcard"] = lambda s, p: g_testcard(s, **p)


def g_mosaic(seed, mode, pal, field="radial", s=52, grout="#14121A"):
    r = np.random.default_rng(seed)
    if mode == "square":
        gh, gw = H // s + 1, W // s + 1
        cy, cx = np.mgrid[0:gh, 0:gw].astype(np.float32)
        px, py = (cx + 0.5) * s, (cy + 0.5) * s
        if field == "radial":
            t = np.sqrt((px - 720) ** 2 + (py - 1000) ** 2) / 1300
            t = (t * 5) % 1.0 * 0.6 + t * 0.4
        elif field == "wave":
            t = 0.5 + 0.5 * np.sin(px / 170 + np.sin(py / 260) * 2.2) * np.cos(py / 400)
        else:
            t = fbm(seed, 2.2, 4, h=gh, w=gw)
        t = np.clip(t + (r.random((gh, gw)) - 0.5) * 0.1, 0, 1)
        cols_ = ramp(t, pal)
        big = cv2.resize(cols_, (gw * s, gh * s), interpolation=cv2.INTER_NEAREST)[:H, :W]
        fx = (XX % s) / s
        fy = (YY % s) / s
        gl = 1 - smooth(0.0, 0.07, np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy)))
        bev = 0.88 + 0.22 * (1 - fx) * (1 - fy) + 0.0
        bev = (0.85 + 0.3 * (0.5 * (1 - fx) + 0.5 * (1 - fy)))
        img = big * bev[..., None]
        img = img * (1 - gl[..., None]) + col(grout) * gl[..., None]
        return img, 1.2, 0.2
    if mode == "hex":
        a = s * 1.0
        hh_ = a * math.sqrt(3)
        q = XX / (a * 1.5)
        rr = YY / hh_
        # axial rounding
        cxh = XX / a
        cyh = YY / a
        # hex grid via two offset rect lattices
        def cell(ox, oy):
            gx = np.floor((XX - ox) / (a * 3)) * 3 * a + ox + a * 1.5
            gy = np.floor((YY - oy) / hh_) * hh_ + oy + hh_ / 2
            return gx, gy
        g1x, g1y = cell(0, 0)
        g2x, g2y = cell(a * 1.5, hh_ / 2)
        d1 = (XX - g1x) ** 2 + (YY - g1y) ** 2
        d2 = (XX - g2x) ** 2 + (YY - g2y) ** 2
        use1 = d1 < d2
        cxs = np.where(use1, g1x, g2x)
        cys = np.where(use1, g1y, g2y)
        dist = np.minimum(d1, d2)
        t = np.clip(0.5 * (cxs / W) + 0.5 * (cys / H) + (fbm(seed, 3, 4)[np.clip(cys.astype(int), 0, H - 1), np.clip(cxs.astype(int), 0, W - 1)] - 0.5) * 0.5, 0, 1)
        hs = np.sin(cxs * 12.9898 + cys * 78.233) * 43758.5453
        jit = hs - np.floor(hs)
        t = np.clip(t + (jit - 0.5) * 0.12, 0, 1)
        img = ramp(t, pal)
        # hex edge distance: approximate using dist to center normalised
        dn = np.sqrt(dist) / (a * 0.866)
        edge = smooth(0.84, 0.97, dn)
        img = img * (0.82 + 0.3 * (1 - dn))[..., None]
        img = img * (1 - edge[..., None]) + col(grout) * edge[..., None]
        return img, 1.2, 0.2
    # pebbles
    st = 62
    P = [(x + (r.random() - 0.5) * st * 0.55, y + (r.random() - 0.5) * st * 0.55) for y in np.arange(-st, H + st, st) for x in np.arange(-st, W + st, st)]
    n = len(P)
    d1, d2, idx = voronoi([p[0] for p in P], [p[1] for p in P], ds=3)
    e = d2 - d1
    cv = r.random(n).astype(np.float32)
    t = np.clip(cv[idx] * 0.55 + fbm(seed + 2, 2.4, 4) * 0.45, 0, 1)
    img = ramp(t, pal) * (0.8 + 0.4 * cv[idx])[..., None]
    shade = 0.7 + 0.45 * smooth(0, 22, e) ** 0.8
    img = img * shade[..., None]
    gl = 1 - smooth(5, 11, e)
    img = img * (1 - gl[..., None]) + col(grout) * gl[..., None]
    return img, 1.4, 0.3


GENS["mosaic"] = lambda s, p: g_mosaic(s, **p)


#### ---- batch C ----
def g_paperfold(seed, mode, pal, bg=None, pw=100, ang=0.0, amp=0.0):
    r = np.random.default_rng(seed)
    if mode == "pleat":
        ca, sa = math.cos(ang), math.sin(ang)
        wob = amp * np.sin(YY / 330.0 + 0.8) + amp * 0.5 * np.sin(YY / 97.0)
        u = XX * ca + YY * sa + wob
        ph = u / pw
        fr = ph - np.floor(ph)
        lit = fr < 0.5
        tri = np.where(lit, fr * 2, (fr - 0.5) * 2)
        base = ramp(np.clip(0.55 * (YY / H) + 0.45 * fbm(seed, 1.6, 3), 0, 1), pal)
        sh = np.where(lit, 1.12 - 0.22 * tri, 0.62 + 0.2 * tri)
        img = base * sh[..., None]
        crease = 1 - smooth(0.0, 0.035, np.minimum(np.abs(fr - 0.5), np.minimum(fr, 1 - fr)))
        img *= (1 - 0.35 * crease)[..., None]
        img += (crease * 0.0)[..., None]
        return img, 1.0, 0.2
    if mode == "miura":
        cw, rh = pw, pw * 1.15
        zz = amp * np.clip(YY / H, 0.15, 1) * 1.0
        row = np.floor(YY / rh)
        fyr = YY / rh - row
        tri = np.abs(2 * ((YY / rh * 0.5) - np.floor(YY / rh * 0.5)) - 1)
        xs = XX + (tri * 2 - 1) * zz * cw
        cidx = np.floor(xs / cw)
        fcx = xs / cw - cidx
        par = ((cidx + row) % 2)
        facing = np.where(((YY / rh * 0.5 - np.floor(YY / rh * 0.5)) < 0.5), 1, -1)
        base = ramp(np.clip(0.5 * (XX / W) + 0.5 * (YY / H), 0, 1), pal)
        v = np.where(par > 0.5, 1.15, 0.7) * (1 + 0.1 * facing * (1 - 2 * (cidx % 2)))
        v = v * (0.9 + 0.2 * fcx)
        img = base * v[..., None]
        cr = 1 - smooth(0.0, 0.05, np.minimum(fcx, 1 - fcx))
        cr2 = 1 - smooth(0.0, 0.05, np.minimum(fyr, 1 - fyr))
        img *= (1 - 0.4 * np.maximum(cr, cr2))[..., None]
        return img, 1.0, 0.25
    # crumple
    h = np.zeros((H, W), np.float32)
    a = 1.0
    for o in range(5):
        n = fbm(seed + o * 3, 3.0 * 2 ** o, 1)
        h += (1 - np.abs(2 * n - 1)) * a
        a *= 0.5
    h = blur(h, 1.6)
    gy, gx = np.gradient(h)
    L = np.clip(0.62 - (gx * 0.8 + gy * 0.6) * 55, 0, 1.6)
    base = ramp(np.clip(0.4 * (YY / H) + 0.6 * fbm(seed + 9, 1.3, 3), 0, 1), pal)
    img = base * (0.35 + 0.8 * L[..., None])
    img *= (0.85 + 0.25 * h / h.max())[..., None]
    fine = fbm(seed + 12, 300, 2, cy=420)
    img *= (0.96 + 0.08 * fine)[..., None]
    return img, 0.8, 0.2


GENS["paperfold"] = lambda s, p: g_paperfold(s, **p)


def g_glitch(seed, mode, pal, nslice=70, maxshift=260, split=14):
    r = np.random.default_rng(seed)
    if mode == "sun":
        img = vgrad([(0, pal[0]), (0.55, pal[1]), (1, pal[2])])
        d = np.sqrt((XX - 720) ** 2 + (YY - 1000) ** 2)
        s = 1 - smooth(398, 402, d)
        sc = ramp(np.clip((YY - 600) / 800, 0, 1), [(0, pal[3]), (1, pal[4])])
        gap = ((YY - 700) % 74) < np.clip((YY - 700) / 12, 0, 40)
        s = s * np.where(gap & (YY > 760), 0.0, 1.0)
        img = img * (1 - s[..., None]) + sc * s[..., None]
        for k in range(14):
            y0 = 1500 + k * 70
            img[y0:y0 + 4] = col(pal[4]) * (0.3 + 0.03 * k)
        bars = ((XX // 60) % 2 == 0) & (YY > 1550)
        img[bars & (YY > 1550)] *= 0.6
    elif mode == "blocks":
        img = vgrad([(0, pal[0]), (1, pal[1])])
        for k in range(26):
            x0, y0 = r.integers(0, W - 200), r.integers(0, H - 100)
            w_, h_ = r.integers(120, 700), r.integers(40, 300)
            c = col(pal[2 + k % (len(pal) - 2)])
            img[y0:y0 + h_, x0:x0 + w_] = c * (0.35 + 0.65 * r.random())
        ring = np.abs(np.sqrt((XX - 720) ** 2 + (YY - 1180) ** 2) - 420) < 26
        img[ring] = col(pal[2])
    else:
        n = fbm(seed, 2.0, 4)
        n2 = fbm(seed + 4, 3.0, 3)
        img = ramp(n, pal) * (0.6 + 0.6 * n2[..., None])
    img = np.ascontiguousarray(img)
    if mode == "sort":
        lum = img.sum(2)
        out = img.copy()
        yy0 = 0
        while yy0 < H:
            hh = int(r.integers(30, 260))
            if r.random() < 0.8:
                lo, hi = float(r.uniform(0.25, 0.9)), float(r.uniform(1.3, 2.4))
                for y in range(yy0, min(H, yy0 + hh)):
                    m = (lum[y] > lo) & (lum[y] < hi)
                    idx = np.flatnonzero(m)
                    if idx.size < 8:
                        continue
                    brk = np.flatnonzero(np.diff(idx) > 1)
                    st = np.concatenate([[0], brk + 1])
                    en = np.concatenate([brk + 1, [idx.size]])
                    for a_, b_ in zip(st, en):
                        seg = idx[a_:b_]
                        if seg.size < 6:
                            continue
                        o = np.argsort(lum[y, seg])
                        out[y, seg] = img[y, seg[o]]
            yy0 += hh
        img = out
    else:
        for _ in range(nslice):
            y0 = int(r.integers(0, H - 10))
            hh = int(r.choice([6, 12, 24, 40, 90, 160]))
            dx = int(r.integers(-maxshift, maxshift))
            img[y0:y0 + hh] = np.roll(img[y0:y0 + hh], dx, axis=1)
            if r.random() < 0.45:
                ch = int(r.integers(0, 3))
                img[y0:y0 + hh, :, ch] = np.roll(img[y0:y0 + hh, :, ch], int(r.integers(-split * 3, split * 3)), axis=1)
        for _ in range(int(nslice * 0.35)):
            x0, y0 = int(r.integers(0, W - 400)), int(r.integers(0, H - 200))
            w_, h_ = int(r.integers(80, 520)), int(r.integers(16, 140))
            sx, sy = int(r.integers(0, W - w_)), int(r.integers(0, H - h_))
            blk = img[sy:sy + h_, sx:sx + w_].copy()
            if r.random() < 0.3:
                blk = 1 - blk
            img[y0:y0 + h_, x0:x0 + w_] = blk[:min(h_, H - y0), :min(w_, W - x0)]
    out = img.copy()
    out[..., 0] = np.roll(img[..., 0], split, axis=1)
    out[..., 2] = np.roll(img[..., 2], -split, axis=1)
    out *= (1 - 0.18 * ((YY % 6) < 2))[..., None]
    return out, 1.2, 0.15


GENS["glitch"] = lambda s, p: g_glitch(s, **p)


def g_isocity(seed, bgstops, top, left, right, win, win_p=0.35, mode="rand", road=None, tw=120, haze=None, floor_h=26):
    r = np.random.default_rng(seed)
    ss = 2
    img0 = vgrad(bgstops)
    im = Image.fromarray((np.clip(img0, 0, 1) * 255).astype(np.uint8)).resize((W * ss, H * ss), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    th = tw / 2
    cx0, cy0 = W / 2, 360
    nh = fbm(seed + 1, 3.0, 3, h=80, w=80)
    N = 76

    def P(u, v, z=0.0):
        return ((cx0 + (u - v) * tw / 2) * ss, (cy0 + (u + v) * th / 2 - z) * ss)

    cells = []
    for s_ in range(0, 2 * N):
        for u in range(0, N):
            v = s_ - u
            if v < 0 or v >= N:
                continue
            cells.append((u, v))
    for (u, v) in cells:
        sxc = cx0 + (u - v) * tw / 2
        syc = cy0 + (u + v) * th / 2
        if sxc < -tw * 2 or sxc > W + tw * 2 or syc < 100 or syc > H + 700:
            continue
        if u % 4 == 3 or v % 4 == 3:
            if road and u % 4 == 3 and v % 4 != 3 and ((u + v) % 3 == 0):
                pass
            if road:
                # road tile with lane marking glow
                pts = [P(u, v), P(u + 1, v), P(u + 1, v + 1), P(u, v + 1)]
                d.polygon(pts, fill=rgb8(road[0]))
                if u % 4 == 3 and v % 4 != 3:
                    d.line([P(u + 0.5, v), P(u + 0.5, v + 1)], fill=rgb8(road[1]), width=3 * ss)
                elif v % 4 == 3 and u % 4 != 3:
                    d.line([P(u, v + 0.5), P(u + 1, v + 0.5)], fill=rgb8(road[1]), width=3 * ss)
            continue
        gx = (sxc / W)
        gyv = (syc / H)
        noise_v = float(nh[min(79, int(max(0, gyv) * 79)), min(79, int(max(0, min(1, gx)) * 79))])
        if mode == "rand":
            hgt = 0.5 + noise_v ** 2 * 11 * float(r.uniform(0.5, 1.2))
        elif mode == "center":
            dd = math.hypot((sxc - W / 2) / 650, (syc - 1250) / 900)
            hgt = 0.5 + 15 * math.exp(-dd * dd * 1.3) * float(r.uniform(0.55, 1.15)) + float(r.random()) * 1.2
        else:
            hgt = 0.6 + float(r.random() ** 3) * 8 + (3 if (u // 4 + v // 4) % 5 == 0 else 0)
        z = hgt * floor_h * 2.2
        s0 = 0.14
        s1 = 0.86
        A, B, C, D = (u + s0, v + s0), (u + s1, v + s0), (u + s1, v + s1), (u + s0, v + s1)
        haze_k = 0.0
        if haze:
            haze_k = float(np.clip((1 - syc / H) * haze[1], 0, 1))
        def mix(c):
            c = col(c)
            if haze:
                c = c * (1 - haze_k) + col(haze[0]) * haze_k
            return rgb8(c)
        tc, lc, rc = mix(top), mix(left), mix(right)
        # faces
        d.polygon([P(*D), P(*C), P(C[0], C[1], z), P(D[0], D[1], z)], fill=lc)
        d.polygon([P(*C), P(*B), P(B[0], B[1], z), P(C[0], C[1], z)], fill=rc)
        d.polygon([P(A[0], A[1], z), P(B[0], B[1], z), P(C[0], C[1], z), P(D[0], D[1], z)], fill=tc)
        # windows
        nf = int(z / floor_h) - 1
        for fl in range(1, nf):
            zz = fl * floor_h
            for face in (0, 1):
                for c_ in range(3):
                    if r.random() > win_p:
                        continue
                    t0 = 0.12 + c_ * 0.29
                    t1 = t0 + 0.17
                    if face == 0:
                        p0, p1 = (D[0] + (C[0] - D[0]) * t0, D[1] + (C[1] - D[1]) * t0), (D[0] + (C[0] - D[0]) * t1, D[1] + (C[1] - D[1]) * t1)
                    else:
                        p0, p1 = (C[0] + (B[0] - C[0]) * t0, C[1] + (B[1] - C[1]) * t0), (C[0] + (B[0] - C[0]) * t1, C[1] + (B[1] - C[1]) * t1)
                    wc = win[int(r.integers(0, len(win)))]
                    if haze:
                        wc = rgb8(col(wc) * (1 - haze_k * 0.7) + col(haze[0]) * haze_k * 0.7)
                    else:
                        wc = rgb8(wc)
                    d.polygon([P(p0[0], p0[1], zz + 4), P(p1[0], p1[1], zz + 4), P(p1[0], p1[1], zz + floor_h * 0.65), P(p0[0], p0[1], zz + floor_h * 0.65)], fill=wc)
    img = np.asarray(im.resize((W, H), Image.LANCZOS), np.float32) / 255.0
    return img, 1.2, 0.25


GENS["isocity"] = lambda s, p: g_isocity(s, **p)


def g_metaballs(seed, mode, pal, bg, n=7, rmin=170, rmax=380):
    r = np.random.default_rng(seed)
    F = np.zeros((H, W), np.float32)
    ys = np.sort(r.random(n))
    balls = []
    for i in range(n):
        rad = float(r.uniform(rmin, rmax))
        x, y = float(r.uniform(0.2, 0.8) * W), float(80 + (0.05 + 0.9 * ys[i]) * (H - 160))
        balls.append((x, y, rad))
    if mode == "bubbles":
        balls = [(float(r.uniform(0, W)), float(r.uniform(0, H)), float(r.uniform(60, 190))) for _ in range(34)]
    for x, y, rad in balls:
        F += (rad * rad) / ((XX - x) ** 2 + (YY - y) ** 2 + 1.0)
    inside = smooth(0.96, 1.04, F)
    if mode == "lava":
        img = vgrad(bg)
        t = np.clip((F - 1) * 0.9, 0, 1)
        c = ramp(0.25 + 0.75 * (1 - np.exp(-(F - 1) * 1.2)) + 0.0 * t, pal)
        c = c * (0.8 + 0.4 * (YY / H))[..., None]
        img = img * (1 - inside[..., None]) + c * inside[..., None]
        img += blur(inside, 40)[..., None] * col(pal[2][1]) * 0.35
        return img, 1.1, 0.25
    if mode == "chrome":
        h = blur(np.clip(F, 0, 3.0), 3.0)
        gy, gx = np.gradient(h)
        nz = 1.0 / np.sqrt(1 + (gx * 90) ** 2 + (gy * 90) ** 2)
        nx, ny = -gx * 90 * nz, -gy * 90 * nz
        refl = 0.5 + 0.5 * ny * 1.3 + 0.18 * nx
        c = ramp(np.clip(refl + 0.2 * (1 - nz), 0, 1), pal)
        spec = np.clip(nx * -0.5 + ny * -0.7 + 0.2, 0, 1) ** 12 * (1 - nz) * 6
        c = c + spec[..., None] * 0.7
        bgimg = vgrad(bg)
        edge = smooth(0.96, 1.05, F)
        img = bgimg * (1 - edge[..., None]) + c * edge[..., None]
        return img, 1.0, 0.25
    # iridescent bubbles
    img = vgrad(bg)
    rim = 0.0
    for x, y, rad in balls:
        d = np.sqrt((XX - x) ** 2 + (YY - y) ** 2) / rad
        m = d < 1.05
        if not m.any():
            continue
    dd = np.sqrt(1.0 / np.maximum(F, 1e-3))
    f = np.clip(F, 0, 4)
    film = ramp((np.sin(f * 7.0 + (XX + YY) / 500.0) * 0.5 + 0.5), pal)
    body = smooth(0.97, 1.03, F)
    edge = np.exp(-((F - 1.0) / 0.22) ** 2)
    img = img * (1 - body[..., None] * 0.35) + body[..., None] * film * 0.28 + edge[..., None] * film * 0.9
    gy, gx = np.gradient(blur(f, 4))
    hl = np.clip(-(gx * 0.6 + gy * 0.8) * 40, 0, 1) ** 2 * body * 0.8
    img += hl[..., None]
    return img, 1.0, 0.3


GENS["metaballs"] = lambda s, p: g_metaballs(s, **p)


def g_guilloche(seed, mode, bg, c1, c2, n=140):
    img = np.zeros((H, W, 3), np.float32)
    img[:] = col(bg)
    canvas = np.zeros((H, W, 3), np.uint8)
    cx, cy = W / 2, H / 2
    if mode == "rosette":
        t = np.linspace(0, 2 * math.pi, 3000)
        for i in range(n):
            ph = i / n * 2 * math.pi
            for (R0, A, k_, sc) in ((500, 120, 13, 1.0), (250, 70, 9, 1.0)):
                rho = R0 + A * np.sin(k_ * t + ph * (k_ / 13.0) * 2.0) * np.cos(2 * t + ph * 0.5)
                pts = np.stack([cx + rho * np.cos(t), cy + rho * np.sin(t)], 1) * 8
                c = col(c1) * (1 - i / n) + col(c2) * (i / n)
                cv2.polylines(canvas, [pts.astype(np.int32).reshape(-1, 1, 2)], True, rgb8(c * 0.8), 1, cv2.LINE_AA, 3)
        for rr_ in (640, 90):
            cv2.circle(canvas, (int(cx * 8), int(cy * 8)), int(rr_ * 8), rgb8(c2), 2, cv2.LINE_AA, 3)
    elif mode == "spiro":
        t = np.linspace(0, 2 * math.pi * 20, 12000)
        specs = [(500, 173, 260, 0.0), (390, 115, 200, 0.4), (280, 53, 150, 0.8), (170, 43, 90, 1.0)]
        cs = [(cx, cy - 560), (cx, cy), (cx, cy + 560)]
        for ci, (ox, oy) in enumerate(cs):
            for j, (R, rr, dd, tt) in enumerate(specs):
                sc = 0.55 + 0.0 * ci
                x = (R - rr) * np.cos(t) + dd * np.cos((R - rr) / rr * t)
                y = (R - rr) * np.sin(t) - dd * np.sin((R - rr) / rr * t)
                pts = np.stack([ox + x * sc, oy + y * sc], 1) * 8
                c = col(c1) * (1 - (tt + ci * 0.3) % 1.0) + col(c2) * ((tt + ci * 0.3) % 1.0)
                cv2.polylines(canvas, [pts.astype(np.int32).reshape(-1, 1, 2)], False, rgb8(c), 1, cv2.LINE_AA, 3)
    else:
        r = np.random.default_rng(seed)
        t = np.linspace(0, 150, 60000)
        for i in range(4):
            f1, f2, f3, f4 = 2 + r.random() * 0.02, 3.0 + r.random() * 0.02, 3.0 + r.random() * 0.02, 2.0 + r.random() * 0.02
            dec = np.exp(-t * 0.012)
            x = cx + 450 * dec * (np.sin(f1 * t + r.random() * 6) + 0.5 * np.sin(f2 * t + r.random() * 6))
            y = cy + 700 * dec * (np.sin(f3 * t + r.random() * 6) + 0.5 * np.sin(f4 * t + r.random() * 6)) * 0.9
            seg = 600
            for s0 in range(0, len(t) - seg, seg - 1):
                k = s0 / len(t)
                c = (col(c1) * (1 - k) + col(c2) * k) * (0.5 + 0.5 * (1 - k))
                pts = np.stack([x[s0:s0 + seg], y[s0:s0 + seg]], 1) * 8
                cv2.polylines(canvas, [pts.astype(np.int32).reshape(-1, 1, 2)], False, rgb8(c), 1, cv2.LINE_AA, 3)
    L = canvas.astype(np.float32) / 255.0
    img = img + L * 1.1 + blur(L, 3) * 0.6 + blur(L, 14) * 0.5
    return img, 1.0, 0.3


GENS["guilloche"] = lambda s, p: g_guilloche(s, **p)


def g_kaleido(seed, folds, pal, bands=5.0, zoom=1.0, cxo=0.5, cyo=0.4, bg="#050510", contrast=1.4):
    fld = fbm(seed, 3.0, 5, h=1400, w=1400)
    dx, dy = XX - W / 2, YY - H / 2
    a = np.arctan2(dy, dx)
    rr = np.sqrt(dx * dx + dy * dy)
    sec = 2 * math.pi / folds
    a2 = np.abs((a % sec) - sec / 2)
    u = rr * np.cos(a2) * zoom * 0.7 + 700 * (cxo * 2 - 1) * 0 + 400
    v = rr * np.sin(a2) * zoom * 0.7 + 400
    f = cv2.remap(fld, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    t = (f * bands) % 1.0
    t2 = np.abs(2 * t - 1)
    c = ramp(np.clip(f * 1.1 + 0.0, 0, 1), pal)
    c = c * (0.45 + 0.7 * smooth(0, 1, t2) ** contrast)[..., None]
    edge = 1 - smooth(0.0, 0.06, np.abs(t2 - 0.5) * 0 + np.minimum(t, 1 - t))
    c += edge[..., None] * col(pal[-1][1]) * 0.35
    env = np.exp(-((rr / 1400) ** 2) * 1.3)
    img = c * env[..., None] + col(bg) * (1 - env[..., None])
    glow(img, W / 2, H / 2, 300, col(pal[-1][1]), 0.25, "exp")
    return img, 1.0, 0.25


GENS["kaleido"] = lambda s, p: g_kaleido(s, **p)


def g_moire(seed, mode, pal, p1=14.0, p2=14.5, bg="#08060F"):
    if mode == "rings":
        d1 = np.sqrt((XX - 470) ** 2 + (YY - 900) ** 2)
        d2 = np.sqrt((XX - 1010) ** 2 + (YY - 1700) ** 2)
        v = 0.5 + 0.25 * np.cos(d1 * 2 * math.pi / p1) + 0.25 * np.cos(d2 * 2 * math.pi / p2)
        v = smooth(0.18, 0.98, v)
    elif mode == "lines":
        a1, a2 = 0.0, 0.035
        u1 = (XX * math.cos(a1) + YY * math.sin(a1)) + 90 * np.sin(YY / 420)
        u2 = (XX * math.cos(a2) + YY * math.sin(a2)) + 90 * np.sin(YY / 420 + 0.7)
        v = 0.5 + 0.25 * np.cos(u1 * 2 * math.pi / p1) + 0.25 * np.cos(u2 * 2 * math.pi / p2)
        v = smooth(0.15, 0.95, v)
    else:
        def grid(ang, per, ox):
            u = ((XX - ox) * math.cos(ang) + (YY - 1280) * math.sin(ang)) / per
            w_ = (-(XX - ox) * math.sin(ang) + (YY - 1280) * math.cos(ang)) / per
            return 0.5 + 0.25 * np.cos(u * 2 * math.pi) + 0.25 * np.cos(w_ * 2 * math.pi)
        v = smooth(0.35, 0.95, 0.5 * grid(0.0, p1, 0) + 0.5 * grid(0.13, p2, 20))
        v = v ** 1.2
    t = np.clip(v * 0.8 + 0.2 * (YY / H), 0, 1)
    img = ramp(t, pal)
    img *= (0.2 + 0.9 * v)[..., None]
    img = img + col(bg) * (1 - v)[..., None] * 0.6
    return img, 0.9, 0.2


GENS["moire"] = lambda s, p: g_moire(s, **p)


def truchet_mask(s, seed, kind, wd, ox=0.0, oy=0.0):
    r = np.random.default_rng(seed)
    gx = int(W / s) + 3
    gy = int(H / s) + 3
    ori = r.integers(0, 2, (gy, gx))
    ix = np.floor((XX + ox) / s).astype(np.int32)
    iy = np.floor((YY + oy) / s).astype(np.int32)
    fx = (XX + ox) / s - ix
    fy = (YY + oy) / s - iy
    o = ori[np.clip(iy, 0, gy - 1), np.clip(ix, 0, gx - 1)]
    if kind == "arc":
        d_a = np.minimum(np.abs(np.hypot(fx, fy) - 0.5), np.abs(np.hypot(1 - fx, 1 - fy) - 0.5))
        d_b = np.minimum(np.abs(np.hypot(1 - fx, fy) - 0.5), np.abs(np.hypot(fx, 1 - fy) - 0.5))
        d = np.where(o == 0, d_a, d_b)
    else:
        d_a = np.abs(fx - fy) / 1.4142
        d_b = np.abs(fx + fy - 1) / 1.4142
        d = np.where(o == 0, d_a, d_b)
    aa = 1.0 / s
    return 1 - smooth(wd - aa, wd + aa, d), ix, iy


def g_truchet(seed, mode, bg, pal, s=90, wd=0.14):
    img = np.ones((H, W, 3), np.float32) * col(bg)
    if mode == "weave":
        m, ix, iy = truchet_mask(s, seed, "arc", wd)
        edge, _, _ = truchet_mask(s, seed, "arc", wd + 0.04)
        t = np.clip((XX / W * 0.4 + YY / H * 0.6), 0, 1)
        c = ramp(t, pal)
        img = img * (1 - edge[..., None]) + col(bg) * 0 + edge[..., None] * (col(pal[0][1]) * 0.35)
        img = img * (1 - m[..., None]) + c * m[..., None]
    elif mode == "maze":
        m, ix, iy = truchet_mask(s, seed, "slash", wd)
        t = np.clip(0.5 + 0.5 * np.sin((ix + iy) * 0.12) * 0 + (YY / H) * 0.7 - 0.1, 0, 1)
        c = ramp(t, pal)
        sh = blur(np.roll(np.roll(m, 4, 0), 4, 1), 4)
        img = img * (1 - 0.5 * sh[..., None])
        img = img * (1 - m[..., None]) + c * m[..., None]
    else:
        regions = fbm(seed + 3, 1.8, 3)
        for lvl, (sc, wdl) in enumerate(((s * 2.0, 0.1), (s, 0.13), (s * 0.5, 0.16))):
            m, ix, iy = truchet_mask(sc, seed + lvl * 17, "arc", wdl)
            act = smooth(lvl * 0.28 + 0.1, lvl * 0.28 + 0.14, regions) * (1 - smooth((lvl + 1) * 0.28 + 0.1, (lvl + 1) * 0.28 + 0.14, regions)) if lvl < 2 else smooth(0.66, 0.7, regions)
            if lvl == 0:
                act = 1 - smooth(0.34, 0.38, regions)
            elif lvl == 1:
                act = smooth(0.34, 0.38, regions) * (1 - smooth(0.66, 0.7, regions))
            c = col(pal[lvl % len(pal)][1])
            mm = m * act
            img = img * (1 - mm[..., None]) + c * mm[..., None]
    return img, 1.0, 0.2


GENS["truchet"] = lambda s, p: g_truchet(s, **p)


def g_fractal(seed, mode, cxy, width, iters, pal, cyc=0.06, c=None, bg="#020208", glow_k=0.6):
    aspect = H / W
    cx, cy = cxy
    re = cx + (XX.astype(np.float64) / W - 0.5) * width
    im_ = cy + (YY.astype(np.float64) / H - 0.5) * width * aspect
    if mode == "julia":
        z = re + 1j * im_
        cc = np.full(z.shape, c, np.complex128)
    elif mode == "ship":
        z = np.zeros_like(re, np.complex128)
        cc = re + 1j * im_
    else:
        z = np.zeros_like(re, np.complex128)
        cc = re + 1j * im_
    n = np.zeros(z.shape, np.float32)
    mu = np.zeros(z.shape, np.float32)
    alive = np.ones(z.shape, bool)
    for i in range(iters):
        if mode == "ship":
            z = np.where(alive, (np.abs(z.real) + 1j * np.abs(z.imag)) ** 2 + cc, z)
        else:
            z = np.where(alive, z * z + cc, z)
        esc = alive & (z.real ** 2 + z.imag ** 2 > 65536.0)
        if esc.any():
            mag = np.log(np.maximum(z.real ** 2 + z.imag ** 2, 1.0001)) * 0.5
            mu = np.where(esc, i + 1 - np.log2(np.maximum(mag, 1e-6) / math.log(256)) , mu)
            alive &= ~esc
        if not alive.any():
            break
    inside = alive
    t = (np.log(np.maximum(mu, 1.0)) * 1.0)
    tt = 0.5 + 0.5 * np.sin(t * 2.2 + mu * cyc)
    img = ramp(np.clip(tt, 0, 1), pal)
    fall = np.clip(np.log(np.maximum(mu, 1)) / math.log(iters), 0, 1)
    img *= (0.12 + 0.9 * smooth(0.05, 0.55, fall) ** 1.2)[..., None]
    img[inside] = col(bg)
    img += blur(img, 10) * glow_k * 0.4
    return img.astype(np.float32), 1.0, 0.2


GENS["fractal"] = lambda s, p: g_fractal(s, **p)


def g_plasma(seed, mode, pal, f=1.0):
    r = np.random.default_rng(seed)
    x = XX / W
    y = YY / H * (H / W)
    ph = r.random(6) * 6.28
    if mode == "classic":
        v = np.sin(x * 9 * f + ph[0]) + np.sin(y * 8 * f + ph[1]) + np.sin((x + y) * 7 * f + ph[2]) + np.sin(np.sqrt((x - 0.5) ** 2 + (y - 0.9) ** 2) * 16 * f + ph[3]) + np.sin(np.sqrt((x - 1.0) ** 2 + (y - 2.3) ** 2) * 13 * f + ph[4])
        t = (v / 5 + 1) * 0.5
        return ramp(t, pal), 1.0, 0.15
    if mode == "poster":
        v = np.sin(x * 11 * f + ph[0] + np.sin(y * 5 + ph[1])) + np.sin(y * 9 * f + ph[2] + np.cos(x * 6)) + np.sin((x * 0.6 - y) * 8 * f + ph[3])
        t = (v / 3 + 1) * 0.5
        q = np.floor(t * (len(pal) + 3)) / (len(pal) + 3)
        img = ramp(q, pal)
        return img, 0.6, 0.15
    # contour lines
    wx = fbm(seed, 2.0, 3)
    wy = fbm(seed + 1, 2.0, 3)
    v = np.sin((x + (wx - 0.5) * 0.8) * 7) + np.sin((y + (wy - 0.5) * 0.8) * 6) + np.sin(np.sqrt((x - 0.4) ** 2 + (y - 1.1) ** 2) * 10) * 1.2
    t = (v / 3.2 + 1) * 0.5
    ln = 0.5 + 0.5 * np.cos(t * 2 * math.pi * 14)
    ln = smooth(0.82, 0.98, ln)
    base = ramp(t, pal)
    img = base * 0.35 + ln[..., None] * ramp(np.clip(t + 0.15, 0, 1), pal) * 1.2
    return img, 1.0, 0.2


GENS["plasma"] = lambda s, p: g_plasma(s, **p)


#### ---- batch D ----
def g_rainwin(seed, bgstops, lights, ndrops=520, veil=0.05, warm="#FFB070", n_lights=240):
    r = np.random.default_rng(seed)
    scene = vgrad(bgstops)
    bokeh(scene, seed + 1, n_lights, lights, 18, 95, alpha=(0.25, 0.9), ring=0.45, region=(0, 700, W, H - 100))
    bokeh(scene, seed + 2, 40, lights, 60, 160, alpha=(0.1, 0.3), ring=0.2, region=(0, 300, W, H))
    blurred = blur(scene, 18)
    sharp = blur(scene, 2.5)
    img = blurred * (1 - veil) + veil * 0.5
    # clear trails
    for i in range(70):
        x, y = r.random() * W, r.random() * (H - 300)
        rad = 8 + r.random() * 18
        L = 120 + r.random() * 600
        x0, x1 = int(max(0, x - rad * 0.4)), int(min(W, x + rad * 0.4))
        y0, y1 = int(max(0, y - L)), int(y)
        if x1 > x0 and y1 > y0:
            fade = np.linspace(0, 1, y1 - y0)[:, None, None] ** 1.5
            img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - 0.6 * fade) + sharp[y0:y1, x0:x1] * 0.6 * fade
    for i in range(ndrops):
        rad = 10 + (r.random() ** 2.2) * 44
        x, y = r.random() * W, r.random() * H
        rx, ry = rad, rad * 1.3
        x0, x1 = int(max(0, x - rx - 3)), int(min(W, x + rx + 3))
        y0, y1 = int(max(0, y - ry - 3)), int(min(H, y + ry + 3))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        px = XX[y0:y1, x0:x1]
        py = YY[y0:y1, x0:x1]
        nx, ny = (px - x) / rx, (py - y) / ry
        rr = np.sqrt(nx * nx + ny * ny)
        m = 1 - smooth(0.92, 1.0, rr)
        sx = np.clip(x - (px - x) * 1.7, 0, W - 1).astype(np.float32)
        sy = np.clip(y - (py - y) * 1.7 + ry * 0.4, 0, H - 1).astype(np.float32)
        samp = cv2.remap(sharp, sx, sy, cv2.INTER_LINEAR)
        samp = samp * 1.15 + 0.02
        rim = smooth(0.7, 1.0, rr) * m
        shade = 1 - 0.4 * rim * np.clip(ny + 0.3, 0, 1)
        hl = np.exp(-(((nx + 0.38) ** 2 + (ny + 0.45) ** 2) / 0.015)) * m
        patch = samp * shade[..., None] + hl[..., None] * 0.75
        reg = img[y0:y1, x0:x1]
        img[y0:y1, x0:x1] = reg * (1 - m[..., None]) + patch * m[..., None]
    return img, 1.4, 0.35


GENS["rainwin"] = lambda s, p: g_rainwin(s, **p)


def g_trails(seed, mode, bg, c1, c2, c3=None):
    r = np.random.default_rng(seed)
    canvas = np.zeros((H, W, 3), np.uint8)
    base = vgrad(bg)
    S = 8

    def poly(pts, color, th):
        cv2.polylines(canvas, [(np.asarray(pts) * S).astype(np.int32).reshape(-1, 1, 2)], False, rgb8(color), th, cv2.LINE_AA, 3)

    if mode == "highway":
        vp = np.array([720.0, 1180.0])
        for i in range(330):
            side = -1 if i % 2 == 0 else 1
            lane = r.uniform(0.04, 1.0) ** 1.2
            ang = math.radians(90 + side * (8 + lane * 70))
            curv = r.uniform(-0.25, 0.25)
            c = col(c1) if side < 0 else col(c2)
            s0 = r.uniform(0.0, 0.7)
            ln = r.uniform(0.15, 0.55)
            n = 40
            pts, ths = [], []
            for k in range(n):
                s = s0 + ln * k / (n - 1)
                dist = (s ** 2.2) * 2300
                a = ang + curv * s * 0.6
                p = vp + np.array([math.cos(a) * dist, math.sin(a) * dist * 0.62 if False else abs(math.sin(a)) * dist * 0.75])
                pts.append(p)
            for k in range(n - 1):
                s = s0 + ln * k / (n - 1)
                th = max(1, int(1 + 9 * s ** 1.6))
                cv2.line(canvas, tuple((pts[k] * S).astype(int)), tuple((pts[k + 1] * S).astype(int)), rgb8(c * (0.35 + 0.65 * s)), th, cv2.LINE_AA, 3)
        L = canvas.astype(np.float32) / 255.0
        img = base + L * 1.1 + blur(L, 4) * 1.0 + blur(L, 20) * 0.8
        glow(img, vp[0], vp[1], 150, col(c1), 0.5, "exp")
        for _ in range(240):
            x = r.uniform(80, 1360)
            y = vp[1] - r.uniform(0, 40) ** 1.0
            hh = r.uniform(10, 70) * (1 - abs(x - 720) / 900)
            img[int(y - hh):int(y), int(x):int(x) + 3] += col("#FFE6A8") * 0.55
        return img, 1.3, 0.35
    if mode == "star":
        pole = np.array([r.uniform(420, 1020), r.uniform(560, 1000)])
        for i in range(1100):
            rho = math.sqrt(r.random()) * 2300 + 20
            a0 = r.random() * 6.283
            span = r.uniform(0.25, 0.55)
            n = max(8, int(rho * span / 16))
            ts = np.linspace(a0, a0 + span, n)
            pts = np.stack([pole[0] + np.cos(ts) * rho, pole[1] + np.sin(ts) * rho], 1)
            tint = [c1, c2, c3 or "#FFFFFF"][int(r.integers(0, 3))]
            b = 0.25 + 0.75 * r.random() ** 2
            th = 1 if r.random() < 0.8 else 2
            poly(pts, col(tint) * b, th)
        L = canvas.astype(np.float32) / 255.0
        img = base + L * 1.1 + blur(L, 2.5) * 0.6
        mw = fbm(seed + 2, 2.0, 4)
        band = np.exp(-(((XX * 0.7 + YY * 0.35) - 1000) / 420) ** 2)
        img += (band * mw ** 1.5 * 0.18)[..., None] * col(c1)
        prof = profile(seed + 3, 2150, 380, 1.4, 5, 0.5, 0.2, 720)
        m = poly_below(prof)
        img = img * (1 - m[..., None]) + col("#04060A") * m[..., None]
        # little tent glow
        glow(img, 880, 2230, 60, col("#FFB050"), 0.5, "exp")
        tent = Image.new("L", (W, H), 0)
        ImageDraw.Draw(tent).polygon([(840, 2250), (880, 2190), (920, 2250)], fill=255)
        tm = np.asarray(tent, np.float32) / 255.0
        img = img * (1 - tm[..., None]) + col("#FFC070") * tm[..., None] * 0.9
        return img, 1.2, 0.3
    # topdown interchange
    img = vgrad(bg)
    # city blocks with windows
    blk = np.zeros((H, W), np.float32)
    for gx in range(0, W, 110):
        for gy in range(0, H, 110):
            if r.random() < 0.55:
                w_, h_ = int(r.integers(30, 90)), int(r.integers(30, 90))
                blk[gy:gy + h_, gx:gx + w_] = r.random() * 0.5 + 0.2
    img += blk[..., None] * col("#241C48") * 0.9
    roads = []
    t = np.linspace(0, 2 * math.pi, 400)
    roads.append(np.stack([720 + 470 * np.cos(t), 1000 + 470 * np.sin(t)], 1))
    roads.append(np.stack([720 + 300 * np.cos(t), 1000 + 300 * np.sin(t)], 1))
    xs = np.linspace(-40, W + 40, 300)
    roads.append(np.stack([xs, 1900 + 170 * np.sin(xs / 210)], 1))
    roads.append(np.stack([xs, 1760 - 120 * np.sin(xs / 230 + 1.2) + 0.1 * xs], 1))
    ys = np.linspace(-40, H + 40, 400)
    roads.append(np.stack([300 + 80 * np.sin(ys / 330), ys], 1))
    roads.append(np.stack([1160 + 60 * np.sin(ys / 280 + 2.0), ys], 1))
    for ri, pts in enumerate(roads):
        for lane in (-14, -5, 5, 14):
            d = np.gradient(pts, axis=0)
            nrm = np.stack([-d[:, 1], d[:, 0]], 1)
            nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-6)
            lp = pts + nrm * lane
            color = c1 if lane < 0 else c2
            k = 0
            while k < len(lp) - 3:
                ln = int(r.integers(4, 26))
                if r.random() < 0.6:
                    poly(lp[k:k + ln], col(color) * (0.5 + 0.5 * r.random()), 2)
                k += ln + int(r.integers(2, 14))
    L = canvas.astype(np.float32) / 255.0
    img = img + L * 1.0 + blur(L, 3) * 0.9 + blur(L, 16) * 0.8
    return img, 1.2, 0.3


GENS["trails"] = lambda s, p: g_trails(s, **p)


def g_minimal(seed, mode, sky, c):
    r = np.random.default_rng(seed)
    img = vgrad(sky)
    if mode == "dunes":
        sx, sy, sr = 900, 780, 200
        d = np.sqrt((XX - sx) ** 2 + (YY - sy) ** 2)
        s = 1 - smooth(sr - 1.2, sr + 1.2, d)
        img = img * (1 - s[..., None]) + col(c[0]) * s[..., None]
        for i, (colr, base, amp, fr) in enumerate(c[1]):
            y = base + amp * np.sin(np.linspace(0, 1, W) * fr + i * 1.7) + amp * 0.35 * np.sin(np.linspace(0, 1, W) * fr * 2.3 + i)
            m = poly_below(np.interp(np.linspace(0, W, 720), np.arange(W), y), 2)
            sh = 1 - 0.12 * np.clip(blur(np.roll(m, 40, 0) * 0 + m, 1) - 0, 0, 1)
            img = img * (1 - m[..., None]) + col(colr) * m[..., None]
        return img, 0.9, 0.0
    if mode == "moonsea":
        sx, sy, sr = 720, 900, 300
        d = np.sqrt((XX - sx) ** 2 + (YY - sy) ** 2)
        s = 1 - smooth(sr - 1.2, sr + 1.2, d)
        img = img * (1 - s[..., None]) + col(c[0]) * s[..., None]
        hor = 1500
        sea = YY > hor
        img[sea] = col(c[1])
        k = np.clip((YY - hor) / (H - hor), 0, 1)
        lines = (YY - hor) % np.maximum(8, 8 + k * 50)
        band = lines < (3 + k * 14)
        refl = np.exp(-(((XX - sx) / (300 * (0.45 + 0.9 * k) * (0.6 + 0.4 * np.sin(YY / 11.0)))) ** 2))
        wave = (np.sin(YY * 0.4) * 0.5 + 0.5) < 0.78
        img = np.where((sea & band)[..., None], col(c[2]) * (0.35 + 0.5 * k[..., None]), img)
        rb = (sea & band & wave)[..., None] * refl[..., None]
        img = img * (1 - rb) + col(c[0]) * rb
        return img, 0.9, 0.1
    # peaks
    sx, sy, sr = 1010, 760, 150
    for i, (cl, cd, px_, base, hh, ww) in enumerate(c[1]):
        pass
    d = np.sqrt((XX - sx) ** 2 + (YY - sy) ** 2)
    s = 1 - smooth(sr - 1.2, sr + 1.2, d)
    img = img * (1 - s[..., None]) + col(c[0]) * s[..., None]
    ss = 2
    for i, (cl, cd, px_, base, hh, ww) in enumerate(c[1]):
        im = Image.new("L", (W * ss, H * ss), 0)
        im2 = Image.new("L", (W * ss, H * ss), 0)
        ImageDraw.Draw(im).polygon([((px_ - ww) * ss, base * ss), (px_ * ss, (base - hh) * ss), ((px_ + ww) * ss, base * ss)], fill=255)
        ImageDraw.Draw(im2).polygon([(px_ * ss, (base - hh) * ss), ((px_ + ww) * ss, base * ss), ((px_ + ww * 0.1) * ss, base * ss)], fill=255)
        m = np.asarray(im.resize((W, H), Image.BOX), np.float32) / 255.0
        m2 = np.asarray(im2.resize((W, H), Image.BOX), np.float32) / 255.0
        img = img * (1 - m[..., None]) + col(cl) * m[..., None]
        img = img * (1 - m2[..., None]) + col(cd) * m2[..., None]
    gm = YY > 1950
    mm = np.clip((YY - 1950) / 2, 0, 1)
    img = img * (1 - mm[..., None]) + col(c[2]) * mm[..., None]
    return img, 0.9, 0.0


GENS["minimal"] = lambda s, p: g_minimal(s, **p)


def gear_poly(cx, cy, R, teeth, depth=0.12, rot=0.0):
    pts = []
    for i in range(teeth * 4):
        a = rot + i / (teeth * 4) * 2 * math.pi
        rr = R * (1 + depth) if (i % 4) in (1, 2) else R
        pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
    return pts


def g_blueprint(seed, mode, bg, line, line2=None):
    r = np.random.default_rng(seed)
    ss = 2
    M = Image.new("L", (W * ss, H * ss), 0)
    d = ImageDraw.Draw(M)
    M2 = Image.new("L", (W * ss, H * ss), 0)
    d2 = ImageDraw.Draw(M2)

    def L(p, w=2, f=255, dd=d):
        dd.line([(a * ss, b * ss) for a, b in p], fill=f, width=int(w * ss), joint="curve")

    def circ(cx, cy, rr, w=2, f=255, dd=d):
        dd.ellipse([(cx - rr) * ss, (cy - rr) * ss, (cx + rr) * ss, (cy + rr) * ss], outline=f, width=int(w * ss))

    def dim(x0, y0, x1, y1):
        L([(x0, y0), (x1, y1)], 2)
        a = math.atan2(y1 - y0, x1 - x0)
        for (x, y, s) in ((x0, y0, 1), (x1, y1, -1)):
            p1 = (x + s * 22 * math.cos(a + 0.25), y + s * 22 * math.sin(a + 0.25))
            p2 = (x + s * 22 * math.cos(a - 0.25), y + s * 22 * math.sin(a - 0.25))
            d.polygon([(x * ss, y * ss), (p1[0] * ss, p1[1] * ss), (p2[0] * ss, p2[1] * ss)], fill=255)
    if mode == "blue":
        for x in range(0, W + 1, 40):
            L([(x, 0), (x, H)], 1 if x % 200 else 2, 90 if x % 200 else 150, d2)
        for y in range(0, H + 1, 40):
            L([(0, y), (W, y)], 1 if y % 200 else 2, 90 if y % 200 else 150, d2)
        gears = [(520, 800, 300, 22), (1000, 1160, 190, 14), (460, 1620, 150, 11)]
        for cx, cy, R, t in gears:
            d.polygon([(a * ss, b * ss) for a, b in gear_poly(cx, cy, R, t, 0.1, 0.0)], outline=255, width=3 * ss)
            circ(cx, cy, R * 0.55, 2)
            circ(cx, cy, R * 0.18, 3)
            circ(cx, cy, R * 0.85, 1, 170)
            L([(cx - R * 1.35, cy), (cx + R * 1.35, cy)], 1, 200)
            L([(cx, cy - R * 1.35), (cx, cy + R * 1.35)], 1, 200)
            for k in range(6):
                a = k / 6 * 2 * math.pi
                L([(cx + math.cos(a) * R * 0.18, cy + math.sin(a) * R * 0.18), (cx + math.cos(a) * R * 0.55, cy + math.sin(a) * R * 0.55)], 2)
        dim(220, 1160, 820, 1160)
        dim(1240, 600, 1240, 1000)
        dim(260, 1900, 1180, 1900)
        for k in range(18):
            L([(260 + k * 52, 1980), (320 + k * 52, 1920)], 1, 200)
        d.rectangle([140 * ss, 2220 * ss, 1300 * ss, 2440 * ss], outline=255, width=3 * ss)
        for x in (500, 900):
            L([(x, 2220), (x, 2440)], 2)
        L([(140, 2330), (1300, 2330)], 2)
        circ(720, 1820, 500, 1, 150)
    elif mode == "chalk":
        for k in range(7):
            cx, cy = r.uniform(250, 1200), r.uniform(300, 2300)
            R = r.uniform(110, 260)
            circ(cx, cy, R, 4)
            if k % 3 == 0:
                pts = [(cx + R * math.cos(a), cy + R * math.sin(a)) for a in (-math.pi / 2, math.pi / 6, 5 * math.pi / 6, -math.pi / 2)]
                L(pts, 4)
            elif k % 3 == 1:
                L([(cx - R, cy), (cx + R, cy)], 3)
                L([(cx, cy - R), (cx, cy + R)], 3)
                circ(cx, cy, R * 0.5, 3)
            else:
                pts = [(cx + R * 0.1 * t * math.cos(t * 0.9), cy + R * 0.1 * t * math.sin(t * 0.9)) for t in np.linspace(0, 20, 120)]
                L(pts, 3)
        xs = np.linspace(80, 1360, 200)
        L(list(zip(xs, 2300 - 190 * np.sin(xs / 120))), 4)
        L([(80, 2340), (1360, 2340)], 3)
        L([(80, 2100), (80, 2340)], 3)
        for k in range(9):
            L([(80 + k * 160, 2330), (80 + k * 160, 2350)], 3)
        for k in range(5):
            x, y = r.uniform(150, 1300), r.uniform(250, 1000)
            pts = []
            for q in range(10):
                a = q / 10 * 2 * math.pi - math.pi / 2
                rr = 70 if q % 2 == 0 else 28
                pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
            pts.append(pts[0])
            L(pts, 3)
    else:
        for x in range(0, W + 1, 30):
            L([(x, 0), (x, H)], 1, 70 if x % 150 else 150, d2)
        for y in range(0, H + 1, 30):
            L([(0, y), (W, y)], 1, 70 if y % 150 else 150, d2)
        L([(60, 1280), (W - 60, 1280)], 3, 255)
        L([(720, 80), (720, H - 80)], 3, 255)
        xs = np.linspace(80, W - 80, 400)
        L(list(zip(xs, 1280 - 420 * np.sin((xs - 720) / 150))), 5, 255, d)
        for k in range(14):
            x = 100 + k * 94
            y = 1280 + 300 * ((x - 720) / 600) ** 2 - 180
            dd = ImageDraw.Draw(M2)
            dd.ellipse([(x - 9) * ss, (y - 9) * ss, (x + 9) * ss, (y + 9) * ss], fill=255)
        ys = [1280 - 600 * (1 - ((x - 720) / 600) ** 2) if abs(x - 720) < 600 else None for x in xs]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        L(pts, 5, 255, d2)
    m = np.asarray(M.resize((W, H), Image.BOX), np.float32) / 255.0
    m2 = np.asarray(M2.resize((W, H), Image.BOX), np.float32) / 255.0
    paper = fbm(seed + 3, 40, 3, cy=56)
    img = np.ones((H, W, 3), np.float32) * col(bg) * (0.93 + 0.12 * paper[..., None])
    if mode == "chalk":
        smudge = fbm(seed + 4, 3, 4)
        img *= (0.9 + 0.16 * smudge[..., None])
        dust = smooth(0.55, 0.8, fbm(seed + 6, 14, 3)) * 0.12
        img += dust[..., None] * col(line)
        grain = fbm(seed + 7, 400, 1, cy=700)
        m = m * (0.55 + 0.6 * grain)
    img += m2[..., None] * col(line2 or line) * (0.45 if mode == "blue" else (0.6 if mode == "graph" else 0.3))
    img += m[..., None] * col(line) * (1.0 if mode != "graph" else 0.9)
    if mode == "graph":
        img = np.ones((H, W, 3), np.float32) * col(bg) * (0.95 + 0.08 * paper[..., None])
        img = img * (1 - m2[..., None] * 0.6) + m2[..., None] * col(line2) * 0.6
        pen = Image.new("L", (W * ss, H * ss), 0)
        dp = ImageDraw.Draw(pen)
        xs = np.linspace(80, W - 80, 400)
        dp.line([(x * ss, (1280 - 420 * math.sin((x - 720) / 150)) * ss) for x in xs], fill=255, width=7 * ss, joint="curve")
        pm = np.asarray(pen.resize((W, H), Image.BOX), np.float32) / 255.0
        parab = Image.new("L", (W * ss, H * ss), 0)
        dq = ImageDraw.Draw(parab)
        dq.line([(x * ss, (1280 + 300 * ((x - 720) / 600) ** 2 - 180) * ss) for x in xs], fill=255, width=7 * ss, joint="curve")
        qm = np.asarray(parab.resize((W, H), Image.BOX), np.float32) / 255.0
        img = img * (1 - m[..., None] * 0.9) + m[..., None] * col("#1A2A3A") * 0.9
        img = img * (1 - pm[..., None]) + pm[..., None] * col("#D8432B")
        img = img * (1 - qm[..., None]) + qm[..., None] * col("#2B6AD8")
    if mode == "blue":
        img = img + blur(m, 2) [..., None] * col(line) * 0.2
    return img, 1.1, 0.3


GENS["blueprint"] = lambda s, p: g_blueprint(s, **p)


def g_geode(seed, mode, pal, bg="#0A080C", bands=14, warp=0.55, crystal="#F4EEFF"):
    n = fbm(seed, 2.0, 5)
    w1 = fbm(seed + 3, 2.4, 4)
    w2 = fbm(seed + 4, 2.4, 4)
    if mode == "slice":
        cx, cy = 720, 1280
        dx = (XX - cx) / 600.0
        dy = (YY - cy) / 1000.0
        nn = warp_sample(n, w1, w2, 260)
        rr = np.sqrt(dx * dx + dy * dy) + (nn - 0.5) * warp * 0.7
        inside = 1 - smooth(0.96, 1.0, rr)
        v = rr / 1.0
        t = np.clip(1 - v, 0, 1)
        band = (rr * bands + (nn - 0.5) * 2.0) % 1.0
        ring = np.abs(2 * band - 1)
        img = ramp(np.clip(rr * 1.05, 0, 1), pal) * (0.65 + 0.5 * smooth(0, 1, ring) ** 0.8)[..., None]
        lines = 1 - smooth(0.0, 0.07, np.minimum(band, 1 - band))
        img = img * (1 - 0.35 * lines[..., None])
        cav = 1 - smooth(0.2, 0.3, rr)
        cr = fbm(seed + 8, 40, 3, cy=70)
        spark = smooth(0.62, 0.85, cr)
        crys = col(crystal) * (0.55 + 0.45 * cr[..., None])
        crys += spark[..., None] * 0.6
        img = img * (1 - cav[..., None]) + crys * cav[..., None]
        rim = np.exp(-((rr - 0.99) / 0.025) ** 2)
        img += rim[..., None] * col(pal[0][1]) * 0.4
        rock = np.ones((H, W, 3), np.float32) * col(bg) * (0.7 + 0.7 * fbm(seed + 9, 14, 4)[..., None])
        out = rock * (1 - inside[..., None]) + img * inside[..., None]
        sh = blur(inside, 40)
        out *= (1 - 0.4 * np.clip(sh - inside, 0, 1))[..., None]
        return out, 1.2, 0.3
    # full-bleed agate / malachite
    nn = warp_sample(n, w1, w2, 360 * warp * 2)
    nn = warp_sample(nn, w2, w1, 140)
    v = nn * bands
    band = v % 1.0
    ring = np.abs(2 * band - 1)
    lines = 1 - smooth(0.0, 0.05, np.minimum(band, 1 - band))
    tone = ramp(nn, pal)
    img = tone * (0.55 + 0.6 * ring[..., None] ** 0.9)
    img = img * (1 - 0.4 * lines[..., None]) + lines[..., None] * col(pal[-1][1]) * 0.25
    gy, gx = np.gradient(blur(nn, 3))
    img *= (0.9 + np.clip(-(gx + gy) * 80, -0.3, 0.45))[..., None]
    return img, 1.0, 0.25


GENS["geode"] = lambda s, p: g_geode(s, **p)


def g_wood(seed, mode, pal, plank=360):
    if mode == "planks":
        img = np.zeros((H, W, 3), np.float32)
        n_p = int(math.ceil(W / plank))
        r = np.random.default_rng(seed)
        for p in range(n_p):
            x0, x1 = p * plank, min(W, (p + 1) * plank)
            sl = slice(x0, x1)
            shift = r.random() * 1000
            tone = r.uniform(0.0, 1.0)
            ox = XX[:, sl] - x0
            yy = YY[:, sl] + shift
            wob = 70 * np.sin(yy / 380 + p) + 25 * np.sin(yy / 90 + p * 3)
            rings = ((ox * 0.8 + wob + 40 * fbm(seed + p, 3, 3, h=H, w=x1 - x0)) / 24.0)
            g = 0.5 + 0.5 * np.sin(rings * 2 * math.pi + 3 * fbm(seed + p + 7, 2.0, 4, h=H, w=x1 - x0))
            fine = fbm(seed + p + 20, 90, 2, cy=4, h=H, w=x1 - x0)
            t = np.clip(0.35 * tone + 0.45 * g + 0.25 * fine, 0, 1)
            img[:, sl] = ramp(t, pal)
            gap = np.minimum(ox, plank - ox)
            img[:, sl] *= (0.5 + 0.5 * smooth(0, 7, gap))[..., None]
            ky = (r.random(3) * 2200 + 150)
            for k in ky:
                dd = np.sqrt((XX[:, sl] - (x0 + plank * r.uniform(0.3, 0.7))) ** 2 + (YY[:, sl] - k) ** 2)
                img[:, sl] *= (1 - 0.55 * np.exp(-(dd / (14 + r.random() * 18)) ** 2))[..., None]
        for p in range(n_p):
            for ny in (140, H - 140):
                for nx in (p * plank + 38, (p + 1) * plank - 38):
                    glow(img, nx, ny, 9, col("#1A0F08"), 0.6, "gauss")
        return img, 1.3, 0.25
    if mode == "rings":
        cx, cy = 720, 1280
        dx, dy = XX - cx, YY - cy
        a = np.arctan2(dy, dx)
        rr = np.sqrt(dx * dx + dy * dy)
        n1 = fbm(seed, 4, 4)
        warpd = rr + 40 * np.sin(a * 3 + 1.0) + 55 * (n1 - 0.5) + 20 * np.sin(a * 7)
        rings = warpd / 27.0
        g = 0.5 + 0.5 * np.sin(rings * 2 * math.pi)
        g = g ** 1.3
        t = np.clip(0.2 + 0.55 * g + 0.2 * (rr / 1000) + 0.1 * fbm(seed + 3, 30, 3), 0, 1)
        img = ramp(t, pal)
        bark = smooth(805, 830, warpd)
        cracks = fbm(seed + 8, 3, 3)
        radial = np.abs(np.sin(a * 5 + cracks * 7))
        crack = (1 - smooth(0.0, 0.02, radial)) * smooth(60, 120, rr) * (1 - smooth(600, 790, rr)) * smooth(0.45, 0.6, fbm(seed + 9, 6, 3))
        img *= (1 - 0.7 * crack)[..., None]
        bark_c = ramp(fbm(seed + 10, 80, 3), [(0, "#1F140D"), (1, "#4A3322")])
        img = img * (1 - bark[..., None]) + bark_c * bark[..., None]
        img *= (1 - 0.55 * smooth(1000, 1500, rr)) [..., None] + 0.0
        img *= 1.0
        return img, 1.3, 0.3
    # figured burl
    n = fbm(seed, 2.0, 5)
    w1 = fbm(seed + 3, 3.2, 4)
    w2 = fbm(seed + 4, 3.2, 4)
    nn = warp_sample(n, w1, w2, 420)
    lines = 0.5 + 0.5 * np.sin((YY * 0.016 + nn * 22.0 + fbm(seed + 6, 1.4, 3) * 9) * 1.0)
    fine = fbm(seed + 8, 220, 2, cy=22)
    t = np.clip(0.2 + 0.55 * lines + 0.25 * fine, 0, 1)
    img = ramp(t, pal)
    gy, gx = np.gradient(blur(nn, 5))
    sheen = np.clip(0.5 + (gx * 0.4 - gy * 0.9) * 90, 0, 1)
    img *= (0.7 + 0.5 * sheen)[..., None]
    gloss = np.exp(-(((XX * 0.4 + YY * 0.9) - 1700) / 260) ** 2) * 0.22
    img += gloss[..., None] * col("#FFEBD0")
    return img, 1.0, 0.3


GENS["wood"] = lambda s, p: g_wood(s, **p)


def g_dither(seed, mode, pal, px=6):
    h, w = -(-H // px), -(-W // px)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if mode == "sphere":
        d = np.sqrt((xx - w * 0.5) ** 2 + (yy - h * 0.38) ** 2)
        R = w * 0.34
        z = np.sqrt(np.clip(1 - (d / R) ** 2, 0, 1))
        lx, ly = -0.5, -0.6
        nxn, nyn = (xx - w * 0.5) / R, (yy - h * 0.38) / R
        lam = np.clip(nxn * lx + nyn * ly + z * 0.62, 0, 1)
        sph = np.where(d < R, lam, -1)
        floor = np.clip((yy - h * 0.62) / (h * 0.38), 0, 1)
        grid = (((xx * 0.6 + 40) % 8 < 1) | (((yy - h * 0.62) ** 1) % 8 < 1)) & (yy > h * 0.62)
        t = np.where(yy > h * 0.62, 0.15 + 0.4 * floor ** 0.7, 0.05 + 0.2 * (1 - yy / (h * 0.62)))
        t = np.where(grid, t * 0.6, t)
        sh = np.exp(-(((xx - w * 0.5) / (R * 1.1)) ** 2 + ((yy - h * 0.75) / (R * 0.22)) ** 2))
        t = np.where(yy > h * 0.62, t * (1 - 0.6 * sh), t)
        t = np.where(sph >= 0, 0.35 + 0.65 * sph, t)
    elif mode == "sunrise":
        t = np.clip(1 - yy / (h * 0.7), 0, 1) ** 1.3 * 0.7
        d = np.sqrt((xx - w * 0.62) ** 2 + (yy - h * 0.45) ** 2)
        t = np.where(d < w * 0.2, 0.98, t + 0.25 * np.exp(-(d / (w * 0.4)) ** 2))
        for i, (base, amp, tone) in enumerate(((0.66, 14, 0.42), (0.74, 11, 0.25), (0.84, 8, 0.1))):
            pr = h * base - amp * (np.sin(xx * 0.09 + i * 2) + 0.5 * np.sin(xx * 0.23 + i))
            t = np.where(yy > pr, tone * (0.8 + 0.2 * (((xx + yy) % 4) < 2)), t)
    else:
        t = 0.5 + 0.5 * np.sin(np.sqrt((xx - w * 0.3) ** 2 + (yy - h * 0.3) ** 2) * 0.22) * 0.0
        stairs = ((xx + yy * 0.5) % 40) / 40.0
        t = 0.15 + 0.7 * stairs * (1 - 0.35 * np.clip(yy / h, 0, 1))
        d = np.sqrt((xx - w * 0.5) ** 2 + (yy - h * 0.55) ** 2)
        t = np.where(d < w * 0.28, 1 - np.clip(d / (w * 0.28), 0, 1) ** 1.4, t)
        t = np.where((d > w * 0.28) & (d < w * 0.31), 0.0, t)
    img = dither_pal(np.clip(t, 0, 1), pal, h, w)
    big = np.repeat(np.repeat(img, px, 0), px, 1)[:H, :W]
    return big, 0.4, 0.0


GENS["dither"] = lambda s, p: g_dither(s, **p)


def g_coderain(seed, pal, cw=24, ch=36, mode="glyph", length=26.0, bgc="#010604"):
    r = np.random.default_rng(seed)
    cols_, rows_ = W // cw, H // ch + 1
    G = 48
    atlas = (r.random((G, 7, 5)) < 0.45).astype(np.float32)
    atlas[:, :, 2] = np.where(r.random((G, 7)) < 0.5, 1, atlas[:, :, 2])
    if mode == "binary":
        zero = np.array([[0, 1, 1, 1, 0], [1, 0, 0, 0, 1], [1, 0, 0, 1, 1], [1, 0, 1, 0, 1], [1, 1, 0, 0, 1], [1, 0, 0, 0, 1], [0, 1, 1, 1, 0]], np.float32)
        one = np.array([[0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 1, 0]], np.float32)
        atlas = np.stack([zero, one])
        G = 2
    gi = r.integers(0, G, (rows_, cols_))
    glyph = atlas[gi]  # rows, cols, 7, 5
    glyph = glyph.transpose(0, 2, 1, 3).reshape(rows_ * 7, cols_ * 5)
    glyph = cv2.resize(glyph, (cols_ * cw, rows_ * ch), interpolation=cv2.INTER_AREA)
    # thin gaps between glyphs
    head = r.random(cols_) * rows_
    speed = 0.7 + r.random(cols_)
    rr = np.arange(rows_)[:, None].astype(np.float32)
    dist = (head[None, :] - rr) % rows_
    ln = length * (0.5 + r.random(cols_))[None, :]
    bright = np.exp(-dist / ln * 3.0) * (dist < ln * 1.6)
    bright = bright * (0.35 + 0.65 * r.random((rows_, cols_)) ** 0.5)
    bigb = cv2.resize(bright.astype(np.float32), (cols_ * cw, rows_ * ch), interpolation=cv2.INTER_NEAREST)
    headm = cv2.resize(((dist < 1.0)).astype(np.float32), (cols_ * cw, rows_ * ch), interpolation=cv2.INTER_NEAREST)
    pad = np.zeros((rows_ * ch, cols_ * cw), np.float32)
    mask = glyph * bigb
    full = np.zeros((H, W), np.float32)
    full[:min(H, mask.shape[0]), :min(W, mask.shape[1])] = mask[:H, :W]
    hm = np.zeros((H, W), np.float32)
    hm[:min(H, mask.shape[0]), :min(W, mask.shape[1])] = (glyph * headm)[:H, :W]
    img = np.ones((H, W, 3), np.float32) * col(bgc)
    img += ramp(np.clip(full * 1.2, 0, 1), pal) * full[..., None] * 1.1
    img += hm[..., None] * col(pal[-1][1]) * 1.2
    img += blur(full, 5)[..., None] * col(pal[len(pal) // 2][1]) * 0.9
    img += blur(hm, 9)[..., None] * col(pal[-1][1]) * 0.8
    return img, 1.2, 0.3


GENS["coderain"] = lambda s, p: g_coderain(s, **p)


def bolt(r, p0, p1, depth, spread, out, width, branch=0.35):
    if depth == 0:
        out.append((p0, p1, width))
        return
    mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    ln = math.hypot(dx, dy) + 1e-6
    off = (r.random() - 0.5) * spread * ln
    mid = (mid[0] - dy / ln * off, mid[1] + dx / ln * off)
    bolt(r, p0, mid, depth - 1, spread, out, width, branch)
    bolt(r, mid, p1, depth - 1, spread, out, width, branch)
    if depth > 3 and r.random() < branch:
        a = math.atan2(dy, dx) + (r.random() - 0.5) * 1.2
        L2 = ln * r.uniform(0.4, 0.8)
        e = (mid[0] + math.cos(a) * L2, mid[1] + math.sin(a) * L2)
        bolt(r, mid, e, depth - 2, spread, out, width * 0.6, branch)


def g_lightning(seed, mode, sky, c, ground="#05060C"):
    r = np.random.default_rng(seed)
    img = vgrad(sky)
    cl = fbm(seed, 2.6, 6, cy=2.2)
    img *= (0.25 + 1.0 * cl[..., None])
    M = Image.new("L", (W * 2, H * 2), 0)
    d = ImageDraw.Draw(M)
    segs = []
    if mode == "storm":
        bolt(r, (900, 100), (560, 1950), 9, 0.55, segs, 7.0, 0.5)
    elif mode == "forks":
        for x0, x1, y1 in ((330, 200, 1500), (1100, 1260, 1750), (720, 640, 1200)):
            bolt(r, (x0, 60), (x1, y1), 8, 0.5, segs, 5.0, 0.45)
    else:
        cx, cy = 720, 1250
        for k in range(16):
            a = r.random() * 6.283
            e = (cx + math.cos(a) * 690, cy + math.sin(a) * 690)
            bolt(r, (cx, cy), e, 7, 0.35, segs, 4.0, 0.3)
    for (a, b, wd) in segs:
        d.line([(a[0] * 2, a[1] * 2), (b[0] * 2, b[1] * 2)], fill=255, width=max(1, int(wd * 2)))
    m = np.asarray(M.resize((W, H), Image.BOX), np.float32) / 255.0
    if mode == "plasma":
        img = np.ones((H, W, 3), np.float32) * col(sky[0][1] if isinstance(sky[0], tuple) else sky[0])
        rr = np.sqrt((XX - 720) ** 2 + (YY - 1250) ** 2)
        glass = 1 - smooth(698, 706, rr)
        img = img * 0.0 + col("#05050E")
        img += glass[..., None] * ramp(rr / 700.0, [(0, c[1]), (0.6, "#2A0A4A"), (1, "#0A0620")]) * 0.55
        rim = np.exp(-((rr - 700) / 6) ** 2)
        img += rim[..., None] * col(c[0]) * 0.6
        m = m * glass
        img += blur(m, 4)[..., None] * col(c[0]) * 1.2 + blur(m, 24)[..., None] * col(c[0]) * 1.2
        img += m[..., None] * col("#FFFFFF") * 1.2
        core = np.exp(-(rr / 90) ** 2)
        img += core[..., None] * col(c[1]) * 1.4
        hl = np.exp(-(((XX - 520) / 120) ** 2 + ((YY - 800) / 200) ** 2)) * 0.25
        img += hl[..., None]
        # base
        base = (YY > 1990) & (np.abs(XX - 720) < 330 + (YY - 1990) * 0.4)
        img = np.where(base[..., None], col("#15151F") * (0.6 + 0.4 * np.clip((YY - 1990) / 200, 0, 1))[..., None], img)
        return img, 1.1, 0.25
    flash = blur(m, 90)
    img += flash[..., None] * col(c[0]) * 3.0 * (0.4 + 1.2 * cl[..., None] ** 1.2)
    img += blur(m, 6)[..., None] * col(c[0]) * 1.4 + blur(m, 24)[..., None] * col(c[0]) * 1.1
    img += m[..., None] * col("#FFFFFF") * 1.3
    prof = profile(seed + 3, 2250, 160, 2.2, 5, 0.5, 0.0, 720)
    if mode == "storm":
        sea = YY > 2150
        wave = np.sin(YY * 0.2 + XX * 0.01) * 0.5 + 0.5
        sk = np.clip(blur(m, 10) * 3, 0, 1)
        img = np.where(sea[..., None], col(ground) * (0.8 + 0.5 * wave[..., None]) + np.roll(sk, 2 * (2560 - 2150), 0)[..., None] * 0, img)
        mirror = np.flipud(img)
        img = np.where(sea[..., None], img * 0.4 + np.roll(mirror, -200, 0) * 0.45, img)
    else:
        mm = poly_below(prof)
        img = img * (1 - mm[..., None]) + col(ground) * mm[..., None]
        # lit windows
        for k in range(26):
            x0 = int(r.integers(40, 1360))
            wb = int(r.integers(40, 90))
            hb = int(r.integers(80, 330))
            yb = int(np.interp(x0, np.linspace(0, W, 720), prof)) + 40
            img[yb - hb:yb + 400, x0:x0 + wb] = col(ground) * 0.9
    return img, 1.2, 0.3


GENS["lightning"] = lambda s, p: g_lightning(s, **p)


def g_clouds(seed, mode, sky, cloud, sun=None, moon=False):
    r = np.random.default_rng(seed)
    img = vgrad(sky)
    cx = 3.0
    dens_n = fbm(seed, cx, 6, cy=cx * 3.4)
    wnx = fbm(seed + 2, 2.0, 3)
    dens_n = warp_sample(dens_n, wnx, wnx, 160)
    gate = smooth(0.0, 1.0, 1 - np.abs(YY / H - (0.6 if mode == "sunset" else 0.62)) * 2.0)
    dens = smooth(0.40 if mode == 'moon' else 0.45, 0.72 if mode == 'moon' else 0.78, dens_n * (0.55 + 0.6 * gate))
    if mode == "sunset":
        sx, sy = 720, 1500
        glow(img, sx, sy, 380, col("#FFB070"), 0.9, "exp")
        disc_m = 1 - smooth(78, 82, np.sqrt((XX - sx) ** 2 + (YY - sy) ** 2))
        img += disc_m[..., None] * col("#FFF0C8") * 1.3
        lx, ly = 0.0, -1.0
    elif mode == "moon":
        sx, sy = 980, 560
        glow(img, sx, sy, 300, col("#9FB8FF"), 0.7, "exp")
        disc_m = 1 - smooth(98, 102, np.sqrt((XX - sx) ** 2 + (YY - sy) ** 2))
        img += disc_m[..., None] * col("#F4F6FF") * 1.2
        stars(img, seed + 4, 900, ymin=0, ymax=H * 0.6, amp=(0.2, 0.8), size=0.7)
        lx, ly = 0.6, -0.7
    else:
        lx, ly = 0.2, -1.0
    M = np.float32([[1, 0, -lx * 26], [0, 1, -ly * 26]])
    sh1 = cv2.warpAffine(dens, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    M2 = np.float32([[1, 0, -lx * 70], [0, 1, -ly * 70]])
    sh2 = cv2.warpAffine(dens, M2, (W, H), borderMode=cv2.BORDER_REFLECT)
    occl = np.clip(blur(sh1, 14) * 0.6 + blur(sh2, 30) * 0.7, 0, 1)
    lit = np.clip(1.0 - occl * 1.25, 0, 1)
    body = ramp(np.clip(lit, 0, 1), cloud)
    if mode == "storm":
        body *= (0.55 + 0.6 * dens_n[..., None])
    img = img * (1 - dens[..., None]) + body * dens[..., None]
    rimd = np.clip(dens - blur(dens, 6), 0, 1) * lit
    rimc = col("#FFC080") if mode == "sunset" else (col("#BFD4FF") if mode == "moon" else col("#BFFFE0"))
    img += rimd[..., None] * rimc * 1.2
    if mode == "storm":
        gap = np.clip(1.0 - dens, 0, 1)
        ang = np.arctan2(YY - (-300), XX - 720)
        rays = 0.5 + 0.5 * np.sin(ang * 16 + fbm(seed + 8, 2, 2) * 6)
        rr = smooth(0.3, 1.0, rays) * np.exp(-(YY / 1500.0)) * gap * smooth(0.2, 0.9, 1 - dens_n)
        img += rr[..., None] * col("#9FFFD0") * 0.12
    return img, 1.2, 0.25


GENS["clouds"] = lambda s, p: g_clouds(s, **p)


def g_caustics(seed, mode, pal, tile="#E6F4F8"):
    r = np.random.default_rng(seed)
    c = np.ones((H, W), np.float32)
    nets = []
    for lvl, st in enumerate((170, 120)):
        P = [(x + (r.random() - 0.5) * st * 0.9, y + (r.random() - 0.5) * st * 0.9) for y in np.arange(-st, H + st, st) for x in np.arange(-st, W + st, st)]
        d1, d2, _ = voronoi([p[0] for p in P], [p[1] for p in P], ds=3)
        net = np.exp(-(d2 - d1) / 7.0)
        net = blur(net, 1.2)
        wx = fbm(seed + 10 + lvl, 3, 3)
        wy = fbm(seed + 20 + lvl, 3, 3)
        mx = (XX + (wx - 0.5) * 90).astype(np.float32)
        my = (YY + (wy - 0.5) * 90).astype(np.float32)
        nets.append(cv2.remap(net.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))
    c = np.clip(nets[0] * 0.8 + nets[1] * 0.6, 0, 1)
    c = blur(c, 1.0) ** 1.2
    if mode == "pool":
        gx = np.minimum(XX % 180, 180 - XX % 180)
        gy = np.minimum(YY % 180, 180 - YY % 180)
        gl = 1 - smooth(2, 7, np.minimum(gx, gy))
        wob = fbm(seed, 3, 3)
        base = ramp(np.clip(0.3 + 0.5 * (YY / H) + (wob - 0.5) * 0.3, 0, 1), pal)
        floor_ = base * (0.8 + 0.2 * (1 - gl[..., None])) * (1 - 0.45 * gl[..., None])
        img = floor_ + c[..., None] * col(tile) * 0.75
        return img, 1.1, 0.25
    if mode == "shallow":
        base = vgrad(pal)
        sand = fbm(seed + 3, 30, 4, cy=50)
        img = base * (0.85 + 0.25 * sand[..., None])
        img += c[..., None] * col("#E8FFF8") * 0.9
        return img, 1.1, 0.25
    # deep: god rays
    base = vgrad(pal)
    ang = (XX - 900) / np.maximum(YY + 400, 1)
    rays = 0.5 + 0.5 * np.sin(ang * 26 + fbm(seed + 2, 3, 3) * 5)
    rays = smooth(0.5, 1.0, rays) * np.exp(-YY / 1500.0)
    base += rays[..., None] * col("#7FD8FF") * 0.22
    top = np.exp(-YY / 700.0)
    base += (c * top)[..., None] * col("#BFF4FF") * 0.45
    r = np.random.default_rng(seed)
    for i in range(110):
        bx, by, br = r.random() * W, r.random() * H, 3 + r.random() ** 3 * 18
        glow(base, bx, by, br, col("#BFEFFF"), 0.35, "gauss")
        glow(base, bx - br * 0.3, by - br * 0.3, br * 0.3, col("#FFFFFF"), 0.8, "gauss")
    return base, 1.2, 0.35


GENS["caustics"] = lambda s, p: g_caustics(s, **p)


def g_equalizer(seed, mode, bg, pal, n=24, seg=18):
    r = np.random.default_rng(seed)
    img = vgrad(bg)
    if mode == "led":
        gap = 14
        bw = (W - 120) / n
        base_y = 2000
        sh = 34
        refl = np.zeros_like(img)
        env = np.exp(-((np.arange(n) - n * 0.5) / (n * 0.35)) ** 2)
        for i in range(n):
            hseg = int(2 + (seg - 2) * min(1, 0.25 + r.random() * 0.9) * (0.5 + 0.6 * env[i]))
            hseg = min(hseg, seg)
            x0 = 60 + i * bw
            for k in range(hseg):
                y1 = base_y - k * (sh + 6)
                t = k / (seg - 1)
                c = ramp(np.array([t], np.float32), pal)[0]
                a = 0.55 + 0.45 * (k == hseg - 1)
                img[int(y1 - sh):int(y1), int(x0 + gap / 2):int(x0 + bw - gap / 2)] = c * a
                # reflection
                ry = base_y + 40 + k * (sh + 6) * 0.5
                if ry + sh * 0.5 < H:
                    img[int(ry):int(ry + sh * 0.5), int(x0 + gap / 2):int(x0 + bw - gap / 2)] = c * 0.18 * (1 - k / seg)
        L = np.clip(img - vgrad(bg), 0, None)
        img += blur(L, 8) * 0.9 + blur(L, 30) * 0.6
        return img, 1.1, 0.25
    if mode == "wave":
        m = Image.new("L", (W * 2, H * 2), 0)
        d = ImageDraw.Draw(m)
        n = 58
        bw = (W - 140) / n
        for i in range(n):
            u = (i - n / 2) / (n / 2)
            h = (60 + 800 * (np.exp(-u * u * 2.2) * (0.35 + 0.65 * r.random()) + 0.06 * r.random()))
            x = 70 + i * bw + bw / 2
            d.rounded_rectangle([(x - bw * 0.3) * 2, (1280 - h / 2) * 2, (x + bw * 0.3) * 2, (1280 + h / 2) * 2], radius=bw * 0.3 * 2, fill=255)
        mm = np.asarray(m.resize((W, H), Image.BOX), np.float32) / 255.0
        t = np.clip(np.abs(YY - 1280) / 520.0, 0, 1)
        c = ramp(t, pal)
        img = img * (1 - mm[..., None]) + c * mm[..., None]
        img += blur(mm, 18)[..., None] * c * 0.7
        return img, 1.0, 0.25
    # radial
    cx, cy = 720, 1160
    m = Image.new("L", (W * 2, H * 2), 0)
    d = ImageDraw.Draw(m)
    nb = 120
    for i in range(nb):
        a = i / nb * 2 * math.pi
        amp = 40 + 330 * (0.3 + 0.7 * abs(math.sin(a * 3 + 1))) * (0.4 + 0.6 * r.random())
        r0 = 330
        d.line([((cx + math.cos(a) * r0) * 2, (cy + math.sin(a) * r0) * 2), ((cx + math.cos(a) * (r0 + amp)) * 2, (cy + math.sin(a) * (r0 + amp)) * 2)], fill=255, width=12)
    mm = np.asarray(m.resize((W, H), Image.BOX), np.float32) / 255.0
    ang = (np.arctan2(YY - cy, XX - cx) / (2 * math.pi)) % 1.0
    c = ramp(np.abs(2 * ang - 1), pal)
    img += mm[..., None] * c * 1.1 + blur(mm, 16)[..., None] * c * 0.9
    rr = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    ring = np.exp(-((rr - 300) / 5) ** 2)
    img += ring[..., None] * col(pal[0][1]) * 0.7 + (np.exp(-(rr / 260) ** 2) * 0.12)[..., None] * col(pal[-1][1])
    return img, 1.0, 0.25


GENS["equalizer"] = lambda s, p: g_equalizer(s, **p)


def g_plaid(seed, mode, a, b, c="#000000", d_="#FFFFFF", scale=1.0):
    if mode == "tartan":
        sett = [(a, 36), (b, 8), (a, 36), (c, 6), (b, 44), (d_, 4), (b, 44), (c, 6), (a, 36)]
        total = sum(w for _, w in sett)
        pat = []
        for cc, w in sett:
            pat += [col(cc)] * int(w * scale * 2)
        pat = np.array(pat, np.float32)
        n = len(pat)
        ix = (XX.astype(np.int32)) % n
        iy = (YY.astype(np.int32)) % n
        warp = pat[ix]
        weft = pat[iy]
        twill = ((XX.astype(np.int32) + YY.astype(np.int32)) // 2) % 2
        cx_ = (XX.astype(np.int32) % 4 < 2).astype(np.float32)
        use_w = (((XX.astype(np.int32) + YY.astype(np.int32)) % 4) < 2)[..., None]
        img = np.where(use_w, warp, weft) * 0.9 + (warp + weft) * 0.05
        thr = fbm(seed, 300, 1, cy=420)
        img *= (0.88 + 0.2 * thr[..., None])
        img *= (0.9 + 0.1 * np.sin(XX * 1.57) * np.sin(YY * 1.57))[..., None] if False else 1
        fuzz = blur(fbm(seed + 2, 6, 3), 1) * 0.0
        return img, 2.0, 0.2
    if mode == "argyle":
        s = 230 * scale
        u = (XX / s) + (YY / (s * 1.7))
        v = (XX / s) - (YY / (s * 1.7))
        fu = u - np.floor(u)
        fv = v - np.floor(v)
        par = (np.floor(u) + np.floor(v)) % 2
        img = np.where(par[..., None] > 0.5, col(a), col(b))
        dd = np.abs(u - np.round(u)) * 0 + np.minimum(np.abs(((XX + 0.0) / s) % 1.0 - 0.5), 9)
        # diamond overlay lines
        ln = (np.abs(((XX / s) % 1.0) - 0.5) < 0.0)
        cxl = np.abs((((XX / s) + (YY / (s * 1.7)) * 0) % 1.0) - 0.5)
        dmd = np.abs(((XX / (s * 0.5)) % 1.0) - 0.5) / 0.5 + np.abs(((YY / (s * 0.85)) % 1.0) - 0.5) / 0.5
        dm = ((np.floor(XX / (s * 0.5)) + np.floor(YY / (s * 0.85))) % 2)
        diam = np.abs(((XX / s) % 1.0) - 0.5) * 2 + np.abs(((YY / (s * 1.7)) % 1.0) - 0.5) * 2
        diam_f = diam < 1.0
        img = np.where(diam_f[..., None], col(a), col(b))
        stitch = (np.abs(diam - 1.0) < 0.03) & (((XX + YY) % 14) < 8)
        img = np.where(stitch[..., None], col(d_), img)
        cross = ((np.abs(((XX / s) % 1.0) - 0.5) < 0.012) | (np.abs(((YY / (s * 1.7)) % 1.0) - 0.5) < 0.006)) & diam_f
        img = np.where(cross[..., None], col(c), img)
        fine = fbm(seed, 260, 1, cy=360)
        img = img * (0.9 + 0.15 * fine[..., None]) * (0.92 + 0.08 * np.sin(XX * 1.2) * np.sin(YY * 1.2) ** 2)[..., None]
        return img, 1.6, 0.2
    # houndstooth
    s = 64 * scale
    xs = (XX / s) % 4
    ys = (YY / s) % 4
    xi, yi = np.floor(xs), np.floor(ys)
    fx, fy = xs - xi, ys - yi
    base = ((xi + yi) % 2)
    tooth = np.where((fx + fy < 1.0) & ((fx > 0.5 * 0) ), 1, 0)
    pat = (((xi + yi) % 2) == 0)
    tri = (fx + fy) < 1.0
    tri2 = (fx - fy) > 0
    val = np.where(pat, np.where(tri, 1, 0), np.where(tri2, 1, 0))
    band = ((xi - yi) % 4) < 2
    val = np.where(((xi + yi) % 2) == 0, np.where(fx + fy < 1.0, 1, 0), np.where(fx > fy, 1, 0))
    # classic: 4x4 checker with teeth; approximate with offset diagonal checker
    chk = ((np.floor(XX / s) + np.floor(YY / s)) % 2).astype(np.float32)
    teeth = (((XX / s) % 1.0) + ((YY / s) % 1.0) < 1.0).astype(np.float32)
    mix = np.where(chk > 0.5, teeth, 1 - teeth)
    stripes = ((np.floor(XX / s + YY / s) % 2)).astype(np.float32)
    v = np.clip(mix * 0.0 + np.where(((np.floor(XX / (2 * s)) + np.floor(YY / (2 * s))) % 2) > 0.5, mix, 1 - mix), 0, 1)
    img = col(a) * (1 - v[..., None]) + col(b) * v[..., None]
    fine = fbm(seed, 280, 1, cy=380)
    img *= (0.88 + 0.2 * fine[..., None])
    g = fbm(seed + 3, 3.0, 3)
    img *= (0.8 + 0.3 * g[..., None])
    return img, 1.6, 0.25


GENS["plaid"] = lambda s, p: g_plaid(s, **p)



def auto_swatch(im):
    """Four representative colours (dark to light) from a rendered wallpaper."""
    small = np.asarray(im.resize((54, 96), Image.BOX), np.float32).reshape(-1, 3)
    cv2.setRNGSeed(7)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, lab, cen = cv2.kmeans(small, 4, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    cen = cen[np.argsort(cen @ np.array([0.299, 0.587, 0.114], np.float32))]
    return ["#%02X%02X%02X" % tuple(int(round(v)) for v in c) for c in cen]


item('stained-glass-arch', 'Stained Glass Arch', 'A cathedral window of jewel-toned glass held in dark lead lines.', 'abstract', True, True, None, 0.1, 'vglass', **{'mode': 'arch', 'pal': [(0, '#1B3FA8'), (0.3, '#C0262F'), (0.55, '#E8A82A'), (0.8, '#2A9A6A'), (1, '#7A3AA8')]})
item('shattered-sunset', 'Shattered Sunset', 'Cracked glass shards fan out from one impact in violet, coral and gold.', 'abstract', True, False, None, 0.1, 'vglass', **{'mode': 'shards', 'pal': [(0, '#1A0B3A'), (0.35, '#7A1F6E'), (0.65, '#E8503A'), (1, '#FFC25A')]})
item('glacier-facets', 'Glacier Facets', 'Bevelled blue ice cells catch cold light along bright frozen seams.', 'abstract', False, False, None, 0.12, 'vglass', **{'mode': 'ice', 'pal': [(0, '#0B2A5A'), (0.4, '#2F7DC2'), (0.75, '#7FD0E8'), (1, '#E8FAFF')]})
item('neon-flow-field', 'Neon Flow Field', 'Thousands of fine cyan and violet threads follow an invisible current.', 'abstract', True, False, None, 0.12, 'flow', **{'mode': 'field', 'bg': '#03040C', 'c1': '#2BD0FF', 'c2': '#C03CFF', 'n': 3500})
item('ink-currents', 'Ink Currents', 'Hand-drawn navy ink lines stream past a golden sun on warm paper.', 'abstract', False, False, None, 0.25, 'flow', **{'mode': 'ink', 'bg': '#EFE3C8', 'c1': '#14213D', 'c2': '#2B3A67', 'n': 700, 'steps': 170, 'step_len': 9, 'width': 2, 'accent': '#D8432B', 'sun': (1000, 700, 260, '#E8B84A'), 'cx': 1.6})
item('twin-vortex', 'Twin Vortex', 'Pink and amber streamlines spiral around two counter-rotating eyes.', 'abstract', True, False, None, 0.12, 'flow', **{'mode': 'swirl', 'bg': '#080314', 'c1': '#FF4FA8', 'c2': '#FFD04A', 'n': 3800, 'steps': 150, 'step_len': 6})
item('leopard-spots', 'Leopard Spots', 'Self-organising brown dots settle into an even coat on warm sand.', 'abstract', False, False, None, 0.2, 'gscott', **{'f': 0.0367, 'k': 0.0649, 'steps': 7000, 'nseed': 220, 'pal': [(0, '#E8D2A4'), (0.5, '#C9904A'), (1, '#3A1E10')], 'mode': 'plain'})
item('turing-maze', 'Turing Maze', 'A glossy teal labyrinth grown by a reaction and diffusion simulation.', 'abstract', True, False, None, 0.12, 'gscott', **{'f': 0.029, 'k': 0.057, 'steps': 5000, 'pal': [(0, '#04121C'), (0.5, '#0C5E6E'), (1, '#7FFFE0')], 'mode': 'emboss'})
item('coral-bloom', 'Coral Bloom', 'Glowing magenta coral ridges wind across a deep blue sea.', 'abstract', True, False, None, 0.12, 'gscott', **{'f': 0.0545, 'k': 0.062, 'steps': 6000, 'nseed': 160, 'pal': [(0, '#05061E'), (0.5, '#4A1A8A'), (1, '#FF4FB0')], 'mode': 'glow'})
item('comic-burst', 'Comic Burst', 'Halftone rays, a bold starburst and a speech bubble in pop colours.', 'patterns', True, False, None, 0.25, 'popart', **{'mode': 'burst', 'pal': ['#FFF4D8', '#FFD21F', '#E4262F', '#1A6BD6', '#FF7FB0']})
item('pop-panels', 'Pop Panels', 'Three slanted comic panels with a sun, a lightning bolt and dots.', 'patterns', False, False, None, 0.25, 'popart', **{'mode': 'panels', 'pal': ['#FFF4D8', '#FFD21F', '#E4262F', '#1A6BD6', '#FF7FB0']})
item('print-misregister', 'Print Misregister', 'Soft colour blobs printed as offset halftone screens on cream paper.', 'patterns', False, False, None, 0.3, 'popart', **{'mode': 'cmyk', 'pal': ['#F6EFDD', '#E8318A', '#12A8DC', '#FFE31A']})
item('indigo-waves', 'Indigo Waves', 'Classic overlapping wave scales in indigo and white fade into the dark.', 'patterns', False, False, None, 0.15, 'seigaiha', **{'R': 110, 'cols_': ['#F2EEE0', '#1C3A7A', '#F2EEE0', '#1C3A7A', '#F2EEE0'], 'bgc': '#10244F', 'fade': (1.0, 0.45)})
item('red-sun-waves', 'Red Sun Waves', 'A crimson sun rises over a sea of red and gold scalloped waves.', 'patterns', True, False, None, 0.25, 'seigaiha', **{'R': 96, 'cols_': ['#F6E3B4', '#C42B2B', '#F6E3B4', '#9E1F26'], 'bgc': '#F0A23C', 'mode': 'horizon', 'sun': (720, 900, 430, '#E83A2A'), 'fade': (1.0, 0.5)})
item('graphite-scales', 'Graphite Scales', 'Tiny silver and charcoal wave scales fading from light to shadow.', 'patterns', False, False, None, 0.12, 'seigaiha', **{'R': 60, 'cols_': ['#D5D9E0', '#3A404E', '#D5D9E0', '#262B38', '#D5D9E0', '#171B26'], 'bgc': '#0B0E16', 'fade': (1.5, 0.7)})
item('star-chart', 'Star Chart', 'A blue celestial chart with degree ticks and linked constellations.', 'space', True, False, None, 0.12, 'constel', **{'bgstops': [(0, '#071A44'), (1, '#0A2E6B')], 'line': '#7FB0FF', 'star': '#E8F2FF', 'chart': 'blue'})
item('gold-zodiac', 'Gold Zodiac', 'Golden star lines trace a ring of constellations on black.', 'space', True, False, None, 0.1, 'constel', **{'bgstops': [(0, '#05050A'), (1, '#0E0A14')], 'line': '#E2B04A', 'star': '#FFE6A8', 'chart': 'gold', 'n': 80})
item('neon-constellation', 'Neon Constellation', 'Pink threads link glittering cyan stars into a drifting network.', 'space', False, False, None, 0.1, 'constel', **{'bgstops': [(0, '#12031E'), (1, '#050818')], 'line': '#FF5FC8', 'star': '#9FF4FF', 'chart': 'net', 'n': 160})
item('pixel-skyline', 'Pixel Skyline', 'A dithered purple dusk over a lit pixel-art city and a pale moon.', 'urban', True, False, None, 0.15, 'pixscene', **{'scene': 'city'})
item('pixel-lighthouse', 'Pixel Lighthouse', 'A striped lighthouse watches a calm pixel sea at golden hour.', 'landscapes', True, True, None, 0.15, 'pixscene', **{'scene': 'harbour'})
item('pixel-desert', 'Pixel Desert', 'Mesas, cacti and a huge moon in a 16-bit desert night.', 'landscapes', False, False, None, 0.12, 'pixscene', **{'scene': 'desert'})
item('green-pcb', 'Green PCB', 'Parallel gold traces and black chips on a deep green circuit board.', 'gaming', False, False, None, 0.15, 'circuit', **{'mode': 'bus', 'bg': '#0B3B24', 'trace': '#C9A24A', 'pad': '#E8D08A', 'chip': '#0F1512'})
item('neon-chip', 'Neon Chip', 'Glowing cyan traces fan out from a central processor.', 'gaming', True, False, None, 0.1, 'circuit', **{'mode': 'radial', 'bg': '#05080F', 'trace': '#1C6A8A', 'pad': '#27E6FF', 'chip': '#0A1220', 'glowc': '#27C8FF'})
item('copper-board', 'Copper Board', 'A tangle of orange copper tracks and chips glowing on black.', 'gaming', True, False, None, 0.1, 'circuit', **{'mode': 'rand', 'bg': '#120A06', 'trace': '#B8651F', 'pad': '#FFB060', 'chip': '#1C120C', 'glowc': '#FF7A18'})
item('papercut-peaks', 'Papercut Peaks', 'Layers of rose and plum paper ridges cast soft shadows below a pale moon.', 'landscapes', True, False, None, 0.15, 'papercut', **{'mode': 'mountains', 'sky': [(0, '#F7C9A0'), (1, '#FBE3C2')], 'layers': [(0, '#F2A58E'), (0.4, '#C9657A'), (0.7, '#7A3F7A'), (1, '#2A1A4A')], 'sun': (900, 520, 190, '#FFF1D6'), 'n': 7})
item('papercut-sea', 'Papercut Sea', 'Layered blue paper waves roll beneath a bright orange sun.', 'landscapes', False, False, None, 0.15, 'papercut', **{'mode': 'waves', 'sky': [(0, '#FFE9B8'), (1, '#FFD08A')], 'layers': [(0, '#9FD8E0'), (0.5, '#2F8FB0'), (1, '#0B3A6E')], 'sun': (720, 700, 330, '#FF7A4A'), 'n': 8})
item('papercut-forest', 'Papercut Forest', 'Rows of jagged pine paper in teal under a round moon.', 'landscapes', True, False, None, 0.12, 'papercut', **{'mode': 'forest', 'sky': [(0, '#0E2B3A'), (1, '#7ACFC0')], 'layers': [(0, '#4FA89A'), (0.5, '#1F6A60'), (1, '#0A2A2E')], 'sun': (430, 620, 150, '#F4F8E0'), 'n': 7})
item('watercolor-dawn', 'Watercolor Dawn', 'Wet pink, gold and blue washes bleed above soft green hills.', 'landscapes', True, False, None, 0.25, 'watercolor', **{'mode': 'sky', 'washes': [(0.0, 0.3, '#F4A3A8', 0.8), (0.2, 0.5, '#F7C37A', 0.7), (0.4, 0.75, '#8DB8E0', 0.7), (0.6, 0.8, '#B79CE0', 0.6)]})
item('watercolor-blooms', 'Watercolor Blooms', 'Round pools of teal, coral and gold pigment on textured paper.', 'abstract', False, False, None, 0.3, 'watercolor', **{'mode': 'blooms', 'washes': [('#2BA7A0', 0.8), ('#F26D5B', 0.8), ('#F2C14E', 0.75), ('#6A5ACD', 0.7), ('#E8789A', 0.7), ('#3C7DC4', 0.7), ('#6BB36A', 0.65)]})
item('watercolor-rain', 'Watercolor Rain', 'Tall streaks of indigo and violet run down the page like rain.', 'abstract', False, False, None, 0.3, 'watercolor', **{'mode': 'streaks', 'paper': '#EEF0F2', 'washes': [('#3B4CCA', 0.8), ('#7A5CC8', 0.7), ('#1E88B8', 0.7), ('#C0488A', 0.5)]})
item('color-bars', 'Color Bars', 'A bent CRT screen shows broadcast colour bars with scanline fuzz.', 'retro', True, False, None, 0.15, 'testcard', **{'mode': 'bars'})
item('tv-static', 'TV Static', 'Cool blue analogue static rolls across a curved tube screen.', 'retro', False, False, None, 0.12, 'testcard', **{'mode': 'snow', 'tint': '#7FD8FF'})
item('amber-scope', 'Amber Scope', 'An amber monitor shows a geometry test grid with glowing circles.', 'retro', False, False, None, 0.12, 'testcard', **{'mode': 'amber', 'tint': '#FFB020', 'bg': '#120900'})
item('tile-mosaic', 'Tile Mosaic', 'Square glass tiles ripple outward in rings of blue and gold.', 'patterns', False, False, None, 0.2, 'mosaic', **{'mode': 'square', 'pal': [(0, '#0B1E5A'), (0.4, '#1F7AC0'), (0.7, '#F2C94C'), (1, '#FFF3C0')], 'field': 'radial'})
item('hex-tiles', 'Hex Tiles', 'Round ceramic tiles shade from plum to peach on a hex grid.', 'patterns', True, False, None, 0.2, 'mosaic', **{'mode': 'hex', 's': 60, 'pal': [(0, '#2A0E3E'), (0.35, '#B83A6E'), (0.7, '#F58A5A'), (1, '#FFE0A0')]})
item('pebble-mosaic', 'Pebble Mosaic', 'Hundreds of earthy stones set in dark grout, in Roman style.', 'patterns', True, False, None, 0.2, 'mosaic', **{'mode': 'pebble', 'pal': [(0, '#4A3A2E'), (0.35, '#8A6E52'), (0.65, '#C9B08A'), (1, '#E8DCC4')], 'grout': '#2A2118'})
item('paper-pleats', 'Paper Pleats', 'Wavy accordion folds of pastel paper catch light on one side.', 'patterns', False, False, None, 0.15, 'paperfold', **{'mode': 'pleat', 'pal': [(0, '#F7B7C8'), (0.5, '#F4D6B0'), (1, '#9EC9E8')], 'pw': 110, 'ang': 0.35, 'amp': 60})
item('folded-chevrons', 'Folded Chevrons', 'A Miura-style fold in blue and mint zigzags down the screen.', 'patterns', True, False, None, 0.15, 'paperfold', **{'mode': 'miura', 'pal': [(0, '#1B2A6B'), (0.5, '#3C8FC0'), (1, '#BDEFD8')], 'pw': 120, 'amp': 0.55})
item('crumpled-paper', 'Crumpled Paper', 'Soft creases and shadows on a sheet of warm cream paper.', 'patterns', False, False, None, 0.25, 'paperfold', **{'mode': 'crumple', 'pal': [(0, '#F4EDE0'), (0.5, '#F1D9C0'), (1, '#E6C3B3')]})
item('glitch-sun', 'Glitch Sun', 'A synthwave sun torn into shifted slices with RGB split.', 'gaming', False, False, None, 0.12, 'glitch', **{'mode': 'sun', 'pal': ['#0B0620', '#7A1A8A', '#FF4F8B', '#FFD04A', '#FF3E8E']})
item('data-blocks', 'Data Blocks', 'Corrupted neon rectangles scatter across a dark navy screen.', 'gaming', False, False, None, 0.1, 'glitch', **{'mode': 'blocks', 'pal': ['#05060F', '#101840', '#27E6FF', '#FF2DB5', '#C6FF2D'], 'nslice': 40, 'maxshift': 120})
item('pixel-sort', 'Pixel Sort', 'Smeared columns of violet, pink and amber melt like a sorted image.', 'abstract', False, False, None, 0.12, 'glitch', **{'mode': 'sort', 'pal': [(0, '#08051A'), (0.35, '#2E1A8A'), (0.6, '#E8318A'), (0.85, '#FFB347'), (1, '#FFF3C0')]})
item('iso-night-city', 'Isometric Night City', 'A neon-edged isometric city grid with thousands of lit windows.', 'urban', True, False, None, 0.1, 'isocity', **{'bgstops': [(0, '#0A0D2A'), (1, '#241048')], 'top': '#2A3566', 'left': '#161C40', 'right': '#0E1230', 'win': ['#FFD27A', '#FFB45A', '#7AD8FF'], 'road': ('#0A0C20', '#27E6FF'), 'mode': 'rand'})
item('iso-day-city', 'Isometric Day City', 'Pale blue towers rise in a bright isometric downtown.', 'urban', False, False, None, 0.2, 'isocity', **{'bgstops': [(0, '#BFE6FF'), (1, '#F8EBD8')], 'top': '#F6F1E8', 'left': '#C9D7E8', 'right': '#9FB6D2', 'win': ['#6B8FB8', '#8FB0D0'], 'win_p': 0.5, 'road': ('#8A94A6', '#F4F4F4'), 'mode': 'grid'})
item('iso-dusk-towers', 'Dusk Towers', 'A stepped pyramid of towers glows orange against a violet dusk.', 'urban', True, True, None, 0.12, 'isocity', **{'bgstops': [(0, '#2B1055'), (0.5, '#C2406E'), (1, '#FFB070')], 'top': '#FFC890', 'left': '#8A3A7E', 'right': '#4A1E6A', 'win': ['#FFE6A8', '#FFFFFF'], 'win_p': 0.3, 'mode': 'center', 'haze': ('#FF9A7A', 1.2)})
item('lava-lamp', 'Lava Lamp', 'Warm molten blobs drift and merge in a purple glass.', 'abstract', False, False, None, 0.12, 'metaballs', **{'n': 10, 'rmin': 110, 'rmax': 250, 'mode': 'lava', 'pal': [(0, '#FF2D6A'), (0.5, '#FF7A2D'), (1, '#FFE066')], 'bg': [(0, '#2A0A4A'), (1, '#0E0620')]})
item('liquid-chrome', 'Liquid Chrome', 'A polished mercury shape reflects a cool studio sky.', 'abstract', True, False, None, 0.1, 'metaballs', **{'mode': 'chrome', 'pal': [(0, '#06080F'), (0.35, '#3A4A6A'), (0.55, '#C8D8F0'), (0.7, '#F4F8FF'), (1, '#6A7A9A')], 'bg': [(0, '#10141E'), (1, '#05060B')], 'n': 5, 'rmin': 200, 'rmax': 420})
item('oil-film', 'Oil Film', 'Iridescent soap-film swirls shimmer over a dark blue base.', 'abstract', False, False, None, 0.12, 'metaballs', **{'mode': 'bubbles', 'pal': [(0, '#4FE0FF'), (0.35, '#C06BFF'), (0.7, '#FF6BB0'), (1, '#FFE08A')], 'bg': [(0, '#0A1030'), (1, '#05050F')]})
item('guilloche-rosette', 'Guilloche Rosette', 'Fine gold curves interlace into a banknote-style rosette.', 'abstract', True, False, None, 0.1, 'guilloche', **{'mode': 'rosette', 'bg': '#0A0710', 'c1': '#E2B04A', 'c2': '#FFE9A8', 'n': 110})
item('spirograph-trio', 'Spirograph Trio', 'Three glowing spirograph rosettes in cyan and magenta on black.', 'abstract', False, False, None, 0.1, 'guilloche', **{'mode': 'spiro', 'bg': '#05060F', 'c1': '#27E6FF', 'c2': '#FF4FB0'})
item('harmonograph', 'Harmonograph', 'Decaying pendulum curves trace amber loops that fade to red.', 'abstract', False, False, None, 0.1, 'guilloche', **{'mode': 'harmo', 'bg': '#0E0A06', 'c1': '#FFB347', 'c2': '#FF3E6A'})
item('frost-kaleidoscope', 'Frost Kaleidoscope', 'Six-fold ice crystals glow in icy blue and white.', 'abstract', False, False, None, 0.12, 'kaleido', **{'folds': 6, 'pal': [(0, '#0B1E5A'), (0.4, '#27B7E8'), (0.7, '#E8F8FF'), (1, '#FFFFFF')], 'bands': 4})
item('ember-mandala', 'Ember Mandala', 'An eight-fold mandala of magenta, orange and cream light.', 'abstract', True, False, None, 0.12, 'kaleido', **{'folds': 8, 'pal': [(0, '#2A0A3E'), (0.4, '#C0306E'), (0.75, '#FF9A4A'), (1, '#FFE9B0')], 'bands': 7, 'zoom': 1.4})
item('jade-flower', 'Jade Flower', 'A twelve-petal kaleidoscope in jade green and lime.', 'abstract', True, False, None, 0.12, 'kaleido', **{'folds': 12, 'pal': [(0, '#052A1E'), (0.4, '#18A878'), (0.7, '#B8F060'), (1, '#FFFFD0')], 'bands': 3, 'zoom': 0.8})
item('moire-rings', 'Moire Rings', 'Two fine ring grids interfere into shifting blue ripples.', 'patterns', False, False, None, 0.15, 'moire', **{'mode': 'rings', 'pal': [(0, '#1A1060'), (0.5, '#4F7CFF'), (1, '#FFD0F0')]})
item('moire-waves', 'Moire Waves', 'Bent red line gratings create flowing optical waves.', 'patterns', False, False, None, 0.2, 'moire', **{'mode': 'lines', 'p1': 9, 'p2': 9.4, 'pal': [(0, '#2A0A0A'), (0.5, '#E8502A'), (1, '#FFE0A0')]})
item('moire-grid', 'Moire Grid', 'Two rotated mint dot grids form a woven interference pattern.', 'patterns', False, False, None, 0.12, 'moire', **{'mode': 'dots', 'p1': 22, 'p2': 23, 'pal': [(0, '#02150E'), (0.5, '#18C890'), (1, '#E0FFF0')]})
item('arc-weave', 'Arc Weave', 'Curving Truchet arcs weave a rainbow ribbon over deep navy.', 'patterns', True, False, None, 0.12, 'truchet', **{'mode': 'weave', 'bg': '#0C0F22', 'pal': [(0, '#FF4F8B'), (0.5, '#FFB347'), (1, '#27E6FF')], 's': 96, 'wd': 0.1})
item('maze-lines', 'Maze Lines', 'Diagonal red strokes form an endless maze on cream paper.', 'patterns', False, False, None, 0.25, 'truchet', **{'mode': 'maze', 'bg': '#F2E8D2', 'pal': [(0, '#1A2E5A'), (1, '#D8432B')], 's': 64, 'wd': 0.09})
item('tangled-tiles', 'Tangled Tiles', 'Truchet arcs at three scales in violet, mint and gold.', 'patterns', False, False, None, 0.12, 'truchet', **{'mode': 'multi', 'bg': '#0E0B18', 'pal': [(0, '#8A5CFF'), (0.5, '#27E6B0'), (1, '#FFD04A')], 's': 120})
item('seahorse-valley', 'Seahorse Valley', 'A deep zoom into the Mandelbrot set, in blue, cream and amber.', 'abstract', True, True, None, 0.1, 'fractal', **{'mode': 'mandel', 'cxy': (-0.7436, 0.1318), 'width': 0.02, 'iters': 380, 'pal': [(0, '#02051E'), (0.3, '#1F4FC0'), (0.6, '#7FE0FF'), (0.85, '#FFE08A'), (1, '#FF7A3A')]})
item('julia-bloom', 'Julia Bloom', 'A violet and pink Julia set glows against a dark fade.', 'abstract', True, False, None, 0.1, 'fractal', **{'mode': 'julia', 'cxy': (0.0, 0.0), 'width': 3.2, 'iters': 160, 'c': (-0.8+0.156j), 'pal': [(0, '#12031E'), (0.4, '#8A2A9A'), (0.75, '#FF5FA8'), (1, '#FFE0F0')]})
item('burning-citadel', 'Burning Citadel', 'Fiery towers rise from the Burning Ship fractal.', 'abstract', True, False, None, 0.1, 'fractal', **{'mode': 'ship', 'cxy': (-1.762, -0.028), 'width': 0.1, 'iters': 300, 'pal': [(0, '#0A0200'), (0.35, '#8A1A0A'), (0.7, '#FF8A1A'), (1, '#FFF0A0')]})
item('retro-plasma', 'Retro Plasma', 'A smooth demoscene plasma of blue, pink and yellow.', 'retro', False, False, None, 0.2, 'plasma', **{'mode': 'classic', 'pal': [(0, '#0A0A5A'), (0.25, '#1AA0FF'), (0.5, '#FF2D95'), (0.75, '#FFD21F'), (1, '#0A0A5A')]})
item('posterized-plasma', 'Posterized Plasma', 'Flat poster-colour bands flow across the screen in violet and pink.', 'retro', False, False, None, 0.2, 'plasma', **{'mode': 'poster', 'pal': [(0, '#1A0B3A'), (0.33, '#5A2AB0'), (0.66, '#FF4F8B'), (1, '#FFD6A8')]})
item('plasma-contours', 'Plasma Contours', 'Glowing teal contour lines trace a rippling energy field.', 'retro', False, False, None, 0.12, 'plasma', **{'mode': 'contour', 'pal': [(0, '#03162A'), (0.5, '#0F6A8A'), (1, '#7FFFD8')]})
item('rainy-night-window', 'Rainy Night Window', 'Raindrops on glass magnify warm city bokeh after dark.', 'urban', True, False, None, 0.12, 'rainwin', **{'bgstops': [(0, '#0A0F2A'), (1, '#2A1648')], 'lights': ['#FFB070', '#FFD9A0', '#FF7A5A', '#8AD0FF']})
item('neon-rain', 'Neon Rain', 'Pink and cyan neon blurs through a rain-covered window.', 'urban', True, False, None, 0.1, 'rainwin', **{'bgstops': [(0, '#12031E'), (1, '#05091E')], 'lights': ['#FF2DB5', '#27E6FF', '#B07AFF', '#FF5E8E'], 'ndrops': 420})
item('misty-morning-glass', 'Misty Morning Glass', 'Soft dawn light glows through a foggy window dotted with drops.', 'urban', False, False, None, 0.2, 'rainwin', **{'bgstops': [(0, '#6A86B8'), (1, '#F2B5A0')], 'lights': ['#FFF0D0', '#FFD0A0', '#C8E0FF'], 'veil': 0.12, 'ndrops': 600, 'n_lights': 150})
item('highway-trails', 'Highway Trails', 'Long-exposure red and white light streaks race toward the horizon.', 'urban', True, False, None, 0.1, 'trails', **{'mode': 'highway', 'bg': [(0, '#05061A'), (0.46, '#1A1038'), (0.47, '#06050E'), (1, '#020208')], 'c1': '#FFE9B0', 'c2': '#FF2A3A'})
item('star-trails', 'Star Trails', 'Concentric star arcs circle the pole above a lone glowing tent.', 'space', True, True, None, 0.1, 'trails', **{'mode': 'star', 'bg': [(0, '#02030E'), (1, '#0B1330')], 'c1': '#BFD7FF', 'c2': '#FFE6C4', 'c3': '#FFFFFF'})
item('interchange-lights', 'Interchange Lights', 'Dashed headlight and taillight streaks loop through a night interchange.', 'urban', False, False, None, 0.1, 'trails', **{'mode': 'top', 'bg': [(0, '#07061A'), (1, '#0C0A24')], 'c1': '#FFF0C0', 'c2': '#FF3A4A'})
item('minimal-dunes', 'Minimal Dunes', 'Flat terracotta dunes roll beneath a pale sun.', 'landscapes', False, False, None, 0.2, 'minimal', **{'mode': 'dunes', 'sky': [(0, '#F7D6B0'), (1, '#F9B8A0')], 'c': ('#FFF0D8', [('#E9856A', 1500, 90, 6.0), ('#C4604F', 1800, 100, 4.5), ('#8E3E4A', 2100, 80, 5.5), ('#4A2440', 2350, 60, 3.5)])})
item('minimal-moonsea', 'Minimal Moon Sea', 'A cream moon above a sea drawn in simple bright stripes.', 'landscapes', False, False, None, 0.1, 'minimal', **{'mode': 'moonsea', 'sky': [(0, '#0B1235'), (1, '#27386E')], 'c': ('#F6F1DC', '#0A1230', '#6A86C8')})
item('minimal-peaks', 'Minimal Peaks', 'Violet two-tone pyramid mountains under a peach sunrise.', 'landscapes', False, False, None, 0.2, 'minimal', **{'mode': 'peaks', 'sky': [(0, '#FF9E80'), (0.5, '#FFC9A0'), (1, '#FFE8C8')], 'c': ('#FFF4E0', [('#7A5C9E', '#5A3E80', 380, 1960, 900, 620), ('#9E7AB8', '#7A5C9E', 1000, 1960, 1100, 760), ('#5A3E80', '#3C2860', 700, 1960, 620, 500)], '#2A1A48')})
item('blueprint-gears', 'Blueprint Gears', 'White technical drawings of gears and dimension lines on blue paper.', 'patterns', True, False, None, 0.15, 'blueprint', **{'mode': 'blue', 'bg': '#17479A', 'line': '#F2F6FF', 'line2': '#A8C4F0'})
item('chalkboard-sketches', 'Chalkboard Sketches', 'Chalk spirals, stars and circles on a dusty green board.', 'patterns', False, False, None, 0.2, 'blueprint', **{'mode': 'chalk', 'bg': '#16241E', 'line': '#F0F0E8'})
item('graph-paper', 'Graph Paper', 'Red and blue pen curves plotted over pale green graph paper.', 'patterns', False, False, None, 0.3, 'blueprint', **{'mode': 'graph', 'bg': '#F8F6EC', 'line': '#1A2A3A', 'line2': '#8FC8A8'})
item('amethyst-geode', 'Amethyst Geode', 'A sliced geode with violet bands and a sparkling quartz core.', 'abstract', True, True, None, 0.1, 'geode', **{'mode': 'slice', 'pal': [(0, '#1A0A2E'), (0.35, '#5A2A9A'), (0.7, '#B88AE8'), (1, '#EADAFF')], 'bands': 16})
item('blue-agate', 'Blue Agate', 'Rippling bands of polished blue agate shine with silver edges.', 'abstract', False, False, None, 0.12, 'geode', **{'mode': 'agate', 'pal': [(0, '#05203A'), (0.4, '#1A6AB0'), (0.75, '#8AD0F0'), (1, '#F4FBFF')], 'bands': 18, 'warp': 0.5})
item('malachite-swirl', 'Malachite Swirl', 'Deep green malachite bands twist into glossy whirls.', 'abstract', False, False, None, 0.12, 'geode', **{'mode': 'agate', 'pal': [(0, '#021A12'), (0.4, '#0F6A44'), (0.75, '#3CC08A'), (1, '#C8F8D8')], 'bands': 11, 'warp': 0.9})
item('oak-planks', 'Oak Planks', 'Warm vertical oak boards with visible grain, knots and nail marks.', 'patterns', False, False, None, 0.2, 'wood', **{'mode': 'planks', 'pal': [(0, '#4A2C18'), (0.5, '#8A5A32'), (1, '#C89A62')]})
item('tree-rings', 'Tree Rings', 'A cut log shows fine growth rings, cracks and dark bark.', 'patterns', False, False, None, 0.2, 'wood', **{'mode': 'rings', 'pal': [(0, '#6A3E1E'), (0.5, '#C08A4E'), (1, '#EBCB96')]})
item('walnut-burl', 'Walnut Burl', 'Glossy swirling walnut veneer with a soft light sheen.', 'patterns', True, False, None, 0.15, 'wood', **{'mode': 'burl', 'pal': [(0, '#1E0F08'), (0.5, '#5A2E16'), (1, '#B8723A')]})
item('pocket-green', 'Pocket Green', 'A four-shade green dithered sphere on a gridded floor.', 'retro', False, False, None, 0.12, 'dither', **{'mode': 'sphere', 'pal': ['#0F380F', '#306230', '#8BAC0F', '#9BBC0F']})
item('mono-sunrise', 'Mono Sunrise', 'A one-bit dithered sun rises over banded black hills.', 'retro', False, False, None, 0.2, 'dither', **{'mode': 'sunrise', 'pal': ['#0A0A0A', '#F4F0E4'], 'px': 5})
item('cga-stripes', 'CGA Stripes', 'Cyan and magenta diagonal bands around a dithered glowing orb.', 'retro', False, False, None, 0.2, 'dither', **{'mode': 'cga', 'pal': ['#000000', '#55FFFF', '#FF55FF', '#FFFFFF'], 'px': 7})
item('green-code-rain', 'Green Code Rain', 'Falling green glyph columns glow with bright leading heads.', 'retro', False, False, None, 0.12, 'coderain', **{'pal': [(0, '#003B12'), (0.6, '#1CFF6A'), (1, '#E8FFEE')]})
item('binary-rain', 'Binary Rain', 'Cyan zeros and ones fall in thin columns through the dark.', 'retro', False, False, None, 0.1, 'coderain', **{'pal': [(0, '#08304A'), (0.6, '#27C8FF'), (1, '#F0FFFF')], 'mode': 'binary', 'cw': 32, 'ch': 44, 'bgc': '#010612'})
item('purple-storm', 'Purple Storm', 'One huge violet bolt splits the night sky over a dark sea.', 'landscapes', True, False, None, 0.1, 'lightning', **{'mode': 'storm', 'sky': [(0, '#0A0A24'), (1, '#2A1A48')], 'c': ('#B49CFF',)})
item('cyan-forks', 'Cyan Forks', 'Forked cyan lightning flashes behind a silhouetted city.', 'gaming', False, False, None, 0.1, 'lightning', **{'mode': 'forks', 'sky': [(0, '#04101C'), (1, '#0A2A3A')], 'c': ('#7FE8FF',)})
item('plasma-globe', 'Plasma Globe', 'Purple filaments arc from a glowing core inside a glass sphere.', 'gaming', True, False, None, 0.1, 'lightning', **{'mode': 'plasma', 'sky': [(0, '#05050E'), (1, '#05050E')], 'c': ('#C06BFF', '#FF3AD0')})
item('cloud-sunset', 'Cloud Sunset', 'Dark clouds with glowing edges drift across a pink and gold sky.', 'landscapes', True, False, None, 0.12, 'clouds', **{'mode': 'sunset', 'sky': [(0, '#1F2A6A'), (0.5, '#C2506E'), (0.64, '#FFB070'), (1, '#FFD8A0')], 'cloud': [(0, '#3A2858'), (0.5, '#E8788A'), (1, '#FFE0B8')]})
item('moonlit-clouds', 'Moonlit Clouds', 'Thin silver clouds slide under a bright moon and stars.', 'landscapes', False, False, None, 0.12, 'clouds', **{'mode': 'moon', 'sky': [(0, '#030614'), (1, '#0E1C48')], 'cloud': [(0, '#0E1A3A'), (0.6, '#6A86C0'), (1, '#E8F0FF')]})
item('storm-front', 'Storm Front', 'A heavy grey-green storm bank with pale shafts of light.', 'landscapes', False, False, None, 0.12, 'clouds', **{'mode': 'storm', 'sky': [(0, '#0A1210'), (1, '#2A3A36')], 'cloud': [(0, '#0A1412'), (0.6, '#4A5A56'), (1, '#C8D8D0')]})
item('pool-caustics', 'Pool Caustics', 'Sunlit ripple patterns dance over blue pool tiles.', 'landscapes', False, False, None, 0.2, 'caustics', **{'mode': 'pool', 'pal': [(0, '#0A5A78'), (1, '#27C0D8')]})
item('shallow-water', 'Shallow Water', 'Turquoise shallows with a web of bright caustic light.', 'landscapes', False, False, None, 0.2, 'caustics', **{'mode': 'shallow', 'pal': [(0, '#0E8A9A'), (1, '#7AE0C8')]})
item('deep-blue', 'Deep Blue', 'Sun rays and rising bubbles fade into dark ocean depths.', 'landscapes', True, False, None, 0.1, 'caustics', **{'mode': 'deep', 'pal': [(0, '#04476B'), (0.5, '#02203A'), (1, '#010A18')]})
item('led-equalizer', 'LED Equalizer', 'A segmented spectrum meter glows from cyan to red with a reflection.', 'retro', False, False, None, 0.1, 'equalizer', **{'mode': 'led', 'bg': [(0, '#05030F'), (1, '#14082A')], 'pal': [(0, '#27E6FF'), (0.5, '#C6FF2D'), (0.8, '#FFB020'), (1, '#FF2D55')]})
item('voice-wave', 'Voice Wave', 'Rounded pink and violet bars form a calm audio waveform.', 'abstract', False, False, None, 0.1, 'equalizer', **{'mode': 'wave', 'bg': [(0, '#0B0F24'), (1, '#1A0F34')], 'pal': [(0, '#FF4FB0'), (0.5, '#9A5CFF'), (1, '#27E6FF')]})
item('radial-spectrum', 'Radial Spectrum', 'Glowing sound bars burst from a ring in cyan and magenta.', 'abstract', True, False, None, 0.1, 'equalizer', **{'mode': 'radial', 'bg': [(0, '#02060F'), (1, '#0A0F26')], 'pal': [(0, '#27E6FF'), (0.5, '#7A5CFF'), (1, '#FF4FD8')]})
item('tartan-plaid', 'Tartan Plaid', 'Woven blue and green tartan with red and cream threads.', 'patterns', True, False, None, 0.2, 'plaid', **{'mode': 'tartan', 'a': '#1F4A8A', 'b': '#0E3A2A', 'c': '#C4262E', 'd_': '#F2E8C8', 'scale': 1.2})
item('argyle', 'Argyle', 'Navy and cream diamonds with fine red stitched lines.', 'patterns', False, False, None, 0.25, 'plaid', **{'mode': 'argyle', 'a': '#2A4A7A', 'b': '#E8DDC4', 'c': '#B8323A', 'd_': '#F6F0E0', 'scale': 1.0})
item('houndstooth', 'Houndstooth', 'A bold black and ivory houndstooth weave.', 'patterns', False, False, None, 0.25, 'plaid', **{'mode': 'hound', 'a': '#15161C', 'b': '#EFE9DD', 'scale': 1.0})


#### ---- WAVE2 BEGIN ----
# ====================================================================== WAVE 2: kit
def C(c):
    return col(c) if isinstance(c, str) else np.asarray(c, np.float32)


def W2(id, name, desc, cat, pro, feat, dim, fn, **params):
    item(id, name, desc, cat, pro, feat, None, dim, fn, **params)
    E[-1]["w2"] = True


def reg(name, f):
    GENS[name] = lambda s, p: f(s, **p)


def poly_mask(polys, ss=2):
    def f(d, s):
        for p in polys:
            d.polygon([(x * s, y * s) for x, y in p], fill=255)
    return draw_mask(f, ss)


def line_mask(paths, width, ss=2, closed=False):
    def f(d, s):
        for p in paths:
            pts = [(x * s, y * s) for x, y in p] + ([(p[0][0] * s, p[0][1] * s)] if closed else [])
            d.line(pts, fill=255, width=max(1, int(width * s)), joint="curve")
    return draw_mask(f, ss)


def circ(img, cx, cy, r, color, k=1.0, soft=1.0):
    x0, x1 = int(max(0, cx - r - 4)), int(min(W, cx + r + 4))
    y0, y1 = int(max(0, cy - r - 4)), int(min(H, cy + r + 4))
    if x1 <= x0 or y1 <= y0:
        return
    d = np.sqrt((XX[y0:y1, x0:x1] - cx) ** 2 + (YY[y0:y1, x0:x1] - cy) ** 2)
    a = ((1 - smooth(r - soft, r + soft, d)) * k)[..., None]
    sub = img[y0:y1, x0:x1]
    sub *= 1 - a
    sub += a * C(color)


def ring(img, cx, cy, r, w, color, k=1.0):
    x0, x1 = int(max(0, cx - r - w - 4)), int(min(W, cx + r + w + 4))
    y0, y1 = int(max(0, cy - r - w - 4)), int(min(H, cy + r + w + 4))
    if x1 <= x0 or y1 <= y0:
        return
    d = np.sqrt((XX[y0:y1, x0:x1] - cx) ** 2 + (YY[y0:y1, x0:x1] - cy) ** 2)
    a = ((1 - smooth(w * 0.5 - 0.8, w * 0.5 + 0.8, np.abs(d - r))) * k)[..., None]
    sub = img[y0:y1, x0:x1]
    sub *= 1 - a
    sub += a * C(color)


def over(img, mask, color, k=1.0):
    m = (mask * k)[..., None]
    img *= 1 - m
    img += m * C(color)


def shadow_under(img, mask, dx=0, dy=16, s=22, k=0.18, color=(0, 0, 0)):
    sh = np.roll(np.roll(blur(mask, s), dy, 0), dx, 1)
    img *= 1 - (sh * k)[..., None] * (1 - np.asarray(color, np.float32))


def height_light(h, light=(-0.6, -0.7, 0.5), k=60.0):
    gy, gx = np.gradient(h)
    n = np.stack([-gx * k, -gy * k, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L = np.array(light, np.float32)
    L /= np.linalg.norm(L)
    return np.clip(n @ L, 0, 1)


def hills_layer(img, seed, y, amp, top, bot, freq=1.4, oct=4, sharp=0.0, fade=500, lit=None, shade=None, lit_k=0.5, skew=0.0):
    prof = profile(seed, y, amp, freq, oct, 0.5, sharp, W // 2, skew)
    return ridge(img, prof, top, bot, fade_y=fade, lit=lit, shade=shade, lit_k=lit_k)


def cellular(scale, seed, ox=0.0, oy=0.0):
    """Jittered-grid cell noise: returns f1, f2 (distances in cell units) and a per-cell random id."""
    r = np.random.default_rng(seed)
    gx = int(W / scale) + 6
    gy = int(H / scale) + 6
    jx = r.random((gy, gx)).astype(np.float32)
    jy = r.random((gy, gx)).astype(np.float32)
    ids = r.random((gy, gx)).astype(np.float32)
    px = (XX + ox) / scale + 1
    py = (YY + oy) / scale + 1
    ix, iy = np.floor(px).astype(np.int32), np.floor(py).astype(np.int32)
    fx, fy = px - ix, py - iy
    f1 = np.full((H, W), 9.0, np.float32)
    f2 = np.full((H, W), 9.0, np.float32)
    cid = np.zeros((H, W), np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy_ = np.clip(iy + dy, 0, gy - 1)
            xx_ = np.clip(ix + dx, 0, gx - 1)
            d = np.sqrt((dx + jx[yy_, xx_] - fx) ** 2 + (dy + jy[yy_, xx_] - fy) ** 2)
            better = d < f1
            f2 = np.where(better, f1, np.minimum(f2, d))
            cid = np.where(better, ids[yy_, xx_], cid)
            f1 = np.minimum(f1, d)
    return f1, f2, cid


def pastel(i):
    return ["#BFD7FF", "#FFD6E0", "#D9F2C9", "#FFE8B8", "#D8C8F5", "#C4F0EC", "#FFCFB8", "#F3E3FF"][i % 8]


# ====================================================================== WAVE 2: AMOLED / pastel / grey
def g_amoled_geo(seed, mode, accent="#5CE1FF"):
    img = np.zeros((H, W, 3), np.float32)
    ac = C(accent)
    if mode == "orbit":
        cx, cy = 720, 1230
        ang = math.radians(-24)
        for i, (rx, a) in enumerate(((230, 0.34), (410, 0.22), (610, 0.14), (830, 0.08))):
            t = np.linspace(0, 2 * math.pi, 400)
            ry = rx * 0.38
            x = rx * np.cos(t)
            y = ry * np.sin(t)
            xr = cx + x * math.cos(ang) - y * math.sin(ang)
            yr = cy + x * math.sin(ang) + y * math.cos(ang)
            m = line_mask([list(zip(xr, yr))], 2.2, 3, True)
            over(img, m, "#FFFFFF" if i != 1 else accent, a * (1.6 if i == 1 else 1))
        for rx, th, rad, c_, g in ((230, 1.2, 8, "#FFFFFF", 0.0), (410, 4.0, 13, accent, 0.5), (610, 2.3, 6, "#FFFFFF", 0.0), (830, 5.4, 9, "#FFFFFF", 0.0)):
            ry = rx * 0.38
            x = rx * math.cos(th)
            y = ry * math.sin(th)
            px_, py_ = cx + x * math.cos(ang) - y * math.sin(ang), cy + x * math.sin(ang) + y * math.cos(ang)
            if g:
                glow(img, px_, py_, 40, C(c_), g, "exp")
            circ(img, px_, py_, rad, c_)
        glow(img, cx, cy, 70, ac, 0.5, "exp")
        circ(img, cx, cy, 16, "#FFFFFF")
    elif mode == "corner":
        for i in range(11):
            R = 260 + i * 150
            a = 0.55 * (1 - i / 12) ** 1.4
            t = np.linspace(0, math.pi / 2, 200)
            m = line_mask([list(zip(R * np.cos(t), R * np.sin(t)))], 2.0, 3)
            over(img, m, accent if i == 4 else "#FFFFFF", a * (2 if i == 4 else 1))
        ring(img, 0, 0, 120, 3, accent, 0.9)
        glow(img, W - 150, H - 190, 40, ac, 0.55, "exp")
        circ(img, W - 150, H - 190, 7, "#FFFFFF")
        ring(img, W - 150, H - 190, 34, 2, "#FFFFFF", 0.35)
    else:  # lines
        for i, x in enumerate((150, 410, 720, 1030, 1290)):
            m = line_mask([[(x, 0), (x, H)]], 1.6, 2)
            over(img, m, "#FFFFFF", 0.11 if i != 2 else 0.0)
        fade = np.exp(-(((YY - 1290) / 520) ** 2))
        m = line_mask([[(720, 0), (720, H)]], 3.0, 2)
        over(img, m * fade, accent, 0.95)
        glow(img, 720, 1290, 36, ac, 0.9, "exp")
        circ(img, 720, 1290, 11, "#FFFFFF")
        for y in (430, 640, 2100, 2290):
            over(img, line_mask([[(150, y), (410, y)]], 1.6, 2), "#FFFFFF", 0.10)
        for y in (900, 1700):
            over(img, line_mask([[(1030, y), (1290, y)]], 1.6, 2), accent, 0.5)
    return img, 0.0, 0.0


def g_amoled_glow(seed, mode, accent="#FF7A2D", accent2="#FFD27A"):
    r = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.float32)
    ac = C(accent)
    if mode == "ember":
        x0, y0 = 720, 1950
        glow(img, x0, y0, 260, ac, 0.16, "exp")
        glow(img, x0, y0, 70, ac, 0.55, "exp")
        glow(img, x0, y0, 24, C(accent2), 0.8, "exp")
        circ(img, x0, y0, 10, accent2)
        for i in range(26):
            h = r.random() ** 1.3
            sx = x0 + r.standard_normal() * (30 + 140 * h)
            sy = y0 - 40 - h * 1050
            rad = 2.4 * (1 - 0.7 * h)
            circ(img, sx, sy, rad, accent2 if r.random() < 0.5 else accent, 0.9 * (1 - h) ** 0.8 + 0.1)
    elif mode == "edge":
        def f(d, ss):
            d.rounded_rectangle([5 * ss, 5 * ss, (W - 5) * ss, (H - 5) * ss], radius=96 * ss, outline=255, width=3 * ss)
        m = draw_mask(f, 2)
        t = (np.arctan2(YY - H / 2, XX - W / 2) / (2 * math.pi) + 0.5)
        colr = ramp(0.5 + 0.5 * np.sin(t * 12.566), [(0, accent), (0.5, accent2), (1, "#7A5CFF")])
        lay = (m + blur(m, 9) * 1.4 + blur(m, 34) * 1.6 + blur(m, 90) * 1.2)[..., None]
        img += colr * lay * 0.9
    else:  # sonar rings
        cx, cy = 720, 1300
        for i in range(9):
            R = 110 + i * 105 + i * i * 6
            ring(img, cx, cy, R, 2.2, accent, 0.65 * (1 - i / 9.5) ** 1.5)
        glow(img, cx, cy, 80, ac, 0.55, "exp")
        circ(img, cx, cy, 12, "#FFFFFF")
    return img, 0.0, 0.0


def g_amoled_sky(seed, mode, accent="#BFD7FF"):
    img = np.zeros((H, W, 3), np.float32)
    ac = C(accent)
    if mode == "stars":
        stars(img, seed, 170, amp=(0.45, 1.0), size=1.1, big=0)
        bx, by = 930, 760
        glow(img, bx, by, 30, ac, 0.6, "exp")
        circ(img, bx, by, 5, "#FFFFFF")
        for a_ in (0, 1):
            m = line_mask([[(bx - 90, by), (bx + 90, by)]] if a_ == 0 else [[(bx, by - 90), (bx, by + 90)]], 1.6, 2)
            fall = np.exp(-np.hypot(XX - bx, YY - by) / 40)
            over(img, m * fall, accent, 0.9)
    elif mode == "crescent":
        stars(img, seed + 4, 90, amp=(0.3, 0.8), size=0.9)
        cx, cy, R = 820, 900, 190
        d1 = np.hypot(XX - cx, YY - cy)
        d2 = np.hypot(XX - cx - 70, YY - cy + 46)
        m = (1 - smooth(R - 1.2, R + 1.2, d1)) * smooth(R * 0.93 - 1.2, R * 0.93 + 1.2, d2)
        img += blur(m, 36)[..., None] * ac * 0.45
        over(img, m, "#F6F1E4")
    else:  # comet
        stars(img, seed + 8, 70, amp=(0.3, 0.8), size=0.9)
        hx, hy = 1010, 620
        u = np.array([-0.62, 0.78], np.float32)
        t = (XX - hx) * u[0] + (YY - hy) * u[1]
        dperp = -(XX - hx) * u[1] + (YY - hy) * u[0]
        tail = np.where(t > 0, np.exp(-t / 520) * np.exp(-(dperp / (3 + 0.05 * t)) ** 2), 0)
        ion = np.where(t > 0, np.exp(-t / 760) * np.exp(-(((dperp - 0.11 * t) / (6 + 0.08 * t)) ** 2)), 0)
        img += tail[..., None] * 0.95 + ion[..., None] * ac * 0.35
        glow(img, hx, hy, 40, ac, 0.8, "exp")
        circ(img, hx, hy, 9, "#FFFFFF")
    return img, 0.0, 0.0


reg("amoled_geo", g_amoled_geo)
reg("amoled_glow", g_amoled_glow)
reg("amoled_sky", g_amoled_sky)


def blob_poly(cx, cy, R, seed, n=120):
    r = np.random.default_rng(seed)
    ph = r.random(4) * 6.28
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rr = R * (1 + 0.10 * np.sin(2 * t + ph[0]) + 0.09 * np.sin(3 * t + ph[1]) + 0.05 * np.sin(5 * t + ph[2]))
    return list(zip(cx + rr * np.cos(t), cy + rr * np.sin(t)))


def g_pastel_soft(seed, mode):
    if mode == "blobs":
        img = vgrad([(0, "#F6F0FA"), (1, "#EEF4FB")])
        spec = [(300, 520, 330), (1150, 760, 290), (700, 1180, 380), (230, 1700, 300), (1180, 1850, 340), (640, 2300, 260), (1250, 330, 160)]
        for i, (x, y, R) in enumerate(spec):
            m = poly_mask([blob_poly(x, y, R, seed + i)])
            shadow_under(img, m, 0, 22, 26, 0.10)
            over(img, m, pastel(i + 1))
            hi = poly_mask([blob_poly(x - R * 0.1, y - R * 0.12, R * 0.78, seed + i + 40)])
            over(img, hi, "#FFFFFF", 0.22)
        return img, 1.0, 0.0
    if mode == "grain":
        img = vgrad([(0, "#FBEFE6"), (1, "#EADDF7")])
        for (x, y, R, c_, k) in ((260, 520, 780, "#FFB6A8", 0.85), (1250, 900, 700, "#C6B3F7", 0.8), (500, 1700, 900, "#A8E4D8", 0.7), (1200, 2300, 620, "#FFE29A", 0.75), (900, 200, 500, "#FFD0E4", 0.6)):
            d = np.hypot(XX - x, YY - y) / R
            a = np.exp(-d * d * 1.8)[..., None] * k
            img = img * (1 - a) + C(c_) * a
        ring(img, 720, 1250, 400, 3, "#FFFFFF", 0.45)
        return img, 8.0, 0.0
    img = vgrad([(0, "#FFF3E2"), (1, "#FCE7DC")])
    cols_ = ["#F7B7A3", "#F9D29D", "#CFE3B4", "#A9D6D0", "#B9B6E8"]

    def arch(img, cx, base, w, k):
        for i in range(5):
            ww = w * (1 - i * 0.17)

            def f(d, ss, ww=ww):
                d.rounded_rectangle([(cx - ww / 2) * ss, (base - ww * 1.35) * ss, (cx + ww / 2) * ss, base * ss], radius=ww / 2 * ss, fill=255)
                d.rectangle([(cx - ww / 2) * ss, (base - ww * 0.6) * ss, (cx + ww / 2) * ss, base * ss], fill=255)
            over(img, draw_mask(f, 2), cols_[(i + k) % 5])
    arch(img, 720, 1560, 980, 0)
    arch(img, 300, 2300, 400, 2)
    arch(img, 1130, 2330, 320, 3)
    circ(img, 1190, 300, 90, "#F7B7A3")
    return img, 3.0, 0.0


def g_pastel_flow(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "waves":
        img = vgrad([(0, "#FDEFF5"), (1, "#E9F1FF")])
        xs = np.linspace(0, W, 400)
        for i in range(6):
            y = 520 + i * 330
            yy = y + 90 * np.sin(xs / W * (5 + i) + i * 1.3) + 40 * np.sin(xs / W * 11 + i)
            m = poly_mask([list(zip(xs, yy)) + [(W, H + 10), (0, H + 10)]])
            shadow_under(img, m, 0, -16, 24, 0.11)
            over(img, m, pastel(i + 2))
        return img, 1.2, 0.0
    if mode == "pills":
        img = vgrad([(0, "#FFF6EC"), (1, "#F3EEFF")])
        for i in range(17):
            x, y = r.random() * W, r.random() * H
            L = 220 + r.random() * 420
            wd = 70 + r.random() * 40
            a = math.radians(-35)
            p0 = (x - math.cos(a) * L / 2, y - math.sin(a) * L / 2)
            p1 = (x + math.cos(a) * L / 2, y + math.sin(a) * L / 2)

            def f(d, ss, p0=p0, p1=p1, wd=wd):
                d.line([(p0[0] * ss, p0[1] * ss), (p1[0] * ss, p1[1] * ss)], fill=255, width=int(wd * ss))
                for p in (p0, p1):
                    d.ellipse([(p[0] - wd / 2) * ss, (p[1] - wd / 2) * ss, (p[0] + wd / 2) * ss, (p[1] + wd / 2) * ss], fill=255)
            m = draw_mask(f, 2)
            shadow_under(img, m, 14, 18, 22, 0.10)
            over(img, m, pastel(i))
        return img, 1.2, 0.0
    img = vgrad([(0, "#EFEAFB"), (1, "#E3F3F1")])
    cx0, cy0 = 1100, 500
    for j in range(2560 // 200 + 1):
        for i in range(8):
            x, y = 90 + i * 180, 90 + j * 200
            d = math.hypot(x - cx0, y - cy0) / 1600
            R = 18 + 68 * (1 - min(1, d)) ** 1.2
            circ(img, x + (90 if j % 2 else 0), y, R, pastel(i + j * 3))
    return img, 1.2, 0.0


def g_grey_study(seed, mode):
    if mode == "stones":
        img = vgrad([(0, "#9A9A9A"), (0.6, "#6A6A6C"), (1, "#3C3C40")])
        glow(img, 720, 900, 700, (0.28, 0.28, 0.28), 0.5, "gauss")
        base = 2160
        spec = [(720, base - 160, 420, 190, 0.05), (700, base - 440, 330, 150, -0.08), (735, base - 680, 250, 118, 0.06), (715, base - 880, 170, 84, -0.05), (725, base - 1030, 105, 56, 0.1)]
        gsh = np.exp(-(((XX - 720) / 560) ** 2 + ((YY - (base + 20)) / 36) ** 2))
        img *= 1 - gsh[..., None] * 0.6
        for (cx, cy, rx, ry, rot) in spec:
            c_, s_ = math.cos(rot), math.sin(rot)
            xx = ((XX - cx) * c_ + (YY - cy) * s_) / rx
            yy = (-(XX - cx) * s_ + (YY - cy) * c_) / ry
            d2 = xx * xx + yy * yy
            z = np.sqrt(np.clip(1 - d2, 0, 1))
            lam = np.clip((-xx * 0.45 - yy * 0.78 + z * 0.65), 0, 1)
            tex = fbm(seed + int(cx), 12, 4)
            t = (0.12 + 0.82 * lam ** 1.1) * (0.93 + 0.14 * tex)
            a = np.clip((1 - np.sqrt(d2)) * min(rx, ry) + 0.5, 0, 1)
            colr = np.repeat(t[..., None], 3, -1) * np.array([1.0, 0.99, 0.97], np.float32)
            colr += (np.exp(-(((xx + 0.35) / 0.22) ** 2 + ((yy + 0.45) / 0.1) ** 2)) * 0.18)[..., None]
            img = img * (1 - a[..., None]) + colr * a[..., None]
        return img, 1.6, 0.25
    if mode == "sand":
        warp = fbm(seed + 1, 3, 4)
        big = fbm(seed + 2, 1.4, 3)
        u = (XX * math.cos(0.5) + YY * math.sin(0.5)) / 70.0 + warp * 5.0 + big * 3
        tri = np.abs((u % 1.0) - 0.5) * 2
        h = (tri ** 1.6) * (0.4 + 0.6 * big) + big * 2.5
        t = np.clip(height_light(h * 12, (-0.7, -0.4, 0.5), 3.0) * 1.15, 0, 1)
        return ramp(t, [(0, "#0C0C0E"), (0.5, "#6C6C70"), (1, "#EFEFEF")]), 1.2, 0.2
    nz = fbm(seed + 3, 1.2, 3, cy=3)
    nz2 = fbm(seed + 4, 3.0, 2, cy=4)
    u = XX / 230.0 + nz * 1.6 + nz2 * 0.25
    h = np.cos(u * 2) * (0.6 + 0.4 * nz) + np.cos(u * 4 + nz * 4) * 0.2
    l = height_light(h * 40, (-0.9, -0.2, 0.55), 1.0)
    ao = 0.5 + 0.5 * smooth(-0.9, 0.7, h)
    t = np.clip(l * ao * 1.2, 0, 1)
    return ramp(t, [(0, "#060607"), (0.55, "#58585C"), (1, "#E6E6E8")]), 1.4, 0.2


reg("amoled_geo", g_amoled_geo)
reg("amoled_glow", g_amoled_glow)
reg("amoled_sky", g_amoled_sky)
reg("pastel_soft", g_pastel_soft)
reg("pastel_flow", g_pastel_flow)
reg("grey_study", g_grey_study)

W2("amoled-orbit", "Orbit Lines", "Fine orbital rings on pure black, with one cyan planet. Great for OLED.", "amoled", False, True, 0.0, "amoled_geo", mode="orbit", accent="#5CE1FF")
W2("amoled-corner-arcs", "Corner Arcs", "Hairline arcs fade out from a corner, with one tiny violet light. Pure black.", "amoled", False, False, 0.0, "amoled_geo", mode="corner", accent="#B58CFF")
W2("amoled-lone-line", "Lone Line", "One glowing mint hairline with a single bead, between pale rules. Pure black.", "amoled", False, False, 0.0, "amoled_geo", mode="lines", accent="#6BFFC4")
W2("amoled-ember", "Black Ember", "A single orange ember and rising sparks in total darkness.", "amoled", False, False, 0.0, "amoled_glow", mode="ember")
W2("amoled-edge-light", "Edge Light", "A thin rainbow glow that wraps the screen edge. The rest stays black.", "amoled", True, True, 0.0, "amoled_glow", mode="edge", accent="#FF5FA8", accent2="#5CE1FF")
W2("amoled-sonar", "Black Sonar", "A tiny blue dot sends quiet rings out into the dark.", "amoled", False, False, 0.0, "amoled_glow", mode="rings", accent="#3D8BFF")
W2("amoled-sparse-stars", "Sparse Stars", "A handful of faint stars and one bright twinkle on true black.", "amoled", False, False, 0.0, "amoled_sky", mode="stars")
W2("amoled-crescent", "Thin Crescent", "A slim warm crescent moon over a few dim stars, on pure black.", "amoled", False, False, 0.0, "amoled_sky", mode="crescent", accent="#FFD9A0")
W2("amoled-comet", "Lone Comet", "A bright comet with a long white tail and a blue ion trail.", "amoled", True, False, 0.0, "amoled_sky", mode="comet", accent="#6FA8FF")
W2("pastel-blob-garden", "Blob Garden", "Soft rounded shapes in pastel blue, pink, mint and cream, like a calm Material layout.", "abstract", False, True, 0.25, "pastel_soft", mode="blobs")
W2("pastel-grain-dawn", "Grain Dawn", "Peach, lilac and mint light melt together under fine film grain.", "abstract", False, False, 0.25, "pastel_soft", mode="grain")
W2("pastel-arches", "Pastel Arches", "Nested boho arches in coral, honey, sage and lilac on warm cream.", "abstract", True, False, 0.25, "pastel_soft", mode="arches")
W2("pastel-soft-waves", "Soft Waves", "Six gently overlapping pastel waves with tender shadows.", "abstract", False, False, 0.25, "pastel_flow", mode="waves")
W2("pastel-pills", "Candy Pills", "Rounded pastel capsules scattered at an angle with soft shadows.", "abstract", False, False, 0.25, "pastel_flow", mode="pills")
W2("pastel-dot-fade", "Dot Fade", "A grid of big pastel dots that shrink away from one bright corner.", "abstract", False, False, 0.25, "pastel_flow", mode="dots")
W2("grey-cairn", "Stone Cairn", "Five smooth grey stones balanced in a quiet monochrome studio.", "abstract", True, False, 0.15, "grey_study", mode="stones")
W2("grey-sand-ripples", "Grey Ripples", "Wind ripples in black and white sand, shaped by low light.", "abstract", False, False, 0.15, "grey_study", mode="sand")
W2("grey-curtain", "Grey Curtain", "Heavy folds of a charcoal curtain lit from one side.", "abstract", False, False, 0.15, "grey_study", mode="folds")


# ====================================================================== WAVE 2: gaming 1
def mesh_render(sky, h, XSC, z0, z1, camh, f, hy, hscale, stops, light=(-0.5, 0.8, 0.35), fogc="#FFFFFF", fogk=0.6, lam_k=0.9, ss=2, flatten=None, edge=None):
    """Flat-shaded perspective heightfield. h is (nz, nx) in 0..1, row 0 = far."""
    nz, nx = h.shape
    sky = sky.copy()
    sky[int(hy):] = C(fogc)
    base = Image.fromarray(np.clip(sky * 255, 0, 255).astype(np.uint8), "RGB").resize((W * ss, H * ss), Image.BILINEAR)
    d = ImageDraw.Draw(base)
    zs = z0 * (z1 / z0) ** (np.arange(nz)[::-1] / (nz - 1))
    xs = (np.arange(nx) / (nx - 1) * 2 - 1)
    X = xs[None, :] * zs[:, None] * XSC
    Y = h * hscale
    Z = np.repeat(zs[:, None], nx, 1)
    sx = W / 2 + X / Z * f
    sy = hy + (camh - Y) / Z * f
    L = np.array(light, np.float32)
    L /= np.linalg.norm(L)
    fc = C(fogc)
    stops_ = stops
    for j in range(nz - 1):
        for i in range(nx - 1):
            for tri in (((j, i), (j, i + 1), (j + 1, i)), ((j, i + 1), (j + 1, i + 1), (j + 1, i))):
                P = np.array([[X[a, b], Y[a, b], Z[a, b]] for a, b in tri])
                n = np.cross(P[1] - P[0], P[2] - P[0])
                nn = np.linalg.norm(n)
                if nn < 1e-9:
                    continue
                n = n / nn
                if n[1] < 0:
                    n = -n
                lam = max(0.0, float(n @ L))
                t = float(np.mean([h[a, b] for a, b in tri]))
                c = ramp(np.array([t], np.float32), stops_)[0]
                c = c * (1 - lam_k + lam_k * (0.25 + 0.95 * lam))
                zf = float(np.mean(P[:, 2]))
                fg = fogk * ((zf - z0) / (z1 - z0)) ** 0.8
                c = c * (1 - fg) + fc * fg
                col8 = tuple(int(np.clip(v, 0, 1) * 255) for v in c)
                pts = [(float(sx[a, b]) * ss, float(sy[a, b]) * ss) for a, b in tri]
                d.polygon(pts, fill=col8, outline=col8)
    return np.asarray(base.resize((W, H), Image.BOX), np.float32) / 255.0


def g_lowpoly_scene(seed, mode):
    if mode == "isle":
        sky = vgrad([(0, "#2A1A5E"), (0.2, "#B04A8E"), (0.33, "#FF9A7A"), (0.39, "#FFC3A0"), (1, "#FFC3A0")])
        circ(sky, 720, 760, 230, "#FFF2C8")
        glow(sky, 720, 760, 300, (1.0, 0.7, 0.5), 0.35, "exp")
        nz, nx = 56, 40
        yy, xx = np.mgrid[0:nz, 0:nx].astype(np.float32)
        r_ = np.hypot((xx - nx / 2) / (nx * 0.34), (yy - nz * 0.55) / (nz * 0.32))
        n = fbm(seed, 4, 4, h=nz, w=nx)
        hh = np.clip(1 - r_ ** 1.6, 0, 1) * (0.55 + 0.9 * n) ** 1.2
        hh = np.where(hh > 0.12, hh, 0.0)
        hh = hh / hh.max()
        hh = np.where(hh > 0, 0.22 + hh * 0.78, 0.04)
        img = mesh_render(sky, hh, 0.95, 1.8, 11.0, 2.0, 1500, 1000, 1.1,
                          [(0, "#103E5E"), (0.1, "#1D6C8A"), (0.2, "#F2D79A"), (0.3, "#6FB06A"), (0.6, "#2E7448"), (0.85, "#8A7A66"), (1, "#F4F1EA")],
                          light=(-0.6, 0.7, 0.3), fogc="#FFC3A0", fogk=0.7)
        return img, 1.2, 0.1
    if mode == "peaks":
        sky = vgrad([(0, "#05071E"), (0.22, "#1B2A6B"), (0.37, "#7C80C0"), (1, "#7C80C0")])
        stars(sky, seed, 420, 0, 800, size=0.7)
        circ(sky, 1020, 520, 120, "#F6F2E6")
        glow(sky, 1020, 520, 200, (0.7, 0.75, 1.0), 0.28, "exp")
        nz, nx = 60, 36
        n = fbm(seed + 3, 3.5, 5, 0.55, h=nz, w=nx)
        rid = 1 - np.abs(2 * fbm(seed + 5, 3.0, 4, h=nz, w=nx) - 1)
        yy = np.arange(nz)[:, None].astype(np.float32)
        env = 0.35 + 0.65 * np.exp(-(((yy - 22) / 20) ** 2))
        hh = np.clip((0.35 * n + 0.65 * rid ** 1.8) * env, 0, 1)
        hh = hh / hh.max() * (1 - 0.92 * smooth(0.55, 0.95, np.arange(nz)[:, None] / (nz - 1.0)))
        img = mesh_render(sky, hh, 1.0, 2.45, 14.0, 3.0, 1450, 950, 2.2,
                          [(0, "#14183A"), (0.35, "#2C3470"), (0.65, "#59629E"), (0.85, "#B6BFE6"), (1, "#FFFFFF")],
                          light=(0.7, 0.7, 0.25), fogc="#7C80C0", fogk=0.75)
        return img, 1.2, 0.15
    sky = vgrad([(0, "#3A2A66"), (0.2, "#D4577A"), (0.39, "#FFB88A"), (1, "#FFB88A")])
    circ(sky, 400, 800, 190, "#FFF0C0")
    glow(sky, 400, 800, 320, (1.0, 0.6, 0.35), 0.4, "exp")
    nz, nx = 60, 34
    n = fbm(seed + 2, 2.6, 4, h=nz, w=nx)
    hh = np.floor(n * 5.0) / 5.0
    yy = np.arange(nz)[:, None].astype(np.float32)
    hh = hh * (0.4 + 0.6 * np.clip((yy - 4) / 24, 0, 1)) + 0.04
    hh = hh / hh.max() * (1 - 0.92 * smooth(0.55, 0.95, np.arange(nz)[:, None] / (nz - 1.0)))
    img = mesh_render(sky, hh, 1.0, 2.1, 13.0, 2.6, 1400, 1000, 1.5,
                      [(0, "#5B1E3A"), (0.3, "#A8402E"), (0.6, "#E07A3C"), (1, "#F6C27A")],
                      light=(-0.8, 0.5, 0.3), fogc="#FFB88A", fogk=0.7)
    return img, 1.2, 0.1


def g_ribbons(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "aero":
        ang, bg = -0.35, [(0, "#0A5BC4"), (0.5, "#2E9BEA"), (1, "#BDEBFF")]
        rib = [(-700, 190, 1.6, 0, 230, "#0B3E94", "#7CD3FF", 0.95), (-380, 160, 1.9, 1.0, 190, "#0E6BC8", "#E8FAFF", 0.9), (-60, 210, 1.5, 2.0, 250, "#2A8CE0", "#FFFFFF", 0.85),
               (290, 170, 1.8, 3.1, 200, "#0A4FB0", "#9AE4FF", 0.9), (640, 200, 1.4, 4.0, 230, "#38B2F0", "#FFFFFF", 0.85), (980, 180, 1.7, 5.1, 200, "#0C5CC0", "#C5F0FF", 0.9)]
    elif mode == "sunburst":
        ang, bg = 1.2, [(0, "#2A0A22"), (0.5, "#9E2A2E"), (1, "#FF9A3A")]
        rib = [(-640, 170, 1.7, 0.5, 210, "#FFD04A", "#FF5E2D", 0.95), (-300, 150, 1.5, 1.7, 180, "#FF8A3C", "#FFE8A0", 0.9), (40, 200, 1.8, 2.6, 230, "#FF4F7A", "#FFC27A", 0.9),
               (380, 160, 1.6, 3.7, 190, "#FFB02E", "#FFF4C8", 0.9), (720, 190, 1.4, 4.8, 220, "#D4313A", "#FF9A5A", 0.92), (1050, 170, 1.9, 5.5, 190, "#FFD04A", "#FF7A3A", 0.9)]
    else:
        ang, bg = 0.25, [(0, "#021A18"), (0.5, "#075B4E"), (1, "#0E8F6C")]
        rib = [(-760, 150, 1.4, 0.3, 140, "#E7C46A", "#6A4B10", 0.95), (-480, 190, 1.7, 1.4, 190, "#0B3D36", "#3EE6B0", 0.9), (-150, 160, 1.5, 2.2, 150, "#F1D88A", "#8A6414", 0.95),
               (150, 200, 1.8, 3.4, 200, "#0A4A40", "#7FFFD0", 0.9), (470, 150, 1.4, 4.4, 140, "#E7C46A", "#6A4B10", 0.95), (800, 180, 1.6, 5.2, 180, "#0B3D36", "#3EE6B0", 0.9)]
    c_, s_ = math.cos(ang), math.sin(ang)
    u = (XX - W / 2) * c_ + (YY - H / 2) * s_
    v = -(XX - W / 2) * s_ + (YY - H / 2) * c_
    t0 = np.clip(v / 1600 + 0.5, 0, 1)
    img = ramp(t0, bg)
    if mode == "aero":
        glow(img, 400, 300, 700, (0.8, 0.95, 1.0), 0.45, "exp")
    for (v0, amp, f, ph, w0, c1, c2, al) in rib:
        fr = f * 0.0016
        cen = v0 + amp * np.sin(u * fr + ph) + amp * 0.4 * np.sin(u * fr * 2.3 + ph * 1.7)
        cs = np.cos(u * fr * 0.8 + ph * 1.3)
        w = w0 * (0.18 + 0.82 * np.abs(cs))
        t = (v - cen) / np.maximum(w, 1.0)
        sg = np.where(cs >= 0, 1.0, -1.0)
        te = t * sg
        a = (1 - smooth(1 - 1.6 / np.maximum(w, 4), 1 + 0.5 / np.maximum(w, 4), np.abs(t)))
        sh = blur(a, 24)
        img *= 1 - (np.roll(sh, 26, 0) * 0.30)[..., None]
        base = ramp(0.5 + 0.5 * te, [(0, c1), (1, c2)])
        spec = np.exp(-(((te + 0.38) / 0.17) ** 2)) * 0.55 + np.exp(-(((te - 0.7) / 0.1) ** 2)) * 0.25
        base = base * (1 - 0.35 * smooth(0.65, 1.0, np.abs(te)))[..., None] + spec[..., None]
        m = (a * al)[..., None]
        img = img * (1 - m) + base * m
    if mode == "aero":
        bokeh(img, seed, 36, ["#FFFFFF", "#BDEBFF"], 14, 70, (0.12, 0.35), 0.8)
    elif mode == "sunburst":
        bokeh(img, seed, 28, ["#FFE9A0", "#FFB060"], 12, 60, (0.1, 0.3), 0.6)
    else:
        bokeh(img, seed, 26, ["#E7C46A", "#7FFFD0"], 10, 50, (0.1, 0.28), 0.6)
    return img, 1.2, 0.12


def bayer4():
    return np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) / 16.0


SPR_BUG = ["#.....#", ".#####.", "#.###.#", "#######", ".#####.", "..#.#.."]
SPR_SAUCER = ["..###..", ".#####.", "###.###", ".#####.", "#.#.#.#", "#.....#"]
SPR_SHIP = ["...#...", "..###..", "..###..", ".#####.", "#######", "##.#.##"]


def g_lcd_matrix(seed, mode, pal):
    r = np.random.default_rng(seed)
    LW, LH = 120, 213
    px = np.zeros((LH, LW), np.int32)
    yy, xx = np.mgrid[0:LH, 0:LW]
    if mode == "hills":
        bay = np.tile(bayer4(), (LH // 4 + 1, LW // 4 + 1))[:LH, :LW]
        g = (yy / LH * 1.25)
        px = np.clip(np.floor(g * 2.0 + bay * 1.0), 0, 2).astype(np.int32) * 0
        px = np.where(g * 2.6 + bay > 1.4, 1, 0)
        px = np.where(g * 2.6 + bay > 2.5, 2, px)
        d = np.hypot(xx - 84, yy - 40)
        px = np.where(d < 14, 0, px)
        px = np.where((d >= 14) & (d < 16), 1, px)
        for k, (base, amp, fr, sh) in enumerate(((120, 22, 0.05, 1), (140, 20, 0.07, 2), (162, 17, 0.09, 3))):
            hy = base + amp * np.sin(xx * fr + k * 2.0) + amp * 0.5 * np.sin(xx * fr * 2.7 + k)
            px = np.where(yy > hy, sh, px)
            px = np.where((yy > hy) & (yy < hy + 1), np.where(sh == 3, 2, 0), px)
        # clouds
        for cx_, cy_, rr in ((26, 66, 8), (50, 82, 6), (96, 96, 7)):
            for dx, dy, q in ((0, 0, 1), (-rr * 0.8, 2, 0.7), (rr * 0.9, 3, 0.7)):
                px = np.where(np.hypot(xx - cx_ - dx, (yy - cy_ - dy) * 1.5) < rr * q, 0, px)
        # grass tufts and ground
        px = np.where(yy > 178, 3, px)
        px = np.where((yy > 178) & ((xx + yy) % 6 == 0), 2, px)
        for x0 in range(4, LW, 11):
            px[176:179, x0] = 3
        # trees
        for x0 in (12, 34, 70, 104):
            h_ = 18
            for i in range(h_):
                wv = 1 + i // 3
                px[170 - h_ + i, max(0, x0 - wv):x0 + wv + 1] = 3
            px[170:179, x0] = 3
    elif mode == "maze":
        cw, ch = 29, 52
        grid = np.full((ch * 4 + 2, cw * 4 + 2), 3, np.int32)
        seen = np.zeros((ch, cw), bool)
        stack = [(0, 0)]
        seen[0, 0] = True
        par = {}
        while stack:
            x, y = stack[-1]
            nb = [(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= x + dx < cw and 0 <= y + dy < ch and not seen[y + dy, x + dx]]
            if not nb:
                stack.pop()
                continue
            nx_, ny_ = nb[int(r.integers(0, len(nb)))]
            seen[ny_, nx_] = True
            par[(nx_, ny_)] = (x, y)
            stack.append((nx_, ny_))
            x0, y0, x1, y1 = 4 * x + 2, 4 * y + 2, 4 * nx_ + 2, 4 * ny_ + 2
            grid[min(y0, y1):max(y0, y1) + 2, min(x0, x1):max(x0, x1) + 2] = 0
        for y in range(ch):
            for x in range(cw):
                grid[4 * y + 2:4 * y + 4, 4 * x + 2:4 * x + 4] = 0
        # solution path from far corner back to start
        cur = (cw - 1, ch - 1)
        while cur != (0, 0):
            p = par[cur]
            x0, y0, x1, y1 = 4 * p[0] + 2, 4 * p[1] + 2, 4 * cur[0] + 2, 4 * cur[1] + 2
            sub = grid[min(y0, y1):max(y0, y1) + 2, min(x0, x1):max(x0, x1) + 2]
            sub[sub == 0] = 1
            cur = p
        grid[2:4, 2:4] = 2
        grid[ch * 4 - 2:ch * 4, cw * 4 - 2:cw * 4] = 2
        ox, oy = (LW - grid.shape[1]) // 2, (LH - grid.shape[0]) // 2
        px[:] = 3
        px[oy:oy + grid.shape[0], ox:ox + grid.shape[1]] = grid
    else:  # shooter
        px[:] = 0
        for _ in range(90):
            px[int(r.integers(0, LH)), int(r.integers(0, LW))] = 1
        for _ in range(26):
            px[int(r.integers(0, LH)), int(r.integers(0, LW))] = 2

        def blit(spr, x0, y0, v=3, sc=1):
            for j, row in enumerate(spr):
                for i, ch_ in enumerate(row):
                    if ch_ == "#":
                        px[y0 + j * sc:y0 + (j + 1) * sc, x0 + i * sc:x0 + (i + 1) * sc] = v
        for rr_, sp in enumerate((SPR_SAUCER, SPR_BUG, SPR_BUG, SPR_SAUCER)):
            for cc in range(5):
                blit(sp, 8 + cc * 22 + (rr_ % 2) * 6, 16 + rr_ * 17, 3 if rr_ % 2 == 0 else 2)
        for x_, y0_, y1_ in ((52, 118, 140), (74, 90, 108)):
            px[y0_:y1_, x_:x_ + 1] = 3
        px[170:176, 20:34] = 3
        px[168:170, 22:32] = 2
        for k in range(18):
            ang = k / 18 * 2 * math.pi
            px[int(100 + 15 * math.sin(ang)), int(86 + 15 * math.cos(ang))] = 3
            px[int(100 + 9 * math.sin(ang + 0.2)), int(86 + 9 * math.cos(ang + 0.2))] = 2
        blit(SPR_SHIP, 56, 186, 3, 2)
        px[160:184, 60:61] = 2
    cell = 12
    big = np.kron(px, np.ones((cell, cell), np.int32))
    pad = H - big.shape[0]
    big = np.pad(big, ((pad // 2, pad - pad // 2), (0, W - big.shape[1])), mode="edge")
    palc = np.array([C(p) for p in pal], np.float32)
    colr = palc[big]
    gy, gx = np.mgrid[0:H, 0:W]
    lit = ((gy % cell) >= 1) & ((gy % cell) < cell - 1) & ((gx % cell) >= 1) & ((gx % cell) < cell - 1)
    litf = lit.astype(np.float32)
    sh = np.roll(np.roll(blur(litf, 1.2), 2, 0), 2, 1)
    base = palc[0] * 0.85 + 0.0
    img = np.where(lit[..., None], colr, base * (0.78 + 0.22 * (1 - sh[..., None])) * 0.9 + colr * 0.10)
    # ghosting: faint smear of lit pixels
    ghost = blur(np.roll(colr * litf[..., None], 14, 0), 3.0)
    img = img + ghost * 0.06
    img *= (0.96 + 0.04 * np.sin(gy / 3.0))[..., None]
    return img, 1.0, 0.12


def g_arcade_carpet(seed, mode):
    r = np.random.default_rng(seed)
    ss = 2
    if mode == "bowling":
        bgc, pal = (0.07, 0.05, 0.31), ["#FF2FA8", "#18D6E8", "#FFD02B", "#FFFFFF", "#8A4DFF"]
    elif mode == "casino":
        bgc, pal = (0.35, 0.05, 0.11), ["#E8B94A", "#F6E6B8", "#1C9AA0", "#B3203A"]
    else:
        bgc, pal = (0.04, 0.22, 0.24), ["#FF8A2D", "#FF5E8E", "#F6E8C8", "#7A5CFF"]
    im = rgb_canvas(ss, bgc)
    d = ImageDraw.Draw(im)
    S = ss
    cell = 190
    for gy in range(-1, int(H / cell) + 2):
        for gx in range(-1, int(W / cell) + 2):
            cx = (gx + 0.5 + (0.5 if gy % 2 else 0)) * cell + r.uniform(-18, 18)
            cy = (gy + 0.5) * cell + r.uniform(-18, 18)
            c = pal[int(r.integers(0, len(pal)))]
            c2 = pal[int(r.integers(0, len(pal)))]
            rot = r.uniform(0, 2 * math.pi)
            ca, sa = math.cos(rot), math.sin(rot)

            def P(x, y):
                return ((cx + x * ca - y * sa) * S, (cy + x * sa + y * ca) * S)
            col8 = rgb8(c)
            col2 = rgb8(c2)
            dk = rgb8(bgc)

            def EL(x, y, rr, fill=None, outline=None, width=0):
                qx, qy = P(x, y)
                d.ellipse([qx - rr * S, qy - rr * S, qx + rr * S, qy + rr * S], fill=fill, outline=outline, width=width)
            if mode == "bowling":
                k = int(r.integers(0, 5))
                if k == 0:
                    d.line([P(x, 28 * math.sin(x / 22.0)) for x in np.linspace(-70, 70, 40)], fill=col8, width=14 * S, joint="curve")
                elif k == 1:
                    d.polygon([P(0, -44), P(40, 30), P(-40, 30)], fill=col8)
                elif k == 2:
                    EL(0, 0, 34, outline=col8, width=11 * S)
                elif k == 3:
                    for q in range(-1, 2):
                        EL(q * 40, 0, 9, fill=col8)
                else:
                    d.polygon([P(-40, -4), P(0, -50), P(40, -4), P(0, 4)], fill=col8)
                    d.polygon([P(-40, 10), P(0, -36), P(40, 10), P(0, 18)], fill=col2)
            elif mode == "casino":
                k = int(r.integers(0, 3))
                if k == 0:
                    for s_, cc in ((52, col8), (36, dk), (22, col2), (9, dk)):
                        d.polygon([P(0, -s_), P(s_, 0), P(0, s_), P(-s_, 0)], fill=cc)
                elif k == 1:
                    for q in range(8):
                        a_ = q * math.pi / 4
                        EL(34 * math.cos(a_), 34 * math.sin(a_), 14, fill=col8)
                    EL(0, 0, 20, fill=col2)
                else:
                    for q in (-30, 0, 30):
                        d.polygon([P(q - 8, -50), P(q + 8, -50), P(q + 8, 50), P(q - 8, 50)], fill=col8 if q else col2)
            else:
                k = int(r.integers(0, 3))
                if k == 0:
                    pts = []
                    for t in np.linspace(0, 1, 40):
                        a_ = t * math.pi * 1.5
                        pts.append(P(50 * math.cos(a_) * (1 - 0.5 * t), 50 * math.sin(a_) * (1 - 0.5 * t)))
                    d.line(pts, fill=col8, width=16 * S, joint="curve")
                elif k == 1:
                    d.polygon([P((52 if q % 2 == 0 else 22) * math.cos(q * math.pi / 8), (52 if q % 2 == 0 else 22) * math.sin(q * math.pi / 8)) for q in range(16)], fill=col8)
                    EL(0, 0, 10, fill=col2)
                else:
                    pts = [P(48 * math.cos(a_), 48 * math.sin(a_)) for a_ in np.linspace(0, math.pi, 24)]
                    d.polygon(pts, fill=col8)
                    d.polygon([P(26 * math.cos(a_), 26 * math.sin(a_)) for a_ in np.linspace(0, math.pi, 24)], fill=col2)
    img = from_pil(im)
    n = fbm(seed, 120, 2, h=H, w=W)
    img *= (0.88 + 0.2 * n)[..., None]
    img *= 0.9
    return img, 3.0, 0.15


def g_crt_bloom(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "aperture":
        pic = vgrad([(0, "#1A0A3A"), (0.5, "#C2308E"), (1, "#1AC8E8")])
        for (x, y, R, c_) in ((720, 1100, 520, "#FFF2D8"), (350, 1700, 300, "#FFE066"), (1150, 600, 260, "#7AF2FF")):
            d = np.hypot(XX - x, YY - y) / R
            a = np.exp(-d * d * 2.4)[..., None]
            pic = pic * (1 - a) + C(c_) * a
        mk = np.zeros((H, W, 3), np.float32)
        xm = (XX % 6).astype(np.int32)
        for ch_ in range(3):
            mk[..., ch_] = np.where(xm // 2 == ch_, 1.0, 0.04)
        mk[..., :] *= (0.55 + 0.45 * smooth(0, 1, 0.5 + 0.5 * np.cos(YY / 8.0 * math.pi * 2)))[..., None]
        out = pic * mk * 1.9
        out += blur(out, 14) * 0.9 + blur(out, 60) * 0.7
        return out, 0.8, 0.25
    if mode == "triad":
        pic = np.zeros((H, W, 3), np.float32)
        t = (YY / H * 1.4 + XX / W * 0.5)
        pic = ramp(np.mod(t * 1.1, 1.0), [(0, "#FF3B1F"), (0.35, "#FFB02E"), (0.6, "#FFF4C8"), (0.8, "#FF3B1F"), (1, "#7A0E1A")])
        pic *= smooth(-0.3, 0.9, np.sin(XX / 210.0) * np.sin(YY / 330.0 + 1.0) + 0.3)[..., None] * 0.8 + 0.25
        pitch = 16.0
        row = np.floor(YY / (pitch * 0.866))
        xo = XX + (row % 2) * pitch / 2
        mk = np.zeros((H, W, 3), np.float32)
        for ch_ in range(3):
            cxm = (np.floor((xo - ch_ * pitch / 3) / pitch) + 0.5) * pitch + ch_ * pitch / 3
            cym = (row + 0.5) * pitch * 0.866
            dd = np.hypot(xo - cxm, YY - cym)
            mk[..., ch_] = 1 - smooth(pitch * 0.17, pitch * 0.30, dd)
        out = pic * (mk * 0.93 + 0.03) * 2.1
        out += blur(out, 10) * 0.8 + blur(out, 50) * 0.5
        return out, 0.8, 0.25
    # persistence streak, monochrome phosphor
    img = np.zeros((H, W, 3), np.float32)
    pts = []
    for t in np.linspace(0, 1, 500):
        x = 220 + 1000 * t + 130 * math.sin(t * 11)
        y = 2100 - 1700 * t ** 0.9 + 90 * math.sin(t * 17 + 1)
        pts.append((x, y))
    m = line_mask([pts], 6, 2)
    tt = np.clip((YY - 400) / 1700, 0, 1)
    img += (m * (0.25 + 0.75 * (1 - tt)))[..., None] * C("#8CFFC0")
    img += blur(m, 7)[..., None] * C("#18C878") * 1.6 + blur(m, 30)[..., None] * C("#0A9060") * 1.5 + blur(m, 110)[..., None] * C("#066A48") * 1.2
    hx, hy = pts[-1]
    glow(img, hx, hy, 40, C("#D8FFE8"), 1.0, "exp")
    glow(img, hx, hy, 170, C("#18C878"), 0.6, "exp")
    scan = 0.62 + 0.38 * (0.5 + 0.5 * np.cos(YY / 5.0 * math.pi * 2))
    img *= scan[..., None]
    img += (fbm(seed, 8, 3, h=H, w=W) * 0.05)[..., None] * C("#18C878")
    return img, 1.4, 0.55


reg("lowpoly_scene", g_lowpoly_scene)
reg("ribbons", g_ribbons)
reg("lcd_matrix", g_lcd_matrix)
reg("arcade_carpet", g_arcade_carpet)
reg("crt_bloom", g_crt_bloom)

W2("lowpoly-sunset-isle", "Low-Poly Isle", "A faceted green island in a flat-shaded sea under a huge peach sun, like a 90s 3D demo.", "gaming", True, True, 0.1, "lowpoly_scene", mode="isle")
W2("lowpoly-night-range", "Low-Poly Night Range", "Angular snow-capped peaks under a pale moon, flat-shaded like an early 3D game.", "gaming", False, False, 0.1, "lowpoly_scene", mode="peaks")
W2("lowpoly-mesa-run", "Low-Poly Mesa Run", "Terraced red mesas in blocky polygons with a low orange sun.", "gaming", False, False, 0.1, "lowpoly_scene", mode="mesa")
W2("y2k-aero-ribbons", "Aero Ribbons", "Glossy blue silk ribbons swirl through a bright sky, with soft bubbles.", "retro", False, True, 0.1, "ribbons", mode="aero")
W2("y2k-sunburst-ribbons", "Sunburst Ribbons", "Twisting ribbons in amber, coral and crimson with a warm early-2000s glow.", "retro", False, False, 0.1, "ribbons", mode="sunburst")
W2("y2k-emerald-ribbons", "Emerald Ribbons", "Emerald and gold ribbons curl across a deep green desktop-style backdrop.", "retro", True, False, 0.1, "ribbons", mode="emerald")
W2("lcd-green-hills", "LCD Green Hills", "A four-shade green dot-matrix landscape with a moon, clouds and trees.", "retro", False, False, 0.15, "lcd_matrix", mode="hills", pal=["#0F380F", "#306230", "#8BAC0F", "#9BBC0F"])
W2("lcd-amber-maze", "LCD Amber Maze", "An amber dot-matrix maze with a glowing path from start to goal.", "retro", False, False, 0.15, "lcd_matrix", mode="maze", pal=["#2B1400", "#7A3D00", "#E08A00", "#FFC84A"])
# dropped in review: W2("lcd-grey-shooter", "LCD Space Shooter", "A grey-blue handheld screen filled with invading ships, bullets and a blast ring.", "retro", True, False, 0.15, "lcd_matrix", mode="shooter", pal=["#14202A", "#3C5666", "#8FB0BE", "#D5ECEF"])
W2("carpet-bowling", "Bowling Carpet", "Neon squiggles, triangles and rings scattered over deep indigo, like an alley carpet.", "patterns", False, False, 0.2, "arcade_carpet", mode="bowling")
W2("carpet-casino", "Casino Carpet", "Gold diamonds, rosettes and bars on rich burgundy, in a plush arcade lounge style.", "patterns", False, False, 0.2, "arcade_carpet", mode="casino")
W2("carpet-cinema", "Cinema Carpet", "Orange and pink starbursts, spirals and half moons on dark teal.", "patterns", False, False, 0.2, "arcade_carpet", mode="cinema")
# dropped in review: W2("crt-aperture-bloom", "Aperture Bloom", "A macro view of a CRT: glowing red, green and blue stripes bloom around a hot white core.", "retro", True, True, 0.1, "crt_bloom", mode="aperture")
# dropped in review: W2("crt-triad-glow", "Triad Glow", "Round phosphor dots in an orange and red glow, as seen under a magnifier.", "retro", False, False, 0.1, "crt_bloom", mode="triad")
W2("crt-green-streak", "Phosphor Streak", "A bright green trace with a long afterglow on a scanlined monochrome screen.", "retro", False, False, 0.1, "crt_bloom", mode="streak")


# ====================================================================== WAVE 2: gaming 2
def parchment(seed, tone=0):
    n = fbm(seed, 3, 5)
    img = ramp(n, [(0, "#BE9A62"), (0.5, "#DEC48E"), (1, "#F1E2B8")])
    st = smooth(0.62, 0.9, fbm(seed + 9, 2.5, 4))
    img *= (1 - 0.22 * st)[..., None]
    fine = fbm(seed + 3, 90, 2)
    img *= (0.94 + 0.12 * fine)[..., None]
    dx = np.minimum(XX, W - XX)
    dy = np.minimum(YY, H - YY)
    dd = np.minimum(dx, dy)
    burn = np.exp(-dd / 150.0) * (0.55 + 0.45 * fbm(seed + 4, 6, 4))
    img = img * (1 - 0.75 * burn)[..., None] + C("#4A2A10") * (0.35 * burn)[..., None]
    return img


def scatter(r, n, ok, mind, tries=4000, size=(W, H)):
    pts = []
    for _ in range(tries):
        x, y = r.random() * (size[0] - 100) + 50, r.random() * (size[1] - 100) + 50
        if not ok(x, y):
            continue
        if all((x - a) ** 2 + (y - b) ** 2 > mind * mind for a, b in pts):
            pts.append((x, y))
            if len(pts) >= n:
                break
    return pts


def g_rpg_map(seed, mode):
    r = np.random.default_rng(seed)
    img = parchment(seed)
    ink = "#47290F"
    if mode in ("continent", "archipelago"):
        n = fbm(seed + 1, 2.2 if mode == "continent" else 3.4, 6)
        fall = 1 - np.clip(np.hypot((XX - 720) / 700, (YY - 1250) / 1050), 0, 1.4)
        L = n * 0.75 + fall * (0.5 if mode == "continent" else 0.3)
        thr = 0.62 if mode == "continent" else 0.58
        land = (L > thr).astype(np.float32)
        lmask = blur(land, 1.4)
        if mode == "archipelago":
            img = img * (1 - 0.30 * (1 - lmask))[..., None] + C("#7FA8A6") * (0.30 * (1 - lmask))[..., None]
        else:
            img = img * (1 - 0.14 * lmask)[..., None] + C("#8E9A55") * (0.14 * lmask)[..., None]
        for k, a in ((0, 0.9), (1, 0.45), (2, 0.28), (3, 0.16)):
            line = np.exp(-(((L - (thr - 0.035 * k)) / (0.004 + 0.002 * k)) ** 2))
            over(img, line, ink, a)
        gm = 0.5 + 0.5 * fbm(seed + 7, 5, 4)
        mn = fbm(seed + 11, 6, 4)
        fn = fbm(seed + 13, 7, 4)
        okm = lambda x, y: land[int(y), int(x)] > 0 and mn[int(y), int(x)] > 0.62 and L[int(y), int(x)] > thr + 0.04
        okf = lambda x, y: land[int(y), int(x)] > 0 and fn[int(y), int(x)] > 0.60 and mn[int(y), int(x)] < 0.58 and L[int(y), int(x)] > thr + 0.03
        mts = scatter(r, 70, okm, 56)
        fts = scatter(r, 160, okf, 34)

        def f(d, ss):
            for (x, y) in mts:
                s = 24 + r.random() * 18
                d.line([((x - s) * ss, (y + s * 0.6) * ss), (x * ss, (y - s) * ss), ((x + s) * ss, (y + s * 0.6) * ss)], fill=255, width=int(3 * ss), joint="curve")
                for q in range(4):
                    xx_ = x + 2 + q * s * 0.2
                    d.line([(xx_ * ss, (y - s * 0.7 + q * s * 0.25) * ss), ((xx_ + s * 0.35) * ss, (y + s * 0.5) * ss)], fill=255, width=int(2 * ss))
            for (x, y) in fts:
                d.ellipse([(x - 11) * ss, (y - 15) * ss, (x + 11) * ss, (y + 7) * ss], outline=255, width=int(2.5 * ss))
                d.line([(x * ss, (y + 7) * ss), (x * ss, (y + 18) * ss)], fill=255, width=int(2.5 * ss))
        over(img, draw_mask(f, 2), ink, 0.85)
        towns = scatter(r, 7, lambda x, y: land[int(y), int(x)] > 0 and L[int(y), int(x)] > thr + 0.05, 260)
        road = [(t[0] + r.uniform(-12, 12), t[1] + r.uniform(-12, 12)) for t in towns]

        def f2(d, ss):
            for (x, y) in towns:
                d.rectangle([(x - 14) * ss, (y - 14) * ss, (x + 14) * ss, (y + 14) * ss], outline=255, width=int(4 * ss))
                d.rectangle([(x - 5) * ss, (y - 5) * ss, (x + 5) * ss, (y + 5) * ss], fill=255)
            for a, b in zip(towns[:-1], towns[1:]):
                n_ = int(math.hypot(b[0] - a[0], b[1] - a[1]) / 26)
                for i in range(0, n_, 2):
                    p0 = (a[0] + (b[0] - a[0]) * i / n_, a[1] + (b[1] - a[1]) * i / n_)
                    p1 = (a[0] + (b[0] - a[0]) * (i + 1) / n_, a[1] + (b[1] - a[1]) * (i + 1) / n_)
                    d.line([(p0[0] * ss, p0[1] * ss), (p1[0] * ss, p1[1] * ss)], fill=255, width=int(3 * ss))
        over(img, draw_mask(f2, 2), "#8E1F12", 0.8)
        # compass rose
        def f3(d, ss):
            cx, cy = 1160, 2160
            for q in range(8):
                a_ = q * math.pi / 4 - math.pi / 2
                Lq = 150 if q % 2 == 0 else 80
                pts = [(cx * ss, cy * ss), ((cx + Lq * math.cos(a_ - 0.18)) * ss * 0.55 + cx * ss * 0.45, (cy + Lq * math.sin(a_ - 0.18)) * ss * 0.55 + cy * ss * 0.45),
                       ((cx + Lq * math.cos(a_)) * ss, (cy + Lq * math.sin(a_)) * ss),
                       ((cx + Lq * math.cos(a_ + 0.18)) * ss * 0.55 + cx * ss * 0.45, (cy + Lq * math.sin(a_ + 0.18)) * ss * 0.55 + cy * ss * 0.45)]
                d.polygon(pts, fill=255 if q % 2 == 0 else 0, outline=255)
            d.ellipse([(cx - 100) * ss, (cy - 100) * ss, (cx + 100) * ss, (cy + 100) * ss], outline=255, width=int(3 * ss))
            d.ellipse([(cx - 168) * ss, (cy - 168) * ss, (cx + 168) * ss, (cy + 168) * ss], outline=255, width=int(2 * ss))
        over(img, draw_mask(f3, 2), ink, 0.85)
    else:
        img = img * np.array([0.95, 0.98, 1.05], np.float32)
        gr = (((XX % 64) < 1.6) | ((YY % 64) < 1.6)).astype(np.float32)
        over(img, gr, "#7A6A4A", 0.22)
        rooms = []
        for _ in range(900):
            w_, h_ = int(r.integers(3, 8)) * 64, int(r.integers(3, 9)) * 64
            x0, y0 = int(r.integers(1, (W - w_) // 64 - 1)) * 64, int(r.integers(1, (H - h_) // 64 - 1)) * 64
            if all(x0 > b[0] + b[2] + 128 or x0 + w_ < b[0] - 128 or y0 > b[1] + b[3] + 128 or y0 + h_ < b[1] - 128 for b in rooms):
                rooms.append((x0, y0, w_, h_))
            if len(rooms) >= 9:
                break
        rooms.sort(key=lambda b: (b[1], b[0]))
        floor = np.zeros((H, W), np.float32)
        wall = np.zeros((H, W), np.float32)

        def fr(d, ss):
            for (x, y, w_, h_) in rooms:
                d.rectangle([x * ss, y * ss, (x + w_) * ss, (y + h_) * ss], fill=255)
            for a, b in zip(rooms[:-1], rooms[1:]):
                ax, ay = a[0] + a[2] // 2, a[1] + a[3] // 2
                bx, by = b[0] + b[2] // 2, b[1] + b[3] // 2
                d.rectangle([(min(ax, bx) - 32) * ss, (ay - 32) * ss, (max(ax, bx) + 32) * ss, (ay + 32) * ss], fill=255)
                d.rectangle([(bx - 32) * ss, (min(ay, by) - 32) * ss, (bx + 32) * ss, (max(ay, by) + 32) * ss], fill=255)
        floor = draw_mask(fr, 2)
        edge = np.clip(blur(floor, 2.2) - floor * 0.0, 0, 1)
        outline = np.clip(np.abs(blur(floor, 3.0) - floor) * 3.0, 0, 1) * (1 - floor * 0.0)
        near = np.clip(blur(floor, 44) * 2.4, 0, 1) * (1 - floor)
        hatch = (((XX + YY) % 18) < 3.2).astype(np.float32) * near
        over(img, floor, "#EBDCB4", 0.65)
        over(img, hatch, ink, 0.55)
        over(img, outline, ink, 0.95)
        gr2 = (((XX % 64) < 1.4) | ((YY % 64) < 1.4)).astype(np.float32) * floor
        over(img, gr2, "#9A8860", 0.35)

        def fd(d, ss):
            for (x, y, w_, h_) in rooms[:7]:
                d.rectangle([(x + w_ // 2 - 20) * ss, (y - 8) * ss, (x + w_ // 2 + 20) * ss, (y + 8) * ss], fill=255)
            x, y, w_, h_ = rooms[2]
            for q in range(6):
                d.line([((x + 20) * ss, (y + 20 + q * 22) * ss), ((x + 140 - q * 8) * ss, (y + 20 + q * 22) * ss)], fill=255, width=int(3 * ss))
        over(img, draw_mask(fd, 2), ink, 0.85)
        x, y, w_, h_ = rooms[-1]
        circ(img, x + w_ / 2, y + h_ / 2, 54, "#8E1F12", 0.0)
        ring(img, x + w_ / 2, y + h_ / 2, 54, 6, "#8E1F12", 0.85)
        ring(img, x + w_ / 2, y + h_ / 2, 30, 4, "#8E1F12", 0.85)
    return img, 1.6, 0.0


def g_dungeon_tiles(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "flag":
        f1, f2, cid = cellular(150, seed)
        gap = smooth(0.03, 0.11, f2 - f1)
        base = ramp(cid, [(0, "#4C4640"), (0.5, "#6A625A"), (1, "#7C7468")])
        n = fbm(seed + 2, 40, 3)
        img = base * (0.8 + 0.4 * n)[..., None]
        bev = np.clip((f2 - f1) * 3.0, 0, 1)
        img *= (0.55 + 0.45 * bev)[..., None] * gap[..., None] + (1 - gap[..., None]) * 0.12
        moss = smooth(0.58, 0.82, fbm(seed + 5, 5, 5)) * (1 - smooth(0.0, 0.45, f2 - f1))
        img = img * (1 - 0.7 * moss)[..., None] + ramp(n, [(0, "#2A4A22"), (1, "#5C8A3A")]) * (0.7 * moss)[..., None]
        d = np.hypot(XX - 720, YY - 1700) / 1500
        img *= (np.clip(1.15 - d, 0, 1) ** 1.2)[..., None] * C("#FFE0B0") * 1.15 + 0.02
        return img, 2.0, 0.2
    if mode == "rune":
        T = 240
        tx, ty = (XX % T), (YY % T)
        gap = smooth(2, 8, np.minimum(np.minimum(tx, T - tx), np.minimum(ty, T - ty)))
        ix, iy = (XX // T).astype(np.int32), (YY // T).astype(np.int32)
        rid = ((ix * 73856093) ^ (iy * 19349663) ^ seed) % 1000 / 1000.0
        n = fbm(seed, 50, 3)
        img = ramp(rid * 0.5 + n * 0.5, [(0, "#1C2230"), (1, "#323C52")])
        bev = np.minimum(np.minimum(tx, T - tx), np.minimum(ty, T - ty))
        img *= (0.55 + 0.45 * smooth(0, 40, bev))[..., None] * gap[..., None] + 0.02
        glyph = np.zeros((H, W), np.float32)

        def f(d, ss):
            for jy in range(H // T + 1):
                for jx in range(W // T + 1):
                    if r.random() < 0.34:
                        cx, cy = (jx + 0.5) * T, (jy + 0.5) * T
                        pts = [(cx + a * 52, cy + b * 52) for a in (-1, 0, 1) for b in (-1, 0, 1)]
                        seq = r.choice(9, int(r.integers(4, 7)), replace=False)
                        path = [pts[i] for i in seq]
                        d.line([(x * ss, y * ss) for x, y in path], fill=255, width=int(7 * ss), joint="curve")
                        d.ellipse([(cx - 84) * ss, (cy - 84) * ss, (cx + 84) * ss, (cy + 84) * ss], outline=255, width=int(4 * ss))
        glyph = draw_mask(f, 2)
        neon(img, glyph, "#3DE0FF", core=0.55, glows=((3, 1.0), (10, 0.9), (30, 0.5)), k=1.0)
        return img, 1.6, 0.35
    f1, f2, cid = cellular(210, seed)
    e = f2 - f1
    gap = smooth(0.05, 0.16, e)
    base = ramp(cid * 0.6 + fbm(seed + 3, 30, 3) * 0.4, [(0, "#16141A"), (1, "#38343C")])
    bev = np.clip(e * 3.5, 0, 1)
    img = base * (0.5 + 0.6 * bev)[..., None]
    lava = (1 - gap)
    lv = ramp(np.clip(0.5 + 0.5 * fbm(seed + 8, 9, 3), 0, 1), [(0, "#C2210A"), (0.55, "#FF7A18"), (1, "#FFE070")])
    img = img * gap[..., None] + lv * lava[..., None] * 1.2
    img += blur(lava, 14)[..., None] * C("#FF5A10") * 0.9 + blur(lava, 60)[..., None] * C("#FF3A08") * 0.5
    return img, 1.6, 0.3


def g_cockpit(seed, mode):
    r = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.float32)
    # windshield area
    wpoly = [(40, 760), (260, 330), (1180, 330), (1400, 760), (1400, 1560), (40, 1560)]
    wmask = poly_mask([wpoly])
    out = np.zeros((H, W, 3), np.float32)
    if mode == "orbit":
        out[:] = vgrad([(0, "#010208"), (1, "#05102A")])
        stars(out, seed, 700, 300, 1560, size=0.8, big=3)
        planet(out, 330, 2000, 1250, [(0, "#0C2A5A"), (0.4, "#2A6EB8"), (0.7, "#4FA6D8"), (1, "#EAF6FF")], light=(0.7, -0.5, 0.45), seed=seed, bands=3.5, atmo="#6FB8FF", atmo_k=1.4, cells=4)
        glow(out, 1180, 520, 120, C("#FFF2D0"), 0.9, "exp")
        glow(out, 1180, 520, 360, C("#FFB070"), 0.25, "exp")
    elif mode == "warp":
        out[:] = vgrad([(0, "#02040E"), (1, "#05143A")])
        cx, cy = 720, 960
        n = 520
        ang = r.random(n) * 2 * math.pi
        r0 = 60 + r.random(n) ** 0.7 * 160
        r1 = r0 + 120 + r.random(n) * 900
        lay = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(lay)
        for a, ra, rb in zip(ang, r0, r1):
            dr.line([(cx + ra * math.cos(a), cy + ra * math.sin(a)), (cx + rb * math.cos(a), cy + rb * math.sin(a))], fill=int(80 + 150 * r.random()), width=int(1 + 3 * r.random()))
        lm = np.asarray(lay, np.float32) / 255
        out += lm[..., None] * C("#BFE0FF") + blur(lm, 6)[..., None] * C("#2D7BFF") * 1.4
        glow(out, cx, cy, 260, C("#4D9BFF"), 0.9, "exp")
        glow(out, cx, cy, 70, C("#FFFFFF"), 0.9, "exp")
    else:
        out[:] = vgrad([(0, "#05030E"), (1, "#0C0620")])
        nebula(out, seed, [(0, "#000000"), (0.4, "#4B1A7A"), (0.7, "#C23A8A"), (1, "#FFC27A")], cx=1.6, warp=0.6, strength=1.0, dust=0.35)
        stars(out, seed + 2, 600, 300, 1560, size=0.8, big=2)
        # distant station
        px, py = 1000, 980
        pts = [(px - 90, py), (px + 90, py)]
        m = line_mask([pts], 8, 2) + line_mask([[(px, py - 50), (px, py + 50)]], 8, 2)
        ringm = np.zeros((H, W), np.float32)
        over(out, np.clip(m, 0, 1), "#0A0B14", 0.95)
        for dx in (-90, 90, 0):
            circ(out, px + dx, py + (0 if dx else 0), 6, "#FFC27A", 0.9)
    img = out * wmask[..., None]
    # frame structure
    frame = np.zeros((H, W, 3), np.float32)
    g = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    frame += ramp(np.broadcast_to(np.linspace(0, 1, H, dtype=np.float32)[:, None], (H, W)), [(0, "#0B0F16"), (0.5, "#18212C"), (1, "#0A0D12")])
    strut = poly_mask([[(40, 1560), (110, 1560), (640, 330), (580, 330)], [(1400, 1560), (1330, 1560), (800, 330), (860, 330)]]) * wmask
    edge = np.clip(np.abs(blur(wmask, 4) - wmask) * 4, 0, 1)
    img = img * (1 - strut[..., None]) + frame * strut[..., None]
    img = img * wmask[..., None] + frame * (1 - wmask[..., None])
    img += (edge * 0.25)[..., None] * C("#7FA8D8")
    # inner bevel on frame
    img *= (1 - 0.35 * blur(1 - wmask, 40) * 0.0)[..., None]
    # panel lines on frame
    for y in (80, 190, 1650, 1820, 2100, 2330):
        ln = np.exp(-(((YY - y) / 1.4) ** 2)) * (1 - wmask)
        img += ln[..., None] * C("#3C4C60") * 0.7
    # top lights
    for x in range(180, 1300, 140):
        circ(img, x, 140, 9, "#FFB45A" if (x // 140) % 3 else "#59E8FF", 0.9)
        glow(img, x, 140, 28, C("#FFB45A" if (x // 140) % 3 else "#59E8FF"), 0.35, "exp")
    # dashboard
    dash = poly_mask([[(0, 1560), (1440, 1560), (1440, 2560), (0, 2560)]]) * 0
    for i in range(7):
        for j in range(3):
            x, y = 130 + i * 190, 1900 + j * 120
            c_ = ["#59E8FF", "#7DFF9A", "#FFB45A", "#FF5A6A"][int(r.integers(0, 4))]
            over(img, poly_mask([[(x - 52, y - 28), (x + 52, y - 28), (x + 52, y + 28), (x - 52, y + 28)]]), "#222C3A", 1.0)
            if r.random() < 0.7:
                circ(img, x, y, 11, c_, 0.95)
                glow(img, x, y, 36, C(c_), 0.5, "exp")
    # gauges
    for gx_ in (300, 1140):
        ring(img, gx_, 1740, 120, 8, "#2A3646", 1.0)
        ring(img, gx_, 1740, 100, 4, "#59E8FF", 0.7)
        a_ = r.uniform(-2.2, 0.6)
        over(img, line_mask([[(gx_, 1740), (gx_ + 90 * math.cos(a_), 1740 + 90 * math.sin(a_))]], 6, 2), "#FFB45A", 0.95)
        glow(img, gx_, 1740, 110, C("#59E8FF"), 0.12, "exp")
    # central screen
    scr = poly_mask([[(520, 1600), (920, 1600), (940, 1860), (500, 1860)]])
    over(img, scr, "#04141A", 1.0)
    for rr in (40, 80, 120):
        ring(img, 720, 1730, rr, 2.2, "#3DE0A8", 0.7 * scr[1730, 720] + 0.0)
    img += blur(scr, 6)[..., None] * C("#2BE0A0") * 0.08
    # HUD
    hud = np.zeros((H, W), np.float32)
    cx, cy = 720, 960
    def rr_(d, ss):
        d.ellipse([(cx - 130) * ss, (cy - 130) * ss, (cx + 130) * ss, (cy + 130) * ss], outline=255, width=int(4 * ss))
        for q in range(12):
            a_ = q * math.pi / 6
            l0, l1 = (130, 160) if q % 3 == 0 else (130, 145)
            d.line([((cx + l0 * math.cos(a_)) * ss, (cy + l0 * math.sin(a_)) * ss), ((cx + l1 * math.cos(a_)) * ss, (cy + l1 * math.sin(a_)) * ss)], fill=255, width=int(3 * ss))
        d.line([((cx - 36) * ss, cy * ss), ((cx + 36) * ss, cy * ss)], fill=255, width=int(4 * ss))
        d.line([(cx * ss, (cy - 36) * ss), (cx * ss, (cy + 36) * ss)], fill=255, width=int(4 * ss))
        for q in range(-3, 4):
            if q == 0:
                continue
            ln = 60 if q % 2 == 0 else 100
            yq = cy + q * 90
            d.line([((cx - 330) * ss, yq * ss), ((cx - 330 + ln) * ss, yq * ss)], fill=255, width=int(3 * ss))
            d.line([((cx + 330 - ln) * ss, yq * ss), ((cx + 330) * ss, yq * ss)], fill=255, width=int(3 * ss))
        for sx_ in (-1, 1):
            xs = cx + sx_ * 520
            d.arc([(xs - 160) * ss, (cy - 300) * ss, (xs + 160) * ss, (cy + 300) * ss], 90 if sx_ < 0 else -90, 270 if sx_ < 0 else 90, fill=255, width=int(3 * ss))
        for (x0, y0, dx, dy) in ((300, 460, 1, 1), (1140, 460, -1, 1), (300, 1360, 1, -1), (1140, 1360, -1, -1)):
            d.line([(x0 * ss, (y0 + dy * 70) * ss), (x0 * ss, y0 * ss), ((x0 + dx * 70) * ss, y0 * ss)], fill=255, width=int(4 * ss))
    hud = draw_mask(rr_, 2) * wmask
    hc = {"orbit": "#59E8FF", "warp": "#9BFFEA", "nebula": "#FFB45A"}[mode]
    neon(img, hud, hc, core=0.5, glows=((2.5, 1.0), (9, 0.7), (26, 0.35)), k=0.95)
    return img, 1.3, 0.3


def g_racing(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "asphalt":
        cxp = 720 + 330 * np.sin(YY / 2560 * 2 * math.pi * 1.2 + 0.4)
        slope = 330 * np.cos(YY / 2560 * 2 * math.pi * 1.2 + 0.4) * (2 * math.pi * 1.2 / 2560)
        dx = (XX - cxp) / np.sqrt(1 + slope * slope)
        hw = 380
        ad = np.abs(dx)
        grass = ramp(fbm(seed, 20, 3), [(0, "#2C5A24"), (1, "#3F7A32")])
        mow = (np.floor(YY / 170) % 2)[..., None]
        grass = grass * (0.88 + 0.16 * mow)
        asp = ramp(fbm(seed + 1, 120, 3), [(0, "#2E3034"), (1, "#45484E")])
        asp *= (0.9 + 0.2 * fbm(seed + 2, 6, 4))[..., None]
        for off in (-130, 130, -210, 210):
            sk = np.exp(-(((dx - off - 22 * np.sin(YY / 190.0 + off)) / 11) ** 2)) * smooth(0.45, 0.75, fbm(seed + 3 + int(off), 3, 4, cy=14))
            asp *= (1 - 0.38 * sk)[..., None]
        inside = 1 - smooth(hw - 1.2, hw + 1.2, ad)
        img = grass * (1 - inside[..., None]) + asp * inside[..., None]
        curb_zone = smooth(hw - 1.2, hw + 1.2, ad) * (1 - smooth(hw + 52, hw + 54.5, ad))
        stripe = (np.floor((YY + dx * slope * 0.0) / 62) % 2)
        cc = np.where(stripe[..., None] > 0, C("#E8E8E8"), C("#D2231F"))
        img = img * (1 - curb_zone[..., None]) + cc * curb_zone[..., None]
        edge = (np.abs(ad - (hw - 26)) < 5).astype(np.float32)
        over(img, edge * inside, "#F2F2F2", 0.85)
        dash = (np.abs(dx) < 7) * ((np.floor(YY / 120) % 2) == 0)
        over(img, dash.astype(np.float32), "#F4C430", 0.9)
        shade = (0.9 + 0.1 * fbm(seed + 9, 3, 3))
        img *= shade[..., None]
        return img, 2.0, 0.25
    if mode == "night":
        hy = 920
        img = vgrad([(0, "#05061A"), (0.25, "#14123A"), (0.36, "#3A1E5A"), (0.4, "#E07A4A"), (1, "#E07A4A")])
        # city glow
        for _ in range(60):
            x = r.random() * W
            hh = 40 + r.random() * 120
            w_ = 30 + r.random() * 70
            m = poly_mask([[(x, hy), (x + w_, hy), (x + w_, hy - hh), (x, hy - hh)]])
            over(img, m, "#0B0A1E", 0.95)
            for _k in range(int(hh / 20)):
                if r.random() < 0.5:
                    circ(img, x + r.random() * w_, hy - r.random() * hh, 2.2, "#FFD27A", 0.9)
        t = np.clip((YY - hy) / (H - hy), 0, 1)
        z = 1.0 / np.maximum(t, 0.004)
        curve = 520 * (1 - t) ** 2.2
        cxr = 720 + curve
        hwr = 30 + 860 * t
        dxp = (XX - cxr)
        u = dxp / np.maximum(hwr, 1)
        below = YY > hy
        road = (np.abs(u) < 1.0) & below
        asph = np.broadcast_to(C("#1A1B22"), (H, W, 3)).copy()
        asph *= (0.75 + 0.5 * fbm(seed, 60, 3))[..., None]
        ph = np.floor(z * 0.55)
        curb = (np.abs(u) >= 0.88) & (np.abs(u) < 1.0) & below
        cc = np.where(((ph % 2) > 0)[..., None], C("#F0F0F0"), C("#D22A24"))
        res = np.where(road[..., None], asph, img)
        res = np.where((curb)[..., None], cc * 0.8, res)
        dash = (np.abs(u) < 0.025) & (((z * 0.9) % 1.0) < 0.5) & below
        res = np.where(dash[..., None], C("#EDEDED") * 0.8, res)
        # barriers outside
        bar = (np.abs(u) >= 1.12) & (np.abs(u) < 1.3) & below
        bc = np.where(((np.floor(z * 0.9) % 2) > 0)[..., None], C("#C8C8D0"), C("#2A2C3A"))
        res = np.where(bar[..., None], bc * 0.6, np.where(((np.abs(u) >= 1.0) & (np.abs(u) < 1.12) & below)[..., None], C("#0C0D14"), res))
        res = np.where(((np.abs(u) >= 1.3) & below)[..., None], C("#06070E"), res)
        img = res
        # lamp posts and pools of light
        for k in range(9):
            zz = 2.0 + k * 1.15 + 0.0
            tt = 1.0 / zz
            ys = hy + (H - hy) * tt * 0.22 * 4.2
            if ys > H + 100:
                continue
            tn = (ys - hy) / (H - hy)
            for sg in (-1, 1):
                xs = 720 + 520 * (1 - tn) ** 2.2 + sg * (30 + 860 * tn) * 1.45
                top = ys - 640 * tn
                over(img, line_mask([[(xs, ys), (xs, top)]], max(1.5, 14 * tn), 2), "#2A2C3A", 0.95)
                glow(img, xs, top, 90 * tn + 12, C("#FFE6B0"), 0.9, "exp")
                circ(img, xs, top, 8 * tn + 3, "#FFFFFF", 0.95)
                pool = np.exp(-(((XX - xs * 0.8 - 720 * 0.2) / (260 * tn + 20)) ** 2 + ((YY - ys) / (36 * tn + 8)) ** 2)) * 0.3
                img += pool[..., None] * C("#FFD9A0")
        glow(img, 720, H - 160, 520, C("#FFF1CC"), 0.35, "gauss")
        return img, 1.6, 0.3
    # waving checkered flag
    u = XX + 90 * np.sin(YY / 240.0 + XX / 420.0) + 40 * np.sin(YY / 90.0 + XX / 200.0)
    v = YY + 120 * np.sin(XX / 330.0 + 0.7) + 60 * np.sin(XX / 140.0 + YY / 300.0)
    ph = XX / 330.0 + 0.7
    cell = 180 * (0.7 + 0.5 * YY / H)
    chk = ((np.floor(u / cell) + np.floor(v / cell)) % 2)[..., None]
    slope = np.cos(ph) * 120 / 330.0 + np.cos(XX / 140.0 + YY / 300.0) * 60 / 140.0
    shade = np.clip(0.78 + 0.5 * slope + 0.12 * np.sin(YY / 240.0 + XX / 420.0), 0.25, 1.15)
    img = np.where(chk > 0, C("#F3EFE6"), C("#15161C")) * shade[..., None]
    weave = (0.94 + 0.06 * np.sin(XX * 1.7) * np.sin(YY * 1.7))
    img *= weave[..., None]
    bgl = ramp(np.clip(1 - np.hypot(XX / W - 0.5, YY / H - 0.4) * 1.5, 0, 1), [(0, "#22262E"), (1, "#C8D0DC")])
    return img * 1.0 + 0.0 * bgl, 1.6, 0.3


def g_arena(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "dojo":
        img = vgrad([(0, "#2C1A12"), (1, "#3A2416")])
        hall = 1500
        # shoji panes
        glo = ramp(np.clip(XX / W * 0.9 + (1 - YY / hall) * 0.3, 0, 1), [(0, "#E8B070"), (0.5, "#FFD79A"), (1, "#FF9A4A")])
        img = np.where((YY < hall)[..., None], glo, img)
        # silhouette branch
        def fb(d, ss):
            def br(x, y, a, L, wd, dep):
                if dep == 0 or L < 14:
                    return
                x2, y2 = x + L * math.cos(a), y + L * math.sin(a)
                d.line([(x * ss, y * ss), (x2 * ss, y2 * ss)], fill=255, width=max(1, int(wd * ss)))
                if dep <= 3:
                    d.ellipse([(x2 - 18) * ss, (y2 - 12) * ss, (x2 + 18) * ss, (y2 + 12) * ss], fill=255)
                br(x2, y2, a - 0.45 + r.uniform(-0.15, 0.15), L * 0.74, wd * 0.7, dep - 1)
                br(x2, y2, a + 0.4 + r.uniform(-0.15, 0.15), L * 0.72, wd * 0.7, dep - 1)
            br(1500, 780, math.pi + 0.15, 420, 22, 7)
            br(1500, 1120, math.pi - 0.2, 360, 16, 6)
        sil = draw_mask(fb, 2) * (YY < hall)
        img = img * (1 - 0.82 * sil[..., None]) + C("#4A2A18") * (0.82 * sil[..., None])
        # lattice
        lat = np.zeros((H, W), np.float32)
        for x in range(0, W + 1, 240):
            lat += np.exp(-(((XX - x) / 7.0) ** 2))
        for y in (0, 300, 750, 1200, 1500):
            lat += np.exp(-(((YY - y) / 8.0) ** 2))
        lat = np.clip(lat, 0, 1) * (YY < hall + 12)
        over(img, lat, "#3A2214", 1.0)
        # floor: perspective planks
        t = np.clip((YY - hall) / (H - hall), 0, 1)
        floor = YY > hall
        uu = (XX - 720) / (0.18 + 1.25 * t) / 150.0
        zz = 1.0 / (0.12 + t)
        plank = np.floor(uu + 0.5)
        pw = ((uu + 0.5) % 1.0)
        base = ramp(fbm(seed, 6, 4, cy=40), [(0, "#7A4A28"), (1, "#B07A48")]) * (0.82 + 0.22 * ((plank * 0.37) % 1.0))[..., None]
        seam = np.exp(-((pw * 150) / 3.0) ** 2) + np.exp(-(((1 - pw) * 150) / 3.0) ** 2)
        seam2 = np.exp(-((((zz * 2.2 + plank * 0.37) % 1.0) / 0.03) ** 2))
        base *= (1 - 0.5 * np.clip(seam + seam2, 0, 1))[..., None]
        refl = blur(np.flipud(np.roll(glo * (YY < hall)[..., None], 0, 0)), 20)
        refl = np.roll(refl, -(H - 2 * hall), 0)
        fl = base * (0.7 - 0.25 * t[..., None]) + glo * (np.exp(-t * 2.6) * 0.4)[..., None] * (0.5 + 0.5 * np.exp(-((XX - 720) / 600.0) ** 2))[..., None]
        img = np.where(floor[..., None], fl, img)
        edge = np.exp(-(((YY - hall) / 6.0) ** 2))
        over(img, edge, "#2A160C", 0.8)
        # hanging lanterns
        for lx, ly, s_ in ((200, 160, 1.0), (1240, 120, 1.2)):
            over(img, line_mask([[(lx, 0), (lx, ly)]], 3, 2), "#2A160C", 0.9)
            glow(img, lx, ly + 90, 160 * s_, C("#FF9A3A"), 0.55, "exp")
            m = poly_mask([[(lx - 50 * s_, ly), (lx + 50 * s_, ly), (lx + 62 * s_, ly + 90 * s_), (lx + 50 * s_, ly + 180 * s_), (lx - 50 * s_, ly + 180 * s_), (lx - 62 * s_, ly + 90 * s_)]])
            over(img, m, "#FF8A2A", 0.95)
            over(img, line_mask([[(lx - 56 * s_, ly + 60 * s_), (lx + 56 * s_, ly + 60 * s_)], [(lx - 56 * s_, ly + 120 * s_), (lx + 56 * s_, ly + 120 * s_)]], 4, 2), "#7A2A10", 0.7)
        return img, 1.6, 0.3
    if mode == "roof":
        img = vgrad([(0, "#2A1550"), (0.2, "#8A2F7A"), (0.34, "#F2705A"), (0.42, "#FFC27A"), (1, "#FFC27A")])
        circ(img, 980, 880, 150, "#FFF2D0")
        glow(img, 980, 880, 420, (1.0, 0.65, 0.4), 0.45, "exp")
        base_y = 1520
        layers = [("#B0607A", 0.55, 150, 380, 90), ("#7A3A6E", 0.75, 200, 560, 110), ("#2A1646", 1.0, 260, 720, 130)]
        for li, (cc, k, hmin, hmax, wmax) in enumerate(layers):
            x = -40
            polys = []
            wins = []
            while x < W + 40:
                w_ = 70 + r.random() * wmax
                h_ = hmin + r.random() * (hmax - hmin)
                y_top = base_y - h_ - li * 40
                polys.append([(x, base_y + 60), (x, y_top), (x + w_, y_top), (x + w_, base_y + 60)])
                if li == 2:
                    for wy in range(int(y_top) + 30, base_y, 34):
                        for wx in range(int(x) + 12, int(x + w_) - 12, 26):
                            if r.random() < 0.28:
                                wins.append((wx, wy))
                x += w_ + r.random() * 12
            over(img, poly_mask(polys), cc, 0.97)
            if li == 2:
                for (wx, wy) in wins:
                    circ(img, wx, wy, 5, "#FFD27A", 0.95)
        # rooftop floor + parapet
        floor = poly_mask([[(0, 1760), (W, 1760), (W, H), (0, H)]])
        conc = ramp(fbm(seed, 30, 4), [(0, "#3C3446"), (1, "#5A5068")])
        over(img, floor, "#000000", 1.0)
        img += conc * floor[..., None]
        t = np.clip((YY - 1760) / 800, 0, 1)
        img += floor[..., None] * C("#FF9A6A") * (np.exp(-t * 2.5) * 0.35)[..., None]
        seams = np.exp(-(((YY - 2060) / 3.0) ** 2)) + np.exp(-(((YY - 2330) / 4.0) ** 2))
        img *= (1 - 0.4 * seams * floor)[..., None]
        over(img, poly_mask([[(0, 1740), (W, 1740), (W, 1790), (0, 1790)]]), "#1A1226", 1.0)
        # chain-link fence
        zone = poly_mask([[(0, 900), (W, 900), (W, 1740), (0, 1740)]])
        a = (((XX + YY) % 52) < 2.2) | (((XX - YY) % 52) < 2.2)
        over(img, a.astype(np.float32) * zone, "#C8C8D8", 0.34)
        for px_ in (60, 720, 1380):
            over(img, line_mask([[(px_, 880), (px_, 1760)]], 14, 2), "#1A1226", 0.95)
        over(img, line_mask([[(0, 900), (W, 900)]], 12, 2), "#1A1226", 0.95)
        # water tower silhouette (right side, behind the fence)
        tw = poly_mask([[(1080, 1050), (1340, 1050), (1320, 1280), (1100, 1280)], [(1100, 1280), (1120, 1280), (1100, 1500), (1080, 1500)], [(1300, 1280), (1320, 1280), (1340, 1500), (1320, 1500)], [(1060, 1050), (1210, 960), (1360, 1050)]])
        over(img, tw, "#1C0F30", 0.9)
        return img, 1.6, 0.3
    # temple courtyard under the moon
    img = vgrad([(0, "#04061C"), (0.3, "#14204E"), (0.55, "#3A4C86"), (1, "#5F72A8")])
    stars(img, seed, 400, 0, 900, size=0.8)
    circ(img, 720, 620, 230, "#F2F0E2")
    glow(img, 720, 620, 520, (0.6, 0.7, 1.0), 0.4, "exp")
    hills_layer(img, seed + 1, 1180, 220, "#2C3868", "#3A4678", freq=1.6, fade=500)
    hills_layer(img, seed + 2, 1290, 140, "#1E2750", "#2A3566", freq=2.0, fade=400)

    def pagoda(d, ss, cx, base, w, tiers):
        for i in range(tiers):
            ww = w * (1 - i * 0.2)
            yb = base - i * w * 0.30
            hh = w * 0.17
            d.rectangle([(cx - ww * 0.36) * ss, (yb - hh) * ss, (cx + ww * 0.36) * ss, yb * ss], fill=255)
            d.polygon([((cx - ww * 0.62) * ss, (yb - hh) * ss), ((cx - ww * 0.5) * ss, (yb - hh - 14) * ss), ((cx + ww * 0.5) * ss, (yb - hh - 14) * ss), ((cx + ww * 0.62) * ss, (yb - hh) * ss),
                       ((cx + ww * 0.52) * ss, (yb - hh + 8) * ss), ((cx - ww * 0.52) * ss, (yb - hh + 8) * ss)], fill=255)
        d.polygon([((cx - 5) * ss, (base - tiers * w * 0.30 - w * 0.17) * ss), ((cx + 5) * ss, (base - tiers * w * 0.30 - w * 0.17) * ss), (cx * ss, (base - tiers * w * 0.30 - w * 0.17 - 90) * ss)], fill=255)
    pm = draw_mask(lambda d, ss: pagoda(d, ss, 720, 1480, 520, 5), 2)
    over(img, pm, "#10162E", 0.97)
    for (px_, py_) in ((690, 1380), (750, 1300), (720, 1210), (700, 1130)):
        circ(img, px_, py_, 5, "#FFC27A", 0.9)
    # steps
    for i in range(9):
        y0 = 1480 + i * 70
        wd = 460 + i * 120
        c_ = ramp(np.array([i / 9.0], np.float32), [(0, "#6C7AA8"), (1, "#2A3156")])[0]
        over(img, poly_mask([[(720 - wd, y0), (720 + wd, y0), (720 + wd, y0 + 70), (720 - wd, y0 + 70)]]), tuple(c_), 1.0)
        over(img, poly_mask([[(720 - wd, y0), (720 + wd, y0), (720 + wd, y0 + 6), (720 - wd, y0 + 6)]]), "#A8B4E0", 0.5)
    over(img, poly_mask([[(0, 2110), (W, 2110), (W, H), (0, H)]]), "#1B2146", 1.0)
    for y in (2180, 2300, 2430):
        over(img, poly_mask([[(0, y), (W, y), (W, y + 4), (0, y + 4)]]), "#2E3868", 0.8)
    for x in (-120, 1560):
        pass
    # red pillars
    for x0 in (60, 1320):
        over(img, poly_mask([[(x0 - 60, 0), (x0 + 60, 0), (x0 + 60, 2110), (x0 - 60, 2110)]]), "#B3261E", 1.0)
        over(img, poly_mask([[(x0 - 60, 0), (x0 - 36, 0), (x0 - 36, 2110), (x0 - 60, 2110)]]), "#E85A3A", 0.5)
        over(img, poly_mask([[(x0 + 34, 0), (x0 + 60, 0), (x0 + 60, 2110), (x0 + 34, 2110)]]), "#4A0A0A", 0.55)
        over(img, poly_mask([[(x0 - 90, 2040), (x0 + 90, 2040), (x0 + 90, 2110), (x0 - 90, 2110)]]), "#3A4166", 1.0)
    # stone lanterns
    for lx in (330, 1110):
        glow(img, lx, 1840, 150, C("#FFB45A"), 0.5, "exp")
        over(img, poly_mask([[(lx - 24, 1900), (lx + 24, 1900), (lx + 18, 2100), (lx - 18, 2100)], [(lx - 70, 1790), (lx + 70, 1790), (lx + 50, 1740), (lx - 50, 1740)], [(lx - 46, 1790), (lx + 46, 1790), (lx + 46, 1900), (lx - 46, 1900)]]), "#262C50", 1.0)
        over(img, poly_mask([[(lx - 24, 1805), (lx + 24, 1805), (lx + 24, 1885), (lx - 24, 1885)]]), "#FFC27A", 0.95)
    fog(img, 1560, 120, "#7E8CBE", 0.45, seed=seed + 4)
    fog(img, 2060, 150, "#7E8CBE", 0.35, seed=seed + 5)
    return img, 1.5, 0.3


reg("rpg_map", g_rpg_map)
reg("dungeon_tiles", g_dungeon_tiles)
reg("cockpit", g_cockpit)
reg("racing", g_racing)
reg("arena", g_arena)

W2("rpg-world-map", "World Map", "A hand-inked parchment continent with mountains, forests, towns and a compass rose.", "gaming", True, True, 0.15, "rpg_map", mode="continent")
W2("rpg-island-chart", "Island Chart", "A sea chart of scattered islands on aged parchment with a teal ocean wash.", "gaming", False, False, 0.15, "rpg_map", mode="archipelago")
W2("rpg-dungeon-plan", "Dungeon Plan", "An inked dungeon floor plan with rooms, doors, hatching and a red goal marker.", "gaming", False, False, 0.15, "rpg_map", mode="dungeon")
W2("tiles-flagstone", "Mossy Flagstones", "Worn flagstone floor with green moss in the cracks, lit by a warm torch.", "gaming", False, False, 0.15, "dungeon_tiles", mode="flag")
W2("tiles-rune-floor", "Rune Floor", "Dark stone tiles with glowing cyan runes carved into every third slab.", "gaming", True, False, 0.15, "dungeon_tiles", mode="rune")
W2("tiles-lava-cracks", "Lava Cracks", "Black basalt plates split by glowing orange lava.", "gaming", False, False, 0.15, "dungeon_tiles", mode="lava")
W2("cockpit-orbit", "Cockpit Orbit", "From a starship cockpit you see a blue planet below, with HUD lines in cyan.", "gaming", True, True, 0.1, "cockpit", mode="orbit")
W2("cockpit-hyperjump", "Cockpit Hyperjump", "Light streaks fly past the cockpit glass as the ship jumps to hyperspace.", "gaming", False, False, 0.1, "cockpit", mode="warp")
W2("cockpit-nebula", "Cockpit Nebula", "An amber HUD over a magenta nebula and a distant space station.", "gaming", False, False, 0.1, "cockpit", mode="nebula")
W2("race-asphalt-bend", "Asphalt Bend", "A top-down race track with tyre marks, red and white kerbs and a bright centre line.", "gaming", False, False, 0.15, "racing", mode="asphalt")
W2("race-night-straight", "Night Straight", "A floodlit night circuit rushes toward a hot orange horizon.", "gaming", True, False, 0.15, "racing", mode="night")
W2("race-chequered-flag", "Chequered Flag", "A black and white chequered flag ripples across the whole screen.", "gaming", False, False, 0.15, "racing", mode="flag")
W2("arena-dojo-dusk", "Dojo at Dusk", "Sunset glows through paper screens onto a polished wooden dojo floor.", "gaming", True, True, 0.15, "arena", mode="dojo")
W2("arena-rooftop", "Rooftop Arena", "A city rooftop behind a chain-link fence, with a low sun and a water tower.", "gaming", False, False, 0.15, "arena", mode="roof")
W2("arena-moon-temple", "Moon Temple", "A moonlit temple courtyard with red pillars, stone lanterns and a pagoda on the steps.", "gaming", True, False, 0.15, "arena", mode="temple")


# ====================================================================== WAVE 2: gaming 3
def flat_hill(img, y, amp, fr, ph, cbody, ctop, stripe=None, sk=0.16):
    xs = np.linspace(0, 1, 721)
    prof = y + amp * np.sin(xs * fr + ph) + amp * 0.45 * np.sin(xs * fr * 2.1 + ph * 1.7)
    m = poly_below(prof, 2)
    top = m * (1 - np.roll(m, 12, 0))
    body = ramp(np.clip((YY - y + amp) / 900.0, 0, 1), [(0, cbody), (1, ctop if False else cbody)])
    over(img, m, cbody)
    if stripe:
        st = (((XX + YY * 0.55) % 120) < 40).astype(np.float32) * m
        over(img, st, stripe, sk)
    over(img, top, ctop, 0.9)
    return m


def cloud_blob(img, cx, cy, s, color="#FFFFFF", shade="#CFE6F5"):
    parts = [(0, 0, 1.0), (-1.1, 0.25, 0.75), (1.1, 0.3, 0.8), (-0.5, -0.45, 0.8), (0.55, -0.4, 0.7)]
    def f(d, ss):
        for (dx, dy, rr) in parts:
            x, y, r_ = cx + dx * s, cy + dy * s * 0.55, rr * s * 0.6
            d.ellipse([(x - r_) * ss, (y - r_) * ss, (x + r_) * ss, (y + r_) * ss], fill=255)
        d.rectangle([(cx - 1.6 * s) * ss, (cy + 0.2 * s) * ss, (cx + 1.7 * s) * ss, (cy + 0.55 * s) * ss], fill=255)
    m = draw_mask(f, 2)
    low = m * (1 - np.roll(m, -int(s * 0.18), 0))
    over(img, m, shade)
    over(img, m * np.roll(m, int(s * 0.16), 0), color)


def g_parallax(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "day":
        img = vgrad([(0, "#3FA9F5"), (0.5, "#9BDDFF"), (1, "#E4F8FF")])
        pal = ("#7FD8D0", "#9CEBE2", "#4BBE62", "#8CE284", "#2E9A4A", "#68D064")
        sun = (1150, 360, "#FFF3A8")
    elif mode == "dusk":
        img = vgrad([(0, "#2B1A5E"), (0.4, "#B2457E"), (0.7, "#FF9A5A"), (1, "#FFD28A")])
        pal = ("#7A3F8E", "#A860A0", "#4E2A78", "#8C4A9A", "#2D1A52", "#5A3A8A")
        sun = (360, 980, "#FFF0B8")
    else:
        img = vgrad([(0, "#FFB6E4"), (0.5, "#FFD6F0"), (1, "#FFF1D6")])
        pal = ("#B8F0E0", "#D4FFF4", "#9ADCC8", "#C4F4E4", "#7CC8B8", "#A8E8D8")
        sun = (260, 380, "#FFFFFF")
    circ(img, sun[0], sun[1], 150, sun[2])
    glow(img, sun[0], sun[1], 360, C(sun[2]), 0.28, "exp")
    for (cx, cy, s) in ((300, 520, 190), (1050, 780, 240), (520, 1000, 150), (1200, 1250, 170)):
        cloud_blob(img, cx, cy, s, "#FFFFFF" if mode != "dusk" else "#FFC9A8", "#D4E6F6" if mode == "day" else ("#C46A9A" if mode == "dusk" else "#F6C6E6"))
    flat_hill(img, 1500, 130, 7.0, 0.5, pal[0], pal[1], None)
    flat_hill(img, 1760, 150, 9.0, 2.0, pal[2], pal[3], pal[4], 0.14)
    flat_hill(img, 2050, 120, 6.0, 4.0, pal[4], pal[5], pal[2], 0.18)
    # ground tiles
    g = poly_mask([[(0, 2300), (W, 2300), (W, H), (0, H)]])
    chk = ((np.floor(XX / 120) + np.floor((YY - 2300) / 120)) % 2)
    over(img, g, "#7A4A2A" if mode != "candy" else "#E8A0C8", 1.0)
    over(img, g * chk, "#6A3C20" if mode != "candy" else "#DC8EB8", 1.0)
    over(img, g * (1 - np.roll(g, 26, 0)), "#58C060" if mode == "day" else ("#7A4AAA" if mode == "dusk" else "#B8F0E0"), 1.0)
    # bushes and coins
    for bx in (210, 760, 1250):
        for dx, dy, rr in ((0, 0, 70), (-70, 20, 52), (70, 26, 56)):
            circ(img, bx + dx, 2290 + dy - 40, rr, pal[2])
    for i in range(9):
        a = i / 8
        x = 330 + 780 * a
        y = 1380 - 220 * math.sin(a * math.pi)
        circ(img, x, y, 30, "#FFC21A")
        circ(img, x, y, 20, "#FFE678")
        over(img, line_mask([[(x, y - 14), (x, y + 14)]], 5, 2), "#E8A010", 0.9)
    if mode == "dusk":
        stars(img, seed, 220, 0, 800, size=0.8)
    if mode == "candy":
        for _ in range(160):
            x, y = r.random() * W, 1500 + r.random() * 800
            a = r.random() * math.pi
            over(img, line_mask([[(x, y), (x + 18 * math.cos(a), y + 18 * math.sin(a))]], 6, 2), ["#FF7AB8", "#7AD0FF", "#FFE066", "#9AFFB8"][int(r.integers(0, 4))], 0.9)
    return img, 0.8, 0.0


def draw_gem(d, cx, cy, R, kind, color, rot, ss):
    n = {"hex": 6, "oct": 8, "tri": 3, "dia": 4, "pent": 5, "sq": 4}[kind]
    off = rot + (math.pi / 4 if kind == "sq" else (math.pi / 2 if kind in ("tri", "pent") else 0))
    if kind == "dia":
        off = rot
    ang = [off + k * 2 * math.pi / n for k in range(n)]
    sx = 0.78 if kind == "dia" else 1.0
    vs = [(cx + R * math.cos(a) * sx, cy + R * math.sin(a)) for a in ang]
    base = np.array(col(color))
    L = np.array([-0.55, -0.8])
    d.polygon([(x * ss, y * ss) for x, y in vs], fill=tuple(int(c * 255 * 0.55) for c in base))
    inner = [(cx + (x - cx) * 0.52, cy + (y - cy) * 0.52) for x, y in vs]
    for k in range(n):
        a0, a1 = vs[k], vs[(k + 1) % n]
        nx_, ny_ = (a0[0] + a1[0]) / 2 - cx, (a0[1] + a1[1]) / 2 - cy
        nl = math.hypot(nx_, ny_) + 1e-6
        lam = max(0.0, (nx_ * L[0] + ny_ * L[1]) / nl)
        c = np.clip(base * (0.55 + 0.9 * lam) + lam * 0.12, 0, 1)
        d.polygon([(a0[0] * ss, a0[1] * ss), (a1[0] * ss, a1[1] * ss), (inner[(k + 1) % n][0] * ss, inner[(k + 1) % n][1] * ss), (inner[k][0] * ss, inner[k][1] * ss)], fill=tuple(int(v * 255) for v in c))
    d.polygon([(x * ss, y * ss) for x, y in inner], fill=tuple(int(np.clip(v * 1.25 + 0.12, 0, 1) * 255) for v in base))
    ti = [(cx + (x - cx) * 0.26 - R * 0.05, cy + (y - cy) * 0.26 - R * 0.07) for x, y in vs]
    d.polygon([(x * ss, y * ss) for x, y in ti], fill=tuple(int(np.clip(v * 0.6 + 0.55, 0, 1) * 255) for v in base))
    hx, hy = cx - R * 0.34, cy - R * 0.42
    d.ellipse([(hx - R * 0.09) * ss, (hy - R * 0.06) * ss, (hx + R * 0.09) * ss, (hy + R * 0.06) * ss], fill=(255, 255, 255))


def g_gems(seed, mode):
    r = np.random.default_rng(seed)
    ss = 2
    kinds = ["hex", "oct", "tri", "dia", "pent", "sq"]
    cols = ["#E8283C", "#2F7BFF", "#2BC866", "#FFC21A", "#B04BFF", "#FF7A1A", "#2BD6E8"]
    if mode == "board":
        bg = vgrad([(0, "#0E1640"), (1, "#1A0E3A")])
        im = Image.fromarray((bg * 255).astype(np.uint8)).resize((W * ss, H * ss), Image.BILINEAR)
        d = ImageDraw.Draw(im)
        cs = 192
        for j in range(13):
            for i in range(7):
                x0, y0 = 48 + i * cs, 96 + j * cs
                tone = 18 if (i + j) % 2 else 30
                d.rounded_rectangle([(x0 + 4) * ss, (y0 + 4) * ss, (x0 + cs - 4) * ss, (y0 + cs - 4) * ss], radius=22 * ss, fill=(tone, tone + 6, tone + 40))
                k = int(r.integers(0, 6))
                draw_gem(d, x0 + cs / 2, y0 + cs / 2, 66, kinds[k], cols[k], 0.0, ss)
        return from_pil(im), 1.4, 0.25
    if mode == "cascade":
        bg = vgrad([(0, "#05081E"), (1, "#101A4A")])
        im = Image.fromarray((bg * 255).astype(np.uint8)).resize((W * ss, H * ss), Image.BILINEAR)
        d = ImageDraw.Draw(im, "RGBA")
        items = []
        for _ in range(46):
            x, y = r.random() * W, r.random() * H
            items.append((y, x, 38 + r.random() * 70, int(r.integers(0, 6)), r.random() * 6.28))
        for (y, x, R, k, rot) in sorted(items):
            c = col(cols[k])
            L_ = R * (3 + 5 * r.random())
            for q in range(6):
                a_ = int(70 * (1 - q / 6))
                d.line([(x * ss, (y - q * L_ / 6 - R * 0.5) * ss), (x * ss, (y - (q + 1) * L_ / 6 - R * 0.5) * ss)], fill=tuple(int(v * 255) for v in c) + (a_,), width=int(R * 1.1 * ss * (1 - q / 8)))
        for (y, x, R, k, rot) in sorted(items):
            draw_gem(d, x, y, R, kinds[k], cols[k], rot, ss)
        img = from_pil(im)
        img += blur(img, 30) * 0.25
        return img, 1.4, 0.3
    bg = vgrad([(0, "#02141C"), (0.5, "#06303A"), (1, "#0B1A1E")])
    glow(bg, 720, 1500, 900, (0.2, 0.55, 0.5), 0.55, "exp")
    im = Image.fromarray((np.clip(bg, 0, 1) * 255).astype(np.uint8)).resize((W * ss, H * ss), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    items = []
    for _ in range(430):
        y = 2560 - (r.random() ** 1.6) * 1500
        x = 720 + (r.standard_normal() * 360) * (0.55 + (y - 1000) / 2600)
        R = 52 + r.random() * 62
        items.append((y, x, R, int(r.integers(0, 6)), r.random() * 6.28))
    for (y, x, R, k, rot) in sorted(items):
        draw_gem(d, x, y, R, kinds[k], cols[k], rot, ss)
    img = from_pil(im)
    img += blur(img, 40) * 0.2
    return img, 1.4, 0.3


def g_bullets(seed, mode):
    r = np.random.default_rng(seed)
    img = vgrad([(0, "#05030E"), (1, "#0C0620")])
    cx, cy = 720, 1150
    glow(img, cx, cy, 800, (0.18, 0.06, 0.32), 0.9, "gauss")

    def bullet(x, y, rad, color):
        if -40 < x < W + 40 and -40 < y < H + 40:
            glow(img, x, y, rad * 2.0, C(color), 0.28, "exp", 3.0)
            circ(img, x, y, rad, color)
            circ(img, x, y, rad * 0.6, "#FFFFFF", 0.95)
    if mode == "spiral":
        cols = ["#FF3F6C", "#FFB02E", "#3FD6FF", "#B04BFF", "#6BFF9A"]
        for arm in range(5):
            for k in range(70):
                th = arm * 2 * math.pi / 5 + k * 0.17
                rr = 70 + k * 24
                bullet(cx + rr * math.cos(th), cy + rr * math.sin(th), 9 + k * 0.28, cols[arm])
        circ(img, cx, cy, 56, "#FFFFFF")
        glow(img, cx, cy, 120, C("#FF7AB0"), 0.9, "exp")
    elif mode == "rings":
        for ring_i in range(9):
            R = 140 + ring_i * 150
            n = 24 + ring_i * 6
            off = ring_i * 0.37
            for k in range(n):
                th = off + k * 2 * math.pi / n
                if ring_i % 3 == 1 and abs(((th % (2 * math.pi)) - math.pi / 2 + math.pi) % (2 * math.pi) - math.pi) < 0.35:
                    continue
                bullet(cx + R * math.cos(th), cy + R * math.sin(th), 11 + ring_i * 1.3, ["#3FD6FF", "#FFFFFF", "#6B8CFF"][ring_i % 3] if ring_i % 2 == 0 else ["#FF4F8A", "#FFB02E"][ring_i % 2 * 0 + (k % 2)])
        circ(img, cx, cy, 44, "#FFE6A0")
        glow(img, cx, cy, 150, C("#FFB02E"), 0.9, "exp")
    else:
        for layer in range(3):
            kk = (5, 7, 4)[layer]
            Rm = (700, 520, 330)[layer]
            colr = (["#FF5FA8", "#FF9AC8"], ["#FFD04A", "#FFF0A0"], ["#6BFF9A", "#B8FFD0"])[layer]
            for q in range(260):
                th = q / 260 * 4 * math.pi
                rr = Rm * abs(math.cos(kk * th / 2))
                bullet(cx + rr * math.cos(th + layer), cy + rr * math.sin(th + layer), 8 + 6 * (rr / Rm), colr[q % 2])
        circ(img, cx, cy, 40, "#FFFFFF")
        glow(img, cx, cy, 140, C("#FF9AC8"), 0.9, "exp")
    return img, 1.2, 0.2


def g_pulp(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "horizon":
        img = vgrad([(0, "#0A2A3A"), (0.6, "#14586A"), (1, "#1E7A7A")])
        stars(img, seed, 150, 0, 1600, size=0.9, tints=("#FFF4D8",))
        hz = 1820
        # big striped planet
        pcx, pcy, pr = 600, 1250, 620
        d = np.hypot(XX - pcx, YY - pcy)
        pm = 1 - smooth(pr - 1.2, pr + 1.2, d)
        bands = 0.5 + 0.5 * np.sin((YY - pcy) / 46.0 + 3 * np.sin(XX / 330.0))
        pc = ramp(bands, [(0, "#E8663A"), (0.5, "#F6A04A"), (1, "#FFD890")])
        lam = np.clip(1 - np.hypot((XX - pcx + 220) / pr, (YY - pcy + 220) / pr) * 0.9, 0.0, 1.0)
        pc = pc * (0.45 + 0.7 * lam[..., None])
        over(img, pm, (0, 0, 0), 0.0)
        img = img * (1 - pm[..., None]) + pc * pm[..., None]
        circ(img, 1170, 560, 110, "#F2E8C8")
        circ(img, 1130, 530, 92, "#14586A", 0.0)
        # ground
        xs = np.linspace(0, 1, 721)
        prof = hz + 40 * np.sin(xs * 9) + 24 * np.sin(xs * 23 + 1)
        g = poly_below(prof, 2)
        over(img, g, "#C2542E")
        gt = ramp(np.clip((YY - hz) / 740, 0, 1), [(0, "#F08A4A"), (1, "#7A2A1A")])
        img = img * (1 - g[..., None]) + gt * g[..., None]
        for _ in range(26):
            x, y = r.random() * W, hz + 80 + r.random() ** 0.7 * 600
            sc = 0.4 + (y - hz) / 600
            def f(dd, ss, x=x, y=y, sc=sc):
                dd.ellipse([(x - 90 * sc) * ss, (y - 22 * sc) * ss, (x + 90 * sc) * ss, (y + 22 * sc) * ss], outline=255, width=int(5 * ss))
            over(img, draw_mask(f, 2), "#4A1A10", 0.6)
        for x in (1000, 1120, 880):
            h_ = 220 + r.random() * 260
            over(img, poly_mask([[(x - 30, hz + 80), (x + 30, hz + 80), (x + 8, hz - h_ + 80), (x - 8, hz - h_ + 80)]]), "#2A1410", 0.95)
        return img, 3.0, 0.2
    if mode == "ringed":
        img = vgrad([(0, "#120A2E"), (1, "#2A1450")])
        stars(img, seed, 260, 0, H, size=0.9, tints=("#FFE8C0", "#FFFFFF"))
        cx, cy, pr = 720, 1180, 420
        ang = -0.38
        c_, s_ = math.cos(ang), math.sin(ang)
        xr = (XX - cx) * c_ + (YY - cy) * s_
        yr = -(XX - cx) * s_ + (YY - cy) * c_
        rho = np.hypot(xr, yr / 0.28)
        ringm = ((rho > 560) & (rho < 820)).astype(np.float32)
        ringm = blur(ringm, 1.0)
        back = ringm * (yr < 0)
        front = ringm * (yr >= 0)
        rc = ramp(np.clip((rho - 560) / 260, 0, 1), [(0, "#F6D9A0"), (0.4, "#E8903A"), (0.7, "#F6D9A0"), (1, "#C2542E")])
        stripe = 0.7 + 0.3 * np.sin(rho / 9.0)
        img = img * (1 - back[..., None]) + rc * stripe[..., None] * back[..., None]
        d = np.hypot(XX - cx, YY - cy)
        pm = 1 - smooth(pr - 1.2, pr + 1.2, d)
        lam = np.clip(1 - np.hypot((XX - cx + 150) / pr, (YY - cy + 170) / pr) * 0.8, 0, 1)
        pcol = ramp(0.5 + 0.5 * np.sin((YY - cy) / 36.0 + np.sin(XX / 120.0)), [(0, "#3FB6B0"), (1, "#C8F0D8")]) * (0.35 + 0.8 * lam[..., None])
        # halftone shadow dots
        cell = 22.0
        gx = ((XX + YY * 0.0) % cell) - cell / 2
        gy = ((YY) % cell) - cell / 2
        dotr = (1 - lam) * cell * 0.62
        dm = (np.hypot(gx, gy) < dotr).astype(np.float32)
        pcol = pcol * (1 - 0.55 * dm[..., None] * smooth(0.25, 0.8, 1 - lam)[..., None])
        img = img * (1 - pm[..., None]) + pcol * pm[..., None]
        img = img * (1 - front[..., None]) + rc * stripe[..., None] * front[..., None]
        sh = pm * front
        return img, 3.0, 0.25
    img = vgrad([(0, "#0A1230"), (0.55, "#2A2A6A"), (1, "#8A3A6A")])
    stars(img, seed, 220, 0, 1700, size=0.9)
    for (cx, cy, rr, c1, c2) in ((520, 760, 330, "#F6E6C0", "#C8B890"), (1080, 1300, 150, "#F2A66A", "#C26A3A")):
        d = np.hypot(XX - cx, YY - cy)
        m = 1 - smooth(rr - 1.2, rr + 1.2, d)
        lam = np.clip(1 - np.hypot((XX - cx + rr * 0.3) / rr, (YY - cy + rr * 0.3) / rr) * 0.75, 0, 1)
        body = ramp(lam, [(0, c2), (1, c1)])
        img = img * (1 - m[..., None]) + body * m[..., None]
        for _ in range(9):
            a = r.random() * 6.28
            dd = r.random() ** 0.5 * rr * 0.7
            crater_r = rr * (0.06 + 0.12 * r.random())
            cx2, cy2 = cx + dd * math.cos(a), cy + dd * math.sin(a)
            ring(img, cx2, cy2, crater_r, 4, c2, 0.8 * (m[int(np.clip(cy2, 0, H - 1)), int(np.clip(cx2, 0, W - 1))] > 0.5))
    xs = np.linspace(0, 1, 721)
    prof = 2000 + 70 * np.sin(xs * 6 + 1) + 40 * np.sin(xs * 17)
    g = poly_below(prof, 2)
    img = img * (1 - g[..., None]) + ramp(np.clip((YY - 1900) / 660, 0, 1), [(0, "#3A1E4A"), (1, "#14082A")]) * g[..., None]
    for x, h_, w_ in ((180, 360, 70), (330, 220, 50), (1040, 460, 80), (1230, 280, 56), (700, 160, 40)):
        over(img, poly_mask([[(x - w_, 2100), (x - w_ * 0.4, 2100 - h_), (x + w_ * 0.4, 2100 - h_ + 40), (x + w_, 2100)]]), "#1A0C30", 0.97)
        over(img, poly_mask([[(x - w_ * 0.4, 2100 - h_), (x + w_ * 0.4, 2100 - h_ + 40), (x + w_, 2100), (x + w_ * 0.2, 2100)]]), "#2E1A4A", 0.9)
    return img, 3.0, 0.25


reg("parallax", g_parallax)
reg("gems", g_gems)
reg("bullets", g_bullets)
reg("pulp", g_pulp)

W2("platform-day-hills", "Platformer Day", "Rounded striped hills, puffy clouds, a coin arc and bushes in bright 2D platformer style.", "gaming", False, True, 0.2, "parallax", mode="day")
W2("platform-dusk-hills", "Platformer Dusk", "The same hill-hopping world at sunset, in violet and orange with early stars.", "gaming", False, False, 0.2, "parallax", mode="dusk")
W2("platform-candy-hills", "Platformer Candy", "A sweet world of mint hills, sprinkles and pink clouds, with a golden coin arc.", "gaming", True, False, 0.2, "parallax", mode="candy")
W2("puzzle-gem-board", "Gem Board", "A grid of faceted red, blue, green, gold and violet gems on a dark game board.", "gaming", False, True, 0.2, "gems", mode="board")
W2("puzzle-gem-cascade", "Gem Cascade", "Faceted gems tumble through a dark blue void leaving soft streaks behind them.", "gaming", False, False, 0.2, "gems", mode="cascade")
W2("puzzle-gem-hoard", "Gem Hoard", "A glittering treasure pile of gems under a teal glow.", "gaming", True, False, 0.2, "gems", mode="hoard")
W2("bullet-spiral-storm", "Spiral Storm", "Five arms of glowing bullets spiral out from a white core, like a shooter boss pattern.", "gaming", False, False, 0.1, "bullets", mode="spiral")
W2("bullet-ring-burst", "Ring Burst", "Nine expanding rings of cyan, white and pink orbs with a gap to dodge through.", "gaming", False, False, 0.1, "bullets", mode="rings")
W2("bullet-petal-bloom", "Petal Bloom", "Rose curves of pink, gold and green bullets bloom like a flower around a bright core.", "gaming", True, False, 0.1, "bullets", mode="petals")
W2("pulp-striped-horizon", "Pulp Horizon", "A huge striped planet and a pale moon above a cratered orange plain, in a 1950s sci-fi poster style.", "retro", False, False, 0.2, "pulp", mode="horizon")
W2("pulp-ringed-world", "Pulp Ringed World", "A teal ringed planet with halftone shading on deep violet, like vintage space-age print art.", "retro", True, False, 0.2, "pulp", mode="ringed")
W2("pulp-twin-moons", "Pulp Twin Moons", "Two cratered moons float over dark rock spires on an alien plain.", "retro", False, False, 0.2, "pulp", mode="moons")


# ====================================================================== WAVE 2: anime scenery + seasons
def cumulus(img, seed, yc, hh, lit, mid, shade, thr=0.5, cx=2.2, light=(-26, -22), blur_s=3.0, k=1.0, xs=None):
    d = fbm(seed, cx, 6, 0.55, cy=cx * 3.0)
    env = np.exp(-(((YY - yc) / hh) ** 2))
    cl = smooth(thr, thr + 0.1, d * 0.75 + env * 0.55)
    cl = blur(cl, blur_s)
    nb = np.roll(np.roll(blur(cl, 9), light[1], 0), light[0], 1)
    litk = np.clip((cl - nb) * 3.2 + 0.45, 0, 1)
    low = np.clip((YY - yc) / hh, -1, 1) * 0.5 + 0.5
    c = ramp(litk * (1 - 0.45 * low), [(0, shade), (0.5, mid), (1, lit)])
    m = (cl * k)[..., None]
    img *= 1 - m
    img += c * m


def wires(img, pts_a, pts_b, n, sag, color="#1A1A28", w=3.0, k=0.9):
    lines = []
    for i in range(n):
        a = (pts_a[0], pts_a[1] + i * 26)
        b = (pts_b[0], pts_b[1] + i * 26)
        xs = np.linspace(a[0], b[0], 80)
        t = np.linspace(0, 1, 80)
        ys = a[1] + (b[1] - a[1]) * t + sag * 4 * t * (1 - t)
        lines.append(list(zip(xs, ys)))
    over(img, line_mask(lines, w, 2), color, k)


def pole(img, x, ytop, ybot, wd, color="#1A1A28", arms=((0, 0.1, 150),), k=0.95):
    over(img, line_mask([[(x, ytop), (x, ybot)]], wd, 2), color, k)
    for (_, off, ln) in arms:
        y = ytop + (ybot - ytop) * off
        over(img, line_mask([[(x - ln, y), (x + ln, y)]], wd * 0.55, 2), color, k)


def g_anime_day(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "sakura":
        img = vgrad([(0, "#74BCEE"), (0.3, "#C6E4F8"), (0.45, "#FFF2F2"), (1, "#FFF2F2")])
        cumulus(img, seed, 560, 300, "#FFFFFF", "#E8F2FB", "#B5CFE8", thr=0.52)
        vp = (720, 1080)
        road = poly_mask([[vp, (-500, H), (W + 500, H)]])
        g = ramp(np.clip((YY - vp[1]) / 1500, 0, 1), [(0, "#E6E0E8"), (1, "#7C8096")])
        over(img, road, "#000000", 1.0)
        img += g * road[..., None]
        side = poly_mask([[vp, (-1200, H), (-500, H)], [vp, (W + 500, H), (W + 1200, H)]])
        img += ramp(np.clip((YY - vp[1]) / 1500, 0, 1), [(0, "#F4E6EC"), (1, "#BCA8B4")]) * side[..., None]
        for q in range(18):
            t0, t1 = (q / 18) ** 1.8, ((q + 0.5) / 18) ** 1.8
            over(img, poly_mask([[(720 - 8 * t0 - 2, vp[1] + 1500 * t0), (720 + 8 * t0 + 2, vp[1] + 1500 * t0), (720 + 8 * t1 + 2, vp[1] + 1500 * t1), (720 - 8 * t1 - 2, vp[1] + 1500 * t1)]]), "#FFFFFF", 0.85)
        glow(img, vp[0], vp[1], 200, C("#FFFFFF"), 0.35, "exp")
        for t in (0.06, 0.1, 0.16, 0.26, 0.42, 0.68, 1.05):
            yg = vp[1] + 1500 * t
            for sd in (-1, 1):
                x = 720 + sd * (900 * t + 230 * t)
                tw = 70 * t
                over(img, poly_mask([[(x - tw, yg), (x + tw, yg), (x + tw * 0.6, yg - 800 * t), (x - tw * 0.6, yg - 800 * t)]]), "#6A4A4A", 0.97)
                cc = (x - sd * 90 * t, yg - 900 * t)
                for _ in range(46):
                    a = r.random() * 6.28
                    dd = r.random() ** 0.6 * 470 * t
                    bx, by = cc[0] + dd * math.cos(a) * 1.15, cc[1] + dd * math.sin(a) * 0.8
                    rr = (70 + r.random() * 90) * t
                    lit = np.clip(0.5 - (bx - cc[0]) / (900 * t) * (-sd) * 0.0 + (cc[1] - by) / (700 * t), 0, 1)
                    colr = ramp(np.array([lit], np.float32), [(0, "#E58AAA"), (0.5, "#F9B3CB"), (1, "#FFE6EE")])[0]
                    circ(img, bx, by, rr, colr, 0.95, soft=max(2, 14 * t))
            if t > 0.3:
                pass
        # utility pole
        pole(img, 250, 350, 2400, 26, "#4A3A4A", arms=((0, 0.06, 150), (0, 0.14, 110)))
        wires(img, (250, 420), (vp[0] - 20, vp[1] - 30), 3, 80, "#4A3A4A", 3, 0.7)
        for _ in range(180):
            x, y = r.random() * W, r.random() * H
            sz = 6 + r.random() * 9
            a = r.random() * 3.14
            m = poly_mask([[(x, y), (x + sz * math.cos(a), y + sz * math.sin(a) * 0.7), (x + sz * 0.5, y + sz * 0.9)]], 1)
            over(img, m, "#FFC3D6" if r.random() < 0.7 else "#FFFFFF", 0.85)
        img += blur(np.clip(img - 0.85, 0, 1), 30) * 0.3
        return img, 1.0, 0.1
    if mode == "crossing":
        img = vgrad([(0, "#262A78"), (0.22, "#8A4A9A"), (0.42, "#F0707A"), (0.55, "#FFB070"), (0.6, "#FFE0A8"), (1, "#FFE0A8")])
        circ(img, 720, 1450, 200, "#FFF6D8")
        glow(img, 720, 1450, 500, (1.0, 0.7, 0.4), 0.5, "exp")
        cumulus(img, seed, 780, 330, "#FFD2A0", "#D06A98", "#4A2E72", thr=0.5, cx=2.0, light=(-20, 16))
        cumulus(img, seed + 3, 1180, 200, "#FFC8A0", "#B8588E", "#5A3A80", thr=0.55, cx=2.8, light=(-20, 16), k=0.8)
        hills_layer(img, seed + 1, 1560, 120, "#6A3A76", "#4A2A62", freq=2.2, fade=300)
        # town silhouettes
        polys = []
        x = 0
        while x < W:
            w_ = 50 + r.random() * 80
            h_ = 30 + r.random() * 110
            polys.append([(x, 1620), (x, 1620 - h_), (x + w_, 1620 - h_), (x + w_, 1620)])
            x += w_ + 6
        over(img, poly_mask(polys), "#3A2258", 0.97)
        ground = poly_mask([[(0, 1620), (W, 1620), (W, H), (0, H)]])
        img = img * (1 - ground[..., None]) + ramp(np.clip((YY - 1620) / 940, 0, 1), [(0, "#4A2A5C"), (1, "#1A1030")]) * ground[..., None]
        vp = (720, 1620)
        for off in (-60, 60):
            over(img, poly_mask([[(vp[0] + off * 0.06, vp[1]), (vp[0] + off * 0.06 + 6, vp[1]), (vp[0] + off * 9 + 14, H), (vp[0] + off * 9 - 14, H)]]), "#C8B0C8", 0.85)
        for q in range(26):
            t = (q / 26) ** 2.1
            y = vp[1] + 940 * t
            wd = 3 + 380 * t
            hh = 3 + 20 * t
            over(img, poly_mask([[(720 - wd, y), (720 + wd, y), (720 + wd, y + hh), (720 - wd, y + hh)]]), "#2A1A2E", 0.85)
        road = poly_mask([[(0, 2040), (W, 2040), (W, 2300), (0, 2300)]])
        over(img, road, "#2A2438", 0.95)
        over(img, poly_mask([[(0, 2040), (W, 2040), (W, 2050), (0, 2050)]]), "#FFB070", 0.5)
        for x_ in (230, 1210):
            pole(img, x_, 1260, 2260, 24, "#1A1228", arms=((0, 0.0, 0),))
            sd = 1 if x_ < 720 else -1
            over(img, line_mask([[(x_ - 90, 1380), (x_ + 90, 1280)], [(x_ - 90, 1280), (x_ + 90, 1380)]], 14, 2), "#F2F2F2", 0.9)
            for dy in (1470, 1560):
                for dx in (-46, 46):
                    glow(img, x_ + dx, dy, 60, C("#FF2A2A"), 0.9, "exp")
                    circ(img, x_ + dx, dy, 22, "#FF5A4A")
                over(img, line_mask([[(x_ - 78, dy), (x_ + 78, dy)]], 4, 2), "#1A1228", 0.0)
            arm_end = 560 if x_ < 720 else 880
            for q in range(10):
                x0 = x_ + (arm_end - x_) * q / 10
                x1 = x_ + (arm_end - x_) * (q + 1) / 10
                over(img, line_mask([[(x0, 1900 - q * 0 + 0), (x1, 1900)]], 22, 2), "#FF3A3A" if q % 2 == 0 else "#F4F4F4", 0.95)
        pole(img, 80, 200, 2100, 34, "#120A1E", arms=((0, 0.04, 200), (0, 0.1, 150)))
        pole(img, 1360, 380, 2100, 28, "#120A1E", arms=((0, 0.05, 160),))
        wires(img, (80, 250), (1360, 430), 4, 170, "#120A1E", 3, 0.85)
        wires(img, (80, 520), (720, 1490), 2, 120, "#120A1E", 2.5, 0.0)
        return img, 1.2, 0.2
    img = vgrad([(0, "#1C74E0"), (0.4, "#5CB4F6"), (0.7, "#BFE6FF"), (1, "#E4F6FF")])
    cumulus(img, seed, 520, 360, "#FFFFFF", "#EAF4FF", "#9CBEE4", thr=0.5, cx=1.8, light=(-30, -28))
    cumulus(img, seed + 4, 1250, 300, "#FFFFFF", "#E2F0FF", "#98B8E0", thr=0.52, cx=2.4, light=(-30, -28))
    hills_layer(img, seed + 2, 1750, 140, "#6A9AC6", "#7AAAD0", freq=1.8, fade=300)
    hills_layer(img, seed + 3, 1900, 90, "#3E8A5A", "#2C7048", freq=2.6, fade=300)
    fld = poly_mask([[(0, 2000), (W, 2000), (W, H), (0, H)]])
    gcol = ramp(np.clip((YY - 2000) / 560, 0, 1), [(0, "#6AD060"), (1, "#2A7A38")])
    img = img * (1 - fld[..., None]) + gcol * fld[..., None]
    for i in range(34):
        y = 2000 + (i / 34) ** 1.6 * 560
        over(img, line_mask([[(0, y), (W, y)]], 2 + i * 0.25, 1), "#8CE070", 0.28)
    for x_, top, wd in ((120, 140, 34), (900, 560, 14), (1180, 700, 9)):
        pole(img, x_, top, 2060, wd, "#12203A", arms=((0, 0.05, wd * 4.5), (0, 0.12, wd * 3.5)))
    wires(img, (120, 220), (900, 640), 3, 150, "#12203A", 4, 0.9)
    wires(img, (900, 640), (1180, 760), 3, 60, "#12203A", 2.5, 0.9)
    wires(img, (120, 220), (-100, 300), 3, 40, "#12203A", 4, 0.9)
    glow(img, 1250, 160, 420, C("#FFF6D0"), 0.55, "exp")
    return img, 1.0, 0.1


def g_anime_night(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "rooftop":
        img = vgrad([(0, "#03041A"), (0.4, "#14104A"), (0.7, "#3A2468"), (0.85, "#8A4A7E"), (1, "#C26A6E")])
        mw = fbm(seed, 2.0, 5)
        band = np.exp(-(((XX - 160 - YY * 0.55) / 320.0) ** 2)) * mw
        img += band[..., None] * C("#7F9CFF") * 0.55 + (band ** 2)[..., None] * C("#FFB8E0") * 0.4
        stars(img, seed, 1500, 0, 1700, size=0.8, big=10)
        for (x, y, L) in ((1080, 360, 380), (500, 700, 300), (1250, 980, 240)):
            over(img, line_mask([[(x, y), (x - L * 0.8, y - L * 0.6)]], 4, 2) * np.exp(-np.hypot(XX - x, YY - y) / (L * 0.5)), "#FFFFFF", 0.9)
            glow(img, x, y, 20, C("#FFFFFF"), 0.9, "exp")
        # skyline
        polys = []
        x = -20
        while x < W:
            w_ = 40 + r.random() * 90
            h_ = 40 + r.random() * 160
            polys.append([(x, 1980), (x, 1980 - h_), (x + w_, 1980 - h_), (x + w_, 1980)])
            for wy in range(int(1980 - h_ + 14), 1970, 24):
                for wx in range(int(x + 8), int(x + w_ - 8), 18):
                    if r.random() < 0.22:
                        circ(img, wx, wy, 3.2, "#FFD27A", 0.9)
            x += w_ + 5
        over(img, poly_mask(polys), "#120C2A", 0.96)
        glow(img, 720, 1990, 700, (1.0, 0.45, 0.4), 0.3, "gauss")
        floor = poly_mask([[(0, 2090), (W, 2090), (W, H), (0, H)]])
        over(img, floor, "#0A0818", 1.0)
        over(img, poly_mask([[(0, 2090), (W, 2090), (W, 2100), (0, 2100)]]), "#6A5A9A", 0.7)
        over(img, poly_mask([[(120, 1700), (460, 1700), (460, 2090), (120, 2090)], [(100, 1660), (480, 1660), (440, 1700), (140, 1700)]]), "#0F0B22", 1.0)
        over(img, line_mask([[(1120, 2090), (1120, 1520)], [(1120, 1620), (1000, 1520)], [(1120, 1620), (1240, 1520)]], 8, 2), "#0F0B22", 1.0)
        glow(img, 1120, 1520, 40, C("#FF3A3A"), 0.9, "exp")
        circ(img, 1120, 1520, 8, "#FF6A5A")
        for x_ in range(0, W + 1, 160):
            over(img, line_mask([[(x_, 2090), (x_, 2300)]], 10, 2), "#0F0B22", 1.0)
        over(img, line_mask([[(0, 2150), (W, 2150)], [(0, 2260), (W, 2260)]], 8, 2), "#0F0B22", 1.0)
        return img, 1.2, 0.2
    if mode == "alley":
        img = vgrad([(0, "#07051A"), (0.5, "#1A0E3A"), (1, "#0A0816")])
        vp = (720, 1150)
        walls = poly_mask([[(-200, -200), (vp[0], vp[1]), (-200, 2800)], [(W + 200, -200), (vp[0], vp[1]), (W + 200, 2800)]])
        over(img, walls, "#1C1540", 1.0)
        sc = np.zeros((H, W, 3), np.float32)
        cols_ = ["#FF2FA8", "#18D6E8", "#FF8A2D", "#9A5CFF", "#6BFF9A"]
        for t in (0.1, 0.14, 0.2, 0.28, 0.38, 0.52, 0.72, 0.98):
            for sd in (-1, 1):
                x = 720 + sd * (130 + 820 * t * 0.9)
                yc = vp[1] - 60 * t + 380 * t * (1 if sd > 0 else 0.8) - 260 * t
                wd, hg = 150 * t, 560 * t
                c = cols_[int(r.integers(0, 5))]
                m = poly_mask([[(x - wd / 2, yc - hg / 2), (x + wd / 2, yc - hg / 2), (x + wd / 2, yc + hg / 2), (x - wd / 2, yc + hg / 2)]])
                neon(sc, m, c, core=0.35, glows=((3 * t + 1, 1.0), (12 * t + 2, 0.7), (40 * t + 5, 0.4)), k=0.8)
                for q in range(5):
                    yy_ = yc - hg / 2 + hg * (q + 0.5) / 5
                    over(sc, line_mask([[(x - wd * 0.25, yy_), (x + wd * 0.25, yy_)]], max(1.5, 10 * t), 2), "#FFFFFF", 0.0)
                sc[...] += 0
                for q in range(int(r.integers(3, 6))):
                    wy = yc + hg * (0.6 + r.random() * 1.2)
                    wx = x + sd * (-20 * t * 8 * 0)
                    if wy < 2300:
                        over(sc, poly_mask([[(wx - 28 * t, wy), (wx + 28 * t, wy), (wx + 28 * t, wy + 40 * t), (wx - 28 * t, wy + 40 * t)]]), "#FFD27A", 0.0)
                        sc += np.exp(-(((XX - wx) / (30 * t + 3)) ** 2 + ((YY - wy) / (22 * t + 3)) ** 2))[..., None] * C("#FFB060") * 0.6
        img += sc
        gnd = poly_mask([[vp, (-600, H), (W + 600, H)]])
        wet = ramp(np.clip((YY - vp[1]) / 1400, 0, 1), [(0, "#1A1432"), (1, "#05040C")])
        img = img * (1 - gnd[..., None]) + wet * gnd[..., None]
        refl = blur(sc, 18)
        refl = blur(refl, 4)
        for sh in (1, 2, 3):
            img += np.roll(refl, (sh * 120 + int(0)), 0) * gnd[..., None] * (0.5 / sh)
        refl2 = np.flipud(sc)
        refl2 = np.roll(refl2, int(2 * vp[1] - H + 0), 0)
        img += blur(refl2, 10) * gnd[..., None] * 0.8 * np.clip((YY - vp[1]) / 600, 0, 1)[..., None]
        fg = fbm(seed + 2, 3, 4, cy=6)
        img += (smooth(0.55, 0.9, fg) * np.exp(-((YY - 1500) / 360.0) ** 2))[..., None] * C("#6A5A9A") * 0.25
        lay = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(lay)
        for _ in range(520):
            x, y = r.random() * (W + 300), r.random() * H
            L = 50 + r.random() * 90
            dr.line([(x, y), (x - L * 0.22, y + L)], fill=int(60 + 140 * r.random()), width=2)
        rn = np.asarray(lay, np.float32) / 255
        img += rn[..., None] * C("#A8B8FF") * 0.45
        return img, 1.3, 0.3
    img = vgrad([(0, "#05081E"), (0.5, "#0C1640"), (1, "#14204E")])
    stars(img, seed, 500, 0, 1300, size=0.8, big=3)
    circ(img, 1100, 360, 90, "#F4F0DA")
    circ(img, 1136, 336, 84, "#05081E", 0.0)
    m = (1 - smooth(88, 91, np.hypot(XX - 1100, YY - 360))) * smooth(80, 83, np.hypot(XX - 1138, YY - 338))
    img += blur(m, 30)[..., None] * C("#8AA0E0") * 0.3
    over(img, m, "#F4F0DA", 1.0)
    for x_, top, wd in ((140, 200, 30), (980, 520, 18)):
        pole(img, x_, top, 2300, wd, "#04060F", arms=((0, 0.05, wd * 4.5), (0, 0.12, wd * 3.5)))
    wires(img, (140, 280), (980, 590), 3, 150, "#04060F", 4, 0.9)
    wires(img, (140, 280), (-100, 340), 3, 40, "#04060F", 4, 0.9)
    over(img, poly_mask([[(0, 1900), (W, 1900), (W, H), (0, H)]]), "#0B1230", 1.0)
    for x_ in range(0, W, 120):
        h_ = 80 + r.random() * 120
        circ(img, x_ + 60, 1900 - h_ * 0.2, 90 + h_ * 0.2, "#050A1C", 0.98)
    # vending machine
    mx, my, mw, mh = 600, 1360, 380, 760
    glow(img, mx + mw / 2, my + mh / 2, 520, C("#7CE8FF"), 0.5, "exp")
    over(img, poly_mask([[(mx, my), (mx + mw, my), (mx + mw, my + mh), (mx, my + mh)]]), "#E8F4FA", 1.0)
    over(img, poly_mask([[(mx + 24, my + 70), (mx + mw - 120, my + 70), (mx + mw - 120, my + mh - 150), (mx + 24, my + mh - 150)]]), "#BFF6FF", 1.0)
    cc = ["#FF4F4F", "#3F7BFF", "#FFD23F", "#4FE08A", "#FFFFFF", "#FF8A3A"]
    for ry in range(5):
        for cx_ in range(5):
            over(img, poly_mask([[(mx + 36 + cx_ * 44, my + 90 + ry * 96), (mx + 66 + cx_ * 44, my + 90 + ry * 96), (mx + 66 + cx_ * 44, my + 150 + ry * 96), (mx + 36 + cx_ * 44, my + 150 + ry * 96)]]), cc[(ry * 3 + cx_) % 6], 1.0)
    over(img, poly_mask([[(mx + mw - 100, my + 80), (mx + mw - 30, my + 80), (mx + mw - 30, my + 400), (mx + mw - 100, my + 400)]]), "#8AA6B8", 1.0)
    over(img, poly_mask([[(mx + 30, my + mh - 120), (mx + mw - 30, my + mh - 120), (mx + mw - 30, my + mh - 60), (mx + 30, my + mh - 60)]]), "#46586A", 1.0)
    over(img, poly_mask([[(mx, my - 30), (mx + mw, my - 30), (mx + mw, my + 40), (mx, my + 40)]]), "#FF4A5A", 1.0)
    cone = np.exp(-(((XX - 790) / 460.0) ** 2 + ((YY - 2300) / 150.0) ** 2))
    img += cone[..., None] * C("#9FEFFF") * 0.55
    refl = np.exp(-(((XX - 790) / 150.0) ** 2) - ((YY - 2300) / 380.0) ** 2) * (YY > 2120)
    img += refl[..., None] * C("#CFF8FF") * 0.4
    for _ in range(24):
        bx, by = 620 + r.random() * 360, 1260 + r.random() * 600
        glow(img, bx, by, 8, C("#FFFFFF"), 0.8, "exp")
    return img, 1.3, 0.2


def g_winter(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "cabin":
        img = vgrad([(0, "#060B2A"), (0.4, "#18285E"), (0.62, "#3C5A94"), (1, "#C8D8F0")])
        stars(img, seed, 500, 0, 1300, size=0.8, big=3)
        circ(img, 300, 420, 100, "#EEF2FF")
        glow(img, 300, 420, 360, (0.6, 0.7, 1.0), 0.35, "exp")
        hills_layer(img, seed + 1, 1500, 150, "#5A78AE", "#7A94C4", freq=1.6, fade=400, lit="#C8D8F0", lit_k=0.4)
        hills_layer(img, seed + 2, 1700, 120, "#8AA4D0", "#B8CCEA", freq=1.9, fade=400)
        m = forest_mask(seed + 3, lambda x: 1700 + 40 * math.sin(x / 200), 46, (180, 340), 0.3)
        over(img, m, "#0E1C3C", 0.96)
        hills_layer(img, seed + 4, 1950, 90, "#D8E4F6", "#B0C4E4", freq=2.2, fade=600)
        # cabin
        cx, cy = 780, 1880
        over(img, poly_mask([[(cx - 190, cy), (cx + 190, cy), (cx + 190, cy - 170), (cx - 190, cy - 170)]]), "#3C2630", 1.0)
        over(img, poly_mask([[(cx - 240, cy - 160), (cx, cy - 330), (cx + 240, cy - 160)]]), "#E8F0FF", 1.0)
        over(img, poly_mask([[(cx - 240, cy - 160), (cx, cy - 330), (cx, cy - 300), (cx - 210, cy - 160)]]), "#9FB6DA", 0.9)
        over(img, poly_mask([[(cx + 100, cy - 310), (cx + 150, cy - 310), (cx + 150, cy - 230), (cx + 100, cy - 250)]]), "#3C2630", 1.0)
        for wx in (cx - 110, cx + 50):
            glow(img, wx + 40, cy - 90, 120, C("#FFB24A"), 0.7, "exp")
            over(img, poly_mask([[(wx, cy - 130), (wx + 80, cy - 130), (wx + 80, cy - 50), (wx, cy - 50)]]), "#FFC25A", 1.0)
            over(img, line_mask([[(wx + 40, cy - 130), (wx + 40, cy - 50)], [(wx, cy - 90), (wx + 80, cy - 90)]], 4, 2), "#7A4A2A", 0.9)
        sm = fbm(seed + 8, 4, 4, cy=8) * np.exp(-(((XX - (cx + 130 - (cy - YY) * 0.15)) / (30 + (cy - YY) * 0.15)) ** 2)) * ((YY < cy - 330) & (YY > cy - 800))
        img += sm[..., None] * C("#C8D4EE") * 0.35
        over(img, poly_mask([[(cx - 240, cy), (cx + 240, cy), (cx + 300, cy + 40), (cx - 300, cy + 40)]]), "#C0D0EC", 1.0)
        bokeh(img, seed, 70, ["#FFFFFF", "#DCE8FF"], 5, 22, (0.15, 0.55), 0.2)
        return img, 1.3, 0.2
    if mode == "birch":
        img = vgrad([(0, "#B8C8F0"), (0.4, "#F2D8E4"), (0.7, "#FFE8D0"), (1, "#FFF6EA")])
        glow(img, 960, 1250, 700, (1.0, 0.85, 0.65), 0.5, "exp")
        for i, t in enumerate((0.3, 0.4, 0.55, 0.75, 1.0)):
            trunks, marks = [], []
            for _ in range(5 + i):
                x = r.random() * W
                wd = 36 * t * (0.8 + 0.4 * r.random())
                trunks.append([(x - wd, -10), (x + wd, -10), (x + wd * 1.15, H + 10), (x - wd * 1.15, H + 10)])
                for _k in range(int(8 + 14 * t)):
                    yk = r.random() * H
                    marks.append([(x - wd, yk), (x - wd * 0.2 + r.random() * wd * 1.0, yk), (x - wd * 0.2, yk + 10 * t + 4), (x - wd, yk + 8 * t + 3)])
            tint = ramp(np.array([t], np.float32), [(0, "#D8C8E0"), (1, "#FFFFFF")])[0]
            over(img, poly_mask(trunks, 1), tuple(tint), 0.5 + 0.5 * t)
            over(img, poly_mask(marks, 1), "#2A2430", (0.5 + 0.4 * t) * 0.7)
        over(img, np.clip(fbm(seed, 3, 3, cy=2) * 1.2 - 0.3, 0, 1) * 0, "#FFFFFF", 0.0)
        over(img, poly_mask([[(0, 2100), (W, 2100), (W, H), (0, H)]]), "#FFFFFF", 0.95)
        sn = ramp(fbm(seed + 3, 4, 3), [(0, "#D8D4EC"), (1, "#FFFFFF")])
        img = img * (1 - 0.0) + 0
        floor = poly_mask([[(0, 2100), (W, 2100), (W, H), (0, H)]])
        img = img * (1 - floor[..., None]) + sn * floor[..., None]
        fog(img, 1800, 320, "#FFF2E8", 0.5, seed=seed + 5)
        bokeh(img, seed, 60, ["#FFFFFF", "#FFE8D0"], 4, 16, (0.2, 0.7), 0.2)
        return img, 1.3, 0.1
    img = vgrad([(0, "#3A2A58"), (1, "#7A4A6A")])
    bokeh(img, seed, 90, ["#FFB24A", "#FFD27A", "#FF7A5A", "#7AD0FF"], 30, 130, (0.25, 0.7), 0.5)
    img = blur(img, 4)
    fr = fbm(seed + 1, 4, 3)
    ang = fr * 6.28
    mf = np.zeros((H, W), np.float32)
    for k in range(3):
        w1 = fbm(seed + 11 + k, 2.2, 3)
        w2 = fbm(seed + 21 + k, 2.2, 3)
        u = XX * (0.01 + 0.004 * k) + w1 * 6
        v = YY * (0.012 + 0.004 * k) + w2 * 6
        mf += np.exp(-(np.abs(np.sin(u * 3.1) * np.sin(v * 3.1))) * 14) * (0.8 - 0.2 * k)
    mf = np.clip(mf, 0, 1)
    icy = ramp(fbm(seed + 6, 3, 4), [(0, "#BFD8F0"), (1, "#F4FAFF")])
    cover = smooth(0.35, 0.7, fbm(seed + 7, 2.4, 5))
    frost = np.clip(cover * (0.35 + 0.8 * mf), 0, 1)
    img = img * (1 - frost[..., None] * 0.8) + icy * frost[..., None] * 0.8
    edge = fbm(seed + 8, 40, 2) * 0.15
    img += (cover * edge)[..., None] * 0.4
    return img, 1.3, 0.15


def g_halloween(seed, mode):
    r = np.random.default_rng(seed)

    def pumpkin(img, cx, cy, R, face=True):
        ps = [(-0.62, 0.78), (0.0, 1.0), (0.62, 0.78)]
        for dx, wx in ((-0.55, 0.62), (0.55, 0.62), (0, 0.78)):
            d = np.hypot((XX - (cx + dx * R)) / (R * wx), (YY - cy) / (R * 0.82))
            a = 1 - smooth(0.98, 1.02, d)
            col_ = ramp(np.clip(1 - d, 0, 1) + 0.1 + (cx - XX) / (R * 3), [(0, "#7A2A08"), (0.5, "#E05A10"), (1, "#FF9A3A")])
            img *= 1 - a[..., None]
            img += col_ * a[..., None]
        over(img, poly_mask([[(cx - 12, cy - R * 0.85), (cx + 14, cy - R * 0.85), (cx + 26, cy - R * 1.15), (cx + 4, cy - R * 1.1)]]), "#3A2A10", 1.0)
        if face:
            f = poly_mask([[(cx - R * 0.45, cy - R * 0.15), (cx - R * 0.15, cy - R * 0.15), (cx - R * 0.3, cy - R * 0.45)],
                           [(cx + R * 0.15, cy - R * 0.15), (cx + R * 0.45, cy - R * 0.15), (cx + R * 0.3, cy - R * 0.45)],
                           [(cx - R * 0.5, cy + R * 0.1), (cx - R * 0.25, cy + R * 0.3), (cx - R * 0.1, cy + R * 0.18), (cx + R * 0.1, cy + R * 0.3), (cx + R * 0.25, cy + R * 0.18), (cx + R * 0.5, cy + R * 0.1), (cx + R * 0.3, cy + R * 0.5), (cx - R * 0.3, cy + R * 0.5)]])
            glow(img, cx, cy, R * 0.8, C("#FFB020"), 0.45, "exp")
            over(img, f, "#FFD23A", 1.0)
            img += blur(f, 8)[..., None] * C("#FF8A10") * 0.6
    if mode == "moon":
        img = vgrad([(0, "#1A0A38"), (0.45, "#5A1E4A"), (0.7, "#C2502A"), (1, "#FF9A3A")])
        circ(img, 720, 840, 340, "#FFD56A")
        circ(img, 720, 840, 340, "#FFE9A0", 0.0)
        glow(img, 720, 840, 700, (1.0, 0.55, 0.15), 0.45, "exp")
        for cx_, cy_, rr in ((620, 760, 60), (850, 900, 40), (700, 1020, 50)):
            circ(img, cx_, cy_, rr, "#F2B650", 0.5)
        cumulus(img, seed, 1200, 170, "#C2502A", "#5A1E4A", "#1A0A38", thr=0.5, k=0.7)
        def tree(d, ss):
            def br(x, y, a, L, w, dep):
                if dep == 0:
                    return
                x2, y2 = x + L * math.cos(a), y + L * math.sin(a)
                d.line([(x * ss, y * ss), (x2 * ss, y2 * ss)], fill=255, width=max(1, int(w * ss)))
                br(x2, y2, a - 0.5 + r.uniform(-0.2, 0.2), L * 0.75, w * 0.68, dep - 1)
                br(x2, y2, a + 0.45 + r.uniform(-0.2, 0.2), L * 0.72, w * 0.68, dep - 1)
                if dep > 3 and r.random() < 0.5:
                    br(x2, y2, a + r.uniform(-0.3, 0.3), L * 0.6, w * 0.6, dep - 2)
            br(220, 2100, -math.pi / 2 - 0.1, 420, 60, 8)
        over(img, draw_mask(tree, 2), "#0A0414", 1.0)
        over(img, poly_mask([[(0, 2060), (W, 2060), (W, H), (0, H)]]), "#0A0414", 1.0)
        for (x, y, R) in ((800, 2260, 190), (1180, 2330, 130), (450, 2380, 100)):
            pumpkin(img, x, y, R)
        return img, 1.3, 0.3
    if mode == "house":
        img = vgrad([(0, "#0A0420"), (0.5, "#2A0E4A"), (0.75, "#6A2A6A"), (1, "#C25A5A")])
        stars(img, seed, 260, 0, 1200, size=0.8)
        circ(img, 1050, 460, 110, "#F2EEDA")
        glow(img, 1050, 460, 360, (0.7, 0.7, 1.0), 0.3, "exp")
        hills_layer(img, seed + 1, 1640, 200, "#1A0C36", "#12082A", freq=1.2, fade=500)
        hx, hy = 620, 1560
        polys = [[(hx - 200, hy), (hx - 200, hy - 260), (hx - 120, hy - 260), (hx - 120, hy - 380), (hx - 60, hy - 330), (hx, hy - 470), (hx + 60, hy - 330), (hx + 120, hy - 380), (hx + 120, hy - 260), (hx + 220, hy - 260), (hx + 220, hy)],
                 [(hx - 230, hy - 260), (hx - 60, hy - 400), (hx + 250, hy - 260)], [(hx - 150, hy - 380), (hx - 120, hy - 540), (hx - 90, hy - 380)]]
        over(img, poly_mask(polys), "#06020E", 1.0)
        for (wx, wy, ww, wh) in ((-150, -200, 40, 70), (50, -220, 40, 70), (-20, -400, 36, 60), (150, -170, 36, 60), (-150, -80, 36, 60)):
            glow(img, hx + wx + ww / 2, hy + wy + wh / 2, 100, C("#FFB020"), 0.45, "exp")
            over(img, poly_mask([[(hx + wx, hy + wy), (hx + wx + ww, hy + wy), (hx + wx + ww, hy + wy + wh), (hx + wx, hy + wy + wh)]]), "#FFC83A", 1.0)
        over(img, poly_mask([[(0, 1620), (W, 1620), (W, H), (0, H)]]), "#08030F", 1.0)
        for x_ in range(40, W, 90):
            over(img, line_mask([[(x_, 1760), (x_, 1990)]], 7, 2), "#08030F", 1.0)
        over(img, line_mask([[(0, 1800), (W, 1800)], [(0, 1940), (W, 1940)]], 7, 2), "#08030F", 1.0)
        for k in range(14):
            x, y = 400 + r.random() * 700, 460 + r.random() * 520
            s = 22 + r.random() * 30
            over(img, poly_mask([[(x, y), (x - s, y - s * 0.4), (x - s * 0.5, y + s * 0.1), (x, y + s * 0.4), (x + s * 0.5, y + s * 0.1), (x + s, y - s * 0.4)]]), "#04010A", 0.95)
        return img, 1.3, 0.3
    img = vgrad([(0, "#18082E"), (0.5, "#5A1E6A"), (1, "#FF7A2A")])
    circ(img, 720, 1250, 480, "#FFB84A")
    glow(img, 720, 1250, 900, (1.0, 0.45, 0.15), 0.5, "exp")
    cumulus(img, seed, 1600, 220, "#FF9A3A", "#8A2A6A", "#2A0E4A", thr=0.5, k=0.8)
    cumulus(img, seed + 5, 700, 240, "#C2502A", "#4A1A5A", "#14062A", thr=0.52, k=0.8)
    for _ in range(120):
        t = r.random()
        a = t * 6.28 * 2.2
        rr = 90 + t * 900
        x = 720 + rr * math.cos(a) * 1.05 + r.standard_normal() * 24
        y = 1250 + rr * math.sin(a) * 1.25 + r.standard_normal() * 24
        s = 18 + 52 * (1 - t) * (0.5 + r.random())
        ph = r.random() * 0.8
        over(img, poly_mask([[(x, y), (x - s, y - s * (0.5 + ph)), (x - s * 0.6, y + s * 0.05), (x - s * 0.3, y - s * 0.1), (x, y + s * 0.35), (x + s * 0.3, y - s * 0.1), (x + s * 0.6, y + s * 0.05), (x + s, y - s * (0.5 + ph))]]), "#07020F", 0.97)
    return img, 1.3, 0.3


reg("anime_day", g_anime_day)
reg("anime_night", g_anime_night)
reg("winter", g_winter)
reg("halloween", g_halloween)

W2("anime-sakura-street", "Sakura Street Noon", "A sunny one-point street lined with cherry trees, pink petals and a utility pole.", "anime", False, True, 0.1, "anime_day", mode="sakura")
W2("anime-train-crossing", "Crossing at Sunset", "Red crossing lights blink beside empty rails under a huge orange sky and power lines.", "anime", True, False, 0.1, "anime_day", mode="crossing")
W2("anime-summer-wires", "Summer Wires", "Towering white clouds, a deep blue sky and power lines over green fields.", "anime", False, False, 0.1, "anime_day", mode="summer")
W2("anime-starry-rooftop", "Starry Rooftop", "A quiet rooftop under the Milky Way, with shooting stars and a glowing city below.", "anime", True, True, 0.1, "anime_night", mode="rooftop")
W2("anime-neon-alley", "Rainy Neon Alley", "A narrow wet alley of glowing signs, rain and reflections in the stones.", "anime", True, False, 0.1, "anime_night", mode="alley")
W2("anime-vending-night", "Vending Machine Night", "A lone vending machine glows cyan on a wet street beneath a thin crescent moon.", "anime", False, False, 0.1, "anime_night", mode="vending")
W2("winter-snowy-cabin", "Snowy Cabin", "A cabin with warm lit windows and smoke, deep in a snowy blue pine forest at night.", "seasonal", False, True, 0.1, "winter", mode="cabin")
W2("winter-birch-dawn", "Birch Dawn", "Pale birch trunks, soft pink mist and drifting snow at winter sunrise.", "seasonal", False, False, 0.1, "winter", mode="birch")
W2("winter-frost-glass", "Frost on Glass", "Icy frost patterns on a window with blurred warm lights from outside.", "seasonal", True, False, 0.15, "winter", mode="frost")
W2("halloween-pumpkin-moon", "Pumpkin Moon", "A huge amber moon, a bare tree and glowing jack-o'-lanterns on a dark hill.", "seasonal", False, False, 0.1, "halloween", mode="moon")
W2("halloween-haunted-hill", "Haunted Hill", "A crooked house with lit windows on a hill, bats and a pale moon over a fence.", "seasonal", True, False, 0.1, "halloween", mode="house")
W2("halloween-bat-swirl", "Bat Swirl", "A swirling flock of bats crosses a giant orange moon in a purple storm sky.", "seasonal", False, False, 0.1, "halloween", mode="bats")


# ====================================================================== WAVE 2: nature
def rays(img, ang, xs, wd, color, k=0.25, y_fade=None):
    c_, s_ = math.cos(ang), math.sin(ang)
    v = (XX - W / 2) * c_ + (YY - H / 2) * s_
    acc = np.zeros((H, W), np.float32)
    for x0, w in zip(xs, wd):
        acc += np.exp(-(((v - x0) / w) ** 2))
    if y_fade is not None:
        acc *= y_fade
    img += acc[..., None] * C(color) * k


def milky(img, seed, ang=0.5, k=0.55, c1="#7F9CFF", c2="#FFB8E0"):
    mw = fbm(seed, 2.4, 5)
    v = (XX - W / 2) * math.sin(ang) - (YY - H / 2) * math.cos(ang)
    band = np.exp(-((v / 300.0) ** 2)) * mw
    img += band[..., None] * C(c1) * k + (band ** 2)[..., None] * C(c2) * k * 0.7


def g_night_desert(seed, mode):
    r = np.random.default_rng(seed)
    img = vgrad([(0, "#02030F"), (0.4, "#0C1650"), (0.7, "#3A3A8A"), (1, "#8A5A8A")])
    if mode == "saguaro":
        milky(img, seed, 0.9)
        stars(img, seed, 1400, 0, 1800, size=0.8, big=6)
        hills_layer(img, seed + 1, 1700, 90, "#2A2660", "#1A1848", freq=1.3, fade=300)
        hills_layer(img, seed + 2, 1900, 120, "#1C1844", "#100E30", freq=1.8, fade=300)
        over(img, poly_mask([[(0, 2080), (W, 2080), (W, H), (0, H)]]), "#0A0820", 1.0)

        def cactus(d, ss, x, base, h, w):
            def rr(x0, y0, x1, y1):
                d.rounded_rectangle([x0 * ss, y0 * ss, x1 * ss, y1 * ss], radius=(x1 - x0) / 2 * ss, fill=255)
            rr(x - w / 2, base - h, x + w / 2, base)
            rr(x - w * 2.0, base - h * 0.62, x - w * 1.0, base - h * 0.3)
            rr(x - w * 2.0, base - h * 0.34, x - w * 0.2, base - h * 0.28)
            rr(x + w * 1.0, base - h * 0.75, x + w * 2.0, base - h * 0.42)
            rr(x + w * 0.2, base - h * 0.5, x + w * 2.0, base - h * 0.44)
        for (x, b, h, w) in ((260, 2090, 780, 66), (1080, 2110, 560, 50), (720, 2060, 300, 34)):
            over(img, draw_mask(lambda d, ss, x=x, b=b, h=h, w=w: cactus(d, ss, x, b, h, w), 2), "#05040E", 1.0)
        return img, 1.2, 0.2
    if mode == "arch":
        milky(img, seed, 0.3, 0.7)
        stars(img, seed, 1600, 0, 2000, size=0.8, big=8)
        def f(d, ss):
            d.polygon([(x * ss, y * ss) for x, y in [(0, 2560), (0, 1150), (240, 920), (480, 1010), (720, 860), (960, 1000), (1200, 930), (1440, 1160), (1440, 2560)]], fill=255)
            hole = [(720 + 430 * math.cos(a_), 1560 - 600 * math.sin(a_)) for a_ in np.linspace(0, math.pi, 40)] + [(290, 2300), (1150, 2300)]
            d.polygon([(x * ss, y * ss) for x, y in hole], fill=0)
        m = draw_mask(f, 2)
        n = fbm(seed + 1, 8, 5, cy=14)
        rock = ramp(n, [(0, "#0A0818"), (1, "#2A1E3E")])
        over(img, m, "#000000", 1.0)
        img += rock * m[..., None]
        glow(img, 720, 2000, 700, (0.7, 0.35, 0.3), 0.18, "gauss")
        return img, 1.2, 0.3
    stars(img, seed, 700, 0, 1500, size=0.8, big=3)
    circ(img, 940, 520, 150, "#F4F0DA")
    m = (1 - smooth(148, 151, np.hypot(XX - 940, YY - 520))) * smooth(130, 133, np.hypot(XX - 990, YY - 490))
    glow(img, 940, 520, 520, (0.5, 0.6, 1.0), 0.28, "exp")
    h = np.zeros((H, W), np.float32)
    for i, (y0, a, f, sp) in enumerate(((1500, 150, 1.5, 0.0), (1760, 230, 1.1, 2.0), (2060, 300, 0.8, 4.0))):
        pass
    yy = np.arange(H, dtype=np.float32)[:, None]
    xx = np.arange(W, dtype=np.float32)[None, :]
    hh = 0
    for i, (y0, a, f, ph, col_top, col_bot) in enumerate(((1450, 120, 5.0, 0.0, "#5A5AA8", "#2A2A68"), (1720, 190, 3.5, 1.7, "#4A4A98", "#1E1E58"), (2050, 260, 2.6, 3.4, "#34347E", "#0F0F3A"))):
        xs = np.linspace(0, 1, 721)
        prof = y0 + a * np.sin(xs * f * 2 + ph) + a * 0.4 * np.sin(xs * f * 5.3 + ph * 2)
        mk = poly_below(prof, 2)
        pr = np.interp(np.arange(W), np.linspace(0, W - 1, 721), prof)[None, :]
        slope = np.gradient(np.interp(np.arange(W), np.linspace(0, W - 1, 721), prof))
        slope = blur(slope[None, :], 3)[0]
        lit = smooth(-1.2, 0.4, -slope)[None, :] * np.exp(-np.clip(YY - pr, 0, None) / 150.0)
        base = ramp(np.clip((YY - pr) / 600.0, 0, 1), [(0, col_top), (1, col_bot)])
        base = base * (1 + 0.55 * lit[..., None])
        over(img, mk, "#000000", 1.0)
        img += base * mk[..., None]
        rip = (np.sin((YY - pr) * 0.21 + XX * 0.02) * 0.5 + 0.5) * np.exp(-np.clip(YY - pr, 0, None) / 180.0)
        img += (rip * mk * 0.05)[..., None] * C("#9AA8FF")
    return img, 1.2, 0.2


def g_geo(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "volcano":
        img = vgrad([(0, "#0A0210"), (0.5, "#3A0A24"), (0.75, "#9A2A1A"), (1, "#FF7A2A")])
        cumulus(img, seed, 520, 300, "#7A2A22", "#2A0A1A", "#0A0210", thr=0.5, cx=1.6, k=0.9, light=(0, 30))
        cx, top = 720, 1180
        cone = poly_mask([[(-300, 2560), (cx - 120, top + 40), (cx - 50, top), (cx + 50, top), (cx + 120, top + 40), (1740, 2560)]])
        n = fbm(seed + 1, 8, 4)
        over(img, cone, "#0C0508", 1.0)
        img += cone[..., None] * ramp(n, [(0, "#000000"), (1, "#2A1010")]) * 0.6
        lava = []
        for i in range(5):
            x0 = cx + r.uniform(-30, 30)
            pts = [(x0, top + 20)]
            for q in range(1, 22):
                y = top + 20 + q * 55
                spread = (y - top) * 0.45
                x = x0 + (i - 2) * (y - top) * 0.22 + 26 * math.sin(q * 0.9 + i)
                pts.append((x, y))
            lava.append(pts)
        lm = line_mask(lava, 7, 2)
        neon(img, lm * cone, "#FF6A1A", core=0.55, glows=((3, 1.0), (14, 0.8), (50, 0.5)), k=1.0)
        glow(img, cx, top, 120, C("#FFAA40"), 0.9, "exp")
        glow(img, cx, top, 420, C("#FF4A10"), 0.55, "exp")
        for _ in range(160):
            x = cx + r.standard_normal() * 220
            y = top - r.random() ** 1.5 * 900
            circ(img, x, y, 2 + r.random() * 3, "#FFC060", 0.9)
        over(img, poly_mask([[(0, 2240), (W, 2240), (W, H), (0, H)]]), "#07030A", 1.0)
        return img, 1.4, 0.3
    if mode == "glacier":
        d = np.hypot((XX - 720) / 560.0, (YY - 1250) / 760.0)
        n = fbm(seed, 4, 6)
        band = 0.5 + 0.5 * np.sin(d * 14 + n * 5)
        img = ramp(np.clip(1 - d * 0.75, 0, 1) * 0.8 + band * 0.2 * np.clip(d, 0, 1), [(0, "#030A22"), (0.4, "#0E3E8A"), (0.7, "#3A9AE0"), (1, "#EAFBFF")])
        fac, f2, cid = cellular(150, seed)
        img *= (0.75 + 0.45 * cid)[..., None] * (0.7 + 0.5 * smooth(0.0, 0.12, f2 - fac))[..., None]
        hole = smooth(0.0, 0.5, 1 - d)
        glow(img, 720, 1250, 380, C("#FFFFFF"), 0.9, "exp")
        for k_ in range(18):
            x = 120 + k_ * 70 + r.uniform(-20, 20)
            L = 120 + r.random() * 380
            over(img, poly_mask([[(x - 24, 0), (x + 24, 0), (x, L)]]), "#9FD8FF", 0.85)
            over(img, poly_mask([[(x - 24, 0), (x - 4, 0), (x - 2, L * 0.9)]]), "#E8FAFF", 0.5)
        fl = poly_mask([[(0, 1750), (W, 1750), (W, H), (0, H)]])
        refl = np.flipud(img)
        refl = np.roll(refl, int(2 * 1750 - H), 0)
        base = ramp(np.clip((YY - 1750) / 810, 0, 1), [(0, "#7ACAF0"), (1, "#04122E")])
        wob = 0.5 + 0.5 * np.sin(YY / 9.0 + fbm(seed + 3, 6, 3) * 9)
        img = img * (1 - fl[..., None]) + (base * 0.6 + blur(refl, 6) * 0.45 * (0.7 + 0.3 * wob[..., None])) * fl[..., None]
        return img, 1.4, 0.25
    xr = 720 + 300 * np.sin(YY / 2560 * 2 * math.pi * 1.5 + 1.0) + 90 * np.sin(YY / 300.0)
    wr = 70 + 34 * np.sin(YY / 410.0)
    rvm = 1 - smooth(wr - 2, wr + 2, np.abs(XX - xr))
    n1 = fbm(seed, 4, 5)
    bands = 0.5 + 0.5 * np.sin(n1 * 26 + YY * 0.006)
    img = ramp(bands * 0.55 + fbm(seed + 2, 12, 4) * 0.45, [(0, "#6A2A1A"), (0.5, "#B8502A"), (1, "#EDA070")])
    prox = blur(rvm, 70)
    img *= (1 - 0.55 * np.clip(prox * 1.8, 0, 1))[..., None]
    rim = blur(rvm, 6) - rvm
    img += np.clip(np.roll(rvm, 14, 1) - rvm, 0, 1)[..., None] * C("#FFD9A0") * 0.4
    water = ramp(np.clip(0.5 + 0.5 * np.sin(YY / 11.0 + fbm(seed + 5, 8, 3) * 7), 0, 1), [(0, "#0A6A6A"), (0.6, "#2FC0B0"), (1, "#C8FFF0")])
    img = img * (1 - rvm[..., None]) + water * rvm[..., None]
    return img, 1.4, 0.2


def g_sea(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "reef":
        img = vgrad([(0, "#2AA8D8"), (0.4, "#0F6AA8"), (1, "#052A5A")])
        rays(img, 0.25, np.linspace(-900, 900, 7), [60, 90, 50, 110, 70, 45, 80], "#BFF0FF", 0.22, np.clip(1 - YY / 1800, 0, 1))
        for c_, base_c in (("#FF5A8A", 0), ("#FF9A3A", 1), ("#B05AFF", 2)):
            def f(d, ss, base_c=base_c):
                def br(x, y, a, L, w, dep):
                    if dep == 0 or L < 8:
                        return
                    x2, y2 = x + L * math.cos(a), y + L * math.sin(a)
                    d.line([(x * ss, y * ss), (x2 * ss, y2 * ss)], fill=255, width=max(2, int(w * ss)))
                    d.ellipse([(x2 - w * 0.7) * ss, (y2 - w * 0.7) * ss, (x2 + w * 0.7) * ss, (y2 + w * 0.7) * ss], fill=255)
                    br(x2, y2, a - 0.42 + r.uniform(-0.1, 0.1), L * 0.78, w * 0.8, dep - 1)
                    br(x2, y2, a + 0.42 + r.uniform(-0.1, 0.1), L * 0.78, w * 0.8, dep - 1)
                for q in range(5):
                    br(120 + base_c * 480 + q * 90 + r.uniform(-40, 40), 2560, -math.pi / 2 + r.uniform(-0.4, 0.4), 150 + r.random() * 120, 26, 6)
            over(img, draw_mask(f, 2), c_, 0.95)
        hills_layer(img, seed + 3, 2400, 80, "#2A5A7A", "#0A2A4A", freq=2.0, fade=200)
        for sch in range(2):
            cx, cy = 450 + sch * 500, 900 + sch * 400
            for _ in range(90):
                a = r.random() * 6.28
                dd = r.random() ** 0.7 * 260
                x, y = cx + dd * math.cos(a) * 1.1, cy + dd * math.sin(a) * 0.7
                s = 14 + r.random() * 6
                dirn = 1 if sch == 0 else -1
                over(img, poly_mask([[(x, y), (x - dirn * s, y - s * 0.4), (x - dirn * s * 1.6, y - s * 0.7), (x - dirn * s * 1.6, y + s * 0.7), (x - dirn * s, y + s * 0.4)]], 1), "#FFD25A" if sch == 0 else "#FFFFFF", 0.95)
        bokeh(img, seed, 40, ["#BFF0FF", "#FFFFFF"], 6, 26, (0.1, 0.4), 0.8)
        return img, 1.3, 0.15
    if mode == "kelp":
        img = vgrad([(0, "#7AD8B0"), (0.3, "#2A9A78"), (1, "#05301E")])
        rays(img, 0.12, np.linspace(-800, 800, 8), [50, 80, 40, 100, 60, 50, 90, 45], "#E8FFD0", 0.28, np.clip(1 - YY / 2000, 0, 1))
        for li, (k_, col_, n) in enumerate(((0.45, "#1E7A4A", 8), (0.7, "#14603A", 7), (1.0, "#0A3C26", 6))):
            polys = []
            for q in range(n):
                x0 = r.random() * W
                amp = 40 * k_ + 10
                ph = r.random() * 6
                pts_l, pts_r = [], []
                for y in range(2600, -100, -20):
                    x = x0 + amp * math.sin(y / 260.0 + ph) * (1.2 - y / 2600)
                    wd = (34 + 20 * math.sin(y / 70 + q)) * k_ * (0.4 + 0.6 * y / 2600)
                    pts_l.append((x - wd, y))
                    pts_r.append((x + wd, y))
                polys.append(pts_l + pts_r[::-1])
            over(img, poly_mask(polys, 2), col_, 0.95 if li else 0.8)
            fog(img, 1300, 800, "#2A9A78", 0.14)
        bokeh(img, seed, 80, ["#E8FFD0", "#BFFFE0"], 3, 14, (0.15, 0.5), 0.3)
        return img, 1.3, 0.2
    img = vgrad([(0, "#02081C"), (1, "#010410")])
    for (cx, cy, R, c1) in ((560, 780, 230, "#FF5FD8"), (950, 1500, 170, "#3DE0FF"), (380, 1800, 120, "#9A6BFF"), (1050, 520, 100, "#6BFF9A")):
        def f(d, ss, cx=cx, cy=cy, R=R):
            d.pieslice([(cx - R) * ss, (cy - R * 0.8) * ss, (cx + R) * ss, (cy + R * 0.8) * ss], 180, 360, fill=255)
            d.rectangle([(cx - R) * ss, cy * ss, (cx + R) * ss, (cy + R * 0.08) * ss], fill=255)
        dome = draw_mask(f, 2)
        tent = []
        for q in range(9):
            x0 = cx - R + (2 * R) * q / 8
            pts = [(x0 + 18 * math.sin(y / 60 + q) * (y / 400), cy + 12 + y) for y in range(0, int(R * 3.2), 14)]
            tent.append(pts)
        tm = line_mask(tent, 3, 2)
        neon(img, dome * 0.6, c1, core=0.2, glows=((8, 0.8), (30, 0.7), (90, 0.4)), k=0.9)
        img += (blur(tm, 2) * 0.8)[..., None] * C(c1) * np.exp(-np.clip(YY - cy, 0, None) / (R * 2.6))[..., None]
        ring_ = np.exp(-((np.hypot((XX - cx) / R, (YY - cy) / (R * 0.8)) - 0.85) / 0.06) ** 2) * (YY < cy)
        img += ring_[..., None] * C(c1) * 0.5
    for _ in range(300):
        x, y = r.random() * W, r.random() * H
        glow(img, x, y, 3 + r.random() * 6, C(["#3DE0FF", "#FF5FD8", "#FFFFFF"][int(r.integers(0, 3))]), 0.5, "exp", 3)
    return img, 1.3, 0.3


def leaf(d, ss, x, y, ang, L, w):
    pts = []
    for t in np.linspace(0, 1, 18):
        wd = w * math.sin(t * math.pi) ** 0.8
        pts.append((x + L * t * math.cos(ang) - wd * math.sin(ang), y + L * t * math.sin(ang) + wd * math.cos(ang)))
    for t in np.linspace(1, 0, 18):
        wd = w * math.sin(t * math.pi) ** 0.8
        pts.append((x + L * t * math.cos(ang) + wd * math.sin(ang), y + L * t * math.sin(ang) - wd * math.cos(ang)))
    d.polygon([(a * ss, b * ss) for a, b in pts], fill=255)


def g_jungle(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "canopy":
        img = vgrad([(0, "#E8FFB0"), (0.5, "#6ACC5A"), (1, "#0E5A2E")])
        glow(img, 960, 260, 700, (1.0, 1.0, 0.7), 0.7, "exp")
        for li, (col_, n, sz) in enumerate((("#2A8A3E", 14, 520), ("#1C7030", 14, 460), ("#0F5224", 14, 520), ("#072F18", 12, 640))):
            def f(d, ss, n=n, sz=sz):
                for _ in range(n):
                    x = r.random() * W
                    y = r.random() * H * 0.9 + (li * 120)
                    leaf(d, ss, x, y, r.uniform(0, 6.28), sz * (0.7 + 0.6 * r.random()), sz * 0.16)
            m = draw_mask(f, 2)
            over(img, m, col_, 0.97)
            rib = m * (np.abs(np.sin((XX * 0.7 + YY) * 0.04)) > 0.985)
            over(img, rib, "#9AE890", 0.18)
        rays(img, 0.45, np.linspace(-700, 900, 6), [40, 70, 35, 60, 45, 50], "#FFFFC0", 0.2, np.clip(1 - YY / 1900, 0, 1))
        fog(img, 1700, 400, "#6AD890", 0.18, seed=seed + 3)
        return img, 1.3, 0.3
    if mode == "savanna":
        img = vgrad([(0, "#2A1A5E"), (0.25, "#9A3A7A"), (0.45, "#F2705A"), (0.6, "#FFB04A"), (0.7, "#FFE08A"), (1, "#FFE08A")])
        circ(img, 720, 1500, 300, "#FFF0B8")
        glow(img, 720, 1500, 800, (1.0, 0.6, 0.2), 0.5, "exp")
        cumulus(img, seed, 1000, 220, "#FFB070", "#B8507A", "#4A2A72", thr=0.55, cx=2.4, light=(-20, 18), k=0.8)
        over(img, poly_mask([[(0, 1700), (W, 1700), (W, H), (0, H)]]), "#14081E", 1.0)
        g = poly_mask([[(0, 1700), (W, 1700), (W, H), (0, H)]])
        img = img * (1 - g[..., None]) + ramp(np.clip((YY - 1700) / 860, 0, 1), [(0, "#7A3A2A"), (1, "#14081E")]) * g[..., None]
        for (x, base, h) in ((260, 1800, 520), (1060, 1760, 380), (700, 1730, 220), (1330, 1900, 560)):
            def acacia(d, ss, x=x, base=base, h=h):
                d.line([(x * ss, base * ss), ((x + 10) * ss, (base - h * 0.5) * ss), ((x - 30) * ss, (base - h * 0.78) * ss)], fill=255, width=int(h * 0.05 * ss), joint="curve")
                d.line([((x + 10) * ss, (base - h * 0.5) * ss), ((x + 100) * ss, (base - h * 0.74) * ss)], fill=255, width=int(h * 0.03 * ss))
                d.ellipse([(x - h * 0.65) * ss, (base - h * 1.0) * ss, (x + h * 0.65) * ss, (base - h * 0.72) * ss], fill=255)
                d.ellipse([(x - h * 0.5) * ss, (base - h * 0.92) * ss, (x + h * 0.5) * ss, (base - h * 0.8) * ss], fill=0)
                d.ellipse([(x - h * 0.65) * ss, (base - h * 0.96) * ss, (x + h * 0.65) * ss, (base - h * 0.78) * ss], fill=255)
            over(img, draw_mask(acacia, 2), "#0A0414", 1.0)
        def gr(d, ss):
            for _ in range(700):
                x = r.random() * W
                y = 1950 + r.random() * 610
                L = 30 + (y - 1950) * 0.12
                d.line([(x * ss, y * ss), ((x + r.uniform(-14, 14)) * ss, (y - L) * ss)], fill=255, width=int(3 * ss))
        over(img, draw_mask(gr, 2), "#0A0414", 0.95)
        for k_ in range(14):
            x, y = 250 + k_ * 60 + r.uniform(-30, 30), 700 + 80 * math.sin(k_ * 0.8) + r.uniform(-50, 50)
            over(img, line_mask([[(x - 16, y - 8), (x, y), (x + 16, y - 8)]], 4, 2), "#1A0A24", 0.9)
        return img, 1.3, 0.3
    img = vgrad([(0, "#DDF5C8"), (0.5, "#9AD48A"), (1, "#3A7A4A")])
    rays(img, 0.1, np.linspace(-700, 700, 5), [70, 110, 60, 90, 70], "#FFFFE0", 0.3, None)
    for li, (k_, col_, n) in enumerate(((0.5, "#8AC27A", 14), (0.75, "#5AA060", 11), (1.0, "#2E7A44", 9))):
        sts = []
        nds = []
        for _ in range(n):
            x = r.random() * W
            wd = 34 * k_
            sts.append([(x - wd, -10), (x + wd, -10), (x + wd, H + 10), (x - wd, H + 10)])
            y = r.random() * 160
            while y < H:
                nds.append([(x - wd * 1.2, y), (x + wd * 1.2, y), (x + wd * 1.2, y + 12 * k_ + 4), (x - wd * 1.2, y + 12 * k_ + 4)])
                y += 300 * k_ + r.random() * 120
        over(img, poly_mask(sts, 1), col_, 0.95)
        over(img, poly_mask(nds, 1), "#1E5A30", 0.7)
        hi = [[(p[0][0] + (p[1][0] - p[0][0]) * 0.15, p[0][1]), (p[0][0] + (p[1][0] - p[0][0]) * 0.3, p[0][1]), (p[0][0] + (p[1][0] - p[0][0]) * 0.3, p[2][1]), (p[0][0] + (p[1][0] - p[0][0]) * 0.15, p[2][1])] for p in sts]
        over(img, poly_mask(hi, 1), "#F2FFD0", 0.25)
        fog(img, 1200 + li * 200, 500, "#E8FFE0", 0.28, seed=seed + li)
    return img, 1.3, 0.1


reg("night_desert", g_night_desert)
reg("geo", g_geo)
reg("sea", g_sea)
reg("jungle", g_jungle)

W2("desert-saguaro-night", "Saguaro Night", "Cactus silhouettes and dark dunes under the Milky Way and a field of stars.", "landscapes", False, False, 0.1, "night_desert", mode="saguaro")
W2("desert-rock-arch", "Rock Arch", "A tall stone arch frames a violet dusk sky and the first stars.", "landscapes", True, False, 0.1, "night_desert", mode="arch")
W2("desert-moon-dunes", "Moonlit Dunes", "Rippled blue dunes glow under a bright moon and a thin sky of stars.", "landscapes", False, False, 0.1, "night_desert", mode="dunes")
W2("volcano-night-glow", "Volcano Glow", "A black volcano with glowing lava streams, sparks and a dark red smoke sky.", "landscapes", True, False, 0.1, "geo", mode="volcano")
W2("glacier-ice-cave", "Glacier Cave", "A blue ice cave with icicles opens to a bright white world and a glassy pool.", "landscapes", False, True, 0.1, "geo", mode="glacier")
W2("canyon-river-view", "Canyon River", "A turquoise river winds through layered red canyon rock, seen from above.", "landscapes", False, False, 0.1, "geo", mode="river")
W2("sea-coral-reef", "Coral Reef", "Sun rays reach a reef of pink and orange coral, with schools of tiny fish.", "landscapes", False, False, 0.1, "sea", mode="reef")
W2("sea-kelp-forest", "Kelp Forest", "Swaying kelp in green layers and golden light beams in the shallow sea.", "landscapes", False, False, 0.1, "sea", mode="kelp")
W2("sea-glow-jellies", "Glow Jellies", "Glowing jellyfish drift through the black deep sea with tiny lights.", "landscapes", True, True, 0.1, "sea", mode="abyss")
W2("jungle-canopy", "Jungle Canopy", "Layers of large leaves and bright light beams in a green tropical canopy.", "landscapes", True, False, 0.1, "jungle", mode="canopy")
W2("savanna-sunset", "Savanna Sunset", "Flat-topped acacia trees and tall grass in front of a huge golden sun.", "landscapes", False, False, 0.1, "jungle", mode="savanna")
W2("bamboo-light", "Bamboo Light", "Misty bamboo stalks in pale green layers with soft sunbeams.", "landscapes", False, False, 0.1, "jungle", mode="bamboo")


# ====================================================================== WAVE 2: art movements and patterns
def paper(seed, base="#F2EBDD", k=0.06):
    n = fbm(seed + 77, 160, 2)
    f = fbm(seed + 78, 6, 3)
    return np.broadcast_to(C(base), (H, W, 3)).copy() * (1 - k + k * 2 * n[..., None] * 0.5 + (f[..., None] - 0.5) * 0.05)


def g_modern(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "swiss":
        img = paper(seed)
        ink = "#17171A"
        over(img, poly_mask([[(120, 0), (1320, 0), (1320, 2560), (120, 2560)]]) * 0, "#000000", 0.0)
        circ(img, 560, 780, 330, "#E2231A")
        over(img, poly_mask([[(120, 1260), (1320, 1260), (1320, 1300), (120, 1300)]]), ink, 1.0)
        for j in range(6):
            for i in range(8):
                circ(img, 170 + i * 150, 1480 + j * 150, 22 - j * 2.5 + (6 if i == 3 else 0), ink, 1.0)
        over(img, poly_mask([[(120, 1260), (500, 1260), (500, 1300), (120, 1300)]]), "#E2231A", 1.0)
        for k_ in range(9):
            over(img, poly_mask([[(120, 190 + k_ * 36), (120 + 160 + (k_ * 97) % 380, 190 + k_ * 36), (120 + 160 + (k_ * 97) % 380, 200 + k_ * 36), (120, 200 + k_ * 36)]]), ink, 0.9)
        def hc(d, ss):
            d.pieslice([780 * ss, 330 * ss, 1320 * ss, 870 * ss], 90, 270, fill=255)
        over(img, draw_mask(hc, 2), ink, 1.0)
        over(img, poly_mask([[(1180, 1480), (1320, 1480), (1320, 2380), (1180, 2380)]]), "#DADAD2", 1.0)
        return img, 1.2, 0.0
    if mode == "destijl":
        img = paper(seed, "#EEE8D6", 0.04)
        xs = [0, 330, 980, W]
        ys = [0, 520, 1210, 1830, H]
        fills = {(0, 0): "#C9241C", (2, 1): "#1F4FA3", (1, 3): "#F2C21A", (2, 3): "#1F4FA3", (0, 2): "#F2C21A", (1, 0): "#F2C21A"}
        for (i, j), c_ in fills.items():
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[j], ys[j + 1]
            over(img, poly_mask([[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]), c_, 1.0)
        for x in xs[1:-1]:
            over(img, poly_mask([[(x - 14, 0), (x + 14, 0), (x + 14, H), (x - 14, H)]]), "#141414", 1.0)
        for y in ys[1:-1]:
            over(img, poly_mask([[(0, y - 14), (W, y - 14), (W, y + 14), (0, y + 14)]]), "#141414", 1.0)
        over(img, poly_mask([[(330, 1210 - 14), (330 + 650, 1210 - 14), (330 + 650, 1210 + 14), (330, 1210 + 14)]]), "#141414", 1.0)
        return img, 1.4, 0.0
    f1, f2, cid = cellular(260, seed)
    pal = ["#C79A52", "#8A5A32", "#4A5560", "#9AA6A8", "#E8D8B0", "#3A6A78", "#B05A3A"]
    idx = np.floor(cid * 7).astype(np.int32) % 7
    pa = np.array([C(p) for p in pal], np.float32)
    img = pa[idx]
    sh = 0.78 + 0.35 * ((XX * 0.0007 + YY * 0.0004 + cid * 5) % 1.0)
    img *= sh[..., None]
    hatch = (((XX + YY * (0.6 + cid)) % 22) < 5) * (cid > 0.55)
    img *= (1 - 0.22 * hatch)[..., None]
    line = 1 - smooth(0.02, 0.07, f2 - f1)
    img = img * (1 - line[..., None]) + C("#14100C") * line[..., None]
    img *= (0.94 + 0.12 * fbm(seed + 3, 120, 2))[..., None]
    return img, 1.6, 0.1


def g_islamic(seed, mode):
    if mode == "star8":
        T = 270.0
        u = (XX % T) / T - 0.5
        v = (YY % T) / T - 0.5
        dA = np.maximum(np.abs(u), np.abs(v))
        dB = np.abs(u) + np.abs(v)
        star = ((dA < 0.32) | (dB < 0.452)).astype(np.float32)
        inner = ((dA < 0.32) & (dB < 0.452)).astype(np.float32)
        img = np.broadcast_to(C("#0B3A4A"), (H, W, 3)).copy()
        img = img * (1 - star[..., None]) + C("#14677A") * star[..., None]
        img = img * (1 - inner[..., None]) + C("#E6D6A8") * inner[..., None]
        dc = np.hypot(u, v)
        img = img * (1 - (dc < 0.1)[..., None] * 1.0) + C("#0B3A4A") * (dc < 0.1)[..., None]
        img = img * (1 - (dc < 0.055)[..., None]) + C("#E0B04A") * (dc < 0.055)[..., None]
        w = 0.0075
        L = np.exp(-(((dA - 0.32) / w) ** 2)) * (dB < 0.5) + np.exp(-(((dB - 0.452) / w) ** 2)) * (dA < 0.34)
        L += np.exp(-(((dc - 0.1) / w) ** 2))
        L = np.clip(L, 0, 1)
        img = img * (1 - L[..., None]) + C("#E0B04A") * L[..., None]
        return img, 1.3, 0.1
    if mode == "hexagram":
        img = np.broadcast_to(C("#3A0E2A"), (H, W, 3)).copy()
        R = 150.0
        dx, dy = R * math.sqrt(3), R * 1.5
        cells = []
        for j in range(-1, int(H / dy) + 2):
            for i in range(-1, int(W / dx) + 2):
                cells.append((i * dx + (dx / 2 if j % 2 else 0), j * dy))

        def f(d, ss):
            for (cx, cy) in cells:
                t1 = [(cx + R * math.cos(math.radians(90 + 120 * k)), cy - R * math.sin(math.radians(90 + 120 * k))) for k in range(3)]
                t2 = [(cx + R * math.cos(math.radians(270 + 120 * k)), cy - R * math.sin(math.radians(270 + 120 * k))) for k in range(3)]
                for t in (t1, t2):
                    d.polygon([(x * ss, y * ss) for x, y in t], outline=255, width=int(5 * ss))
        m = draw_mask(f, 2)
        fill = np.zeros((H, W), np.float32)
        def f2(d, ss):
            for (cx, cy) in cells:
                hexp = [(cx + R * 0.58 * math.cos(math.radians(30 + 60 * k)), cy + R * 0.58 * math.sin(math.radians(30 + 60 * k))) for k in range(6)]
                d.polygon([(x * ss, y * ss) for x, y in hexp], fill=255)
        hm = draw_mask(f2, 2)
        over(img, hm, "#7A1E4A", 1.0)
        over(img, m, "#F0C860", 0.95)
        img += blur(m, 5)[..., None] * C("#FFB040") * 0.4
        return img, 1.3, 0.2
    img = vgrad([(0, "#0A1A4A"), (1, "#14307A")])
    R = 170.0
    dx, dy = R, R * math.sqrt(3) / 2
    centers = []
    for j in range(-2, int(H / dy) + 3):
        for i in range(-2, int(W / dx) + 3):
            centers.append((i * dx + (dx / 2 if j % 2 else 0), j * dy))
    for (cx, cy) in centers:
        if (cx + cy) % 2 < 1:
            circ(img, cx, cy, R * 0.98, "#18489A", 0.35)
    for (cx, cy) in centers:
        ring(img, cx, cy, R, 4.5, "#F6DC9A", 0.9)
    for (cx, cy) in centers:
        circ(img, cx, cy, 8, "#F6DC9A", 0.9)
    img += blur(np.clip(img - 0.55, 0, 1), 8) * 0.5
    return img, 1.3, 0.2


def g_jp(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "asanoha":
        img = np.broadcast_to(C("#14295A"), (H, W, 3)).copy()
        R = 130.0
        dx, dy = R * math.sqrt(3), R * 1.5
        segs = []
        tris = []
        for j in range(-1, int(H / dy) + 2):
            for i in range(-1, int(W / dx) + 2):
                cx, cy = i * dx + (dx / 2 if j % 2 else 0), j * dy
                V = [(cx + R * math.cos(math.radians(90 + 60 * k)), cy - R * math.sin(math.radians(90 + 60 * k))) for k in range(6)]
                for k in range(6):
                    a, b = V[k], V[(k + 1) % 6]
                    g = ((cx + a[0] + b[0]) / 3, (cy + a[1] + b[1]) / 3)
                    segs += [[g, (cx, cy)], [g, a], [g, b], [a, b]]
                    if (i + j + k) % 2 == 0:
                        tris.append([(cx, cy), a, b])
        over(img, poly_mask(tris, 2), "#1E3B7E", 1.0)
        over(img, line_mask(segs, 2.6, 2), "#F2EEE0", 0.92)
        return img, 1.2, 0.1
    if mode == "kikko":
        img = np.broadcast_to(C("#0C0C10"), (H, W, 3)).copy()
        R = 112.0
        dx, dy = R * math.sqrt(3), R * 1.5
        outer, inner, fills = [], [], []
        for j in range(-1, int(H / dy) + 2):
            for i in range(-1, int(W / dx) + 2):
                cx, cy = i * dx + (dx / 2 if j % 2 else 0), j * dy
                for rr, lst in ((R * 0.97, outer), (R * 0.72, inner)):
                    lst.append([(cx + rr * math.cos(math.radians(90 + 60 * k)), cy - rr * math.sin(math.radians(90 + 60 * k))) for k in range(6)])
        fm = poly_mask(outer, 2)
        shade = fbm(seed, 6, 3)
        img = img * (1 - fm[..., None]) + ramp(shade, [(0, "#1A2A4A"), (1, "#3A1A2A")]) * fm[..., None]
        over(img, line_mask(outer, 4.5, 2, True), "#D8B050", 0.95)
        over(img, line_mask(inner, 2.5, 2, True), "#D8B050", 0.7)
        return img, 1.2, 0.2
    img = np.broadcast_to(C("#F1E8D4"), (H, W, 3)).copy()
    cols_ = ["#B3261E", "#14295A", "#F1E8D4", "#2E5A4A"]
    polys = [[] for _ in cols_]
    for col_i in range(13):
        x0 = col_i * 120 - 40
        for row in range(-1, 26):
            y = row * 120 + (60 if col_i % 2 else 0)
            polys[(row + col_i * 2) % 4 if False else (col_i + row) % 4].append([(x0, y + 120), (x0 + 60, y), (x0 + 120, y + 120), (x0 + 120, y + 180), (x0 + 60, y + 60), (x0, y + 180)])
    for p, c_ in zip(polys, cols_):
        over(img, poly_mask(p, 2), c_, 1.0)
    over(img, line_mask([q for p in polys for q in p], 2.0, 2, True), "#2A1A10", 0.5)
    img *= (0.92 + 0.12 * fbm(seed, 200, 2))[..., None]
    return img, 1.4, 0.1


def g_steps(seed, mode):
    cell = 36
    cx_ = (XX // cell).astype(np.int32)
    cy_ = (YY // cell).astype(np.int32)
    if mode == "aztec":
        pal = np.array([C(c) for c in ("#14100E", "#C2502A", "#F0E0BC", "#1F9A92")], np.float32)
        T = 14
        i = (cx_ % T) - T // 2
        j = (cy_ % T) - T // 2
        d = np.abs(i) + np.abs(j)
        k = (d + (cx_ // T + cy_ // T) % 2 * 2) % 4
        img = pa = pal[k]
        edge = ((cx_ % T) == 0) | ((cy_ % T) == 0)
        img = np.where(edge[..., None], pal[0], img)
        sh = ((XX % cell) < 3) | ((YY % cell) < 3)
        img = img * (1 - 0.10 * sh[..., None])
        return img.astype(np.float32), 1.6, 0.1
    if mode == "andean":
        pal = np.array([C(c) for c in ("#120A0A", "#B3261E", "#F0E0BC", "#E0A02A", "#2A5A7A")], np.float32)
        band = cy_ // 9
        rr = cy_ % 9
        tri = np.abs((cx_ % 12) - 6)
        k = (rr + tri * np.where(band % 2 == 0, 1, -1) + band) % 5
        img = pal[k]
        edge = (rr == 0) | (rr == 8)
        img = np.where(edge[..., None], pal[0], img)
        return (img * (1 - 0.1 * (((YY % cell) < 3)[..., None]))).astype(np.float32), 1.6, 0.1
    pal = np.array([C(c) for c in ("#2A1A12", "#C2502A", "#F0E0BC", "#6A8A5A")], np.float32)
    T = 12
    i = np.abs((cx_ % T) - T // 2 + 0.5)
    j = cy_ % 8
    inside = j >= (np.floor(i) * 8 / (T // 2 + 1))
    base = np.where(((cx_ // T) + (cy_ // 8)) % 2 == 0, 1, 2)
    kk = np.where(inside, base, 0)
    kk = np.where((cy_ % 8) == 7, 3, kk)
    img = pal[kk]
    return (img * (1 - 0.1 * (((XX % cell) < 3)[..., None]))).astype(np.float32), 1.6, 0.1


def g_craft(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "batik":
        T = 250.0
        out = np.broadcast_to(C("#34200F"), (H, W, 3)).copy()
        for oy in (0, T / 2):
            ox = 0 if oy == 0 else T / 2
            u = ((XX + ox) % T) / T - 0.5
            v = ((YY + oy) % T) / T - 0.5
            e1 = np.sqrt(((np.abs(u) - 0.25) / 0.25) ** 2 + (v / 0.1) ** 2)
            e2 = np.sqrt((u / 0.1) ** 2 + ((np.abs(v) - 0.25) / 0.25) ** 2)
            e = np.minimum(e1, e2)
            ins = (e < 1.0).astype(np.float32)
            ln = np.exp(-(((e - 1.0) / 0.06) ** 2))
            if oy == 0:
                out = out * (1 - ins[..., None]) + C("#E8D4A0") * ins[..., None]
                out = out * (1 - 0.7 * ln[..., None]) + C("#14080A") * 0.7 * ln[..., None]
                dc = np.hypot(u, v)
                out = np.where((dc < 0.05)[..., None], C("#34200F"), out)
            else:
                out = out * (1 - ins[..., None] * 0.55) + C("#B8864A") * ins[..., None] * 0.55
        cr = fbm(seed, 14, 5)
        crack = 1 - smooth(0.0, 0.03, np.abs(cr - 0.5))
        out = out * (1 - 0.35 * crack[..., None]) + C("#F4E6C0") * 0.35 * crack[..., None] * 0.6
        out *= (0.9 + 0.2 * fbm(seed + 2, 100, 2))[..., None]
        return out, 1.4, 0.2
    if mode == "shibori":
        T = 300.0
        ox = np.where((YY // T) % 2 == 0, 0.0, T / 2)
        u = ((XX + ox) % T) - T / 2
        v = (YY % T) - T / 2
        wob = fbm(seed, 10, 3) * 40
        d = np.hypot(u, v) + wob
        ring_ = 0.5 + 0.5 * np.cos(d / 8.0)
        fall = 1 - smooth(110, 150, d)
        val = np.clip(ring_ * fall * 0.9 + (1 - fall) * 0.12, 0, 1)
        val = blur(val, 2.0)
        img = ramp(val, [(0, "#0B1A46"), (0.45, "#1E3F8A"), (1, "#EEF3FA")])
        img *= (0.9 + 0.2 * fbm(seed + 2, 140, 3))[..., None]
        return img, 1.5, 0.2
    base = paper(seed, "#C9C4BA", 0.1)
    base *= (0.9 + 0.2 * fbm(seed + 5, 90, 3))[..., None]
    base *= (0.96 + 0.08 * fbm(seed + 6, 4, 3))[..., None]
    cols_ = ["#C2502A", "#5A7A4A", "#F0E4CC", "#2A2A32", "#E8A0A0", "#2E7A82", "#D8A030"]
    img = base
    for ci, c_ in enumerate(cols_):
        polys = []
        for _ in range(34):
            cx, cy = r.random() * W, r.random() * H
            n = int(r.integers(5, 8))
            R = 26 + r.random() ** 1.6 * 100
            ang = np.sort(r.random(n) * 6.28)
            rad = R * (0.55 + 0.6 * r.random(n))
            polys.append([(cx + a * 0 + rr_ * math.cos(a), cy + rr_ * math.sin(a)) for a, rr_ in zip(ang, rad)])
        m = poly_mask(polys, 2)
        shadow_under(img, m, 3, 4, 2, 0.28)
        over(img, m, c_, 1.0)
    return img, 1.6, 0.0


reg("modern", g_modern)
reg("islamic", g_islamic)
reg("jp", g_jp)
reg("steps", g_steps)
reg("craft", g_craft)

W2("modern-swiss-poster", "Swiss Poster", "A red circle, rule lines and dot grids on an off-white page, in the Swiss grid style.", "abstract", False, False, 0.3, "modern", mode="swiss")
W2("modern-destijl", "De Stijl Grid", "Black grid lines and flat red, blue and yellow planes on warm paper.", "abstract", False, False, 0.3, "modern", mode="destijl")
W2("modern-cubist-shards", "Cubist Shards", "Fragmented ochre, brown and blue planes with black edges and hatching, like a cubist still life.", "abstract", True, False, 0.2, "modern", mode="cubist")
W2("islamic-star-tiles", "Octagon Star Tiles", "Cream octagon stars with gold outlines repeat across a deep teal tile.", "patterns", True, True, 0.15, "islamic", mode="star8")
W2("islamic-hexagrams", "Plum Hexagrams", "Golden interlaced six-point stars on deep plum with glowing hexagon centres.", "patterns", False, False, 0.15, "islamic", mode="hexagram")
W2("islamic-circle-lattice", "Circle Lattice", "Overlapping gold circles form a flower lattice on deep blue.", "patterns", False, False, 0.15, "islamic", mode="circles")
W2("japan-asanoha", "Asanoha Hemp Leaf", "A traditional hemp-leaf star lattice in white lines on indigo.", "patterns", False, False, 0.15, "jp", mode="asanoha")
W2("japan-kikko", "Kikko Tortoise Shell", "Gold hexagon shells with inner outlines on dark navy and plum.", "patterns", True, False, 0.15, "jp", mode="kikko")
W2("japan-yabane", "Yabane Arrow Feathers", "Rows of arrow-feather chevrons in red, indigo, cream and green.", "patterns", False, False, 0.25, "jp", mode="yabane")
W2("steps-aztec-diamonds", "Stepped Diamonds", "Concentric stepped diamonds in terracotta, cream, turquoise and black.", "patterns", False, False, 0.2, "steps", mode="aztec")
W2("steps-andean-weave", "Andean Weave", "Stepped zigzag bands in red, gold, cream and blue, like a woven textile.", "patterns", True, False, 0.2, "steps", mode="andean")
W2("steps-terrace-pyramids", "Terrace Pyramids", "Rows of stepped pyramids in clay, cream and olive on dark brown.", "patterns", False, False, 0.2, "steps", mode="pyramids")
W2("craft-batik-kawung", "Batik Kawung", "Cream ellipses on dark brown cloth with fine wax cracks, in the classic kawung design.", "patterns", True, False, 0.2, "craft", mode="batik")
W2("craft-indigo-shibori", "Indigo Shibori", "Resist-dyed ripples in indigo and white, like hand-folded cloth.", "patterns", False, True, 0.2, "craft", mode="shibori")
W2("craft-terrazzo", "Terrazzo", "Colourful stone chips scattered across a pale cement floor.", "patterns", False, False, 0.25, "craft", mode="terrazzo")


# ====================================================================== WAVE 2: cozy, space, textures
def steam(img, x, y, h, seed, color="#FFFFFF", k=0.35, w=40):
    n = fbm(seed, 6, 3, cy=10)
    t = np.clip((y - YY) / h, 0, 1)
    sway = 30 * np.sin(YY / 60.0 + seed)
    m = np.exp(-(((XX - x - sway * t) / (w * (0.6 + t))) ** 2)) * (1 - t) * (YY < y) * smooth(0.3, 0.8, n)
    img += blur(m, 6)[..., None] * C(color) * k


def g_cozy_room(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "lamp":
        img = vgrad([(0, "#0E1A2E"), (1, "#1A2640")])
        # window
        over(img, poly_mask([[(120, 260), (700, 260), (700, 1320), (120, 1320)]]), "#07102A", 1.0)
        stars(img, seed, 60, 270, 1300, size=0.8)
        for x in range(0, 700):
            pass
        circ(img, 520, 520, 70, "#EEF2FF")
        over(img, line_mask([[(410, 260), (410, 1320)], [(120, 790), (700, 790)]], 12, 2), "#2A1E18", 1.0)
        over(img, line_mask([[(120, 260), (700, 260), (700, 1320), (120, 1320)]], 22, 2, True), "#2A1E18", 1.0)
        # wall shelf + frame
        over(img, poly_mask([[(920, 400), (1280, 400), (1280, 700), (920, 700)]]), "#0A121F", 1.0)
        over(img, poly_mask([[(940, 420), (1260, 420), (1260, 680), (940, 680)]]), "#24406A", 1.0)
        over(img, poly_mask([[(940, 600), (1100, 520), (1260, 640), (1260, 680), (940, 680)]]), "#2E5A7A", 1.0)
        # desk
        over(img, poly_mask([[(0, 1760), (W, 1760), (W, H), (0, H)]]), "#3A2418", 1.0)
        img += (YY > 1760)[..., None] * ramp(np.clip((YY - 1760) / 800, 0, 1), [(0, "#6A4228"), (1, "#2A180E")]) * 0.8
        # lamp
        lx, ly = 1020, 1760
        over(img, poly_mask([[(lx - 110, ly), (lx + 110, ly), (lx + 90, ly - 30), (lx - 90, ly - 30)]]), "#1A1A22", 1.0)
        over(img, line_mask([[(lx, ly - 30), (lx - 160, ly - 480), (lx - 380, ly - 560)]], 14, 2), "#1A1A22", 1.0)
        hx, hy = lx - 380, ly - 560
        over(img, poly_mask([[(hx - 100, hy + 50), (hx + 100, hy + 50), (hx + 50, hy - 40), (hx - 50, hy - 40)]]), "#E0A030", 1.0)
        cone = poly_mask([[(hx - 90, hy + 50), (hx + 90, hy + 50), (hx + 560, 1760), (hx - 560, 1760)]])
        gl = ramp(np.clip((YY - hy) / 1200, 0, 1), [(0, "#FFE4A0"), (1, "#C26A1A")])
        img += cone[..., None] * gl * (0.45 * np.clip(1 - (YY - hy) / 1500, 0.2, 1))[..., None]
        glow(img, hx, hy + 60, 260, C("#FFC060"), 0.7, "exp")
        pool = np.exp(-(((XX - hx) / 560.0) ** 2 + ((YY - 1790) / 120.0) ** 2))
        img += pool[..., None] * C("#FFB050") * 0.6
        # books, mug, plant
        for i, (bw, bh, c_) in enumerate(((60, 280, "#8A3A3A"), (50, 250, "#3A6A8A"), (66, 300, "#C8A040"), (46, 230, "#4A7A4A"))):
            x0 = 160 + sum(b[0] for b in ((60,), (50,), (66,), (46,))[:i]) + i * 6
            over(img, poly_mask([[(x0, 1760 - bh), (x0 + bw, 1760 - bh), (x0 + bw, 1760), (x0, 1760)]]), c_, 1.0)
            over(img, poly_mask([[(x0, 1760 - bh), (x0 + bw * 0.25, 1760 - bh), (x0 + bw * 0.25, 1760), (x0, 1760)]]), "#FFFFFF", 0.12)
        over(img, poly_mask([[(520, 1620), (660, 1620), (650, 1760), (530, 1760)]]), "#E8DCC8", 1.0)
        over(img, poly_mask([[(520, 1620), (660, 1620), (660, 1640), (520, 1640)]]), "#4A2A18", 1.0)
        ring(img, 676, 1690, 30, 12, "#E8DCC8", 1.0)
        steam(img, 590, 1620, 260, seed, "#FFEBD0", 0.5, 22)
        for (px_, ang) in ((1250, -1.9), (1290, -1.5), (1230, -1.2), (1300, -2.2)):
            def lf(d, ss, px_=px_, ang=ang):
                leaf(d, ss, px_, 1620, ang, 320, 70)
            over(img, draw_mask(lf, 2), "#2A6A44", 1.0)
        over(img, poly_mask([[(1200, 1620), (1340, 1620), (1320, 1760), (1220, 1760)]]), "#B0603A", 1.0)
        return img, 1.4, 0.3
    if mode == "cafe":
        img = vgrad([(0, "#2A1A12"), (1, "#14100E")])
        out = np.zeros((H, W, 3), np.float32)
        out += vgrad([(0, "#2A3A5A"), (1, "#0E1424")])
        bokeh(out, seed, 70, ["#FFB060", "#FFD890", "#FF7A5A", "#7AC8FF", "#FFE8B0"], 40, 150, (0.25, 0.8), 0.6)
        out = blur(out, 5)
        win = poly_mask([[(60, 160), (1380, 160), (1380, 1500), (60, 1500)]])
        img = img * (1 - win[..., None]) + out * win[..., None]
        lay = Image.new("L", (W, H), 0)
        dr = ImageDraw.Draw(lay)
        for _ in range(240):
            x, y = r.random() * W, 160 + r.random() * 1340
            rr_ = 4 + r.random() * 12
            dr.ellipse([x - rr_, y - rr_ * 1.4, x + rr_, y + rr_ * 1.4], fill=int(110 + 120 * r.random()))
            if r.random() < 0.5:
                dr.line([(x, y + rr_), (x + r.uniform(-3, 3), y + rr_ + 60 + 140 * r.random())], fill=70, width=3)
        dm = np.asarray(lay, np.float32) / 255
        img += blur(dm, 1.5)[..., None] * 0.35 * win[..., None] + np.roll(blur(dm, 4), 4, 0)[..., None] * C("#FFE0B0") * 0.2 * win[..., None]
        over(img, line_mask([[(60, 160), (1380, 160), (1380, 1500), (60, 1500)]], 30, 2, True), "#1A100A", 1.0)
        over(img, line_mask([[(720, 160), (720, 1500)]], 20, 2), "#1A100A", 1.0)
        # table
        over(img, poly_mask([[(0, 1620), (W, 1620), (W, H), (0, H)]]), "#3A2214", 1.0)
        img += (YY > 1620)[..., None] * ramp(np.clip((YY - 1620) / 940, 0, 1), [(0, "#7A4A2A"), (1, "#2A160A")]) * 0.8
        img += (YY > 1620)[..., None] * (np.exp(-((YY - 1700) / 60.0) ** 2) * 0.25)[..., None] * C("#FFB070")
        # cup and saucer
        cx, cy = 720, 2000
        over(img, poly_mask([[(cx - 380, cy + 40), (cx + 380, cy + 40), (cx + 300, cy + 110), (cx - 300, cy + 110)]]), "#E8DCC8", 1.0)
        def cup(d, ss):
            d.pieslice([(cx - 220) * ss, (cy - 280) * ss, (cx + 220) * ss, (cy + 160) * ss], 0, 180, fill=255)
            d.rectangle([(cx - 220) * ss, (cy - 60) * ss, (cx + 220) * ss, (cy - 50) * ss], fill=255)
        over(img, draw_mask(cup, 2), "#F2E8D6", 1.0)
        over(img, poly_mask([[(cx - 220, cy - 110), (cx + 220, cy - 110), (cx + 215, cy - 80), (cx - 215, cy - 80)]]), "#4A2A18", 1.0)
        ring(img, cx + 250, cy - 20, 70, 30, "#F2E8D6", 1.0)
        latte = np.hypot((XX - cx) / 200.0, (YY - (cy - 110)) / 40.0)
        lm = 1 - smooth(0.98, 1.02, latte)
        over(img, lm, "#B07A4A", 1.0)
        for q in range(3):
            ring(img, cx, cy - 110, 1, 1, "#000000", 0.0)
        hrt = np.hypot((XX - cx) / np.maximum(120.0 - 30 * np.abs((YY - (cy - 110)) / 40.0), 8.0), (YY - (cy - 110)) / 30.0)
        over(img, (1 - smooth(0.9, 1.0, hrt)) * lm, "#F2E4CC", 0.9)
        steam(img, cx - 30, cy - 120, 520, seed + 2, "#FFEBD0", 0.45, 50)
        steam(img, cx + 60, cy - 120, 420, seed + 3, "#FFEBD0", 0.35, 40)
        return img, 1.5, 0.3
    img = vgrad([(0, "#1A0E0A"), (1, "#2A1810")])
    glow(img, 720, 1500, 1100, (0.55, 0.28, 0.1), 0.55, "gauss")
    pal = ["#8A3A3A", "#C28A3A", "#3A5A7A", "#4A6A4A", "#6A3A6A", "#B05A3A", "#D8C8A0", "#2A3A4A"]
    for sh in range(6):
        y0 = 220 + sh * 400
        over(img, poly_mask([[(0, y0 + 330), (W, y0 + 330), (W, y0 + 370), (0, y0 + 370)]]), "#4A2A18", 1.0)
        x = 20
        books = []
        while x < W - 40:
            if r.random() < 0.08:
                x += 80 + r.random() * 90
                continue
            bw = 36 + r.random() * 44
            bh = 230 + r.random() * 90
            lean = 0.0
            c_ = pal[int(r.integers(0, len(pal)))]
            over(img, poly_mask([[(x, y0 + 330 - bh), (x + bw, y0 + 330 - bh), (x + bw, y0 + 330), (x, y0 + 330)]]), c_, 1.0)
            over(img, poly_mask([[(x, y0 + 330 - bh), (x + bw * 0.2, y0 + 330 - bh), (x + bw * 0.2, y0 + 330), (x, y0 + 330)]]), "#FFFFFF", 0.1)
            over(img, poly_mask([[(x, y0 + 330 - bh * 0.7), (x + bw, y0 + 330 - bh * 0.7), (x + bw, y0 + 330 - bh * 0.62), (x, y0 + 330 - bh * 0.62)]]), "#E8D8A0", 0.55)
            x += bw + 3
    over(img, poly_mask([[(0, 0), (60, 0), (60, H), (0, H)], [(W - 60, 0), (W, 0), (W, H), (W - 60, H)]]), "#3A2012", 1.0)
    glow(img, 1060, 870, 200, C("#FFC060"), 0.55, "exp")
    over(img, poly_mask([[(990, 800), (1130, 800), (1100, 700), (1020, 700)]]), "#F6C060", 1.0)
    over(img, poly_mask([[(1050, 800), (1070, 800), (1070, 930), (1050, 930)]]), "#2A1810", 1.0)
    img *= 0.88 + 0.12 * np.exp(-(((YY - 1280) / 1500.0) ** 2))[..., None]
    return img, 1.5, 0.35


def g_cozy_corner(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "plant":
        img = vgrad([(0, "#F2E4D0"), (1, "#E0CDB4")])
        over(img, poly_mask([[(0, 0), (520, 0), (520, 2060), (0, 2300)]]), "#D8C4A8", 1.0)
        over(img, poly_mask([[(0, 2150), (W, 1980), (W, H), (0, H)]]), "#B08A62", 1.0)
        win = poly_mask([[(700, 220), (1300, 220), (1300, 1500), (700, 1500)]])
        over(img, win, "#CDEBFF", 1.0)
        skyw = vgrad([(0, "#8AD0F8"), (1, "#EAF8FF")])
        cumulus(skyw, seed, 700, 200, "#FFFFFF", "#F4FAFF", "#CDE4F6", thr=0.55, cx=3.0, k=0.9)
        img = img * (1 - win[..., None]) + skyw * win[..., None]
        over(img, line_mask([[(700, 220), (1300, 220), (1300, 1500), (700, 1500)]], 26, 2, True), "#F8F2E8", 1.0)
        over(img, line_mask([[(1000, 220), (1000, 1500)], [(700, 860), (1300, 860)]], 16, 2), "#F8F2E8", 1.0)
        beam = poly_mask([[(700, 1500), (1300, 1500), (1100, 2560), (-100, 2560)]])
        img += beam[..., None] * C("#FFF2C8") * 0.16
        # pot and plant
        px, py = 360, 2140
        over(img, poly_mask([[(px - 170, py - 240), (px + 170, py - 240), (px + 120, py + 120), (px - 120, py + 120)]]), "#C2603A", 1.0)
        over(img, poly_mask([[(px - 170, py - 240), (px + 170, py - 240), (px + 160, py - 190), (px - 160, py - 190)]]), "#D87A52", 1.0)
        cols_ = ["#2A7A4A", "#3A9A5A", "#1E6A3E"]
        for ci, c_ in enumerate(cols_):
            def pl(d, ss, ci=ci):
                for q in range(8):
                    a = -math.pi / 2 + (q - 3.5) * 0.3 + ci * 0.07
                    leaf(d, ss, px + (q - 3.5) * 8, py - 220, a, 520 + 160 * ((q * 37 + ci * 11) % 5) / 5, 130)
            over(img, draw_mask(pl, 2), c_, 0.98 if ci == 0 else 0.9)
        shadow_under(img, poly_mask([[(px - 170, py + 120), (px + 520, py + 120), (px + 420, py + 160), (px - 120, py + 160)]]), 0, 4, 18, 0.3)
        return img, 1.4, 0.15
    if mode == "cat":
        img = vgrad([(0, "#06081E"), (0.6, "#14204A"), (1, "#3A2A5A")])
        stars(img, seed, 400, 0, 1300, size=0.8, big=2)
        circ(img, 980, 520, 120, "#F4F0DA")
        glow(img, 980, 520, 440, (0.5, 0.6, 1.0), 0.3, "exp")
        polys = []
        x = 0
        while x < W:
            w_ = 60 + r.random() * 120
            h_ = 120 + r.random() * 420
            polys.append([(x, 1520), (x, 1520 - h_), (x + w_, 1520 - h_), (x + w_, 1520)])
            for wy in range(int(1520 - h_ + 20), 1500, 30):
                for wx in range(int(x + 10), int(x + w_ - 10), 24):
                    if r.random() < 0.18:
                        circ(img, wx, wy, 4, "#FFD27A", 0.9)
            x += w_ + 8
        over(img, poly_mask(polys), "#0A0E24", 0.97)
        over(img, poly_mask([[(0, 1520), (W, 1520), (W, H), (0, H)]]), "#0A0E24", 1.0)
        over(img, poly_mask([[(0, 1980), (W, 1980), (W, H), (0, H)]]), "#6A4A30", 1.0)
        over(img, poly_mask([[(0, 1980), (W, 1980), (W, 2010), (0, 2010)]]), "#9A7450", 1.0)
        # curtains
        for sd in (0, 1):
            xs = [0, 220] if sd == 0 else [W - 220, W]
            fold = 0.5 + 0.5 * np.sin(XX / 34.0)
            m = poly_mask([[(xs[0], 0), (xs[1], 0), (xs[1] + (40 if sd == 0 else -40), 1980), (xs[0] - (0 if sd == 0 else 40), 1980)]])
            c = ramp(fold, [(0, "#5A2A4A"), (1, "#8A4A6A")])
            over(img, m, "#000000", 1.0)
            img += c * m[..., None]
        # window frame
        over(img, line_mask([[(0, 0), (W, 0)], [(0, 1520), (W, 1520)], [(720, 0), (720, 1520)]], 24, 2), "#E8DCC8", 1.0)
        # cat
        def cat(d, ss):
            d.ellipse([(430) * ss, (1700) * ss, (1150) * ss, (2030) * ss], fill=255)
            d.ellipse([(330) * ss, (1780) * ss, (580) * ss, (2020) * ss], fill=255)
            d.polygon([(350 * ss, 1810 * ss), (350 * ss, 1690 * ss), (430 * ss, 1760 * ss)], fill=255)
            d.polygon([(500 * ss, 1790 * ss), (540 * ss, 1690 * ss), (560 * ss, 1810 * ss)], fill=255)
            d.line([(1100 * ss, 1960 * ss), (1220 * ss, 1990 * ss), (1180 * ss, 2040 * ss), (800 * ss, 2040 * ss)], fill=255, width=int(56 * ss), joint="curve")
        cm = draw_mask(cat, 2)
        over(img, cm, "#0A0710", 1.0)
        rim = cm * (1 - np.roll(cm, 8, 0))
        over(img, rim, "#6A5A9A", 0.55)
        over(img, line_mask([[(380, 1900), (440, 1920)], [(470, 1895), (520, 1915)]], 5, 2), "#3A3050", 0.9)
        return img, 1.4, 0.3
    img = vgrad([(0, "#1A0C08"), (1, "#0E0605")])
    # brick wall
    f1, f2, cid = cellular(120, seed)
    bw, bh = 150, 60
    row = (YY // bh).astype(np.int32)
    bx = (XX + (row % 2) * bw / 2) % bw
    by = YY % bh
    mortar = (bx < 6) | (by < 6)
    brick = ramp(((row * 7 + ((XX + (row % 2) * bw / 2) // bw) * 13) % 10) / 10.0, [(0, "#6A2A1E"), (1, "#9A4A32")]) * (0.8 + 0.4 * fbm(seed, 80, 2))[..., None]
    wall = np.where(mortar[..., None], C("#2A1A14"), brick)
    img = wall * 0.5
    arch = poly_mask([[(300, 2200), (300, 1100), (420, 900), (720, 800), (1020, 900), (1140, 1100), (1140, 2200)]])
    over(img, arch, "#06020A", 1.0)
    # flames
    for (fx, fh, fw, c1, c2) in ((720, 620, 220, "#FFE070", "#FF8A1A"), (560, 420, 160, "#FFC040", "#E8501A"), (880, 460, 170, "#FFC040", "#E8501A"), (720, 380, 110, "#FFFFFF", "#FFD060")):
        def fl(d, ss, fx=fx, fh=fh, fw=fw):
            pts = []
            for t in np.linspace(0, 1, 30):
                y = 2160 - fh * t
                wd = fw * (1 - t) ** 0.6 * (0.7 + 0.3 * math.sin(t * 9 + fx))
                pts.append((fx - wd + 30 * math.sin(t * 6 + fx) * t, y))
            for t in np.linspace(1, 0, 30):
                y = 2160 - fh * t
                wd = fw * (1 - t) ** 0.6 * (0.7 + 0.3 * math.sin(t * 9 + fx + 1))
                pts.append((fx + wd + 30 * math.sin(t * 6 + fx) * t, y))
            d.polygon([(a * ss, b * ss) for a, b in pts], fill=255)
        m = draw_mask(fl, 2)
        over(img, m, c2, 0.95)
        over(img, m * np.roll(m, 40, 0), c1, 0.9)
    glow(img, 720, 1900, 700, (1.0, 0.45, 0.12), 0.9, "exp")
    glow(img, 720, 2000, 1400, (0.8, 0.3, 0.08), 0.35, "gauss")
    over(img, poly_mask([[(200, 780), (1240, 780), (1240, 840), (200, 840)]]), "#3A2012", 1.0)
    for cx_ in (360, 1080):
        over(img, poly_mask([[(cx_ - 26, 640), (cx_ + 26, 640), (cx_ + 26, 780), (cx_ - 26, 780)]]), "#F2E8D0", 1.0)
        glow(img, cx_, 610, 60, C("#FFC060"), 0.8, "exp")
        circ(img, cx_, 618, 12, "#FFE8A0")
    over(img, poly_mask([[(0, 2200), (W, 2200), (W, H), (0, H)]]), "#2A1608", 1.0)
    img += (YY > 2200)[..., None] * np.exp(-(((XX - 720) / 700.0) ** 2) - ((YY - 2300) / 160.0) ** 2)[..., None] * C("#FF8A30") * 0.6
    over(img, poly_mask([[(260, 2200), (1180, 2200), (1260, 2330), (180, 2330)]]), "#6A2A3A", 0.95)
    return img, 1.5, 0.3


def g_blackhole(seed, mode):
    r = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.float32) + C("#010206")
    stars(img, seed, 1200, 0, H, size=0.8, big=4)
    cx, cy = 720, 1220
    if mode == "lensed":
        Rh = 190
        xp, yp = XX - cx, YY - cy
        rr = np.hypot(xp, yp)
        disk = (np.abs(yp) < 8 + 0.022 * np.abs(xp)) & (np.abs(xp) > Rh * 0.9) & (np.abs(xp) < 760)
        dop = 0.55 + 0.6 * smooth(-500, 500, -xp)
        dcol = ramp(np.clip(np.abs(xp) / 760, 0, 1), [(0, "#FFF4D0"), (0.3, "#FFB040"), (1, "#7A2A10")])
        dm = disk.astype(np.float32)
        dm = blur(dm, 2.0)
        img += dm[..., None] * dcol * dop[..., None] * 1.6
        img += blur(dm, 16)[..., None] * C("#FF8A20") * 0.5 + blur(dm, 60)[..., None] * C("#FF5A10") * 0.3
        for rf, wd, kk in ((1.55, 14, 1.0), (1.75, 24, 0.5)):
            arc = np.exp(-(((rr - Rh * rf) / wd) ** 2)) * ((yp < 40) * 1.0 + (yp >= 40) * 0.0)
            arc2 = np.exp(-(((np.hypot(xp, yp * 0.6 - 0) - Rh * rf * 0.95) / wd) ** 2)) * (yp > 0) * 0.5
            img += (arc + arc2 * 0.4)[..., None] * C("#FFD090") * kk * 0.9 * (0.7 + 0.5 * smooth(-500, 500, -xp))[..., None]
        glow(img, cx, cy, Rh * 1.1, C("#FFB060"), 0.3, "exp")
        circ(img, cx, cy, Rh, "#000000")
        ring(img, cx, cy, Rh * 1.02, 4, "#FFE8C0", 0.5)
        img *= 1 - 0.0
        return img, 1.2, 0.2
    if mode == "faceon":
        xp, yp = XX - cx, YY - cy
        rr = np.hypot(xp, yp)
        th = np.arctan2(yp, xp)
        sp = np.sin(th * 3 + np.log(np.maximum(rr, 1.0)) * 5.5) * 0.5 + 0.5
        n = fbm(seed, 4, 5)
        v = np.clip(1 - (rr - 200) / 620, 0, 1) ** 1.6 * (0.45 + 0.7 * sp) * (0.7 + 0.6 * n) * smooth(190, 240, rr)
        img += v[..., None] * ramp(np.clip(1 - (rr - 200) / 620, 0, 1) * (0.7 + 0.4 * sp), [(0, "#7A1A08"), (0.4, "#FF6A18"), (0.75, "#FFC060"), (1, "#FFFFFF")]) * 1.4
        circ(img, cx, cy, 190, "#000000")
        ring(img, cx, cy, 196, 7, "#FFFFFF", 0.95)
        glow(img, cx, cy, 200, C("#FFE0A0"), 0.5, "exp")
        return img, 1.2, 0.25
    xp, yp = XX - cx, YY - cy
    ang = 0.35
    c_, s_ = math.cos(ang), math.sin(ang)
    u = xp * c_ + yp * s_
    v = -xp * s_ + yp * c_
    n = fbm(seed, 3, 5, cy=9)
    for sgn in (-1, 1):
        t = np.clip(sgn * u, 0, None)
        wd = 10 + 0.12 * t
        beam = np.exp(-((v / wd) ** 2)) * np.exp(-t / 900.0) * (sgn * u > 0) * (0.6 + 0.8 * n)
        img += beam[..., None] * C("#7AB8FF") * 0.9 + blur(beam, 18)[..., None] * C("#3A6AFF") * 0.7
        cone = np.exp(-((v / (40 + 0.35 * t)) ** 2)) * np.exp(-t / 700.0) * (sgn * u > 0)
        img += (cone * 0.35 * n)[..., None] * C("#9A7AFF")
    disk = np.exp(-((v / 18.0) ** 2)) * np.exp(-((u / 360.0) ** 2) * 0.0) * smooth(-420, -200, u) * (1 - smooth(200, 420, u))
    disk = np.exp(-((yp / 16.0) ** 2)) * (np.abs(xp) < 520) * (1 - smooth(300, 520, np.abs(xp)))
    img += disk[..., None] * C("#FFB070") * 1.2 + blur(disk, 20)[..., None] * C("#FF7A30") * 0.8
    glow(img, cx, cy, 90, C("#FFFFFF"), 0.9, "exp")
    glow(img, cx, cy, 300, C("#7AB8FF"), 0.4, "exp")
    circ(img, cx, cy, 36, "#000000")
    return img, 1.2, 0.2


def g_alien_sky(seed, mode):
    r = np.random.default_rng(seed)
    if mode == "binary":
        img = vgrad([(0, "#1A0E4A"), (0.3, "#5A2A8A"), (0.5, "#D0508A"), (0.62, "#FF9A5A"), (0.7, "#FFD08A"), (1, "#FFD08A")])
        glow(img, 520, 1620, 700, (1.0, 0.5, 0.2), 0.55, "exp")
        glow(img, 980, 1500, 400, (0.5, 0.8, 1.0), 0.45, "exp")
        circ(img, 520, 1620, 260, "#FFB058")
        circ(img, 520, 1620, 200, "#FFDC90")
        circ(img, 980, 1500, 70, "#E8F6FF")
        stars(img, seed, 200, 0, 1000, size=0.8)
        for i, (y0, a, f, ph, c1, c2) in enumerate(((1700, 90, 5.0, 0.2, "#7A2A6A", "#3A1A5A"), (1900, 130, 3.4, 1.4, "#4A1E5E", "#1E0E3A"), (2160, 170, 2.4, 3.0, "#220E3A", "#0A0518"))):
            xs = np.linspace(0, 1, 721)
            prof = y0 + a * np.sin(xs * f * 2 + ph) + a * 0.35 * np.sin(xs * f * 5.5)
            m = poly_below(prof, 2)
            over(img, m, "#000000", 1.0)
            img += ramp(np.clip((YY - y0) / 700, 0, 1), [(0, c1), (1, c2)]) * m[..., None]
        for (x, base, h) in ((260, 2300, 360), (1180, 2360, 280)):
            over(img, poly_mask([[(x - 18, base), (x + 18, base), (x + 6, base - h), (x - 6, base - h)]]), "#050210", 1.0)
            circ(img, x, base - h, 38, "#050210")
        return img, 1.2, 0.2
    if mode == "moonrise":
        img = vgrad([(0, "#02030C"), (0.5, "#0A1030"), (1, "#2A2A6A")])
        stars(img, seed, 900, 0, 1800, size=0.8, big=4)
        ringp = [(0, "#8A6A4A"), (0.5, "#E8D0A0"), (1, "#8A6A4A")]
        ring_system(img, 720, 800, 520, 1.3, 2.1, 0.22, -0.28, ringp, False, seed=3, alpha=0.8)
        planet(img, 720, 800, 520, [(0, "#5A3A2A"), (0.5, "#B88A5A"), (1, "#F0D8A8")], light=(-0.6, -0.4, 0.7), seed=seed, bands=7, atmo="#FFD8A0", atmo_k=0.4, cells=4)
        ring_system(img, 720, 800, 520, 1.3, 2.1, 0.22, -0.28, [(0, "#8A6A4A"), (0.5, "#E8D0A0"), (1, "#8A6A4A")], True, seed=3, alpha=0.85)
        for i, (y0, a, f, ph, c1, c2) in enumerate(((1900, 120, 4.0, 0.3, "#2A2A5A", "#12122E"), (2150, 160, 2.8, 1.8, "#14143A", "#06061A"))):
            xs = np.linspace(0, 1, 721)
            prof = y0 + a * np.sin(xs * f * 2 + ph) + a * 0.35 * np.sin(xs * f * 5.5)
            m = poly_below(prof, 2)
            over(img, m, "#000000", 1.0)
            img += ramp(np.clip((YY - y0) / 600, 0, 1), [(0, c1), (1, c2)]) * m[..., None]
        for x, h_, w_ in ((180, 340, 60), (1260, 420, 70), (1080, 200, 40)):
            over(img, poly_mask([[(x - w_, 2200), (x - w_ * 0.3, 2200 - h_), (x + w_ * 0.3, 2200 - h_ + 30), (x + w_, 2200)]]), "#040410", 1.0)
        return img, 1.2, 0.25
    img = vgrad([(0, "#02101A"), (1, "#041E24")])
    glow(img, 720, 2300, 1100, (0.1, 0.5, 0.45), 0.5, "gauss")
    pal = [("#3DFFC8", "#1AA890"), ("#B05CFF", "#6A2AB0"), ("#FF5FA8", "#B02A70"), ("#5CD8FF", "#2A88B0")]
    for k_ in range(14):
        x0 = r.random() * W
        base = 2300 + r.random() * 120
        h_ = 500 + r.random() * 1100
        sw = r.uniform(-260, 260)
        pts = [(x0 + sw * (t ** 2) + 30 * math.sin(t * 4 + k_), base - h_ * t) for t in np.linspace(0, 1, 30)]
        c1, c2 = pal[k_ % 4]
        m = line_mask([pts], 9, 2)
        neon(img, m, c2, core=0.2, glows=((4, 0.9), (16, 0.6), (50, 0.3)), k=0.8)
        ex, ey = pts[-1]
        glow(img, ex, ey, 70, C(c1), 0.6, "exp")
        circ(img, ex, ey, 26, c1, 0.95)
        circ(img, ex - 8, ey - 8, 9, "#FFFFFF", 0.8)
    for _ in range(280):
        x, y = r.random() * W, r.random() * 2300
        glow(img, x, y, 4 + r.random() * 8, C(pal[int(r.integers(0, 4))][0]), 0.55, "exp", 3)
    over(img, poly_mask([[(0, 2420), (W, 2400), (W, H), (0, H)]]), "#020A0E", 1.0)
    return img, 1.2, 0.3


def g_hard_tex(seed, mode):
    if mode == "carbon":
        T = 24
        i = (XX // (T / 2)).astype(np.int32)
        j = (YY // (T / 2)).astype(np.int32)
        up = ((i + j) % 4) < 2
        lx = (XX % (T / 2)) / (T / 2)
        ly = (YY % (T / 2)) / (T / 2)
        v = np.where(up, ly, lx)
        base = 0.1 + 0.2 * np.where(((i // 2 + j // 2) % 2) == 0, v, 1 - v)
        sp = np.exp(-(((XX + YY) % 1200 - 600) / 220.0) ** 2) * 0.35
        img = np.repeat(base[..., None], 3, -1) * np.array([1.0, 1.02, 1.08], np.float32) + sp[..., None] * np.array([0.7, 0.75, 0.85]) * (0.5 + 0.5 * np.where(up, 1, 0.4))[..., None]
        img *= (0.9 + 0.2 * fbm(seed, 4, 3))[..., None]
        return img.astype(np.float32), 0.8, 0.25
    if mode == "brushed":
        r = np.random.default_rng(seed)
        n = r.random((H, W // 8)).astype(np.float32)
        n = np.asarray(Image.fromarray(n, "F").resize((W, H), Image.BICUBIC))
        n = blur(n, 1.0)
        g = (n - 0.5) * 0.5
        c = 0.5 + 0.5 * np.cos((XX - 720) / 420.0 * math.pi * 0.9)
        shine = (0.28 + 0.52 * c ** 2) + 0.25 * np.exp(-(((XX - 300) / 70.0) ** 2))
        v = np.clip(shine + g * 0.5 + 0.05, 0, 1)
        img = ramp(v, [(0, "#2A2C32"), (0.5, "#8A8E98"), (1, "#EEF0F4")])
        for y in (320, 2240):
            for x in (140, 1300):
                circ(img, x, y, 34, "#5A5E68")
                circ(img, x - 4, y - 4, 26, "#B8BCC6")
                over(img, line_mask([[(x - 18, y - 2), (x + 18, y + 4)]], 5, 2), "#3A3E48", 0.9)
        return img, 0.8, 0.2
    n = fbm(seed, 4, 6)
    fine = fbm(seed + 1, 140, 2)
    img = ramp(n * 0.7 + fine * 0.3, [(0, "#6A6A68"), (0.5, "#9A9A96"), (1, "#C4C4BE")])
    rng = np.random.default_rng(seed)
    pores = np.zeros((H, W), np.float32)
    for _ in range(900):
        x, y, rr_ = rng.random() * W, rng.random() * H, 1.5 + rng.random() * 5
        pores[int(y), int(x)] = rr_
    pm = blur((pores > 0).astype(np.float32) * 6, 2.0)
    img *= (1 - 0.5 * np.clip(pm, 0, 1))[..., None]
    for gx in (0, 720):
        for gy in range(0, H, 640):
            seam = np.exp(-((XX - gx) / 2.0) ** 2) + np.exp(-((YY - gy) / 2.0) ** 2)
            img *= (1 - 0.35 * np.clip(seam, 0, 1))[..., None]
    for x in (360, 1080):
        for y in range(320, H, 640):
            circ(img, x, y, 22, "#4A4A48", 0.8)
            ring(img, x, y, 22, 4, "#B8B8B2", 0.4)
    st = smooth(0.6, 0.85, fbm(seed + 3, 3, 4, cy=10))
    img *= (1 - 0.22 * st)[..., None]
    return img, 1.4, 0.3


def g_textile(seed, mode):
    if mode == "denim":
        u = (XX + YY) / 5.0
        warp = 0.5 + 0.5 * np.sin(u * math.pi * 2 / 2)
        weft = fbm(seed, 300, 2, cy=300)
        base = ramp(0.35 + 0.35 * warp * (0.6 + 0.4 * weft) + (fbm(seed + 1, 8, 4) - 0.5) * 0.25, [(0, "#0F2A5A"), (0.5, "#2A5A9A"), (1, "#7AA6D8")])
        speck = (fbm(seed + 2, 220, 1, cy=220) > 0.82).astype(np.float32)
        img = base * (1 - 0.5 * speck[..., None]) + C("#D8E8F4") * 0.5 * speck[..., None]
        for y in (560, 600):
            dash = ((XX % 70) < 46).astype(np.float32)
            m = np.exp(-(((YY - y) / 4.5) ** 2)) * dash
            img = img * (1 - m[..., None]) + C("#E0A040") * m[..., None]
            img *= 1 - 0.25 * np.exp(-(((YY - y - 8) / 4.0) ** 2))[..., None]
        seam = np.exp(-(((YY - 580) / 26.0) ** 2))
        img *= 1 + 0.12 * seam[..., None]
        return img.astype(np.float32), 1.5, 0.25
    if mode == "leather":
        f1, f2, cid = cellular(46, seed)
        e = np.clip((f2 - f1) * 4, 0, 1)
        img = ramp(0.25 + 0.5 * e * (0.7 + 0.3 * cid) + 0.2 * fbm(seed + 1, 6, 3), [(0, "#2A0F06"), (0.5, "#6A2E14"), (1, "#B8723A")])
        sheen = np.exp(-(((XX - 400) / 520.0) ** 2) - ((YY - 700) / 900.0) ** 2) * 0.22
        img += sheen[..., None] * C("#FFD8A0")
        for x in (90, 1350):
            dash = ((YY % 56) < 36).astype(np.float32)
            m = np.exp(-(((XX - x) / 4.0) ** 2)) * dash
            img = img * (1 - 0.8 * m[..., None]) + C("#E8D2A0") * 0.8 * m[..., None]
            img *= 1 - 0.3 * (np.exp(-(((XX - x - 6) / 3.0) ** 2)) * dash)[..., None]
        return img.astype(np.float32), 1.5, 0.3
    T = 14.0
    hth = 0.5 + 0.5 * np.cos(XX / T * 2 * math.pi)
    vth = 0.5 + 0.5 * np.cos(YY / T * 2 * math.pi)
    over_ = ((np.floor(XX / T) + np.floor(YY / T)) % 2) < 1
    v = np.where(over_, hth, vth)
    n = fbm(seed, 60, 3)
    img = ramp(0.3 + 0.5 * v * (0.7 + 0.5 * n) + (fbm(seed + 2, 5, 3) - 0.5) * 0.2, [(0, "#4A3420"), (0.5, "#9A7A4A"), (1, "#D8BC84")])
    img *= (0.9 + 0.2 * fbm(seed + 3, 150, 1, cy=150))[..., None]
    return img.astype(np.float32), 1.6, 0.25


reg("cozy_room", g_cozy_room)
reg("cozy_corner", g_cozy_corner)
reg("blackhole", g_blackhole)
reg("alien_sky", g_alien_sky)
reg("hard_tex", g_hard_tex)
reg("textile", g_textile)

W2("cozy-desk-lamp", "Desk Lamp Night", "A warm desk lamp lights books, a steaming mug and a plant while the moon shines outside.", "cozy", True, True, 0.25, "cozy_room", mode="lamp")
W2("cozy-cafe-rain", "Cafe Rain", "A latte with a heart on a wooden table, with warm blurred lights beyond a rainy window.", "cozy", False, False, 0.25, "cozy_room", mode="cafe")
W2("cozy-bookshelf-glow", "Bookshelf Glow", "Rows of soft-coloured books on dark shelves, with one small lamp glowing.", "cozy", False, False, 0.25, "cozy_room", mode="shelf")
W2("cozy-plant-corner", "Plant Corner", "A sunlit corner with a terracotta pot and a big leafy plant by a window.", "cozy", False, False, 0.2, "cozy_corner", mode="plant")
W2("cozy-cat-window", "Cat on the Sill", "A sleeping cat curled on a windowsill, with curtains and a moonlit city behind.", "cozy", True, False, 0.2, "cozy_corner", mode="cat")
W2("cozy-fireplace", "Fireplace Glow", "A brick fireplace with dancing flames, two candles and a soft rug in warm light.", "cozy", False, False, 0.2, "cozy_corner", mode="fire")
W2("space-lensed-black-hole", "Lensed Black Hole", "A black hole bends the light of its glowing disk into a bright arc over the stars.", "space", True, True, 0.1, "blackhole", mode="lensed")
W2("space-accretion-spiral", "Accretion Spiral", "Seen from above, a white-hot spiral disk falls into a perfectly black centre.", "space", False, False, 0.1, "blackhole", mode="faceon")
W2("space-quasar-jets", "Quasar Jets", "Two blue jets shoot from a tiny bright core with a thin edge-on disk.", "space", False, False, 0.1, "blackhole", mode="jets")
W2("alien-binary-sunset", "Binary Sunset", "Two suns, one big orange and one small blue, set over purple alien dunes.", "space", False, False, 0.1, "alien_sky", mode="binary")
W2("alien-ringed-moonrise", "Ringed Moonrise", "A giant ringed planet rises above dark rock spires on a small moon.", "space", True, False, 0.1, "alien_sky", mode="moonrise")
W2("alien-glow-flora", "Glow Flora", "Tall glowing alien plants and drifting spores in a dark teal night.", "space", False, False, 0.1, "alien_sky", mode="flora")
W2("tex-carbon-fibre", "Carbon Fibre", "A tight twill weave of black carbon fibre with a diagonal sheen.", "patterns", False, False, 0.25, "hard_tex", mode="carbon")
W2("tex-brushed-steel", "Brushed Steel", "Fine horizontal brushing on cool steel with a soft highlight and two screws.", "patterns", False, False, 0.25, "hard_tex", mode="brushed")
W2("tex-raw-concrete", "Raw Concrete", "Pale grey concrete with pores, panel seams and tie holes.", "patterns", False, False, 0.25, "hard_tex", mode="concrete")
W2("tex-blue-denim", "Blue Denim", "Twill-woven blue denim with white flecks and a golden stitched seam.", "patterns", False, False, 0.25, "textile", mode="denim")
W2("tex-brown-leather", "Brown Leather", "Pebbled brown leather with a soft sheen and cream stitching on both edges.", "patterns", True, False, 0.25, "textile", mode="leather")
# dropped in review: W2("tex-natural-linen", "Natural Linen", "A coarse natural linen weave with soft fibres and warm tones.", "patterns", False, False, 0.25, "textile", mode="linen")


# ====================================================================== WAVE 2: categories
CATEGORIES += [("amoled", "AMOLED"), ("cozy", "Cozy"), ("anime", "Anime")]
NEW_CAT_LABELS.update({
    "amoled": {"ar": "أموليد", "cs": "AMOLED", "de": "AMOLED", "es": "AMOLED", "fa": "آمولد", "fr": "AMOLED", "hi": "AMOLED", "id": "AMOLED", "it": "AMOLED", "ja": "AMOLED", "ko": "AMOLED", "pl": "AMOLED", "pt": "AMOLED", "ru": "AMOLED", "tr": "AMOLED", "uk": "AMOLED", "zh": "AMOLED"},
    "cozy": {"ar": "أجواء دافئة", "cs": "Útulno", "de": "Gemütlich", "es": "Acogedor", "fa": "دنج", "fr": "Cosy", "hi": "आरामदायक", "id": "Nyaman", "it": "Accogliente", "ja": "くつろぎ", "ko": "아늑함", "pl": "Przytulnie", "pt": "Aconchegante", "ru": "Уют", "tr": "Rahat", "uk": "Затишок", "zh": "温馨"},
    "anime": {"ar": "أنمي", "cs": "Anime", "de": "Anime", "es": "Anime", "fa": "انیمه", "fr": "Anime", "hi": "एनीमे", "id": "Anime", "it": "Anime", "ja": "アニメ", "ko": "애니", "pl": "Anime", "pt": "Anime", "ru": "Аніме".replace("і", "и"), "tr": "Anime", "uk": "Аніме", "zh": "动漫"},
})

_W2_PRO = {"cockpit-nebula", "puzzle-gem-cascade", "anime-sakura-street", "sea-coral-reef", "glacier-ice-cave", "bullet-ring-burst", "cozy-fireplace", "tiles-lava-cracks", "lowpoly-night-range", "desert-saguaro-night", "space-accretion-spiral", "y2k-aero-ribbons"}
_W2_FEAT = {"amoled-edge-light", "lowpoly-sunset-isle", "rpg-world-map", "arena-dojo-dusk", "anime-starry-rooftop", "sea-glow-jellies", "cozy-desk-lamp", "space-lensed-black-hole", "islamic-star-tiles", "craft-indigo-shibori"}
for _e in E:
    if _e.get("w2"):
        if _e["id"] in _W2_PRO:
            _e["pro"] = True
        _e["feat"] = _e["id"] in _W2_FEAT
#### ---- WAVE2 END ----
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
        "requiresPro": bool(e["pro"]), "minAppVersionCode": 0, "swatch": (e["swatch"] or ["#000000"] * 4)[:4],
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
    auto_sw = e["swatch"] is None
    if auto_sw:
        try:
            lst["swatch"] = json.loads((ldir / "listing.json").read_text(encoding="utf-8")).get("swatch") or ["#000000"] * 4
        except (OSError, ValueError):
            lst["swatch"] = ["#000000"] * 4
    lst = keep_locales(lst, ldir / "listing.json")
    ddir = ROOT / "dist" / e["id"]
    ldir.mkdir(parents=True, exist_ok=True)
    ddir.mkdir(parents=True, exist_ok=True)
    (ldir / "listing.json").write_text(json.dumps(lst, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    h = hashlib.sha256()
    h.update(json.dumps({k: v for k, v in lst.items() if not (auto_sw and k == "swatch")}, sort_keys=True).encode())
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
    if auto_sw:
        lst["swatch"] = auto_swatch(im)
        (ldir / "listing.json").write_text(json.dumps(lst, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
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
    (ROOT / "categories.json").write_text(json.dumps(keep_label_locales([{"id": i, "labels": {"en": n, **NEW_CAT_LABELS.get(i, {})}} for i, n in CATEGORIES], ROOT / "categories.json"), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    full_src = Path(__file__).read_text(encoding="utf-8")
    w2_a, w2_b = full_src.find("#### ---- WAVE2 BEGIN ----"), full_src.find("#### ---- WAVE2 END ----")
    w2_src = full_src[w2_a:w2_b] if w2_a >= 0 else ""
    script_hash = hashlib.sha256((full_src[:w2_a] + full_src[w2_b:] if w2_a >= 0 else full_src).encode()).hexdigest()
    w2_hash = hashlib.sha256(w2_src.encode()).hexdigest()
    only = set(x for x in a.only.split(",") if x)
    todo = [(e, script_hash + (w2_hash if e.get("w2") else ""), a.force) for e in E if not only or e["id"] in only]
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
