#!/usr/bin/env python3
"""Synthesizes the EmuCore Store UI sound packs (100% original, generated from math + seeded noise).

Output (idempotent, deterministic):
  store/sounds/categories.json
  store/sounds/src/<id>/listing.json
  store/sounds/dist/<id>/{item.ecxsounds, cover.webp, waveform.webp, build.json}

Usage: python scripts/build-store-sounds.py [--force] [--only <id>[,<id>...]]
Pure stdlib for audio (wave, math, random, zipfile); Pillow for the images.
"""
import colorsys
import hashlib
import io
import json
import math
import os
import random
import sys
import wave
import zipfile
import zlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SR = 22050
TAU = 2 * math.pi
ROOT = Path(__file__).resolve().parent.parent
SOUNDS = ROOT / "store" / "sounds"
SELF = Path(__file__).resolve()
PUBLISHED_AT = "2026-11-24T12:00:00Z"
PEAK_SHORT = 10 ** (-3 / 20)
PEAK_MUSIC = 10 ** (-6 / 20)

R = random.Random(0)


def seed(name):
    R.seed(zlib.crc32(name.encode()))


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def ns(d):
    return max(1, int(d * SR))


# ----------------------------------------------------------------------------- primitives
def tone(f, dur, wave="sine", a=0.003, tau=0.1, hold=0.0, glide=1.0, glide_t=0.05, vib=0.0,
         vib_rate=5.5, duty=0.5, fm=None, h2=0.0, h3=0.0, vol=1.0, rel=0.012):
    n = ns(dur)
    out = [0.0] * n
    ph = 0.0
    mph = 0.0
    inv = 1.0 / SR
    sin = math.sin
    exp = math.exp
    for i in range(n):
        t = i * inv
        fr = f
        if glide != 1.0:
            fr = f * (1 + (glide - 1) * exp(-t / glide_t))
        if vib:
            fr *= 1 + vib * sin(TAU * vib_rate * t)
        e = t / a if t < a else 1.0
        if t > hold:
            e *= exp(-(t - hold) / tau)
        r = (dur - t) / rel
        if r < 1:
            e *= r
        ph += fr * inv
        ph -= int(ph)
        if wave == "sine":
            if fm:
                mph += fr * fm[0] * inv
                mph -= int(mph)
                s = sin(TAU * ph + fm[1] * exp(-t / fm[2]) * sin(TAU * mph))
            else:
                s = sin(TAU * ph)
            if h2:
                s += h2 * sin(2 * TAU * ph)
            if h3:
                s += h3 * sin(3 * TAU * ph)
        elif wave == "tri":
            s = 4 * abs(ph - 0.5) - 1
        elif wave == "sq":
            s = 1.0 if ph < duty else -1.0
        else:  # saw
            s = 2 * ph - 1
        out[i] = s * e * vol
    return out


def sweep(f0, f1, dur, wave="sine", a=0.008, rel=0.03, duty=0.5, vol=1.0, tau=None):
    n = ns(dur)
    out = [0.0] * n
    ph = 0.0
    ratio = f1 / f0
    for i in range(n):
        t = i / SR
        fr = f0 * ratio ** (t / dur)
        ph += fr / SR
        ph -= int(ph)
        if wave == "sine":
            s = math.sin(TAU * ph)
        elif wave == "sq":
            s = 1.0 if ph < duty else -1.0
        elif wave == "tri":
            s = 4 * abs(ph - 0.5) - 1
        else:
            s = 2 * ph - 1
        e = min(1.0, t / a, (dur - t) / rel)
        if tau:
            e *= math.exp(-t / tau)
        out[i] = s * e * vol
    return out


def additive(f, dur, parts, a=0.002, rel=0.015, vol=1.0, vib=0.0):
    n = ns(dur)
    out = [0.0] * n
    for ratio, amp, tau in parts:
        fr = f * ratio
        if fr > SR * 0.45:
            continue
        w = TAU * fr / SR
        d = math.exp(-1.0 / (SR * tau))
        amp *= vol
        sin = math.sin
        if vib:
            for i in range(n):
                out[i] += amp * sin(w * i + vib * sin(TAU * 5.2 * i / SR))
                amp *= d
        else:
            for i in range(n):
                out[i] += amp * sin(w * i)
                amp *= d
    na = max(1, int(a * SR))
    for i in range(min(na, n)):
        out[i] *= i / na
    nr = max(1, int(rel * SR))
    for i in range(min(nr, n)):
        out[n - 1 - i] *= i / nr
    return out


def ks(f, dur, tau=0.6, bright=0.5, vol=1.0):
    """Karplus-Strong plucked string with fractional delay."""
    n = ns(dur)
    D = SR / f - 0.5
    Di = int(D)
    fr = D - Di
    out = [0.0] * n
    init = min(n, Di + 3)
    prev = 0.0
    for i in range(init):  # smoothed noise burst excitation
        prev = prev + bright * (R.uniform(-1, 1) - prev)
        out[i] = prev
    g = math.exp(-1.0 / (f * tau))
    for i in range(Di + 3, n):
        a0 = out[i - Di]
        a1 = out[i - Di - 1]
        a2 = out[i - Di - 2]
        out[i] = g * 0.5 * (a0 * (1 - fr) + a1 * fr + a1 * (1 - fr) + a2 * fr)
    nr = int(0.012 * SR)
    for i in range(min(nr, n)):
        out[n - 1 - i] *= i / nr
    return [v * vol for v in out]


def noise(dur, a=0.001, tau=0.03, vol=1.0):
    n = ns(dur)
    out = [0.0] * n
    na = max(1, int(a * SR))
    for i in range(n):
        e = math.exp(-i / (SR * tau))
        if i < na:
            e *= i / na
        out[i] = R.uniform(-1, 1) * e * vol
    return out


def lp(x, fc):
    a = 1 - math.exp(-TAU * fc / SR)
    y = 0.0
    out = [0.0] * len(x)
    for i, v in enumerate(x):
        y += a * (v - y)
        out[i] = y
    return out


def hp(x, fc):
    l = lp(x, fc)
    return [v - w for v, w in zip(x, l)]


def bq(x, b0, b1, b2, a1, a2):
    x1 = x2 = y1 = y2 = 0.0
    out = [0.0] * len(x)
    for i, v in enumerate(x):
        y = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, v
        y2, y1 = y1, y
        out[i] = y
    return out


def bp(x, fc, q=1.0):
    w = TAU * fc / SR
    al = math.sin(w) / (2 * q)
    a0 = 1 + al
    return bq(x, al / a0, 0.0, -al / a0, -2 * math.cos(w) / a0, (1 - al) / a0)


def bp_sweep(x, f0, f1, q=1.5):
    n = len(x)
    out = [0.0] * n
    x1 = x2 = y1 = y2 = 0.0
    for i in range(n):
        fc = f0 * (f1 / f0) ** (i / max(1, n - 1))
        w = TAU * fc / SR
        al = math.sin(w) / (2 * q)
        a0 = 1 + al
        b0 = al / a0
        a1 = -2 * math.cos(w) / a0
        a2 = (1 - al) / a0
        v = x[i]
        y = b0 * v - b0 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, v
        y2, y1 = y1, y
        out[i] = y
    return out


def lp_sweep(x, f0, f1, tau):
    y = 0.0
    out = [0.0] * len(x)
    for i, v in enumerate(x):
        fc = f1 + (f0 - f1) * math.exp(-i / (SR * tau))
        a = 1 - math.exp(-TAU * fc / SR)
        y += a * (v - y)
        out[i] = y
    return out


def bitcrush(x, bits=6, hold=1):
    q = 2 ** (bits - 1)
    out = [0.0] * len(x)
    cur = 0.0
    for i, v in enumerate(x):
        if i % hold == 0:
            cur = round(v * q) / q
        out[i] = cur
    return out


def echo(x, delay, g, taps=3):
    d = int(delay * SR)
    n = len(x) + d * taps
    out = x + [0.0] * (n - len(x))
    for k in range(1, taps + 1):
        for i, v in enumerate(x):
            out[i + d * k] += v * (g ** k)
    return out


def gain(x, g):
    return [v * g for v in x]


def layer(items):
    """items: (offset_seconds, samples, gain)"""
    n = max(int(o * SR) + len(s) for o, s, g in items)
    out = [0.0] * n
    for o, s, g in items:
        k = int(o * SR)
        for i, v in enumerate(s):
            out[k + i] += v * g
    return out


def phrase(v, notes, gap, last=0.3, d=None, tail_gain=1.0):
    items = []
    for k, m in enumerate(notes):
        dur = last if k == len(notes) - 1 else (d or gap * 3)
        items.append((k * gap, v(hz(m), dur), tail_gain if k == len(notes) - 1 else 1.0))
    return layer(items)


def finish(x, peak=PEAK_SHORT, fades=True):
    n = len(x)
    m = sum(x) / n
    x = [v - m for v in x]
    fi = min(int(0.002 * SR), n // 4) if fades else 0
    fo = min(int(0.02 * SR), n // 3) if fades else 0
    for i in range(fi):
        x[i] *= i / fi
    for i in range(fo):
        x[n - 1 - i] *= i / fo
    pk = max(abs(v) for v in x) or 1.0
    k = peak / pk
    return [v * k for v in x]


TARGET_SEL_RMS = 10 ** (-18 / 20)
TARGET_MUSIC_RMS = 10 ** (-24 / 20)


def limit(x, peak):
    """Transparent below 70 % of the ceiling, smooth tanh knee above it; never exceeds peak."""
    knee = peak * 0.7
    out = []
    for v in x:
        a = abs(v)
        if a > knee:
            a = knee + (peak - knee) * math.tanh((a - knee) / (peak - knee))
        out.append(a if v >= 0 else -a)
    return out


def active_rms(x):
    thr = 0.05 * max(abs(v) for v in x)
    act = [v for v in x if abs(v) >= thr]
    return math.sqrt(sum(v * v for v in act) / len(act))


def loudness_match(shorts):
    k = TARGET_SEL_RMS / active_rms(shorts["menu_select"])
    return {n: limit([v * k for v in x], PEAK_SHORT) for n, x in shorts.items()}


def music_match(x):
    rms = math.sqrt(sum(v * v for v in x) / len(x))
    return limit([v * (TARGET_MUSIC_RMS / rms) for v in x], PEAK_MUSIC)


def write_ogg(samples, quality=0.45):
    """Music loops ship as OGG Vorbis (about a tenth of the WAV size); Android loops OGG gaplessly."""
    import numpy as np
    import soundfile as sf
    buf = io.BytesIO()
    data = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    # Written in small blocks: one big write overflows the stack inside libsndfile's Vorbis encoder on Windows.
    with sf.SoundFile(buf, "w", samplerate=SR, channels=1, format="OGG", subtype="VORBIS",
                      compression_level=1.0 - quality) as out:
        for start in range(0, len(data), 4096):
            out.write(data[start:start + 4096])
    return buf.getvalue()


def write_wav(samples):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(
            int(max(-32767, min(32767, round(v * 32767)))).to_bytes(2, "little", signed=True) for v in samples))
    return buf.getvalue()


# ----------------------------------------------------------------------------- packs
def std(v, open_n, close_n, sel_n, gap, last=0.35, sel_gap=None, click=None, click_f=84):
    click = click if click is not None else v(hz(click_f), 0.06)
    return dict(
        click=click,
        menu_open=phrase(v, open_n, gap, last),
        menu_close=phrase(v, close_n, gap, last * 0.8),
        menu_select=phrase(v, sel_n, sel_gap or gap * 1.4, last * 1.2))


def p_soft_pop():
    v = lambda f, d: tone(f, d, a=0.004, tau=0.09, fm=(2, 0.7, 0.05), h2=0.15)
    c = tone(620, 0.06, a=0.002, tau=0.02, glide=2.0, glide_t=0.012, h2=0.2)
    return std(v, [67, 71, 74], [74, 71, 67], [72, 79], 0.055, 0.3, click=c)


def tap_glass(f, d=0.5, vol=1.0):
    t = additive(f, d, [(1, 1.0, 0.1), (2.76, 0.6, 0.05), (5.4, 0.3, 0.03), (8.93, 0.12, 0.02)], vol=vol)
    tick = hp(noise(0.01, tau=0.002, vol=0.5), 4000)
    return layer([(0, t, 1), (0, tick, 0.6)])


def p_glass_tap():
    return dict(
        click=tap_glass(2800, 0.07),
        menu_open=layer([(0, tap_glass(2000, 0.3), 1), (0.07, tap_glass(3000, 0.4), 1)]),
        menu_close=layer([(0, tap_glass(3000, 0.3), 1), (0.07, tap_glass(2000, 0.35), 0.9)]),
        menu_select=layer([(0, tap_glass(2400, 0.3), 1), (0.06, tap_glass(3600, 0.6), 1.1)]))


CHIME = [(1, 1.0, 0.6), (2.0, 0.4, 0.4), (3.01, 0.3, 0.3), (4.07, 0.15, 0.2), (5.4, 0.1, 0.12)]


def chime_v(f, d):
    return additive(f, d, CHIME, vib=0.4)


def p_crystal_chime():
    return std(chime_v, [79, 83, 86, 91], [91, 86, 83, 79], [86, 91], 0.065, 0.9, sel_gap=0.1, click_f=91)


def wood_hit(f, d=0.09, vol=1.0):
    body = tone(f, d, a=0.0008, tau=0.022, glide=1.12, glide_t=0.01, h2=0.35, vol=0.8)
    knock = bp(noise(d, tau=0.01, vol=1.0), f * 1.5, 3.0)
    return layer([(0, body, 1), (0, knock, 1.2 * vol)])


def p_wood_block():
    return dict(
        click=wood_hit(900, 0.05),
        menu_open=layer([(0, wood_hit(700), 1), (0.08, wood_hit(1050, 0.12), 1)]),
        menu_close=layer([(0, wood_hit(1050), 1), (0.08, wood_hit(700, 0.12), 0.9)]),
        menu_select=layer([(0, wood_hit(850), 1), (0.07, wood_hit(850), 0.7), (0.15, wood_hit(1300, 0.16), 1.1)]))


def drop(f, d=0.25, vol=1.0):
    return tone(f, d, a=0.003, tau=0.075, glide=0.55, glide_t=0.035, h2=0.12, vol=vol)


def p_water_drop():
    sel = layer([(0, drop(900, 0.25), 1), (0.09, drop(1350, 0.4), 0.9)])
    return dict(
        click=drop(1400, 0.06),
        menu_open=layer([(0, drop(700, 0.2), 1), (0.07, drop(900, 0.2), 1), (0.14, drop(1200, 0.3), 1)]),
        menu_close=layer([(0, drop(1200, 0.2), 1), (0.07, drop(900, 0.2), 0.9), (0.14, drop(700, 0.3), 0.8)]),
        menu_select=echo(sel, 0.09, 0.3, 3))


def key_hit(fc, thump, d=0.08, vol=1.0):
    n = bp(noise(d, a=0.0005, tau=0.012), fc, 1.6)
    th = tone(thump, d, a=0.001, tau=0.02, vol=0.9)
    return layer([(0, n, 1.1 * vol), (0, th, 0.8 * vol)])


def p_mech_keys():
    ping = additive(3800, 0.2, [(1, 0.5, 0.06), (2.3, 0.2, 0.03)])
    return dict(
        click=key_hit(3200, 140, 0.05),
        menu_open=layer([(0, key_hit(2600, 150), 1), (0.07, key_hit(3600, 160), 1)]),
        menu_close=layer([(0, key_hit(3600, 160), 1), (0.07, key_hit(2400, 120), 0.9)]),
        menu_select=layer([(0, key_hit(1200, 90, 0.12), 1.2), (0.0, ping, 0.25)]))


def tw_strike(vol=1.0, d=0.09):
    n = bp(noise(d, a=0.0005, tau=0.02), 1800, 0.9)
    th = tone(110, d, a=0.001, tau=0.025, vol=0.9)
    return layer([(0, n, vol), (0, th, vol * 0.7)])


def ratchet(n_ticks, f0, f1, rev=False):
    items = []
    t = 0.0
    gap = 0.07
    for k in range(n_ticks):
        fc = f0 * (f1 / f0) ** (k / max(1, n_ticks - 1))
        items.append((t, bp(noise(0.02, a=0.0003, tau=0.004), fc, 2.5), 1.0))
        t += gap
        gap *= 0.8 if not rev else 1.25
    return layer(items)


def p_typewriter():
    bell = additive(2600, 0.7, [(1, 1.0, 0.5), (2.4, 0.5, 0.3), (3.7, 0.25, 0.2), (5.9, 0.1, 0.1)])
    return dict(
        click=tw_strike(1.0, 0.06),
        menu_open=ratchet(6, 1500, 3500),
        menu_close=ratchet(6, 3500, 1500, rev=True),
        menu_select=layer([(0, tw_strike(), 1), (0.1, bell, 0.7)]))


def swish(f0, f1, d, q=1.2, shape=0.5):
    n = ns(d)
    nz = [R.uniform(-1, 1) for _ in range(n)]
    nz = bp_sweep(nz, f0, f1, q)
    peak_i = int(n * shape)
    for i in range(n):
        e = i / peak_i if i < peak_i else (n - i) / (n - peak_i)
        nz[i] *= e * e
    return nz


def p_paper_flick():
    tap = layer([(0, tone(200, 0.05, tau=0.015, a=0.001), 0.6), (0, noise(0.04, tau=0.008), 0.5)])
    return dict(
        click=swish(3000, 6000, 0.06, 1.0, 0.25),
        menu_open=swish(800, 4500, 0.28, 1.2, 0.65),
        menu_close=swish(4500, 800, 0.28, 1.2, 0.35),
        menu_select=layer([(0, swish(1500, 5500, 0.12, 1.0, 0.3), 1), (0.12, tap, 0.9)]))


def p_chip_blips():
    v = lambda f, d: bitcrush(tone(f, d, "sq", a=0.001, tau=0.06, duty=0.5, rel=0.004), 6)
    items_open = [(k * 0.05, v(hz(m), 0.06), 1) for k, m in enumerate([72, 76, 79, 84])]
    items_open[-1] = (0.15, v(hz(84), 0.2), 1)
    return dict(
        click=v(1320, 0.03),
        menu_open=layer(items_open),
        menu_close=layer([(k * 0.05, v(hz(m), 0.06 if k < 3 else 0.2), 1) for k, m in enumerate([84, 79, 76, 72])]),
        menu_select=layer([(0, v(hz(76), 0.07), 1), (0.07, tone(hz(83), 0.22, "tri", a=0.001, tau=0.12), 1.2)]))


def p_arcade_coin():
    v = lambda f, d: bitcrush(tone(f, d, "sq", a=0.001, tau=0.1, duty=0.25, rel=0.004), 5)
    run = [(k * 0.035, v(hz(m), 0.05), 1) for k, m in enumerate([67, 71, 74, 79, 83])]
    run.append((5 * 0.035, bitcrush(tone(hz(91), 0.3, "sq", a=0.001, tau=0.15, duty=0.25, vib=0.02), 5), 1))
    return dict(
        click=v(1800, 0.03),
        menu_open=layer([(0, bitcrush(sweep(400, 1600, 0.17, "sq", duty=0.25, a=0.004, rel=0.01), 5), 1),
                         (0.17, v(hz(88), 0.15), 1)]),
        menu_close=layer([(0, bitcrush(sweep(1600, 400, 0.17, "sq", duty=0.25, a=0.004, rel=0.01), 5), 1),
                          (0.17, v(hz(55), 0.15), 1)]),
        menu_select=layer(run))


def saw_pluck(f, d, cut=4500):
    a = tone(f, d, "saw", a=0.002, tau=0.2)
    b = tone(f * 1.0045, d, "saw", a=0.002, tau=0.2)
    x = [(p + q) * 0.5 for p, q in zip(a, b)]
    return lp_sweep(x, cut, 350, 0.1)


def p_synth_pluck():
    v = lambda f, d: saw_pluck(f, d)
    sel = layer([(0, saw_pluck(hz(64), 0.3), 1), (0.0, saw_pluck(hz(71), 0.3), 0.8), (0.09, saw_pluck(hz(76), 0.5), 1),
                 (0.09, saw_pluck(hz(83), 0.5), 0.7)])
    return dict(
        click=saw_pluck(hz(84), 0.06, 6000),
        menu_open=phrase(v, [57, 60, 64, 69], 0.06, 0.4),
        menu_close=phrase(v, [69, 64, 60, 57], 0.06, 0.35),
        menu_select=sel)


def scifi_v(f, d):
    return tone(f, d, a=0.002, tau=0.05, fm=(1.414, 2.5, 0.06), vol=1.0)


def p_scifi_console():
    hiss = bp(noise(0.12, tau=0.04, vol=0.5), 5000, 1.5)
    steps_up = [(k * 0.04, scifi_v(f, 0.045), 1) for k, f in enumerate([1200, 1500, 1350, 1900, 1700, 2400])]
    steps_dn = [(k * 0.04, scifi_v(f, 0.045), 1) for k, f in enumerate([2400, 1900, 2100, 1500, 1650, 1100])]
    sel = layer([(0, scifi_v(1500, 0.08), 1), (0.09, scifi_v(2250, 0.2), 1)])
    return dict(
        click=scifi_v(2200, 0.04),
        menu_open=layer(steps_up + [(0, hiss, 0.3)]),
        menu_close=layer(steps_dn + [(0, hiss, 0.3)]),
        menu_select=echo(sel, 0.08, 0.35, 2))


def p_laser_ui():
    lx = lambda a, b, d, w="saw", vol=1.0: lp(sweep(a, b, d, w, a=0.004, rel=0.02, vol=vol), 6000)
    sel = layer([(0, lx(2000, 800, 0.1), 1), (0.11, lx(2600, 1000, 0.1), 1), (0.22, tone(3200, 0.2, tau=0.1), 0.5)])
    return dict(
        click=lx(5000, 2200, 0.045, "sine"),
        menu_open=layer([(0, lx(300, 2400, 0.28), 1), (0, lx(302, 2410, 0.28), 0.6)]),
        menu_close=layer([(0, lx(2400, 300, 0.28), 1), (0, lx(2410, 302, 0.28), 0.6)]),
        menu_select=sel)


def bubble(f, d=0.3, vol=1.0):
    b = tone(f, d, a=0.004, tau=0.12, glide=0.6, glide_t=0.06, vib=0.03, vib_rate=11, h2=0.1, vol=vol)
    pop = lp(noise(0.02, tau=0.004, vol=0.4), 1500)
    return layer([(0, b, 1), (0, pop, 0.5)])


def p_deep_bubble():
    return dict(
        click=bubble(300, 0.08),
        menu_open=layer([(0, bubble(160, 0.25), 1), (0.09, bubble(210, 0.25), 1), (0.18, bubble(290, 0.35), 1)]),
        menu_close=layer([(0, bubble(290, 0.25), 1), (0.09, bubble(210, 0.25), 0.9), (0.18, bubble(150, 0.4), 0.9)]),
        menu_select=layer([(0, bubble(220, 0.4), 1), (0.12, bubble(660, 0.12), 0.7)]))


def marimba_v(f, d):
    bar = additive(f, d, [(1, 1.0, 0.3), (3.99, 0.4, 0.07), (9.9, 0.1, 0.025)])
    mal = lp(noise(0.006, tau=0.002, vol=0.3), 2500)
    return layer([(0, bar, 1), (0, mal, 0.5)])


def p_marimba():
    return std(marimba_v, [60, 64, 67, 72], [72, 67, 64, 60], [67, 72], 0.07, 0.6, sel_gap=0.09, click_f=79)


def kalimba_v(f, d):
    tine = additive(f, d, [(1, 1.0, 0.45), (5.45, 0.22, 0.08), (9.0, 0.07, 0.04)])
    th = lp(noise(0.008, tau=0.003, vol=0.5), 1800)
    return layer([(0, tine, 1), (0, th, 0.4)])


def p_kalimba():
    return std(kalimba_v, [74, 78, 81, 86], [86, 81, 78, 74], [81, 86], 0.07, 0.7, sel_gap=0.1, click_f=88)


def toy_v(f, d):
    bar = additive(f, d, [(1, 1.0, 0.25), (2.9, 0.6, 0.12), (5.1, 0.35, 0.07), (7.7, 0.2, 0.04)])
    ham = hp(noise(0.008, tau=0.002, vol=0.5), 2500)
    return layer([(0, bar, 1), (0, ham, 0.5)])


def p_toy_piano():
    return std(toy_v, [79, 83, 86, 91], [91, 86, 83, 79], [84, 91], 0.06, 0.55, sel_gap=0.09, click_f=96)


def tick(fc, d=0.03, vol=1.0):
    x = [0.0] * ns(d)
    x[0] = 1.0
    x[1] = -0.6
    return gain(bp(x, fc, 4.0), 12 * vol)


def p_minimal_tick():
    tock = lp(tone(900, 0.05, tau=0.012, a=0.001), 2000)
    return dict(
        click=tick(3000),
        menu_open=layer([(0, tick(2500), 1), (0.06, tick(3600), 1)]),
        menu_close=layer([(0, tick(3600), 1), (0.06, tick(2500), 0.9)]),
        menu_select=layer([(0, tick(3200), 1), (0.05, tick(4400), 1), (0.05, tock, 0.4)]))


def relay(vol=1.0):
    th = tone(90, 0.06, a=0.001, tau=0.015)
    nz = bp(noise(0.03, a=0.0003, tau=0.006), 1200, 1.2)
    return layer([(0, th, vol), (0, nz, 1.1 * vol)])


def p_analog_click():
    blip = lp(tone(440, 0.14, "saw", a=0.003, tau=0.05), 2200)
    return dict(
        click=relay(),
        menu_open=layer([(0, relay(), 1), (0.02, lp(sweep(80, 240, 0.2, "sq", a=0.01, rel=0.05), 900), 0.7)]),
        menu_close=layer([(0, lp(sweep(240, 80, 0.2, "sq", a=0.01, rel=0.05), 900), 0.7), (0.19, relay(0.9), 1)]),
        menu_select=layer([(0, relay(), 1), (0.03, blip, 0.8)]))


def felt_v(f, d):
    x = tone(f, d, a=0.012, tau=0.24, h2=0.3, h3=0.08)
    sub = tone(f * 0.5, d, a=0.012, tau=0.2, vol=0.2)
    return lp([p + q for p, q in zip(x, sub)], 1800)


def p_warm_felt():
    return std(felt_v, [55, 60, 64], [64, 60, 55], [60, 67], 0.075, 0.55, sel_gap=0.1, click_f=72)


BELL = [(0.5, 0.4, 0.9), (1.0, 1.0, 0.9), (1.19, 0.5, 0.7), (1.56, 0.3, 0.5), (2.0, 0.4, 0.45), (2.51, 0.2, 0.3),
        (3.0, 0.15, 0.25), (4.1, 0.1, 0.15)]


def bell_v(f, d):
    return additive(f, d, BELL, a=0.002)


def p_temple_bell():
    return dict(
        click=bell_v(hz(84), 0.07),
        menu_open=layer([(0, bell_v(hz(60), 1.0), 1), (0.12, bell_v(hz(67), 1.1), 0.9)]),
        menu_close=layer([(0, bell_v(hz(67), 0.9), 0.9), (0.12, bell_v(hz(60), 1.2), 1)]),
        menu_select=layer([(0, bell_v(hz(72), 1.2), 1), (0.08, bell_v(hz(79), 1.3), 0.9)]))


def p_handheld_beep():
    b = lambda f, d: bitcrush(tone(f, d, "sq", a=0.001, tau=1.0, rel=0.004), 5, 2)
    return dict(
        click=b(2400, 0.025),
        menu_open=layer([(0, b(1000, 0.07), 1), (0.1, b(1500, 0.12), 1)]),
        menu_close=layer([(0, b(1500, 0.07), 1), (0.1, b(1000, 0.12), 1)]),
        menu_select=layer([(0, b(1800, 0.05), 1), (0.07, b(2400, 0.14), 1)]))


def glitch(dur, base, up=0.0, bits=5):
    out = []
    while len(out) / SR < dur:
        seg = R.choice([0.004, 0.006, 0.009])
        f = base * R.choice([1, 1.25, 1.5, 2, 3]) * (1 + up * len(out) / SR / dur)
        out += tone(f, seg, "sq", a=0.0004, tau=1.0, rel=0.0005, duty=R.choice([0.5, 0.25, 0.125]))
        if R.random() < 0.25:
            out += [R.uniform(-1, 1) * 0.5 for _ in range(ns(0.002))]
    x = bitcrush(out, bits, 2)
    n = len(x)
    return [v * math.exp(-3 * i / n) for i, v in enumerate(x)]


def p_cyber_glitch():
    return dict(
        click=glitch(0.035, 1500),
        menu_open=glitch(0.28, 500, up=3.0),
        menu_close=glitch(0.28, 1800, up=-0.7),
        menu_select=layer([(0, glitch(0.1, 900), 1), (0.11, tone(1760, 0.18, "sq", a=0.001, tau=0.08, duty=0.25), 0.8)]))


def ks_v(f, d):
    return lp(ks(f, d, tau=0.5, bright=0.55), 4500)


def strum(notes, gap, d):
    return layer([(k * gap, ks_v(hz(m), d), 1.0) for k, m in enumerate(notes)])


def p_cozy_pluck():
    return dict(
        click=ks_v(hz(84), 0.07),
        menu_open=strum([60, 64, 67, 72], 0.045, 0.6),
        menu_close=strum([72, 67, 64, 60], 0.045, 0.5),
        menu_select=strum([67, 72, 76], 0.06, 0.7))


# ----------------------------------------------------------------------------- music
CH = {"maj7": [0, 4, 7, 11], "m7": [0, 3, 7, 10], "add9": [0, 4, 7, 14], "m9": [0, 3, 7, 14],
      "sus2": [0, 2, 7, 14], "6": [0, 4, 7, 9]}
PROG_A = [(48, "maj7"), (45, "m7"), (41, "maj7"), (43, "6"), (48, "add9"), (40, "m7"), (41, "maj7"), (43, "sus2")]
PROG_B = [(45, "m7"), (41, "maj7"), (48, "add9"), (43, "6"), (45, "m9"), (41, "maj7"), (40, "m7"), (43, "sus2")]
PROG_C = [(38, "sus2"), (36, "m9"), (41, "maj7"), (43, "sus2"), (38, "m7"), (36, "m9"), (34, "maj7"), (43, "sus2")]
PROG_D = [(43, "6"), (40, "m7"), (48, "maj7"), (45, "m7"), (43, "add9"), (41, "maj7"), (48, "add9"), (38, "sus2")]


def pad_note(buf, f, start, length, g, kind):
    N = len(buf)
    w1 = TAU * f / SR
    w2 = TAU * f * 1.0025 / SR
    pi = math.pi
    sin = math.sin
    if kind == "saw":
        a = 1 - math.exp(-TAU * 900 / SR)
        y = 0.0
        step = f / SR
        step2 = f * 1.003 / SR
        p1 = p2 = 0.0
        for i in range(length):
            p1 += step
            p1 -= int(p1)
            p2 += step2
            p2 -= int(p2)
            y += a * ((p1 + p2) - 1.0 - y)
            e = sin(pi * i / length)
            buf[(start + i) % N] += y * e * e * g * 1.6
        return
    h2 = {"soft": 0.0, "warm": 0.35, "glass": 0.18}[kind]
    h3 = 0.08 if kind == "glass" else 0.0
    for i in range(length):
        e = sin(pi * i / length)
        e *= e
        a1 = w1 * i
        v = sin(a1) + sin(w2 * i) + h2 * sin(2 * a1) + h3 * sin(3 * a1)
        buf[(start + i) % N] += v * e * g


def music_loop(cfg):
    secs = cfg["secs"]
    prog = cfg["prog"]
    nch = len(prog)
    if cfg.get("bpm"):
        bar = 240.0 / cfg["bpm"]
        secs = nch * (2 if nch * 2 * bar >= 31 else 4) * bar
    clen = int(secs / nch * SR)
    N = clen * nch
    buf = [0.0] * N
    ov = int(1.6 * SR)
    sh = cfg.get("shift", 0)
    for ci, (root, typ) in enumerate(prog):
        chord = [root + sh + x for x in CH[typ]]
        start = ci * clen
        pad_g = cfg.get("pad", 0.12)
        for k, m in enumerate(chord):
            pad_note(buf, hz(m + 12), start - ov, clen + 2 * ov, pad_g * (1.0 if k < 3 else 0.7), cfg["padkind"])
        pad_note(buf, hz(chord[0]), start - ov, clen + 2 * ov, pad_g * 1.3, "soft")
        pool = sorted(m + o for m in chord for o in cfg["octs"])
        step = int(cfg["step"] * SR)
        count = clen // step
        idx = 0
        direction = 1
        for k in range(count):
            if R.random() < cfg["rest"]:
                continue
            pat = cfg["pattern"]
            if pat == "up":
                idx = k % len(pool)
            elif pat == "updown":
                idx += direction
                if idx >= len(pool) - 1 or idx <= 0:
                    direction = -direction
                idx = max(0, min(len(pool) - 1, idx))
            else:
                idx = R.randrange(len(pool))
            vel = 0.55 + 0.45 * R.random()
            x = cfg["voice"](hz(pool[idx]), cfg["arp_d"])
            off = start + k * step
            g = cfg["arp"] * vel
            for i, v in enumerate(x):
                buf[(off + i) % N] += v * g
    for snd, pat, g in cfg.get("bed", ()):
        stp = 60.0 / cfg["bpm"] / 4 * SR
        k = 0
        pos = 0.0
        while pos < N:
            ch = pat[k % len(pat)]
            if ch != ".":
                o_ = int(pos)
                lv = g * (1.0 if ch == "x" else 0.5)
                for i, v in enumerate(snd):
                    buf[(o_ + i) % N] += v * lv
            k += 1
            pos += stp
    for snd, prob, g in cfg.get("sparse", ()):
        stp = int(cfg["step"] * SR)
        for k in range(N // stp):
            if R.random() < prob:
                lv = g * (0.6 + 0.4 * R.random())
                for i, v in enumerate(snd):
                    buf[(k * stp + i) % N] += v * lv
    # circular reverb (delay lines wrap) => stays seamless
    wet = [0.0] * N
    combs = [(0.173, 0.55), (0.241, 0.5), (0.337, 0.45), (0.443, 0.4)]
    for dl, g in combs:
        d = int(dl * SR)
        c = buf[:]
        for _ in range(2):
            for i in range(N):
                c[i] = buf[i] + g * c[i - d]
        for i in range(N):
            wet[i] += c[i] - buf[i]
    mix = cfg.get("reverb", 0.5)
    out = [buf[i] + mix * wet[i] * 0.5 for i in range(N)]
    # circular low-pass: warm up the state with the tail so the loop point stays continuous
    a = 1 - math.exp(-TAU * cfg.get("lp", 5000) / SR)
    y = 0.0
    for v in out[-int(0.3 * SR):]:
        y += a * (v - y)
    for i in range(N):
        y += a * (out[i] - y)
        out[i] = y
    return finish(out, PEAK_MUSIC, fades=False)


MUSIC = {
    "snd-crystal-chime": dict(secs=40, prog=PROG_D, shift=7, padkind="glass", voice=chime_v, arp_d=1.8, octs=(24, 36),
                              step=0.5, rest=0.2, pattern="updown", arp=0.11, pad=0.07),
    "snd-synth-pluck": dict(secs=32, prog=PROG_B, padkind="saw", voice=lambda f, d: saw_pluck(f, d, 3500), arp_d=0.9,
                            octs=(24, 36), step=0.25, rest=0.25, pattern="updown", arp=0.1, pad=0.04, lp=3500),
    "snd-sci-fi-console": dict(secs=44, prog=PROG_C, padkind="glass", voice=scifi_v, arp_d=0.6, octs=(36, 48), step=0.75,
                               rest=0.5, pattern="rand", arp=0.05, pad=0.09, reverb=0.7, lp=4000),
    "snd-marimba": dict(secs=36, prog=PROG_A, shift=2, padkind="warm", voice=marimba_v, arp_d=1.2, octs=(24, 36),
                        step=0.375, rest=0.3, pattern="rand", arp=0.15, pad=0.07),
    "snd-kalimba": dict(secs=40, prog=PROG_D, shift=-2, padkind="soft", voice=kalimba_v, arp_d=1.4, octs=(24, 36),
                        step=0.5, rest=0.25, pattern="rand", arp=0.14, pad=0.08),
    "snd-warm-felt": dict(secs=44, prog=PROG_A, shift=-3, padkind="warm", voice=felt_v, arp_d=1.6, octs=(12, 24, 36),
                          step=1.0, rest=0.3, pattern="rand", arp=0.12, pad=0.09, lp=3000),
    "snd-temple-bell": dict(secs=44, prog=PROG_C, shift=5, padkind="soft", voice=bell_v, arp_d=2.2, octs=(24, 36),
                            step=2.0, rest=0.35, pattern="rand", arp=0.1, pad=0.09, reverb=0.6),
    "snd-cozy-pluck": dict(secs=36, prog=PROG_B, shift=3, padkind="warm", voice=ks_v, arp_d=1.2, octs=(24, 36), step=0.5,
                           rest=0.2, pattern="updown", arp=0.18, pad=0.07),
}

# ----------------------------------------------------------------------------- expansion packs (v2)
# Helpers for the second wave of packs. Everything is synthesized from math + seeded noise.
def L(*items):
    return layer(list(items))


def fseq(freqs, sd, wave="sq", duty=0.5, pwm=None, a=0.001, rel=0.004, tau=None, vol=1.0, vib=0.0, bits=None):
    """Continuous-phase tone whose pitch steps through `freqs` (one step every `sd` seconds)."""
    k = max(1, int(sd * SR))
    n = k * len(freqs)
    out = [0.0] * n
    ph = 0.0
    tot = n / SR
    for i in range(n):
        t = i / SR
        f = freqs[i // k]
        if vib:
            f *= 1 + vib * math.sin(TAU * 6 * t)
        ph += f / SR
        ph -= int(ph)
        if wave == "sq":
            du = duty + (pwm[1] * math.sin(TAU * pwm[0] * t) if pwm else 0.0)
            s = 1.0 if ph < du else -1.0
        elif wave == "tri":
            s = 4 * abs(ph - 0.5) - 1
        elif wave == "saw":
            s = 2 * ph - 1
        else:
            s = math.sin(TAU * ph)
        e = min(1.0, t / a, (tot - t) / rel)
        if tau:
            e *= math.exp(-t / tau)
        out[i] = s * e * vol
    return bitcrush(out, bits) if bits else out


def lfsr(d, rate, tau, short=False, vol=1.0):
    n = ns(d)
    k = max(1, int(SR / rate))
    pat = [R.choice((-1.0, 1.0)) for _ in range(31)]
    out = [0.0] * n
    v = 1.0
    for i in range(n):
        if i % k == 0:
            v = pat[(i // k) % 31] if short else R.choice((-1.0, 1.0))
        out[i] = v * math.exp(-i / (SR * tau)) * vol * min(1.0, i / 20.0)
    return out


def varispeed(x, sp):
    out = []
    pos = 0.0
    n = len(x)
    while pos < n - 1:
        i = int(pos)
        fr = pos - i
        out.append(x[i] * (1 - fr) + x[i + 1] * fr)
        pos += sp(len(out) / SR)
    return out


def bend(x, cents, tt):
    return varispeed(x, lambda t: 2 ** (cents / 1200.0 * math.exp(-t / tt)))


def trem(x, rate, depth):
    return [v * (1 - depth + depth * math.sin(TAU * rate * i / SR)) for i, v in enumerate(x)]


def fadeio(x, a=0.004, r=0.02):
    n = len(x)
    na, nr = min(int(a * SR), n // 3), min(int(r * SR), n // 3)
    x = list(x)
    for i in range(na):
        x[i] *= i / na
    for i in range(nr):
        x[n - 1 - i] *= i / nr
    return x


def sat(x, k):
    return [math.tanh(k * v) for v in x]


def hiss(d, vol=1.0, lo=0):
    x = noise(d, a=0.01, tau=1e9, vol=vol)
    return hp(x, lo) if lo else x


def thump(f, d=0.12, vol=1.0, glide=1.8, tau=0.04):
    return tone(f, d, a=0.001, tau=tau, glide=glide, glide_t=0.02, vol=vol)


def click_n(fc, q=1.5, d=0.02, tau=0.004, vol=1.0):
    return bp(noise(d, a=0.0004, tau=tau, vol=vol), fc, q)


def stagger(v, notes, gap, d, gain_=1.0):
    return layer([(k * gap, v(hz(m), d), gain_) for k, m in enumerate(notes)])


def cascade(v, notes, gap, d, jitter=0.0):
    return layer([(k * gap + (R.random() * jitter), v(hz(m), d), 0.6 + 0.4 * R.random()) for k, m in enumerate(notes)])


# ---------------------------------------------------------------- chip family
def p_pulse_quest():
    v = lambda f, d: tone(f, d, "sq", a=0.001, tau=0.09, duty=0.125, rel=0.004)
    t = lambda f, d: tone(f, d, "tri", a=0.001, tau=0.2, rel=0.004)
    sel = [(k * 0.04, v(hz(m), 0.04), 1) for k, m in enumerate([76, 79, 76, 79])] + [(0.16, v(hz(84), 0.25), 1), (0.16, t(hz(60), 0.25), 0.9)]
    return dict(click=v(hz(88), 0.025), menu_open=phrase(v, [60, 64, 67, 72, 76], 0.045, 0.18),
                menu_close=phrase(v, [76, 72, 67, 64, 60], 0.045, 0.15), menu_select=layer(sel))


def p_chip_drums():
    kick = lambda: sweep(190, 45, 0.13, "tri", a=0.001, tau=0.07)
    snare = lambda: layer([(0, lfsr(0.12, 7000, 0.04), 0.9), (0, sweep(260, 150, 0.08, "tri", tau=0.03), 0.6)])
    hat = lambda: lfsr(0.05, 16000, 0.012, vol=0.7)
    bass = tone(hz(38), 0.25, "tri", a=0.001, tau=0.15)
    return dict(click=hat(),
                menu_open=L((0, kick(), 1), (0.1, snare(), 0.9), (0.2, hat(), 0.8), (0.25, kick(), 1)),
                menu_close=L((0, snare(), 0.9), (0.09, kick(), 1), (0.19, kick(), 0.8)),
                menu_select=L((0, kick(), 1), (0, bass, 0.8), (0.08, hat(), 0.8), (0.12, lfsr(0.16, 3000, 0.07, short=True), 0.9)))


def p_pc_speaker():
    sq = lambda fs, sd: fseq(fs, sd, "sq", a=0.0005, rel=0.003)
    return dict(click=sq([2400], 0.02),
                menu_open=sq([392, 523, 659, 784, 1047, 1319], 0.028),
                menu_close=sq([1319, 1047, 784, 659, 523, 392], 0.028),
                menu_select=L((0, sq([880], 0.06), 1), (0.09, sq([1760], 0.12), 1)))


def p_wavetable_pocket():
    tab = [R.randrange(16) for _ in range(32)]
    tab = [(a_ + b_) / 2 for a_, b_ in zip(tab, tab[1:] + tab[:1])]
    tab = [(round(v) - 7.5) / 8 for v in tab]

    def wt(f, d, tau=0.1):
        n = ns(d)
        ph = 0.0
        out = [0.0] * n
        for i in range(n):
            ph += f / SR
            ph -= int(ph)
            t = i / SR
            out[i] = tab[int(ph * 32)] * math.exp(-t / tau) * min(1.0, t / 0.001, (d - t) / 0.004)
        return out
    v = lambda m, d=0.1: wt(hz(m), d)
    return dict(click=wt(1500, 0.025, 0.02),
                menu_open=L(*[(k * 0.06, v(m), 1) for k, m in enumerate([62, 65, 69, 74])]),
                menu_close=L(*[(k * 0.06, v(m), 1) for k, m in enumerate([74, 69, 65, 62])]),
                menu_select=L((0, wt(hz(50), 0.2, 0.15), 0.9), (0.0, v(74, 0.08), 1), (0.09, v(81, 0.25), 1)))


def p_coin_slot():
    coin = lambda f, d=0.45: additive(f, d, [(1, 1.0, 0.22), (2.32, 0.7, 0.14), (4.25, 0.5, 0.09), (6.63, 0.3, 0.05)])
    ch = lambda f, d: tone(f, d, "sq", a=0.001, tau=0.1, duty=0.5, rel=0.004)
    slot = ratchet(4, 900, 2400)
    return dict(click=layer([(0, click_n(5000, 2.0, 0.02), 1), (0, coin(3300, 0.05), 0.6)]),
                menu_open=L((0, coin(2400), 1), (0.11, coin(3100, 0.3), 0.9)),
                menu_close=L((0, slot, 1)),
                menu_select=L((0, coin(2700, 0.3), 0.9), (0.06, ch(hz(83), 0.07), 0.8), (0.13, ch(hz(88), 0.28), 0.9)))


def p_sixteen_bit():
    v = lambda f, d: bitcrush(lp(tone(f, d, a=0.003, tau=0.18, fm=(2, 1.2, 0.12), h2=0.2), 5500), 8, 3)
    stab = lambda ms, d=0.3: layer([(k * 0.012, v(hz(m), d), 1) for k, m in enumerate(ms)])
    return dict(click=v(hz(79), 0.05),
                menu_open=L((0, stab([55, 62, 67]), 1), (0.12, stab([60, 67, 72, 76], 0.4), 1)),
                menu_close=L((0, stab([60, 67, 72, 76]), 1), (0.12, stab([55, 62, 67], 0.35), 0.9)),
                menu_select=L((0, stab([67, 71, 74, 79], 0.15), 1), (0.1, stab([72, 76, 79, 84], 0.45), 1)))


def p_pwm_tracker():
    def arp(chord, reps, sd=0.02, d_tau=0.18):
        fs = [hz(m) for _ in range(reps) for m in chord]
        return fseq(fs, sd, "sq", duty=0.5, pwm=(6, 0.35), tau=d_tau, rel=0.01)
    return dict(click=fseq([hz(81)], 0.03, "sq", pwm=(40, 0.4), tau=0.03),
                menu_open=layer([(0, arp([57, 60, 64], 4), 1), (0.24, arp([60, 64, 69], 5, 0.02, 0.25), 1)]),
                menu_close=layer([(0, arp([60, 64, 69], 4), 1), (0.24, arp([57, 60, 64], 5, 0.02, 0.25), 0.9)]),
                menu_select=layer([(0, fseq([hz(m) for m in [64, 67, 71, 76] * 2], 0.025, "sq", pwm=(7, 0.35), tau=0.3), 1)]))


# ---------------------------------------------------------------- keys / bells
def p_fm_bells():
    v = lambda f, d: layer([(0, tone(f, d, a=0.001, tau=0.55, fm=(3.5, 3.0, 0.25)), 1), (0, tone(f * 2.001, d, a=0.001, tau=0.2, fm=(1.4, 1.5, 0.1)), 0.35)])
    return dict(click=v(hz(96), 0.08),
                menu_open=stagger(v, [76, 83, 88], 0.09, 0.7),
                menu_close=stagger(v, [88, 83, 76], 0.09, 0.6),
                menu_select=L((0, v(hz(84), 0.5), 1), (0.1, v(hz(91), 0.8), 1)))


def rhodes_v(f, d):
    a = tone(f, d, a=0.002, tau=0.55, fm=(1, 1.3, 0.3), h2=0.1)
    b = tone(f * 1.004, d, a=0.002, tau=0.5, fm=(1, 1.1, 0.25))
    tine = tone(f * 7, min(d, 0.12), a=0.001, tau=0.02, vol=0.5)
    return layer([(0, a, 1), (0, b, 0.7), (0, tine, 0.4)])


def p_rhodes():
    stab = lambda ms, d=0.6, g=0.012: layer([(k * g, rhodes_v(hz(m), d), 1) for k, m in enumerate(ms)])
    return dict(click=rhodes_v(hz(84), 0.07),
                menu_open=L((0, stab([50, 57, 62, 65, 69]), 1), (0.15, stab([55, 59, 62, 65, 69], 0.7), 1)),
                menu_close=L((0, stab([55, 59, 62, 65, 69]), 1), (0.15, stab([48, 55, 60, 64, 71], 0.7), 0.9)),
                menu_select=stab([60, 64, 67, 71, 74], 0.9, 0.02))


def p_vibraphone():
    v = lambda f, d: trem(additive(f, d, [(1, 1.0, 0.9), (4, 0.35, 0.4), (10, 0.12, 0.15)], a=0.003), 5.5, 0.45)
    return dict(click=additive(hz(88), 0.06, [(1, 1.0, 0.04), (4, 0.4, 0.02)]),
                menu_open=stagger(v, [60, 64, 67, 71], 0.1, 0.9),
                menu_close=stagger(v, [71, 67, 64, 60], 0.1, 0.8),
                menu_select=L((0, v(hz(72), 0.8), 1), (0.12, v(hz(79), 1.0), 1)))


def p_glockenspiel():
    v = lambda f, d: layer([(0, additive(f, d, [(1, 1.0, 0.3), (2.0, 0.4, 0.2), (4.1, 0.25, 0.12), (6.3, 0.12, 0.08)], a=0.0008), 1),
                            (0, hp(noise(0.012, tau=0.003, vol=0.5), 5000), 0.7)])
    return dict(click=v(hz(100), 0.07),
                menu_open=layer([(k * 0.04, v(hz(m), 0.5), 1) for k, m in enumerate([84, 88, 91, 96, 100])]),
                menu_close=layer([(k * 0.04, v(hz(m), 0.45), 1) for k, m in enumerate([100, 96, 91, 88, 84])]),
                menu_select=L((0, v(hz(91), 0.4), 1), (0.0, v(hz(103), 0.4), 0.6), (0.1, v(hz(96), 0.7), 1), (0.1, v(hz(108), 0.5), 0.5)))


def p_drawbar_organ():
    parts = [(0.5, 0.8, 9), (1, 1.0, 9), (1.5, 0.7, 9), (2, 0.8, 9), (3, 0.4, 9), (4, 0.35, 9), (6, 0.2, 9), (8, 0.15, 9)]

    def v(f, d):
        x = trem(additive(f, d, parts, a=0.006, rel=0.04, vol=0.3), 6.5, 0.15)
        kc = click_n(f * 5, 1.2, 0.012, 0.003, 0.6)
        return layer([(0, x, 1), (0, kc, 0.5)])
    ch = lambda ms, d: layer([(0, v(hz(m), d), 1) for m in ms])
    return dict(click=v(hz(72), 0.05),
                menu_open=L((0, ch([48, 55, 60], 0.14), 1), (0.15, ch([53, 57, 65], 0.2), 1)),
                menu_close=L((0, ch([53, 57, 65], 0.14), 1), (0.15, ch([48, 55, 60], 0.2), 1)),
                menu_select=L((0, ch([48, 52, 55, 60], 0.18), 1), (0.2, ch([55, 59, 62, 67], 0.3), 1)))


def p_music_box():
    v = lambda f, d: additive(f, d, [(1, 1.0, 0.25), (3.0, 0.4, 0.12), (5.8, 0.2, 0.06)], a=0.001)
    crank = layer([(k * 0.035, click_n(1800 + k * 150, 3.0, 0.015, 0.004, 0.7), 1) for k in range(5)])
    pent = [84, 86, 88, 91, 93, 96, 98, 100]
    return dict(click=v(hz(96), 0.06),
                menu_open=L((0, crank, 0.6), (0.18, v(hz(84), 0.5), 1), (0.26, v(hz(88), 0.5), 1), (0.34, v(hz(91), 0.7), 1)),
                menu_close=L((0, v(hz(96), 0.5), 1), (0.1, v(hz(91), 0.5), 1), (0.2, v(hz(88), 0.7), 0.9)),
                menu_select=layer([(k * 0.06, v(hz(pent[(k * 3) % 8]), 0.45 if k < 4 else 0.9), 1) for k in range(5)]))


def p_harp():
    v = lambda f, d: lp(ks(f, d, tau=1.0, bright=0.4), 5500)
    up = [51, 55, 58, 63, 67, 70, 75]
    return dict(click=v(hz(87), 0.2),
                menu_open=stagger(v, up, 0.04, 0.9),
                menu_close=stagger(v, up[::-1], 0.04, 0.8),
                menu_select=L((0, v(hz(58), 0.9), 1), (0.0, v(hz(70), 0.9), 0.8), (0.09, v(hz(82), 1.0), 1)))


# ---------------------------------------------------------------- plucked strings
def p_nylon_guitar():
    v = lambda f, d: lp(ks(f, d, tau=0.7, bright=0.3), 3800)
    body = lambda: thump(120, 0.05, 0.5)
    return dict(click=layer([(0, v(hz(76), 0.12), 1), (0, body(), 0.4)]),
                menu_open=layer([(0, body(), 1)] + [(k * 0.022, v(hz(m), 0.9), 1) for k, m in enumerate([40, 47, 52, 55, 59, 64])]),
                menu_close=layer([(0, body(), 1)] + [(k * 0.022, v(hz(m), 0.8), 1) for k, m in enumerate([64, 59, 55, 52, 47, 40])]),
                menu_select=L((0, v(hz(64), 0.6), 1), (0.09, v(hz(71), 0.5), 1), (0.18, v(hz(76), 1.0), 1.1)))


def p_steel_string():
    v = lambda f, d: ks(f, d, tau=1.2, bright=0.9)
    pick = lambda: click_n(4500, 1.2, 0.01, 0.002, 0.5)
    seq = [43, 50, 55, 59, 62]
    return dict(click=layer([(0, pick(), 1), (0, v(hz(79), 0.1), 0.8)]),
                menu_open=layer([(k * 0.05, v(hz(m), 0.7), 1) for k, m in enumerate(seq)] + [(0, pick(), 1)]),
                menu_close=layer([(k * 0.05, v(hz(m), 0.6), 1) for k, m in enumerate(seq[::-1])] + [(0, pick(), 1)]),
                menu_select=L((0, v(hz(55), 0.5), 1), (0.0, v(hz(62), 0.5), 0.9), (0.0, v(hz(67), 0.5), 0.9), (0.08, v(hz(79), 1.1), 1.1)))


def p_koto():
    v = lambda f, d: ks(f, d, tau=0.5, bright=0.95)
    oshi = lambda m, d=0.8: bend(v(hz(m), d), 60, 0.12)
    hira = [62, 64, 65, 69, 70, 74, 76]
    return dict(click=layer([(0, click_n(3000, 3.0, 0.01, 0.003, 0.7), 1), (0, v(hz(81), 0.1), 0.8)]),
                menu_open=stagger(v, [hira[0], hira[2], hira[3], hira[5], hira[6]], 0.06, 0.7),
                menu_close=stagger(v, [hira[6], hira[5], hira[3], hira[2], hira[0]], 0.06, 0.6),
                menu_select=L((0, v(hz(62), 0.5), 1), (0.1, oshi(70, 1.0), 1.1)))


def p_shamisen():
    def v(f, d):
        x = sat(ks(f, d, tau=0.25, bright=1.0), 2.5)
        return x
    bachi = lambda: layer([(0, thump(150, 0.05, 0.7, glide=1.3, tau=0.015), 1), (0, click_n(2500, 1.0, 0.03, 0.01, 0.8), 1)])
    return dict(click=bachi(),
                menu_open=L((0, bachi(), 0.8), (0.06, v(hz(57), 0.3), 1), (0.14, v(hz(64), 0.3), 1), (0.22, v(hz(69), 0.5), 1)),
                menu_close=L((0, v(hz(69), 0.3), 1), (0.08, v(hz(64), 0.3), 1), (0.16, v(hz(57), 0.3), 1), (0.16, bachi(), 0.8)),
                menu_select=L((0, bachi(), 1), (0.0, v(hz(52), 0.4), 1), (0.07, v(hz(59), 0.35), 1), (0.07, bachi(), 0.7), (0.15, v(hz(64), 0.7), 1)))


def p_pizzicato():
    v = lambda f, d: ks(f, d, tau=0.1, bright=0.6)
    return dict(click=v(hz(84), 0.07),
                menu_open=stagger(v, [67, 72, 76, 79, 84], 0.055, 0.2),
                menu_close=stagger(v, [84, 79, 76, 72, 67], 0.055, 0.2),
                menu_select=L((0, stagger(v, [60, 67, 76], 0.0, 0.25), 1), (0.12, stagger(v, [64, 72, 79], 0.0, 0.3), 1)))


# ---------------------------------------------------------------- tuned percussion
def p_steel_drum():
    def v(f, d=0.5):
        a = additive(f, d, [(1, 1.0, 0.4), (2, 0.8, 0.3), (3, 0.4, 0.2), (4.01, 0.15, 0.12)], a=0.002)
        b = additive(f * 1.005, d, [(1, 0.6, 0.4), (2, 0.5, 0.3)], a=0.002)
        return layer([(0, a, 1), (0, b, 0.6)])
    roll = layer([(k * 0.035, v(hz(72), 0.12), 0.8) for k in range(5)] + [(0.18, v(hz(79), 0.7), 1)])
    return dict(click=v(hz(84), 0.07),
                menu_open=L((0, v(hz(67)), 1), (0.09, v(hz(72)), 1), (0.18, v(hz(76), 0.6), 1)),
                menu_close=L((0, v(hz(76)), 1), (0.09, v(hz(72)), 1), (0.18, v(hz(67), 0.6), 1)),
                menu_select=roll)


def p_handpan():
    def v(f, d=0.9):
        a = additive(f, d, [(1, 1.0, 0.8), (2, 0.45, 0.5), (3, 0.2, 0.3)], a=0.004)
        return layer([(0, a, 1), (0, thump(f * 0.5, 0.06, 0.3, 1.3, 0.02), 0.8)])
    d3 = [50, 57, 58, 60, 62, 65, 69]
    return dict(click=v(hz(74), 0.1),
                menu_open=echo(L((0, v(hz(d3[0] + 12)), 1), (0.1, v(hz(d3[4] + 12)), 1), (0.2, v(hz(d3[6] + 12), 0.7), 1)), 0.15, 0.25, 2),
                menu_close=echo(L((0, v(hz(d3[6] + 12)), 1), (0.1, v(hz(d3[4] + 12)), 1), (0.2, v(hz(d3[0] + 12), 0.7), 1)), 0.15, 0.25, 2),
                menu_select=echo(L((0, v(hz(62), 0.8), 1), (0.12, v(hz(69), 0.9), 1)), 0.2, 0.3, 2))


def p_gamelan():
    def v(f, d=0.6):
        a = additive(f, d, [(1, 1.0, 0.7), (2.4, 0.5, 0.4), (3.9, 0.3, 0.25), (5.6, 0.18, 0.15)], a=0.002)
        b = additive(f * 1.007, d, [(1, 0.8, 0.7), (2.4, 0.4, 0.4)], a=0.002)
        return layer([(0, a, 1), (0, b, 0.7)])
    sl = lambda k: 293.7 * 2 ** (k * 240 / 1200.0)
    gong = additive(65, 1.4, [(1, 1.0, 1.5), (2.08, 0.7, 1.1), (3.2, 0.5, 0.8)], a=0.02)
    return dict(click=v(sl(8), 0.1),
                menu_open=layer([(k * 0.08, v(sl(k + 1), 0.6), 1) for k in range(4)]),
                menu_close=layer([(k * 0.08, v(sl(5 - k), 0.5), 1) for k in range(4)]),
                menu_select=L((0, gong, 0.8), (0.0, v(sl(5), 0.7), 1), (0.15, v(sl(7), 0.9), 1)))


def p_xylophone():
    v = lambda f, d=0.22: layer([(0, additive(f, d, [(1, 1.0, 0.08), (3, 0.5, 0.04), (6, 0.2, 0.02)], a=0.0005), 1),
                                 (0, click_n(f * 1.2, 2.0, 0.01, 0.003, 0.5), 0.7)])
    return dict(click=v(hz(96), 0.08),
                menu_open=layer([(k * 0.05, v(hz(m)), 1) for k, m in enumerate([72, 76, 79, 84, 88])]),
                menu_close=layer([(k * 0.05, v(hz(m)), 1) for k, m in enumerate([88, 84, 79, 76, 72])]),
                menu_select=L((0, v(hz(79)), 1), (0.05, v(hz(79)), 0.7), (0.1, v(hz(91), 0.4), 1.1)))


def p_tubular_bells():
    v = lambda f, d=1.2: layer([(0, additive(f, d, [(1, 1.0, 1.3), (2.43, 0.5, 0.9), (3.9, 0.3, 0.5), (5.3, 0.2, 0.3)], a=0.001), 1),
                                (0, click_n(f * 3, 1.0, 0.02, 0.006, 0.6), 0.6)])
    return dict(click=v(hz(72), 0.08),
                menu_open=layer([(k * 0.2, v(hz(m), 1.0), 1) for k, m in enumerate([60, 64, 67])]),
                menu_close=layer([(k * 0.2, v(hz(m), 1.0), 1) for k, m in enumerate([67, 64, 60])]),
                menu_select=L((0, v(hz(55), 1.6), 1), (0.0, v(hz(62), 1.6), 0.7), (0.25, v(hz(67), 1.5), 1)))


def p_singing_bowl():
    def v(f, d=1.4):
        x = additive(f, d, [(1, 1.0, 1.3), (2.71, 0.5, 0.9), (5.2, 0.2, 0.5)], a=0.03)
        y = additive(f * 1.006, d, [(1, 0.8, 1.3)], a=0.03)
        return trem(layer([(0, x, 1), (0, y, 0.8)]), 4.2, 0.3)
    strike = lambda f: layer([(0, thump(f, 0.02, 0.4, 1.0, 0.005), 1), (0, click_n(f * 3, 2.0, 0.02, 0.004, 0.5), 0.5)])
    return dict(click=layer([(0, strike(900), 1), (0, additive(900, 0.12, [(1, 1.0, 0.06), (2.71, 0.4, 0.03)]), 0.8)]),
                menu_open=L((0, v(hz(57)), 1)),
                menu_close=L((0, v(hz(52), 1.2), 1)),
                menu_select=L((0, strike(500), 1), (0, v(hz(64), 1.5), 1), (0.4, v(hz(71), 1.2), 0.5)))


def p_wind_chimes():
    v = lambda f, d=0.9: additive(f, d, [(1, 1.0, 0.45), (2.76, 0.3, 0.2), (5.4, 0.12, 0.1)], a=0.0015)
    pent = [79, 81, 84, 86, 88, 91, 93, 96]
    idx = [R.randrange(8) for _ in range(9)]
    idx2 = [R.randrange(8) for _ in range(7)]
    opn = layer([(k * 0.055 + R.random() * 0.02, v(hz(pent[(i + k) % 8]), 0.7), 0.5 + 0.5 * R.random()) for k, i in enumerate(sorted(idx))])
    cls = layer([(k * 0.07 + R.random() * 0.02, v(hz(pent[i]), 0.5), 1.0 - k * 0.1) for k, i in enumerate(sorted(idx2, reverse=True))])
    return dict(click=v(hz(93), 0.1),
                menu_open=opn, menu_close=cls,
                menu_select=L((0, v(hz(84), 1.0), 1), (0.05, v(hz(91), 1.0), 0.8), (0.1, v(hz(96), 1.1), 0.8)))


def p_bamboo():
    def knock(f, d=0.12):
        return layer([(0, bp(noise(d, a=0.0005, tau=0.015), f, 14), 2.5), (0, tone(f, d, a=0.0005, tau=0.03, h2=0.2), 0.6)])

    def flute(f, d=0.45):
        br = bp(noise(d, a=0.04, tau=0.3), f, 6)
        t = tone(f, d, a=0.05, tau=0.3, vib=0.006, vib_rate=5, rel=0.08, h2=0.05)
        return layer([(0, t, 1), (0, br, 1.2)])
    return dict(click=knock(1100, 0.05),
                menu_open=L((0, knock(700), 1), (0.1, knock(950), 1), (0.2, knock(1250, 0.2), 1)),
                menu_close=L((0, knock(1250), 1), (0.1, knock(950), 1), (0.2, knock(700, 0.2), 1)),
                menu_select=L((0, knock(850), 1), (0.1, flute(hz(81), 0.6), 1)))


def p_hand_drum():
    ghe = lambda: tone(95, 0.3, a=0.001, tau=0.1, glide=1.5, glide_t=0.08, h2=0.2)
    na = lambda: layer([(0, additive(420, 0.25, [(1, 1.0, 0.1), (2, 0.5, 0.07), (3, 0.3, 0.05)], a=0.0005), 1), (0, click_n(2500, 2.0, 0.01, 0.003, 0.6), 0.6)])
    tin = lambda: additive(630, 0.35, [(1, 1.0, 0.2), (2, 0.6, 0.12), (3, 0.35, 0.08)], a=0.0005)
    return dict(click=layer([(0, click_n(3000, 2.0, 0.02, 0.005, 1.0), 1), (0, additive(630, 0.04, [(1, 1.0, 0.02)]), 0.5)]),
                menu_open=L(*[(k * 0.05, na(), 0.7 + k * 0.1) for k in range(4)], (0.22, tin(), 1)),
                menu_close=L((0, tin(), 1), (0.12, na(), 0.8), (0.2, ghe(), 1)),
                menu_select=L((0, ghe(), 1), (0.0, na(), 0.8), (0.1, tin(), 1)))


# ---------------------------------------------------------------- mechanical
def p_blue_switch():
    def key(fc=3800, th=130, d=0.09):
        jacket = click_n(fc, 2.5, 0.012, 0.002, 1.0)
        leaf = additive(5200, 0.05, [(1, 1.0, 0.008), (1.5, 0.5, 0.005)])
        bottom = layer([(0.012, click_n(1500, 1.5, 0.04, 0.012, 1.0), 1), (0.012, thump(th, 0.05, 0.8, 1.2, 0.015), 1)])
        return layer([(0, jacket, 1), (0, leaf, 0.4), (0, bottom, 0.9)])
    return dict(click=key(3800, 150, 0.06),
                menu_open=L((0, key(3200, 140), 1), (0.09, key(4200, 170), 1)),
                menu_close=L((0, key(4200, 170), 1), (0.09, key(3000, 130), 0.9)),
                menu_select=L((0, key(2800, 90), 1), (0.12, key(3600, 120), 1.2)))


def p_thock_board():
    def th(f=95, d=0.12, vol=1.0):
        body = thump(f, d, 1.0, 1.4, 0.05)
        top = lp(noise(0.04, a=0.0005, tau=0.012), 1800)
        return layer([(0, body, 1), (0, top, 0.8), (0, tone(f * 3.1, 0.06, a=0.001, tau=0.02), 0.2)])
    return dict(click=th(140, 0.06),
                menu_open=L((0, th(120), 1), (0.08, th(150), 1)),
                menu_close=L((0, th(150), 1), (0.08, th(110), 1)),
                menu_select=L((0, th(80, 0.18), 1.2), (0.1, th(115), 0.8)))


def p_camera_shutter():
    def shut(vol=1.0):
        a = layer([(0, click_n(2600, 1.5, 0.012, 0.003), 1), (0.0, thump(300, 0.03, 0.6, 1.2, 0.01), 1)])
        b = layer([(0.035, click_n(2000, 1.5, 0.012, 0.003), 0.8), (0.035, thump(240, 0.03, 0.5, 1.1, 0.01), 0.8)])
        return layer([(0, a, vol), (0, b, vol)])
    beep = lambda: tone(2200, 0.06, a=0.002, tau=0.05)
    return dict(click=shut(),
                menu_open=L((0, ratchet(5, 1800, 3200), 0.8), (0.4, shut(), 1)),
                menu_close=L((0, shut(), 1), (0.12, ratchet(4, 2800, 1500, rev=True), 0.7)),
                menu_select=L((0, beep(), 0.7), (0.09, beep(), 0.7), (0.2, shut(), 1.2)))


def p_rotary_dial():
    def tk(fc=1600):
        return click_n(fc, 2.0, 0.02, 0.004, 1.0)
    spin = lambda n_, g0, rev: layer([(sum(g0 * (1.18 if not rev else 0.85) ** j for j in range(k)), tk(1400 + 50 * k), 1) for k in range(n_)])
    ring = layer([(k * 0.04, tone(1200 if k % 2 == 0 else 1500, 0.05, a=0.001, tau=0.03, h2=0.4), 1) for k in range(8)] + [(0, additive(2400, 0.4, [(1, 1.0, 0.2)]), 0.2)])
    return dict(click=tk(1800),
                menu_open=spin(8, 0.03, False),
                menu_close=L((0, thump(180, 0.06, 1.0, 1.3, 0.02), 1), (0, tk(900), 1), (0.05, tk(1400), 0.8)),
                menu_select=ring)


def p_cassette_deck():
    clunk = lambda: layer([(0, thump(110, 0.08, 1.0, 1.3, 0.025), 1), (0, click_n(1800, 1.5, 0.03, 0.008), 1)])
    spin = lambda up: layer([(0, bp_sweep(hiss(0.35, 0.8), 800 if up else 3200, 3200 if up else 800, 1.0), 0.9),
                             (0, sweep(60 if up else 140, 140 if up else 60, 0.35, "saw", tau=None, a=0.04, rel=0.06, vol=0.2), 0.7)])
    return dict(click=layer([(0, click_n(2600, 1.8, 0.02, 0.004), 1)]),
                menu_open=L((0, clunk(), 1), (0.08, spin(True), 0.6)),
                menu_close=L((0, spin(False), 0.6), (0.3, clunk(), 1)),
                menu_select=L((0, clunk(), 0.9), (0.1, click_n(3500, 1.0, 0.02, 0.005), 0.8), (0.14, tone(1100, 0.05, a=0.002, tau=0.03), 0.4)))


def p_toggle_panel():
    def tog(f=1800, vol=1.0):
        snap = click_n(f, 1.5, 0.015, 0.003)
        spring = additive(f * 1.3, 0.12, [(1, 0.5, 0.05), (2.7, 0.3, 0.03)])
        body = thump(160, 0.05, 0.7, 1.2, 0.012)
        return layer([(0, snap, vol), (0.008, spring, 0.4 * vol), (0.012, body, 0.8 * vol)])
    return dict(click=tog(2400, 0.8),
                menu_open=L((0, tog(1700), 1), (0.12, tog(2200), 1)),
                menu_close=L((0, tog(2200), 1), (0.12, tog(1500), 1)),
                menu_select=L((0, tog(1300), 1.2), (0.08, tog(1900, 0.8), 0.9), (0.08, tone(1760, 0.1, a=0.002, tau=0.06), 0.3)))


def p_office_desk():
    pen = lambda: L((0, click_n(3600, 3.0, 0.012, 0.003), 1), (0.12, click_n(2800, 3.0, 0.012, 0.003), 0.8))
    staple = lambda: L((0, thump(200, 0.04, 1.0, 1.2, 0.012), 0.8), (0, click_n(3000, 1.2, 0.03, 0.006), 1), (0.03, additive(4200, 0.1, [(1, 1.0, 0.02), (1.7, 0.5, 0.015)]), 0.5))
    clip = lambda: L((0, thump(130, 0.05, 1.0, 1.3, 0.015), 1), (0, click_n(1200, 0.8, 0.05, 0.015), 1))
    return dict(click=click_n(3600, 3.0, 0.012, 0.003),
                menu_open=pen(),
                menu_close=L((0, clip(), 1), (0.1, clip(), 0.7)),
                menu_select=L((0, staple(), 1.1), (0.14, click_n(2000, 1.0, 0.04, 0.01), 0.4)))


def p_clockwork():
    tick = lambda f=2200: layer([(0, click_n(f, 3.0, 0.015, 0.003), 1), (0, additive(f * 0.5, 0.03, [(1, 1.0, 0.008)]), 0.5)])
    wind = layer([(k * 0.032, click_n(1500 + 90 * k, 2.5, 0.012, 0.003), 1) for k in range(10)])
    ding = additive(2640, 0.6, [(1, 1.0, 0.3), (2.76, 0.4, 0.15), (5.4, 0.15, 0.07)])
    return dict(click=tick(2600),
                menu_open=L((0, wind, 1), (0.34, tick(2200), 1)),
                menu_close=L((0, tick(2200), 1), (0.12, tick(1700), 1), (0.24, tick(1300), 0.8)),
                menu_select=L((0, tick(2200), 1), (0.12, ding, 0.7)))


def p_metal_clank():
    def clank(f=700, d=0.5, vol=1.0):
        x = additive(f, d, [(1, 1.0, 0.12), (2.7, 0.7, 0.08), (4.8, 0.5, 0.05), (7.3, 0.3, 0.03)], a=0.0005)
        return layer([(0, x, vol), (0, click_n(f * 3, 1.0, 0.02, 0.004), 0.9 * vol)])
    return dict(click=clank(1300, 0.08, 0.8),
                menu_open=L((0, clank(600, 0.3), 1), (0.1, clank(850, 0.4), 1)),
                menu_close=L((0, clank(850, 0.3), 1), (0.1, clank(500, 0.4), 1)),
                menu_select=L((0, clank(420, 0.6), 1.1), (0.12, clank(630, 0.7), 1)))


def p_blade_ring():
    ring = lambda f=3600, d=0.8: additive(f, d, [(1, 1.0, 0.5), (2.01, 0.4, 0.3), (3.5, 0.2, 0.15)], a=0.001)
    shing = lambda up: bp_sweep(noise(0.25, a=0.02, tau=0.2), 2500 if up else 8000, 8000 if up else 2500, 1.2)
    return dict(click=additive(4200, 0.08, [(1, 1.0, 0.03), (2.01, 0.4, 0.02)]),
                menu_open=L((0, shing(True), 1), (0.2, ring(3800, 0.5), 0.6)),
                menu_close=L((0, shing(False), 1), (0.22, thump(180, 0.05, 1.0, 1.2, 0.015), 0.9), (0.22, click_n(1800, 1.5, 0.02, 0.005), 0.8)),
                menu_select=L((0, click_n(5000, 1.0, 0.012, 0.003), 1), (0, ring(3300, 1.0), 1), (0, ring(4400, 0.8), 0.6)))


def p_abacus_beads():
    bead = lambda f=2300, vol=1.0: layer([(0, tone(f, 0.04, a=0.0004, tau=0.008, h2=0.3), vol), (0, click_n(f * 0.8, 3.0, 0.015, 0.003), vol)])
    slide = lambda up, n_=6: layer([(k * 0.03 + R.random() * 0.01, bead((1700 + k * 150) if up else (2600 - k * 150), 0.8), 1) for k in range(n_)])
    return dict(click=bead(2600),
                menu_open=slide(True),
                menu_close=slide(False),
                menu_select=L((0, slide(True, 4), 0.8), (0.15, bead(3000), 1), (0.17, bead(3000), 0.7)))


# ---------------------------------------------------------------- foley
def p_cardboard():
    def scrape(d, fc=700, up=True):
        x = lp(noise(d, a=0.01, tau=1e9), 2200)
        x = [v * (0.5 + 0.5 * math.sin(TAU * 38 * i / SR + R.random())) for i, v in enumerate(x)]
        return fadeio(bp_sweep(x, fc, fc * 2.5, 1.0) if up else bp_sweep(x, fc * 2.5, fc, 1.0), 0.02, 0.05)
    thud = lambda f=130: layer([(0, thump(f, 0.1, 0.9, 1.3, 0.03), 1), (0, lp(noise(0.06, tau=0.015), 900), 0.9)])
    return dict(click=layer([(0, thud(200), 0.8), (0, lp(noise(0.03, tau=0.006), 1500), 0.8)]),
                menu_open=L((0, scrape(0.3, 600, True), 1), (0.3, thud(150), 0.8)),
                menu_close=L((0, thud(150), 0.8), (0.05, scrape(0.25, 600, False), 1)),
                menu_select=L((0, thud(110), 1.0), (0.1, thud(170), 0.8)))


def p_bubble_wrap():
    def pop(f=1400, vol=1.0):
        return layer([(0, bp(noise(0.03, a=0.0002, tau=0.006), f, 3.0), 2.0 * vol), (0, tone(f * 0.7, 0.03, a=0.0003, tau=0.007), 0.4 * vol),
                      (0.004, lp(noise(0.05, tau=0.012), 800), 0.4 * vol)])
    return dict(click=pop(1800, 0.9),
                menu_open=L((0, pop(1100), 1), (0.07, pop(1400), 0.9), (0.12, pop(1700), 1), (0.2, pop(2000), 0.9)),
                menu_close=L((0, pop(1900), 1), (0.09, pop(1500), 0.9), (0.2, pop(1100), 1)),
                menu_select=L((0, pop(900, 1.2), 1), (0.0, pop(1300), 0.7), (0.07, pop(2100, 0.8), 0.7)))


def p_ice_cubes():
    clink = lambda f, d=0.4: layer([(0, additive(f, d, [(1, 1.0, 0.08), (2.3, 0.5, 0.05), (4.1, 0.2, 0.03)], a=0.0005), 1), (0, click_n(f * 1.5, 1.5, 0.01, 0.002), 0.5)])

    def pour(d=0.35):
        x = bp_sweep(hiss(d, 1.0), 1500, 3800, 1.2)
        x = [v * (0.6 + 0.4 * math.sin(TAU * 22 * i / SR + math.sin(i / 90.0))) for i, v in enumerate(x)]
        return fadeio(x, 0.02, 0.08)
    crack = lambda: layer([(k * 0.02 + R.random() * 0.01, click_n(4500 + R.randrange(2000), 3.0, 0.006, 0.0015), 0.8) for k in range(4)])
    return dict(click=clink(3600, 0.08),
                menu_open=L((0, pour(), 0.7), (0.3, clink(2800), 0.8), (0.4, crack(), 0.5)),
                menu_close=L((0, clink(2400, 0.4), 1), (0.04, crack(), 0.6)),
                menu_select=L((0, clink(3000), 1), (0.09, clink(4000, 0.6), 1)))


def p_rain_window():
    drip = lambda f=2200, vol=1.0: layer([(0, click_n(f * 1.3, 2.0, 0.008, 0.0015), 0.8 * vol), (0, tone(f, 0.04, a=0.0004, tau=0.01, glide=0.7, glide_t=0.01), 0.5 * vol)])
    pit = lambda n_, span: layer([(R.random() * span, drip(1500 + R.randrange(1800), 0.5 + 0.5 * R.random()), 1) for _ in range(n_)])
    bed = lambda d, g: gain(fadeio(hp(lp(hiss(d, 1.0), 6000), 2500), 0.05, 0.1), g)
    return dict(click=drip(2400),
                menu_open=L((0, bed(0.45, 0.12), 1), (0.0, pit(9, 0.4), 1)),
                menu_close=L((0, bed(0.4, 0.1), 1), (0.0, layer([(k * 0.06 * (1 + k * 0.3), drip(2000 - k * 150), 1.0 - k * 0.1) for k in range(6)]), 1)),
                menu_select=L((0, drip(2200), 1), (0.1, drip(3000), 1), (0.12, echo(tone(3000, 0.05, a=0.001, tau=0.02), 0.07, 0.3, 3), 0.3)))


def p_jazz_brushes():
    def swish2(d=0.3, up=True):
        x = bp_sweep(noise(d, a=0.05, tau=1e9), 3000, 5200, 0.7)
        n = len(x)
        return [v * math.sin(math.pi * (i / n if up else 1 - i / n) ** 0.7) ** 1.5 for i, v in enumerate(x)]
    tap = lambda: layer([(0, click_n(3200, 0.8, 0.04, 0.012), 1), (0, thump(220, 0.03, 0.4, 1.0, 0.01), 0.5)])
    kick = lambda: thump(70, 0.15, 1.0, 1.8, 0.07)
    return dict(click=tap(),
                menu_open=L((0, swish2(0.3, True), 1), (0.3, tap(), 0.9)),
                menu_close=L((0, tap(), 0.9), (0.04, swish2(0.3, False), 1)),
                menu_select=L((0, kick(), 1), (0.08, tap(), 0.8), (0.16, swish2(0.2, True), 0.7), (0.2, tap(), 1)))


# ---------------------------------------------------------------- nature
def chirp(f0, f1, d, vib=0.0, vol=1.0):
    n = ns(d)
    out = [0.0] * n
    ph = 0.0
    for i in range(n):
        t = i / SR
        u = t / d
        f = (f0 + (f1 - f0) * u) * (1 + vib * math.sin(TAU * 30 * t))
        ph += f / SR
        e = math.sin(math.pi * u) ** 0.6
        out[i] = (math.sin(TAU * ph) + 0.15 * math.sin(2 * TAU * ph)) * e * vol
    return out


def p_songbird():
    tw = lambda f, d=0.09: chirp(f, f * 1.35, d, 0.01)
    trill = layer([(k * 0.045, chirp(3800, 3000, 0.04, 0.0), 1) for k in range(7)])
    return dict(click=chirp(4200, 5200, 0.03),
                menu_open=L((0, tw(2800), 1), (0.12, tw(3400), 1), (0.24, tw(4000, 0.14), 1)),
                menu_close=L((0, chirp(4000, 2800, 0.12), 1), (0.17, chirp(3300, 2300, 0.16), 0.8)),
                menu_select=L((0, tw(3000, 0.07), 1), (0.1, tw(3000, 0.07), 1), (0.22, trill, 0.9)))


def p_frog_pond():
    def ribbit(f=170, d=0.16):
        n = ns(d)
        out = [0.0] * n
        ph = 0.0
        for i in range(n):
            t = i / SR
            ph += f * (1 + 0.3 * math.sin(math.pi * t / d)) / SR
            ph -= int(ph)
            out[i] = (2 * ph - 1) * (0.5 + 0.5 * math.sin(TAU * 45 * t)) * math.sin(math.pi * t / d)
        return lp(out, 1400)
    tree = lambda f=2400: layer([(k * 0.035, tone(f, 0.03, a=0.002, tau=0.015, glide=1.3, glide_t=0.01), 1) for k in range(3)])
    plop = lambda: tone(500, 0.1, a=0.001, tau=0.04, glide=0.5, glide_t=0.03)
    return dict(click=tree(2800),
                menu_open=L((0, ribbit(150, 0.14), 1), (0.17, ribbit(190, 0.18), 1)),
                menu_close=L((0, ribbit(200, 0.14), 1), (0.17, ribbit(150, 0.2), 0.9)),
                menu_select=L((0, plop(), 0.8), (0.1, ribbit(180, 0.12), 1), (0.25, tree(2300), 0.8)))


def p_crickets():
    def cr(f=4400, n_=3, vol=1.0):
        out = []
        for k in range(n_):
            seg = tone(f, 0.03, a=0.004, tau=1e9, rel=0.006)
            seg = [v * (0.5 + 0.5 * math.sin(TAU * 110 * i / SR)) for i, v in enumerate(seg)]
            out.append((k * 0.04, seg, vol))
        return layer(out)
    return dict(click=cr(4600, 1, 0.8),
                menu_open=L((0, cr(4200, 2), 1), (0.12, cr(4400, 3), 1), (0.28, cr(4700, 4), 1)),
                menu_close=L((0, cr(4700, 4), 1), (0.22, cr(4300, 3, 0.7), 1), (0.4, cr(4100, 1, 0.5), 1)),
                menu_select=L((0, cr(4500, 6), 1), (0.3, cr(4500, 2), 1), (0.42, cr(4500, 2), 1)))


def p_night_owl():
    def hoo(f=420, d=0.35, drop_=0.82):
        n = ns(d)
        out = [0.0] * n
        ph = 0.0
        for i in range(n):
            t = i / SR
            u = t / d
            fr = f * (1 - (1 - drop_) * u ** 2)
            ph += fr / SR
            e = math.sin(math.pi * min(1.0, u * 1.0)) ** 1.2
            out[i] = (math.sin(TAU * ph) + 0.12 * math.sin(2 * TAU * ph)) * e
        br = gain(lp(noise(d, a=0.05, tau=0.2), 900), 0.12)
        return [a + b for a, b in zip(out, br)]
    flutter = layer([(k * 0.02, lp(noise(0.03, a=0.005, tau=0.01), 1500), 1) for k in range(5)])
    return dict(click=hoo(500, 0.07, 0.9),
                menu_open=L((0, hoo(380, 0.25), 1), (0.3, hoo(430, 0.4), 1)),
                menu_close=L((0, hoo(430, 0.3), 1), (0.35, hoo(340, 0.45, 0.75), 0.9)),
                menu_select=L((0, flutter, 0.4), (0.1, hoo(450, 0.25), 1), (0.4, hoo(450, 0.25), 1)))


def p_tide_swell():
    def wave(d=0.7, peak=0.5, lo=300, hi=2400):
        x = noise(d, a=0.01, tau=1e9)
        out = []
        y = 0.0
        n = len(x)
        for i, v in enumerate(x):
            u = i / n
            e = (u / peak) if u < peak else ((1 - u) / (1 - peak)) ** 1.5
            fc = lo + (hi - lo) * e
            a_ = 1 - math.exp(-TAU * fc / SR)
            y += a_ * (v - y)
            out.append(y * e)
        return out
    return dict(click=layer([(0, thump(700, 0.05, 0.8, 0.6, 0.015), 1), (0, lp(noise(0.03, tau=0.01), 2000), 0.5)]),
                menu_open=wave(0.7, 0.75),
                menu_close=wave(0.7, 0.25),
                menu_select=L((0, wave(0.45, 0.6, 400, 3000), 0.8), (0.4, thump(900, 0.1, 0.7, 0.55, 0.04), 0.7), (0.5, thump(1300, 0.1, 0.5, 0.55, 0.04), 0.6)))


# ---------------------------------------------------------------- sci-fi / digital
def p_hud_blips():
    v = lambda f, d=0.03: lp(tone(f, d, a=0.001, tau=0.02, rel=0.004, h3=0.2), 6000)
    data = lambda fs, g=0.032: layer([(k * g, v(f), 1) for k, f in enumerate(fs)])
    r = lambda: [R.choice([1200, 1500, 1800, 2100, 2400, 3000]) for _ in range(7)]
    return dict(click=v(2600, 0.02),
                menu_open=data(sorted(r())),
                menu_close=data(sorted(r(), reverse=True)),
                menu_select=L((0, data([1800, 1800]), 1), (0.09, tone(3200, 0.18, a=0.002, tau=0.1, vib=0.01, vib_rate=14), 1)))


def p_sonar_ping():
    ping = lambda f=1300, d=0.22: tone(f, d, a=0.004, tau=0.07, glide=1.04, glide_t=0.02)
    return dict(click=tone(1800, 0.05, a=0.002, tau=0.02),
                menu_open=echo(ping(1100), 0.17, 0.4, 3),
                menu_close=echo(ping(700), 0.15, 0.35, 3),
                menu_select=L((0, ping(1500, 0.15), 1), (0.12, echo(ping(1500, 0.15), 0.14, 0.4, 3), 1)))


def p_geiger():
    tk = lambda: layer([(0, click_n(2500 + R.randrange(1500), 1.5, 0.008, 0.0015), 1)])
    burst = lambda n_, span, f=1.0: layer([(sorted([R.random() * span for _ in range(n_)])[k], tk(), 0.6 + 0.4 * R.random()) for k in range(n_)])
    def ramp(n_, span, up):
        ts = [span * ((k / n_) ** (0.6 if up else 1.6)) for k in range(n_)]
        return layer([(t, tk(), 1.0) for t in (ts if up else [span - t for t in ts])])
    return dict(click=layer([(0, click_n(3000, 1.5, 0.01, 0.002), 1)]),
                menu_open=ramp(14, 0.45, True),
                menu_close=ramp(14, 0.45, False),
                menu_select=L((0, burst(10, 0.18), 1), (0.2, tone(1500, 0.18, a=0.003, tau=0.1), 0.8)))


def p_analog_sweeps():
    def pad(f, d, c0, c1):
        x = layer([(0, tone(f, d, "saw", a=0.01, tau=1e9, rel=0.04), 1), (0, tone(f * 1.006, d, "saw", a=0.01, tau=1e9, rel=0.04), 1)])
        return fadeio(bp_sweep(x, c0, c1, 3.0), 0.01, 0.04)
    return dict(click=bp(layer([(0, tone(220, 0.05, "saw", a=0.001, tau=0.02), 1)]), 1400, 3.0),
                menu_open=pad(110, 0.5, 250, 5000),
                menu_close=pad(110, 0.5, 5000, 250),
                menu_select=L((0, pad(110, 0.2, 800, 800), 1), (0.15, pad(165, 0.4, 600, 4500), 1)))


def p_hologram():
    def grains(n_, span, f0, f1, d=0.12, conv=None):
        items = []
        for k in range(n_):
            t = R.random() * span
            f = f0 * (f1 / f0) ** R.random() if conv is None else conv * (1 + 0.03 * R.random())
            items.append((t, fadeio(tone(f, d, a=0.01, tau=0.04, rel=0.04), 0.01, 0.03), 0.4 + 0.6 * R.random()))
        return layer(items)
    return dict(click=grains(3, 0.02, 3000, 5000, 0.04),
                menu_open=grains(26, 0.45, 1200, 6000),
                menu_close=gain(fadeio(grains(26, 0.45, 1200, 6000), 0.0, 0.0)[::-1], 1.0),
                menu_select=L((0, grains(18, 0.3, 1500, 5500), 0.7), (0.3, grains(8, 0.05, 0, 0, 0.4, conv=2400), 1)))


def p_dialup():
    dt = lambda f1, f2, d=0.05: layer([(0, tone(f1, d, a=0.002, tau=1e9, rel=0.004), 1), (0, tone(f2, d, a=0.002, tau=1e9, rel=0.004), 1)])
    hand = layer([(0, fseq([1300, 2100, 1300, 2250, 1650, 2100], 0.04, "sine", a=0.002, rel=0.01), 1), (0, gain(bp(hiss(0.24, 1.0), 2500, 1.0), 0.25), 1)])
    return dict(click=dt(697, 1209, 0.04),
                menu_open=hand,
                menu_close=fseq([2100, 1650, 1300, 980], 0.05, "sine", a=0.002, rel=0.02),
                menu_select=L((0, dt(852, 1477, 0.05), 1), (0.07, dt(941, 1336, 0.05), 1), (0.14, dt(770, 1633, 0.12), 1)))


def p_robot_voice():
    def vowel(f0, d, f1a, f1b, f2a, f2b, vol=1.0):
        x = tone(f0, d, "saw", a=0.01, tau=1e9, rel=0.03)
        y = layer([(0, bp_sweep(x, f1a, f1b, 6.0), 1), (0, bp_sweep(x, f2a, f2b, 8.0), 0.6)])
        return fadeio(gain(y, 3.0 * vol), 0.01, 0.04)
    return dict(click=bp(tone(110, 0.04, "saw", a=0.001, tau=0.02), 1800, 4.0),
                menu_open=vowel(110, 0.4, 300, 700, 2300, 1200),
                menu_close=vowel(100, 0.4, 700, 300, 1200, 2300),
                menu_select=L((0, vowel(120, 0.15, 300, 550, 2300, 1900), 1), (0.17, vowel(140, 0.3, 550, 450, 1900, 1100), 1)))


def p_theremin():
    def th(f0, f1, d=0.45):
        n = ns(d)
        out = [0.0] * n
        ph = 0.0
        for i in range(n):
            t = i / SR
            u = t / d
            f = (f0 + (f1 - f0) * (1 - math.exp(-5 * u))) * (1 + 0.012 * math.sin(TAU * 6 * t))
            ph += f / SR
            out[i] = (math.sin(TAU * ph) + 0.2 * math.sin(2 * TAU * ph)) * min(1.0, t / 0.04, (d - t) / 0.08)
        return out
    return dict(click=th(1000, 1100, 0.07),
                menu_open=th(330, 780, 0.5),
                menu_close=th(780, 300, 0.5),
                menu_select=L((0, th(520, 520, 0.2), 1), (0.22, th(520, 880, 0.5), 1)))


def p_warp_drive():
    def whoosh(up, d=0.5):
        s = sweep(70 if up else 900, 900 if up else 70, d, "saw", a=0.05, rel=0.08, vol=0.5)
        s = lp_sweep(s, 3000 if up else 400, 400 if up else 3000, 0.2)
        n = bp_sweep(hiss(d, 0.8), 400 if up else 5000, 5000 if up else 400, 1.0)
        return fadeio(layer([(0, s, 1), (0, n, 0.5)]), 0.03, 0.08)
    return dict(click=layer([(0, sweep(600, 1800, 0.05, "sine", tau=0.03), 1)]),
                menu_open=whoosh(True),
                menu_close=whoosh(False),
                menu_select=L((0, thump(60, 0.2, 1.0, 1.6, 0.1), 1.2), (0.0, whoosh(True, 0.18), 0.6), (0.12, additive(2000, 0.5, [(1, 1.0, 0.25), (1.5, 0.5, 0.2), (2.0, 0.3, 0.15)]), 0.6)))


# ---------------------------------------------------------------- lo-fi
def p_lofi_tape():
    def v(f, d):
        x = layer([(0, tone(f, d, a=0.004, tau=0.4, fm=(1, 1.2, 0.2), h2=0.1), 1), (0, tone(f * 1.005, d, a=0.004, tau=0.35, fm=(1, 1.0, 0.2)), 0.6)])
        x = varispeed(x, lambda t: 1 + 0.012 * math.sin(TAU * 0.9 * t) + 0.004 * math.sin(TAU * 7 * t))
        x = lp(x, 3200)
        h = gain(hiss(len(x) / SR, 0.03), 1.0)
        return bitcrush([a + b for a, b in zip(x, h)], 9, 2)
    ch = lambda ms, d=0.5: layer([(k * 0.015, v(hz(m), d), 1) for k, m in enumerate(ms)])
    return dict(click=v(hz(79), 0.06),
                menu_open=L((0, ch([50, 57, 60, 64]), 1), (0.2, ch([55, 59, 62, 67]), 1)),
                menu_close=L((0, ch([55, 59, 62, 67]), 1), (0.2, ch([48, 55, 60, 64], 0.6), 1)),
                menu_select=L((0, ch([53, 60, 64, 69], 0.3), 1), (0.15, ch([55, 62, 67, 71], 0.7), 1)))


def p_vinyl_pop():
    def crackle(d, dens=300):
        x = [0.0] * ns(d)
        for _ in range(int(d * dens)):
            i = R.randrange(len(x))
            x[i] += R.uniform(-1, 1) * R.random() ** 2
        return lp(x, 6000)
    pop = lambda: layer([(0, click_n(1800, 0.8, 0.02, 0.004), 1), (0, thump(100, 0.03, 0.5, 1.0, 0.01), 0.5)])
    note = lambda m, d=0.35: lp(additive(hz(m), d, [(1, 1.0, 0.2), (2, 0.4, 0.1), (3, 0.15, 0.06)], a=0.003), 2800)
    needle = L((0, thump(55, 0.1, 1.0, 1.0, 0.04), 1.0), (0.02, crackle(0.25, 700), 0.7))
    return dict(click=pop(),
                menu_open=L((0, needle, 1), (0.14, note(64), 0.7), (0.0, crackle(0.5, 250), 0.4)),
                menu_close=L((0, note(64), 0.6), (0.05, hp(noise(0.12, a=0.02, tau=0.08), 2000), 0.6), (0.08, crackle(0.2, 600), 0.5)),
                menu_select=L((0, pop(), 1), (0.02, note(67, 0.45), 1), (0.1, note(72, 0.5), 1), (0, crackle(0.5, 200), 0.4)))


def p_vaporwave():
    def v(f, d=0.9):
        x = layer([(0, tone(f, d, a=0.01, tau=0.5, fm=(3.01, 1.8, 0.3)), 1), (0, tone(f * 1.01, d, a=0.01, tau=0.5, fm=(3.01, 1.8, 0.3)), 0.8)])
        return lp(x, 3000)
    return dict(click=v(hz(76), 0.1),
                menu_open=echo(L((0, v(hz(57)), 1), (0.16, v(hz(64)), 1), (0.32, v(hz(69), 1.0), 1)), 0.22, 0.35, 2),
                menu_close=echo(L((0, v(hz(69)), 1), (0.16, v(hz(64)), 1), (0.32, v(hz(57), 1.0), 1)), 0.22, 0.35, 2),
                menu_select=echo(L((0, v(hz(60)), 1), (0.0, v(hz(64)), 0.8), (0.0, v(hz(67)), 0.8), (0.0, v(hz(71)), 0.7)), 0.25, 0.4, 2))


# ---------------------------------------------------------------- haptic
def p_haptic_tick():
    t_ = lambda f=240, d=0.012: tone(f, d, a=0.0008, tau=0.004, rel=0.004)
    return dict(click=t_(260, 0.01), menu_open=L((0, t_(200), 1), (0.05, t_(280), 1)),
                menu_close=L((0, t_(280), 1), (0.05, t_(200), 1)), menu_select=L((0, t_(220), 1), (0.045, t_(220), 1), (0.09, t_(330, 0.02), 1.2)))


def p_haptic_thud():
    th = lambda f=110, d=0.09: layer([(0, tone(f, d, a=0.002, tau=0.03, glide=1.25, glide_t=0.02, h2=0.5, h3=0.2), 1)])
    return dict(click=th(160, 0.05), menu_open=L((0, th(110), 1), (0.09, th(140), 1)),
                menu_close=L((0, th(140), 1), (0.09, th(100), 1)), menu_select=L((0, th(90, 0.14), 1.2), (0.12, th(130, 0.1), 1)))


def p_haptic_buzz():
    def bz(d=0.07, f=175):
        x = tone(f, d, "saw", a=0.006, tau=1e9, rel=0.015)
        return fadeio(lp(x, 900), 0.006, 0.015)
    return dict(click=bz(0.03, 190), menu_open=L((0, bz(0.05), 0.6), (0.07, bz(0.09), 1)),
                menu_close=L((0, bz(0.09), 1), (0.11, bz(0.05), 0.6)), menu_select=L((0, bz(0.05), 1), (0.08, bz(0.05), 1), (0.16, bz(0.12, 210), 1)))


# ----------------------------------------------------------------------------- wave 2 (v3): helpers
def mk(c, o, cl, s):
    return dict(click=c, menu_open=o, menu_close=cl, menu_select=s)


def svf_lp(x, fc, q=0.25):
    """Chamberlin state-variable low-pass (resonant). fc: Hz or f(i)->Hz; q: damping (small = squelchy)."""
    low = band = 0.0
    out = [0.0] * len(x)
    for i, v in enumerate(x):
        f = fc(i) if callable(fc) else fc
        f1 = 2 * math.sin(math.pi * min(f, 3500.0) / SR)
        low += f1 * band
        high = v - low - q * band
        band += f1 * high
        out[i] = low
    return out


def pump(x, rate, depth=0.7, ph=0.0):
    return [v * (1 - depth * math.exp(-(((i / SR) * rate + ph) % 1.0) / 0.18)) for i, v in enumerate(x)]


def stack(f, d, n=5, det=0.012, wave="saw", **kw):
    items = [(0, tone(f * (1 + det * (k - (n - 1) / 2) / ((n - 1) / 2 or 1)), d, wave, **kw), 1.0 / n) for k in range(n)]
    return layer(items)


def hat(d=0.05, fc=7000, vol=1.0):
    return bp(noise(d, a=0.0003, tau=d * 0.25, vol=vol), fc, 0.8)


def kick(f=55, d=0.3, tau=0.1, vol=1.0):
    return layer([(0, tone(f, d, a=0.001, tau=tau, glide=3.2, glide_t=0.025, h2=0.15, vol=vol), 1),
                  (0, click_n(1800, 1.0, 0.01, 0.002), 0.4 * vol)])


def snare(f=190, d=0.2, vol=1.0):
    return layer([(0, tone(f, d, a=0.001, tau=0.05, glide=1.4, glide_t=0.02, vol=vol), 0.8),
                  (0, bp(noise(d, a=0.0005, tau=0.07), 2500, 0.7), 1.2 * vol)])


def clap(vol=1.0):
    return layer([(k * 0.012, bp(noise(0.05, a=0.0003, tau=0.012), 1500, 1.0), vol) for k in range(3)] +
                 [(0.036, bp(noise(0.12, a=0.0003, tau=0.05), 1400, 1.0), 0.9 * vol)])


def pad_v(f, d, a=0.15, lpf=2500, det=1.004):
    x = layer([(0, tone(f, d, "saw", a=a, tau=1e9, rel=0.2), 0.5), (0, tone(f * det, d, "saw", a=a, tau=1e9, rel=0.2), 0.5),
               (0, tone(f / 2, d, "sine", a=a, tau=1e9, rel=0.2), 0.4)])
    return lp(x, lpf)


def chord(v, ms, d, g=0.0, vol=1.0):
    return layer([(k * g, v(hz(m), d), vol) for k, m in enumerate(ms)])


def shaker(d=0.12, fc=5500, vol=1.0):
    return bp(noise(d, a=d * 0.35, tau=d * 0.4, vol=vol), fc, 1.0)


def knock(f, d=0.08, vol=1.0, body=1800):
    return layer([(0, tone(f, d, a=0.001, tau=d * 0.3, glide=1.15, glide_t=0.01), vol),
                  (0, click_n(body, 1.2, 0.015, 0.003), 0.5 * vol)])


def creak(f0, f1, d, wob=40.0, vol=1.0):
    n = ns(d)
    out = [0.0] * n
    ph = 0.0
    for i in range(n):
        t = i / SR
        f = (f0 + (f1 - f0) * t / d) * (1 + 0.12 * math.sin(TAU * wob * t) * (0.5 + R.random()))
        ph += f / SR
        ph -= int(ph)
        out[i] = (2 * ph - 1) * min(1.0, t / 0.02, (d - t) / 0.03) * vol
    return bp(out, (f0 + f1) / 2 * 2.2, 3.0)


# ----------------------------------------------------------------------------- wave 2: world instruments
def sitar_v(f, d):
    s = ks(f, d, tau=0.9, bright=0.75)
    return layer([(0, s, 0.7), (0, sat(gain(s, 4.0), 1.0), 0.45), (0, additive(f * 2, d, [(1, 0.25, 0.7), (1.5, 0.1, 0.6)]), 0.25)])


def p_sitar():
    v = lambda f, d: bend(sitar_v(f, d), -150, 0.09)
    return std(v, [60, 64, 67, 68, 72], [72, 68, 67, 64, 60], [67, 72, 76], 0.09, last=0.8, sel_gap=0.11)


def p_tabla():
    dayan = lambda f=330, d=0.3: layer([(0, additive(f, d, [(1, 1.0, 0.12), (2.0, 0.5, 0.08), (3.0, 0.3, 0.05)]), 1),
                                       (0, click_n(2200, 1.5, 0.012, 0.003), 0.4)])
    bayan = lambda f=95, d=0.4: layer([(0, tone(f, d, a=0.002, tau=0.16, glide=0.8, glide_t=0.12, h2=0.2), 1),
                                      (0, click_n(600, 1.0, 0.02, 0.005), 0.4)])
    tin = lambda f=440: tone(f, 0.18, a=0.001, tau=0.05, h2=0.3)
    return mk(dayan(560, 0.12),
              L((0, bayan(), 1), (0.12, dayan(330, 0.2), 1), (0.22, tin(), 0.8), (0.3, dayan(330, 0.25), 0.9)),
              L((0, tin(), 0.8), (0.08, dayan(330, 0.2), 1), (0.18, bayan(80, 0.5), 1.1)),
              L((0, dayan(330, 0.2), 1), (0.09, dayan(330, 0.2), 0.8), (0.17, bayan(), 1), (0.17, dayan(440, 0.5), 1)))


def p_oud():
    v = lambda f, d: lp(layer([(0, ks(f, d, 0.45, 0.35), 1), (0, ks(f * 1.004, d, 0.45, 0.35), 0.7)]), 3800)
    return std(v, [50, 54, 55, 57, 62], [62, 58, 57, 55, 50], [57, 62, 66], 0.085, last=0.8, sel_gap=0.1)


def p_erhu():
    def v(f, d):
        x = tone(f, d, "saw", a=0.05, tau=1e9, vib=0.007, glide=0.95, glide_t=0.1, rel=0.08)
        return layer([(0, lp(x, 3000), 0.8), (0, bp(x, 1400, 3.0), 1.2)])
    return std(v, [62, 66, 69, 74], [74, 71, 69, 66, 62], [66, 69, 74], 0.14, last=0.8, sel_gap=0.15)


def p_balafon():
    v = lambda f, d: layer([(0, additive(f, d, [(1, 1, 0.22), (3.0, 0.3, 0.1), (5.1, 0.1, 0.05)]), 1),
                            (0, bp(noise(d, a=0.0005, tau=0.09, vol=0.5), 2600, 2.5), 0.35)])
    return std(v, [60, 62, 64, 67, 69, 72], [72, 69, 67, 64, 62, 60], [64, 67, 72], 0.07, last=0.5, sel_gap=0.09)


def p_didgeridoo():
    def drone(f, d, lo=300, hi=1500, rev=False):
        x = tone(f, d, "sq", duty=0.35, a=0.04, tau=1e9, rel=0.08)
        x = lp(x, 700)
        return bp_sweep(x, hi, lo, 2.0) if rev else bp_sweep(x, lo, hi, 2.0)
    pop = lambda: layer([(0, thump(160, 0.06, 1.0, 1.5, 0.02), 1), (0, click_n(900, 1.0, 0.02, 0.005), 0.5)])
    yell = lambda: tone(hz(70), 0.4, "sine", a=0.02, tau=0.2, glide=1.2, glide_t=0.15, fm=(1.5, 2.0, 0.3))
    return mk(pop(), drone(65, 1.0), drone(65, 0.9, rev=True),
              L((0, pop(), 1), (0.08, drone(65, 0.9), 0.8), (0.3, yell(), 0.7)))


def p_bagpipe():
    def reed(f, d):
        x = tone(f, d, "saw", a=0.01, tau=1e9, rel=0.03)
        return layer([(0, lp(x, 3200), 0.6), (0, bp(x, 1500, 2.0), 1.0)])
    drone = lambda d: lp(tone(hz(45), d, "saw", a=0.08, tau=1e9, rel=0.1), 500)

    def tune(ms, gap, d):
        items = []
        for k, m in enumerate(ms):
            items.append((k * gap, reed(hz(m + 7), 0.03), 0.8))
            items.append((k * gap + 0.03, reed(hz(m), d), 1.0))
        return layer(items)
    return mk(reed(hz(76), 0.07),
              L((0, drone(0.9), 0.6), (0, tune([69, 71, 73, 74], 0.17, 0.15), 1)),
              L((0, drone(0.8), 0.6), (0, tune([74, 73, 71, 69], 0.17, 0.15), 1)),
              L((0, drone(1.0), 0.6), (0, tune([71, 74, 76], 0.2, 0.3), 1)))


def p_accordion():
    def v(f, d):
        x = layer([(0, tone(f, d, "saw", a=0.06, tau=1e9, rel=0.08), 0.5), (0, tone(f * 1.007, d, "saw", a=0.06, tau=1e9, rel=0.08), 0.5),
                   (0, tone(f * 2, d, "sq", duty=0.3, a=0.06, tau=1e9, rel=0.08), 0.2)])
        return lp(x, 2800)
    ch = lambda ms, d: chord(v, ms, d)
    return mk(v(hz(72), 0.07), ch([60, 64, 67], 0.7), ch([67, 64, 60], 0.6),
              L((0, ch([57, 60, 64], 0.2), 1), (0.22, ch([62, 65, 69, 74], 0.8), 1)))


def p_harmonica():
    def v(f, d, bnd=0.96):
        x = tone(f, d, "sq", duty=0.32, a=0.03, tau=1e9, glide=bnd, glide_t=0.07, rel=0.06)
        br = bp(noise(d, a=0.03, tau=d * 0.5, vol=0.25), f * 2, 2.0)
        return lp(layer([(0, x, 0.7), (0, br, 0.6)]), 3400)
    ch = lambda ms, d: chord(v, ms, d, 0.0, 0.6)
    return mk(v(hz(79), 0.07, 1.0),
              L((0, v(hz(67), 0.14), 1), (0.15, v(hz(72), 0.14), 1), (0.3, v(hz(76), 0.5), 1)),
              L((0, v(hz(76), 0.14), 1), (0.15, v(hz(72), 0.14), 1), (0.3, v(hz(67), 0.5), 1)),
              ch([67, 72, 76], 0.6))


def p_ukulele():
    v = lambda f, d: ks(f, d, 0.35, 0.4)
    st = lambda ms, d=0.5, g=0.014, up=False: layer([(k * g, v(hz(m), d), 0.9) for k, m in enumerate(ms[::-1] if up else ms)])
    return mk(v(hz(79), 0.08),
              L((0, st([67, 60, 64, 69]), 1), (0.22, st([67, 60, 64, 69], 0.3, 0.01, True), 0.7), (0.34, st([65, 60, 65, 69], 0.5), 1)),
              L((0, st([65, 60, 65, 69]), 1), (0.22, st([67, 60, 64, 69], 0.5), 1)),
              L((0, st([67, 62, 66, 69]), 1), (0.2, st([71, 62, 67, 74], 0.7, 0.02), 1)))


def p_banjo():
    def v(f, d):
        s = ks(f, d, 0.25, 0.95)
        return layer([(0, s, 0.9), (0, bp(s, 900, 1.2), 0.5), (0, click_n(2500, 1.5, 0.01, 0.002), 0.4)])
    roll = lambda ms, g=0.055: layer([(k * g, v(hz(m), 0.3), 1) for k, m in enumerate(ms)])
    return mk(v(hz(79), 0.07), roll([67, 62, 67, 71, 74, 71]), roll([74, 71, 67, 62, 59], 0.06),
              L((0, roll([67, 71, 74, 79]), 1), (0.25, chord(v, [62, 67, 71, 74], 0.7, 0.01), 1)))


def p_taiko():
    don = lambda f=70, d=0.5: layer([(0, tone(f, d, a=0.002, tau=0.22, glide=1.5, glide_t=0.05, h2=0.2), 1),
                                    (0, bp(noise(0.1, a=0.0005, tau=0.025), 700, 0.8), 0.7)])
    ka = lambda: knock(380, 0.07, 1.0, 2200)
    return mk(ka(), L((0, don(70, 0.4), 1), (0.2, don(70, 0.4), 0.9), (0.4, don(85, 0.7), 1.1)),
              L((0, ka(), 0.8), (0.1, don(85, 0.4), 1), (0.3, don(60, 0.8), 1.1)),
              L((0, ka(), 0.8), (0.07, ka(), 0.8), (0.16, don(75, 0.8), 1.2)))


def p_cajon():
    bass = lambda: layer([(0, tone(95, 0.25, a=0.001, tau=0.09, glide=1.6, glide_t=0.03), 1),
                          (0, lp(noise(0.12, a=0.0005, tau=0.03), 700), 0.9)])
    slap = lambda: layer([(0, bp(noise(0.1, a=0.0003, tau=0.04), 2300, 0.8), 1),
                          (0, hp(noise(0.14, a=0.002, tau=0.07), 5000), 0.4)])
    return mk(slap(), L((0, bass(), 1), (0.15, slap(), 0.9), (0.3, bass(), 1), (0.4, slap(), 0.9)),
              L((0, slap(), 0.9), (0.1, bass(), 1)), L((0, bass(), 1), (0.12, slap(), 1), (0.2, slap(), 0.8)))


def p_latin_percussion():
    cow = lambda f=560, d=0.18: layer([(0, tone(f, d, "sq", a=0.001, tau=0.06), 0.6), (0, tone(f * 1.5, d, "sq", a=0.001, tau=0.05), 0.5)])
    cow = (lambda c: (lambda f=560, d=0.18: bp(c(f, d), f * 1.3, 1.5)))(cow)
    claves = lambda: tone(2300, 0.06, a=0.0005, tau=0.012, fm=(1.0, 1.0, 0.01))
    guiro = lambda up=True: layer([(k * 0.028, click_n(3000 + (k * 90 if up else -k * 90), 2.0, 0.015, 0.004), 1) for k in range(10)])
    return mk(claves(), L((0, guiro(), 0.8), (0.3, cow(), 1)), L((0, guiro(False), 0.8), (0.3, claves(), 1)),
              L((0, cow(), 1), (0.15, cow(740), 1), (0.3, claves(), 0.9), (0.4, shaker(0.15), 0.6)))


def p_pan_flute():
    def v(f, d):
        br = bp(noise(d, a=0.04, tau=d * 0.6, vol=1.0), f, 14.0)
        return layer([(0, tone(f, d, a=0.045, tau=d * 0.7, vib=0.004, h3=0.04), 0.8), (0, br, 2.5)])
    return std(v, [72, 76, 79, 83], [83, 79, 76, 72], [76, 79, 84], 0.14, last=0.7, sel_gap=0.16)


# ---------------------------------------------------------------- classical keys and bells
def p_celesta():
    v = lambda f, d: layer([(0, additive(f, d, [(1, 1, 0.5), (2.0, 0.35, 0.3), (3.0, 0.12, 0.2), (5.4, 0.05, 0.08)], a=0.002), 1),
                            (0, tone(f * 0.5, 0.05, a=0.001, tau=0.015), 0.3)])
    return std(v, [72, 76, 79, 83, 88], [88, 83, 79, 76, 72], [79, 84, 91], 0.08, last=0.9, sel_gap=0.1)


def p_harpsichord():
    def v(f, d):
        a = ks(f, d, 0.5, 1.0)
        b = ks(f * 2, d, 0.4, 1.0)
        return layer([(0, a, 1), (0, b, 0.45), (0, click_n(3500, 1.5, 0.008, 0.002), 0.5)])
    trill = lambda m, n: layer([(k * 0.06, v(hz(m + (2 if k % 2 else 0)), 0.2), 1) for k in range(n)])
    return mk(v(hz(79), 0.07), L((0, trill(67, 4), 1), (0.25, v(hz(72), 0.4), 1)),
              L((0, v(hz(76), 0.15), 1), (0.14, v(hz(72), 0.15), 1), (0.28, v(hz(67), 0.5), 1)),
              chord(v, [60, 67, 72, 76], 0.8, 0.025))


def p_pipe_organ():
    def v(f, d):
        x = additive(f, d, [(0.5, 0.5, 9), (1, 1, 9), (2, 0.6, 9), (3, 0.3, 9), (4, 0.3, 9), (6, 0.12, 9), (8, 0.08, 9)], a=0.05, rel=0.12)
        chiff = bp(noise(0.08, a=0.002, tau=0.02, vol=0.8), f * 3, 2.0)
        return layer([(0, x, 1), (0, chiff, 0.4)])
    wide = lambda ms, d: echo(chord(v, ms, d), 0.17, 0.35, 3)
    return mk(v(hz(79), 0.1), wide([43, 55, 59, 62], 0.7), wide([62, 59, 55, 43], 0.6), wide([48, 60, 64, 67, 72], 0.9))


def p_church_bells():
    P = [(0.5, 0.5, 1.3), (1.0, 1.0, 1.1), (1.2, 0.6, 0.9), (1.5, 0.35, 0.7), (2.0, 0.5, 0.6), (2.5, 0.2, 0.4), (3.0, 0.15, 0.3)]
    v = lambda f, d: layer([(0, additive(f, d, P, a=0.001, rel=0.05), 1), (0, additive(f * 1.006, d, P[:3], a=0.001, rel=0.05), 0.4),
                            (0, click_n(1500, 1.0, 0.012, 0.003), 0.3)])
    return mk(v(hz(60), 0.35), echo(L((0, v(hz(55), 1.0), 1), (0.3, v(hz(60), 1.0), 0.9), (0.6, v(hz(64), 1.0), 0.9)), 0.3, 0.3, 1),
              echo(L((0, v(hz(64), 1.0), 1), (0.3, v(hz(60), 1.0), 0.9), (0.6, v(hz(55), 1.0), 0.9)), 0.3, 0.3, 1),
              L((0, v(hz(48), 1.7), 1.1), (0.0, v(hz(60), 1.5), 0.7)))


# ---------------------------------------------------------------- electronic genres
def p_house_stabs():
    def stab(ms, d=0.28):
        x = layer([(0, stack(hz(m), d, 3, 0.01, a=0.002, tau=0.12), 1) for m in ms])
        return lp_sweep(x, 5000, 700, 0.09)
    return mk(hat(0.04, 8000), L((0, kick(), 1), (0.25, stab([57, 60, 64, 67]), 0.8), (0.5, stab([59, 62, 66, 69]), 0.8)),
              L((0, stab([59, 62, 66, 69]), 0.8), (0.25, stab([57, 60, 64, 67], 0.4), 0.8), (0.25, kick(48, 0.4), 1)),
              L((0, kick(), 1), (0.0, clap(), 0.8), (0.0, stab([60, 64, 67, 71, 74], 0.7), 1)))


def p_techno_blips():
    bl = lambda f, d=0.12: sat(lp_sweep(tone(f, d, "sq", a=0.001, tau=0.06), 3000, 600, 0.04), 2.0)
    m = lambda x: echo(x, 0.1875, 0.4, 3)
    return mk(m(bl(hz(57), 0.05)), m(L((0, kick(48, 0.3, 0.12), 1), (0.19, bl(hz(45)), 0.8), (0.375, bl(hz(57)), 0.8))),
              m(L((0, bl(hz(57)), 0.8), (0.19, bl(hz(45)), 0.8), (0.375, kick(44, 0.4, 0.14), 1))),
              m(L((0, kick(48, 0.3), 1), (0.0, hat(0.05, 9000), 0.7), (0.19, bl(hz(52), 0.2), 1), (0.375, bl(hz(64), 0.25), 0.9))))


def wobble(f, d, rate, lo=200, hi=1800):
    x = tone(f, d, "saw", a=0.01, tau=1e9, rel=0.05)
    y = svf_lp(x, lambda i: lo + (hi - lo) * (0.5 + 0.5 * math.sin(TAU * rate * i / SR)), 0.35)
    return fadeio(sat(gain(y, 2.0), 1.0), 0.01, 0.04)


def p_dubstep_wobble():
    return mk(layer([(0, thump(70, 0.08, 1.0, 2.5, 0.03), 1), (0, click_n(2200, 1.0, 0.01, 0.002), 0.4)]),
              L((0, wobble(hz(33), 0.9, 3.0, 150, 1800), 1)),
              L((0, wobble(hz(36), 0.9, 6.0, 1800, 150), 1)),
              L((0, kick(), 1), (0.1, wobble(hz(31), 0.7, 5.0, 150, 2000), 1), (0.1, snare(), 0.7)))


def p_dnb_breaks():
    st = 60 / 174 / 4
    br = lambda pat: layer([(k * st, h, 1) for k, h in enumerate(pat) if h is not None])
    K, S, H = kick(58, 0.18, 0.06), snare(200, 0.15), hat(0.04, 8500, 0.7)
    brk = br([K, H, None, H, S, H, K, None, None, K, K, H, S, None, H, S])
    reese = lp(layer([(0, tone(hz(31), 0.7, "saw", a=0.01, tau=1e9, rel=0.1), 0.6), (0, tone(hz(31) * 1.01, 0.7, "saw", a=0.01, tau=1e9, rel=0.1), 0.6)]), 400)
    return mk(S, L((0, brk, 1)), L((0, br([S, H, S, H, K, None, K, None, S]), 1)),
              L((0, br([K, H, S, None, K]), 1), (0.05, reese, 0.9)))


def p_trance_plucks():
    def pl(f, d=0.2):
        return lp_sweep(stack(f, d, 7, 0.02, a=0.001, tau=0.1), 5500, 900, 0.12)
    arp = lambda ms, g=0.1875 / 2: echo(layer([(k * g, pl(hz(m)), 1) for k, m in enumerate(ms)]), 0.1875, 0.35, 2)
    return mk(pl(hz(81), 0.1), arp([69, 72, 76, 81, 76, 72]), arp([81, 76, 72, 69, 64]), arp([72, 76, 79, 84, 88], 0.07))


def p_idm_glitch():
    def grain(f, d=0.04, g=0.9):
        return lp(bitcrush(tone(f, d, "sine", a=0.001, tau=d * 0.5, fm=(2.0, 3.0, d * 0.3), vol=g), 5, 2), 5000)
    def burst(n, base, spread, gap):
        return layer([(k * gap + R.random() * 0.01, grain(base * 2 ** (R.randrange(-spread, spread) / 5.0), R.choice((0.03, 0.05, 0.08))), 0.8) for k in range(n)])
    tk = lambda: click_n(R.choice((3000, 5000)), 2.0, 0.01, 0.002)
    return mk(L((0, grain(1800, 0.02), 1), (0.0, tk(), 0.5)),
              L((0, burst(9, 400, 6, 0.045), 1), (0.05, kick(52, 0.2), 0.8)),
              L((0, burst(7, 900, 6, 0.06), 1), (0.3, tk(), 0.7), (0.34, kick(60, 0.2), 0.8)),
              L((0, burst(5, 600, 4, 0.037), 1), (0.22, grain(880, 0.3, 0.9), 0.9), (0.22, grain(1320, 0.3, 0.9), 0.5)))


def acid(f, d, cut0=2800, cut1=250, slide=None):
    x = tone(f, d, "saw", a=0.002, tau=1e9, glide=slide or 1.0, glide_t=0.04, rel=0.02)
    y = svf_lp(x, lambda i: cut1 + (cut0 - cut1) * math.exp(-i / (SR * 0.12)), 0.18)
    return fadeio(sat(gain(y, 1.6), 1.0), 0.002, 0.02)


def p_acid_squelch():
    return mk(acid(hz(40), 0.08), L((0, acid(hz(36), 0.12), 1), (0.15, acid(hz(48), 0.12), 1), (0.3, acid(hz(43), 0.3, slide=0.8), 1)),
              L((0, acid(hz(48), 0.12), 1), (0.15, acid(hz(43), 0.12), 1), (0.3, acid(hz(36), 0.4, 3200, 200), 1)),
              L((0, kick(), 0.9), (0.0, acid(hz(36), 0.12), 1), (0.15, acid(hz(48), 0.1), 1), (0.25, acid(hz(55), 0.4, 3400), 1)))


def p_future_bass():
    def ch(ms, d=0.7):
        x = layer([(0, stack(hz(m), d, 5, 0.018, a=0.01, tau=1e9, rel=0.1), 0.6) for m in ms])
        return pump(lp(x, 3200), 3.0, 0.8)
    return mk(layer([(0, tone(1100, 0.07, a=0.001, tau=0.03, fm=(2, 2, 0.03)), 1)]),
              L((0, ch([58, 62, 65, 69, 72], 0.9), 1)), L((0, ch([56, 60, 63, 67, 70], 0.8), 1)),
              L((0, kick(), 0.8), (0.0, ch([60, 64, 67, 71, 74, 79], 1.0), 1)))


def p_ambient_pads():
    sw = lambda ms, d: echo(chord(lambda f, dd: pad_v(f, dd, 0.25, 1800), ms, d), 0.3, 0.3, 2)
    return mk(lp(tone(hz(72), 0.4, a=0.08, tau=0.15), 1500), sw([48, 55, 62], 1.2), sw([55, 50, 43], 1.2), sw([48, 55, 59, 64, 71], 1.3))


def p_hyperpop():
    def lead(f, d=0.15):
        return sat(gain(tone(f, d, "saw", a=0.002, tau=0.1, glide=1.3, glide_t=0.03), 2.5), 1.0)
    return mk(bitcrush(tone(2200, 0.05, a=0.001, tau=0.02, fm=(1.4, 3, 0.02)), 5, 2),
              L((0, lead(hz(76)), 0.7), (0.06, lead(hz(83)), 0.7), (0.12, lead(hz(88)), 0.7), (0.18, lead(hz(95), 0.3), 0.7)),
              L((0, lead(hz(95)), 0.7), (0.06, lead(hz(88)), 0.7), (0.12, lead(hz(83)), 0.7), (0.18, kick(40, 0.4), 1.2)),
              L((0, sat(gain(kick(45, 0.35), 3.0), 1.0), 1), (0.1, bitcrush(chord(lambda f, d: tone(f, d, "sq", a=0.002, tau=0.15), [84, 88, 91, 95], 0.4, 0.0), 6, 2), 0.7)))


def p_trap_hats():
    rl = lambda n, g0, g1, f=8500: layer([(k * (g0 + (g1 - g0) * k / max(1, n - 1)), hat(0.03, f, 0.7 + 0.3 * (k % 2)), 1) for k in range(n)])
    s808 = lambda f=44, d=0.9: tone(f, d, a=0.003, tau=0.4, glide=1.6, glide_t=0.08, h2=0.2)
    return mk(hat(0.03, 9000), rl(10, 0.09, 0.03), rl(10, 0.03, 0.09, 7000),
              L((0, hat(0.03, 9000), 0.7), (0.06, hat(0.03, 9000), 0.7), (0.12, s808(), 1.2), (0.12, clap(), 0.8)))


def p_drum_box():
    cl = lambda: tone(2800, 0.04, "sq", a=0.0005, tau=0.012)
    cow = lambda: bp(tone(540, 0.2, "sq", a=0.001, tau=0.08), 800, 1.5)
    tom = lambda f: tone(f, 0.3, a=0.001, tau=0.12, glide=1.6, glide_t=0.04)
    return mk(cl(), L((0, tom(180), 1), (0.1, tom(140), 1), (0.2, tom(100), 1)), L((0, tom(100), 1), (0.1, tom(140), 1), (0.2, tom(180), 1)),
              L((0, kick(), 1), (0.12, clap(), 0.9), (0.12, cow(), 0.6)))


# ---------------------------------------------------------------- retro hardware
def p_floppy_seek():
    def seek(n, f0, f1):
        return layer([(k * 0.032, tone(f0 + (f1 - f0) * k / max(1, n - 1), 0.03, "sq", a=0.001, tau=0.02), 1) for k in range(n)])
    grind = lambda d: bp(noise(d, a=0.02, tau=d * 0.6, vol=0.8), 500, 1.0)
    return mk(layer([(0, thump(130, 0.03, 1.0, 1.3, 0.01), 1), (0, click_n(2000, 1.5, 0.008, 0.002), 0.6)]),
              L((0, thump(80, 0.1), 1), (0.05, grind(0.2), 0.5), (0.1, seek(8, 150, 260), 0.9)),
              L((0, seek(6, 260, 140), 0.9), (0.22, thump(70, 0.15), 1)),
              L((0, seek(10, 120, 320), 0.9), (0.4, tone(1500, 0.15, "sq", a=0.002, tau=0.07), 0.5)))


def p_crt_degauss():
    def hum(d, f=60):
        return layer([(0, tone(f, d, a=0.03, tau=1e9, rel=0.1, h2=0.5, h3=0.2), 0.7)])
    deg = lambda d=0.9: [v * math.exp(-i / (SR * d * 0.35)) for i, v in enumerate(tone(60, d, "sine", a=0.002, tau=1e9, h2=0.6, glide=2.0, glide_t=0.3))]
    whine = lambda d, f0, f1: sweep(f0, f1, d, "sine", a=0.05, rel=0.1, vol=0.3)
    return mk(layer([(0, thump(95, 0.05), 1), (0, click_n(3500, 1.0, 0.01, 0.002), 0.6)]),
              L((0, thump(55, 0.3, 1.2, 2.0, 0.1), 1), (0.05, deg(), 1), (0.3, whine(0.6, 600, 4200), 0.7)),
              L((0, whine(0.35, 4200, 200), 0.8), (0.3, thump(80, 0.1), 1)),
              L((0, thump(60, 0.15), 1), (0.05, tone(880, 0.35, a=0.004, tau=0.15, vol=0.6), 1), (0.05, hum(0.4), 0.5)))


def p_dot_matrix():
    def head(n, f0, f1):
        return layer([(k * 0.011, tone(f0 + (f1 - f0) * (k % 7) / 6.0, 0.012, "sq", a=0.0005, tau=0.01), 0.8) for k in range(n)])
    def feed(n):
        return layer([(k * 0.04, click_n(900, 2.0, 0.012, 0.003), 1) for k in range(n)])
    return mk(head(3, 1800, 2400), L((0, head(40, 1600, 3000), 0.8), (0.46, feed(3), 1)),
              L((0, feed(5), 1), (0.22, head(18, 3000, 1600), 0.8)),
              L((0, head(12, 2000, 3200), 0.8), (0.15, tone(2400, 0.2, a=0.002, tau=0.08), 0.6), (0.15, click_n(1500, 1.0, 0.02, 0.005), 0.6)))


def p_pinball():
    flip = lambda: layer([(0, thump(110, 0.06, 1.0, 1.5, 0.02), 1), (0, click_n(1200, 1.2, 0.015, 0.004), 0.9)])
    bump = lambda f=1100: layer([(0, additive(f, 0.35, [(1, 1, 0.12), (2.4, 0.5, 0.08), (3.9, 0.3, 0.05)]), 0.8), (0, flip(), 0.4)])
    spring = lambda: sweep(120, 700, 0.35, "sq", a=0.05, rel=0.05, vol=0.4)
    return mk(flip(), L((0, ratchet(6, 600, 1200), 0.8), (0.4, spring(), 0.8), (0.52, flip(), 1.1)),
              L((0, sweep(900, 120, 0.4, "tri", vol=0.8), 1), (0.4, thump(70, 0.2), 1)),
              L((0, bump(1100), 1), (0.07, bump(1400), 0.9), (0.14, bump(1750), 0.8)))


def p_slot_reels():
    def spin(n, rev=False):
        gaps = [0.03 * 1.18 ** k for k in range(n)]
        t = 0
        items = []
        for k, g in enumerate(gaps):
            items.append((t, layer([(0, click_n(1400 - (k * 25 if not rev else -k * 25), 2.0, 0.012, 0.003), 1), (0, tone(220, 0.02, a=0.0005, tau=0.008), 0.5)]), 1))
            t += g
        return layer(items)
    bellp = lambda f: additive(f, 0.7, [(1, 1, 0.3), (2.76, 0.4, 0.2), (5.4, 0.2, 0.1)])
    return mk(L((0, click_n(1400, 2.0, 0.012, 0.003), 1), (0, tone(220, 0.02, a=0.0005, tau=0.008), 0.5)),
              L((0, spin(14), 1)), L((0, spin(10, True), 0.9), (0.9, thump(90, 0.15), 1)),
              L((0, thump(90, 0.12), 1), (0.1, bellp(1320), 1), (0.22, bellp(1760), 1), (0.34, bellp(2200), 0.9)))


def p_jukebox():
    cheap = lambda x: lp(hp(x, 300), 3000)
    v = lambda f, d: cheap(layer([(0, tone(f, d, "sq", duty=0.4, a=0.01, tau=d * 0.5), 0.6), (0, tone(f * 2, d, "sine", a=0.01, tau=d * 0.4), 0.3)]))
    relay_ = lambda: layer([(0, relay(), 1), (0.05, relay(0.8), 0.8)])
    drop = layer([(0, thump(60, 0.08), 0.8), (0.05, cheap(noise(0.4, a=0.05, tau=0.15, vol=0.5)), 0.7)])
    return mk(relay(), L((0, relay_(), 1), (0.2, drop, 1), (0.5, stagger(v, [60, 64, 67], 0.1, 0.25), 0.9)),
              L((0, stagger(v, [67, 64, 60], 0.09, 0.2), 0.9), (0.4, relay_(), 1)),
              L((0, relay_(), 1), (0.15, stagger(v, [60, 67, 72, 76], 0.07, 0.3), 1)))


def p_telegraph_morse():
    def snd(down=True):
        return layer([(0, click_n(1100 if down else 800, 2.0, 0.02, 0.005), 1), (0, thump(130 if down else 100, 0.04, 0.8, 1.2, 0.01), 0.8)])
    def code(pat):
        items, t = [], 0.0
        for ch in pat:
            ln = 0.05 if ch == "." else 0.15
            items.append((t, snd(True), 1))
            items.append((t + ln, snd(False), 0.8))
            t += ln + 0.09
        return layer(items)
    return mk(L((0, snd(True), 1), (0.05, snd(False), 0.8)), code(".-"), code("-."), code("---"))


def p_cash_register():
    ch = lambda f: additive(f, 0.9, [(1, 1, 0.35), (2.7, 0.5, 0.25), (4.1, 0.25, 0.15), (6.3, 0.1, 0.08)])
    key = lambda: layer([(0, knock(260, 0.04, 1.0, 1800), 1)])
    drawer = lambda: layer([(0, ratchet(5, 700, 1500), 0.7), (0.3, thump(80, 0.15, 1.2), 1)])
    return mk(key(), L((0, key(), 1), (0.1, drawer(), 1)), L((0, thump(90, 0.1), 1), (0.04, ch(1800), 0.6)),
              L((0, key(), 1), (0.1, ch(2100), 1), (0.16, ch(2800), 0.7)))


def p_shortwave_radio():
    stat = lambda d, vol=0.6: bp(noise(d, a=0.02, tau=d * 0.5, vol=vol), 1800, 0.6)
    het = lambda d, f0, f1: sweep(f0, f1, d, "sine", a=0.05, rel=0.1, vol=0.5)
    return mk(layer([(0, click_n(2500, 1.0, 0.015, 0.004), 1)]),
              L((0, stat(0.9), 0.7), (0.0, het(0.8, 400, 1800), 1)), L((0, stat(0.7), 0.7), (0.0, het(0.6, 1800, 300), 1)),
              L((0, stat(0.15), 1), (0.1, tone(700, 0.45, a=0.01, tau=0.3), 0.9), (0.1, stat(0.5, 0.15), 1)))


# ---------------------------------------------------------------- game moods
def p_rpg_menu():
    v = lambda f, d: layer([(0, lp(ks(f, d, 0.7, 0.5), 5000), 0.9), (0, additive(f * 2, d * 0.6, [(1, 0.3, 0.3), (3.0, 0.1, 0.15)]), 0.4)])
    return std(v, [67, 71, 74, 79], [79, 74, 71, 67], [72, 76, 79, 84], 0.09, last=0.7, sel_gap=0.08)


def p_fighter_stinger():
    hit = lambda f=70: layer([(0, thump(f, 0.25, 1.0, 2.5, 0.08), 1.2), (0, lp(noise(0.12, a=0.0005, tau=0.04), 3500), 1.0),
                              (0, additive(520, 0.5, [(1, 0.5, 0.2), (2.4, 0.3, 0.12), (3.7, 0.2, 0.08)]), 0.35)])
    whoosh = lambda up=True: swish(500, 4000, 0.25, 1.0, 0.8) if up else swish(4000, 500, 0.25, 1.0, 0.3)
    brass = lambda ms: lp_sweep(chord(lambda f, d: tone(f, d, "saw", a=0.01, tau=0.3), ms, 0.5, 0.0), 3500, 800, 0.2)
    return mk(layer([(0, thump(110, 0.06, 1.0, 2.0, 0.02), 1), (0, lp(noise(0.05, a=0.0004, tau=0.015), 3000), 1)]),
              L((0, whoosh(), 0.8), (0.24, hit(), 1)), L((0, whoosh(False), 0.7), (0.2, thump(55, 0.3, 1.0, 1.4, 0.1), 1)),
              L((0, hit(), 1), (0.14, hit(90), 1), (0.14, brass([52, 59, 64]), 0.8)))


def p_racing_gears():
    shift = lambda: layer([(0, click_n(1400, 1.2, 0.02, 0.006), 1), (0, thump(120, 0.05, 0.9, 1.3, 0.015), 0.8)])
    rev = lambda f0, f1, d: lp(sweep(f0, f1, d, "saw", a=0.03, rel=0.08, vol=0.7), 1200)
    whistle = lambda: sweep(900, 3000, 0.5, "sine", a=0.1, rel=0.1, vol=0.35)
    pssh = lambda: bp(noise(0.3, a=0.005, tau=0.1, vol=0.9), 4500, 0.7)
    return mk(shift(), L((0, rev(80, 280, 0.8), 1), (0.8, shift(), 1)), L((0, rev(280, 70, 0.7), 1), (0.7, shift(), 0.9)),
              L((0, whistle(), 0.8), (0.0, bp(noise(0.5, a=0.1, tau=0.5, vol=0.4), 3000, 1.0), 0.5), (0.5, pssh(), 1), (0.5, shift(), 0.6)))


def p_platform_hop():
    def hop(f, d=0.22, up=True):
        n = ns(d)
        out = [0.0] * n
        ph = 0.0
        for i in range(n):
            t = i / n
            fr = f * (1 + 0.6 * (1 - (2 * t - 1) ** 2) if up else 1 - 0.4 * t)
            ph += fr / SR
            out[i] = (math.sin(TAU * ph) + 0.3 * math.sin(2 * TAU * ph)) * min(1.0, i / 60, (n - i) / 400) * (1 - 0.5 * t)
        return out
    ding = lambda f: tone(f, 0.18, a=0.001, tau=0.07, h2=0.4)
    return mk(hop(440, 0.12), L((0, hop(330), 1), (0.2, hop(440), 1), (0.4, hop(587, 0.3), 1)),
              L((0, hop(587, 0.15, False), 1), (0.17, hop(440, 0.15, False), 1), (0.34, thump(80, 0.15), 1)),
              L((0, ding(988), 1), (0.08, ding(1319), 1), (0.2, hop(660, 0.2), 0.8)))


def p_puzzle_match():
    pop = lambda: layer([(0, thump(300, 0.05, 1.0, 1.8, 0.015), 0.7), (0, click_n(1800, 1.0, 0.01, 0.003), 0.5)])
    sp = lambda f, d=0.4: layer([(0, additive(f, d, [(1, 1, 0.15), (2.0, 0.3, 0.1), (3.0, 0.15, 0.06)]), 1), (0, pop(), 0.3)])
    return mk(L((0, pop(), 1), (0, sp(1047, 0.2), 0.6)),
              stagger(sp, [72, 76, 79, 84], 0.07, 0.4), stagger(sp, [84, 79, 76, 72], 0.07, 0.3),
              L((0, pop(), 1), (0.06, stagger(sp, [72, 74, 76, 79, 81, 84, 88], 0.05, 0.45), 1)))


def p_horror_stinger():
    heart = lambda: L((0, thump(55, 0.15, 1.0, 1.5, 0.05), 1), (0.17, thump(48, 0.2, 0.8, 1.5, 0.06), 0.8))
    screech = lambda f, d=0.5: trem(lp(tone(f, d, "saw", a=0.03, tau=0.4, glide=0.9, glide_t=0.2, vib=0.01, vib_rate=7), 3000), 11, 0.3)
    return mk(thump(60, 0.16, 1.0, 1.5, 0.05),
              L((0, creak(150, 500, 0.8, 30), 1), (0.6, heart(), 0.8)),
              L((0, creak(500, 140, 0.7, 25), 1), (0.6, thump(45, 0.4, 1.0, 1.3, 0.14), 1)),
              echo(L((0, screech(hz(75)), 0.8), (0, screech(hz(76)), 0.8), (0, screech(hz(81)), 0.6), (0, thump(40, 0.5, 1.0, 1.4, 0.2), 1)), 0.25, 0.3, 2))


def p_starfighter():
    laser = lambda f0=2200, f1=350, d=0.18: sweep(f0, f1, d, "sq", a=0.001, rel=0.04, duty=0.4, vol=0.7, tau=d)
    boom = lambda: L((0, lp_sweep(noise(0.8, a=0.002, tau=0.25), 3500, 250, 0.2), 1.1), (0, thump(50, 0.6, 1.0, 2.2, 0.15), 1))
    power = lambda: stagger(lambda f, d: tone(f, d, "sq", duty=0.25, a=0.001, tau=0.1), [72, 76, 79, 84, 88], 0.05, 0.2)
    return mk(laser(1800, 600, 0.08), L((0, laser(), 1), (0.12, laser(), 0.9), (0.24, laser(2600, 500), 0.9)),
              L((0, boom(), 1)), L((0, power(), 0.8), (0.3, laser(900, 3000, 0.25), 0.8)))


def p_shadow_steps():
    step = lambda f=180, vol=1.0: layer([(0, lp(noise(0.07, a=0.003, tau=0.02), 700), 1.2 * vol), (0, thump(f, 0.05, 0.6 * vol, 1.2, 0.015), 1)])
    dull = lambda f: lp(tone(f, 0.18, a=0.004, tau=0.09, h2=0.2), 900)
    return mk(step(210, 0.9), L((0, step(170), 1), (0.3, step(190), 0.9), (0.62, dull(180), 0.8)),
              L((0, step(190), 0.9), (0.3, step(170), 1), (0.62, dull(120), 0.8)),
              echo(L((0, step(180), 0.8), (0.28, dull(330), 1), (0.4, dull(440), 0.8)), 0.2, 0.3, 2))


def p_sports_whistle():
    def whistle(d=0.35, f=2300):
        n = ns(d)
        out = [0.0] * n
        ph = 0.0
        for i in range(n):
            t = i / SR
            ph += f * (1 + 0.02 * math.sin(TAU * 30 * t)) / SR
            out[i] = math.sin(TAU * ph + 1.2 * math.sin(TAU * 60 * t)) * (0.6 + 0.4 * math.sin(TAU * 38 * t)) * min(1.0, t / 0.02, (d - t) / 0.03)
        return out
    buzz = lambda d=0.4: lp(tone(180, d, "saw", a=0.005, tau=1e9, rel=0.03, h2=0.2), 1500)
    dribble = lambda: layer([(0, thump(140, 0.1, 1.0, 1.3, 0.03), 1), (0, click_n(900, 1.0, 0.02, 0.005), 0.4)])
    swoosh = lambda: swish(1500, 4500, 0.2, 1.0, 0.5)
    return mk(dribble(), L((0, whistle(0.18), 0.9), (0.28, whistle(0.45), 1)), L((0, buzz(0.5), 0.8)),
              L((0, dribble(), 1), (0.15, dribble(), 0.8), (0.3, swoosh(), 0.9), (0.45, whistle(0.3, 2600), 0.8)))


def p_rhythm_hits():
    tapv = lambda f, g=1.0: layer([(0, click_n(2800, 1.5, 0.015, 0.004), g), (0, tone(f, 0.12, a=0.001, tau=0.04, h2=0.3), 0.8 * g)])
    return mk(tapv(1320), L((0, tapv(660, 0.8), 1), (0.25, tapv(660, 0.8), 1), (0.5, tapv(660, 0.8), 1), (0.75, tapv(990), 1.2)),
              L((0, thump(160, 0.3, 1.0, 0.7, 0.2), 1), (0.0, bp(noise(0.3, a=0.01, tau=0.1, vol=0.5), 600, 1.0), 0.8)),
              L((0, tapv(1320), 1), (0.07, tapv(1760), 1), (0.14, stagger(lambda f, d: tone(f, d, a=0.001, tau=0.1, h2=0.3), [84, 88, 91], 0.05, 0.4), 0.9)))


def p_level_up():
    v = lambda f, d: layer([(0, tone(f, d, "sq", duty=0.25, a=0.002, tau=d * 0.5), 0.5), (0, additive(f * 2, d, [(1, 0.5, 0.3), (3.0, 0.2, 0.15)]), 0.5)])
    return mk(tone(1568, 0.07, a=0.001, tau=0.03, h2=0.4), stagger(v, [60, 64, 67, 72, 76], 0.07, 0.35), stagger(v, [76, 72, 67, 64, 60], 0.07, 0.25),
              L((0, stagger(v, [67, 72, 76, 79], 0.08, 0.3), 1), (0.4, chord(v, [72, 76, 79, 84], 0.9), 1)))


def p_treasure_chest():
    clink = lambda f: additive(f, 0.25, [(1, 1, 0.08), (2.6, 0.5, 0.05), (4.4, 0.25, 0.03)])
    pour = lambda: layer([(k * 0.03 + R.random() * 0.02, clink(R.choice((1900, 2300, 2700, 3100))), 0.4 + 0.6 * R.random()) for k in range(14)])
    sparkle = lambda: stagger(lambda f, d: additive(f, d, [(1, 1, 0.2), (2.0, 0.3, 0.12)]), [84, 88, 91, 96], 0.06, 0.5)
    return mk(clink(2400), L((0, creak(260, 520, 0.5, 35), 1), (0.5, thump(80, 0.1), 1), (0.55, sparkle(), 0.8)),
              L((0, creak(520, 260, 0.4, 35), 0.9), (0.4, thump(70, 0.15), 1.1), (0.4, click_n(900, 1.0, 0.02, 0.005), 0.6)),
              L((0, pour(), 1), (0.35, sparkle(), 0.9)))


def p_quiz_show():
    ding = lambda f=1320: layer([(0, tone(f, 0.5, a=0.001, tau=0.2, fm=(1, 1.2, 0.1)), 1), (0, tone(f * 1.5, 0.5, a=0.001, tau=0.15), 0.5)])
    wrong = lambda: lp(layer([(0, tone(110, 0.5, "saw", a=0.005, tau=1e9, rel=0.05), 1), (0, tone(116, 0.5, "saw", a=0.005, tau=1e9, rel=0.05), 1)]), 900)
    tk = lambda f=900: tone(f, 0.04, a=0.0005, tau=0.012)
    return mk(tk(1100), L(*[(k * 0.14, tk(900 + (k == 3) * 600), 1) for k in range(4)]), L((0, wrong(), 0.9)), L((0, ding(1047), 1), (0.12, ding(1568), 1)))


# ---------------------------------------------------------------- holiday
def p_sleigh_bells():
    def jingle(d=0.5, fc=4200):
        rng = [bp(noise(d, a=0.003, tau=d * 0.3, vol=1.0), fc * r, 6.0) for r in (0.8, 1.0, 1.3)]
        return layer([(k * 0.011, x, 0.7) for k, x in enumerate(rng)])
    bell1 = lambda f=3000: additive(f, 0.3, [(1, 1, 0.1), (2.3, 0.4, 0.07), (3.8, 0.2, 0.04)])
    return mk(bell1(), L((0, jingle(0.3), 1), (0.15, jingle(0.4, 4600), 1), (0.32, jingle(0.6, 3800), 1)),
              L((0, jingle(0.4, 4600), 1), (0.17, jingle(0.5, 3600), 0.9)), L((0, jingle(0.3), 1), (0.12, bell1(2600), 0.8), (0.2, bell1(3500), 0.8)))


def p_spooky_organ():
    v = lambda f, d: trem(additive(f, d, [(1, 1, 9), (2, 0.5, 9), (3, 0.7, 9), (5, 0.3, 9), (7, 0.15, 9)], a=0.05, rel=0.15, vib=0.4), 5.5, 0.3)
    wail = lambda d=0.9: lp(sweep(300, 900, d, "saw", a=0.2, rel=0.3, vol=0.5), 1800)
    return mk(v(hz(55), 0.12), echo(chord(v, [45, 51, 57], 0.7), 0.25, 0.3, 2), echo(chord(v, [57, 51, 45], 0.6), 0.25, 0.3, 2),
              echo(L((0, chord(v, [48, 54, 55, 60], 0.6), 1), (0.3, wail(), 0.5)), 0.25, 0.3, 2))


def p_fireworks():
    launch = lambda: layer([(0, sweep(500, 2200, 0.5, "sine", a=0.05, rel=0.05, vol=0.3), 0.8), (0, bp(noise(0.5, a=0.05, tau=1.0, vol=0.4), 3000, 1.0), 0.6)])
    boom = lambda: layer([(0, thump(60, 0.45, 1.0, 2.0, 0.14), 1.1), (0, lp_sweep(noise(0.5, a=0.002, tau=0.16), 3500, 300, 0.12), 1.0)])
    crackle = lambda d=0.6: layer([(R.random() * d, click_n(R.choice((3000, 4500, 6000)), 2.0, 0.01, 0.003), 0.4 + 0.5 * R.random()) for _ in range(30)])
    return mk(layer([(0, thump(100, 0.05, 1.0, 1.6, 0.015), 1), (0, click_n(2200, 1.0, 0.01, 0.003), 0.8)]),
              L((0, launch(), 1), (0.5, boom(), 1), (0.55, crackle(), 0.7)), L((0, crackle(0.8), 0.7), (0.0, boom(), 0.5)),
              L((0, boom(), 1), (0.05, crackle(0.5), 0.8), (0.1, stagger(lambda f, d: additive(f, d, [(1, 1, 0.2)]), [96, 100, 103], 0.07, 0.4), 0.5)))


# ---------------------------------------------------------------- asmr
def rub(d, lo, hi, depth=0.6, vol=1.0, pk=None):
    n = ns(d)
    x = bp_sweep([R.uniform(-1, 1) for _ in range(n)], lo, hi, 1.2)
    out = []
    ph = R.random() * 6
    for i, v in enumerate(x):
        e = (1 - depth + depth * (0.5 + 0.5 * math.sin(ph + i / SR * R.choice((38, 38)))))
        out.append(v * e * math.sin(math.pi * i / n) ** 1.5 * vol)
    return out


def p_book_pages():
    page = lambda d=0.45, up=True: rub(d, 900, 3800, 0.5, 1.4) if up else rub(d, 3800, 900, 0.5, 1.4)
    thud = lambda: layer([(0, thump(95, 0.1, 1.0, 1.2, 0.03), 0.9), (0, lp(noise(0.08, a=0.003, tau=0.025), 1100), 0.7)])
    return mk(rub(0.1, 1500, 3500, 0.2, 1.4), L((0, thud(), 0.8), (0.12, page(0.5), 1)), L((0, page(0.4, False), 1), (0.4, thud(), 1)),
              L((0, page(0.25), 1), (0.3, page(0.25), 0.9), (0.55, knock(300, 0.05, 0.6, 1200), 0.8)))


def p_pencil_sketch():
    stroke = lambda d=0.25: rub(d, 2500, 5500, 0.8, 1.2)
    erase = lambda d=0.3: rub(d, 700, 1600, 0.4, 1.2)
    tipt = lambda: layer([(0, knock(1100, 0.03, 1.0, 3000), 1)])
    return mk(tipt(), L((0, tipt(), 1), (0.1, stroke(0.2), 1), (0.35, stroke(0.3), 0.9)), L((0, erase(), 1), (0.3, tipt(), 0.8)),
              L((0, stroke(0.12), 1), (0.15, stroke(0.12), 1), (0.3, tipt(), 1)))


def p_cloth_rustle():
    fab = lambda d=0.4, lo=400, hi=1800: rub(d, lo, hi, 0.9, 1.2)
    return mk(fab(0.08, 600, 1400), L((0, fab(0.5, 300, 1600), 1), (0.45, fab(0.2, 800, 2000), 0.7)), L((0, fab(0.5, 1800, 300), 1)),
              L((0, fab(0.15, 500, 1800), 1), (0.18, fab(0.3, 800, 2400), 1), (0.4, knock(250, 0.04, 0.5, 1000), 0.5)))


def p_mic_brush():
    br = lambda d=0.6, lo=700, hi=3500: layer([(0, rub(d, lo, hi, 0.3, 1.0), 1), (0, lp(noise(d, a=d * 0.4, tau=d * 0.6, vol=0.4), 300), 0.7)])
    foam = lambda: layer([(0, lp(noise(0.08, a=0.004, tau=0.025), 900), 1.2), (0, thump(120, 0.06, 0.6, 1.2, 0.02), 0.6)])
    return mk(foam(), L((0, br(0.7, 600, 3800), 1)), L((0, br(0.7, 3800, 600), 1)), L((0, foam(), 1), (0.12, foam(), 0.9), (0.25, br(0.5), 0.9)))


def p_finger_taps():
    tp = lambda f=900, g=1.0: layer([(0, tone(f, 0.06, a=0.001, tau=0.015, h2=0.2), 0.8 * g), (0, click_n(f * 2.5, 1.5, 0.008, 0.002), 0.4 * g)])
    scr = lambda: layer([(k * 0.016, click_n(2600 + 120 * (k % 4), 2.0, 0.008, 0.002), 0.5 + 0.1 * (k % 3)) for k in range(14)])
    return mk(tp(1000), L((0, tp(800), 0.9), (0.09, tp(900), 1), (0.18, tp(1000), 1), (0.27, tp(1150), 1)),
              L((0, tp(1150), 1), (0.09, tp(900), 1), (0.18, tp(700), 0.9)), L((0, scr(), 0.7), (0.26, tp(1000), 1), (0.35, tp(1300), 1)))


# ---------------------------------------------------------------- kids
def p_squeaky_toys():
    def sq(f0, f1, d=0.15):
        n = ns(d)
        out = [0.0] * n
        ph = 0.0
        for i in range(n):
            t = i / n
            f = f0 + (f1 - f0) * t + 60 * math.sin(TAU * 25 * i / SR)
            ph += f / SR
            out[i] = (math.sin(TAU * ph) + 0.35 * math.sin(2 * TAU * ph) + 0.15 * math.sin(3 * TAU * ph)) * min(1.0, i / 80, (n - i) / 200)
        return out
    return mk(sq(900, 1500, 0.07), L((0, sq(700, 1300), 1), (0.17, sq(1000, 1700, 0.2), 1)), L((0, sq(1500, 900), 1), (0.17, sq(1200, 600, 0.2), 1)),
              L((0, sq(800, 1400, 0.1), 1), (0.12, sq(1100, 1800, 0.1), 1), (0.24, sq(1400, 2000, 0.25), 1)))


def p_boing_springs():
    def boing(f, d=0.5, rate=22.0):
        n = ns(d)
        out = [0.0] * n
        ph = 0.0
        for i in range(n):
            t = i / SR
            fr = f * (1 + 0.25 * math.exp(-t / 0.15) * math.sin(TAU * rate * t))
            ph += fr / SR
            out[i] = (math.sin(TAU * ph) + 0.3 * math.sin(2 * TAU * ph)) * math.exp(-t / (d * 0.5)) * min(1.0, i / 30, (n - i) / 250)
        return out
    return mk(boing(500, 0.15, 30), L((0, boing(260, 0.6), 1), (0.3, boing(390, 0.5, 26), 0.9)), L((0, boing(430, 0.5), 1), (0.25, boing(250, 0.6, 18), 1)),
              L((0, boing(330, 0.4, 28), 1), (0.12, boing(520, 0.7, 24), 1)))


def p_kazoo_fun():
    def kz(f, d=0.2):
        x = tone(f, d, "saw", a=0.012, tau=1e9, vib=0.01, vib_rate=6, rel=0.05)
        return trem(layer([(0, bp(x, 1300, 2.5), 1.6), (0, lp(x, 2500), 0.3)]), 90, 0.4)
    ph = lambda ms, g, d=0.14, last=0.4: layer([(k * g, kz(hz(m), last if k == len(ms) - 1 else d), 1) for k, m in enumerate(ms)])
    return mk(kz(hz(72), 0.08), ph([60, 64, 67, 72], 0.13), ph([72, 67, 64, 60], 0.13), ph([67, 72, 67, 76], 0.12, 0.12, 0.5))


# ---------------------------------------------------------------- cinematic
def p_cine_whoosh():
    w = lambda lo, hi, d, sh=0.55: layer([(0, swish(lo, hi, d, 0.9, sh), 1.3), (0, lp(tone(lo / 4, d, a=d * 0.5, tau=1e9, rel=d * 0.4), 400), 0.4)])
    return mk(swish(1500, 4500, 0.1, 1.0, 0.3), L((0, w(300, 4500, 0.9, 0.75), 1)), L((0, w(4500, 300, 0.9, 0.25), 1)),
              L((0, w(400, 3500, 0.35, 0.7), 1), (0.3, w(3500, 300, 0.4, 0.2), 0.8)))


def p_trailer_hit():
    hit = lambda: layer([(0, thump(45, 1.0, 1.0, 2.6, 0.3), 1.3), (0, lp_sweep(noise(0.9, a=0.001, tau=0.3), 4000, 300, 0.25), 1.0),
                         (0, additive(300, 1.2, [(1, 0.4, 0.5), (1.5, 0.3, 0.4), (2.2, 0.2, 0.3)]), 0.3)])
    return mk(layer([(0, thump(70, 0.15, 1.0, 2.0, 0.05), 1), (0, click_n(1500, 1.0, 0.02, 0.005), 0.5)]),
              L((0, swish(300, 3500, 0.7, 1.0, 0.95), 1.2), (0.72, hit(), 1)), L((0, hit(), 0.7), (0.0, sweep(180, 40, 0.9, "sine", vol=0.6), 0.6)),
              L((0, hit(), 1), (0.35, thump(50, 0.5, 0.7, 1.8, 0.15), 0.7)))


def p_tension_riser():
    def rise(d, f0, f1):
        x = layer([(k * 0.0, sweep(f0 * (1 + 0.01 * k), f1 * (1 + 0.01 * k), d, "saw", a=d * 0.3, rel=0.05, vol=0.3), 1) for k in range(4)])
        return layer([(0, lp(x, 3500), 1), (0, swish(400, 5000, d, 0.9, 0.97), 0.7)])
    pulse = lambda d: layer([(k * (0.2 - 0.12 * k / (d / 0.15)), thump(70, 0.08), 0.6) for k in range(5)])
    return mk(tone(1200, 0.04, a=0.001, tau=0.01), rise(1.3, 150, 900), L((0, sweep(900, 120, 0.9, "saw", a=0.01, rel=0.1, vol=0.5), 1), (0.85, thump(45, 0.5, 1.0, 1.5, 0.15), 1)),
              L((0, rise(0.7, 220, 1200), 1), (0.72, kick(50, 0.4), 1), (0.72, lp(noise(0.4, a=0.001, tau=0.1), 2500), 0.7)))


def p_sub_drop():
    drop = lambda f=90, d=1.2: sat(gain(tone(f, d, a=0.002, tau=0.5, glide=0.45, glide_t=0.4, h2=0.2), 1.8), 1.0)
    return mk(thump(60, 0.15, 1.0, 1.8, 0.05), L((0, sweep(60, 220, 0.5, "sine", a=0.1, rel=0.05, vol=0.6), 1), (0.5, drop(), 1.1)),
              L((0, drop(110, 0.9), 1)), L((0, drop(), 1), (0.0, lp(noise(0.3, a=0.001, tau=0.08), 1500), 0.8)))


def p_orchestral_swell():
    def st(f, d, a=0.4):
        x = layer([(0, tone(f * (1 + 0.004 * k), d, "saw", a=a, tau=1e9, rel=0.2, vib=0.004), 0.3) for k in (-1, 0, 1)])
        return lp(x, 2200)
    ch = lambda ms, d, a: chord(lambda f, dd: st(f, dd, a), ms, d, 0.0)
    tim = lambda: layer([(0, thump(80, 0.5, 1.0, 1.5, 0.15), 1), (0, lp(noise(0.1, a=0.001, tau=0.03), 1500), 0.7)])
    return mk(lp(tone(hz(60), 0.12, "saw", a=0.01, tau=0.06), 1800), L((0, ch([48, 55, 60, 64, 67], 1.5, 0.7), 1)), L((0, ch([67, 64, 60, 55, 48], 1.2, 0.05), 1)),
              L((0, tim(), 1), (0.0, ch([48, 55, 60, 64, 72], 1.2, 0.08), 1)))


# ---------------------------------------------------------------- accessibility
def pure(f, d, a=0.006, rel=0.02, vol=1.0, h2=0.0):
    return tone(f, d, "sine", a=a, tau=1e9, rel=rel, h2=h2, vol=vol)


def p_clear_tones():
    return mk(pure(880, 0.08), L((0, pure(660, 0.12), 1), (0.14, pure(990, 0.2), 1)), L((0, pure(990, 0.12), 1), (0.14, pure(660, 0.2), 1)),
              L((0, pure(660, 0.12), 1), (0.14, pure(830, 0.12), 1), (0.28, pure(990, 0.3), 1)))


def p_soft_lows():
    v = lambda f, d: lp(tone(f, d, "sine", a=0.02, tau=1e9, rel=0.05, h2=0.3), 800)
    return mk(v(220, 0.1), L((0, v(196, 0.15), 1), (0.17, v(262, 0.3), 1)), L((0, v(262, 0.15), 1), (0.17, v(196, 0.3), 1)),
              L((0, v(196, 0.14), 1), (0.16, v(247, 0.14), 1), (0.32, v(294, 0.35), 1)))


def p_pulse_code():
    b = lambda f=800, d=0.07: tone(f, d, "tri", a=0.004, tau=1e9, rel=0.015)
    return mk(b(), L((0, b(), 1), (0.12, b(), 1)), L((0, b(700), 1), (0.12, b(700), 1), (0.24, b(700), 1)), L((0, b(800, 0.3), 1)))


def p_pitch_ladder():
    t_ = lambda m, d=0.1: tone(hz(m), d, "tri", a=0.005, tau=1e9, rel=0.025)
    return mk(t_(72, 0.07), stagger(lambda f, d: tone(f, d, "tri", a=0.005, tau=1e9, rel=0.025), [67, 72, 76], 0.1, 0.11),
              stagger(lambda f, d: tone(f, d, "tri", a=0.005, tau=1e9, rel=0.025), [76, 72, 67], 0.1, 0.11),
              L((0, t_(67, 0.12), 1), (0.14, t_(79, 0.3), 1)))


# ---------------------------------------------------------------- materials
def p_stone_tap():
    v = lambda f, d=0.4: layer([(0, additive(f, d, [(1, 1, 0.12), (2.3, 0.5, 0.06), (3.7, 0.25, 0.04)]), 1), (0, knock(f * 0.5, 0.03, 0.6, 2000), 0.6)])
    return std(v, [55, 58, 62, 65], [65, 62, 58, 55], [58, 62, 67], 0.1, last=0.4, click=knock(500, 0.05, 1.0, 2500))


def p_marble_clack():
    cl = lambda f=3200: additive(f, 0.12, [(1, 1, 0.012), (1.9, 0.4, 0.008)])
    bounce = lambda n, g0: layer([(sum(g0 * 0.72 ** j for j in range(k)), cl(3000 + 150 * (k % 3)), 0.9 ** k) for k in range(n)])
    return mk(cl(), bounce(7, 0.14), bounce(7, 0.1), L((0, cl(2800), 1), (0.05, cl(3400), 1), (0.15, bounce(5, 0.07), 0.8)))


def p_rubber_bounce():
    bn = lambda f=140: layer([(0, tone(f, 0.12, a=0.001, tau=0.04, glide=1.5, glide_t=0.03), 1), (0, lp(noise(0.04, a=0.001, tau=0.01), 800), 0.5)])
    ball = lambda n, g0, f=140: layer([(sum(g0 * 0.7 ** j for j in range(k)), bn(f * (1 + 0.08 * k)), 0.85 ** k) for k in range(n)])
    twang = lambda: bend(tone(300, 0.3, a=0.001, tau=0.12), 500, 0.1)
    return mk(bn(), ball(6, 0.3), ball(6, 0.22, 110), L((0, twang(), 1), (0.25, bn(180), 1)))


def p_gel_squish():
    sq = lambda f0, f1, d=0.2: layer([(0, sweep(f0, f1, d, "sine", a=0.01, rel=0.03, vol=0.9, tau=d * 0.8), 1), (0, bp(noise(d, a=0.005, tau=d * 0.4, vol=0.6), 700, 1.2), 0.6)])
    return mk(sq(300, 700, 0.07), L((0, sq(200, 600, 0.25), 1), (0.2, sq(300, 800, 0.2), 0.9)), L((0, sq(800, 200, 0.3), 1)),
              L((0, sq(250, 900, 0.15), 1), (0.12, sq(400, 1200, 0.15), 1), (0.26, thump(300, 0.05, 0.7, 1.4, 0.02), 0.8)))


def p_crystal_gems():
    g = lambda f, d=0.5: additive(f, d, [(1, 1, 0.3), (2.76, 0.5, 0.2), (5.4, 0.25, 0.1), (8.9, 0.1, 0.05)], a=0.001)
    scatter = lambda n: layer([(k * 0.03 + R.random() * 0.02, g(R.choice((1200, 1600, 2000, 2600)), 0.2), 0.5 + 0.5 * R.random()) for k in range(n)])
    return mk(g(1800, 0.12), L((0, scatter(10), 1)), L((0, scatter(8), 0.9), (0.0, g(900, 0.5), 0.7)), L((0, g(1568, 0.7), 1), (0.1, g(2349, 0.7), 0.8), (0.12, scatter(6), 0.6)))


def p_sand_shift():
    gr = lambda d, fc=3500, a=None: bp(noise(d, a=a or d * 0.3, tau=d * 0.5, vol=1.0), fc, 0.5)
    return mk(gr(0.06, 4000, 0.004), L((0, gr(0.7, 2500), 1.4)), L((0, gr(0.7, 4500, 0.1), 1.4), (0.5, thump(70, 0.1, 0.5), 0.5)),
              L((0, gr(0.12, 3500, 0.01), 1), (0.15, gr(0.3, 3000, 0.05), 1.2)))


def p_leather_creak():
    snap = lambda: layer([(0, click_n(2200, 1.0, 0.015, 0.004), 1), (0, thump(180, 0.04, 0.8, 1.2, 0.01), 0.7)])
    return mk(snap(), L((0, creak(180, 360, 0.5, 22), 1.3), (0.5, snap(), 0.9)), L((0, creak(360, 160, 0.45, 25), 1.3), (0.4, thump(100, 0.1), 0.8)),
              L((0, snap(), 1), (0.1, creak(220, 300, 0.2, 30), 1)))


def p_cork_pop():
    pop = lambda f=520: layer([(0, thump(70, 0.06, 1.0, 1.5, 0.02), 0.8), (0, tone(f, 0.25, a=0.002, tau=0.07, glide=0.6, glide_t=0.08), 1)])
    glug = lambda n: layer([(k * 0.08, tone(300 + 60 * (k % 3), 0.1, a=0.003, tau=0.03, glide=0.7, glide_t=0.05), 0.8) for k in range(n)])
    clink = lambda f=2400: additive(f, 0.3, [(1, 1, 0.1), (2.4, 0.4, 0.06)])
    return mk(pop(700), L((0, pop(), 1), (0.25, clink(), 0.8)), L((0, glug(5), 1), (0.45, clink(1900), 0.7)), L((0, clink(2400), 1), (0.1, pop(600), 1)))


def p_porcelain_clink():
    c = lambda f, d=0.5: additive(f, d, [(1, 1, 0.1), (2.76, 0.4, 0.07), (5.4, 0.15, 0.04)])
    stir = lambda: layer([(k * 0.07, c(2300 + 100 * (k % 3), 0.1), 0.6) for k in range(6)])
    return mk(c(2600, 0.15), L((0, c(2000, 0.4), 1), (0.12, c(2600, 0.5), 1)), L((0, stir(), 1), (0.4, c(1700, 0.3), 0.8)),
              L((0, c(2400), 1), (0.1, c(3000, 0.6), 1)))


def p_chess_pieces():
    tok = lambda f=300, g=1.0: layer([(0, tone(f, 0.12, a=0.001, tau=0.03, glide=1.1, glide_t=0.01), g), (0, click_n(1500, 1.0, 0.02, 0.004), 0.7 * g)])
    slide = lambda: rub(0.2, 600, 1800, 0.3, 0.8)
    clk = lambda: layer([(0, click_n(2000, 1.5, 0.015, 0.004), 1), (0, thump(160, 0.04, 0.9, 1.2, 0.012), 0.8)])
    return mk(tok(), L((0, slide(), 0.7), (0.2, tok(), 1), (0.4, clk(), 0.8)), L((0, clk(), 0.8), (0.1, slide(), 0.6), (0.3, tok(260), 1)),
              L((0, tok(), 1), (0.08, tok(340, 0.9), 1), (0.2, clk(), 1)))


def p_domino_clack():
    cl = lambda f=1100, g=1.0: layer([(0, click_n(f, 1.5, 0.02, 0.004), g), (0, tone(f / 4, 0.04, a=0.0005, tau=0.01), 0.5 * g)])
    cascade_ = lambda n, g0: layer([(sum(g0 * 0.93 ** j for j in range(k)), cl(900 + 40 * (k % 5), 0.6 + 0.4 * ((k % 2))), 1) for k in range(n)])
    return mk(cl(), cascade_(14, 0.06), cascade_(10, 0.05), L((0, cascade_(8, 0.05), 1), (0.45, thump(100, 0.1, 1.0, 1.3, 0.03), 1)))

# ----------------------------------------------------------------------------- wave 2: catalog
# id, name, description, categories, pro, featured, hue, builder
W2_PACKS = [
    ("snd-sitar", "Sitar", "Buzzing, gliding strings in a raga-like scale with a drifting drone loop.", ["world", "musical"], True, True, 0.07, p_sitar),
    ("snd-tabla", "Tabla", "Bright treble taps and a bending bass drum, played as short rhythmic phrases.", ["world", "musical"], False, False, 0.03, p_tabla),
    ("snd-oud", "Oud", "Warm double-course lute plucks in a Middle Eastern mode.", ["world", "musical"], True, False, 0.09, p_oud),
    ("snd-erhu", "Erhu", "A two-string bowed fiddle that sings and slides between notes.", ["world", "musical"], True, False, 0.95, p_erhu),
    ("snd-balafon", "Balafon", "Wooden bars with a buzzing gourd resonance in a bouncy pentatonic.", ["world", "musical"], False, False, 0.11, p_balafon),
    ("snd-didgeridoo", "Drone Pipe", "A deep, vowel-like drone that opens and closes like a breath.", ["world", "ambient"], False, False, 0.08, p_didgeridoo),
    ("snd-bagpipe", "Highland Pipes", "A nasal chanter with grace notes over a steady drone.", ["world", "musical"], False, False, 0.31, p_bagpipe),
    ("snd-accordion", "Accordion", "Breathing reeds with a musette shimmer and bellows swells.", ["world", "musical"], True, False, 0.00, p_accordion),
    ("snd-harmonica", "Harmonica", "Breathy blues-harp notes with a bend into the pitch.", ["world", "musical"], False, False, 0.58, p_harmonica),
    ("snd-ukulele", "Ukulele", "Bright, sunny four-string strums, down and up.", ["world", "musical", "soft"], False, False, 0.13, p_ukulele),
    ("snd-banjo", "Banjo", "Twangy picked rolls with a drum-head snap.", ["world", "musical"], False, False, 0.10, p_banjo),
    ("snd-taiko", "Taiko", "Big stretched-skin drum hits with sharp rim clicks.", ["world", "mechanical"], True, True, 0.02, p_taiko),
    ("snd-cajon", "Cajon", "A wooden box drum: deep bass thumps and snappy slaps.", ["world", "mechanical"], False, False, 0.07, p_cajon),
    ("snd-latin-percussion", "Latin Percussion", "Cowbell, claves, guiro scrapes and shakers in a lively groove.", ["world", "mechanical"], False, False, 0.12, p_latin_percussion),
    ("snd-pan-flute", "Pan Flute", "Breathy bamboo pipes with a soft chiff at the start of each note.", ["world", "soft", "ambient"], True, False, 0.36, p_pan_flute),
    ("snd-celesta", "Celesta", "A tiny hammered-bell keyboard, like a lullaby in a snow globe.", ["musical", "soft"], True, False, 0.57, p_celesta),
    ("snd-harpsichord", "Harpsichord", "Plucked-key baroque tones with a dry jack click and little trills.", ["musical"], False, False, 0.09, p_harpsichord),
    ("snd-pipe-organ", "Pipe Organ", "A cathedral organ with breathy chiff and a long echoing tail.", ["musical", "ambient"], True, False, 0.62, p_pipe_organ),
    ("snd-church-bells", "Church Bells", "Heavy tower bells ringing out in a slow, descending peal.", ["musical", "ambient"], True, False, 0.12, p_church_bells),
    ("snd-house-stabs", "House Stabs", "Filtered chord stabs over a four-on-the-floor kick.", ["electronic"], False, False, 0.78, p_house_stabs),
    ("snd-techno-blips", "Techno Blips", "Dark, saturated blips with a ticking echo and a heavy kick.", ["electronic"], False, False, 0.60, p_techno_blips),
    ("snd-dubstep-wobble", "Wobble Bass", "Growling bass that wobbles open and shut with a sub thump.", ["electronic"], True, False, 0.33, p_dubstep_wobble),
    ("snd-dnb-breaks", "Jungle Breaks", "Chopped fast breakbeats with a deep reese bass.", ["electronic"], True, False, 0.28, p_dnb_breaks),
    ("snd-trance-plucks", "Trance Plucks", "Supersaw arpeggio plucks with a bouncing dotted echo.", ["electronic"], True, True, 0.72, p_trance_plucks),
    ("snd-idm-glitch", "Micro Glitch", "Tiny bit-crushed grains and clicks in off-beat patterns.", ["electronic", "sci-fi"], False, False, 0.45, p_idm_glitch),
    ("snd-acid-squelch", "Acid Squelch", "A resonant, sweeping filter on a rolling saw bass line.", ["electronic", "retro"], True, False, 0.17, p_acid_squelch),
    ("snd-future-bass", "Future Bass", "Wide, pumping super-saw chords with a bright, melodic lift.", ["electronic"], True, True, 0.88, p_future_bass),
    ("snd-ambient-pads", "Drift Pads", "Slow swelling synth pads washed in soft echo.", ["electronic", "ambient", "soft"], False, False, 0.51, p_ambient_pads),
    ("snd-hyperpop", "Hyper Pop", "Sugary, distorted leads, crushed chimes and a bright snap.", ["electronic"], True, False, 0.91, p_hyperpop),
    ("snd-trap-hats", "Trap Hats", "Rolling hi-hat patterns, snaps and a sliding sub bass.", ["electronic"], False, False, 0.40, p_trap_hats),
    ("snd-drum-box", "Drum Box", "Boomy analog drum-machine kicks, toms, claps and cowbell.", ["electronic", "retro"], False, False, 0.05, p_drum_box),
    ("snd-floppy-seek", "Floppy Seek", "Stepper-motor chatter and head seeks of an old disk drive.", ["hardware", "retro"], False, False, 0.55, p_floppy_seek),
    ("snd-crt-degauss", "CRT Degauss", "The thump, hum and rising whine of a tube monitor.", ["hardware", "retro"], False, False, 0.48, p_crt_degauss),
    ("snd-dot-matrix", "Dot Matrix", "Zipping print-head buzz and a ratcheting paper feed.", ["hardware", "retro"], False, False, 0.70, p_dot_matrix),
    ("snd-pinball", "Pinball Table", "Flipper thwacks, bumper dings and a spring-loaded plunger.", ["hardware", "retro", "mechanical"], True, True, 0.02, p_pinball),
    ("snd-slot-reels", "Reel Spinner", "Slowing reel ticks, a heavy stop and a bright winning bell.", ["hardware", "mechanical"], False, False, 0.14, p_slot_reels),
    ("snd-jukebox", "Jukebox", "Selector relays, a record arm and a tinny tune through a small speaker.", ["hardware", "retro"], True, False, 0.06, p_jukebox),
    ("snd-telegraph-morse", "Telegraph Key", "A brass sounder clacking out dots and dashes.", ["hardware", "mechanical"], False, False, 0.10, p_telegraph_morse),
    ("snd-cash-register", "Cash Register", "Key clacks, a sliding drawer and a ringing bell.", ["hardware", "mechanical"], False, False, 0.15, p_cash_register),
    ("snd-shortwave-radio", "Shortwave Radio", "Tuning whistles, crackling static and a locked-in carrier tone.", ["hardware", "retro", "sci-fi"], False, False, 0.42, p_shortwave_radio),
    ("snd-rpg-menu", "Quest Menu", "Harp-like plucks and bell sparkles for an adventure-style menu.", ["game", "musical"], True, True, 0.34, p_rpg_menu),
    ("snd-fighter-stinger", "Fight Stingers", "Punchy impacts, whooshes and brass hits for versus-style menus.", ["game", "cinematic"], False, False, 0.01, p_fighter_stinger),
    ("snd-racing-gears", "Racing Gears", "Shifter clunks, engine revs, a turbo whistle and blow-off.", ["game", "mechanical"], True, False, 0.04, p_racing_gears),
    ("snd-platform-hop", "Hop and Bounce", "Springy, elastic jumps and cheerful coin dings.", ["game", "kids"], False, False, 0.16, p_platform_hop),
    ("snd-puzzle-match", "Puzzle Match", "Soft pops and sparkling, rising chain-reaction notes.", ["game", "soft"], True, False, 0.50, p_puzzle_match),
    ("snd-horror-stinger", "Horror Stingers", "Creaks, heartbeats and a dissonant string screech.", ["game", "cinematic"], True, False, 0.75, p_horror_stinger),
    ("snd-starfighter", "Starfighter", "Laser bolts, booming explosions and power-up arpeggios.", ["game", "sci-fi"], False, False, 0.64, p_starfighter),
    ("snd-shadow-steps", "Shadow Steps", "Quiet footfalls and dull, muffled blips for stealth games.", ["game", "soft"], False, False, 0.68, p_shadow_steps),
    ("snd-sports-whistle", "Sports Whistle", "A referee whistle, a buzzer and a bouncing ball.", ["game", "mechanical"], False, False, 0.57, p_sports_whistle),
    ("snd-rhythm-hits", "Rhythm Hits", "Crisp tap feedback with a count-in and a perfect-hit sparkle.", ["game", "musical"], True, False, 0.82, p_rhythm_hits),
    ("snd-level-up", "Level Up", "Bright pulse-wave arpeggios and a triumphant little fanfare.", ["game", "retro"], True, True, 0.15, p_level_up),
    ("snd-treasure-chest", "Treasure Chest", "A creaking lid, spilling gold and a magical sparkle.", ["game", "mechanical"], False, False, 0.13, p_treasure_chest),
    ("snd-quiz-show", "Quiz Show", "Countdown ticks, a happy ding and a wrong-answer buzz.", ["game", "retro"], False, False, 0.20, p_quiz_show),
    ("snd-sleigh-bells", "Sleigh Bells", "Shaken winter jingles with tiny bright bells.", ["holiday", "soft"], True, False, 0.52, p_sleigh_bells),
    ("snd-spooky-organ", "Spooky Organ", "Haunted-house organ chords with a ghostly vibrato and a wail.", ["holiday", "musical"], True, False, 0.80, p_spooky_organ),
    ("snd-fireworks", "Fireworks", "Launch whistles, thumping bursts and a glittering crackle.", ["holiday"], False, False, 0.96, p_fireworks),
    ("snd-book-pages", "Book Pages", "Heavy page turns, a soft cover thump and a finger tap.", ["asmr", "soft"], False, False, 0.09, p_book_pages),
    ("snd-pencil-sketch", "Pencil Sketch", "Scratchy graphite strokes, eraser rubs and tip taps.", ["asmr", "soft"], False, False, 0.00, p_pencil_sketch),
    ("snd-cloth-rustle", "Cloth Rustle", "Gentle fabric folds and silky sweeps, very close and quiet.", ["asmr", "soft"], False, False, 0.88, p_cloth_rustle),
    ("snd-mic-brush", "Mic Brush", "Slow brushing across a microphone and soft foam taps.", ["asmr", "soft"], True, False, 0.25, p_mic_brush),
    ("snd-finger-taps", "Finger Taps", "Delicate fingertip taps and nail scratches on a hollow surface.", ["asmr", "soft"], False, False, 0.06, p_finger_taps),
    ("snd-squeaky-toys", "Squeaky Toys", "Rubber toy squeaks that chirp up and down.", ["kids", "soft"], False, False, 0.12, p_squeaky_toys),
    ("snd-boing-springs", "Boing Springs", "Wobbly springs that bounce and settle with a boing.", ["kids"], False, False, 0.17, p_boing_springs),
    ("snd-kazoo-fun", "Kazoo Party", "Buzzy, humming kazoo tunes for a silly mood.", ["kids", "musical"], False, False, 0.10, p_kazoo_fun),
    ("snd-cine-whoosh", "Cinema Whoosh", "Big transition sweeps with a low swell underneath.", ["cinematic"], False, False, 0.58, p_cine_whoosh),
    ("snd-trailer-hit", "Trailer Hit", "Deep boom impacts with a reverse swell lead-in.", ["cinematic"], True, True, 0.00, p_trailer_hit),
    ("snd-tension-riser", "Tension Riser", "Rising layers that build and release in a pounding hit.", ["cinematic", "sci-fi"], True, False, 0.78, p_tension_riser),
    ("snd-sub-drop", "Sub Drop", "A bass drop that falls through the floor.", ["cinematic", "electronic"], False, False, 0.69, p_sub_drop),
    ("snd-orchestral-swell", "Orchestral Swell", "Big string-section swells with a timpani hit.", ["cinematic", "musical"], True, False, 0.07, p_orchestral_swell),
    ("snd-clear-tones", "Clear Tones", "Pure, easily distinguished tones, each with its own clear pitch pattern.", ["accessibility"], False, False, 0.56, p_clear_tones),
    ("snd-soft-lows", "Soft Lows", "Warm low tones with no sharp highs. Gentle for sensitive hearing.", ["accessibility", "soft"], False, False, 0.09, p_soft_lows),
    ("snd-pulse-code", "Pulse Code", "Each action has its own count of beeps, easy to tell apart without looking.", ["accessibility"], False, False, 0.62, p_pulse_code),
    ("snd-pitch-ladder", "Pitch Ladder", "Open climbs, close descends and select leaps: direction you can hear.", ["accessibility"], False, False, 0.40, p_pitch_ladder),
    ("snd-stone-tap", "Stone Tap", "Dense, dry tapping on slate and river stones.", ["materials", "mechanical"], False, False, 0.08, p_stone_tap),
    ("snd-marble-clack", "Marble Clack", "Glass marbles clinking and bouncing to rest.", ["materials"], False, False, 0.53, p_marble_clack),
    ("snd-rubber-bounce", "Rubber Bounce", "A rubber ball bouncing and a twanging band.", ["materials", "kids"], False, False, 0.03, p_rubber_bounce),
    ("snd-gel-squish", "Gel Squish", "Squelchy, wet gel presses with a glossy slide.", ["materials", "soft"], False, False, 0.47, p_gel_squish),
    ("snd-crystal-gems", "Gem Cascade", "Scattering gems and a glittering, ringing pickup.", ["materials", "musical"], True, False, 0.63, p_crystal_gems),
    ("snd-sand-shift", "Sand Shift", "Fine grains pouring and shifting under a hand.", ["materials", "soft", "nature"], False, False, 0.11, p_sand_shift),
    ("snd-leather-creak", "Leather Creak", "A creaking saddle, a snapping strap and a soft thump.", ["materials", "mechanical"], False, False, 0.07, p_leather_creak),
    ("snd-cork-pop", "Cork Pop", "A bottle cork popping, glugging liquid and a glass clink.", ["materials", "soft"], False, False, 0.14, p_cork_pop),
    ("snd-porcelain-clink", "Porcelain Clink", "A teacup, a saucer and a stirring spoon.", ["materials", "soft"], False, False, 0.95, p_porcelain_clink),
    ("snd-chess-pieces", "Chess Pieces", "Wooden pieces knocking on a board and a clock press.", ["materials", "mechanical"], False, False, 0.08, p_chess_pieces),
    ("snd-domino-clack", "Domino Run", "A row of dominoes toppling in a quickening clatter.", ["materials", "mechanical"], False, False, 0.02, p_domino_clack),
]

# id, name, desc, categories, pro, featured, hue, builder
NEW_PACKS = [
    ("snd-pulse-quest", "Pulse Quest", "Thin 12.5% pulse-wave arpeggios with a triangle bass, like a tiny fantasy adventure.", ["retro"], False, False, 0.30, p_pulse_quest),
    ("snd-chip-drums", "Chip Drums", "Noise-channel kicks, snares and hats in a punchy tracker-style groove.", ["retro", "mechanical"], False, False, 0.02, p_chip_drums),
    ("snd-pc-speaker", "PC Speaker", "Bare square-wave beeps that step through pitches, no envelope at all.", ["retro"], False, False, 0.13, p_pc_speaker),
    ("snd-wavetable-pocket", "Wavetable Pocket", "Grainy 4-bit custom waveforms with a bouncy bass note on confirm.", ["retro"], False, False, 0.40, p_wavetable_pocket),
    ("snd-coin-slot", "Coin Slot", "Metallic coin drops, a ratchet slot and a short jingle for confirms.", ["retro", "mechanical"], False, False, 0.11, p_coin_slot),
    ("snd-sixteen-bit", "Sixteen Bit", "Warm FM chord stabs crunched to a sampled-cartridge sound.", ["retro", "musical"], True, False, 0.67, p_sixteen_bit),
    ("snd-pwm-tracker", "PWM Tracker", "Fast chord arpeggios with swirling pulse-width modulation.", ["retro", "musical"], True, False, 0.78, p_pwm_tracker),
    ("snd-fm-bells", "FM Bells", "Glassy frequency-modulated bells with a floating ambient loop.", ["musical", "ambient"], True, False, 0.50, p_fm_bells),
    ("snd-rhodes", "Jazz E-Piano", "Jazzy electric-piano chords with a tine bark and a late-night loop.", ["musical", "ambient"], True, True, 0.06, p_rhodes),
    ("snd-vibraphone", "Vibraphone", "Mellow metal bars shimmering with a slow motor tremolo.", ["musical"], True, False, 0.58, p_vibraphone),
    ("snd-glockenspiel", "Glockenspiel", "Bright, tiny metal bars that sparkle in quick runs.", ["musical", "soft"], False, False, 0.55, p_glockenspiel),
    ("snd-drawbar-organ", "Drawbar Organ", "Tonewheel organ chords with key click and a rotating-speaker wobble.", ["musical"], True, False, 0.04, p_drawbar_organ),
    ("snd-music-box", "Music Box", "A wound-up comb of tiny tines with a cranking ratchet.", ["musical", "soft", "ambient"], True, False, 0.90, p_music_box),
    ("snd-harp", "Harp", "Silky harp glissandi with a dreamy, drifting loop.", ["musical", "ambient"], True, False, 0.44, p_harp),
    ("snd-nylon-guitar", "Nylon Guitar", "Warm fingerstyle strums on a classical guitar body.", ["musical", "soft"], False, False, 0.09, p_nylon_guitar),
    ("snd-steel-string", "Steel String", "Bright fingerpicked steel strings with a crisp pick attack.", ["musical"], False, False, 0.60, p_steel_string),
    ("snd-koto", "Koto", "Plucked silk strings in the hirajoshi scale with expressive pitch bends.", ["musical"], True, True, 0.97, p_koto),
    ("snd-shamisen", "Shamisen", "Buzzing, sharp plucks and plectrum slaps in a pentatonic style.", ["musical"], True, False, 0.00, p_shamisen),
    ("snd-pizzicato", "Pizzicato", "Dry, bouncy orchestral string plucks.", ["musical", "soft"], False, False, 0.20, p_pizzicato),
    ("snd-steel-drum", "Steel Drum", "Sunny steel pan notes with quick tremolo rolls.", ["musical"], True, True, 0.14, p_steel_drum),
    ("snd-handpan", "Handpan", "Soft, resonant dome notes with an echoing hollow body.", ["musical", "ambient", "soft"], True, False, 0.47, p_handpan),
    ("snd-gamelan", "Gamelan", "Beating bronze metallophones in a slendro tuning with a deep gong.", ["musical", "ambient"], True, False, 0.12, p_gamelan),
    ("snd-xylophone", "Xylophone", "Hard-mallet wooden bars: dry, quick and cheerful.", ["musical", "soft"], False, False, 0.10, p_xylophone),
    ("snd-tubular-bells", "Tubular Bells", "Long, ringing orchestral chimes struck one after another.", ["musical", "ambient"], True, False, 0.62, p_tubular_bells),
    ("snd-singing-bowl", "Singing Bowl", "Slow, beating bowl tones for a calm, meditative feel.", ["ambient", "soft"], False, False, 0.08, p_singing_bowl),
    ("snd-wind-chimes", "Wind Chimes", "A random breeze of tuned tubes tinkling across the menu.", ["ambient", "soft", "nature"], False, False, 0.52, p_wind_chimes),
    ("snd-bamboo", "Bamboo", "Hollow bamboo knocks answered by a breathy flute.", ["nature", "soft"], False, False, 0.25, p_bamboo),
    ("snd-hand-drum", "Hand Drums", "Pitched hand-drum slaps and rolls, deep on the left and bright on the right.", ["musical", "mechanical"], False, False, 0.07, p_hand_drum),
    ("snd-blue-switch", "Blue Switches", "Loud, clicky two-stage keyboard switches with a metal leaf ping.", ["mechanical"], False, False, 0.64, p_blue_switch),
    ("snd-thock-board", "Thock Board", "Deep, creamy keyboard thocks with almost no top-end.", ["mechanical", "soft"], False, False, 0.72, p_thock_board),
    ("snd-camera-shutter", "Camera Shutter", "Shutter clacks, film winding and an autofocus beep.", ["mechanical"], True, False, 0.57, p_camera_shutter),
    ("snd-rotary-dial", "Rotary Dial", "Spinning dial ticks, a handset clunk and a telephone bell.", ["mechanical", "retro"], False, False, 0.16, p_rotary_dial),
    ("snd-cassette-deck", "Cassette Deck", "Heavy transport buttons, tape hiss and motor spin-ups.", ["mechanical", "retro"], True, True, 0.03, p_cassette_deck),
    ("snd-toggle-panel", "Toggle Panel", "Chunky metal toggle switches snapping on a control panel.", ["mechanical"], False, False, 0.35, p_toggle_panel),
    ("snd-office-desk", "Office Desk", "A clicking pen, a stapler and a snapping binder clip.", ["mechanical", "soft"], False, False, 0.23, p_office_desk),
    ("snd-clockwork", "Clockwork", "Escapement ticks, spring winding and a tiny chime.", ["mechanical", "ambient"], True, False, 0.13, p_clockwork),
    ("snd-metal-clank", "Metal Clank", "Struck steel plates and pipes with a hard, clangy ring.", ["mechanical"], False, False, 0.57, p_metal_clank),
    ("snd-blade-ring", "Blade Ring", "A drawn sword's whisper and a bright ring of steel on steel.", ["mechanical", "sci-fi"], False, False, 0.54, p_blade_ring),
    ("snd-abacus-beads", "Abacus Beads", "Clattering wooden beads that slide and click into place.", ["mechanical", "soft"], False, False, 0.06, p_abacus_beads),
    ("snd-cardboard", "Cardboard Box", "Dull thuds and scrapes of flaps on a corrugated box.", ["soft", "mechanical"], False, False, 0.09, p_cardboard),
    ("snd-bubble-wrap", "Bubble Wrap", "Satisfying pops with a tiny gasp of air.", ["soft"], False, False, 0.49, p_bubble_wrap),
    ("snd-ice-cubes", "Ice Cubes", "Cold glass clinks, a pour and crackling ice.", ["soft", "ambient"], True, False, 0.53, p_ice_cubes),
    ("snd-rain-window", "Rain on Glass", "Tiny drips tapping a windowpane over a hush of rain.", ["soft", "ambient", "nature"], False, False, 0.60, p_rain_window),
    ("snd-jazz-brushes", "Jazz Brushes", "Whispering brush swishes with a soft kick drum.", ["musical", "soft"], True, False, 0.08, p_jazz_brushes),
    ("snd-songbird", "Songbird", "Synthesized chirps, tweets and trills of a morning bird.", ["nature", "ambient"], True, True, 0.28, p_songbird),
    ("snd-frog-pond", "Frog Pond", "Throaty ribbits, tree-frog peeps and little plops.", ["nature"], False, False, 0.33, p_frog_pond),
    ("snd-crickets", "Night Crickets", "Pulsing high chirps of crickets on a warm summer night.", ["nature", "ambient"], False, False, 0.66, p_crickets),
    ("snd-night-owl", "Night Owl", "Soft, breathy hoots from deep in the woods.", ["nature", "ambient"], False, False, 0.74, p_night_owl),
    ("snd-tide-swell", "Tide Swell", "Rolling waves that rise and fall with a few water pings.", ["nature", "ambient"], False, False, 0.56, p_tide_swell),
    ("snd-hud-blips", "HUD Blips", "Quick bursts of data ticks, like a helmet display reading out.", ["sci-fi"], False, False, 0.45, p_hud_blips),
    ("snd-sonar-ping", "Sonar Ping", "Deep-sea pings that echo away into the dark.", ["sci-fi", "ambient"], False, False, 0.57, p_sonar_ping),
    ("snd-geiger-counter", "Geiger Counter", "Random clicks that speed up and slow down like a radiation meter.", ["sci-fi", "mechanical"], False, False, 0.19, p_geiger),
    ("snd-analog-sweeps", "Analog Sweeps", "Resonant filter sweeps over detuned saw waves.", ["sci-fi", "retro"], False, False, 0.84, p_analog_sweeps),
    ("snd-hologram", "Hologram", "A shimmering cloud of tiny grains that gathers into a chord.", ["sci-fi", "ambient"], True, True, 0.50, p_hologram),
    ("snd-dialup-tones", "Dial-Up Tones", "Touch-tone pairs and a screeching handshake of an old modem.", ["retro", "sci-fi"], False, False, 0.76, p_dialup),
    ("snd-robot-voice", "Robot Voice", "Buzzing vowels from a talking machine.", ["sci-fi"], True, False, 0.38, p_robot_voice),
    ("snd-theremin", "Theremin", "Eerie gliding tones with a hand-waved vibrato.", ["sci-fi", "musical"], True, False, 0.82, p_theremin),
    ("snd-warp-drive", "Warp Drive", "Rising and falling whooshes with a hull-shaking thump.", ["sci-fi"], True, False, 0.62, p_warp_drive),
    ("snd-lofi-tape", "Lo-Fi Tape", "Wobbly, hissy electric-piano chords with a mellow chill loop.", ["musical", "ambient", "retro"], True, False, 0.10, p_lofi_tape),
    ("snd-vinyl-pop", "Vinyl Pop", "Needle drops, crackle and warm piano notes from an old record.", ["retro", "soft"], False, False, 0.07, p_vinyl_pop),
    ("snd-vaporwave", "Vaporwave Haze", "Slowed, detuned chimes drenched in echo, with a sleepy loop.", ["ambient", "musical", "retro"], True, False, 0.88, p_vaporwave),
    ("snd-haptic-tick", "Haptic Tick", "Tiny, dry taps that feel like a light vibration motor.", ["soft", "mechanical"], False, False, 0.55, p_haptic_tick),
    ("snd-haptic-thud", "Haptic Thud", "Short low bumps you can almost feel in your palm.", ["soft", "mechanical"], False, False, 0.02, p_haptic_thud),
    ("snd-haptic-buzz", "Haptic Buzz", "Quick motor buzzes in short, long and rising patterns.", ["mechanical"], False, False, 0.15, p_haptic_buzz),
]

# loops for the richer packs (reuse the music engine)
NEW_MUSIC = {
    "snd-rhodes": dict(secs=40, prog=PROG_A, shift=0, padkind="warm", voice=rhodes_v, arp_d=1.2, octs=(24, 36), step=0.5, rest=0.25,
                       pattern="rand", arp=0.16, pad=0.06, lp=3500),
    "snd-music-box": dict(secs=36, prog=PROG_D, shift=7, padkind="soft", voice=lambda f, d: additive(f, d, [(1, 1.0, 0.25), (3.0, 0.4, 0.12)], a=0.001),
                          arp_d=1.0, octs=(36, 48), step=0.25, rest=0.2, pattern="updown", arp=0.14, pad=0.05),
    "snd-handpan": dict(secs=44, prog=PROG_B, shift=-2, padkind="soft", voice=lambda f, d: additive(f, d, [(1, 1.0, 0.8), (2, 0.45, 0.5), (3, 0.2, 0.3)], a=0.004),
                        arp_d=1.6, octs=(24, 36), step=0.75, rest=0.3, pattern="rand", arp=0.15, pad=0.07, reverb=0.6),
    "snd-harp": dict(secs=40, prog=PROG_C, shift=3, padkind="glass", voice=lambda f, d: lp(ks(f, d, tau=1.0, bright=0.4), 5500),
                     arp_d=1.5, octs=(24, 36), step=0.5, rest=0.2, pattern="updown", arp=0.15, pad=0.06, reverb=0.6),
    "snd-lofi-tape": dict(secs=40, prog=PROG_A, shift=-2, padkind="warm", voice=rhodes_v, arp_d=1.0, octs=(24, 36), step=0.375, rest=0.35,
                          pattern="rand", arp=0.13, pad=0.08, lp=2600),
    "snd-vaporwave": dict(secs=44, prog=PROG_C, shift=2, padkind="glass", voice=lambda f, d: lp(tone(f, d, a=0.01, tau=0.5, fm=(3.01, 1.8, 0.3)), 3000),
                          arp_d=2.0, octs=(24, 36), step=1.0, rest=0.3, pattern="rand", arp=0.1, pad=0.09, reverb=0.8, lp=2800),
    "snd-fm-bells": dict(secs=40, prog=PROG_D, shift=5, padkind="glass", voice=lambda f, d: tone(f, d, a=0.001, tau=0.55, fm=(3.5, 3.0, 0.25)),
                         arp_d=1.6, octs=(24, 36), step=0.75, rest=0.3, pattern="rand", arp=0.1, pad=0.07, reverb=0.6),
    "snd-singing-bowl": dict(secs=44, prog=PROG_C, shift=-5, padkind="soft", voice=lambda f, d: additive(f, d, [(1, 1.0, 1.3), (2.71, 0.5, 0.9)], a=0.03),
                             arp_d=2.4, octs=(24, 36), step=2.0, rest=0.4, pattern="rand", arp=0.1, pad=0.1, reverb=0.7),
}


# ----------------------------------------------------------------------------- catalog
# id, name, description, categories, pro, featured, hue (0..1), builder
PACKS = [
    ("snd-soft-pop", "Soft Pop", "Round, friendly bubbles of sound. Gentle on the ears for everyday menus.", ["soft"], False, True, 0.93, p_soft_pop),
    ("snd-glass-tap", "Glass Tap", "Bright, delicate taps like a fingernail on a crystal glass.", ["soft", "musical"], True, False, 0.52, p_glass_tap),
    ("snd-crystal-chime", "Crystal Chime", "Shimmering chimes with an ambient, floating music loop.", ["musical", "ambient"], True, True, 0.55, p_crystal_chime),
    ("snd-wood-block", "Wood Block", "Warm, hollow knocks of a wooden block. Natural and tactile.", ["mechanical", "soft"], False, False, 0.08, p_wood_block),
    ("snd-water-drop", "Water Drop", "Clean droplets that bloop upward when menus open.", ["soft", "ambient"], False, False, 0.58, p_water_drop),
    ("snd-mech-keys", "Mechanical Keys", "Crisp keyboard switch clacks with a deep space-bar thock.", ["mechanical"], False, True, 0.62, p_mech_keys),
    ("snd-typewriter", "Typewriter", "Strikes, ratchets and a bright carriage bell for confirms.", ["mechanical"], False, False, 0.1, p_typewriter),
    ("snd-paper-flick", "Paper Flick", "Soft swishes and flicks of paper being turned.", ["soft"], False, False, 0.12, p_paper_flick),
    ("snd-chip-blips", "Chip Blips", "Square-wave 8-bit blips and quick arpeggios.", ["retro"], False, False, 0.33, p_chip_blips),
    ("snd-arcade-coin", "Arcade Run", "Sparkling pulse-wave slides and runs in an arcade spirit.", ["retro"], False, False, 0.15, p_arcade_coin),
    ("snd-synth-pluck", "Synthwave Pluck", "Filtered saw plucks and a pulsing night-drive music loop.", ["retro", "musical", "ambient"], True, True, 0.8, p_synth_pluck),
    ("snd-sci-fi-console", "Sci-Fi Console", "Data chirps from a starship console, with a deep space drone loop.", ["sci-fi", "ambient"], True, True, 0.48, p_scifi_console),
    ("snd-laser-ui", "Laser UI", "Fast laser sweeps and zaps for a futuristic interface.", ["sci-fi"], True, False, 0.97, p_laser_ui),
    ("snd-deep-bubble", "Deep Bubble", "Low, round bubbles rising from the deep.", ["soft", "ambient"], False, False, 0.6, p_deep_bubble),
    ("snd-marimba", "Marimba", "Woody mallet notes with a relaxed marimba music loop.", ["musical", "ambient"], True, False, 0.07, p_marimba),
    ("snd-kalimba", "Kalimba", "Plucked metal tines in a calm pentatonic mood, with a loop.", ["musical", "ambient"], True, True, 0.38, p_kalimba),
    ("snd-toy-piano", "Toy Piano", "Tiny, bright toy piano bars. Playful and cute.", ["musical", "soft"], True, False, 0.85, p_toy_piano),
    ("snd-minimal-tick", "Minimal Tick", "Barely-there ticks. Quiet, clean and precise.", ["mechanical"], False, False, 0.55, p_minimal_tick),
    ("snd-analog-click", "Analog Click", "Relay clunks and warm analog blips from a vintage panel.", ["mechanical", "retro"], False, False, 0.05, p_analog_click),
    ("snd-warm-felt", "Warm Felt", "Muffled felt-piano tones and a soft, sleepy music loop.", ["soft", "musical", "ambient"], True, False, 0.02, p_warm_felt),
    ("snd-temple-bell", "Temple Bell", "Deep resonant bells with a meditative, drifting loop.", ["musical", "ambient"], True, False, 0.12, p_temple_bell),
    ("snd-handheld-beep", "Handheld Beep", "Lo-fi beeps from a pocket console of old.", ["retro"], False, False, 0.25, p_handheld_beep),
    ("snd-cyber-glitch", "Cyber Glitch", "Bit-crushed stutters and digital glitches.", ["sci-fi", "retro"], True, False, 0.72, p_cyber_glitch),
    ("snd-cozy-pluck", "Cozy Pluck", "Warm plucked strings and a cosy fireside music loop.", ["soft", "musical", "ambient"], True, False, 0.07, p_cozy_pluck),
]

PACKS += NEW_PACKS
PACKS += W2_PACKS
MUSIC.update(NEW_MUSIC)

CATEGORIES = [
    ("soft", "Soft"), ("retro", "Retro"), ("mechanical", "Mechanical"),
    ("musical", "Musical"), ("sci-fi", "Sci-Fi"), ("ambient", "Ambient"), ("nature", "Nature"),
    ("world", "World"), ("electronic", "Electronic"), ("hardware", "Retro hardware"), ("game", "Game moods"),
    ("holiday", "Holiday"), ("asmr", "ASMR"), ("kids", "Kids"), ("cinematic", "Cinematic"),
    ("accessibility", "Accessibility"), ("materials", "Materials"),
]


# ----------------------------------------------------------------------------- images
def hexc(rgb):
    return "#%02X%02X%02X" % rgb


def hsv(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return (int(r * 255 + 0.5), int(g * 255 + 0.5), int(b * 255 + 0.5))


def palette(h):
    return dict(bg=hsv(h, 0.7, 0.16), mid=hsv(h, 0.65, 0.38), main=hsv(h, 0.65, 0.95),
                accent=hsv(h + 0.09, 0.5, 1.0), light=hsv(h - 0.04, 0.2, 1.0))


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def vgrad(w, h, top, bot):
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        d.line([(0, y), (w, y)], fill=lerp(top, bot, y / (h - 1)))
    return img


def glow(img, cx, cy, r, color, alpha):
    layer_ = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer_).ellipse([cx - r, cy - r, cx + r, cy + r], fill=color + (alpha,))
    layer_ = layer_.filter(ImageFilter.GaussianBlur(r * 0.5))
    img.paste(layer_, (0, 0), layer_)


def envelope_bars(sounds, nbars):
    cat = []
    for s in sounds:
        cat += s + [0.0] * int(0.05 * SR)
    seg = len(cat) / nbars
    out = []
    for i in range(nbars):
        chunk = cat[int(i * seg):int((i + 1) * seg)] or [0.0]
        out.append(max(abs(v) for v in chunk))
    m = max(out) or 1
    return [v / m for v in out]


def speaker(d, cx, cy, s, color):
    body = [(cx - 0.55 * s, cy - 0.22 * s), (cx - 0.3 * s, cy - 0.22 * s), (cx - 0.3 * s, cy + 0.22 * s),
            (cx - 0.55 * s, cy + 0.22 * s)]
    cone = [(cx - 0.3 * s, cy - 0.22 * s), (cx + 0.05 * s, cy - 0.5 * s), (cx + 0.05 * s, cy + 0.5 * s),
            (cx - 0.3 * s, cy + 0.22 * s)]
    d.polygon(body, fill=color)
    d.polygon(cone, fill=color)
    for k, r in enumerate((0.3, 0.55, 0.8)):
        d.arc([cx + 0.05 * s - r * s * 0.6, cy - r * s, cx + 0.05 * s + r * s * 0.6, cy + r * s], -45, 45,
              fill=color, width=max(2, int(s * 0.07)))


COVER_STYLE = {
    "soft": ["snd-soft-pop", "snd-water-drop", "snd-paper-flick", "snd-deep-bubble", "snd-glass-tap"],
    "retro": ["snd-chip-blips", "snd-arcade-coin", "snd-handheld-beep", "snd-synth-pluck", "snd-analog-click"],
    "mechanical": ["snd-wood-block", "snd-mech-keys", "snd-typewriter", "snd-minimal-tick"],
    "musical": ["snd-toy-piano", "snd-marimba", "snd-kalimba", "snd-cozy-pluck"],
    "sci-fi": ["snd-sci-fi-console", "snd-laser-ui", "snd-cyber-glitch"],
    "ambient": ["snd-crystal-chime", "snd-warm-felt", "snd-temple-bell"],
}
STYLE_OF = {pid: st for st, ids in COVER_STYLE.items() for pid in ids}
CW, CH_ = 1280, 720


def overlay(img, fn, blur=0):
    lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    fn(ImageDraw.Draw(lay, "RGBA"))
    if blur:
        lay = lay.filter(ImageFilter.GaussianBlur(blur))
    img.alpha_composite(lay)


def base_img(pal, mid=None):
    return vgrad(CW, CH_, pal["bg"], mid or pal["mid"]).convert("RGBA")


PIX_SPK = ["...#....#..", "..##.....#.", "####..#...#", "####..#...#", "####..#...#", "..##.....#.", "...#....#.."]


def cover_soft(pal, sounds, rnd):
    img = base_img(pal)
    cols = [pal["main"], pal["accent"], pal["light"], pal["mid"]]
    for _ in range(9):
        cx, cy, r = rnd.randrange(CW), rnd.randrange(CH_), rnd.randrange(110, 280)
        c = rnd.choice(cols)
        overlay(img, lambda d, cx=cx, cy=cy, r=r, c=c: d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c + (90,)), blur=14)
    bars = envelope_bars(sounds, 14)

    def fg(d):
        d.ellipse([110, 210, 410, 510], fill=pal["light"] + (60,), outline=pal["light"] + (150,), width=6)
        speaker(d, 255, 360, 190, pal["light"])
        for i, v in enumerate(bars):
            h = 70 + v * 330
            xx = 470 + i * 55
            d.rounded_rectangle([xx, 360 - h / 2, xx + 38, 360 + h / 2], radius=19,
                                fill=lerp(pal["light"], pal["accent"], i / 13) + (225,))
    overlay(img, fg)
    return img


def cover_retro(pal, sounds, rnd):
    img = base_img(pal, pal["bg"])
    bars = envelope_bars(sounds, 24)

    def fg(d):
        for r, row in enumerate(PIX_SPK):
            for c, ch in enumerate(row):
                if ch == "#":
                    d.rectangle([60 + c * 30, 200 + r * 30, 60 + c * 30 + 27, 200 + r * 30 + 27], fill=pal["light"] + (255,))
        for i, v in enumerate(bars):
            n = 2 + int(v * 11)
            for k in range(n):
                col = lerp(pal["main"], pal["accent"], k / 12) if k < 9 else (255, 120, 120)
                x = 400 + i * 33
                d.rectangle([x, 640 - k * 38 - 32, x + 28, 640 - k * 38], fill=col + (255,))
    overlay(img, fg)
    overlay(img, lambda d: [d.line([(0, y), (CW, y)], fill=(0, 0, 0, 70), width=2) for y in range(0, CH_, 6)])
    overlay(img, lambda d: d.rectangle([0, 0, CW - 1, CH_ - 1], outline=pal["main"] + (160,), width=10))
    return img


def cover_mech(pal, sounds, rnd):
    img = base_img(pal)

    def bg(d):
        for r in range(6):
            for c in range(13):
                x, y = 20 + c * 100, 20 + r * 120
                d.rounded_rectangle([x, y, x + 84, y + 84], radius=16, fill=(255, 255, 255, 14),
                                    outline=pal["main"] + (60,), width=2)
    overlay(img, bg)
    bars = envelope_bars(sounds, 20)

    def fg(d):
        d.rounded_rectangle([110, 190, 450, 560], radius=48, fill=(0, 0, 0, 120))
        d.rounded_rectangle([100, 160, 440, 520], radius=48, fill=pal["mid"] + (255,), outline=pal["accent"] + (255,), width=5)
        d.rounded_rectangle([140, 190, 400, 440], radius=34, fill=lerp(pal["mid"], pal["light"], 0.35) + (255,))
        speaker(d, 268, 318, 170, pal["bg"])
        for i, v in enumerate(bars):
            h = 40 + v * 340
            x = 520 + i * 36
            d.rectangle([x, 520 - h, x + 26, 520], fill=lerp(pal["main"], pal["accent"], i / 19) + (240,))
            d.rectangle([x, 520 - h, x + 26, 520 - h + 8], fill=pal["light"] + (255,))
    overlay(img, fg)
    return img


def cover_music(pal, sounds, rnd):
    img = base_img(pal)
    n = 13
    bars = envelope_bars(sounds, n)
    ys = [250 + k * 58 for k in range(5)]

    def fg(d):
        d.ellipse([40, 240, 240, 440], fill=pal["bg"] + (150,), outline=pal["accent"] + (210,), width=5)
        speaker(d, 138, 340, 130, pal["light"])
        for y in ys:
            d.line([(290, y), (1230, y)], fill=pal["light"] + (110,), width=3)
        pts = []
        for i, v in enumerate(bars):
            pts.append((340 + i * 70, ys[4] + 40 - v * (ys[4] - ys[0] + 80)))
        d.line(pts, fill=pal["main"] + (200,), width=5, joint="curve")
        for i, (x, y) in enumerate(pts):
            col = lerp(pal["main"], pal["accent"], i / (n - 1))
            d.line([(x + 20, y), (x + 20, y - 100)], fill=col + (255,), width=6)
            d.ellipse([x - 22, y - 16, x + 22, y + 16], fill=col + (255,), outline=pal["light"] + (255,), width=3)
    overlay(img, fg)
    return img


def cover_scifi(pal, sounds, rnd):
    img = base_img(pal, pal["bg"])
    cx, cy = 640, 360
    nb = 120
    bars = envelope_bars(sounds, 60)
    bars = bars + bars[::-1]

    def fg(d):
        for x in range(0, CW, 64):
            d.line([(x, 0), (x, CH_)], fill=pal["main"] + (22,), width=1)
        for y in range(0, CH_, 64):
            d.line([(0, y), (CW, y)], fill=pal["main"] + (22,), width=1)
        for k, (r, w, a) in enumerate([(330, 4, 170), (285, 2, 120), (170, 6, 200)]):
            for s in range(0, 360, 30 + 15 * k):
                d.arc([cx - r, cy - r, cx + r, cy + r], s + k * 7, s + 18 + 5 * k, fill=pal["accent"] + (a,), width=w)
        for i, v in enumerate(bars):
            ang = TAU * i / nb - math.pi / 2
            r0, r1 = 190, 190 + 25 + v * 70
            col = lerp(pal["main"], pal["accent"], v)
            d.line([(cx + r0 * math.cos(ang), cy + r0 * math.sin(ang)), (cx + r1 * math.cos(ang), cy + r1 * math.sin(ang))],
                   fill=col + (255,), width=5)
        for i in range(72):
            ang = TAU * i / 72
            rr = 360 if i % 6 == 0 else 352
            d.line([(cx + 345 * math.cos(ang), cy + 345 * math.sin(ang)), (cx + rr * math.cos(ang), cy + rr * math.sin(ang))],
                   fill=pal["light"] + (200,), width=3)
        d.ellipse([cx - 120, cy - 120, cx + 120, cy + 120], fill=pal["bg"] + (200,), outline=pal["light"] + (230,), width=4)
        speaker(d, cx - 6, cy, 140, pal["light"])
    overlay(img, fg)
    return img


def cover_ambient(pal, sounds, rnd):
    img = base_img(pal)
    cols = [pal["main"], pal["accent"], pal["light"]]
    for k in range(5):
        ph = rnd.random() * TAU
        amp = rnd.randrange(50, 110)
        yb = 180 + k * 55
        col = cols[k % 3]

        def rib(d, ph=ph, amp=amp, yb=yb, col=col):
            top = [(x, yb + amp * math.sin(x / 210 + ph) + 40 * math.sin(x / 90 + ph * 2)) for x in range(0, CW + 20, 20)]
            d.polygon(top + [(x, y + 150) for x, y in reversed(top)], fill=col + (85,))
        overlay(img, rib, blur=26)
    bars = envelope_bars(sounds, 160)
    d0 = ImageDraw.Draw(img, "RGBA")
    for _ in range(70):
        x, y, r = rnd.randrange(CW), rnd.randrange(300), rnd.choice([1, 2, 2, 3])
        d0.ellipse([x - r, y - r, x + r, y + r], fill=pal["light"] + (180,))

    def fg(d):
        cy = 520
        top, bot = [], []
        for i, v in enumerate(bars):
            x = 80 + i * (CW - 160) / 159
            h = 10 + v * 110
            top.append((x, cy - h))
            bot.append((x, cy + h))
        d.polygon(top + bot[::-1], fill=pal["light"] + (110,))
        d.line(top, fill=pal["light"] + (230,), width=3)
        d.line(bot, fill=pal["light"] + (230,), width=3)
        d.ellipse([560, 100, 720, 260], fill=pal["bg"] + (120,), outline=pal["light"] + (210,), width=4)
        speaker(d, 636, 180, 100, pal["light"])
    overlay(img, fg)
    return img


COVERS = {"soft": cover_soft, "retro": cover_retro, "mechanical": cover_mech, "musical": cover_music,
          "sci-fi": cover_scifi, "ambient": cover_ambient}


# ----------------------------------------------------------------------------- expansion covers (v2): one motif per pack
def _C(d, x, y, r, fill=None, out=None, w=3):
    d.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline=out, width=w)


def _RR(d, x0, y0, x1, y1, r, fill=None, out=None, w=3):
    d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fill, outline=out, width=w)


def _A(c, a=255):
    return tuple(c) + (a,)


def _star(d, x, y, r, fill):
    d.polygon([(x, y - r), (x + r * 0.2, y - r * 0.2), (x + r, y), (x + r * 0.2, y + r * 0.2), (x, y + r),
               (x - r * 0.2, y + r * 0.2), (x - r, y), (x - r * 0.2, y - r * 0.2)], fill=fill)


def _pix(d, rows, x0, y0, px, fill):
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                d.rectangle([x0 + c * px, y0 + r * px, x0 + c * px + px - 2, y0 + r * px + px - 2], fill=fill)


def _gear(d, cx, cy, ro, ri, teeth, fill, hole, rot=0.0):
    pts = []
    for i in range(teeth * 4):
        a = rot + TAU * i / (teeth * 4)
        r = ro if (i % 4) in (1, 2) else ri
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    d.polygon(pts, fill=fill)
    _C(d, cx, cy, hole, fill=_A((0, 0, 0), 120))


def _wave(d, x0, x1, cy, amp, per, ph, col, w=4, f=None):
    pts = [(x, cy + amp * (f(x / per + ph) if f else math.sin(x / per + ph))) for x in range(int(x0), int(x1), 6)]
    d.line(pts, fill=col, width=w, joint="curve")


def m_pulse_quest(d, p, b, r):
    sword = ["......##", ".....###", "....###.", "...###..", "#.###...", ".###....", "..#.....", ".#.#...."]
    heart = [".##.##.", "#######", "#######", ".#####.", "..###..", "...#..."]
    _pix(d, sword, 150, 150, 44, _A(p["light"]))
    for i in range(3):
        _pix(d, heart, 620 + i * 190, 160, 22, _A(p["accent"]))
    for i, v in enumerate(b[:24]):
        h = int(v * 8) + 1
        for k in range(h):
            d.rectangle([560 + i * 28, 640 - k * 30, 560 + i * 28 + 24, 640 - k * 30 - 26], fill=_A(p["main"]))


def m_chip_drums(d, p, b, r):
    for i, (x, c) in enumerate(zip((150, 420, 690, 960), (p["main"], p["accent"], p["light"], p["mid"]))):
        d.rectangle([x, 200, x + 250, 450], fill=_A(c, 230))
        for _ in range(110):
            gx, gy = r.randrange(7), r.randrange(7)
            if r.random() < 0.5:
                d.rectangle([x + gx * 35, 200 + gy * 35, x + gx * 35 + 30, 200 + gy * 35 + 30], fill=_A(p["bg"], 190))
    for i, v in enumerate(b[:32]):
        d.rectangle([140 + i * 34, 640 - v * 130, 140 + i * 34 + 28, 640], fill=_A(p["light"]))


def m_pc_speaker(d, p, b, r):
    _C(d, 400, 360, 270, fill=_A(p["bg"], 200), out=_A(p["light"]), w=8)
    for y in range(110, 620, 34):
        for x in range(150, 660, 34):
            if (x - 400) ** 2 + (y - 360) ** 2 < 235 ** 2:
                _C(d, x, y, 9, fill=_A(p["main"]))
    xs = []
    for k in range(9):
        y0, y1 = (260, 460) if k % 2 == 0 else (460, 260)
        xs += [(720 + k * 60, y0), (720 + k * 60 + 60, y0)]
        xs += [(720 + k * 60 + 60, y1)]
    d.line(xs, fill=_A(p["accent"]), width=10)


def m_wavetable(d, p, b, r):
    hs = [r.randrange(1, 16) for _ in range(32)]
    for y in range(8):
        d.line([(80, 150 + y * 60), (1200, 150 + y * 60)], fill=_A(p["light"], 40), width=2)
    for i, h in enumerate(hs):
        x = 90 + i * 35
        d.rectangle([x, 640 - h * 30, x + 33, 640], fill=_A(p["main"], 200))
        d.rectangle([x, 640 - h * 30, x + 33, 640 - h * 30 + 8], fill=_A(p["light"]))


def m_coin_slot(d, p, b, r):
    _RR(d, 120, 120, 520, 600, 30, fill=_A(p["bg"], 220), out=_A(p["accent"]), w=6)
    _RR(d, 290, 220, 350, 420, 20, fill=(0, 0, 0, 255), out=_A(p["light"]), w=4)
    _C(d, 840, 360, 230, fill=_A(p["accent"]), out=_A(p["light"]), w=14)
    _C(d, 840, 360, 170, out=_A(p["light"]), w=8)
    _star(d, 840, 360, 120, _A(p["light"]))
    d.polygon([(320, 120), (400, 60), (240, 60)], fill=_A(p["main"]))


def m_sixteen(d, p, b, r):
    for k in range(9):
        d.rectangle([280, 380 - k * 22, 1000, 380 - k * 22 + 11], fill=_A(p["accent"], 200 - k * 18))
    _C(d, 640, 380, 190, fill=_A(p["accent"], 255))
    for k in range(6):
        d.rectangle([440, 380 + k * 22, 840, 380 + k * 22 + 4 + k * 3], fill=_A(p["bg"], 255))
    for off, col, hgt in ((0, p["mid"], 200), (60, p["bg"], 140)):
        pts = [(0, 720)]
        for x in range(0, 1300, 80):
            pts.append((x, 720 - hgt - off - 70 * abs(math.sin(x / 190 + off))))
        d.polygon(pts + [(1280, 720)], fill=_A(col, 255))
    for _ in range(40):
        x, y = r.randrange(1280), r.randrange(220)
        d.rectangle([x, y, x + 8, y + 8], fill=_A(p["light"]))


def m_pwm_tracker(d, p, b, r):
    cols = [p["main"], p["accent"], p["light"], p["mid"]]
    for c in range(4):
        x = 140 + c * 280
        d.rectangle([x, 80, x + 250, 650], fill=_A(p["bg"], 190), outline=_A(cols[c], 150), width=3)
        for row in range(14):
            y = 100 + row * 40
            if r.random() < 0.65:
                w = r.randrange(40, 200)
                d.rectangle([x + 12, y, x + 12 + w, y + 24], fill=_A(cols[c], 235))
            if row % 4 == 0:
                d.rectangle([x, y - 6, x + 250, y - 3], fill=_A(p["light"], 90))


def m_fm_bells(d, p, b, r):
    for k, (n, col) in enumerate(((5, p["main"]), (7, p["accent"]), (3, p["light"]))):
        pts = []
        for i in range(721):
            a = TAU * i / 720
            rr = (320 - k * 55) * math.cos(n * a / 2 + k)
            pts.append((640 + rr * math.cos(a), 360 + rr * math.sin(a)))
        d.line(pts, fill=_A(col, 230), width=5)
    _C(d, 640, 360, 24, fill=_A(p["light"]))


def m_rhodes(d, p, b, r):
    for i in range(14):
        x = 70 + i * 82
        d.rectangle([x, 330, x + 78, 680], fill=_A(p["light"], 235), outline=_A(p["bg"], 255), width=3)
    for i in (0, 1, 3, 4, 5, 7, 8, 10, 11, 12):
        x = 70 + i * 82 + 56
        d.rectangle([x, 330, x + 52, 540], fill=_A(p["bg"], 255))
    for i, v in enumerate(b[:14]):
        x = 110 + i * 82
        d.line([(x, 300), (x, 300 - 30 - v * 170)], fill=_A(p["accent"]), width=8)
        _C(d, x, 300 - 30 - v * 170, 12, fill=_A(p["main"]))


def m_vibraphone(d, p, b, r):
    for i in range(9):
        w = 120 - i * 7
        x = 120 + i * 125
        d.rounded_rectangle([x, 150 + i * 14, x + w, 190 + i * 14], radius=10, fill=_A(p["light"], 240))
        d.rounded_rectangle([x + w // 2 - 12, 215 + i * 14, x + w // 2 + 12, 520 - i * 14], radius=12, fill=_A(p["main"], 200))
    _C(d, 640, 610, 50, out=_A(p["light"]), w=6)
    d.line([(600, 610), (680, 610)], fill=_A(p["light"]), width=6)


def m_glock(d, p, b, r):
    for i in range(10):
        x = 100 + i * 108
        h = 380 - i * 24
        d.rounded_rectangle([x, 580 - h, x + 80, 580], radius=10, fill=_A(lerp(p["main"], p["light"], i / 9), 240))
        d.ellipse([x + 33, 585 - h + 10, x + 47, 585 - h + 24], fill=_A(p["bg"]))
    for _ in range(14):
        _star(d, r.randrange(80, 1200), r.randrange(40, 300), r.randrange(10, 30), _A(p["light"], 230))


def m_drawbar(d, p, b, r):
    cols = [(120, 70, 40)] * 2 + [(240, 240, 240)] * 4 + [(30, 30, 30)] * 2 + [(240, 240, 240)] * 1
    hs = [0.8, 0.5, 0.9, 0.4, 0.1, 0.0, 0.6, 0.3, 0.2]
    for i, (c, h) in enumerate(zip(cols, hs)):
        x = 150 + i * 115
        d.rectangle([x, 120, x + 40, 600], fill=_A(p["bg"], 200), outline=_A(p["light"], 120), width=2)
        d.rounded_rectangle([x - 20, 140 + h * 350, x + 60, 190 + h * 350], radius=8, fill=_A(c))
        d.line([(x - 20, 165 + h * 350), (x + 60, 165 + h * 350)], fill=_A(p["accent"]), width=4)


def m_music_box(d, p, b, r):
    _RR(d, 120, 160, 1160, 480, 40, fill=_A(p["bg"], 200), out=_A(p["accent"]), w=6)
    for row in range(5):
        for c in range(26):
            if r.random() < 0.42:
                _C(d, 160 + c * 38, 205 + row * 55, 8, fill=_A(p["light"]))
    for i in range(24):
        d.rectangle([160 + i * 41, 520, 160 + i * 41 + 22, 600 - (i % 6) * 18], fill=_A(p["main"]))
    d.rectangle([140, 505, 1140, 520], fill=_A(p["accent"]))


def m_harp(d, p, b, r):
    d.arc([160, 40, 760, 440], 180, 270, fill=_A(p["light"]), width=18)
    d.line([(160, 240), (160, 650)], fill=_A(p["light"]), width=24)
    d.line([(160, 650), (940, 650)], fill=_A(p["light"]), width=24)
    d.line([(460, 40), (940, 650)], fill=_A(p["accent"]), width=18)
    for i in range(14):
        t = i / 13
        d.line([(200 + t * 230, 644), (460 + t * 440, 70 + t * 550)], fill=_A(p["main"], 255), width=3)


def m_nylon(d, p, b, r):
    _C(d, 460, 360, 240, fill=_A(p["bg"], 255), out=_A(p["accent"]), w=10)
    for k, rr in enumerate((200, 190, 175, 165)):
        _C(d, 460, 360, rr, out=_A(lerp(p["main"], p["light"], k / 3)), w=8 if k % 2 == 0 else 4)
    _C(d, 460, 360, 130, fill=(0, 0, 0, 255))
    for k in range(6):
        d.line([(80, 270 + k * 36), (1200, 270 + k * 36)], fill=_A(p["light"], 230), width=2 + (5 - k) // 2)
    d.rectangle([760, 220, 780, 500], fill=_A(p["light"]))


def m_steel(d, p, b, r):
    for i in range(9):
        x = 120 + i * 125
        d.line([(x, 110), (x, 610)], fill=_A(p["light"], 200), width=6 if i else 16)
    for k in range(6):
        d.line([(80, 150 + k * 80), (1200, 150 + k * 80)], fill=_A(lerp(p["main"], p["light"], k / 5)), width=2 + (5 - k) // 2)
    for x in (3, 5, 7):
        _C(d, 120 + x * 125 - 62, 650, 14, fill=_A(p["accent"]))
    for k, v in enumerate(b[:10]):
        d.line([(120 + k * 120, 360), (120 + k * 120, 360 - v * 120)], fill=_A(p["accent"], 120), width=5)


def m_koto(d, p, b, r):
    d.polygon([(100, 470), (1180, 330), (1180, 430), (100, 580)], fill=_A(p["mid"]), outline=_A(p["accent"]))
    for k in range(13):
        y0 = 480 + k * 7
        d.line([(110, y0 + 4), (1170, 340 + k * 7 + 4)], fill=_A(p["light"], 220), width=2)
    for k in range(13):
        t = 0.08 + k * 0.07
        x = 110 + t * 1060
        y = 478 + k * 7 - t * 140 + 4
        d.polygon([(x - 14, y + 30), (x + 14, y + 30), (x, y - 36)], fill=_A(p["light"]))
    d.arc([300, 70, 980, 520], 200, 340, fill=_A(p["accent"], 130), width=6)


def m_shamisen(d, p, b, r):
    d.polygon([(220, 420), (350, 300), (480, 420), (350, 540)], fill=_A(p["light"], 240), outline=_A(p["accent"]))
    d.polygon([(220, 420), (350, 300), (480, 420), (350, 540)], fill=_A(p["bg"], 255), outline=_A(p["light"]), width=8)
    d.line([(480, 420), (1180, 200)], fill=_A(p["light"]), width=30)
    for k in range(3):
        d.line([(350, 420 + (k - 1) * 14), (1180, 200 + (k - 1) * 14)], fill=_A(p["accent"]), width=3)
    d.polygon([(780, 570), (980, 570), (900, 690), (860, 690)], fill=_A(p["main"]))


def m_pizz(d, p, b, r):
    d.polygon([(380, 120), (900, 120), (940, 350), (900, 600), (380, 600), (340, 350)], fill=_A(p["bg"], 255), outline=_A(p["accent"]), width=6)
    for sx in (480, 800):
        for y in range(240, 460, 4):
            d.point((sx + 20 * math.sin((y - 240) / 35), y), fill=_A(p["light"]))
        d.line([(sx - 6, 235), (sx + 14, 235)], fill=_A(p["light"]), width=8)
    d.line([(640, 40), (640, 680)], fill=_A(p["light"], 120), width=3)
    for k, a in enumerate((60, 40, 22)):
        d.arc([640 - a, 330 - 140, 640 + a, 330 + 140], 90, 270, fill=_A(p["main"], 230 - k * 50), width=5)
        d.arc([640 - a, 330 - 140, 640 + a, 330 + 140], 270, 90, fill=_A(p["main"], 230 - k * 50), width=5)


def m_steel_drum(d, p, b, r):
    d.ellipse([160, 120, 1120, 640], fill=_A(p["bg"], 255), outline=_A(p["light"]), width=14)
    for k, (cx, cy, rx, ry) in enumerate([(640, 380, 150, 110), (400, 300, 110, 80), (880, 300, 110, 80), (330, 450, 90, 70), (950, 450, 90, 70),
                                          (500, 210, 80, 55), (780, 210, 80, 55), (640, 540, 100, 55)]):
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=_A(lerp(p["main"], p["accent"], k / 7), 255), outline=_A(p["light"]), width=4)


def m_handpan(d, p, b, r):
    _C(d, 640, 360, 330, fill=_A(p["mid"]), out=_A(p["light"]), w=10)
    _C(d, 640, 360, 150, fill=_A(p["bg"]), out=_A(p["accent"]), w=6)
    for k in range(8):
        a = TAU * k / 8
        _C(d, 640 + 235 * math.cos(a), 360 + 235 * math.sin(a), 52, fill=_A(p["main"]), out=_A(p["light"]), w=4)
    _C(d, 640, 360, 38, fill=_A(p["light"]))


def m_gamelan(d, p, b, r):
    d.rectangle([80, 120, 1200, 150], fill=_A(p["light"]))
    for k in range(7):
        x = 170 + k * 150
        d.line([(x, 150), (x, 250)], fill=_A(p["light"], 200), width=4)
        rad = 60 + (k % 4) * 10
        _C(d, x, 250 + rad, rad, fill=_A(p["main"]), out=_A(p["light"]), w=5)
        _C(d, x, 250 + rad, rad * 0.35, fill=_A(p["accent"]))
    _C(d, 640, 560, 100, fill=_A(p["bg"]), out=_A(p["accent"]), w=8)


def m_xylo(d, p, b, r):
    for i in range(9):
        x0 = 120 + i * 120
        y0 = 190 + i * 20
        y1 = 620 - i * 10
        col = hsv(i / 12.0 + 0.0, 0.6, 0.95)
        d.rounded_rectangle([x0, y0, x0 + 100, y1], radius=14, fill=_A(col))
    d.line([(980, 90), (780, 260)], fill=_A(p["light"]), width=10)
    _C(d, 770, 270, 28, fill=_A(p["accent"]))
    d.line([(1100, 120), (900, 310)], fill=_A(p["light"]), width=10)
    _C(d, 890, 320, 28, fill=_A(p["accent"]))


def m_tubular(d, p, b, r):
    d.rectangle([120, 70, 1160, 100], fill=_A(p["light"]))
    for i in range(8):
        x = 190 + i * 125
        h = 300 + (7 - i) * 28
        d.line([(x + 25, 100), (x + 25, 140)], fill=_A(p["light"]), width=3)
        d.rounded_rectangle([x, 140, x + 50, 140 + h], radius=22, fill=_A(lerp(p["main"], p["light"], i / 7)))
        d.rectangle([x + 10, 150, x + 17, 130 + h], fill=_A((255, 255, 255), 90))


def m_singing_bowl(d, p, b, r):
    for k in range(6):
        d.ellipse([240 - k * 40, 560 - k * 14, 1040 + k * 40, 660 + k * 14], outline=_A(p["accent"], 200 - k * 30), width=4)
    d.pieslice([300, 160, 980, 700], 0, 180, fill=_A(p["mid"]), outline=_A(p["light"]), width=8)
    d.ellipse([300, 300, 980, 420], fill=_A(p["bg"]), outline=_A(p["light"]), width=8)
    d.line([(980, 130), (770, 340)], fill=_A(p["light"]), width=18)
    _C(d, 770, 340, 20, fill=_A(p["accent"]))


def m_wind_chimes(d, p, b, r):
    _C(d, 640, 150, 150, fill=_A(p["mid"]), out=_A(p["light"]), w=8)
    _C(d, 640, 150, 28, fill=_A(p["bg"]))
    d.line([(640, 0), (640, 122)], fill=_A(p["light"]), width=4)
    for k in range(6):
        x = 520 + k * 48
        d.line([(x, 215), (x, 270)], fill=_A(p["light"], 200), width=2)
        d.rounded_rectangle([x - 10, 270, x + 10, 350 + (3 - abs(k - 2.5)) * 50 + 80], radius=10, fill=_A(lerp(p["main"], p["accent"], k / 5)))
    d.line([(640, 190), (640, 520)], fill=_A(p["light"], 180), width=2)
    d.polygon([(640, 520), (600, 600), (680, 600)], fill=_A(p["accent"]))


def m_bamboo(d, p, b, r):
    for i, x in enumerate((220, 430, 640, 850, 1060)):
        w = 70 - (i % 3) * 8
        d.rectangle([x - w // 2, 0, x + w // 2, 720], fill=_A(lerp(p["mid"], p["main"], 0.4 + 0.15 * (i % 3))))
        for y in range(70 + i * 40, 720, 190):
            d.rectangle([x - w // 2 - 6, y, x + w // 2 + 6, y + 14], fill=_A(p["bg"]))
        d.rectangle([x - w // 2 + 8, 0, x - w // 2 + 16, 720], fill=_A((255, 255, 255), 70))
    for k in range(8):
        x, y = 640 + (k - 4) * 70, 120 + (k % 3) * 70
        d.polygon([(x, y), (x + 120, y - 30), (x + 20, y + 14)], fill=_A(p["light"], 210))


def m_hand_drum(d, p, b, r):
    for cx, cy, rad, col in ((400, 380, 250, p["mid"]), (900, 330, 180, p["accent"])):
        d.rectangle([cx - rad, cy, cx + rad, cy + rad * 0.7], fill=_A(p["bg"]))
        d.ellipse([cx - rad, cy - rad * 0.35, cx + rad, cy + rad * 0.35], fill=_A(col), outline=_A(p["light"]), width=8)
        d.ellipse([cx - rad * 0.4, cy - rad * 0.14, cx + rad * 0.4, cy + rad * 0.14], fill=_A(p["bg"]), outline=_A(p["light"]), width=4)
        for k in range(10):
            x = cx - rad + k * rad / 5
            d.line([(x, cy + 14), (x + rad / 10, cy + rad * 0.7)], fill=_A(p["light"], 170), width=3)


def m_blue_switch(d, p, b, r):
    _RR(d, 240, 130, 1040, 590, 60, fill=_A((30, 90, 220), 255), out=_A(p["light"]), w=8)
    _RR(d, 330, 200, 950, 500, 30, fill=_A((60, 130, 250), 255))
    d.rectangle([610, 270, 670, 430], fill=_A(p["bg"]))
    d.rectangle([560, 320, 720, 380], fill=_A(p["bg"]))
    for k in range(3):
        d.arc([490 - k * 60, 210 - k * 60, 790 + k * 60, 510 + k * 60], 200, 340, fill=_A(p["light"], 190 - k * 50), width=6)


def m_thock(d, p, b, r):
    for row in range(4):
        for c in range(9):
            pressed = (row, c) == (1, 4)
            x, y = 90 + c * 120 + row * 12, 120 + row * 130 + (14 if pressed else 0)
            col = lerp(p["mid"], p["main"], ((row + c) % 5) / 5)
            _RR(d, x, y, x + 104, y + 104, 18, fill=_A(col, 255), out=_A(p["bg"]), w=4)
            _RR(d, x + 12, y + 10, x + 92, y + 78, 12, fill=_A((255, 255, 255), 40))


def m_camera(d, p, b, r):
    cx, cy = 640, 360
    _C(d, cx, cy, 310, fill=_A(p["bg"]), out=_A(p["light"]), w=10)
    for k in range(7):
        a = TAU * k / 7
        pts = [(cx + 300 * math.cos(a), cy + 300 * math.sin(a)), (cx + 300 * math.cos(a + 0.9), cy + 300 * math.sin(a + 0.9)),
               (cx + 110 * math.cos(a + 1.9), cy + 110 * math.sin(a + 1.9))]
        d.polygon(pts, fill=_A(lerp(p["mid"], p["main"], k / 6), 245), outline=_A(p["light"]))
    _C(d, cx, cy, 70, fill=_A((0, 0, 0)), out=_A(p["accent"]), w=5)


def m_rotary(d, p, b, r):
    cx, cy = 640, 360
    _C(d, cx, cy, 330, fill=_A(p["bg"]), out=_A(p["light"]), w=10)
    for k in range(10):
        a = math.radians(40 + k * 28)
        _C(d, cx + 245 * math.cos(a), cy + 245 * math.sin(a), 46, fill=_A((0, 0, 0)), out=_A(p["light"]), w=5)
    _C(d, cx, cy, 90, fill=_A(p["mid"]), out=_A(p["accent"]), w=6)
    d.arc([cx - 300, cy - 300, cx + 300, cy + 300], 330, 40, fill=_A(p["accent"]), width=14)


def m_cassette(d, p, b, r):
    _RR(d, 190, 150, 1090, 590, 30, fill=_A(p["mid"]), out=_A(p["light"]), w=8)
    _RR(d, 250, 190, 1030, 360, 18, fill=_A(p["light"], 235))
    _RR(d, 440, 400, 840, 560, 26, fill=_A(p["bg"]))
    for x in (420, 860):
        _C(d, x, 275, 82, fill=_A(p["bg"]))
        for k in range(6):
            a = TAU * k / 6
            d.line([(x, 275), (x + 70 * math.cos(a), 275 + 70 * math.sin(a))], fill=_A(p["light"]), width=7)
    d.line([(420, 360), (860, 360)], fill=_A(p["accent"]), width=6)
    for x in (500, 640, 780):
        _C(d, x, 480, 16, fill=_A(p["main"]))


def m_toggle(d, p, b, r):
    _RR(d, 90, 150, 1190, 570, 30, fill=_A(p["bg"]), out=_A(p["light"], 200), w=6)
    for i in range(6):
        x = 190 + i * 180
        on = (i * 5 + 1) % 3 != 0
        _C(d, x, 230, 14, fill=_A((90, 255, 120) if on else (60, 70, 60)))
        _RR(d, x - 28, 300, x + 28, 480, 28, fill=_A(p["mid"]), out=_A(p["light"]), w=4)
        d.polygon([(x - 14, 390), (x + 14, 390), (x + 10, 300 if on else 480), (x - 10, 300 if on else 480)], fill=_A(p["light"]))
        _C(d, x, 300 if on else 480, 22, fill=_A(p["accent"]), out=_A(p["light"]), w=3)


def m_office(d, p, b, r):
    for k in range(8):
        x = 120 + k * 60
        d.line([(x, 520), (x, 560), (x + 36, 560), (x + 36, 520)], fill=_A(p["light"]), width=7)
    d.rounded_rectangle([160, 170, 1000, 220], radius=20, fill=_A(p["accent"]))
    d.polygon([(1000, 170), (1120, 195), (1000, 220)], fill=_A(p["light"]))
    pts = [(300, 380), (300, 300), (420, 300), (420, 420), (360, 420), (360, 340)]
    d.line(pts, fill=_A(p["main"]), width=10, joint="curve")
    pts2 = [(560, 400), (560, 280), (740, 280), (740, 440), (650, 440), (650, 340)]
    d.line(pts2, fill=_A(p["light"]), width=10, joint="curve")
    d.rectangle([860, 300, 1160, 330], fill=_A(p["mid"]))
    d.rectangle([860, 360, 1160, 480], fill=_A(p["light"], 220))


def m_clockwork(d, p, b, r):
    _gear(d, 470, 360, 270, 220, 14, _A(p["main"]), 60, 0.1)
    _gear(d, 850, 270, 170, 135, 9, _A(p["accent"]), 40, 0.3)
    _gear(d, 880, 560, 110, 85, 7, _A(p["light"]), 28, 0.0)
    _C(d, 470, 360, 30, fill=_A(p["light"]))


def m_metal_clank(d, p, b, r):
    _RR(d, 120, 110, 1160, 610, 24, fill=_A(lerp(p["mid"], (200, 200, 210), 0.35)), out=_A(p["light"]), w=8)
    for x in (170, 1110):
        for y in (160, 560):
            _C(d, x, y, 22, fill=_A(p["bg"]), out=_A(p["light"]), w=4)
    for k in range(5):
        d.line([(300, 200 + k * 70), (980, 200 + k * 70 + (k % 2) * 14)], fill=_A(p["bg"], 160), width=4)
    for k in range(9):
        a = TAU * k / 9
        d.line([(640 + 60 * math.cos(a), 360 + 60 * math.sin(a)), (640 + 190 * math.cos(a), 360 + 190 * math.sin(a))], fill=_A(p["accent"]), width=6)
    _C(d, 640, 360, 38, fill=_A(p["light"]))


def m_blade(d, p, b, r):
    for ang, col in ((-35, p["light"]), (35, p["accent"])):
        a = math.radians(ang)
        ux, uy = math.cos(a), math.sin(a)
        x0, y0 = 640 - 480 * ux, 360 - 480 * uy
        x1, y1 = 640 + 480 * ux, 360 + 480 * uy
        d.polygon([(x0 + 18 * uy, y0 - 18 * ux), (x0 - 18 * uy, y0 + 18 * ux), (x1, y1)], fill=_A(col, 240))
        gx, gy = 640 - 330 * ux, 360 - 330 * uy
        d.line([(gx + 70 * uy, gy - 70 * ux), (gx - 70 * uy, gy + 70 * ux)], fill=_A(p["main"]), width=18)
    for k in range(12):
        a = TAU * k / 12
        d.line([(640 + 40 * math.cos(a), 360 + 40 * math.sin(a)), (640 + 110 * math.cos(a), 360 + 110 * math.sin(a))], fill=_A((255, 255, 255)), width=5)


def m_abacus(d, p, b, r):
    _RR(d, 120, 110, 1160, 610, 24, out=_A(p["light"]), w=18)
    d.rectangle([120, 250, 1160, 270], fill=_A(p["light"]))
    for k in range(7):
        x = 230 + k * 135
        d.line([(x, 120), (x, 600)], fill=_A(p["accent"], 230), width=7)
        d.ellipse([x - 40, 160, x + 40, 235], fill=_A(p["main"]))
        n = 1 + (k * 2) % 4
        for j in range(4):
            y = 300 + j * 62 + (0 if j < n else 70)
            d.ellipse([x - 40, y, x + 40, y + 55], fill=_A(lerp(p["main"], p["accent"], j / 3)))


def m_cardboard(d, p, b, r):
    base = (200, 150, 95)
    d.rectangle([280, 130, 1000, 600], fill=_A(base))
    d.polygon([(280, 130), (1000, 130), (900, 40), (380, 40)], fill=_A((180, 130, 80)))
    d.polygon([(1000, 130), (1000, 600), (1130, 500), (1130, 40)], fill=_A((160, 112, 66)))
    d.rectangle([590, 130, 690, 600], fill=_A((230, 200, 140), 230))
    for y in range(160, 600, 26):
        d.line([(285, y), (995, y)], fill=_A((150, 105, 60), 90), width=2)
    d.rectangle([700, 330, 960, 410], fill=_A(p["light"], 230))
    for k in range(4):
        d.line([(730 + k * 52, 345), (730 + k * 52, 395)], fill=_A(p["bg"]), width=6)


def m_bubble(d, p, b, r):
    for row in range(5):
        for c in range(11):
            x, y = 130 + c * 100 + (row % 2) * 50, 120 + row * 120
            if r.random() < 0.28:
                d.arc([x - 36, y - 36, x + 36, y + 36], 20, 160, fill=_A(p["light"], 190), width=4)
                for kk in range(3):
                    a = r.random() * TAU
                    d.line([(x, y), (x + 50 * math.cos(a), y + 50 * math.sin(a))], fill=_A(p["accent"], 200), width=3)
            else:
                _C(d, x, y, 44, fill=_A(p["light"], 60), out=_A(p["light"], 190), w=3)
                _C(d, x - 14, y - 14, 10, fill=_A((255, 255, 255), 190))


def m_ice(d, p, b, r):
    d.polygon([(380, 120), (900, 120), (840, 640), (440, 640)], fill=_A(p["light"], 60), outline=_A(p["light"]), width=8)
    d.polygon([(395, 260), (885, 260), (840, 640), (440, 640)], fill=_A(p["main"], 90))
    for cx, cy, a in ((560, 330, 0.3), (700, 400, -0.2), (600, 500, 0.5), (760, 280, 0.1)):
        pts = [(cx + 60 * math.cos(a + k * math.pi / 2 + 0.78), cy + 60 * math.sin(a + k * math.pi / 2 + 0.78)) for k in range(4)]
        d.polygon(pts, fill=_A((255, 255, 255), 130), outline=_A(p["light"]))
    for k in range(6):
        _C(d, 480 + k * 60, 400 + (k % 3) * 70, 6, fill=_A((255, 255, 255), 170))


def m_rain(d, p, b, r):
    _RR(d, 160, 60, 1120, 660, 20, fill=_A(p["bg"], 120), out=_A(p["light"]), w=14)
    d.line([(640, 60), (640, 660)], fill=_A(p["light"]), width=12)
    d.line([(160, 360), (1120, 360)], fill=_A(p["light"]), width=12)
    for _ in range(46):
        x, y, rad = r.randrange(190, 1090), r.randrange(90, 600), r.randrange(8, 20)
        d.line([(x, y), (x + r.randrange(-6, 6), y + r.randrange(40, 140))], fill=_A(p["light"], 90), width=3)
        d.ellipse([x - rad * 0.7, y - rad, x + rad * 0.7, y + rad], fill=_A(p["accent"], 150), outline=_A(p["light"], 220))


def m_jazz(d, p, b, r):
    _C(d, 520, 380, 280, fill=_A(p["bg"]), out=_A(p["light"]), w=14)
    _C(d, 520, 380, 245, fill=_A((235, 235, 225), 235))
    for k in range(10):
        a = TAU * k / 10
        d.rectangle([520 + 285 * math.cos(a) - 12, 380 + 285 * math.sin(a) - 12, 520 + 285 * math.cos(a) + 12, 380 + 285 * math.sin(a) + 12], fill=_A(p["accent"]))
    for k in range(9):
        a = math.radians(-70 + k * 12)
        d.line([(1100, 640), (1100 + 600 * math.cos(math.pi + a) * 0.9, 640 + 600 * math.sin(math.pi + a) * 0.9)], fill=_A(p["main"], 210), width=3)
    d.line([(1100, 640), (1180, 700)], fill=_A(p["light"]), width=20)


def m_songbird(d, p, b, r):
    d.line([(60, 560), (1220, 500)], fill=_A(p["mid"]), width=22)
    d.ellipse([360, 280, 800, 540], fill=_A(p["main"]))
    d.polygon([(800, 380), (1100, 520), (780, 470)], fill=_A(p["mid"]))
    _C(d, 420, 280, 96, fill=_A(p["main"]))
    d.polygon([(330, 270), (240, 300), (330, 320)], fill=_A(p["accent"]))
    _C(d, 405, 262, 14, fill=_A(p["bg"]))
    d.ellipse([440, 330, 720, 470], fill=_A(p["accent"], 240))
    for k, (x, y) in enumerate(((190, 150), (130, 80), (250, 70))):
        d.arc([x - 60 - k * 20, y - 40, x + 60 + k * 20, y + 60], 200, 340, fill=_A(p["light"], 220), width=6)


def m_frog(d, p, b, r):
    for k in range(5):
        d.ellipse([140 - k * 40, 440 + k * 12, 1140 + k * 40, 680 - k * 12], outline=_A(p["light"], 90 - k * 12), width=3)
    for cx, cy, rr in ((260, 560, 120), (1010, 540, 140), (650, 600, 110)):
        d.pieslice([cx - rr, cy - rr * 0.45, cx + rr, cy + rr * 0.45], 20, 340, fill=_A(p["mid"]), outline=_A(p["light"]))
    d.ellipse([430, 220, 850, 540], fill=_A(p["main"]))
    for x in (500, 780):
        _C(d, x, 230, 62, fill=_A(p["main"]), out=_A(p["bg"]), w=5)
        _C(d, x, 230, 34, fill=_A(p["light"]))
        _C(d, x, 230, 17, fill=_A(p["bg"]))
    d.arc([520, 330, 760, 460], 10, 170, fill=_A(p["bg"]), width=8)


def m_crickets(d, p, b, r):
    _C(d, 960, 200, 120, fill=_A(p["light"], 250))
    _C(d, 1000, 175, 112, fill=_A(p["bg"], 255))
    for _ in range(70):
        x = r.randrange(0, 1280, 13)
        h = r.randrange(120, 360)
        d.polygon([(x, 720), (x + 12, 720), (x + r.randrange(-30, 40), 720 - h)], fill=_A(lerp(p["mid"], p["main"], r.random()), 235))
    d.ellipse([380, 410, 640, 490], fill=_A(p["bg"], 255), outline=_A(p["light"]), width=4)
    d.line([(440, 430), (360, 340), (300, 380)], fill=_A(p["light"]), width=6)
    d.line([(520, 430), (480, 330), (420, 360)], fill=_A(p["light"]), width=6)
    for k in range(3):
        d.arc([240 - k * 40, 330 - k * 40, 440 + k * 40, 530 + k * 40], 200, 270, fill=_A(p["accent"], 200 - k * 50), width=5)


def m_owl(d, p, b, r):
    _C(d, 640, 270, 230, fill=_A(p["light"], 60))
    d.polygon([(440, 140), (520, 250), (400, 300)], fill=_A(p["mid"]))
    d.polygon([(840, 140), (760, 250), (880, 300)], fill=_A(p["mid"]))
    d.ellipse([420, 200, 860, 640], fill=_A(p["mid"]))
    for x in (540, 740):
        _C(d, x, 370, 96, fill=_A(p["accent"]), out=_A(p["light"]), w=8)
        _C(d, x, 370, 50, fill=_A(p["bg"]))
        _C(d, x - 14, 356, 14, fill=_A((255, 255, 255)))
    d.polygon([(610, 430), (670, 430), (640, 500)], fill=_A(p["main"]))
    d.line([(60, 660), (1220, 640)], fill=_A(p["bg"]), width=40)


def m_tide(d, p, b, r):
    for k in range(7):
        y = 230 + k * 70
        _wave(d, 0, 1290, y, 40 + k * 4, 70 + k * 12, k * 0.9, _A(lerp(p["accent"], p["bg"], k / 7), 255), 3)
        pts = [(x, y + (40 + k * 4) * math.sin(x / (70 + k * 12) + k * 0.9)) for x in range(0, 1290, 6)]
        d.polygon(pts + [(1280, 720), (0, 720)], fill=_A(lerp(p["main"], p["bg"], (k + 1) / 8), 200))
    _C(d, 980, 120, 70, fill=_A(p["light"], 230))


def m_hud(d, p, b, r):
    for x0, y0, sx, sy in ((90, 70, 1, 1), (1190, 70, -1, 1), (90, 650, 1, -1), (1190, 650, -1, -1)):
        d.line([(x0, y0 + sy * 150), (x0, y0), (x0 + sx * 150, y0)], fill=_A(p["accent"]), width=8)
    _C(d, 640, 360, 190, out=_A(p["main"]), w=5)
    _C(d, 640, 360, 20, out=_A(p["light"]), w=4)
    for a in range(0, 360, 90):
        d.line([(640 + 150 * math.cos(math.radians(a)), 360 + 150 * math.sin(math.radians(a))), (640 + 260 * math.cos(math.radians(a)), 360 + 260 * math.sin(math.radians(a)))], fill=_A(p["light"]), width=5)
    for i, v in enumerate(b[:12]):
        d.rectangle([150, 160 + i * 36, 150 + 40 + v * 200, 186 + i * 36], fill=_A(p["main"], 220))
        d.rectangle([1130 - 40 - b[(i + 5) % 12] * 200, 160 + i * 36, 1130, 186 + i * 36], fill=_A(p["accent"], 220))


def m_sonar(d, p, b, r):
    cx, cy = 640, 370
    for k in range(1, 6):
        _C(d, cx, cy, k * 62, out=_A(p["main"], 200), w=3)
    d.line([(cx - 330, cy), (cx + 330, cy)], fill=_A(p["main"], 120), width=2)
    d.line([(cx, cy - 330), (cx, cy + 330)], fill=_A(p["main"], 120), width=2)
    d.pieslice([cx - 320, cy - 320, cx + 320, cy + 320], -60, 0, fill=_A(p["accent"], 130))
    for x, y in ((820, 250), (520, 450), (760, 480)):
        _C(d, x, y, 14, fill=_A(p["light"]))
        _C(d, x, y, 30, out=_A(p["light"], 140), w=3)


def m_geiger(d, p, b, r):
    cx, cy = 640, 560
    d.pieslice([cx - 450, cy - 450, cx + 450, cy + 450], 180, 360, fill=_A(p["bg"]), outline=_A(p["light"]), width=8)
    for k in range(21):
        a = math.radians(180 + k * 9)
        l = 40 if k % 5 == 0 else 22
        d.line([(cx + 400 * math.cos(a), cy + 400 * math.sin(a)), (cx + (400 - l) * math.cos(a), cy + (400 - l) * math.sin(a))], fill=_A(p["light"]), width=5)
    a = math.radians(180 + 118)
    d.line([(cx, cy), (cx + 360 * math.cos(a), cy + 360 * math.sin(a))], fill=_A(p["accent"]), width=10)
    _C(d, cx, cy, 30, fill=_A(p["main"]))
    for k in range(3):
        d.pieslice([540, 130, 740, 330], 270 + k * 120 - 30, 270 + k * 120 + 30, fill=_A(p["main"], 230))
    _C(d, 640, 230, 26, fill=_A(p["light"]))


def m_analog(d, p, b, r):
    d.rectangle([100, 100, 800, 620], fill=_A(p["bg"], 230), outline=_A(p["light"], 160), width=4)
    for k in range(1, 7):
        d.line([(100 + k * 100, 100), (100 + k * 100, 620)], fill=_A(p["light"], 40), width=2)
    for k in range(1, 5):
        d.line([(100, 100 + k * 104), (800, 100 + k * 104)], fill=_A(p["light"], 40), width=2)
    pts = [(100 + x, 520 - 330 * math.exp(-((x - 430) / 180) ** 2) - 120 * math.exp(-((x - 560) / 40) ** 2) * (1 if x > 400 else 0)) for x in range(0, 701, 6)]
    d.polygon(pts + [(800, 620), (100, 620)], fill=_A(p["main"], 90))
    d.line(pts, fill=_A(p["accent"]), width=7, joint="curve")
    for i, (cx, cy) in enumerate(((960, 230), (1100, 230), (960, 470), (1100, 470))):
        _C(d, cx, cy, 58, fill=_A(p["mid"]), out=_A(p["light"]), w=5)
        a = math.radians(-120 + i * 70)
        d.line([(cx, cy), (cx + 48 * math.cos(a), cy + 48 * math.sin(a))], fill=_A(p["accent"]), width=7)


def m_hologram(d, p, b, r):
    d.polygon([(640, 140), (400, 560), (880, 560)], fill=_A(p["main"], 40))
    d.ellipse([360, 520, 920, 640], fill=_A(p["bg"]), outline=_A(p["light"]), width=6)
    d.ellipse([440, 545, 840, 615], outline=_A(p["accent"]), width=4)
    cx, cy, s = 640, 340, 130
    A = [(cx - s * 0.6, cy - s * 0.2), (cx + s * 0.6, cy - s * 0.2), (cx + s * 0.6, cy + s), (cx - s * 0.6, cy + s)]
    B = [(x + 70, y - 80) for x, y in A]
    for P in (A, B):
        d.line(P + [P[0]], fill=_A(p["light"]), width=5)
    for a_, b_ in zip(A, B):
        d.line([a_, b_], fill=_A(p["accent"]), width=4)
    for k in range(40):
        x, y = r.randrange(300, 980), r.randrange(80, 520)
        _C(d, x, y, r.randrange(2, 6), fill=_A(p["light"], 200))


def m_dialup(d, p, b, r):
    for i in range(12):
        x, y = 120 + (i % 3) * 120, 130 + (i // 3) * 120
        _RR(d, x, y, x + 100, y + 100, 20, fill=_A(p["mid"]), out=_A(p["light"]), w=3)
        _C(d, x + 50, y + 50, 14, fill=_A(p["light"], 220))
    d.line([(520, 360), (580, 360)], fill=_A(p["accent"]), width=6)
    for k in range(3):
        _wave(d, 600, 1230, 200 + k * 160, 50, 28 - k * 8, k, _A([p["main"], p["accent"], p["light"]][k]), 4, f=lambda u: 1.0 if math.sin(u) > 0 else -1.0)
    for i in range(20):
        d.text((600 + i * 30, 620), "01"[r.randrange(2)], fill=_A(p["light"], 190))


def m_robot(d, p, b, r):
    _RR(d, 330, 90, 950, 630, 70, fill=_A(p["mid"]), out=_A(p["light"]), w=10)
    d.line([(640, 90), (640, 30)], fill=_A(p["light"]), width=8)
    _C(d, 640, 24, 22, fill=_A(p["accent"]))
    for x in (470, 810):
        _RR(d, x - 80, 190, x + 80, 330, 40, fill=_A(p["bg"]), out=_A(p["accent"]), w=6)
        _C(d, x, 260, 34, fill=_A(p["light"]))
    _RR(d, 400, 410, 880, 590, 30, fill=_A(p["bg"]))
    for i, v in enumerate(b[:16]):
        h = 20 + v * 130
        d.rectangle([425 + i * 28, 570 - h, 445 + i * 28, 570], fill=_A(lerp(p["main"], p["accent"], i / 15)))


def m_theremin(d, p, b, r):
    d.rounded_rectangle([320, 460, 960, 660], radius=24, fill=_A(p["mid"]), outline=_A(p["light"]), width=6)
    d.line([(400, 460), (400, 80)], fill=_A(p["light"]), width=14)
    d.arc([700, 120, 1000, 520], 270, 90, fill=_A(p["light"]), width=14)
    for k in range(5):
        d.arc([160 - k * 50, 240 - k * 50, 640 + k * 50, 720 + k * 50], 270, 340, fill=_A(p["main"], 220 - k * 35), width=4)
        d.arc([640 - k * 40, 90 - k * 40, 1180 + k * 40, 600 + k * 40], 190, 260, fill=_A(p["accent"], 220 - k * 35), width=4)
    _wave(d, 100, 1180, 400, 36, 22, 0, _A(p["light"], 150), 3)
    _C(d, 400, 70, 22, fill=_A(p["accent"]))


def m_warp(d, p, b, r):
    cx, cy = 640, 360
    for _ in range(190):
        a = r.random() * TAU
        r0 = r.randrange(40, 260)
        r1 = r0 + r.randrange(80, 520)
        d.line([(cx + r0 * math.cos(a), cy + r0 * math.sin(a)), (cx + r1 * math.cos(a), cy + r1 * math.sin(a))], fill=_A(lerp(p["main"], p["light"], r.random()), r.randrange(80, 255)), width=r.randrange(2, 7))
    _C(d, cx, cy, 36, fill=_A((255, 255, 255)))
    _C(d, cx, cy, 110, out=_A(p["accent"], 190), w=4)


def m_lofi(d, p, b, r):
    d.arc([300, 100, 980, 740], 180, 360, fill=_A(p["light"]), width=26)
    for x in (320, 960):
        _RR(d, x - 90, 330, x + 90, 570, 60, fill=_A(p["main"]), out=_A(p["light"]), w=8)
        _RR(d, x - 55, 380, x + 55, 540, 40, fill=_A(p["bg"]))
    for k in range(4):
        _wave(d, 420, 860, 440 + (k - 1.5) * 30, 16 + k * 5, 20, k * 0.7, _A(lerp(p["accent"], p["light"], k / 3), 230), 4)
    _C(d, 640, 150, 36, fill=_A(p["accent"], 230))


def m_vinyl(d, p, b, r):
    cx, cy = 500, 360
    _C(d, cx, cy, 330, fill=_A((12, 12, 14)), out=_A(p["light"], 160), w=4)
    for rr in range(120, 320, 12):
        _C(d, cx, cy, rr, out=_A((70, 70, 80), 200), w=1)
    d.pieslice([cx - 320, cy - 320, cx + 320, cy + 320], 300, 340, fill=_A((255, 255, 255), 30))
    _C(d, cx, cy, 110, fill=_A(p["main"]))
    _C(d, cx, cy, 10, fill=_A(p["bg"]))
    d.line([(1100, 100), (1100, 330), (770, 480)], fill=_A(p["light"]), width=14, joint="curve")
    _C(d, 1100, 100, 36, fill=_A(p["accent"]))
    d.polygon([(770, 480), (740, 450), (810, 440)], fill=_A(p["light"]))


def m_vapor(d, p, b, r):
    _C(d, 640, 290, 220, fill=_A(p["accent"]))
    for k in range(7):
        d.rectangle([420, 300 + k * 26, 860, 300 + k * 26 + 4 + k * 3], fill=_A(p["bg"]))
    d.rectangle([0, 420, 1280, 720], fill=_A(p["bg"]))
    for k in range(12):
        y = 420 + 300 * ((k / 11) ** 2)
        d.line([(0, y), (1280, y)], fill=_A(p["main"]), width=3)
    for k in range(-12, 13):
        d.line([(640 + k * 20, 420), (640 + k * 140, 720)], fill=_A(p["main"]), width=3)
    for k in range(3):
        d.line([(160 + k * 24, 420), (150 + k * 30, 300 - k * 25)], fill=_A(p["light"]), width=7)


def m_hap_tick(d, p, b, r):
    _RR(d, 450, 70, 830, 650, 56, fill=_A(p["bg"]), out=_A(p["light"]), w=8)
    _RR(d, 480, 130, 800, 580, 20, fill=_A(p["mid"]))
    for sx in (-1, 1):
        for k in range(3):
            x = 640 + sx * (260 + k * 70)
            d.line([(x, 300 + k * 20), (x, 420 - k * 20)], fill=_A(p["accent"], 230 - k * 60), width=10)
    _C(d, 640, 360, 14, fill=_A(p["light"]))


def m_hap_thud(d, p, b, r):
    for k in range(8):
        _C(d, 640, 360, 50 + k * 45, out=_A(lerp(p["light"], p["main"], k / 7), 255 - k * 28), w=18 - k * 2)
    _C(d, 640, 360, 38, fill=_A(p["light"]))


def m_hap_buzz(d, p, b, r):
    _RR(d, 470, 70, 810, 650, 56, fill=_A(p["bg"]), out=_A(p["light"]), w=8)
    for dx in (-18, 0, 18):
        _RR(d, 470 + dx + 14, 70, 810 + dx - 14, 650, 56, out=_A(p["accent"], 150), w=3)
    for sx in (-1, 1):
        for k in range(5):
            x0 = 640 + sx * (230 + k * 40)
            pts = [(x0 + (12 if i % 2 else -12), 200 + i * 30) for i in range(9)]
            d.line(pts, fill=_A(p["main"], 240 - k * 40), width=6)
    _C(d, 640, 360, 70, fill=_A(p["mid"]), out=_A(p["light"]), w=5)


# (cover motif, background style) per pack
NEW_COVER = {
    "snd-pulse-quest": (m_pulse_quest, "dots"), "snd-chip-drums": (m_chip_drums, "stripes"), "snd-pc-speaker": (m_pc_speaker, "flat"),
    "snd-wavetable-pocket": (m_wavetable, "grid"), "snd-coin-slot": (m_coin_slot, "rad"), "snd-sixteen-bit": (m_sixteen, "v"),
    "snd-pwm-tracker": (m_pwm_tracker, "grid"), "snd-fm-bells": (m_fm_bells, "rad"), "snd-rhodes": (m_rhodes, "v"),
    "snd-vibraphone": (m_vibraphone, "rad"), "snd-glockenspiel": (m_glock, "dots"), "snd-drawbar-organ": (m_drawbar, "stripes"),
    "snd-music-box": (m_music_box, "v"), "snd-harp": (m_harp, "rad"), "snd-nylon-guitar": (m_nylon, "v"),
    "snd-steel-string": (m_steel, "stripes"), "snd-koto": (m_koto, "rad"), "snd-shamisen": (m_shamisen, "dots"),
    "snd-pizzicato": (m_pizz, "v"), "snd-steel-drum": (m_steel_drum, "rad"), "snd-handpan": (m_handpan, "rad"),
    "snd-gamelan": (m_gamelan, "v"), "snd-xylophone": (m_xylo, "flat"), "snd-tubular-bells": (m_tubular, "v"),
    "snd-singing-bowl": (m_singing_bowl, "rad"), "snd-wind-chimes": (m_wind_chimes, "v"), "snd-bamboo": (m_bamboo, "flat"),
    "snd-hand-drum": (m_hand_drum, "stripes"), "snd-blue-switch": (m_blue_switch, "grid"), "snd-thock-board": (m_thock, "flat"),
    "snd-camera-shutter": (m_camera, "rad"), "snd-rotary-dial": (m_rotary, "dots"), "snd-cassette-deck": (m_cassette, "stripes"),
    "snd-toggle-panel": (m_toggle, "grid"), "snd-office-desk": (m_office, "flat"), "snd-clockwork": (m_clockwork, "dots"),
    "snd-metal-clank": (m_metal_clank, "stripes"), "snd-blade-ring": (m_blade, "rad"), "snd-abacus-beads": (m_abacus, "flat"),
    "snd-cardboard": (m_cardboard, "flat"), "snd-bubble-wrap": (m_bubble, "rad"), "snd-ice-cubes": (m_ice, "v"),
    "snd-rain-window": (m_rain, "v"), "snd-jazz-brushes": (m_jazz, "rad"), "snd-songbird": (m_songbird, "v"),
    "snd-frog-pond": (m_frog, "v"), "snd-crickets": (m_crickets, "v"), "snd-night-owl": (m_owl, "rad"),
    "snd-tide-swell": (m_tide, "v"), "snd-hud-blips": (m_hud, "grid"), "snd-sonar-ping": (m_sonar, "grid"),
    "snd-geiger-counter": (m_geiger, "stripes"), "snd-analog-sweeps": (m_analog, "grid"), "snd-hologram": (m_hologram, "rad"),
    "snd-dialup-tones": (m_dialup, "grid"), "snd-robot-voice": (m_robot, "dots"), "snd-theremin": (m_theremin, "rad"),
    "snd-warp-drive": (m_warp, "rad"), "snd-lofi-tape": (m_lofi, "v"), "snd-vinyl-pop": (m_vinyl, "rad"),
    "snd-vaporwave": (m_vapor, "v"), "snd-haptic-tick": (m_hap_tick, "flat"), "snd-haptic-thud": (m_hap_thud, "rad"),
    "snd-haptic-buzz": (m_hap_buzz, "stripes"),
}



# ----------------------------------------------------------------------------- wave 2 covers: one motif per pack
def _bars_bottom(d, p, b, y=690, h=70, n=32, x0=140, step=31):
    for i, v in enumerate(b[:n]):
        d.rectangle([x0 + i * step, y - v * h, x0 + i * step + step - 6, y], fill=_A(p["light"], 120))


def c_sitar(d, p, b, r):
    _C(d, 290, 500, 175, fill=_A(p["mid"]), out=_A(p["light"], 230), w=7)
    d.line([(400, 430), (1190, 190)], fill=_A(p["light"]), width=40)
    for k in range(10):
        t = 0.1 + k * 0.09
        x, y = 400 + t * 790, 430 - t * 240
        d.line([(x - 8, y - 28), (x + 8, y + 28)], fill=_A(p["accent"]), width=5)
    for k in range(-2, 3):
        d.line([(300, 500 + k * 7), (1190, 190 + k * 7)], fill=_A(p["accent"], 200), width=2)
    for i in range(5):
        _C(d, 290, 500, 40 + i * 28, out=_A(p["main"], 120), w=2)


def c_tabla(d, p, b, r):
    for cx, cy, w, h in ((380, 400, 170, 250), (800, 450, 240, 200)):
        d.rectangle([cx - w, cy, cx + w, cy + h], fill=_A(p["mid"]))
        d.ellipse([cx - w, cy - 55, cx + w, cy + 55], fill=_A(p["light"], 240), outline=_A(p["accent"]), width=6)
        d.ellipse([cx - w * 0.55, cy - 28, cx + w * 0.55, cy + 28], fill=_A(p["bg"]))
        for k in range(10):
            x = cx - w + k * (2 * w / 9)
            d.line([(x, cy + 30), (x + (k % 2 * 2 - 1) * 20, cy + h)], fill=_A(p["accent"], 200), width=3)
    _C(d, 640, 150, 28, fill=_A(p["accent"], 200))


def c_oud(d, p, b, r):
    _C(d, 460, 480, 210, fill=_A(p["mid"]), out=_A(p["light"]), w=7)
    d.polygon([(560, 420), (1050, 320), (1050, 380), (580, 540)], fill=_A(p["light"], 235))
    d.polygon([(1050, 320), (1190, 250), (1190, 330), (1050, 380)], fill=_A(p["accent"]))
    _C(d, 450, 470, 60, fill=_A(p["bg"]), out=_A(p["accent"]), w=5)
    for a in range(0, 360, 30):
        d.line([(450 + 60 * math.cos(math.radians(a)), 470 + 60 * math.sin(math.radians(a))),
                (450 + 90 * math.cos(math.radians(a)), 470 + 90 * math.sin(math.radians(a)))], fill=_A(p["accent"], 200), width=3)


def c_erhu(d, p, b, r):
    d.line([(640, 40), (640, 520)], fill=_A(p["light"]), width=26)
    d.polygon([(560, 520), (640, 470), (720, 520), (720, 640), (640, 690), (560, 640)], fill=_A(p["mid"]), outline=_A(p["accent"]), width=8)
    for off in (-5, 5):
        d.line([(640 + off, 70), (640 + off, 600)], fill=_A(p["main"]), width=2)
    d.line([(330, 470), (950, 300)], fill=_A(p["accent"]), width=7)
    for k in range(4):
        d.arc([380 + k * 70, 330 - k * 12, 520 + k * 70, 480 - k * 12], 200, 340, fill=_A(p["main"], 130), width=3)


def c_balafon(d, p, b, r):
    for i in range(11):
        t = i / 10
        x = 160 + i * 96
        h = 260 - 100 * t
        y = 190 + 90 * math.sin(t * math.pi) * 0.4
        _RR(d, x - 40, y, x + 40, y + h, 14, fill=_A(p["light"], 235), out=_A(p["accent"]), w=3)
        _C(d, x, y + h + 80, 50 - 20 * t, fill=_A(p["mid"]), out=_A(p["main"], 200), w=3)


def c_didgeridoo(d, p, b, r):
    d.polygon([(100, 640), (180, 700), (1130, 230), (1170, 120)], fill=_A(p["mid"]), outline=_A(p["accent"]))
    for k in range(9):
        t = 0.08 + k * 0.1
        x, y = 140 + t * 1000, 670 - t * 470
        d.line([(x - 25, y + 35), (x + 25, y - 30)], fill=_A(p["light"], 220), width=7)
        _C(d, x + 8, y + 20, 6, fill=_A(p["accent"]))
    for k in range(4):
        d.arc([1050 - k * 70, 70 - k * 70, 1250 + k * 70 * 0, 270 + k * 0], 90, 270, fill=_A(p["main"], 200 - k * 40), width=5)


def c_bagpipe(d, p, b, r):
    d.ellipse([380, 300, 900, 600], fill=_A(p["mid"]), outline=_A(p["light"]), width=7)
    for i, x in enumerate((520, 640, 760)):
        d.rectangle([x - 14, 90 + i * 25, x + 14, 330], fill=_A(p["light"]))
        d.rectangle([x - 20, 85 + i * 25, x + 20, 115 + i * 25], fill=_A(p["accent"]))
    d.polygon([(900, 520), (1120, 660), (1090, 700), (870, 570)], fill=_A(p["accent"]))
    for k in range(6):
        _C(d, 960 + k * 28, 560 + k * 18, 6, fill=_A(p["bg"]))
    for a in range(0, 300, 40):
        d.arc([420, 340, 860, 570], a, a + 20, fill=_A(p["main"], 150), width=4)


def c_accordion(d, p, b, r):
    d.rectangle([120, 190, 260, 560], fill=_A(p["mid"]), outline=_A(p["accent"]), width=5)
    d.rectangle([1020, 190, 1160, 560], fill=_A(p["mid"]), outline=_A(p["accent"]), width=5)
    for k in range(14):
        x = 260 + k * 54
        sh = 18 if k % 2 else 0
        d.polygon([(x, 200 + sh), (x + 54, 200 + 18 - sh), (x + 54, 550 - 18 + sh), (x, 550 - sh)], fill=_A(p["light"] if k % 2 else p["main"], 230), outline=_A(p["bg"]), width=2)
    for k in range(5):
        _C(d, 190, 250 + k * 65, 20, fill=_A(p["accent"]))
        _C(d, 1090, 250 + k * 65, 20, fill=_A(p["light"]))


def c_harmonica(d, p, b, r):
    _RR(d, 180, 250, 1100, 470, 24, fill=_A(p["light"], 245), out=_A(p["accent"]), w=6)
    for k in range(12):
        x = 230 + k * 74
        _RR(d, x, 300, x + 46, 360, 8, fill=_A(p["bg"], 255))
        d.line([(x + 6, 405), (x + 40, 405)], fill=_A(p["mid"]), width=4)
    _RR(d, 150, 230, 210, 490, 14, fill=_A(p["main"]))
    _RR(d, 1070, 230, 1130, 490, 14, fill=_A(p["main"]))


def c_ukulele(d, p, b, r):
    d.ellipse([300, 400, 640, 700], fill=_A(p["mid"]), outline=_A(p["light"]), width=7)
    d.ellipse([360, 290, 580, 480], fill=_A(p["mid"]), outline=_A(p["light"]), width=7)
    _C(d, 470, 470, 50, fill=_A(p["bg"]), out=_A(p["accent"]), w=5)
    d.rectangle([450, 40, 494, 330], fill=_A(p["light"]))
    for k in range(7):
        d.line([(450, 80 + k * 38), (494, 80 + k * 38)], fill=_A(p["accent"]), width=3)
    for k in range(4):
        d.line([(458 + k * 9, 40), (458 + k * 9, 600)], fill=_A(p["main"]), width=2)
    for i in range(4):
        _star(d, 820 + i * 100, 250 + (i % 2) * 120, 26, _A(p["accent"], 220))


def c_banjo(d, p, b, r):
    _C(d, 520, 430, 250, fill=_A(p["light"], 240), out=_A(p["accent"]), w=14)
    _C(d, 520, 430, 200, out=_A(p["mid"]), w=3)
    for a in range(0, 360, 20):
        _C(d, 520 + 232 * math.cos(math.radians(a)), 430 + 232 * math.sin(math.radians(a)), 7, fill=_A(p["mid"]))
    d.line([(760, 400), (1200, 250)], fill=_A(p["main"]), width=40)
    for k in range(4):
        d.line([(520, 430 + (k - 1.5) * 12), (1200, 250 + (k - 1.5) * 12)], fill=_A(p["bg"], 200), width=2)
    d.rectangle([480, 560, 560, 600], fill=_A(p["bg"]))


def c_taiko(d, p, b, r):
    d.polygon([(250, 260), (1030, 260), (1080, 420), (1030, 600), (250, 600), (200, 420)], fill=_A(p["mid"]), outline=_A(p["accent"]), width=8)
    d.ellipse([250, 180, 1030, 340], fill=_A(p["light"], 240), outline=_A(p["accent"]), width=8)
    for row in (300, 560):
        for k in range(14):
            _C(d, 240 + k * 58, row + (k % 2) * 10, 9, fill=_A(p["accent"]))
    d.line([(160, 130), (520, 270)], fill=_A(p["light"]), width=16)
    d.line([(1120, 130), (760, 270)], fill=_A(p["light"]), width=16)


def c_cajon(d, p, b, r):
    d.polygon([(300, 140), (980, 140), (1020, 640), (260, 640)], fill=_A(p["mid"]), outline=_A(p["light"]), width=7)
    _C(d, 640, 380, 95, fill=_A(p["bg"]), out=_A(p["accent"]), w=8)
    d.line([(300, 140), (260, 640)], fill=_A(p["accent"]), width=3)
    for k in range(6):
        d.line([(310, 190 + k * 80), (970, 190 + k * 80)], fill=_A(p["light"], 60), width=2)
    for i in range(3):
        d.arc([100 - i * 40, 330 - i * 40, 260 + i * 0, 430 + i * 40], 270, 90, fill=_A(p["main"], 200 - i * 50), width=6)


def c_latin(d, p, b, r):
    d.polygon([(240, 230), (480, 230), (540, 480), (190, 480)], fill=_A(p["accent"]), outline=_A(p["light"]), width=6)
    _C(d, 365, 330, 30, fill=_A(p["bg"], 200))
    for cx in (780, 940):
        d.ellipse([cx - 60, 200, cx + 60, 380], fill=_A(p["main"]), outline=_A(p["light"]), width=5)
        d.rectangle([cx - 14, 370, cx + 14, 560], fill=_A(p["light"]))
    d.line([(640, 620), (1080, 520)], fill=_A(p["light"]), width=22)
    d.line([(640, 540), (1080, 640)], fill=_A(p["light"]), width=22)


def c_panflute(d, p, b, r):
    for i in range(9):
        x = 250 + i * 86
        h = 520 - i * 38
        _RR(d, x, 100 + i * 24, x + 66, 100 + i * 24 + h, 28, fill=_A(p["light"], 235), out=_A(p["accent"]), w=3)
        _C(d, x + 33, 130 + i * 24, 13, fill=_A(p["bg"]))
    d.line([(240, 330), (1050, 460)], fill=_A(p["main"]), width=22)
    d.line([(240, 520), (1050, 560)], fill=_A(p["main"]), width=22)


def c_celesta(d, p, b, r):
    for i in range(12):
        w = 110 - i * 4
        x = 130 + i * 90
        d.rectangle([x, 330 - (11 - i) * 6, x + w - 18, 520], fill=_A(p["light"], 235), outline=_A(p["accent"]), width=3)
        d.rectangle([x, 520, x + w - 18, 560], fill=_A(p["mid"]))
    for _ in range(14):
        _star(d, r.randrange(80, 1200), r.randrange(60, 280), r.randrange(8, 30), _A(p["accent"], 220))
    d.line([(950, 200), (1100, 420)], fill=_A(p["main"]), width=10)
    _C(d, 1100, 420, 22, fill=_A(p["main"]))


def c_harpsichord(d, p, b, r):
    d.polygon([(130, 380), (1150, 380), (1190, 460), (90, 460)], fill=_A(p["mid"]), outline=_A(p["accent"]), width=4)
    d.polygon([(110, 460), (1170, 460), (1220, 600), (60, 600)], fill=_A(p["light"], 245), outline=_A(p["bg"]), width=3)
    for k in range(24):
        x = 100 + k * 44
        d.line([(x - 10 + k * 0.1, 460), (x - 22, 600)], fill=_A(p["bg"]), width=2)
    for k in (1, 2, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16, 18, 19, 20):
        d.polygon([(90 + k * 44 + 5, 460), (90 + k * 44 + 30, 460), (90 + k * 44 + 22, 540), (90 + k * 44 + 2, 540)], fill=_A(p["bg"]))
    d.arc([200, 80, 1080, 560], 190, 350, fill=_A(p["accent"]), width=12)
    for k in range(8):
        d.arc([260 + k * 22, 120 + k * 12, 1020 - k * 22, 500], 200, 340, fill=_A(p["main"], 150), width=3)


def c_pipeorgan(d, p, b, r):
    hs = [380, 470, 540, 600, 650, 600, 540, 470, 380]
    for i, h in enumerate(hs):
        x = 140 + i * 110
        d.rectangle([x, 700 - h, x + 84, 700], fill=_A(p["light"], 240), outline=_A(p["accent"]), width=3)
        d.polygon([(x, 700 - h), (x + 84, 700 - h), (x + 42, 700 - h - 40)], fill=_A(p["light"], 240), outline=_A(p["accent"]))
        d.rectangle([x + 14, 640 - h * 0.1, x + 70, 664 - h * 0.1], fill=_A(p["bg"]))
    d.rectangle([100, 700, 1180, 720], fill=_A(p["mid"]))


def c_churchbells(d, p, b, r):
    for cx, s, rot in ((460, 1.0, 0), (860, 0.7, 1)):
        top = 150 + rot * 90
        d.polygon([(cx - 40 * s, top), (cx + 40 * s, top), (cx + 190 * s, top + 330 * s), (cx + 230 * s, top + 400 * s),
                   (cx - 230 * s, top + 400 * s), (cx - 190 * s, top + 330 * s)], fill=_A(p["accent"], 240), outline=_A(p["light"]), width=6)
        _C(d, cx, top + 420 * s, 36 * s, fill=_A(p["mid"]))
        d.rectangle([cx - 60 * s, top - 40 * s, cx + 60 * s, top], fill=_A(p["mid"]))
    for k in range(3):
        d.arc([120 - k * 40, 250 - k * 40, 280, 520 + k * 40], 120, 240, fill=_A(p["main"], 180 - k * 50), width=5)


def c_house(d, p, b, r):
    for i in range(7):
        h = 140 + (i % 3) * 70
        for k in range(3):
            d.rectangle([140 + i * 150, 560 - k * 70 - (i % 3) * 14, 140 + i * 150 + 120, 560 - k * 70 - 56 - (i % 3) * 14], fill=_A(p["main"] if k else p["accent"], 200))
    _C(d, 640, 150, 90, fill=_A(p["light"], 150), out=_A(p["light"]), w=5)
    _C(d, 640, 150, 40, fill=_A(p["light"]))


def c_techno(d, p, b, r):
    for i in range(7):
        for j in range(4):
            if (i + j * 2) % 3 == 0:
                x, y = 160 + i * 150, 140 + j * 130
                d.polygon([(x, y - 50), (x + 50, y), (x, y + 50), (x - 50, y)], fill=_A(p["accent"] if (i + j) % 2 else p["main"], 220))
    for k in range(4):
        _C(d, 640, 360, 90 + k * 80, out=_A(p["light"], 130 - k * 25), w=3)


def c_dubstep(d, p, b, r):
    for k in range(4):
        pts = [(x, 360 + (150 - k * 30) * math.sin(x / (50 + k * 8) + k) * math.sin(x / 400 + 0.5)) for x in range(60, 1230, 6)]
        d.line(pts, fill=_A([p["main"], p["accent"], p["light"], p["mid"]][k], 240 - k * 40), width=16 - k * 3, joint="curve")
    d.rectangle([60, 640, 1220, 650], fill=_A(p["light"], 120))


def c_dnb(d, p, b, r):
    pat = ["x...x...x...x...", "....x.......x...", "x.x.x.x.x.x.x.x.", "x..x..x...x.x..."]
    cols = [p["main"], p["accent"], p["light"], p["mid"]]
    for j, row in enumerate(pat):
        for i, ch in enumerate(row):
            x, y = 130 + i * 69, 160 + j * 120
            _RR(d, x, y, x + 58, y + 96, 10, fill=_A(cols[j], 245 if ch == "x" else 40))
    _bars_bottom(d, p, b, 700, 60)


def c_trance(d, p, b, r):
    for k in range(7):
        ph = k * 0.4
        pts = []
        for x in range(60, 1230, 8):
            t = (x / 110 + ph) % 1
            pts.append((x, 380 + (t * 2 - 1) * 150 * (1 - k * 0.07)))
        d.line(pts, fill=_A(lerp(p["main"], p["accent"], k / 6), 220), width=5)
    for k in range(5):
        d.line([(640, 40), (200 + k * 220, 700)], fill=_A(p["light"], 60), width=3)


def c_idm(d, p, b, r):
    for _ in range(46):
        x, y = r.randrange(40, 1200), r.randrange(40, 640)
        w, h = r.choice((20, 40, 90, 160)), r.choice((8, 14, 30, 60))
        d.rectangle([x, y, x + w, y + h], fill=_A(r.choice((p["main"], p["accent"], p["light"], p["mid"])), r.randrange(80, 240)))
    for k in range(5):
        y = 120 + k * 120
        d.line([(0, y), (1280, y + r.randrange(-30, 30))], fill=_A(p["light"], 90), width=2)
    for k in range(8):
        _C(d, 160 + k * 140, 360, 6, fill=_A(p["light"]))


def c_acid(d, p, b, r):
    _C(d, 420, 380, 230, fill=_A(p["mid"]), out=_A(p["accent"]), w=10)
    for a in range(-135, 136, 15):
        ra = math.radians(a - 90)
        d.line([(420 + 250 * math.cos(ra), 380 + 250 * math.sin(ra)), (420 + 275 * math.cos(ra), 380 + 275 * math.sin(ra))], fill=_A(p["light"], 200), width=4)
    ra = math.radians(35 - 90)
    d.line([(420, 380), (420 + 190 * math.cos(ra), 380 + 190 * math.sin(ra))], fill=_A(p["light"]), width=14)
    _C(d, 420, 380, 26, fill=_A(p["accent"]))
    pts = [(780 + x, 560 - 220 * math.exp(-((x - 230) / 70) ** 2) - 40 * math.exp(-x / 200)) for x in range(0, 420, 6)]
    d.line(pts, fill=_A(p["main"]), width=9, joint="curve")


def c_futurebass(d, p, b, r):
    for i, (x, h) in enumerate(zip((220, 370, 520, 670, 820, 970), (300, 420, 340, 480, 380, 300))):
        _RR(d, x - 55, 640 - h, x + 55, 640, 55, fill=_A(lerp(p["main"], p["accent"], i / 5), 215))
    for _ in range(20):
        _star(d, r.randrange(80, 1200), r.randrange(60, 250), r.randrange(6, 20), _A(p["light"], 230))


def c_ambientpads(d, p, b, r):
    for k in range(7):
        y = 150 + k * 80
        d.ellipse([-100 + k * 60, y - 60, 900 + k * 60, y + 60], fill=_A(lerp(p["mid"], p["main"], k / 7), 70))
    d.ellipse([820, 130, 1060, 370], fill=_A(p["light"], 100))


def c_hyperpop(d, p, b, r):
    for i in range(5):
        _star(d, 200 + i * 220, 180 + (i % 2) * 220, 80 - i * 6, _A([p["main"], p["accent"], p["light"]][i % 3], 240))
    d.polygon([(640, 650), (440, 470), (500, 380), (640, 470), (780, 380), (840, 470)], fill=_A(p["accent"], 245))
    for k in range(12):
        _C(d, r.randrange(60, 1220), r.randrange(60, 660), r.randrange(4, 14), fill=_A(p["light"], 200))


def c_traphats(d, p, b, r):
    for y, s in ((260, 1.0), (420, 0.8)):
        d.ellipse([240, y - 24, 760, y + 24], fill=_A(p["light"], 235), outline=_A(p["accent"]), width=4)
        d.ellipse([440, y - 38, 560, y - 10], fill=_A(p["accent"]))
    d.line([(500, 470), (500, 700)], fill=_A(p["light"]), width=10)
    for k in range(12):
        _C(d, 830 + k * 34, 330 + (k % 2) * 10, 7 + (k % 3) * 3, fill=_A(p["main"]))


def c_drumbox(d, p, b, r):
    for i in range(4):
        for j in range(4):
            on = (i * 3 + j * 5) % 4 == 0
            x, y = 330 + i * 150, 130 + j * 140
            _RR(d, x, y, x + 122, y + 112, 18, fill=_A(p["accent"] if on else p["mid"], 255), out=_A(p["light"], 160), w=3)
    for k in range(5):
        _C(d, 130, 160 + k * 110, 22, fill=_A(p["main"], 220))


def c_floppy(d, p, b, r):
    _RR(d, 330, 80, 950, 680, 26, fill=_A(p["mid"]), out=_A(p["light"]), w=7)
    d.rectangle([450, 80, 830, 270], fill=_A(p["light"], 230))
    d.rectangle([700, 105, 780, 245], fill=_A(p["bg"]))
    d.rectangle([420, 380, 860, 680], fill=_A(p["light"], 245))
    for k in range(4):
        d.line([(450, 440 + k * 55), (830, 440 + k * 55)], fill=_A(p["accent"], 200), width=3)
    _C(d, 640, 330, 12, fill=_A(p["bg"]))


def c_crt(d, p, b, r):
    _RR(d, 250, 100, 1030, 600, 70, fill=_A(p["mid"]), out=_A(p["light"]), w=8)
    _RR(d, 310, 160, 970, 540, 50, fill=_A(p["bg"]))
    for y in range(170, 540, 9):
        d.line([(330, y), (950, y)], fill=_A(p["main"], 60), width=2)
    for k in range(3):
        d.arc([620 - 100 - k * 40, 350 - 100 - k * 40, 660 + 100 + k * 40, 350 + 100 + k * 40], 0, 360, fill=_A(p["accent"], 180 - k * 50), width=4)
    d.rectangle([520, 600, 760, 660], fill=_A(p["mid"]))


def c_dotmatrix(d, p, b, r):
    d.rectangle([320, 40, 960, 700], fill=_A(p["light"], 245))
    for y in range(60, 700, 60):
        _C(d, 345, y, 9, fill=_A(p["bg"], 200))
        _C(d, 935, y, 9, fill=_A(p["bg"], 200))
    for j in range(7):
        n = r.randrange(10, 22)
        for i in range(n):
            _C(d, 400 + i * 24, 150 + j * 70, 5, fill=_A(p["mid"], 255))
    d.rectangle([150, 120, 330, 160], fill=_A(p["accent"]))


def c_pinball(d, p, b, r):
    for cx, cy, rr in ((420, 220, 70), (700, 180, 60), (860, 330, 70)):
        _C(d, cx, cy, rr, fill=_A(p["accent"]), out=_A(p["light"]), w=8)
        _C(d, cx, cy, rr * 0.45, fill=_A(p["light"]))
    d.polygon([(300, 560), (570, 640), (570, 690), (290, 610)], fill=_A(p["main"]))
    d.polygon([(980, 560), (710, 640), (710, 690), (990, 610)], fill=_A(p["main"]))
    _C(d, 640, 460, 34, fill=_A(p["light"]))
    d.line([(640, 400), (700, 330)], fill=_A(p["light"], 120), width=4)


def c_slotreels(d, p, b, r):
    for i in range(3):
        x = 220 + i * 290
        _RR(d, x, 130, x + 250, 600, 22, fill=_A(p["light"], 245), out=_A(p["accent"]), w=8)
        if i == 0:
            _C(d, x + 125, 360, 80, fill=_A(p["accent"]))
        elif i == 1:
            d.rectangle([x + 40, 300, x + 210, 420], fill=_A(p["main"]))
            d.rectangle([x + 40, 330, x + 210, 390], fill=_A(p["bg"], 120))
        else:
            for k in range(3):
                d.line([(x + 60 + k * 0, 250 + k * 0), (x + 190, 250)], fill=_A(p["mid"]), width=26)
            d.line([(x + 190, 250), (x + 110, 480)], fill=_A(p["mid"]), width=26)
        d.line([(x - 20, 365), (x + 270, 365)], fill=_A(p["bg"], 130), width=2)
    _star(d, 1190, 90, 50, _A(p["accent"]))


def c_jukebox(d, p, b, r):
    d.pieslice([300, 60, 980, 740], 180, 360, fill=_A(p["mid"]), outline=_A(p["accent"]), width=10)
    d.rectangle([300, 400, 980, 700], fill=_A(p["mid"]), outline=_A(p["accent"]), width=10)
    for k in range(6):
        d.arc([360 + k * 30, 120 + k * 30, 920 - k * 30, 680 - k * 30], 180, 360, fill=_A(lerp(p["main"], p["accent"], k / 5), 230), width=8)
    _C(d, 640, 560, 110, fill=_A(p["bg"]), out=_A(p["light"]), w=5)
    _C(d, 640, 560, 30, fill=_A(p["accent"]))
    for a in range(0, 360, 40):
        _C(d, 640 + 70 * math.cos(math.radians(a)), 560 + 70 * math.sin(math.radians(a)), 3, fill=_A(p["light"], 150))


def c_telegraph(d, p, b, r):
    d.polygon([(260, 560), (1020, 560), (1060, 640), (220, 640)], fill=_A(p["mid"]), outline=_A(p["accent"]), width=5)
    d.line([(300, 470), (900, 420)], fill=_A(p["accent"]), width=26)
    _C(d, 960, 410, 52, fill=_A(p["light"]), out=_A(p["accent"]), w=6)
    d.rectangle([380, 470, 420, 560], fill=_A(p["light"]))
    x = 140
    for ln in (30, 90, 30, 30, 90, 90, 30):
        d.rectangle([x, 150, x + ln, 180], fill=_A(p["main"]))
        x += ln + 36


def c_cashregister(d, p, b, r):
    d.polygon([(230, 360), (1050, 360), (1090, 600), (190, 600)], fill=_A(p["mid"]), outline=_A(p["accent"]), width=7)
    d.polygon([(330, 130), (950, 130), (1000, 360), (280, 360)], fill=_A(p["light"], 245))
    for k in range(5):
        d.rectangle([390 + k * 100, 190, 460 + k * 100, 260], fill=_A(p["bg"]))
    for j in range(3):
        for i in range(6):
            _C(d, 330 + i * 124, 420 + j * 50, 18, fill=_A(p["accent"] if (i + j) % 2 else p["light"]))
    d.rectangle([230, 640, 1050, 690], fill=_A(p["accent"]))
    _C(d, 1130, 160, 36, fill=_A(p["main"]))


def c_shortwave(d, p, b, r):
    _RR(d, 160, 140, 1120, 640, 40, fill=_A(p["mid"]), out=_A(p["accent"]), w=7)
    d.pieslice([300, 160, 900, 760], 215, 325, fill=_A(p["bg"]))
    for k in range(25):
        a = math.radians(220 + k * 4.2)
        L_ = 40 if k % 5 == 0 else 22
        d.line([(600 + 330 * math.cos(a), 460 + 330 * math.sin(a)), (600 + (330 - L_) * math.cos(a), 460 + (330 - L_) * math.sin(a))], fill=_A(p["light"]), width=3)
    a = math.radians(262)
    d.line([(600, 460), (600 + 320 * math.cos(a), 460 + 320 * math.sin(a))], fill=_A(p["accent"]), width=6)
    for k in range(3):
        _C(d, 980, 330 + k * 80, 30, fill=_A(p["light"], 220))


def c_rpg(d, p, b, r):
    d.polygon([(300, 120), (400, 220), (300, 330), (200, 220)], fill=_A(p["accent"], 240), outline=_A(p["light"]), width=5)
    d.line([(200, 220), (400, 220)], fill=_A(p["light"], 140), width=3)
    for k in range(4):
        _RR(d, 560, 130 + k * 140, 1140, 220 + k * 140, 18, fill=_A(p["mid"], 255 if k != 1 else 255), out=_A(p["accent"] if k == 1 else p["light"], 180), w=5 if k == 1 else 2)
        d.line([(600, 175 + k * 140), (600 + 120 + k * 60, 175 + k * 140)], fill=_A(p["light"], 220), width=8)
    d.polygon([(500, 148 + 140), (540, 175 + 140), (500, 202 + 140)], fill=_A(p["accent"]))


def c_fighter(d, p, b, r):
    pts = []
    for i in range(24):
        a = TAU * i / 24
        rr = 330 if i % 2 == 0 else 160
        pts.append((640 + rr * math.cos(a), 360 + rr * 0.85 * math.sin(a)))
    d.polygon(pts, fill=_A(p["accent"], 240), outline=_A(p["light"]), width=6)
    d.polygon([(640 + (c * 0.5) * math.cos(TAU * i / 24), 360 + c * 0.42 * math.sin(TAU * i / 24)) for i, c in ((i, 330 if i % 2 == 0 else 160) for i in range(24))], fill=_A(p["main"], 255))
    for k in range(6):
        d.line([(80, 100 + k * 110), (330, 130 + k * 100)], fill=_A(p["light"], 130), width=5)


def c_racing(d, p, b, r):
    d.arc([140, 100, 700, 660], 150, 390, fill=_A(p["mid"]), width=40)
    for k in range(13):
        a = math.radians(150 + k * 20)
        d.line([(420 + 270 * math.cos(a), 380 + 270 * math.sin(a)), (420 + 230 * math.cos(a), 380 + 230 * math.sin(a))], fill=_A(p["light"], 230), width=5)
    a = math.radians(300)
    d.line([(420, 380), (420 + 220 * math.cos(a), 380 + 220 * math.sin(a))], fill=_A(p["accent"]), width=10)
    _C(d, 420, 380, 22, fill=_A(p["accent"]))
    for x in (820, 940, 1060):
        d.line([(x, 190), (x, 580)], fill=_A(p["light"], 180), width=8)
    d.line([(820, 385), (1060, 385)], fill=_A(p["light"], 180), width=8)
    _C(d, 940, 385, 56, fill=_A(p["main"]), out=_A(p["light"]), w=5)


def c_platform(d, p, b, r):
    for x, y, w in ((100, 520, 330), (560, 400, 250), (940, 300, 250)):
        d.rectangle([x, y, x + w, y + 160], fill=_A(p["mid"]), outline=_A(p["light"]), width=4)
        for k in range(int(w // 50)):
            d.rectangle([x + k * 50 + 4, y + 8, x + k * 50 + 46, y + 40], fill=_A(p["main"], 200))
    for k in range(18):
        t = k / 17
        x = 280 + t * 700
        y = 460 - 330 * math.sin(math.pi * t)
        _C(d, x, y, 8, fill=_A(p["light"], 200))
    _C(d, 1060, 230, 28, fill=_A(p["accent"]), out=_A(p["light"]), w=4)
    d.ellipse([240, 440, 320, 520], fill=_A(p["accent"]))


def c_puzzle(d, p, b, r):
    cols = [p["main"], p["accent"], p["light"], p["mid"]]
    for i in range(7):
        for j in range(5):
            c = cols[(i * 2 + j * 3 + (i * j)) % 4]
            _RR(d, 190 + i * 130, 90 + j * 120, 190 + i * 130 + 108, 90 + j * 120 + 100, 22, fill=_A(c, 230))
    for k in range(4):
        _RR(d, 190 + (k + 1) * 130 - 6, 90 + 2 * 120 - 6, 190 + (k + 1) * 130 + 114, 90 + 2 * 120 + 106, 26, out=_A(p["light"]), w=6)
    _star(d, 1100, 100, 40, _A(p["light"]))


def c_horror(d, p, b, r):
    d.polygon([(100, 360), (400, 160), (880, 160), (1180, 360), (880, 560), (400, 560)], fill=_A(p["light"], 235), outline=_A(p["accent"]), width=6)
    _C(d, 640, 360, 150, fill=_A(p["accent"]))
    d.ellipse([610, 230, 670, 490], fill=_A(p["bg"]))
    for k in range(7):
        a = math.radians(-30 + k * 10)
        d.line([(100, 360), (60, 360 + (k - 3) * 40)], fill=_A(p["main"], 200), width=3)
    for x in (140, 1140):
        d.line([(x, 40), (x, 140 + (x > 600) * 40)], fill=_A(p["main"], 200), width=3)


def c_starfighter(d, p, b, r):
    d.polygon([(640, 120), (780, 520), (640, 450), (500, 520)], fill=_A(p["light"], 245), outline=_A(p["accent"]), width=5)
    d.polygon([(500, 520), (360, 580), (500, 450)], fill=_A(p["mid"]))
    d.polygon([(780, 520), (920, 580), (780, 450)], fill=_A(p["mid"]))
    d.polygon([(610, 520), (640, 640), (670, 520)], fill=_A(p["accent"]))
    for x in (280, 1000):
        d.line([(x, 80), (x + 70, 340)], fill=_A(p["main"]), width=8)
    for _ in range(40):
        _C(d, r.randrange(40, 1240), r.randrange(30, 690), r.randrange(1, 4), fill=_A(p["light"], 200))


def c_shadow(d, p, b, r):
    _C(d, 980, 170, 100, fill=_A(p["light"], 120))
    for k in range(6):
        x, y = 140 + k * 170, 560 - k * 70
        a = 255 - k * 38
        d.ellipse([x, y, x + 50, y + 100], fill=_A(p["accent"], a))
        d.ellipse([x + 60, y + 70, x + 110, y + 170], fill=_A(p["main"], a))
        for t in range(4):
            _C(d, x + 8 + t * 11, y - 12 + (t % 2) * 4, 7, fill=_A(p["accent"], a))


def c_whistle(d, p, b, r):
    _C(d, 460, 380, 190, fill=_A(p["light"], 245), out=_A(p["accent"]), w=8)
    d.polygon([(560, 260), (1100, 260), (1100, 380), (610, 380)], fill=_A(p["light"], 245), outline=_A(p["accent"]))
    _C(d, 460, 380, 100, fill=_A(p["bg"]))
    d.rectangle([520, 200, 620, 270], fill=_A(p["accent"]))
    d.arc([200, 60, 700, 460], 220, 320, fill=_A(p["main"]), width=8)
    for k in range(3):
        d.arc([1080 + k * 40, 220 - k * 30, 1160 + k * 60, 420 + k * 30], 300, 60, fill=_A(p["main"], 220 - k * 60), width=6)


def c_rhythmhits(d, p, b, r):
    for i in range(4):
        x = 260 + i * 200
        d.rectangle([x, 40, x + 150, 560], fill=_A(p["mid"], 150))
        for k in range(3):
            y = 80 + ((i * 97 + k * 190) % 440)
            _RR(d, x + 14, y, x + 136, y + 44, 14, fill=_A([p["main"], p["accent"], p["light"], p["main"]][i], 240))
    d.rectangle([240, 560, 1050, 580], fill=_A(p["light"]))
    for i in range(4):
        _C(d, 335 + i * 200, 570, 56 if i == 2 else 40, out=_A(p["accent"] if i == 2 else p["light"], 220), w=6)


def c_levelup(d, p, b, r):
    for k in range(4):
        y = 560 - k * 120
        d.line([(440, y), (640, y - 100), (840, y)], fill=_A(lerp(p["main"], p["accent"], k / 3), 255 - k * 40), width=36, joint="curve")
    for _ in range(18):
        _star(d, r.randrange(120, 1160), r.randrange(60, 330), r.randrange(8, 26), _A(p["light"], 230))
    _RR(d, 240, 640, 1040, 680, 16, fill=_A(p["mid"]))
    _RR(d, 240, 640, 880, 680, 16, fill=_A(p["accent"]))


def c_treasure(d, p, b, r):
    for a in range(200, 340, 10):
        d.line([(640, 320), (640 + 600 * math.cos(math.radians(a)), 320 + 600 * math.sin(math.radians(a)))], fill=_A(p["light"], 45), width=14)
    d.rectangle([260, 380, 1020, 650], fill=_A(p["mid"]), outline=_A(p["accent"]), width=8)
    d.polygon([(260, 380), (320, 220), (960, 220), (1020, 380)], fill=_A(p["main"], 245), outline=_A(p["accent"]), width=6)
    d.rectangle([590, 360, 690, 480], fill=_A(p["accent"]))
    for k in range(5):
        d.line([(260 + k * 190, 380), (260 + k * 190, 650)], fill=_A(p["accent"], 160), width=5)
    for x, y in ((520, 340), (640, 300), (760, 340)):
        d.polygon([(x, y - 40), (x + 35, y), (x, y + 40), (x - 35, y)], fill=_A(p["light"]))


def c_quiz(d, p, b, r):
    _C(d, 640, 380, 190, fill=_A(p["accent"]), out=_A(p["light"]), w=10)
    _C(d, 640, 380, 130, fill=_A(p["main"], 255))
    for a in range(0, 360, 18):
        _C(d, 640 + 260 * math.cos(math.radians(a)), 380 + 260 * math.sin(math.radians(a)), 14, fill=_A(p["light"] if (a // 18) % 2 else p["accent"]))
    d.rectangle([590, 330, 690, 350], fill=_A(p["light"]))
    d.rectangle([630, 330, 650, 440], fill=_A(p["light"]))


def c_sleighbells(d, p, b, r):
    d.arc([140, 40, 1140, 520], 20, 160, fill=_A(p["accent"]), width=16)
    for k in range(5):
        a = math.radians(30 + k * 30)
        cx, cy = 640 + 500 * math.cos(a), 280 + 240 * math.sin(a)
        _C(d, cx, cy + 70, 70, fill=_A(p["light"], 245), out=_A(p["accent"]), w=5)
        d.line([(cx - 55, cy + 85), (cx + 55, cy + 85)], fill=_A(p["bg"]), width=5)
        _C(d, cx, cy + 128, 10, fill=_A(p["bg"]))
    for _ in range(14):
        _star(d, r.randrange(60, 1220), r.randrange(500, 700), r.randrange(6, 16), _A(p["light"], 220))


def c_spookyorgan(d, p, b, r):
    _C(d, 640, 400, 250, fill=_A(p["accent"], 245), out=_A(p["main"]), w=8)
    d.polygon([(500, 340), (570, 340), (535, 280)], fill=_A(p["bg"]))
    d.polygon([(710, 340), (780, 340), (745, 280)], fill=_A(p["bg"]))
    d.polygon([(480, 460), (540, 500), (590, 450), (640, 510), (690, 450), (740, 500), (800, 460), (780, 560), (500, 560)], fill=_A(p["bg"]))
    d.rectangle([620, 120, 660, 200], fill=_A(p["mid"]))
    for x, y in ((190, 150), (1090, 190)):
        d.polygon([(x - 90, y), (x - 40, y - 30), (x, y + 10), (x + 40, y - 30), (x + 90, y), (x + 30, y + 40), (x, y + 80), (x - 30, y + 40)], fill=_A(p["mid"]))


def c_fireworks(d, p, b, r):
    for cx, cy, col, rr in ((380, 280, p["main"], 220), (860, 220, p["accent"], 180), (700, 460, p["light"], 130)):
        for a in range(0, 360, 15):
            ra = math.radians(a)
            d.line([(cx + rr * 0.35 * math.cos(ra), cy + rr * 0.35 * math.sin(ra)), (cx + rr * math.cos(ra), cy + rr * math.sin(ra))], fill=_A(col, 220), width=4)
            _C(d, cx + rr * math.cos(ra), cy + rr * math.sin(ra), 6, fill=_A(col))
    d.rectangle([0, 680, 1280, 720], fill=_A(p["bg"]))


def c_book(d, p, b, r):
    d.polygon([(140, 200), (640, 250), (640, 640), (140, 590)], fill=_A(p["light"], 245), outline=_A(p["accent"]), width=5)
    d.polygon([(1140, 200), (640, 250), (640, 640), (1140, 590)], fill=_A(p["light"], 235), outline=_A(p["accent"]), width=5)
    for k in range(5):
        d.line([(180, 270 + k * 60), (600, 310 + k * 60)], fill=_A(p["mid"], 150), width=4)
        d.line([(1100, 270 + k * 60), (680, 310 + k * 60)], fill=_A(p["mid"], 150), width=4)
    d.polygon([(640, 250), (900, 120), (1080, 150), (700, 330)], fill=_A(p["light"], 200), outline=_A(p["accent"]), width=3)
    d.rectangle([120, 590, 1160, 625], fill=_A(p["mid"]))


def c_pencil(d, p, b, r):
    d.polygon([(900, 80), (980, 120), (480, 600), (400, 560)], fill=_A(p["accent"]))
    d.polygon([(400, 560), (480, 600), (330, 680)], fill=_A(p["light"]))
    d.polygon([(330, 680), (362, 640), (350, 670)], fill=_A(p["bg"]))
    d.polygon([(900, 80), (980, 120), (1010, 70), (950, 40)], fill=_A(p["main"]))
    for k in range(4):
        pts = [(160 + x, 640 - k * 40 + 24 * math.sin(x / 40 + k)) for x in range(0, 360, 10)]
        d.line(pts, fill=_A(p["light"], 170), width=3)
    d.arc([140, 150, 420, 430], 200, 20, fill=_A(p["main"], 200), width=3)


def c_cloth(d, p, b, r):
    for k in range(7):
        pts = [(x, 140 + k * 70 + 60 * math.sin(x / 110 + k * 0.7) + 25 * math.sin(x / 41 + k)) for x in range(0, 1290, 10)]
        d.line(pts, fill=_A(lerp(p["mid"], p["light"], k / 6), 230), width=48, joint="curve")
    for k in range(3):
        d.line([(0, 600 + k * 20), (1280, 580 + k * 25)], fill=_A(p["bg"], 120), width=4)


def c_micbrush(d, p, b, r):
    _RR(d, 560, 120, 720, 400, 80, fill=_A(p["light"], 245), out=_A(p["accent"]), w=6)
    for k in range(8):
        d.line([(570, 160 + k * 30), (710, 160 + k * 30)], fill=_A(p["mid"], 150), width=3)
    d.arc([500, 380, 780, 520], 0, 180, fill=_A(p["accent"]), width=8)
    d.line([(640, 520), (640, 650)], fill=_A(p["light"]), width=12)
    d.rectangle([540, 650, 740, 670], fill=_A(p["mid"]))
    for k in range(6):
        d.line([(150, 200 + k * 55), (440, 150 + k * 55)], fill=_A(p["main"], 120), width=14)
        d.line([(840, 150 + k * 55), (1130, 200 + k * 55)], fill=_A(p["main"], 120), width=14)


def c_fingertaps(d, p, b, r):
    for k in range(5):
        _C(d, 640, 380, 80 + k * 70, out=_A(lerp(p["main"], p["accent"], k / 4), 220 - k * 38), w=5)
    for i, (x, y) in enumerate(((450, 330), (560, 260), (680, 250), (790, 300), (870, 390))):
        _RR(d, x - 28, y - 70, x + 28, y + 40, 28, fill=_A(p["light"], 230))


def c_squeaky(d, p, b, r):
    d.ellipse([320, 340, 860, 640], fill=_A(p["accent"], 250), outline=_A(p["light"]), width=6)
    _C(d, 720, 260, 120, fill=_A(p["accent"], 250), out=_A(p["light"]), w=6)
    d.polygon([(820, 260), (980, 290), (820, 330)], fill=_A(p["main"]))
    _C(d, 740, 230, 14, fill=_A(p["bg"]))
    d.arc([380, 400, 700, 600], 190, 320, fill=_A(p["light"], 200), width=8)
    for k in range(3):
        d.arc([930 + k * 40, 150 - k * 30, 1010 + k * 60, 330 + k * 30], 300, 60, fill=_A(p["light"], 220 - k * 60), width=6)


def c_boing(d, p, b, r):
    pts = []
    for i in range(0, 501, 4):
        t = i / 500
        pts.append((640 + (150 - 60 * t) * math.sin(i / 14), 640 - i * 1.0))
    d.line(pts, fill=_A(p["light"]), width=14, joint="curve")
    _C(d, 640, 130, 90, fill=_A(p["accent"]), out=_A(p["light"]), w=6)
    d.rectangle([420, 640, 860, 690], fill=_A(p["mid"]))
    for k in range(3):
        d.arc([260 - k * 50, 70, 420, 330 + k * 20], 120, 240, fill=_A(p["main"], 220 - k * 60), width=6)


def c_kazoo(d, p, b, r):
    d.polygon([(200, 330), (940, 230), (940, 460), (200, 410)], fill=_A(p["main"]), outline=_A(p["light"]), width=6)
    _C(d, 600, 240, 50, fill=_A(p["light"]), out=_A(p["accent"]), w=5)
    d.ellipse([940, 240, 1060, 450], fill=_A(p["accent"]))
    for x, y in ((1020, 120), (1130, 220), (1090, 330)):
        _C(d, x, y, 24, fill=_A(p["light"]))
        d.line([(x + 22, y), (x + 22, y - 90)], fill=_A(p["light"]), width=8)
    for k in range(3):
        d.line([(260 + k * 40, 560), (330 + k * 40, 620 - k * 10)], fill=_A(p["accent"], 200), width=10)


def c_cinewhoosh(d, p, b, r):
    for k in range(9):
        y = 260 + k * 24
        d.polygon([(40 + k * 10, y), (1240 - k * 30, 360 + (k - 4) * 6), (60 + k * 10, y + 14)], fill=_A(lerp(p["main"], p["accent"], k / 8), 130))
    _C(d, 880, 360, 60, fill=_A(p["light"], 240))
    d.line([(100, 360), (1180, 360)], fill=_A(p["light"], 200), width=5)
    d.line([(880, 200), (880, 520)], fill=_A(p["light"], 90), width=3)
    d.rectangle([0, 60, 1280, 110], fill=_A((0, 0, 0), 190))
    d.rectangle([0, 610, 1280, 660], fill=_A((0, 0, 0), 190))


def c_trailerhit(d, p, b, r):
    for k in range(6):
        _C(d, 640, 360, 70 + k * 70, out=_A(lerp(p["accent"], p["bg"], k / 6), 255 - k * 28), w=14 - k * 2)
    for a in range(0, 360, 30):
        ra = math.radians(a + 7)
        d.line([(640 + 60 * math.cos(ra), 360 + 60 * math.sin(ra)), (640 + (420 + (a % 60)) * math.cos(ra), 360 + (330 + (a % 60)) * math.sin(ra))], fill=_A(p["light"], 120), width=4)
    _C(d, 640, 360, 50, fill=_A(p["light"]))


def c_riser(d, p, b, r):
    for i in range(14):
        h = 60 + i * 38
        _RR(d, 120 + i * 78, 660 - h, 120 + i * 78 + 60, 660, 8, fill=_A(lerp(p["mid"], p["main"], i / 13), 245))
    d.polygon([(960, 150), (1100, 150), (1100, 90), (1220, 200), (1100, 310), (1100, 250), (960, 250)], fill=_A(p["light"], 0))
    d.line([(160, 440), (1100, 130)], fill=_A(p["accent"]), width=10)
    d.polygon([(1100, 130), (1040, 130), (1090, 190)], fill=_A(p["accent"]))


def c_subdrop(d, p, b, r):
    for k in range(7):
        y = 120 + k * 80
        rr = 90 - k * 8
        d.line([(640, y), (640, y + 40)], fill=_A(p["light"], 90), width=4)
        d.ellipse([640 - rr * 2, y + 40 - rr * 0.4, 640 + rr * 2, y + 40 + rr * 0.4], outline=_A(lerp(p["accent"], p["main"], k / 6), 255 - k * 18), width=6)
    _C(d, 640, 90, 36, fill=_A(p["light"]))
    pts = [(x, 690 - 40 * math.exp(-abs(x - 640) / 250) * math.sin(x / 12)) for x in range(80, 1200, 6)]
    d.line(pts, fill=_A(p["main"]), width=8)


def c_orchestral(d, p, b, r):
    for k in range(7):
        d.arc([120 + k * 30, 80 + k * 38, 1160 - k * 30, 900 - k * 20], 195, 345, fill=_A(lerp(p["main"], p["accent"], k / 6), 240), width=10)
    d.line([(980, 250), (1190, 90)], fill=_A(p["light"]), width=10)
    _C(d, 970, 258, 16, fill=_A(p["light"]))
    for _ in range(18):
        _star(d, r.randrange(120, 1160), r.randrange(480, 680), r.randrange(6, 16), _A(p["light"], 190))


def c_cleartones(d, p, b, r):
    for k, (rr, w) in enumerate(((60, 30), (130, 24), (200, 18), (270, 12))):
        _C(d, 640, 360, rr, out=_A(p["light"] if k % 2 == 0 else p["accent"]), w=w)
    d.line([(100, 360), (1180, 360)], fill=_A(p["light"], 160), width=6)


def c_softlows(d, p, b, r):
    for k in range(5):
        pts = [(x, 520 - 100 * math.sin(x / (240 + k * 40) + k) - k * 40) for x in range(0, 1290, 10)]
        d.polygon(pts + [(1280, 720), (0, 720)], fill=_A(lerp(p["main"], p["mid"], k / 4), 120))
    _C(d, 640, 200, 100, fill=_A(p["light"], 200))


def c_pulsecode(d, p, b, r):
    for j, n in enumerate((1, 2, 3)):
        for i in range(n):
            _C(d, 300 + i * 150, 180 + j * 150, 46, fill=_A([p["light"], p["accent"], p["main"]][j]))
    d.rounded_rectangle([760, 310, 1130, 400], radius=44, fill=_A(p["light"]))
    for j in range(3):
        d.line([(700, 180 + j * 150), (1150, 180 + j * 150)], fill=_A(p["mid"], 100), width=3)


def c_pitchladder(d, p, b, r):
    d.line([(420, 60), (420, 680)], fill=_A(p["light"]), width=16)
    d.line([(860, 60), (860, 680)], fill=_A(p["light"]), width=16)
    for k in range(8):
        d.line([(420, 110 + k * 78), (860, 110 + k * 78)], fill=_A(lerp(p["accent"], p["main"], k / 7)), width=14)
    d.polygon([(1030, 140), (1130, 260), (1060, 260), (1060, 360), (1000, 360), (1000, 260), (930, 260)], fill=_A(p["accent"]))
    d.polygon([(250, 580), (150, 460), (220, 460), (220, 360), (280, 360), (280, 460), (350, 460)], fill=_A(p["main"]))


def c_stone(d, p, b, r):
    for cx, cy, w, h in ((640, 600, 400, 110), (640, 500, 300, 100), (630, 410, 220, 90), (645, 330, 150, 76), (640, 262, 90, 58)):
        d.ellipse([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], fill=_A(lerp(p["mid"], p["light"], (cy % 7) / 7), 255), outline=_A(p["accent"], 200), width=4)
    for k in range(3):
        d.arc([300 - k * 40, 240 - k * 30, 500, 560 + k * 30], 120, 240, fill=_A(p["main"], 200 - k * 60), width=5)


def c_marble(d, p, b, r):
    for cx, cy, rr, c in ((420, 380, 190, p["main"]), (820, 440, 140, p["accent"]), (900, 220, 90, p["light"])):
        _C(d, cx, cy, rr, fill=_A(c, 235), out=_A(p["light"], 200), w=4)
        for k in range(5):
            d.arc([cx - rr * 0.8, cy - rr * 0.8 + k * rr * 0.1, cx + rr * 0.8, cy + rr * 0.8], 30 + k * 40, 150 + k * 40, fill=_A(p["bg"], 140), width=6)
        d.ellipse([cx - rr * 0.55, cy - rr * 0.6, cx - rr * 0.15, cy - rr * 0.3], fill=_A((255, 255, 255), 190))


def c_rubberbounce(d, p, b, r):
    for k in range(5):
        x = 140 + k * 220
        y = 600 - 420 * math.exp(-k * 0.45) * 1.0
        _C(d, x, y, 40, fill=_A(p["accent"], 90 + k * 30))
    d.ellipse([940, 540, 1180, 620], fill=_A(p["main"], 245))
    d.ellipse([920, 650, 1200, 700], fill=_A(p["bg"], 200))
    pts = [(140 + t * 14, 600 - 420 * abs(math.sin(t * math.pi / 55)) * math.exp(-t / 110)) for t in range(0, 900, 6)]
    d.line(pts, fill=_A(p["light"], 140), width=4)


def c_gel(d, p, b, r):
    d.polygon([(260, 600), (320, 340), (480, 200), (700, 170), (900, 260), (1010, 450), (1000, 600)], fill=_A(p["main"], 200), outline=_A(p["light"], 220), width=6)
    d.ellipse([380, 250, 560, 340], fill=_A((255, 255, 255), 150))
    for x, y, rr in ((300, 640, 40), (1060, 620, 30), (700, 660, 26)):
        _C(d, x, y, rr, fill=_A(p["main"], 190))
    d.ellipse([250, 600, 1010, 680], fill=_A(p["main"], 120))


def c_gems(d, p, b, r):
    cols = [p["main"], p["accent"], p["light"]]
    for i, (cx, cy, s) in enumerate(((420, 360, 190), (800, 280, 140), (880, 520, 110), (230, 560, 90))):
        c = cols[i % 3]
        pts = [(cx - s, cy - s * 0.3), (cx - s * 0.5, cy - s * 0.8), (cx + s * 0.5, cy - s * 0.8), (cx + s, cy - s * 0.3), (cx, cy + s)]
        d.polygon(pts, fill=_A(c, 235), outline=_A(p["light"]), width=4)
        d.line([(cx - s, cy - s * 0.3), (cx + s, cy - s * 0.3)], fill=_A(p["bg"], 140), width=3)
        d.line([(cx - s * 0.5, cy - s * 0.8), (cx - s * 0.25, cy - s * 0.3), (cx, cy + s), (cx + s * 0.25, cy - s * 0.3), (cx + s * 0.5, cy - s * 0.8)], fill=_A(p["bg"], 120), width=3)
    for _ in range(14):
        _star(d, r.randrange(60, 1220), r.randrange(40, 680), r.randrange(6, 16), _A(p["light"], 220))


def c_sand(d, p, b, r):
    for k in range(5):
        pts = [(x, 330 + k * 70 + 90 * math.sin(x / 190 + k * 1.3)) for x in range(0, 1290, 10)]
        d.polygon(pts + [(1280, 720), (0, 720)], fill=_A(lerp(p["light"], p["mid"], k / 4), 220))
    for _ in range(120):
        _C(d, r.randrange(30, 1250), r.randrange(30, 330), r.randrange(1, 4), fill=_A(p["light"], 190))
    _C(d, 1050, 130, 70, fill=_A(p["accent"], 200))


def c_leather(d, p, b, r):
    _RR(d, 200, 160, 1080, 600, 36, fill=_A(p["mid"]), out=_A(p["light"]), w=5)
    _RR(d, 240, 200, 1040, 560, 20, out=_A(p["accent"], 220), w=3)
    for k in range(34):
        t = k / 33
        d.line([(250 + t * 780, 200), (262 + t * 780, 212)], fill=_A(p["accent"]), width=5)
        d.line([(250 + t * 780, 548), (262 + t * 780, 560)], fill=_A(p["accent"]), width=5)
    _RR(d, 560, 280, 720, 460, 12, out=_A(p["light"]), w=8)
    d.rectangle([615, 340, 665, 420], fill=_A(p["light"]))
    d.line([(100, 380), (200, 380)], fill=_A(p["main"], 200), width=26)


def c_cork(d, p, b, r):
    d.polygon([(500, 420), (780, 420), (780, 700), (500, 700)], fill=_A(p["mid"]), outline=_A(p["light"]), width=5)
    d.polygon([(560, 220), (720, 220), (720, 420), (560, 420)], fill=_A(p["mid"]), outline=_A(p["light"]), width=5)
    d.rectangle([555, 130, 725, 230], fill=_A(p["accent"]))
    for _ in range(18):
        _C(d, r.randrange(565, 715), r.randrange(140, 220), 4, fill=_A(p["bg"], 180))
    d.rectangle([520, 520, 760, 680], fill=_A(p["main"], 140))
    for k, (x, y) in enumerate(((880, 160), (960, 90), (1040, 200), (1100, 110))):
        _C(d, x, y, 14 + k * 3, out=_A(p["light"], 220), w=4)
    for a in range(200, 340, 20):
        d.line([(640 + 110 * math.cos(math.radians(a)), 170 + 110 * math.sin(math.radians(a))), (640 + 190 * math.cos(math.radians(a)), 170 + 190 * math.sin(math.radians(a)))], fill=_A(p["light"], 150), width=5)


def c_porcelain(d, p, b, r):
    d.polygon([(380, 340), (900, 340), (860, 560), (420, 560)], fill=_A(p["light"], 250), outline=_A(p["accent"]), width=6)
    d.arc([840, 360, 1010, 520], 270, 90, fill=_A(p["light"]), width=24)
    d.ellipse([280, 540, 1000, 640], fill=_A(p["light"], 250), outline=_A(p["accent"]), width=6)
    for k in range(5):
        d.line([(430 + k * 90, 380), (430 + k * 90, 520)], fill=_A(p["accent"], 130), width=3)
    for x in (520, 640, 760):
        pts = [(x + 28 * math.sin(i / 18), 300 - i) for i in range(0, 170, 4)]
        d.line(pts, fill=_A(p["light"], 130), width=8, joint="curve")


def c_chess(d, p, b, r):
    for i in range(8):
        for j in range(5):
            if (i + j) % 2 == 0:
                d.rectangle([140 + i * 125, 440 + j * 56 - 20, 140 + i * 125 + 125, 440 + j * 56 + 36], fill=_A(p["light"], 230))
            else:
                d.rectangle([140 + i * 125, 440 + j * 56 - 20, 140 + i * 125 + 125, 440 + j * 56 + 36], fill=_A(p["mid"], 255))
    _C(d, 640, 130, 60, fill=_A(p["accent"]), out=_A(p["light"]), w=5)
    d.polygon([(590, 190), (690, 190), (730, 420), (550, 420)], fill=_A(p["accent"]), outline=_A(p["light"]), width=5)
    d.ellipse([520, 400, 760, 470], fill=_A(p["accent"]), outline=_A(p["light"]), width=5)
    d.rectangle([600, 100, 680, 110], fill=_A(p["light"], 0))


def c_domino(d, p, b, r):
    for i in range(8):
        a = i * 7
        x = 150 + i * 135
        pts = [(0, 0), (70, 0), (70, 330), (0, 330)]
        ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
        tp = [(x + px * ca - (py - 330) * sa, 640 + px * sa + (py - 330) * ca) for px, py in pts]
        d.polygon(tp, fill=_A(p["light"], 245), outline=_A(p["accent"]))
        for k, (px, py) in enumerate(((35, 80), (35, 165), (35, 250))):
            _C(d, x + px * ca - (py - 330) * sa, 640 + px * sa + (py - 330) * ca, 11, fill=_A(p["bg"]))


W2_COVER = {
    "snd-sitar": (c_sitar, "rad"), "snd-tabla": (c_tabla, "dots"), "snd-oud": (c_oud, "v"), "snd-erhu": (c_erhu, "flat"),
    "snd-balafon": (c_balafon, "stripes"), "snd-didgeridoo": (c_didgeridoo, "rad"), "snd-bagpipe": (c_bagpipe, "flat"),
    "snd-accordion": (c_accordion, "stripes"), "snd-harmonica": (c_harmonica, "grid"), "snd-ukulele": (c_ukulele, "v"),
    "snd-banjo": (c_banjo, "dots"), "snd-taiko": (c_taiko, "rad"), "snd-cajon": (c_cajon, "flat"), "snd-latin-percussion": (c_latin, "stripes"),
    "snd-pan-flute": (c_panflute, "v"), "snd-celesta": (c_celesta, "rad"), "snd-harpsichord": (c_harpsichord, "v"),
    "snd-pipe-organ": (c_pipeorgan, "flat"), "snd-church-bells": (c_churchbells, "rad"), "snd-house-stabs": (c_house, "grid"),
    "snd-techno-blips": (c_techno, "grid"), "snd-dubstep-wobble": (c_dubstep, "v"), "snd-dnb-breaks": (c_dnb, "grid"),
    "snd-trance-plucks": (c_trance, "rad"), "snd-idm-glitch": (c_idm, "dots"), "snd-acid-squelch": (c_acid, "flat"),
    "snd-future-bass": (c_futurebass, "rad"), "snd-ambient-pads": (c_ambientpads, "v"), "snd-hyperpop": (c_hyperpop, "dots"),
    "snd-trap-hats": (c_traphats, "flat"), "snd-drum-box": (c_drumbox, "grid"), "snd-floppy-seek": (c_floppy, "grid"),
    "snd-crt-degauss": (c_crt, "flat"), "snd-dot-matrix": (c_dotmatrix, "stripes"), "snd-pinball": (c_pinball, "dots"),
    "snd-slot-reels": (c_slotreels, "rad"), "snd-jukebox": (c_jukebox, "rad"), "snd-telegraph-morse": (c_telegraph, "flat"),
    "snd-cash-register": (c_cashregister, "stripes"), "snd-shortwave-radio": (c_shortwave, "grid"), "snd-rpg-menu": (c_rpg, "v"),
    "snd-fighter-stinger": (c_fighter, "stripes"), "snd-racing-gears": (c_racing, "stripes"), "snd-platform-hop": (c_platform, "dots"),
    "snd-puzzle-match": (c_puzzle, "flat"), "snd-horror-stinger": (c_horror, "rad"), "snd-starfighter": (c_starfighter, "flat"),
    "snd-shadow-steps": (c_shadow, "v"), "snd-sports-whistle": (c_whistle, "rad"), "snd-rhythm-hits": (c_rhythmhits, "grid"),
    "snd-level-up": (c_levelup, "rad"), "snd-treasure-chest": (c_treasure, "v"), "snd-quiz-show": (c_quiz, "dots"),
    "snd-sleigh-bells": (c_sleighbells, "v"), "snd-spooky-organ": (c_spookyorgan, "rad"), "snd-fireworks": (c_fireworks, "flat"),
    "snd-book-pages": (c_book, "flat"), "snd-pencil-sketch": (c_pencil, "dots"), "snd-cloth-rustle": (c_cloth, "v"),
    "snd-mic-brush": (c_micbrush, "rad"), "snd-finger-taps": (c_fingertaps, "rad"), "snd-squeaky-toys": (c_squeaky, "dots"),
    "snd-boing-springs": (c_boing, "v"), "snd-kazoo-fun": (c_kazoo, "stripes"), "snd-cine-whoosh": (c_cinewhoosh, "flat"),
    "snd-trailer-hit": (c_trailerhit, "rad"), "snd-tension-riser": (c_riser, "grid"), "snd-sub-drop": (c_subdrop, "v"),
    "snd-orchestral-swell": (c_orchestral, "rad"), "snd-clear-tones": (c_cleartones, "flat"), "snd-soft-lows": (c_softlows, "v"),
    "snd-pulse-code": (c_pulsecode, "grid"), "snd-pitch-ladder": (c_pitchladder, "flat"), "snd-stone-tap": (c_stone, "flat"),
    "snd-marble-clack": (c_marble, "rad"), "snd-rubber-bounce": (c_rubberbounce, "v"), "snd-gel-squish": (c_gel, "rad"),
    "snd-crystal-gems": (c_gems, "rad"), "snd-sand-shift": (c_sand, "v"), "snd-leather-creak": (c_leather, "flat"),
    "snd-cork-pop": (c_cork, "dots"), "snd-porcelain-clink": (c_porcelain, "v"), "snd-chess-pieces": (c_chess, "flat"),
    "snd-domino-clack": (c_domino, "stripes"),
}
NEW_COVER.update(W2_COVER)



class _SafeDraw:
    """Normalises box corners so motif code never trips PIL's x1>=x0 / y1>=y0 checks."""

    def __init__(self, d):
        self._d = d

    def __getattr__(self, name):
        fn = getattr(self._d, name)
        if name in ("rectangle", "rounded_rectangle", "ellipse", "pieslice", "arc"):
            def wrapped(box, *a, **k):
                x0, y0, x1, y1 = box
                return fn([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], *a, **k)
            return wrapped
        return fn


def new_cover(pid, pal, sounds, rnd):
    fn, style = NEW_COVER[pid]
    img = base_img(pal, pal["bg"] if style in ("flat", "grid") else None)
    if style == "rad":
        glow(img, 640, 360, 420, pal["mid"], 200)
    elif style == "dots":
        overlay(img, lambda d: [_C(d, x, y, 4, fill=_A(pal["light"], 40)) for x in range(20, CW, 50) for y in range(20, CH_, 50)])
    elif style == "stripes":
        overlay(img, lambda d: [d.polygon([(x, 0), (x + 40, 0), (x - 280 + 40, CH_), (x - 280, CH_)], fill=_A(pal["mid"], 70)) for x in range(0, CW + 400, 120)])
    elif style == "grid":
        overlay(img, lambda d: ([d.line([(x, 0), (x, CH_)], fill=_A(pal["main"], 26), width=1) for x in range(0, CW, 40)] +
                                [d.line([(0, y), (CW, y)], fill=_A(pal["main"], 26), width=1) for y in range(0, CH_, 40)]))
    bars = envelope_bars(sounds, 32)
    overlay(img, lambda d: fn(_SafeDraw(d), pal, bars, rnd))
    return img


def make_cover(pal, sounds, rnd, pid):
    img = (new_cover(pid, pal, sounds, rnd) if pid in NEW_COVER else COVERS[STYLE_OF[pid]](pal, sounds, rnd)).convert("RGB")
    return img.resize((640, 360), Image.LANCZOS)


def draw_icon(d, kind, cx, cy, s, col):
    w = max(3, s // 9)
    if kind == "click":
        d.ellipse([cx - s * .15, cy - s * .15, cx + s * .15, cy + s * .15], fill=col)
        d.ellipse([cx - s * .38, cy - s * .38, cx + s * .38, cy + s * .38], outline=col, width=w)
    elif kind == "open":
        for k in (-0.18, 0.12):
            d.line([(cx - s * .3, cy + s * (k + .16)), (cx, cy + s * (k - .12)), (cx + s * .3, cy + s * (k + .16))],
                   fill=col, width=w, joint="curve")
    elif kind == "close":
        for k in (-0.12, 0.18):
            d.line([(cx - s * .3, cy + s * (k - .16)), (cx, cy + s * (k + .12)), (cx + s * .3, cy + s * (k - .16))],
                   fill=col, width=w, joint="curve")
    else:
        d.line([(cx - s * .32, cy + s * .02), (cx - s * .08, cy + s * .26), (cx + s * .34, cy - s * .24)], fill=col,
               width=w + 2, joint="curve")


def make_waveform(pal, four):
    W, H = 1280, 720
    SS = 1
    img = vgrad(W, H, pal["bg"], lerp(pal["bg"], pal["mid"], 0.7))
    d = ImageDraw.Draw(img, "RGBA")
    rowh = 150
    top = 50
    kinds = ["click", "open", "close", "select"]
    for r, (kind, s) in enumerate(zip(kinds, four)):
        y0 = top + r * (rowh + 10)
        cy = y0 + rowh // 2
        d.rounded_rectangle([30, y0, W - 30, y0 + rowh], radius=22, fill=(255, 255, 255, 14),
                            outline=pal["main"] + (60,), width=2)
        d.rounded_rectangle([50, cy - 50, 150, cy + 50], radius=22, fill=pal["mid"] + (255,),
                            outline=pal["accent"] + (200,), width=3)
        draw_icon(d, kind, 100, cy, 76, pal["light"])
        x0, x1 = 190, W - 60
        for gx in range(5):
            xx = x0 + (x1 - x0) * gx / 4
            d.line([(xx, y0 + 12), (xx, y0 + rowh - 12)], fill=pal["light"] + (28,), width=1)
        d.line([(x0, cy), (x1, cy)], fill=pal["light"] + (60,), width=1)
        width = x1 - x0
        n = len(s)
        span = max(n, int(0.3 * SR))  # very short sounds stay readable but keep their relative length
        cols = width
        for cx in range(cols):
            a = int(cx * span / cols)
            b = max(a + 1, int((cx + 1) * span / cols))
            if a >= n:
                break
            seg = s[a:b]
            hi = max(seg)
            lo = min(seg)
            t = cx / cols
            col = lerp(pal["main"], pal["accent"], t)
            d.line([(x0 + cx, cy - hi * 58 - 1), (x0 + cx, cy - lo * 58 + 1)], fill=col + (255,), width=1)
    return img


def save_webp(img, path):
    img.save(path, "WEBP", quality=88, method=6)


# ----------------------------------------------------------------------------- build
def zip_pack(pid, files, sounds_json):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        def add(name, data):
            zi = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zi.create_system = 3
            z.writestr(zi, data, compresslevel=9)

        add("sounds.json", sounds_json)
        for name in sorted(files):
            add(name, files[name])
    return buf.getvalue()


def write_if_changed(path, data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    if path.exists() and path.read_bytes() == data:
        return
    path.write_bytes(data)



# ----------------------------------------------------------------------------- wave 2: music + achievement for every pack
CH.update({"maj": [0, 4, 7, 12], "min": [0, 3, 7, 12], "dim7": [0, 3, 6, 9], "sus4": [0, 5, 7, 10], "7": [0, 4, 7, 10],
           "m11": [0, 3, 7, 10, 14, 17], "phr": [0, 1, 7, 12], "m6": [0, 3, 7, 9], "5": [0, 7, 12, 19], "lyd": [0, 4, 7, 11, 18]})
PROGS = {
    "A": PROG_A, "B": PROG_B, "C": PROG_C, "D": PROG_D,
    "E": [(45, "m7"), (41, "maj7"), (43, "7"), (45, "m11"), (40, "m7"), (38, "m6"), (41, "maj7"), (40, "7")],
    "F": [(48, "maj"), (43, "maj"), (45, "m7"), (41, "maj7"), (48, "6"), (43, "sus4"), (45, "m7"), (41, "add9")],
    "G": [(38, "phr"), (39, "maj"), (38, "m7"), (36, "maj7"), (38, "phr"), (41, "m7"), (39, "maj7"), (38, "5")],
    "H": [(36, "5"), (32, "maj7"), (34, "sus2"), (31, "m7"), (36, "m9"), (32, "lyd"), (34, "maj7"), (31, "sus2")],
    "I": [(38, "dim7"), (37, "m7"), (38, "phr"), (34, "dim7"), (38, "m6"), (35, "dim7"), (37, "phr"), (38, "5")],
    "J": [(41, "lyd"), (43, "6"), (41, "maj7"), (45, "m9"), (41, "lyd"), (40, "m7"), (43, "add9"), (38, "sus2")],
    "K": [(45, "m7"), (45, "m7"), (41, "maj7"), (43, "7"), (45, "m7"), (40, "m7"), (41, "maj7"), (43, "7")],
    "L": [(48, "add9"), (43, "6"), (45, "m7"), (41, "maj7"), (48, "maj7"), (40, "m7"), (41, "add9"), (43, "sus4")],
}


def _flute(f, d):
    return layer([(0, tone(f, d, a=0.06, tau=d * 0.7, vib=0.004), 0.8), (0, bp(noise(d, a=0.04, tau=d * 0.6), f, 14.0), 2.0)])


VOICES = {
    "bell": chime_v, "mallet": marimba_v, "kal": kalimba_v, "felt": felt_v, "rhodes": rhodes_v, "pluck": ks_v,
    "harp": lambda f, d: lp(ks(f, d, tau=1.0, bright=0.4), 5500),
    "saw": lambda f, d: saw_pluck(f, d, 3500),
    "fm": lambda f, d: tone(f, d, a=0.001, tau=0.55, fm=(3.5, 3.0, 0.25)),
    "organ": lambda f, d: additive(f, d, [(1, 1, 2.5), (2, 0.5, 2.0), (3, 0.3, 1.5)], a=0.08, rel=0.2),
    "flute": _flute,
    "chip": lambda f, d: tone(f, d, "sq", duty=0.25, a=0.002, tau=0.18),
    "glass": lambda f, d: additive(f, d, [(1, 1, 0.5), (2.76, 0.4, 0.3), (5.4, 0.15, 0.15)], a=0.003),
    "str": lambda f, d: lp(tone(f, d, "saw", a=0.4, tau=d * 0.5, vib=0.004), 2200),
    "toy": toy_v,
    "vibes": lambda f, d: trem(additive(f, d, [(1, 1.0, 0.9), (4, 0.35, 0.4), (10, 0.12, 0.15)], a=0.003), 5.5, 0.45),
    "pad": lambda f, d: lp(tone(f, d, "saw", a=0.5, tau=d * 0.5), 1500),
    "drone": lambda f, d: additive(f, d, [(1, 1, 1.5), (2.01, 0.4, 1.0), (3.0, 0.2, 0.7)], a=0.05),
}
_ST = dict(octs=(24, 36), step=0.5, rest=0.25, pattern="rand", arp=0.1, pad=0.07, reverb=0.5)
STYLES = {
    "bell": dict(_ST, padkind="glass", arp_d=1.8, reverb=0.6), "mallet": dict(_ST, padkind="warm", arp_d=1.2, step=0.375, rest=0.3, arp=0.15),
    "kal": dict(_ST, padkind="soft", arp_d=1.4, arp=0.14, pad=0.08), "felt": dict(_ST, padkind="warm", arp_d=1.6, octs=(12, 24, 36), step=1.0, rest=0.3, arp=0.12, pad=0.09, lp=3000),
    "rhodes": dict(_ST, padkind="warm", arp_d=1.2, arp=0.16, pad=0.06, lp=3500), "pluck": dict(_ST, padkind="warm", arp_d=1.2, rest=0.2, pattern="updown", arp=0.18),
    "harp": dict(_ST, padkind="glass", arp_d=1.5, rest=0.2, pattern="updown", arp=0.15, pad=0.06, reverb=0.6),
    "saw": dict(_ST, padkind="saw", arp_d=0.9, step=0.25, pattern="updown", pad=0.04, lp=3500),
    "fm": dict(_ST, padkind="glass", arp_d=1.6, step=0.75, rest=0.3, reverb=0.6),
    "organ": dict(_ST, padkind="soft", arp_d=2.0, octs=(12, 24), step=1.0, rest=0.4, arp=0.08, pad=0.09),
    "flute": dict(_ST, padkind="soft", arp_d=1.8, step=0.75, rest=0.35, pad=0.08),
    "chip": dict(_ST, padkind="soft", arp_d=0.4, step=0.25, rest=0.2, pattern="updown", arp=0.08, pad=0.05, lp=4500),
    "glass": dict(_ST, padkind="glass", arp_d=1.6, step=0.5, rest=0.3, reverb=0.7), "str": dict(_ST, padkind="saw", arp_d=3.0, octs=(12, 24), step=1.5, rest=0.3, arp=0.08, pad=0.09, lp=2500),
    "toy": dict(_ST, padkind="soft", arp_d=1.0, step=0.25, rest=0.2, pattern="updown", arp=0.12, pad=0.05),
    "vibes": dict(_ST, padkind="warm", arp_d=1.6, step=0.75, rest=0.3, pad=0.07, reverb=0.6),
    "pad": dict(_ST, padkind="soft", arp_d=2.4, octs=(12, 24), step=2.0, rest=0.5, arp=0.05, pad=0.12, reverb=0.7, lp=2500),
    "drone": dict(_ST, padkind="soft", arp_d=2.4, octs=(12, 24), step=2.0, rest=0.4, arp=0.1, pad=0.1, reverb=0.7, lp=2200),
}
PAT = {"four": "x...x...x...x...", "off": "..x...x...x...x.", "hat8": "x.x.x.x.x.x.x.x.", "sync": "x..x..x...x.x...",
       "backbeat": "....x.......x...", "sparse": "x......."}
# suffix style prog shift [secs=N] [bpm=N] [ach=mode] [bed=sound@pattern@gain,...]  (sounds: click open close sel kick snare hat clap)
PLAN_TEXT = """
soft-pop felt F 0
glass-tap bell J 3
wood-block mallet A 0 bed=click@sparse@0.25
water-drop kal J -2
mech-keys rhodes K 0 bpm=96 bed=click@sync@0.14
typewriter pluck A 0 bpm=92 bed=click@sync@0.12
paper-flick felt F 4 bed=click@sparse@0.15
chip-blips chip F 0 bpm=120 bed=kick@four@0.25,hat@off@0.12
arcade-coin chip L 2 bpm=128 bed=kick@four@0.25,hat@hat8@0.1
laser-ui saw I 0 bpm=110 bed=click@sync@0.12
deep-bubble pad J -3
toy-piano toy L 0
minimal-tick kal A 0 bed=click@sparse@0.15
analog-click saw K 0 bpm=100 bed=click@sync@0.14
handheld-beep chip L -2 bpm=132 bed=kick@four@0.2
cyber-glitch saw I 3 bpm=124 bed=click@sync@0.15,kick@four@0.25
pulse-quest chip L 5 bpm=116 bed=hat@off@0.1
chip-drums chip K 0 bpm=118 bed=kick@four@0.3,snare@backbeat@0.2,hat@hat8@0.1
pc-speaker chip A 0 bpm=108
wavetable-pocket chip F 3 bpm=124 bed=kick@four@0.2
coin-slot chip L 0 bpm=112 bed=click@sync@0.12
sixteen-bit saw L 2 bpm=108 bed=kick@four@0.2
pwm-tracker chip K 3 bpm=134 bed=kick@four@0.22,hat@off@0.1
vibraphone vibes J 0
glockenspiel bell L 5
drawbar-organ organ K 0
nylon-guitar pluck F 0
steel-string pluck L -2
koto harp G 4
shamisen pluck G 7
pizzicato pluck L 0
steel-drum mallet L 3 bpm=104 bed=click@sync@0.15
gamelan bell G 0
xylophone mallet L 4
tubular-bells bell H 0
wind-chimes bell J 5
bamboo flute J 0 bed=click@sparse@0.25
hand-drum mallet G 0 bpm=96 bed=click@sync@0.3,open@sparse@0.1
blue-switch rhodes K 0 bpm=104 bed=click@hat8@0.15
thock-board felt K -2 bpm=84 bed=click@sync@0.2
camera-shutter pluck F 0 bed=click@sparse@0.2
rotary-dial pluck K 0 bed=click@sparse@0.18
cassette-deck rhodes A -3 bed=click@sparse@0.18
toggle-panel saw K 0 bpm=100 bed=click@sync@0.16
office-desk mallet F 0 bed=click@sparse@0.18
clockwork bell A 0 bpm=100 bed=click@hat8@0.15
metal-clank drone I 0 bed=click@sparse@0.2
blade-ring drone G 0 bed=click@sparse@0.12
abacus-beads mallet F 0 bed=click@sparse@0.2
cardboard felt A 2 bed=click@sparse@0.2
bubble-wrap kal L 0 bed=click@sparse@0.2
ice-cubes bell J 0 bed=click@sparse@0.12
rain-window felt J -2
jazz-brushes rhodes A -2 bpm=88 bed=click@sync@0.1
songbird flute F 2
frog-pond mallet J 0 bed=click@sparse@0.2
crickets kal J 0 bed=click@sparse@0.12
night-owl pad C 0
tide-swell pad J 0
hud-blips saw I 0 bpm=112 bed=click@sync@0.15
sonar-ping drone H 0 bed=click@sparse@0.15
geiger-counter drone I 0 bed=click@sparse@0.3
analog-sweeps saw C 0
hologram glass J 3
dialup-tones saw I 0 bpm=100 bed=click@sync@0.14
robot-voice saw K 0 bpm=108 bed=click@sync@0.14
theremin flute I 0
warp-drive pad H 0
vinyl-pop felt A 0 bed=click@sparse@0.12
haptic-tick kal A 0 bed=click@sparse@0.2
haptic-thud felt A -3 bed=click@sparse@0.25
haptic-buzz saw K 0 bpm=96 bed=click@sync@0.2
sitar harp G 0 secs=44
tabla mallet G 2 bpm=92 bed=click@sync@0.2,open@sparse@0.08
oud pluck G -2
erhu str E 2 ach=minor
balafon mallet L 0 bpm=108 bed=click@sync@0.15
didgeridoo drone H -5 bed=click@sparse@0.15
bagpipe organ L 0 ach=fan
accordion organ F 0 bpm=100
harmonica flute K 0 bpm=88
ukulele pluck L 5 bpm=104 bed=click@off@0.12
banjo pluck F -2 bpm=124 bed=click@sync@0.15
taiko drone G -5 bpm=84 bed=click@sync@0.3,open@sparse@0.1
cajon mallet K 0 bpm=96 bed=click@sync@0.25,open@sparse@0.1
latin-percussion mallet L 0 bpm=108 bed=click@sync@0.2,sel@sparse@0.1
pan-flute flute J 0
celesta glass J 5
harpsichord pluck E 0 bpm=96
pipe-organ organ H 0 secs=48
church-bells bell H -3 secs=48
house-stabs saw K 0 bpm=124 bed=kick@four@0.3,clap@backbeat@0.2,hat@off@0.12
techno-blips saw I -2 bpm=130 bed=kick@four@0.35,hat@off@0.12,click@sync@0.12
dubstep-wobble saw I 0 bpm=140 bed=kick@sync@0.3,snare@backbeat@0.22
dnb-breaks saw E 0 bpm=172 bed=kick@sync@0.28,snare@backbeat@0.22,hat@hat8@0.1
trance-plucks saw L 0 bpm=136 bed=kick@four@0.3,hat@off@0.12
idm-glitch saw I 2 bpm=110 bed=click@sync@0.2,hat@sparse@0.12
acid-squelch saw K -2 bpm=126 bed=kick@four@0.3,hat@off@0.12
future-bass str L 0 bpm=150 bed=snare@backbeat@0.22,hat@hat8@0.08
ambient-pads pad J 0
hyperpop chip L 4 bpm=160 bed=kick@sync@0.3,clap@backbeat@0.2,hat@hat8@0.1
trap-hats saw E 0 bpm=140 bed=hat@hat8@0.12,snare@backbeat@0.2,kick@sync@0.3
drum-box rhodes K 0 bpm=104 bed=kick@four@0.3,snare@backbeat@0.22,hat@hat8@0.1
floppy-seek saw I 0 bpm=96 bed=click@sync@0.18
crt-degauss drone H 0 bed=click@sparse@0.15
dot-matrix saw K 0 bpm=108 bed=click@hat8@0.12,open@sparse@0.06
pinball chip K 0 bpm=130 bed=click@sync@0.15,sel@sparse@0.1
slot-reels chip L 0 bpm=118 bed=click@hat8@0.12
jukebox rhodes A 0 bpm=92 bed=click@sync@0.1
telegraph-morse pluck E 0 bed=click@sparse@0.25
cash-register mallet L 0 bpm=100 bed=click@sync@0.15
shortwave-radio drone I 0 bed=click@sparse@0.15
rpg-menu harp J 0 secs=44
fighter-stinger saw I 0 bpm=128 bed=kick@sync@0.3,click@backbeat@0.2
racing-gears saw K 2 bpm=140 bed=kick@four@0.3,snare@backbeat@0.2,hat@off@0.1
platform-hop chip L 0 bpm=124 bed=kick@four@0.2,hat@off@0.1
puzzle-match kal L 3
horror-stinger drone I 0 ach=minor
starfighter saw E 0 bpm=136 bed=kick@four@0.3,hat@off@0.12
shadow-steps drone H 0 bed=click@sparse@0.2
sports-whistle chip F 0 bpm=120 bed=kick@four@0.25,snare@backbeat@0.2
rhythm-hits chip K 0 bpm=128 bed=click@off@0.18,kick@four@0.25
level-up chip L 0 bpm=120 ach=fan
treasure-chest harp G 0 secs=44
quiz-show chip F 2 bpm=112 bed=click@hat8@0.12,kick@four@0.2
sleigh-bells bell L 0 bpm=100 bed=click@hat8@0.14
spooky-organ organ I 0 ach=minor
fireworks glass L 0 bed=click@sparse@0.2
book-pages felt A 0 bed=click@sparse@0.15
pencil-sketch kal F 0 bed=click@sparse@0.15
cloth-rustle pad J 0 bed=click@sparse@0.1
mic-brush felt A -3 bed=click@sparse@0.15
finger-taps kal J 0 bed=click@sparse@0.22
squeaky-toys toy L 2 bpm=116 bed=click@sync@0.14
boing-springs toy F 0 bpm=108 bed=click@sync@0.14
kazoo-fun toy L 0 bpm=120 bed=click@off@0.12,kick@four@0.2
cine-whoosh str H 0 ach=minor
trailer-hit str I -2 ach=minor
tension-riser str I 0 ach=minor
sub-drop pad H -5 ach=minor
orchestral-swell str H 0 secs=48
clear-tones bell L 0
soft-lows pad F 0
pulse-code kal A 0 bpm=100 bed=click@sync@0.15
pitch-ladder mallet L 0
stone-tap mallet G 0 bed=click@sparse@0.25
marble-clack glass J 0 bed=click@sparse@0.2
rubber-bounce mallet F 0 bpm=104 bed=click@off@0.14
gel-squish kal J 0 bed=click@sparse@0.15
crystal-gems glass L 3
sand-shift pad J 0 bed=click@sparse@0.12
leather-creak drone A -4 bed=click@sparse@0.2
cork-pop felt F 0 bed=click@sparse@0.15
porcelain-clink bell F 2 bed=click@sparse@0.15
chess-pieces felt E 0 bed=click@sparse@0.15
domino-clack mallet E 0 bpm=100 bed=click@sync@0.18
"""
PLAN = {}
for _ln in PLAN_TEXT.strip().splitlines():
    _p = _ln.split()
    _kw = dict(x.split("=", 1) for x in _p[4:])
    PLAN["snd-" + _p[0]] = dict(style=_p[1], prog=_p[2], shift=int(_p[3]), **_kw)


def music_cfg(pid, raw):
    """Loop config for a pack: its explicit MUSIC entry (original wave) or the plan-based style."""
    if pid in MUSIC:
        return MUSIC[pid]
    pl = PLAN[pid]
    cfg = dict(STYLES[pl["style"]], voice=VOICES[pl["style"]], prog=PROGS[pl["prog"]], shift=pl["shift"], secs=int(pl.get("secs", 40)))
    if "bpm" in pl:
        cfg["bpm"] = float(pl["bpm"])
    gens = {"click": raw["click"], "open": raw["menu_open"], "close": raw["menu_close"], "sel": raw["menu_select"],
            "kick": kick(55, 0.3), "snare": snare(), "hat": hat(0.05, 8000), "clap": clap()}
    bed, sparse = [], []
    for ent in filter(None, pl.get("bed", "").split(",")):
        s, p_, g = ent.split("@")
        x = gens[s][: int(0.6 * SR)]
        pk = max(abs(v) for v in x) or 1.0
        x = [v / pk for v in x]
        if "bpm" in pl and p_ != "sparse":
            bed.append((x, PAT[p_], float(g)))
        elif "bpm" in pl:
            sparse.append((x, 0.3, float(g)))
        else:
            sparse.append((x, 0.4, float(g)))
    if bed:
        cfg["bed"] = bed
    if sparse:
        cfg["sparse"] = sparse
    return cfg


def make_achievement(pid, raw, cfg):
    pl = PLAN.get(pid)
    voice = cfg["voice"]
    low = (pl or {}).get("style") in ("organ", "str", "drone", "pad")
    b = (48 if low else 60) + ((cfg.get("shift", 0) + 6) % 12 - 6)
    prog = (pl or {}).get("prog", "A")
    mode = (pl or {}).get("ach") or ("minor" if prog in "BCEGHI" else "major")
    iv = {"major": [0, 4, 7, 12, 16], "minor": [0, 3, 7, 12, 15], "fan": [0, 0, 7, 12, 19]}[mode]
    tri = [0, 4, 7] if mode != "minor" else [0, 3, 7]
    gap = 0.07 if mode != "fan" else 0.11
    items = [(k * gap, voice(hz(b + m), 0.55), 1.0) for k, m in enumerate(iv)]
    t0 = gap * len(iv)
    items += [(t0 + k * 0.018, voice(hz(b + 12 + m), 1.15), 0.85) for k, m in enumerate(tri)]
    items.append((t0, voice(hz(b - 12), 1.2), 0.6))
    if pl and "click@" in pl.get("bed", ""):
        items += [(k * gap, raw["click"], 0.5) for k in range(len(iv))]
    x = echo(layer(items), 0.17, 0.3, 2)
    return x[: int(2.3 * SR)]


W2_IDS = {q[0] for q in W2_PACKS}


def tags_from_zip(path):
    import zipfile as _z
    with _z.ZipFile(path) as z:
        keys = set(json.loads(z.read("sounds.json"))["files"])
    t = []
    if "click" in keys:
        t.append("contains:click")
    if {"menu_open", "menu_close", "menu_select"} <= keys:
        t.append("contains:menu")
    if "music" in keys:
        t.append("contains:music")
    if "achievement" in keys:
        t.append("contains:achievement")
    return sorted(t)


PACK_META = {p[0]: p for p in PACKS}


def build_one(task):
    pid, digest, force = task
    meta = PACK_META[pid]
    dist = SOUNDS / "dist" / pid
    seed(pid)
    raw = meta[7]()
    shorts = loudness_match({k: finish(raw[k]) for k in ("click", "menu_open", "menu_close", "menu_select")})
    files = {f"{k}.wav": write_wav(v) for k, v in shorts.items()}
    if pid in W2_IDS:
        for k_ in shorts:
            m_ = sum(shorts[k_]) / len(shorts[k_])
            y_ = fadeio([v - m_ for v in shorts[k_]], 0.002, 0.02)
            m2_ = sum(y_) / len(y_)
            shorts[k_] = limit(fadeio([v - m2_ for v in y_], 0.001, 0.01), PEAK_SHORT)
        files = {f"{k}.wav": write_wav(v) for k, v in shorts.items()}
    cfg = music_cfg(pid, raw)
    seed(pid + "/music")
    music = music_match(music_loop(cfg))
    files["music.ogg"] = write_ogg(music)
    seed(pid + "/ach")
    ach = finish(make_achievement(pid, raw, cfg))
    k = TARGET_SEL_RMS / active_rms(ach)
    ach = limit([v * k for v in ach], PEAK_SHORT)
    files["achievement.wav"] = write_wav(ach)
    fmap = {k_: f"{k_}.wav" for k_ in shorts}
    fmap["music"] = "music.ogg"
    fmap["achievement"] = "achievement.wav"
    sj = json.dumps({"formatVersion": 1, "id": pid, "version": 1, "files": fmap}, indent=2) + "\n"
    write_if_changed(dist / "item.ecxsounds", zip_pack(pid, files, sj))
    order = [shorts["click"], shorts["menu_open"], shorts["menu_close"], shorts["menu_select"]]
    pal = palette(meta[6])
    rnd = random.Random(zlib.crc32(pid.encode()))
    save_webp(make_cover(pal, order, rnd, pid), dist / "cover.webp")
    save_webp(make_waveform(pal, order), dist / "waveform.webp")
    write_if_changed(dist / "build.json", json.dumps({"sourceSha256": digest}, indent=2) + "\n")
    if os.environ.get("SOUNDS_REVIEW"):
        rv = ROOT / ".workspace" / "review" / "sounds"
        rv.mkdir(parents=True, exist_ok=True)
        gap = [0.0] * int(0.25 * SR)
        (rv / f"{pid}.wav").write_bytes(write_wav(order[0] + gap + order[1] + gap + order[2] + gap + order[3]))
        (rv / f"{pid}-music.wav").write_bytes(write_wav(music))
        (rv / f"{pid}-achievement.wav").write_bytes(write_wav(ach))
    return pid


def main():
    force = "--force" in sys.argv
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    SOUNDS.mkdir(parents=True, exist_ok=True)
    # Keep translations already present in the files; this table only owns the English text.
    old_cats = {}
    if (SOUNDS / "categories.json").exists():
        old_cats = {c["id"]: c.get("labels", {}) for c in json.loads((SOUNDS / "categories.json").read_text(encoding="utf-8"))}
    write_if_changed(SOUNDS / "categories.json", json.dumps(
        [{"id": i, "labels": {**old_cats.get(i, {}), "en": l}} for i, l in CATEGORIES], indent=2, ensure_ascii=False) + "\n")
    script_bytes = SELF.read_bytes()
    todo, bases = [], {}
    for pid, name, desc, cats, pro, featured, hue, builder in PACKS:
        if only and pid not in only:
            continue
        assert len(name) <= 40 and len(desc) <= 160, pid
        assert pid in PLAN or pid in MUSIC, pid
        pal = palette(hue)
        listing = {
            "version": 1, "categories": cats, "tags": [], "featured": featured, "publishedAt": PUBLISHED_AT,
            "requiresPro": pro, "minAppVersionCode": 0,
            "swatch": [hexc(pal["bg"]), hexc(pal["main"]), hexc(pal["accent"]), hexc(pal["light"])],
            "name": {"en": name}, "description": {"en": desc},
        }
        src = SOUNDS / "src" / pid
        src.mkdir(parents=True, exist_ok=True)
        (SOUNDS / "dist" / pid).mkdir(parents=True, exist_ok=True)
        if (src / "listing.json").exists():
            old = json.loads((src / "listing.json").read_text(encoding="utf-8"))
            listing["name"] = {**old.get("name", {}), "en": name}
            listing["description"] = {**old.get("description", {}), "en": desc}
        bases[pid] = listing
        dist = SOUNDS / "dist" / pid
        digest = hashlib.sha256((json.dumps(listing, sort_keys=True, ensure_ascii=False)).encode() + b"\0" + script_bytes).hexdigest()
        bj = dist / "build.json"
        if (not force and bj.exists() and json.loads(bj.read_text())["sourceSha256"] == digest
                and all((dist / f).exists() for f in ("item.ecxsounds", "cover.webp", "waveform.webp"))):
            print(f"= {pid} (up to date)")
        else:
            todo.append((pid, digest, force))
    if todo:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
            for pid in ex.map(build_one, todo):
                print(f"+ {pid}", flush=True)
    for pid, listing in bases.items():
        listing["tags"] = tags_from_zip(SOUNDS / "dist" / pid / "item.ecxsounds")
        write_if_changed(SOUNDS / "src" / pid / "listing.json", (json.dumps(listing, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
