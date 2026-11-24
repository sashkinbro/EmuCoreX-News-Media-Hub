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

CATEGORIES = [
    ("soft", "Soft"), ("retro", "Retro"), ("mechanical", "Mechanical"),
    ("musical", "Musical"), ("sci-fi", "Sci-Fi"), ("ambient", "Ambient"),
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


def make_cover(pal, sounds, rnd, pid):
    img = COVERS[STYLE_OF[pid]](pal, sounds, rnd).convert("RGB")
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


def main():
    force = "--force" in sys.argv
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    SOUNDS.mkdir(parents=True, exist_ok=True)
    write_if_changed(SOUNDS / "categories.json", json.dumps(
        [{"id": i, "labels": {"en": l}} for i, l in CATEGORIES], indent=2, ensure_ascii=False) + "\n")
    script_bytes = SELF.read_bytes()
    for pid, name, desc, cats, pro, featured, hue, builder in PACKS:
        if only and pid not in only:
            continue
        has_music = pid in MUSIC
        pro = pro or has_music
        assert len(name) <= 40 and len(desc) <= 160
        pal = palette(hue)
        listing = {
            "version": 1, "categories": cats, "tags": [], "featured": featured, "publishedAt": PUBLISHED_AT,
            "requiresPro": pro, "minAppVersionCode": 0,
            "swatch": [hexc(pal["bg"]), hexc(pal["main"]), hexc(pal["accent"]), hexc(pal["light"])],
            "name": {"en": name}, "description": {"en": desc},
        }
        src = SOUNDS / "src" / pid
        dist = SOUNDS / "dist" / pid
        src.mkdir(parents=True, exist_ok=True)
        dist.mkdir(parents=True, exist_ok=True)
        listing_bytes = (json.dumps(listing, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        write_if_changed(src / "listing.json", listing_bytes)
        digest = hashlib.sha256(listing_bytes + b"\0" + script_bytes).hexdigest()
        bj = dist / "build.json"
        if (not force and bj.exists() and json.loads(bj.read_text())["sourceSha256"] == digest
                and all((dist / f).exists() for f in ("item.ecxsounds", "cover.webp", "waveform.webp"))):
            print(f"= {pid} (up to date)")
            continue
        seed(pid)
        raw = builder()
        shorts = loudness_match({k: finish(raw[k]) for k in ("click", "menu_open", "menu_close", "menu_select")})
        files = {f"{k}.wav": write_wav(v) for k, v in shorts.items()}
        music = None
        if has_music:
            seed(pid + "/music")
            music = music_match(music_loop(MUSIC[pid]))
            files["music.wav"] = write_wav(music)
        fmap = {k: f"{k}.wav" for k in shorts}
        if has_music:
            fmap["music"] = "music.wav"
        sj = json.dumps({"formatVersion": 1, "id": pid, "version": 1, "files": fmap}, indent=2) + "\n"
        write_if_changed(dist / "item.ecxsounds", zip_pack(pid, files, sj))
        order = [shorts["click"], shorts["menu_open"], shorts["menu_close"], shorts["menu_select"]]
        rnd = random.Random(zlib.crc32(pid.encode()))
        save_webp(make_cover(pal, order, rnd, pid), dist / "cover.webp")
        save_webp(make_waveform(pal, order), dist / "waveform.webp")
        write_if_changed(bj, json.dumps({"sourceSha256": digest}, indent=2) + "\n")
        print(f"+ {pid}" + (" (music)" if has_music else ""))


if __name__ == "__main__":
    main()
