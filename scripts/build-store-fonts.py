#!/usr/bin/env python3
"""EmuCore Store: Fonts section builder.

Every font comes from github.com/google/fonts (SIL OFL 1.1 or Apache-2.0 only), one file per item.
Writes store/font/{categories.json, src/<id>/listing.json,
dist/<id>/{item.ecxfont, cover.webp, gallery-1.webp, gallery-2.webp, build.json}}.
Covers and gallery images are rendered with the font itself (Pillow). Deterministic and idempotent
(skips items whose build.json hash matches). Non-English translations already present in
listing.json / categories.json are kept unless --retranslate is given.

  python scripts/build-store-fonts.py --fetch           # download missing font files into .workspace/fonts-cache/<id>/
  python scripts/build-store-fonts.py                   # build everything that changed
  python scripts/build-store-fonts.py --force           # rebuild all
  python scripts/build-store-fonts.py --only a,b        # selected ids
  python scripts/build-store-fonts.py --sheet out.png   # also write a contact sheet of covers
  python scripts/build-store-fonts.py --retranslate     # overwrite stored translations with the built-in ones

Downloaded files are untrusted data: they are only parsed with fontTools and Pillow, never executed.
"""
import argparse
import colorsys
import hashlib
import io
import json
import math
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HUB = Path(__file__).resolve().parents[1]
ROOT = HUB / "store" / "font"
CACHE = HUB / ".workspace" / "fonts-cache"
PUBLISHED = "2026-11-24T12:00:00Z"
RAW = "https://raw.githubusercontent.com/google/fonts/main"
LOCALES = ["en", "ar", "cs", "de", "es", "fa", "fr", "hi", "id", "it", "ja", "ko", "pl", "pt", "ru", "tr", "uk", "zh"]
LICENSE_IDS = {"ofl": "OFL-1.1", "apache": "Apache-2.0"}
MAX_PACKAGE = 4 * 1024 * 1024

# (family, google/fonts directory, license directory, file name)
SOURCES = [
    ("Press Start 2P", "pressstart2p", "ofl", "PressStart2P-Regular.ttf"),
    ("Pixelify Sans", "pixelifysans", "ofl", "PixelifySans[wght].ttf"),
    ("Silkscreen", "silkscreen", "ofl", "Silkscreen-Regular.ttf"),
    ("VT323", "vt323", "ofl", "VT323-Regular.ttf"),
    ("Jersey 10", "jersey10", "ofl", "Jersey10-Regular.ttf"),
    ("Tiny5", "tiny5", "ofl", "Tiny5[BLED,JITT,ROND,wdth,wght].ttf"),
    ("DotGothic16", "dotgothic16", "ofl", "DotGothic16-Regular.ttf"),
    ("Handjet", "handjet", "ofl", "Handjet[ELGR,ELSH,wght].ttf"),
    ("Rubik Pixels", "rubikpixels", "ofl", "RubikPixels-Regular.ttf"),
    ("JetBrains Mono", "jetbrainsmono", "ofl", "JetBrainsMono[wght].ttf"),
    ("Fira Code", "firacode", "ofl", "FiraCode[wght].ttf"),
    ("IBM Plex Mono", "ibmplexmono", "ofl", "IBMPlexMono-Regular.ttf"),
    ("Space Mono", "spacemono", "ofl", "SpaceMono-Regular.ttf"),
    ("Share Tech Mono", "sharetechmono", "ofl", "ShareTechMono-Regular.ttf"),
    ("Roboto Mono", "robotomono", "ofl", "RobotoMono[wght].ttf"),
    ("Source Code Pro", "sourcecodepro", "ofl", "SourceCodePro[wght].ttf"),
    ("Cousine", "cousine", "ofl", "Cousine-Regular.ttf"),
    ("Anonymous Pro", "anonymouspro", "ofl", "AnonymousPro-Regular.ttf"),
    ("Martian Mono", "martianmono", "ofl", "MartianMono[wdth,wght].ttf"),
    ("Victor Mono", "victormono", "ofl", "VictorMono[wght].ttf"),
    ("Inter", "inter", "ofl", "Inter[opsz,wght].ttf"),
    ("Manrope", "manrope", "ofl", "Manrope[wght].ttf"),
    ("Nunito", "nunito", "ofl", "Nunito[wght].ttf"),
    ("Rubik", "rubik", "ofl", "Rubik[wght].ttf"),
    ("Montserrat", "montserrat", "ofl", "Montserrat[wght].ttf"),
    ("Onest", "onest", "ofl", "Onest[wght].ttf"),
    ("Golos Text", "golostext", "ofl", "GolosText[wght].ttf"),
    ("Unbounded", "unbounded", "ofl", "Unbounded[wght].ttf"),
    ("Raleway", "raleway", "ofl", "Raleway[wght].ttf"),
    ("Open Sans", "opensans", "ofl", "OpenSans[wdth,wght].ttf"),
    ("PT Sans", "ptsans", "ofl", "PT_Sans-Web-Regular.ttf"),
    ("Source Sans 3", "sourcesans3", "ofl", "SourceSans3[wght].ttf"),
    ("Fira Sans", "firasans", "ofl", "FiraSans-Regular.ttf"),
    ("IBM Plex Sans", "ibmplexsans", "ofl", "IBMPlexSans[wdth,wght].ttf"),
    ("Wix Madefor Display", "wixmadefordisplay", "ofl", "WixMadeforDisplay[wght].ttf"),
    ("Commissioner", "commissioner", "ofl", "Commissioner[FLAR,VOLM,slnt,wght].ttf"),
    ("Geologica", "geologica", "ofl", "Geologica[CRSV,SHRP,slnt,wght].ttf"),
    ("Oswald", "oswald", "ofl", "Oswald[wght].ttf"),
    ("Bebas Neue", "bebasneue", "ofl", "BebasNeue-Regular.ttf"),
    ("Teko", "teko", "ofl", "Teko[wght].ttf"),
    ("Saira Condensed", "sairacondensed", "ofl", "SairaCondensed-Regular.ttf"),
    ("Roboto Condensed", "robotocondensed", "ofl", "RobotoCondensed[wght].ttf"),
    ("Fira Sans Condensed", "firasanscondensed", "ofl", "FiraSansCondensed-Regular.ttf"),
    ("Yanone Kaffeesatz", "yanonekaffeesatz", "ofl", "YanoneKaffeesatz[wght].ttf"),
    ("Anton", "anton", "ofl", "Anton-Regular.ttf"),
    ("Big Shoulders Display", "bigshouldersdisplay", "ofl", "BigShouldersDisplay[wght].ttf"),
    ("Comfortaa", "comfortaa", "ofl", "Comfortaa[wght].ttf"),
    ("Kosugi Maru", "kosugimaru", "apache", "KosugiMaru-Regular.ttf"),
    ("Fredoka", "fredoka", "ofl", "Fredoka[wdth,wght].ttf"),
    ("Baloo 2", "baloo2", "ofl", "Baloo2[wght].ttf"),
    ("Varela Round", "varelaround", "ofl", "VarelaRound-Regular.ttf"),
    ("Marmelad", "marmelad", "ofl", "Marmelad-Regular.ttf"),
    ("Zen Maru Gothic", "zenmarugothic", "ofl", "ZenMaruGothic-Regular.ttf"),
    ("Bellota", "bellota", "ofl", "Bellota-Regular.ttf"),
    ("Bangers", "bangers", "ofl", "Bangers-Regular.ttf"),
    ("Comic Neue", "comicneue", "ofl", "ComicNeue-Regular.ttf"),
    ("Balsamiq Sans", "balsamiqsans", "ofl", "BalsamiqSans-Regular.ttf"),
    ("Pangolin", "pangolin", "ofl", "Pangolin-Regular.ttf"),
    ("Playpen Sans", "playpensans", "ofl", "PlaypenSans[wght].ttf"),
    ("Underdog", "underdog", "ofl", "Underdog-Regular.ttf"),
    ("Lora", "lora", "ofl", "Lora[wght].ttf"),
    ("Merriweather", "merriweather", "ofl", "Merriweather[opsz,wdth,wght].ttf"),
    ("PT Serif", "ptserif", "ofl", "PT_Serif-Web-Regular.ttf"),
    ("Playfair Display", "playfairdisplay", "ofl", "PlayfairDisplay[wght].ttf"),
    ("Literata", "literata", "ofl", "Literata[opsz,wght].ttf"),
    ("Source Serif 4", "sourceserif4", "ofl", "SourceSerif4[opsz,wght].ttf"),
    ("Bitter", "bitter", "ofl", "Bitter[wght].ttf"),
    ("Roboto Slab", "robotoslab", "apache", "RobotoSlab[wght].ttf"),
    ("Vollkorn", "vollkorn", "ofl", "Vollkorn[wght].ttf"),
    ("EB Garamond", "ebgaramond", "ofl", "EBGaramond[wght].ttf"),
    ("Old Standard TT", "oldstandardtt", "ofl", "OldStandard-Regular.ttf"),
    ("Caveat", "caveat", "ofl", "Caveat[wght].ttf"),
    ("Pacifico", "pacifico", "ofl", "Pacifico-Regular.ttf"),
    ("Kalam", "kalam", "ofl", "Kalam-Regular.ttf"),
    ("Neucha", "neucha", "ofl", "Neucha.ttf"),
    ("Marck Script", "marckscript", "ofl", "MarckScript-Regular.ttf"),
    ("Bad Script", "badscript", "ofl", "BadScript-Regular.ttf"),
    ("Amatic SC", "amaticsc", "ofl", "AmaticSC-Regular.ttf"),
    ("Great Vibes", "greatvibes", "ofl", "GreatVibes-Regular.ttf"),
    ("Zen Kurenaido", "zenkurenaido", "ofl", "ZenKurenaido-Regular.ttf"),
    ("UnifrakturMaguntia", "unifrakturmaguntia", "ofl", "UnifrakturMaguntia-Book.ttf"),
    ("Pirata One", "pirataone", "ofl", "PirataOne-Regular.ttf"),
    ("Cinzel", "cinzel", "ofl", "Cinzel[wght].ttf"),
    ("Cinzel Decorative", "cinzeldecorative", "ofl", "CinzelDecorative-Regular.ttf"),
    ("New Rocker", "newrocker", "ofl", "NewRocker-Regular.ttf"),
    ("Metal Mania", "metalmania", "ofl", "MetalMania-Regular.ttf"),
    ("Ruslan Display", "ruslandisplay", "ofl", "RuslanDisplay-Regular.ttf"),
    ("Yeseva One", "yesevaone", "ofl", "YesevaOne-Regular.ttf"),
    ("Stalinist One", "stalinistone", "ofl", "StalinistOne-Regular.ttf"),
    ("Kelly Slab", "kellyslab", "ofl", "KellySlab-Regular.ttf"),
    ("Rampart One", "rampartone", "ofl", "RampartOne-Regular.ttf"),
    ("Dela Gothic One", "delagothicone", "ofl", "DelaGothicOne-Regular.ttf"),
    ("Reggae One", "reggaeone", "ofl", "ReggaeOne-Regular.ttf"),
    ("Rubik Wet Paint", "rubikwetpaint", "ofl", "RubikWetPaint-Regular.ttf"),
    ("Monoton", "monoton", "ofl", "Monoton-Regular.ttf"),
    ("Bungee", "bungee", "ofl", "Bungee-Regular.ttf"),
    ("Righteous", "righteous", "ofl", "Righteous-Regular.ttf"),
    ("Lobster", "lobster", "ofl", "Lobster-Regular.ttf"),
    ("Faster One", "fasterone", "ofl", "FasterOne-Regular.ttf"),
    ("Rye", "rye", "ofl", "Rye-Regular.ttf"),
    ("Rubik 80s Fade", "rubik80sfade", "ofl", "Rubik80sFade-Regular.ttf"),
    ("Days One", "daysone", "ofl", "DaysOne-Regular.ttf"),
    ("Orbitron", "orbitron", "ofl", "Orbitron[wght].ttf"),
    ("Exo 2", "exo2", "ofl", "Exo2[wght].ttf"),
    ("Audiowide", "audiowide", "ofl", "Audiowide-Regular.ttf"),
    ("Russo One", "russoone", "ofl", "RussoOne-Regular.ttf"),
    ("Oxanium", "oxanium", "ofl", "Oxanium[wght].ttf"),
    ("Chakra Petch", "chakrapetch", "ofl", "ChakraPetch-Regular.ttf"),
    ("Michroma", "michroma", "ofl", "Michroma-Regular.ttf"),
    ("Turret Road", "turretroad", "ofl", "TurretRoad-Regular.ttf"),
    ("Major Mono Display", "majormonodisplay", "ofl", "MajorMonoDisplay-Regular.ttf"),
    ("Play", "play", "ofl", "Play-Regular.ttf"),
    ("Tektur", "tektur", "ofl", "Tektur[wdth,wght].ttf"),
    ("Zen Dots", "zendots", "ofl", "ZenDots-Regular.ttf"),
    ("Atkinson Hyperlegible", "atkinsonhyperlegible", "ofl", "AtkinsonHyperlegible-Regular.ttf"),
    ("Atkinson Hyperlegible Next", "atkinsonhyperlegiblenext", "ofl", "AtkinsonHyperlegibleNext[wght].ttf"),
    ("Lexend", "lexend", "ofl", "Lexend[wght].ttf"),
    ("Andika", "andika", "ofl", "Andika-Regular.ttf"),
]

# family -> (category, extra style tag or "", pro, featured, cover theme "name:variant")
META = {
    "Press Start 2P": ("pixel", "", 0, 1, "pixgrid:0"),
    "Pixelify Sans": ("pixel", "", 0, 1, "pixgrid:1"),
    "Silkscreen": ("pixel", "", 0, 0, "terminal:1"),
    "VT323": ("pixel", "terminal", 0, 0, "terminal:0"),
    "Jersey 10": ("pixel", "", 1, 0, "pixgrid:2"),
    "Tiny5": ("pixel", "", 0, 0, "pixgrid:3"),
    "DotGothic16": ("pixel", "", 1, 0, "pixgrid:0"),
    "Handjet": ("pixel", "", 1, 0, "synth:1"),
    "Rubik Pixels": ("pixel", "", 1, 0, "neon:1"),
    "JetBrains Mono": ("mono", "code", 0, 1, "editor:0"),
    "Fira Code": ("mono", "code", 0, 0, "editor:1"),
    "IBM Plex Mono": ("mono", "code", 0, 0, "slate:0"),
    "Space Mono": ("mono", "geometric", 0, 0, "blueprint:0"),
    "Share Tech Mono": ("mono", "terminal", 0, 0, "terminal:0"),
    "Roboto Mono": ("mono", "code", 0, 0, "editor:2"),
    "Source Code Pro": ("mono", "code", 0, 0, "editor:3"),
    "Cousine": ("mono", "", 0, 0, "light:0"),
    "Anonymous Pro": ("mono", "terminal", 0, 0, "paper:0"),
    "Martian Mono": ("mono", "wide", 1, 0, "neon:0"),
    "Victor Mono": ("mono", "code", 1, 0, "editor:1"),
    "Inter": ("sans", "neutral", 0, 1, "slate:0"),
    "Manrope": ("sans", "geometric", 0, 0, "light:1"),
    "Nunito": ("sans", "friendly", 0, 0, "pastel:0"),
    "Rubik": ("sans", "friendly", 0, 0, "slate:1"),
    "Montserrat": ("sans", "geometric", 0, 0, "light:0"),
    "Onest": ("sans", "neutral", 0, 0, "slate:2"),
    "Golos Text": ("sans", "neutral", 0, 0, "light:2"),
    "Unbounded": ("sans", "wide", 1, 1, "neon:2"),
    "Raleway": ("sans", "elegant", 0, 0, "light:3"),
    "Open Sans": ("sans", "humanist", 0, 0, "light:0"),
    "PT Sans": ("sans", "humanist", 0, 0, "slate:0"),
    "Source Sans 3": ("sans", "humanist", 0, 0, "light:1"),
    "Fira Sans": ("sans", "humanist", 0, 0, "slate:1"),
    "IBM Plex Sans": ("sans", "neutral", 0, 0, "slate:2"),
    "Wix Madefor Display": ("sans", "modern", 0, 0, "pastel:1"),
    "Commissioner": ("sans", "modern", 1, 0, "light:2"),
    "Geologica": ("sans", "modern", 1, 0, "slate:3"),
    "Oswald": ("condensed", "sport", 0, 1, "stripes:0"),
    "Bebas Neue": ("condensed", "sport", 0, 0, "stripes:1"),
    "Teko": ("condensed", "sport", 0, 0, "stripes:2"),
    "Saira Condensed": ("condensed", "sport", 0, 0, "stripes:3"),
    "Roboto Condensed": ("condensed", "neutral", 0, 0, "slate:0"),
    "Fira Sans Condensed": ("condensed", "humanist", 0, 0, "stripes:0"),
    "Yanone Kaffeesatz": ("condensed", "vintage", 1, 0, "poster:0"),
    "Anton": ("condensed", "impact", 1, 0, "poster:1"),
    "Big Shoulders Display": ("condensed", "sport", 1, 0, "stripes:1"),
    "Comfortaa": ("rounded", "geometric", 0, 1, "pastel:0"),
    "Kosugi Maru": ("rounded", "anime", 0, 0, "pastel:2"),
    "Fredoka": ("rounded", "friendly", 0, 0, "pastel:1"),
    "Baloo 2": ("rounded", "friendly", 0, 0, "pastel:3"),
    "Varela Round": ("rounded", "geometric", 0, 0, "pastel:2"),
    "Marmelad": ("rounded", "humanist", 0, 0, "light:1"),
    "Zen Maru Gothic": ("rounded", "anime", 1, 0, "pastel:0"),
    "Bellota": ("rounded", "playful", 1, 0, "pastel:3"),
    "Bangers": ("comic", "impact", 0, 1, "halftone:0"),
    "Comic Neue": ("comic", "casual", 0, 0, "halftone:1"),
    "Balsamiq Sans": ("comic", "sketch", 0, 0, "notebook:0"),
    "Pangolin": ("comic", "casual", 0, 0, "notebook:1"),
    "Playpen Sans": ("comic", "casual", 1, 0, "pastel:1"),
    "Underdog": ("comic", "impact", 1, 0, "halftone:2"),
    "Lora": ("serif", "book", 0, 0, "paper:0"),
    "Merriweather": ("serif", "book", 0, 0, "paper:1"),
    "PT Serif": ("serif", "book", 0, 0, "paper:2"),
    "Playfair Display": ("serif", "elegant", 0, 1, "paper:3"),
    "Literata": ("serif", "book", 0, 0, "paper:0"),
    "Source Serif 4": ("serif", "book", 0, 0, "paper:1"),
    "Bitter": ("serif", "slab", 0, 0, "light:0"),
    "Roboto Slab": ("serif", "slab", 0, 0, "paper:2"),
    "Vollkorn": ("serif", "book", 0, 0, "parchment:0"),
    "EB Garamond": ("serif", "classic", 0, 0, "parchment:1"),
    "Old Standard TT": ("serif", "classic", 1, 0, "parchment:2"),
    "Caveat": ("handwriting", "script", 0, 1, "notebook:0"),
    "Pacifico": ("handwriting", "brush", 0, 0, "synth:2"),
    "Kalam": ("handwriting", "script", 0, 0, "notebook:1"),
    "Neucha": ("handwriting", "chalk", 0, 0, "chalk:0"),
    "Marck Script": ("handwriting", "calligraphy", 1, 0, "parchment:1"),
    "Bad Script": ("handwriting", "script", 0, 0, "notebook:0"),
    "Amatic SC": ("handwriting", "chalk", 0, 0, "chalk:1"),
    "Great Vibes": ("handwriting", "calligraphy", 1, 0, "paper:3"),
    "Zen Kurenaido": ("handwriting", "anime", 1, 0, "notebook:1"),
    "UnifrakturMaguntia": ("fantasy", "blackletter", 1, 0, "parchment:2"),
    "Pirata One": ("fantasy", "blackletter", 1, 0, "stone:0"),
    "Cinzel": ("fantasy", "roman", 0, 1, "stone:1"),
    "Cinzel Decorative": ("fantasy", "roman", 1, 0, "stone:2"),
    "New Rocker": ("fantasy", "blackletter", 1, 0, "stone:0"),
    "Metal Mania": ("fantasy", "metal", 1, 0, "stone:3"),
    "Ruslan Display": ("display", "folk", 1, 0, "poster:2"),
    "Yeseva One": ("display", "elegant", 0, 0, "paper:3"),
    "Stalinist One": ("display", "constructivist", 1, 0, "poster:1"),
    "Kelly Slab": ("display", "slab", 0, 0, "slate:0"),
    "Rampart One": ("display", "anime", 1, 0, "stripes:2"),
    "Dela Gothic One": ("display", "anime", 1, 0, "neon:1"),
    "Reggae One": ("display", "anime", 1, 0, "poster:0"),
    "Rubik Wet Paint": ("display", "paint", 1, 0, "neon:2"),
    "Monoton": ("retro", "neon", 1, 1, "synth:0"),
    "Bungee": ("retro", "signage", 1, 0, "poster:0"),
    "Righteous": ("retro", "geometric", 0, 0, "synth:2"),
    "Lobster": ("retro", "script", 0, 0, "poster:2"),
    "Faster One": ("retro", "speed", 1, 0, "synth:1"),
    "Rye": ("retro", "western", 1, 0, "poster:3"),
    "Rubik 80s Fade": ("retro", "eighties", 1, 0, "synth:3"),
    "Days One": ("retro", "impact", 0, 0, "stripes:2"),
    "Orbitron": ("scifi", "geometric", 0, 1, "neon:0"),
    "Exo 2": ("scifi", "technical", 0, 0, "blueprint:0"),
    "Audiowide": ("scifi", "rounded", 0, 0, "neon:1"),
    "Russo One": ("scifi", "impact", 0, 0, "stripes:3"),
    "Oxanium": ("scifi", "technical", 0, 0, "neon:3"),
    "Chakra Petch": ("scifi", "technical", 0, 0, "blueprint:1"),
    "Michroma": ("scifi", "wide", 1, 0, "slate:3"),
    "Turret Road": ("scifi", "technical", 1, 0, "terminal:0"),
    "Major Mono Display": ("scifi", "mono", 1, 0, "terminal:1"),
    "Play": ("scifi", "technical", 0, 0, "blueprint:2"),
    "Tektur": ("scifi", "technical", 1, 0, "neon:0"),
    "Zen Dots": ("scifi", "wide", 1, 0, "synth:0"),
    "Atkinson Hyperlegible": ("accessibility", "", 0, 1, "contrast:0"),
    "Atkinson Hyperlegible Next": ("accessibility", "", 0, 0, "contrast:1"),
    "Lexend": ("accessibility", "", 0, 0, "light:0"),
    "Andika": ("accessibility", "", 0, 0, "contrast:2"),
}

CATEGORIES = ["pixel", "mono", "sans", "display", "condensed", "rounded", "comic", "serif", "handwriting", "fantasy", "retro", "scifi", "accessibility"]

CAT_LABELS = {
    "pixel": dict(en="Pixel", ar="بكسلي", cs="Pixelové", de="Pixel", es="Píxeles", fa="پیکسلی", fr="Pixel", hi="पिक्सेल", id="Piksel", it="Pixel", ja="ピクセル", ko="픽셀", pl="Piksele", pt="Pixel", ru="Пиксельные", tr="Piksel", uk="Піксельні", zh="像素"),
    "mono": dict(en="Monospace", ar="ثابت العرض", cs="Neproporcionální", de="Festbreite", es="Monoespaciadas", fa="تک‌فاصله", fr="Chasse fixe", hi="मोनोस्पेस", id="Monospasi", it="Monospazio", ja="等幅", ko="고정폭", pl="Stałej szerokości", pt="Monoespaçadas", ru="Моноширинные", tr="Tek aralıklı", uk="Моноширинні", zh="等宽"),
    "sans": dict(en="Sans-serif", ar="بلا زوائد", cs="Bezpatkové", de="Serifenlos", es="Sans serif", fa="بدون سریف", fr="Sans empattement", hi="सैन्स-सेरिफ़", id="Sans-serif", it="Sans serif", ja="サンセリフ", ko="산세리프", pl="Bezszeryfowe", pt="Sem serifa", ru="Без засечек", tr="Tırnaksız", uk="Без засічок", zh="无衬线"),
    "display": dict(en="Display", ar="عرض", cs="Titulkové", de="Display", es="Display", fa="نمایشی", fr="Titrage", hi="डिस्प्ले", id="Display", it="Display", ja="ディスプレイ", ko="디스플레이", pl="Ozdobne", pt="Display", ru="Акцидентные", tr="Gösteri", uk="Акцидентні", zh="展示"),
    "condensed": dict(en="Condensed", ar="مضغوط", cs="Úzké", de="Schmal", es="Condensadas", fa="باریک", fr="Condensées", hi="संकुचित", id="Ramping", it="Condensati", ja="コンデンスド", ko="콘덴스드", pl="Wąskie", pt="Condensadas", ru="Узкие", tr="Dar", uk="Вузькі", zh="窄体"),
    "rounded": dict(en="Rounded", ar="مستدير", cs="Zaoblené", de="Rund", es="Redondeadas", fa="گرد", fr="Arrondies", hi="गोल", id="Membulat", it="Arrotondati", ja="丸ゴシック", ko="둥근", pl="Zaokrąglone", pt="Arredondadas", ru="Округлые", tr="Yuvarlak", uk="Округлі", zh="圆体"),
    "comic": dict(en="Comic", ar="كوميك", cs="Komiksové", de="Comic", es="Cómic", fa="کمیک", fr="BD", hi="कॉमिक", id="Komik", it="Fumetto", ja="コミック", ko="만화풍", pl="Komiksowe", pt="Quadrinhos", ru="Комиксные", tr="Çizgi roman", uk="Комікс", zh="漫画"),
    "serif": dict(en="Serif", ar="سيريف", cs="Patkové", de="Serifen", es="Serif", fa="سریف‌دار", fr="Empattements", hi="सेरिफ़", id="Serif", it="Serif", ja="セリフ", ko="세리프", pl="Szeryfowe", pt="Serifadas", ru="С засечками", tr="Tırnaklı", uk="Із засічками", zh="衬线"),
    "handwriting": dict(en="Handwriting", ar="خط اليد", cs="Ručně psané", de="Handschrift", es="Manuscritas", fa="دست‌نویس", fr="Manuscrites", hi="हस्तलिखित", id="Tulisan tangan", it="Calligrafici", ja="手書き", ko="손글씨", pl="Odręczne", pt="Manuscritas", ru="Рукописные", tr="El yazısı", uk="Рукописні", zh="手写"),
    "fantasy": dict(en="Fantasy", ar="خيالي", cs="Fantasy", de="Fantasy", es="Fantasía", fa="فانتزی", fr="Fantasy", hi="फ़ैंटेसी", id="Fantasi", it="Fantasy", ja="ファンタジー", ko="판타지", pl="Fantasy", pt="Fantasia", ru="Фэнтези", tr="Fantezi", uk="Фентезі", zh="奇幻"),
    "retro": dict(en="Retro", ar="كلاسيكي", cs="Retro", de="Retro", es="Retro", fa="قدیمی", fr="Rétro", hi="रेट्रो", id="Retro", it="Retrò", ja="レトロ", ko="레트로", pl="Retro", pt="Retrô", ru="Ретро", tr="Retro", uk="Ретро", zh="复古"),
    "scifi": dict(en="Sci-Fi", ar="خيال علمي", cs="Sci-fi", de="Sci-Fi", es="Ciencia ficción", fa="علمی-تخیلی", fr="Science-fiction", hi="साइंस-फ़िक्शन", id="Fiksi ilmiah", it="Fantascienza", ja="SF", ko="SF", pl="Sci-fi", pt="Ficção científica", ru="Sci-Fi", tr="Bilim kurgu", uk="Sci-Fi", zh="科幻"),
    "accessibility": dict(en="Accessibility", ar="سهولة القراءة", cs="Čitelné", de="Barrierefrei", es="Accesibles", fa="دسترس‌پذیر", fr="Accessibilité", hi="सुगम्यता", id="Aksesibilitas", it="Accessibili", ja="読みやすさ", ko="접근성", pl="Dostępność", pt="Acessibilidade", ru="Доступные", tr="Erişilebilirlik", uk="Доступність", zh="无障碍"),
}

# One sentence per category and locale; {n} is the font family name (fonts keep their names everywhere).
DESC = {
    "pixel": dict(
        en="{n} is a pixel-art typeface that brings classic console-game screens to your menus.",
        ar="{n} خط بكسلي يضفي على قوائمك مظهر شاشات ألعاب الكونسول الكلاسيكية.",
        cs="{n} je pixelové písmo, které dodá menu vzhled klasických konzolových her.",
        de="{n} ist eine Pixelschrift, die deinen Menüs den Look klassischer Konsolenspiele verleiht.",
        es="{n} es una tipografía de píxeles que da a tus menús el aspecto de las consolas clásicas.",
        fa="{n} یک قلم پیکسلی است که به منوهای شما ظاهر بازی‌های کلاسیک کنسولی می‌دهد.",
        fr="{n} est une police en pixels qui donne à vos menus l'allure des jeux de console d'antan.",
        hi="{n} एक पिक्सेल टाइपफ़ेस है जो आपके मेनू को क्लासिक कंसोल गेम जैसा रूप देता है।",
        id="{n} adalah huruf piksel yang memberi menu Anda tampilan gim konsol klasik.",
        it="{n} è un carattere a pixel che dà ai tuoi menu l'aspetto dei giochi per console di una volta.",
        ja="{n}は、メニューをクラシックなゲーム機の画面風に見せるピクセル書体です。",
        ko="{n}은(는) 메뉴를 클래식 콘솔 게임 화면처럼 보이게 하는 픽셀 서체입니다.",
        pl="{n} to pikselowa czcionka, która nadaje menu wygląd klasycznych gier konsolowych.",
        pt="{n} é uma fonte em pixels que dá aos seus menus o visual dos jogos de console clássicos.",
        ru="{n} — пиксельный шрифт, придающий меню вид экранов классических консольных игр.",
        tr="{n}, menülerinize klasik konsol oyunlarının görünümünü kazandıran piksel bir yazı tipidir.",
        uk="{n} — піксельний шрифт, що надає меню вигляду екранів класичних консольних ігор.",
        zh="{n}是一款像素字体，让菜单拥有经典主机游戏的画面感。",
    ),
    "mono": dict(
        en="{n} is a monospaced font with even letter widths, crisp for stats, logs and code.",
        ar="{n} خط ثابت العرض بحروف متساوية، واضح للإحصاءات والسجلات والشيفرة.",
        cs="{n} je neproporcionální písmo s pravidelnou šířkou znaků, ostré pro statistiky, logy i kód.",
        de="{n} ist eine Festbreitenschrift mit gleichmäßigen Zeichen, klar für Werte, Logs und Code.",
        es="{n} es una fuente monoespaciada de caracteres uniformes, nítida para estadísticas, registros y código.",
        fa="{n} یک قلم تک‌فاصله با حروف هم‌عرض است که برای آمار، گزارش‌ها و کد خوانا و شفاف است.",
        fr="{n} est une police à chasse fixe aux caractères réguliers, nette pour les stats, les journaux et le code.",
        hi="{n} समान चौड़ाई वाले अक्षरों का मोनोस्पेस फ़ॉन्ट है, जो आँकड़ों, लॉग और कोड के लिए साफ़ दिखता है।",
        id="{n} adalah fon monospasi dengan lebar huruf seragam, tajam untuk statistik, log, dan kode.",
        it="{n} è un carattere monospaziato dalle lettere uniformi, nitido per statistiche, log e codice.",
        ja="{n}は文字幅が揃った等幅フォントで、数値やログ、コードをくっきり表示します。",
        ko="{n}은(는) 글자 폭이 일정한 고정폭 글꼴로, 수치와 로그, 코드를 또렷하게 보여 줍니다.",
        pl="{n} to czcionka o stałej szerokości znaków, czytelna w statystykach, logach i kodzie.",
        pt="{n} é uma fonte monoespaçada de letras uniformes, nítida para estatísticas, registros e código.",
        ru="{n} — моноширинный шрифт с одинаковой шириной знаков, чёткий для статистики, логов и кода.",
        tr="{n}, eşit harf genişliğine sahip tek aralıklı bir yazı tipidir; istatistik, günlük ve kod için nettir.",
        uk="{n} — моноширинний шрифт з однаковою шириною знаків, чіткий для статистики, журналів і коду.",
        zh="{n}是一款等宽字体，字符宽度一致，适合清晰显示数据、日志和代码。",
    ),
    "sans": dict(
        en="{n} is a clean sans-serif that stays easy to read across menus, lists and settings.",
        ar="{n} خط بلا زوائد نظيف يسهل قراءته في القوائم والإعدادات.",
        cs="{n} je čisté bezpatkové písmo, které se dobře čte v menu, seznamech i nastavení.",
        de="{n} ist eine klare Groteskschrift, die in Menüs, Listen und Einstellungen gut lesbar bleibt.",
        es="{n} es una tipografía sans serif limpia, fácil de leer en menús, listas y ajustes.",
        fa="{n} یک قلم بدون سریف و تمیز است که در منوها، فهرست‌ها و تنظیمات به‌راحتی خوانده می‌شود.",
        fr="{n} est une police sans empattement épurée, facile à lire dans les menus, les listes et les réglages.",
        hi="{n} एक साफ़-सुथरा सैन्स-सेरिफ़ फ़ॉन्ट है, जो मेनू, सूचियों और सेटिंग्स में आसानी से पढ़ा जाता है।",
        id="{n} adalah fon sans-serif yang bersih dan mudah dibaca di menu, daftar, dan pengaturan.",
        it="{n} è un sans serif pulito, facile da leggere in menu, elenchi e impostazioni.",
        ja="{n}は、メニューやリスト、設定画面でも読みやすい、すっきりしたサンセリフ体です。",
        ko="{n}은(는) 메뉴와 목록, 설정 화면에서 읽기 쉬운 깔끔한 산세리프 서체입니다.",
        pl="{n} to czysta czcionka bezszeryfowa, czytelna w menu, listach i ustawieniach.",
        pt="{n} é uma fonte sem serifa limpa, fácil de ler em menus, listas e configurações.",
        ru="{n} — аккуратный шрифт без засечек, который удобно читать в меню, списках и настройках.",
        tr="{n}, menülerde, listelerde ve ayarlarda rahat okunan sade bir tırnaksız yazı tipidir.",
        uk="{n} — охайний шрифт без засічок, який зручно читати в меню, списках і налаштуваннях.",
        zh="{n}是一款简洁的无衬线字体，在菜单、列表和设置中都清晰易读。",
    ),
    "condensed": dict(
        en="{n} is a tall, condensed face with a sporty punch, great for titles and scoreboards.",
        ar="{n} خط مضغوط طويل بطابع رياضي قوي، رائع للعناوين ولوحات النتائج.",
        cs="{n} je vysoké úzké písmo se sportovním nádechem, skvělé pro nadpisy a výsledkové tabule.",
        de="{n} ist eine hohe, schmale Schrift mit sportlichem Biss, ideal für Titel und Anzeigetafeln.",
        es="{n} es una tipografía condensada y alta de aire deportivo, ideal para títulos y marcadores.",
        fa="{n} یک قلم باریک و بلند با حال‌وهوای ورزشی است که برای عنوان‌ها و جدول امتیازها عالی است.",
        fr="{n} est une police condensée et haute au style sportif, parfaite pour les titres et les tableaux de score.",
        hi="{n} खेल-जैसे अंदाज़ वाला ऊँचा, संकुचित फ़ॉन्ट है, जो शीर्षकों और स्कोरबोर्ड के लिए बढ़िया है।",
        id="{n} adalah huruf tinggi yang ramping bergaya olahraga, pas untuk judul dan papan skor.",
        it="{n} è un carattere alto e condensato dallo stile sportivo, ottimo per titoli e tabelloni.",
        ja="{n}は、スポーティーな力強さを持つ縦長のコンデンスド書体で、タイトルやスコアボードに最適です。",
        ko="{n}은(는) 스포티한 힘이 느껴지는 길쭉한 콘덴스드 서체로, 제목과 점수판에 잘 어울립니다.",
        pl="{n} to wysoka, wąska czcionka o sportowym charakterze, świetna do tytułów i tablic wyników.",
        pt="{n} é uma fonte alta e condensada de ar esportivo, ótima para títulos e placares.",
        ru="{n} — высокий узкий шрифт со спортивным характером, отлично подходит для заголовков и табло.",
        tr="{n}, spor havasında uzun ve dar bir yazı tipidir; başlıklar ve skor tabloları için idealdir.",
        uk="{n} — високий вузький шрифт зі спортивним характером, чудовий для заголовків і табло.",
        zh="{n}是一款带运动气息的高窄字体，非常适合标题和计分板。",
    ),
    "rounded": dict(
        en="{n} has soft, rounded letters that give your interface a friendly, playful feel.",
        ar="{n} خط بحروف ناعمة ومستديرة يمنح واجهتك طابعًا ودودًا ومرحًا.",
        cs="{n} má měkká zaoblená písmena, která dodají rozhraní přátelský a hravý ráz.",
        de="{n} hat weiche, runde Buchstaben, die der Oberfläche einen freundlichen, verspielten Charakter geben.",
        es="{n} tiene letras suaves y redondeadas que dan a la interfaz un aire amable y divertido.",
        fa="{n} حروفی نرم و گرد دارد که به رابط شما حالتی دوستانه و بازیگوش می‌دهد.",
        fr="{n} a des lettres douces et arrondies qui donnent à l'interface un ton amical et ludique.",
        hi="{n} के कोमल, गोल अक्षर आपके इंटरफ़ेस को मिलनसार और खिलंदड़ा अंदाज़ देते हैं।",
        id="{n} memiliki huruf lembut dan membulat yang membuat antarmuka terasa ramah dan ceria.",
        it="{n} ha lettere morbide e arrotondate che rendono l'interfaccia amichevole e giocosa.",
        ja="{n}は、やわらかな丸みのある文字で、インターフェースに親しみやすく楽しい雰囲気を与えます。",
        ko="{n}은(는) 부드럽고 둥근 글자로 인터페이스에 친근하고 경쾌한 느낌을 더합니다.",
        pl="{n} ma miękkie, zaokrąglone litery, które nadają interfejsowi przyjazny, zabawny charakter.",
        pt="{n} tem letras suaves e arredondadas que dão à interface um ar amigável e divertido.",
        ru="{n} — мягкие округлые буквы, которые придают интерфейсу дружелюбный и игривый вид.",
        tr="{n}, arayüze samimi ve eğlenceli bir hava katan yumuşak, yuvarlak harflere sahiptir.",
        uk="{n} — м'які округлі літери, що надають інтерфейсу дружнього й грайливого вигляду.",
        zh="{n}的字形圆润柔和，让界面显得亲切而活泼。",
    ),
    "comic": dict(
        en="{n} is a lively, comic-style typeface with a hand-lettered feel for fun, casual UI.",
        ar="{n} خط حيوي بطابع القصص المصورة وكأنه مكتوب بخط اليد، مناسب لواجهة مرحة وغير رسمية.",
        cs="{n} je živé písmo v komiksovém stylu s ručně psaným nádechem pro zábavné a nenucené rozhraní.",
        de="{n} ist eine lebhafte Comic-Schrift mit Handschrift-Charme für ein lockeres, spaßiges Interface.",
        es="{n} es una tipografía de cómic, viva y con aire de rotulado a mano, para una interfaz divertida e informal.",
        fa="{n} قلمی پرجنب‌وجوش با سبک کمیک و حس دست‌نویس برای رابطی بامزه و غیررسمی است.",
        fr="{n} est une police de style bande dessinée, vivante et façon lettrage à la main, pour une interface décontractée.",
        hi="{n} कॉमिक शैली का जीवंत, हाथ से लिखे जैसा फ़ॉन्ट है, जो मज़ेदार और अनौपचारिक इंटरफ़ेस के लिए है।",
        id="{n} adalah huruf bergaya komik yang hidup dengan nuansa tulisan tangan untuk antarmuka santai dan seru.",
        it="{n} è un carattere vivace in stile fumetto, con un tocco di lettering a mano, per un'interfaccia informale e divertente.",
        ja="{n}は、手描き風の躍動感あるコミック調書体で、楽しくカジュアルな画面にぴったりです。",
        ko="{n}은(는) 손글씨 느낌의 생동감 있는 만화풍 서체로, 재미있고 캐주얼한 화면에 어울립니다.",
        pl="{n} to żywa czcionka w stylu komiksu z ręcznie pisanym charakterem, dla swobodnego, zabawnego interfejsu.",
        pt="{n} é uma fonte viva de estilo quadrinhos, com jeito de letra feita à mão, para uma interface divertida e descontraída.",
        ru="{n} — живой комиксный шрифт с ощущением рукописных надписей для весёлого неформального интерфейса.",
        tr="{n}, eğlenceli ve rahat bir arayüz için el yazısı havasında, canlı bir çizgi roman yazı tipidir.",
        uk="{n} — живий шрифт у стилі коміксів із відчуттям рукописних написів для веселого неформального інтерфейсу.",
        zh="{n}是一款生动的漫画风字体，带有手写感，适合轻松有趣的界面。",
    ),
    "serif": dict(
        en="{n} is a refined serif face that makes descriptions and long text easy on the eyes.",
        ar="{n} خط سيريف أنيق يجعل الأوصاف والنصوص الطويلة مريحة للعين.",
        cs="{n} je vybroušené patkové písmo, které je příjemné na oči u popisů i delších textů.",
        de="{n} ist eine elegante Serifenschrift, die Beschreibungen und lange Texte angenehm lesbar macht.",
        es="{n} es una tipografía serif refinada que hace cómodas a la vista las descripciones y los textos largos.",
        fa="{n} یک قلم سریف‌دار و ظریف است که توضیحات و متن‌های بلند را برای چشم راحت می‌کند.",
        fr="{n} est une police à empattements raffinée, qui rend les descriptions et les longs textes agréables à lire.",
        hi="{n} एक परिष्कृत सेरिफ़ फ़ॉन्ट है, जो विवरण और लंबे पाठ को आँखों के लिए आरामदायक बनाता है।",
        id="{n} adalah huruf serif yang elegan sehingga deskripsi dan teks panjang nyaman dibaca.",
        it="{n} è un carattere serif raffinato che rende descrizioni e testi lunghi piacevoli da leggere.",
        ja="{n}は、説明文や長い文章を目にやさしく読ませてくれる、洗練されたセリフ体です。",
        ko="{n}은(는) 설명과 긴 글을 눈이 편하게 읽을 수 있게 해 주는 세련된 세리프 서체입니다.",
        pl="{n} to wytworna czcionka szeryfowa, dzięki której opisy i długie teksty czyta się wygodnie.",
        pt="{n} é uma fonte serifada refinada que deixa descrições e textos longos agradáveis à vista.",
        ru="{n} — изысканный шрифт с засечками, делающий описания и длинные тексты приятными для глаз.",
        tr="{n}, açıklamaları ve uzun metinleri göze rahat getiren şık bir tırnaklı yazı tipidir.",
        uk="{n} — вишуканий шрифт із засічками, що робить описи та довгі тексти приємними для очей.",
        zh="{n}是一款精致的衬线字体，让说明和长文本读起来更舒适。",
    ),
    "handwriting": dict(
        en="{n} looks hand-written, adding a personal, notebook-like touch to your interface.",
        ar="{n} خط يبدو مكتوبًا بخط اليد ويضيف لمسة شخصية تشبه دفتر الملاحظات إلى واجهتك.",
        cs="{n} vypadá jako psané rukou a dodá rozhraní osobní nádech poznámkového bloku.",
        de="{n} wirkt handgeschrieben und verleiht deiner Oberfläche eine persönliche Notizbuch-Note.",
        es="{n} parece escrita a mano y da a tu interfaz un toque personal de cuaderno.",
        fa="{n} دست‌نویس به نظر می‌رسد و به رابط شما حسی شخصی و شبیه دفترچه یادداشت می‌دهد.",
        fr="{n} imite l'écriture manuscrite et donne à votre interface une touche personnelle de carnet.",
        hi="{n} हाथ से लिखा हुआ दिखता है और आपके इंटरफ़ेस को नोटबुक जैसा निजी स्पर्श देता है।",
        id="{n} tampak seperti tulisan tangan dan memberi sentuhan pribadi ala buku catatan pada antarmuka Anda.",
        it="{n} sembra scritto a mano e dà alla tua interfaccia un tocco personale da quaderno.",
        ja="{n}は手書きのような書体で、画面にノートのような温かみを加えます。",
        ko="{n}은(는) 손으로 쓴 듯한 서체로 인터페이스에 노트처럼 개성 있는 느낌을 더합니다.",
        pl="{n} wygląda jak pismo odręczne i nadaje interfejsowi osobisty, zeszytowy charakter.",
        pt="{n} parece escrita à mão e dá à sua interface um toque pessoal de caderno.",
        ru="{n} выглядит как рукописный текст и добавляет интерфейсу личный блокнотный штрих.",
        tr="{n}, el yazısı görünümüyle arayüzünüze kişisel, defter benzeri bir dokunuş katar.",
        uk="{n} виглядає як рукописний текст і додає інтерфейсу особистого блокнотного штриха.",
        zh="{n}看起来像手写体，为界面增添笔记本般的个性气质。",
    ),
    "fantasy": dict(
        en="{n} is a fantasy typeface with a medieval, gothic flavour, made for epic game titles.",
        ar="{n} خط خيالي بنكهة قروسطية وقوطية، مصمم لعناوين الألعاب الملحمية.",
        cs="{n} je fantasy písmo s nádechem středověku a gotiky, stvořené pro epické herní názvy.",
        de="{n} ist eine Fantasy-Schrift mit mittelalterlicher, gotischer Note, gemacht für epische Spieltitel.",
        es="{n} es una tipografía de fantasía con sabor medieval y gótico, hecha para títulos de juegos épicos.",
        fa="{n} یک قلم فانتزی با رنگ‌وبوی قرون وسطایی و گوتیک است که برای عنوان بازی‌های حماسی ساخته شده.",
        fr="{n} est une police fantasy aux accents médiévaux et gothiques, faite pour les titres de jeux épiques.",
        hi="{n} मध्ययुगीन और गॉथिक अंदाज़ वाला फ़ैंटेसी फ़ॉन्ट है, जो महाकाव्य गेम शीर्षकों के लिए बना है।",
        id="{n} adalah huruf fantasi bernuansa abad pertengahan dan gotik, cocok untuk judul gim epik.",
        it="{n} è un carattere fantasy dal sapore medievale e gotico, fatto per i titoli di giochi epici.",
        ja="{n}は、中世やゴシックの雰囲気をまとったファンタジー書体で、壮大なゲームタイトルにふさわしい一品です。",
        ko="{n}은(는) 중세와 고딕 분위기를 담은 판타지 서체로, 웅장한 게임 타이틀에 어울립니다.",
        pl="{n} to fantasy czcionka o średniowiecznym, gotyckim sznycie, stworzona do epickich tytułów gier.",
        pt="{n} é uma fonte de fantasia com sabor medieval e gótico, feita para títulos de jogos épicos.",
        ru="{n} — фэнтезийный шрифт со средневековым готическим колоритом, созданный для эпических игровых заголовков.",
        tr="{n}, orta çağ ve gotik havası taşıyan, destansı oyun başlıkları için yapılmış bir fantezi yazı tipidir.",
        uk="{n} — фентезійний шрифт із середньовічним готичним колоритом, створений для епічних ігрових заголовків.",
        zh="{n}是一款带有中世纪与哥特风格的奇幻字体，专为史诗级游戏标题打造。",
    ),
    "display": dict(
        en="{n} is a bold display face with strong personality, made to stand out in headlines.",
        ar="{n} خط عرض عريض بشخصية قوية، صُمم ليبرز في العناوين.",
        cs="{n} je výrazné titulkové písmo se silnou osobností, které vynikne v nadpisech.",
        de="{n} ist eine kräftige Display-Schrift mit starkem Charakter, gemacht für auffällige Überschriften.",
        es="{n} es una tipografía display contundente y con mucha personalidad, hecha para destacar en los titulares.",
        fa="{n} یک قلم نمایشی پررنگ با شخصیتی قوی است که برای دیده‌شدن در عنوان‌ها ساخته شده.",
        fr="{n} est une police d'affichage audacieuse et pleine de personnalité, faite pour se démarquer dans les titres.",
        hi="{n} मज़बूत व्यक्तित्व वाला गहरा डिस्प्ले फ़ॉन्ट है, जो शीर्षकों में अलग से उभरता है।",
        id="{n} adalah huruf display tebal dengan karakter kuat, dibuat agar menonjol di judul.",
        it="{n} è un carattere display deciso e di forte personalità, fatto per distinguersi nei titoli.",
        ja="{n}は、見出しで存在感を放つ、個性の強い太めのディスプレイ書体です。",
        ko="{n}은(는) 개성이 강한 굵은 디스플레이 서체로, 제목에서 돋보이도록 만들어졌습니다.",
        pl="{n} to wyrazista czcionka ozdobna o mocnym charakterze, stworzona, by wyróżniać się w nagłówkach.",
        pt="{n} é uma fonte display marcante e cheia de personalidade, feita para se destacar em títulos.",
        ru="{n} — выразительный акцидентный шрифт с сильным характером, созданный, чтобы выделяться в заголовках.",
        tr="{n}, başlıklarda öne çıkmak için tasarlanmış, güçlü karakterli kalın bir gösteri yazı tipidir.",
        uk="{n} — виразний акцидентний шрифт із сильним характером, створений, щоб вирізнятися в заголовках.",
        zh="{n}是一款个性鲜明的粗体展示字体，专为在标题中脱颖而出而设计。",
    ),
    "retro": dict(
        en="{n} brings a retro arcade and vintage-poster mood to titles and menus.",
        ar="{n} يضفي أجواء صالات الألعاب الكلاسيكية والملصقات القديمة على العناوين والقوائم.",
        cs="{n} dodá nadpisům a menu atmosféru retro arkád a starých plakátů.",
        de="{n} bringt Retro-Arcade- und Vintage-Plakat-Stimmung in Titel und Menüs.",
        es="{n} aporta un aire de arcade retro y cartel vintage a títulos y menús.",
        fa="{n} حال‌وهوای آرکیدهای قدیمی و پوسترهای قدیمی را به عنوان‌ها و منوها می‌آورد.",
        fr="{n} apporte une ambiance de salle d'arcade rétro et d'affiche vintage aux titres et aux menus.",
        hi="{n} शीर्षकों और मेनू में रेट्रो आर्केड और पुराने पोस्टर जैसा माहौल लाता है।",
        id="{n} menghadirkan suasana arkade retro dan poster jadul pada judul dan menu.",
        it="{n} porta atmosfera da sala giochi retrò e da vecchio manifesto in titoli e menu.",
        ja="{n}は、タイトルやメニューにレトロなアーケードとヴィンテージポスターの雰囲気を添えます。",
        ko="{n}은(는) 제목과 메뉴에 레트로 아케이드와 빈티지 포스터의 분위기를 더합니다.",
        pl="{n} wnosi do tytułów i menu klimat retro automatów i dawnych plakatów.",
        pt="{n} traz um clima de fliperama retrô e cartaz vintage para títulos e menus.",
        ru="{n} привносит в заголовки и меню атмосферу ретро-аркад и старых афиш.",
        tr="{n}, başlıklara ve menülere retro atari salonu ile nostaljik afiş havası katar.",
        uk="{n} привносить у заголовки та меню атмосферу ретро-аркад і старих афіш.",
        zh="{n}为标题和菜单带来复古街机与老海报的氛围。",
    ),
    "scifi": dict(
        en="{n} is a futuristic typeface with a technical, sci-fi look for a high-tech interface.",
        ar="{n} خط مستقبلي بمظهر تقني وخيال علمي لواجهة عالية التقنية.",
        cs="{n} je futuristické písmo s technickým sci-fi vzhledem pro high-tech rozhraní.",
        de="{n} ist eine futuristische Schrift im technischen Sci-Fi-Look für eine Hightech-Oberfläche.",
        es="{n} es una tipografía futurista de aspecto técnico y de ciencia ficción para una interfaz de alta tecnología.",
        fa="{n} یک قلم آینده‌نگرانه با ظاهری فنی و علمی-تخیلی برای رابطی پیشرفته است.",
        fr="{n} est une police futuriste à l'allure technique de science-fiction, pour une interface high-tech.",
        hi="{n} तकनीकी, साइंस-फ़िक्शन रूप वाला भविष्यवादी फ़ॉन्ट है, जो हाई-टेक इंटरफ़ेस के लिए है।",
        id="{n} adalah huruf futuristik bergaya teknis dan fiksi ilmiah untuk antarmuka berteknologi tinggi.",
        it="{n} è un carattere futuristico dall'aspetto tecnico e fantascientifico per un'interfaccia hi-tech.",
        ja="{n}は、ハイテクな画面にふさわしい、技術的でSFらしい未来的な書体です。",
        ko="{n}은(는) 첨단 인터페이스에 어울리는 기술적이고 SF 느낌의 미래적인 서체입니다.",
        pl="{n} to futurystyczna czcionka o technicznym, sci-fi wyglądzie, idealna do nowoczesnego interfejsu.",
        pt="{n} é uma fonte futurista de visual técnico e de ficção científica para uma interface de alta tecnologia.",
        ru="{n} — футуристичный шрифт с техничным научно-фантастическим видом для хай-тек интерфейса.",
        tr="{n}, yüksek teknolojili bir arayüz için teknik ve bilim kurgu görünümlü fütüristik bir yazı tipidir.",
        uk="{n} — футуристичний шрифт із технічним науково-фантастичним виглядом для хай-тек інтерфейсу.",
        zh="{n}是一款带有科技与科幻外观的未来感字体，适合高科技风格的界面。",
    ),
    "accessibility": dict(
        en="{n} is designed for clear reading: distinct letter shapes that stay legible at any size.",
        ar="{n} مصمم للقراءة الواضحة بأشكال حروف متمايزة تبقى مقروءة بأي حجم.",
        cs="{n} je navržené pro jasné čtení: výrazné tvary písmen zůstávají čitelné v každé velikosti.",
        de="{n} ist für klares Lesen gestaltet: unterscheidbare Buchstabenformen, die in jeder Größe lesbar bleiben.",
        es="{n} está diseñada para leer con claridad: formas de letra distintas que se leen bien en cualquier tamaño.",
        fa="{n} برای خواندن آسان طراحی شده است: شکل حروف متمایز که در هر اندازه خوانا می‌ماند.",
        fr="{n} est conçue pour une lecture claire : des formes de lettres distinctes, lisibles à toutes les tailles.",
        hi="{n} स्पष्ट पढ़ने के लिए बना है: अलग-अलग अक्षर-आकृतियाँ, जो हर आकार में पढ़ने लायक रहती हैं।",
        id="{n} dirancang untuk keterbacaan jelas: bentuk huruf yang berbeda dan tetap terbaca di ukuran apa pun.",
        it="{n} è pensato per una lettura chiara: forme delle lettere distinte, leggibili a qualsiasi dimensione.",
        ja="{n}は読みやすさを重視した書体で、字形がはっきり区別でき、どんなサイズでも読みやすさを保ちます。",
        ko="{n}은(는) 또렷한 가독성을 위해 설계되어, 구별되는 글자 모양이 어떤 크기에서도 잘 읽힙니다.",
        pl="{n} został zaprojektowany z myślą o czytelności: wyraźne kształty liter są czytelne w każdym rozmiarze.",
        pt="{n} foi desenhada para leitura clara: formas de letra distintas que continuam legíveis em qualquer tamanho.",
        ru="{n} создан для чёткого чтения: различимые формы букв остаются разборчивыми при любом размере.",
        tr="{n}, net okunabilirlik için tasarlandı: ayırt edilebilir harf biçimleri her boyutta okunaklı kalır.",
        uk="{n} створено для чіткого читання: розрізнювані форми літер залишаються розбірливими за будь-якого розміру.",
        zh="{n}专为清晰阅读而设计，字形区分明显，任何字号下都清晰易读。",
    ),
}

SCRIPT_ORDER = ["latin", "latin-ext", "cyrillic", "greek", "vietnamese", "arabic", "devanagari", "cjk"]
LATIN = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
LATIN_EXT = "ĀāĂăĄąĆćČčĎďĐđĒēĘęĚěĞğĪīİıŁłŃńŇňŌōŐőŒœŘřŚśŞşŠšŤťŪūŮůŰűŹźŻżŽž"
CYR_BASE = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
CYR_UK = "ІіЇїЄєҐґ"
GREEK = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩαβγδεζηθικλμνξοπρστυφχψω"
VIET = "ĂăÂâÊêÔôƠơƯưĐđẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ"
KANJI = "一二三四五六七八九十日月火水木金土人大小中国学生年"
HIRAGANA = "".join(chr(c) for c in range(0x3042, 0x3094))


def slug(family):
    return re.sub(r"[^a-z0-9]+", "-", family.lower()).strip("-")


def font_id(family):
    return "font-" + slug(family)


def source_url(directory, lic, file):
    return f"{RAW}/{lic}/{directory}/{urllib.parse.quote(file)}"


# ====================================================================== fetch
def fetch_one(src):
    family, directory, lic, file = src
    folder = CACHE / font_id(family)
    target = folder / "font.ttf"
    lic_target = folder / "LICENSE.txt"
    if target.exists() and lic_target.exists():
        return family, "have", target.stat().st_size
    folder.mkdir(parents=True, exist_ok=True)
    try:
        font = _get(source_url(directory, lic, file))
        if font[:4] not in (b"\x00\x01\x00\x00", b"true", b"OTTO"):
            return family, "BAD-MAGIC", 0
        text = None
        for name in ("OFL.txt", "LICENSE.txt"):
            try:
                text = _get(f"{RAW}/{lic}/{directory}/{name}")
                break
            except urllib.error.HTTPError:
                continue
        if text is None:
            return family, "NO-LICENSE", 0
        target.write_bytes(font)
        lic_target.write_bytes(text)
        return family, "fetched", len(font)
    except Exception as exc:  # network errors
        return family, f"ERR {exc}", 0


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "font-catalog-builder"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


# ====================================================================== font analysis
def analyse(path):
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.ttLib import TTFont
    tt = TTFont(str(path), lazy=True)
    cmap = tt.getBestCmap() or {}
    have = set(cmap)

    def frac(chars):
        return sum(1 for c in chars if ord(c) in have) / len(chars)

    scripts = []
    if frac(LATIN) == 1.0:
        scripts.append("latin")
    if frac(LATIN_EXT) >= 0.9:
        scripts.append("latin-ext")
    partial_cyrillic = False
    if frac(CYR_BASE) == 1.0:
        if frac(CYR_UK) == 1.0:
            scripts.append("cyrillic")
        else:
            partial_cyrillic = True
    if frac(GREEK) >= 0.98:
        scripts.append("greek")
    if frac(VIET) >= 0.95:
        scripts.append("vietnamese")
    arabic = "".join(chr(c) for c in range(0x0627, 0x064B))
    if frac(arabic) >= 0.75:
        scripts.append("arabic")
    deva = "".join(chr(c) for c in range(0x0915, 0x093A))
    if frac(deva) >= 0.8:
        scripts.append("devanagari")
    if frac(HIRAGANA) >= 0.8 and frac(KANJI) >= 0.8:
        scripts.append("cjk")
    upm = tt["head"].unitsPerEm
    scale = 1.0
    try:
        gs = tt.getGlyphSet()
        pen = BoundsPen(gs)
        gs[cmap[ord("x")]].draw(pen)
        if pen.bounds:
            ratio = pen.bounds[3] / upm
            if ratio > 0.05:
                scale = max(0.9, min(1.25, round(0.52 / ratio / 0.05) * 0.05))
                scale = round(scale, 2)
    except Exception:
        scale = 1.0
    axes = [(a.axisTag, a.minValue, a.defaultValue, a.maxValue) for a in tt["fvar"].axes] if "fvar" in tt else []
    return {"cmap": have, "scripts": scripts, "partial_cyrillic": partial_cyrillic, "sizeScale": scale, "axes": axes}


class Face:
    def __init__(self, path, cmap, axes=()):
        self.path = str(path)
        self.cmap = cmap
        self.axes = axes
        self._cache = {}

    def at(self, size):
        size = max(6, int(round(size)))
        if size not in self._cache:
            font = ImageFont.truetype(self.path, size)
            if self.axes:  # variable font: render the Regular (wght 400) instance, other axes at their defaults
                font.set_variation_by_axes([min(max(400, lo), hi) if tag == "wght" else dflt for tag, lo, dflt, hi in self.axes])
            self._cache[size] = font
        return self._cache[size]

    def ok(self, text):
        return "".join(ch for ch in text if ch == " " or ord(ch) in self.cmap)

    def width(self, text, size):
        return self.at(size).getlength(text)

    def bbox(self, text, size):
        return self.at(size).getbbox(text, anchor="ls")


# ====================================================================== colours / backdrops
def hexc(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mix(a, b, t):
    return tuple(int(round(a[i] * (1 - t) + b[i] * t)) for i in range(3))


def lum(c):
    return (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255.0


def hue_shift(c, dh, ds=1.0, dv=1.0):
    h, s, v = colorsys.rgb_to_hsv(*(x / 255.0 for x in c))
    r, g, b = colorsys.hsv_to_rgb((h + dh) % 1.0, min(1, s * ds), min(1, v * dv))
    return (int(r * 255), int(g * 255), int(b * 255))


# (bg0, bg1, ink, sub, accent)
PALETTES = {
    "pixgrid": [("#0B1026", "#162150", "#EAF2FF", "#8FA3D6", "#35F0FF"), ("#1A0B2E", "#2E1458", "#FFEAF6", "#C6A5E8", "#FF4FB0"),
                ("#06140A", "#0F2E18", "#E1FFE8", "#8FD6A0", "#5CFF7A"), ("#240A0A", "#471616", "#FFF0E0", "#E0A890", "#FFB02E")],
    "terminal": [("#020A04", "#06180A", "#6DFF8A", "#3FB05B", "#6DFF8A"), ("#0C0700", "#1B1000", "#FFB547", "#B9812C", "#FFB547")],
    "paper": [("#F6EEDC", "#EADFC4", "#2B2118", "#6E5E4A", "#B5492E"), ("#F3F1EA", "#E4E0D3", "#1F2430", "#5B6270", "#2F6F8F"),
              ("#FAF4E8", "#EFE3CC", "#3A2418", "#7A5A44", "#9C2F2F"), ("#F1ECE6", "#DDD2C6", "#2A1E2E", "#6A5A70", "#7A3E9D")],
    "blueprint": [("#0F3B7A", "#0A2A5C", "#F2F7FF", "#A9C6F2", "#7FE3FF"), ("#123A4A", "#0A2A36", "#EAFBFF", "#9CCBD8", "#FFD166"),
                  ("#1B2F6B", "#101E4C", "#FFFFFF", "#B1BCF0", "#FF8FA3")],
    "neon": [("#07040F", "#160A30", "#E7FBFF", "#8E9BD9", "#27E8FF"), ("#0A0414", "#220A3A", "#FFE6FA", "#C79BE0", "#FF3DB8"),
             ("#030B10", "#052A2E", "#E6FFF7", "#7ECDB7", "#3DFFB0"), ("#0E0410", "#2B0B2A", "#FFF2E0", "#E0A9C7", "#FF8A3D")],
    "synth": [("#14022E", "#7A1B8F", "#FFF3FF", "#F3B6E8", "#FF4FA8"), ("#07062B", "#36146B", "#F0F6FF", "#B7C1FF", "#2FE0FF"),
              ("#1B0524", "#B8326A", "#FFF6E8", "#FFC6B0", "#FFD43D"), ("#0A0A2E", "#1E5AA8", "#FFFFFF", "#BFE0FF", "#FF6FD8")],
    "slate": [("#10141C", "#1D2430", "#F1F5FB", "#93A0B5", "#4F8CFF"), ("#121014", "#241F2E", "#F6F2FB", "#A99DBB", "#B77CFF"),
              ("#0F1714", "#1B2B25", "#EEF8F3", "#8DAA9E", "#2FD6A0"), ("#171210", "#2B2320", "#FBF3EE", "#B8A59B", "#FF7A45")],
    "light": [("#F6F8FC", "#E6ECF7", "#121826", "#5C6779", "#3563E9"), ("#FBF7F4", "#F1E6DD", "#2A1D17", "#76645A", "#E0552B"),
              ("#F4FAF7", "#DDF0E8", "#0F2A21", "#4F6E62", "#12A474"), ("#F9F5FC", "#EADFF3", "#25143A", "#6F5E86", "#8E4FD6")],
    "pastel": [("#FFE9F1", "#FFD0E2", "#4B2A57", "#8C6296", "#FF6FA8"), ("#E6F6FF", "#CBE8FF", "#1F3A5C", "#5F7FA3", "#3FA3FF"),
               ("#FFF4D9", "#FFE3A8", "#4A3320", "#8D7050", "#FF9A3D"), ("#EAFBE8", "#CDEFC8", "#1F4A2A", "#5E8A66", "#3DBE5A")],
    "halftone": [("#FFE24A", "#FFC400", "#16121C", "#5A4A1F", "#E5322D"), ("#FFB3D1", "#FF7FB2", "#1E1030", "#5A2A4A", "#2E5BFF"),
                 ("#9BE8FF", "#4FC3F7", "#10162B", "#1F4A66", "#FF3D3D")],
    "notebook": [("#FBFBF4", "#F1F1E4", "#1E2A5A", "#6A74A8", "#D8404B"), ("#FFFDF2", "#F7F1D5", "#2A2A2A", "#7A7A5A", "#2C7DE0")],
    "parchment": [("#E9D7A8", "#C9A96B", "#3A2410", "#7A5A32", "#8C2B1E"), ("#EBDCB7", "#CFB680", "#2D1E14", "#6E5434", "#2E5A44"),
                  ("#F0E2BE", "#D7BE86", "#24150E", "#6B4C2B", "#5B2C83")],
    "stone": [("#120B0B", "#2A1414", "#F2E2B8", "#B09A6A", "#D4A63A"), ("#0B0F14", "#18232E", "#E8EEF5", "#8FA2B8", "#C9B27A"),
              ("#100A14", "#271634", "#F3E6FF", "#B093CC", "#D9B45A"), ("#0D120C", "#1D2B1A", "#EAF2D8", "#98AE80", "#C9A24A")],
    "stripes": [("#101318", "#1B2028", "#FFFFFF", "#9AA6B8", "#FFC400"), ("#0A1424", "#10284A", "#FFFFFF", "#9FB7DA", "#2FD0FF"),
                ("#180A0A", "#331212", "#FFFFFF", "#D6A0A0", "#FF3D3D"), ("#081A12", "#0F3524", "#FFFFFF", "#9FD1B8", "#7CFF4F")],
    "poster": [("#FFB347", "#E8452C", "#2A0F08", "#5C2412", "#FFF1C9"), ("#FF9E80", "#C2185B", "#FFF6E5", "#FFD2C0", "#FFD54A"),
               ("#FFD28A", "#2A9D8F", "#1F1A14", "#4F3F2C", "#E76F51"), ("#F6D7A7", "#D2691E", "#2B1608", "#6B3F1D", "#7A1F1F")],
    "editor": [("#1E1F29", "#282A36", "#F8F8F2", "#8A8FA8", "#50FA7B"), ("#1A1B26", "#24283B", "#C0CAF5", "#7A88B8", "#7AA2F7"),
               ("#272822", "#33342B", "#F8F8F2", "#9A9A88", "#A6E22E"), ("#002B36", "#073642", "#EAF2F2", "#7FA0A8", "#B58900")],
    "contrast": [("#000000", "#141414", "#FFFFFF", "#D8D8D8", "#FFD60A"), ("#FFFFFF", "#EDEDED", "#000000", "#303030", "#0057D8"),
                 ("#0A0A14", "#0F1030", "#FFFFFF", "#E0E0FF", "#00E5FF")],
    "chalk": [("#173A2E", "#0F2A22", "#F4F7F2", "#A9C4B8", "#FFD966"), ("#1C2B3A", "#12202D", "#F2F6FA", "#A6B8CC", "#FF9AA8")],
}
FX = {"pixgrid": "hard", "terminal": "glow", "neon": "glow", "synth": "shadow", "halftone": "outline", "poster": "shadow", "stripes": None}

W, H = 1280, 720


class Theme:
    def __init__(self, spec):
        name, var = spec.split(":")
        pal = PALETTES[name][int(var) % len(PALETTES[name])]
        self.name = name
        self.bg0, self.bg1, self.ink, self.sub, self.accent = (hexc(c) for c in pal)
        self.dark = lum(self.bg0) < 0.5 if name not in ("synth", "poster") else lum(self.ink) > 0.6
        base = self.bg1
        if self.dark:
            self.card = mix(base, self.ink, 0.10)
        else:
            self.card = mix(self.bg0, (255, 255, 255), 0.6)
        if name in ("poster", "synth"):
            self.card = mix(self.bg0, (0, 0, 0), 0.55) if name == "synth" else mix((255, 255, 255), self.bg0, 0.25)
            self.cardInk = self.ink if name == "synth" else self.ink
        self.fx = FX.get(name)

    @property
    def panel_ink(self):
        return self.ink


def grad(w, h, c0, c1, angle=90.0):
    a = math.radians(angle)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    t = (xs / max(1, w - 1)) * math.cos(a) + (ys / max(1, h - 1)) * math.sin(a)
    t = (t - t.min()) / max(1e-6, t.max() - t.min())
    c0 = np.array(c0, np.float32)
    c1 = np.array(c1, np.float32)
    arr = c0[None, None, :] * (1 - t[..., None]) + c1[None, None, :] * t[..., None]
    return Image.fromarray(arr.astype(np.uint8), "RGB").convert("RGBA")


def overlay(im, fn):
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    fn(ImageDraw.Draw(layer))
    im.alpha_composite(layer)


def radial(im, cx, cy, r, color, k=1.0):
    ys, xs = np.mgrid[0:im.size[1], 0:im.size[0]].astype(np.float32)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / r
    a = np.clip(1 - d, 0, 1) ** 2 * k
    layer = np.zeros((im.size[1], im.size[0], 4), np.uint8)
    layer[..., 0], layer[..., 1], layer[..., 2] = color
    layer[..., 3] = (a * 255).astype(np.uint8)
    im.alpha_composite(Image.fromarray(layer, "RGBA"))


def grain(im, rng, amount=6):
    arr = np.asarray(im.convert("RGB"), np.int16)
    n = rng.integers(-amount, amount + 1, size=arr.shape[:2])[..., None]
    out = np.clip(arr + n, 0, 255).astype(np.uint8)
    res = Image.fromarray(out, "RGB").convert("RGBA")
    im.paste(res)


def vignette(im, k=0.5):
    w, h = im.size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xs - w / 2) / (w / 2)) ** 2 + ((ys - h / 2) / (h / 2)) ** 2)
    a = np.clip((d - 0.55) / 0.9, 0, 1) ** 1.6 * k
    layer = np.zeros((h, w, 4), np.uint8)
    layer[..., 3] = (a * 255).astype(np.uint8)
    im.alpha_composite(Image.fromarray(layer, "RGBA"))


def backdrop(theme, rng, w=W, h=H):
    n = theme.name
    t = theme
    if n == "pixgrid":
        im = grad(w, h, t.bg0, t.bg1, 70)

        def deco(d):
            line = t.accent + (26,)
            for x in range(0, w, 40):
                d.line([(x, 0), (x, h)], fill=line, width=1)
            for y in range(0, h, 40):
                d.line([(0, y), (w, y)], fill=line, width=1)
            for _ in range(46):
                x, y = int(rng.integers(0, w // 8)) * 8, int(rng.integers(0, h // 8)) * 8
                s = int(rng.choice([8, 8, 16]))
                d.rectangle([x, y, x + s, y + s], fill=t.accent + (int(rng.integers(40, 150)),))
            d.rectangle([0, h - 16, w, h], fill=t.accent + (170,))
        overlay(im, deco)
    elif n == "terminal":
        im = grad(w, h, t.bg1, t.bg0, 90)
        radial(im, w / 2, h / 2, w * 0.7, t.accent, 0.16)

        def deco(d):
            for y in range(0, h, 4):
                d.line([(0, y), (w, y)], fill=(0, 0, 0, 70), width=2)
            d.rectangle([24, 24, w - 24, h - 24], outline=t.accent + (70,), width=2)
        overlay(im, deco)
        vignette(im, 0.6)
    elif n in ("paper", "parchment"):
        im = grad(w, h, t.bg0, t.bg1, 60)
        grain(im, rng, 7)
        if n == "parchment":
            radial(im, w / 2, h / 2, w * 0.75, mix(t.bg0, (255, 255, 255), 0.4), 0.5)
            vignette(im, 0.75)

        def deco(d):
            d.rectangle([34, 34, w - 34, h - 34], outline=t.sub + (120,), width=3)
            d.rectangle([46, 46, w - 46, h - 46], outline=t.sub + (70,), width=1)
        overlay(im, deco)
    elif n == "blueprint":
        im = grad(w, h, t.bg0, t.bg1, 80)

        def deco(d):
            for x in range(0, w, 32):
                d.line([(x, 0), (x, h)], fill=(255, 255, 255, 18 if x % 160 else 46), width=1)
            for y in range(0, h, 32):
                d.line([(0, y), (w, y)], fill=(255, 255, 255, 18 if y % 160 else 46), width=1)
            d.ellipse([w - 330, h - 330, w - 30, h - 30], outline=t.accent + (90,), width=2)
            d.line([(w - 180, h - 360), (w - 180, h)], fill=t.accent + (60,), width=1)
            for cx, cy in ((40, 40), (w - 40, 40), (40, h - 40), (w - 40, h - 40)):
                d.line([(cx - 16, cy), (cx + 16, cy)], fill=t.ink + (160,), width=2)
                d.line([(cx, cy - 16), (cx, cy + 16)], fill=t.ink + (160,), width=2)
        overlay(im, deco)
    elif n == "neon":
        im = grad(w, h, t.bg0, t.bg1, 120)
        radial(im, w * 0.2, h * 0.9, w * 0.6, t.accent, 0.5)
        radial(im, w * 0.85, h * 0.1, w * 0.55, hue_shift(t.accent, 0.38), 0.45)

        def deco(d):
            d.rounded_rectangle([40, 40, w - 40, h - 40], radius=26, outline=t.accent + (200,), width=4)
            d.rounded_rectangle([54, 54, w - 54, h - 54], radius=18, outline=hue_shift(t.accent, 0.38) + (110,), width=2)
        overlay(im, deco)
    elif n == "synth":
        im = grad(w, h, t.bg0, t.bg1, 90)
        hz = int(h * 0.76)
        sun_c = (w // 2, int(h * 0.50))
        sun = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sd = ImageDraw.Draw(sun)
        r = 190
        for i in range(r * 2):
            y = sun_c[1] - r + i
            f = i / (2 * r)
            col = mix(t.accent, hue_shift(t.accent, 0.12), f) + (255,)
            half = math.sqrt(max(0, r * r - (i - r) ** 2))
            sd.line([(sun_c[0] - half, y), (sun_c[0] + half, y)], fill=col)
        mask = Image.new("L", (w, h), 255)
        md = ImageDraw.Draw(mask)
        for k in range(8):
            y0 = sun_c[1] + 10 + k * 26
            md.rectangle([0, y0, w, y0 + 3 + k * 2], fill=0)
        sun.putalpha(Image.composite(sun.getchannel("A"), Image.new("L", (w, h), 0), mask).point(lambda v: int(v * 0.5)))
        im.alpha_composite(sun)

        def deco(d):
            d.rectangle([0, hz, w, h], fill=mix(t.bg0, (0, 0, 0), 0.55) + (255,))
            d.line([(0, hz), (w, hz)], fill=t.accent + (255,), width=3)
            for i in range(-12, 13):
                d.line([(w / 2 + i * 22, hz), (w / 2 + i * 150, h)], fill=t.accent + (130,), width=2)
            y, step = hz, 8
            while y < h:
                y += step
                step *= 1.35
                d.line([(0, y), (w, y)], fill=t.accent + (110,), width=2)
        overlay(im, deco)
    elif n == "slate":
        im = grad(w, h, t.bg0, t.bg1, 35)
        radial(im, w * 0.85, h * 0.2, w * 0.45, t.accent, 0.22)
        radial(im, w * 0.1, h * 0.95, w * 0.4, hue_shift(t.accent, 0.1), 0.14)
        overlay(im, lambda d: d.rectangle([0, 0, 10, h], fill=t.accent + (255,)))
    elif n == "light":
        im = grad(w, h, t.bg0, t.bg1, 40)
        radial(im, w * 0.9, h * 0.1, w * 0.5, t.accent, 0.20)
        radial(im, w * 0.05, h * 0.9, w * 0.4, hue_shift(t.accent, 0.12), 0.16)
        overlay(im, lambda d: d.rectangle([0, 0, 10, h], fill=t.accent + (255,)))
    elif n == "pastel":
        im = grad(w, h, t.bg0, t.bg1, 50)

        def deco(d):
            for _ in range(26):
                x, y, r = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(18, 70))
                c = hue_shift(t.accent, float(rng.random()) * 0.3 - 0.15)
                d.ellipse([x - r, y - r, x + r, y + r], fill=c + (int(rng.integers(30, 80)),))
            d.rounded_rectangle([36, 36, w - 36, h - 36], radius=48, outline=t.accent + (160,), width=5)
        overlay(im, deco)
    elif n == "halftone":
        im = grad(w, h, t.bg0, t.bg1, 60)

        def deco(d):
            step = 22
            for yi in range(0, h // step + 2):
                for xi in range(0, w // step + 2):
                    x, y = xi * step + (step // 2 if yi % 2 else 0), yi * step
                    f = max(0.0, 1 - math.hypot(x - w, y - h) / (w * 0.75))
                    r = 9 * f
                    if r > 0.8:
                        d.ellipse([x - r, y - r, x + r, y + r], fill=t.accent + (200,))
            d.rectangle([14, 14, w - 14, h - 14], outline=t.ink + (255,), width=10)
        overlay(im, deco)
    elif n == "notebook":
        im = grad(w, h, t.bg0, t.bg1, 90)

        def deco(d):
            for y in range(120, h, 46):
                d.line([(0, y), (w, y)], fill=(80, 130, 210, 110), width=2)
            d.line([(110, 0), (110, h)], fill=t.accent + (170,), width=3)
            for y in range(70, h, 140):
                d.ellipse([34, y, 60, y + 26], fill=(0, 0, 0, 55))
        overlay(im, deco)
        grain(im, rng, 4)
    elif n == "stone":
        im = grad(w, h, t.bg1, t.bg0, 90)
        radial(im, w / 2, h * 0.45, w * 0.6, t.accent, 0.12)
        grain(im, rng, 5)
        vignette(im, 0.65)

        def deco(d):
            d.rectangle([30, 30, w - 30, h - 30], outline=t.accent + (230,), width=4)
            d.rectangle([44, 44, w - 44, h - 44], outline=t.accent + (110,), width=2)
            for cx, cy in ((30, 30), (w - 30, 30), (30, h - 30), (w - 30, h - 30)):
                d.polygon([(cx, cy - 20), (cx + 20, cy), (cx, cy + 20), (cx - 20, cy)], fill=t.accent + (255,))
        overlay(im, deco)
    elif n == "stripes":
        im = grad(w, h, t.bg0, t.bg1, 20)

        def deco(d):
            for i in range(-2, 5):
                x = w * 0.9 + i * 62
                d.polygon([(x, 0), (x + 30, 0), (x - 120 + 30, h), (x - 120, h)], fill=t.accent + (255 if i % 3 == 0 else 46,))
            d.rectangle([0, h - 18, w, h], fill=t.accent + (255,))
        overlay(im, deco)
    elif n == "poster":
        im = grad(w, h, t.bg0, t.bg1, 90)

        def deco(d):
            cx, cy = w / 2, h * 1.05
            for i in range(36):
                a0 = math.pi + i * math.pi / 36
                a1 = a0 + math.pi / 72
                d.polygon([(cx, cy), (cx + 1800 * math.cos(a0), cy + 1800 * math.sin(a0)), (cx + 1800 * math.cos(a1), cy + 1800 * math.sin(a1))],
                          fill=(255, 255, 255, 36))
            d.rectangle([22, 22, w - 22, h - 22], outline=t.ink + (230,), width=8)
        overlay(im, deco)
        grain(im, rng, 5)
    elif n == "editor":
        im = grad(w, h, t.bg0, t.bg1, 90)

        def deco(d):
            d.rectangle([0, 0, 96, h], fill=(0, 0, 0, 55))
            y = 40
            for i in range(1, 16):
                d.rectangle([40, y + 6, 40 + 18 + (i % 3) * 6, y + 20], fill=t.sub + (120,))
                x = 130 + (i % 4) * 36
                for _ in range(int(rng.integers(2, 5))):
                    wd = int(rng.integers(40, 180))
                    c = [t.accent, hue_shift(t.accent, 0.25), hue_shift(t.accent, 0.55), t.sub][int(rng.integers(0, 4))]
                    d.rounded_rectangle([x, y + 4, x + wd, y + 22], radius=5, fill=c + (42,))
                    x += wd + 16
                y += 42
        overlay(im, deco)
    elif n == "contrast":
        im = grad(w, h, t.bg0, t.bg0, 0)
        overlay(im, lambda d: (d.rectangle([0, h - 28, w, h], fill=t.accent + (255,)), d.rectangle([0, 0, 28, h], fill=t.accent + (255,))))
    elif n == "chalk":
        im = grad(w, h, t.bg0, t.bg1, 70)
        grain(im, rng, 9)

        def deco(d):
            for _ in range(14):
                x, y, rr = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(60, 220))
                d.ellipse([x - rr, y - rr // 2, x + rr, y + rr // 2], fill=(255, 255, 255, 8))
            d.rectangle([0, 0, w, h], outline=(120, 84, 48, 255), width=26)
        overlay(im, deco)
    else:
        raise ValueError(n)
    return im


# ====================================================================== text helpers
def put_text(im, face, xy, text, size, fill, anchor="ms", fx=None, accent=None, ink_stroke=None):
    """Draw text at baseline-anchored xy using the font's own glyphs."""
    text = face.ok(text)
    if not text.strip():
        return
    font = face.at(size)
    x, y = xy
    if fx == "glow":
        glow = Image.new("L", im.size, 0)
        ImageDraw.Draw(glow).text((x, y), text, font=font, fill=255, anchor=anchor)
        for rad, k in ((22, 0.9), (8, 1.0)):
            g = glow.filter(ImageFilter.GaussianBlur(rad)).point(lambda v, k=k: min(255, int(v * 2.2 * k)))
            layer = Image.new("RGBA", im.size, (accent or fill) + (0,))
            layer.putalpha(g)
            im.alpha_composite(layer)
    d = ImageDraw.Draw(im)
    if fx == "shadow":
        off = max(3, int(size * 0.045))
        d.text((x + off, y + off), text, font=font, fill=(accent or fill), anchor=anchor)
    elif fx == "hard":
        off = max(3, int(size * 0.06))
        d.text((x + off, y + off), text, font=font, fill=(accent or fill), anchor=anchor)
    elif fx == "outline":
        off = max(4, int(size * 0.05))
        d.text((x + off, y + off), text, font=font, fill=(accent or fill), anchor=anchor)
        d.text((x, y), text, font=font, fill=(255, 255, 255), anchor=anchor, stroke_width=max(3, int(size * 0.035)), stroke_fill=(255, 255, 255))
    d.text((x, y), text, font=font, fill=fill, anchor=anchor)


def text_extent(face, text, size):
    text = face.ok(text)
    b = face.bbox(text, size)
    return b[2] - b[0], b[3] - b[1], b


def fit_size(face, text, max_w, max_h, hi, lo=14):
    size = hi
    while size > lo:
        w_, h_, _ = text_extent(face, text, size)
        if w_ <= max_w and h_ <= max_h:
            return size
        size = int(size * 0.94)
    return lo


def wrap_words(face, text, size, max_w):
    words = face.ok(text).split()
    lines, cur = [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        if face.width(trial, size) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def center_baseline(face, text, size, cy):
    """Baseline y that vertically centres the ink of text on cy."""
    b = face.bbox(face.ok(text) or "H", size)
    return cy - (b[1] + b[3]) / 2


def sample_line(face, scripts):
    parts = ["Aa"]
    if "cyrillic" in scripts:
        parts.append("Бб Жж")
    elif "greek" in scripts:
        parts.append("Ωω")
    parts.append("0123")
    line = " ".join(parts)
    for extra in ("★", "♥"):
        if ord(extra) in face.cmap:
            line += " " + extra
            break
    return face.ok(line)


# ====================================================================== images
def render_cover(face, family, scripts, theme, seed):
    rng = np.random.default_rng(seed)
    im = backdrop(theme, rng)
    margin = 110
    max_w = W - 2 * margin
    sample = sample_line(face, scripts)
    # name: single line if big enough, else two lines
    size1 = fit_size(face, family, max_w, 250, 230)
    lines = [family]
    size = size1
    if size1 < 120 and " " in family:
        words = family.split()
        best = None
        for k in range(1, len(words)):
            a, b = " ".join(words[:k]), " ".join(words[k:])
            s = min(fit_size(face, a, max_w, 190, 200), fit_size(face, b, max_w, 190, 200))
            if best is None or s > best[0]:
                best = (s, [a, b])
        if best and best[0] > size1 * 1.15:
            size, lines = best
    sample_size = fit_size(face, sample, max_w, 110, min(int(size * 0.62), 108))
    gap = 56
    line_h = [text_extent(face, ln, size)[1] for ln in lines]
    sh = text_extent(face, sample, sample_size)[1]
    total = sum(line_h) + (len(lines) - 1) * int(size * 0.28) + gap + sh
    y = (H - total) / 2
    for ln, lh in zip(lines, line_h):
        b = face.bbox(face.ok(ln), size)
        base = y - b[1]
        put_text(im, face, (W / 2, base), ln, size, theme.ink, "ms", theme.fx, theme.accent)
        y += lh + int(size * 0.28)
    y += gap - int(size * 0.28)
    b = face.bbox(sample, sample_size)
    base = y - b[1]
    sub = theme.accent if theme.name in ("terminal", "contrast", "editor", "pixgrid", "neon") else theme.sub
    put_text(im, face, (W / 2, base), sample, sample_size, sub, "ms")
    return im.convert("RGB").resize((640, 360), Image.LANCZOS)


def rounded_mask(size, radius):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=radius, fill=255)
    return m


def card_art(size, accent, kind, rng):
    w, h = size
    c0 = hue_shift(accent, [0.0, 0.62, 0.78, 0.08][kind], 0.9, 0.55)
    c1 = hue_shift(accent, [0.1, 0.72, 0.9, 0.0][kind], 1.0, 1.0)
    art = grad(w, h, c0, c1, 75)

    def deco(d):
        if kind == 0:  # crystal ruins
            for i in range(5):
                x = 30 + i * 50
                top = 40 + (i % 3) * 22
                d.polygon([(x, h), (x + 22, top), (x + 44, h)], fill=(255, 255, 255, 70))
                d.polygon([(x + 22, top), (x + 44, h), (x + 22, h)], fill=(0, 0, 0, 50))
            d.ellipse([w - 90, 18, w - 40, 68], fill=(255, 255, 255, 140))
        elif kind == 1:  # neon drift
            for i in range(7):
                d.polygon([(i * 46, h), (i * 46 + 22, h), (w / 2 + (i - 3) * 6 + 3, h * 0.45), (w / 2 + (i - 3) * 6 - 3, h * 0.45)], fill=(255, 255, 255, 60))
            d.line([(0, h * 0.45), (w, h * 0.45)], fill=(255, 255, 255, 200), width=3)
            d.ellipse([w / 2 - 28, h * 0.45 - 28, w / 2 + 28, h * 0.45 + 28], fill=(255, 255, 255, 110))
        elif kind == 2:  # star harbor
            for _ in range(26):
                x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
                d.ellipse([x, y, x + 3, y + 3], fill=(255, 255, 255, 200))
            d.ellipse([w - 120, h - 110, w + 30, h + 40], fill=(255, 255, 255, 90))
            d.ellipse([30, 40, 90, 100], fill=(255, 255, 255, 120))
            d.arc([-40, 90, w + 40, h + 120], 200, 340, fill=(255, 255, 255, 150), width=3)
        else:  # moonlit tavern
            d.ellipse([w - 110, 20, w - 40, 90], fill=(255, 250, 220, 230))
            d.polygon([(0, h), (0, h - 60), (70, h - 90), (150, h - 50), (w, h - 80), (w, h)], fill=(0, 0, 0, 90))
            d.rectangle([50, h - 80, 110, h - 40], fill=(0, 0, 0, 130))
            d.rectangle([68, h - 66, 80, h - 52], fill=(255, 220, 140, 230))
    overlay(art, deco)
    return art


def render_gallery_library(face, scripts, theme, seed):
    rng = np.random.default_rng(seed + 1)
    im = backdrop(theme, rng)
    ink, sub, accent, card = theme.ink, theme.sub, theme.accent, theme.card
    card_ink = ink if (theme.dark or theme.name in ("poster",)) and lum(card) < 0.55 else (theme.ink if lum(card) < 0.5 else (20, 20, 20) if lum(theme.ink) > 0.5 else theme.ink)
    if lum(card) >= 0.5 and lum(ink) > 0.5:
        card_ink = (24, 24, 30)
    elif lum(card) < 0.5 and lum(ink) <= 0.5:
        card_ink = (245, 245, 245)
    else:
        card_ink = ink
    card_sub = mix(card_ink, card, 0.45)
    # panel behind the UI so any backdrop stays readable
    overlay(im, lambda d: d.rounded_rectangle([34, 34, W - 34, H - 34], radius=34, fill=mix(theme.bg0, theme.bg1, 0.5) + (222,)))
    put_text(im, face, (84, 118), "Library", fit_size(face, "Library", 400, 70, 66), ink, "ls")
    put_text(im, face, (W - 84, 108), "Search games", 26, sub, "rs")
    titles = [("Crystal Ruins", "Adventure - 24 h"), ("Neon Drift", "Racing - 8 h"), ("Star Harbor", "Space - 41 h"), ("Moonlit Tavern", "Puzzle - 6 h")]
    cw, gap, x0, y0, ch = 262, 24, 84, 168, 372
    for i, (title, meta) in enumerate(titles):
        x = x0 + i * (cw + gap)
        tile = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        mask = rounded_mask((cw, ch), 22)
        base = Image.new("RGBA", (cw, ch), card + (255,))
        tile.paste(base, (0, 0), mask)
        art = card_art((cw, 218), accent, i, rng)
        tile.paste(art, (0, 0), rounded_mask((cw, 218), 22))
        # square the lower art corners
        tile.paste(art.crop((0, 120, cw, 218)), (0, 120))
        d = ImageDraw.Draw(tile)
        ts = fit_size(face, title, cw - 36, 38, 30)
        base_y = center_baseline(face, title, ts, 252)
        d.text((18, base_y), face.ok(title), font=face.at(ts), fill=card_ink, anchor="ls")
        ms = fit_size(face, meta, cw - 36, 26, 20)
        d.text((18, center_baseline(face, meta, ms, 292)), face.ok(meta), font=face.at(ms), fill=card_sub, anchor="ls")
        d.rounded_rectangle([18, 328, cw - 18, 338], radius=5, fill=mix(card, card_ink, 0.16))
        d.rounded_rectangle([18, 328, 18 + int((cw - 36) * [0.72, 0.35, 0.9, 0.12][i]), 338], radius=5, fill=accent)
        im.alpha_composite(tile, (x, y0))
    # continue pill
    overlay(im, lambda d: d.rounded_rectangle([84, 566, 330, 636], radius=35, fill=accent + (255,)))
    pill_ink = (255, 255, 255) if lum(accent) < 0.62 else (20, 20, 20)
    ps = fit_size(face, "Continue", 190, 36, 30)
    put_text(im, face, (207, center_baseline(face, "Continue", ps, 601)), "Continue", ps, pill_ink, "ms")
    put_text(im, face, (360, center_baseline(face, "Last played: Neon Drift", 26, 601)), "Last played: Neon Drift", 26, sub, "ls")
    return im.convert("RGB")


def render_gallery_text(face, scripts, theme, seed):
    rng = np.random.default_rng(seed + 2)
    im = backdrop(theme, rng)
    ink, sub, accent = theme.ink, theme.sub, theme.accent
    overlay(im, lambda d: d.rounded_rectangle([34, 34, W - 34, H - 34], radius=34, fill=mix(theme.bg0, theme.bg1, 0.5) + (222,)))
    # settings list
    put_text(im, face, (84, 118), "Settings", fit_size(face, "Settings", 400, 70, 60), ink, "ls")
    rows = [("Graphics", "High", None), ("Audio", "Surround", None), ("Controls", "Gamepad", None), ("Show FPS", "", True), ("Auto save", "", False)]
    y = 160
    line = mix(theme.bg1, ink, 0.2)
    for label, value, toggle in rows:
        ls = fit_size(face, label, 250, 40, 30)
        put_text(im, face, (84, center_baseline(face, label, ls, y + 44)), label, ls, ink, "ls")
        if toggle is None:
            vs = fit_size(face, value, 170, 34, 24)
            put_text(im, face, (592, center_baseline(face, value, vs, y + 44)), value, vs, sub, "rs")
        else:
            on = toggle
            overlay(im, lambda d, y=y, on=on: (d.rounded_rectangle([522, y + 22, 592, y + 66], radius=22, fill=(accent if on else mix(theme.bg1, ink, 0.28)) + (255,)),
                                              d.ellipse([(556 if on else 528), y + 28, (586 if on else 558), y + 58], fill=(255, 255, 255, 255))))
        overlay(im, lambda d, y=y: d.line([(84, y + 88), (592, y + 88)], fill=line + (255,), width=2))
        y += 96
    # pangrams
    overlay(im, lambda d: d.line([(640, 96), (640, H - 96)], fill=line + (255,), width=2))
    x0, max_w = 682, W - 84 - 682
    put_text(im, face, (x0, 118), "Aa", fit_size(face, "Aa", 120, 70, 62), accent, "ls")
    blocks = ["The quick brown fox jumps over the lazy dog."]
    if "cyrillic" in scripts:
        blocks.append("Чуєш їх, доцю, га? Кумедна ж ти, прощайся без ґольфів!")
        blocks.append("Съешь же ещё этих мягких французских булок.")
    blocks.append("0123456789 ABCDEFGHIJKLM")
    size = 36
    while size > 16:
        total = 0
        wrapped = []
        for text in blocks:
            lines = wrap_words(face, text, size, max_w)
            wrapped.append(lines)
            total += len(lines) * size * 1.38 + 26
        if total <= 440:
            break
        size = int(size * 0.92)
    y = 172
    for lines in wrapped:
        for ln in lines:
            put_text(im, face, (x0, center_baseline(face, ln, size, y + size * 0.7)), ln, size, ink, "ls")
            y += size * 1.38
        y += 26
    return im.convert("RGB")


# ====================================================================== build
def webp_bytes(im, q=84, method=5):
    b = io.BytesIO()
    im.save(b, "WEBP", quality=q, method=method)
    return b.getvalue()


def det_zip(files):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for name, data in files:
            zi = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zi.create_system = 3
            z.writestr(zi, data, compresslevel=9)
    return b.getvalue()


def seed_of(fid):
    return int(hashlib.sha256(fid.encode()).hexdigest()[:8], 16) % 100000


def entries():
    out = []
    for family, directory, lic, file in SOURCES:
        cat, style, pro, feat, theme = META[family]
        out.append({"family": family, "dir": directory, "lic": lic, "file": file, "id": font_id(family), "cat": cat,
                    "style": style, "pro": bool(pro), "feat": bool(feat), "theme": theme})
    return out


def keep_locales(lst, path, retranslate):
    """Carry non-English name/description translations over from the existing listing.json."""
    if retranslate:
        return lst
    try:
        old = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return lst
    for key in ("name", "description"):
        for loc, txt in (old.get(key) or {}).items():
            if loc != "en" and loc in LOCALES and txt:
                lst[key][loc] = txt
    return lst


def keep_label_locales(cats, path, retranslate):
    if retranslate:
        return cats
    try:
        old = {c["id"]: c.get("labels", {}) for c in json.loads(Path(path).read_text(encoding="utf-8"))}
    except (OSError, ValueError, KeyError, TypeError):
        return cats
    for c in cats:
        for loc, txt in old.get(c["id"], {}).items():
            if loc != "en" and loc in LOCALES and txt:
                c["labels"][loc] = txt
    return cats


def ordered(mapping):
    return {loc: mapping[loc] for loc in LOCALES if loc in mapping}


def build_one(args):
    try:
        return build_one_(args)
    except Exception:
        import traceback
        traceback.print_exc()
        return args[0]["id"], "FAIL", 0


def build_one_(args):
    e, script_hash, force, retranslate = args
    folder = CACHE / e["id"]
    font_path = folder / "font.ttf"
    lic_path = folder / "LICENSE.txt"
    if not font_path.exists() or not lic_path.exists():
        return e["id"], "NOT-FETCHED", 0
    info = analyse(font_path)
    scripts = [s for s in SCRIPT_ORDER if s in info["scripts"]]
    if "latin" not in scripts:
        return e["id"], "NO-LATIN", 0
    theme = Theme(e["theme"])
    swatch = ["#%02X%02X%02X" % c for c in (theme.bg0, theme.bg1, theme.accent, theme.ink)]
    tags = [f"script:{s}" for s in scripts] + [f"style:{e['cat']}"]
    if e["style"] and e["style"] != e["cat"]:
        tags.append(f"style:{e['style']}")
    lst = {
        "version": 1, "categories": [e["cat"]], "tags": tags, "featured": e["feat"], "publishedAt": PUBLISHED,
        "requiresPro": e["pro"], "minAppVersionCode": 0, "swatch": swatch,
        "name": ordered({loc: e["family"] for loc in LOCALES}),
        "description": ordered({loc: DESC[e["cat"]][loc].replace("{n}", e["family"]) for loc in LOCALES}),
    }
    ldir = ROOT / "src" / e["id"]
    ddir = ROOT / "dist" / e["id"]
    ldir.mkdir(parents=True, exist_ok=True)
    ddir.mkdir(parents=True, exist_ok=True)
    lst = keep_locales(lst, ldir / "listing.json", retranslate)
    (ldir / "listing.json").write_text(json.dumps(lst, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    font_bytes = font_path.read_bytes()
    h = hashlib.sha256()
    h.update(json.dumps(lst, sort_keys=True, ensure_ascii=False).encode())
    h.update(script_hash.encode())
    h.update(hashlib.sha256(font_bytes).digest())
    sha = h.hexdigest()
    pkg_path = ddir / "item.ecxfont"
    bj = ddir / "build.json"
    outputs = [pkg_path, ddir / "cover.webp", ddir / "gallery-1.webp", ddir / "gallery-2.webp"]
    if not force and bj.exists() and all(p.exists() for p in outputs):
        try:
            if json.loads(bj.read_text())["sourceSha256"] == sha:
                return e["id"], "skip", pkg_path.stat().st_size
        except Exception:
            pass

    meta = {
        "formatVersion": 1, "id": e["id"], "family": e["family"], "file": "font.ttf", "license": LICENSE_IDS[e["lic"]],
        "source": source_url(e["dir"], e["lic"], e["file"]), "scripts": scripts, "sizeScale": info["sizeScale"],
    }
    pkg = det_zip([
        ("font.json", (json.dumps(meta, indent=2, ensure_ascii=False) + "\n").encode("utf-8")),
        ("font.ttf", font_bytes),
        ("LICENSE.txt", lic_path.read_bytes()),
    ])
    if len(pkg) > MAX_PACKAGE:
        return e["id"], "TOO-BIG", len(pkg)
    pkg_path.write_bytes(pkg)

    face = Face(font_path, info["cmap"], info["axes"])
    seed = seed_of(e["id"])
    imgs = {
        "cover.webp": render_cover(face, e["family"], scripts, theme, seed),
        "gallery-1.webp": render_gallery_library(face, scripts, theme, seed),
        "gallery-2.webp": render_gallery_text(face, scripts, theme, seed),
    }
    for name, im in imgs.items():
        q = 84
        data = webp_bytes(im, q)
        while len(data) > 300_000 and q > 50:
            q -= 8
            data = webp_bytes(im, q)
        (ddir / name).write_bytes(data)
    bj.write_text(json.dumps({"sourceSha256": sha}, indent=2) + "\n", encoding="utf-8", newline="\n")
    note = "built" + (" partial-cyr" if info["partial_cyrillic"] else "")
    return e["id"], note, len(pkg)


def contact_sheet(path, items):
    from PIL import ImageFont as IF
    cols, cw, chh, cap = 6, 320, 180, 40
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (cw + 10) + 10, rows * (chh + cap + 10) + 10), (22, 22, 26))
    font = IF.load_default(size=15)
    d = ImageDraw.Draw(sheet)
    for i, e in enumerate(items):
        x, y = 10 + (i % cols) * (cw + 10), 10 + (i // cols) * (chh + cap + 10)
        f = ROOT / "dist" / e["id"] / "cover.webp"
        if f.exists():
            sheet.paste(Image.open(f).convert("RGB").resize((cw, chh), Image.LANCZOS), (x, y))
        listing = json.loads((ROOT / "src" / e["id"] / "listing.json").read_text(encoding="utf-8")) if (ROOT / "src" / e["id"] / "listing.json").exists() else {}
        marks = " ".join({"latin": "L", "latin-ext": "Lx", "cyrillic": "C", "greek": "G", "vietnamese": "V", "arabic": "A", "devanagari": "D", "cjk": "J"}[t.split(":")[1]]
                         for t in listing.get("tags", []) if t.startswith("script:"))
        d.text((x, y + chh + 4), e["id"], font=font, fill=(235, 235, 240))
        d.text((x, y + chh + 21), ("PRO" if e["pro"] else "Free") + "  " + marks, font=font, fill=(255, 196, 64) if e["pro"] else (130, 220, 150))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--sheet", default="")
    ap.add_argument("--retranslate", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    only = set(x for x in a.only.split(",") if x)
    todo_entries = [e for e in entries() if not only or e["id"] in only]
    assert len({e["id"] for e in entries()}) == len(SOURCES), "duplicate ids"
    for e in entries():
        assert e["family"] in META and e["cat"] in CATEGORIES, e["family"]
        for loc in LOCALES:
            text = DESC[e["cat"]][loc].replace("{n}", e["family"])
            assert len(text) <= 160, (e["id"], loc, len(text))
        assert len(e["family"]) <= 40 and len(e["id"]) <= 48, e["id"]
    if a.fetch:
        with ProcessPoolExecutor(8) as ex:
            for family, status, size in ex.map(fetch_one, [s for s in SOURCES if not only or font_id(s[0]) in only]):
                print(f"{status:10} {family} {size // 1024} KB", flush=True)
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    cats = keep_label_locales([{"id": c, "labels": ordered(CAT_LABELS[c])} for c in CATEGORIES], ROOT / "categories.json", a.retranslate)
    (ROOT / "categories.json").write_text(json.dumps(cats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    todo = [(e, script_hash, a.force, a.retranslate) for e in todo_entries]
    results = []
    if a.jobs > 1 and len(todo) > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            for r in ex.map(build_one, todo):
                results.append(r)
                print(f"{r[1]:18} {r[0]} {r[2] // 1024} KB", flush=True)
    else:
        for t in todo:
            r = build_one(t)
            results.append(r)
            print(f"{r[1]:18} {r[0]} {r[2] // 1024} KB", flush=True)
    bad = [r for r in results if r[1] not in ("built", "skip", "built partial-cyr")]
    if a.sheet:
        contact_sheet(a.sheet, [e for e in entries() if (ROOT / "dist" / e["id"] / "cover.webp").exists()])
    if bad:
        print("PROBLEMS:", [(r[0], r[1]) for r in bad], file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
