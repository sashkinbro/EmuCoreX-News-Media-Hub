# EmuCore Store — catalog and package formats (v1)

The Store is served from this repository like the rest of the Hub: one catalog file referenced by the
signed manifest, and package files pinned by commit + SHA-256. Old apps ignore it.

```
manifest.json  ──store──►  catalog/v1/store.json
                              ├─ categories[]   (per kind, 18-locale labels)
                              └─ items[]        (themes, wallpapers, sounds, controls, fonts)
```

## Item (store.json → items[])

| Field | Notes |
| --- | --- |
| `id` | `[a-z0-9-]{3,48}`, unique across all kinds, stable forever |
| `kind` | `theme` · `wallpaper` · `sounds` · `controls` · `font` |
| `version` | positive int; a higher version shows "Update" |
| `requiresPro` | Pro items can be previewed and installed by everyone, applied only with Pro |
| `minAppVersionCode` | older apps show "Requires a newer app" |
| `featured`, `publishedAt` | store ordering |
| `categories[]`, `tags[]` | filter chips |
| `name{18}`, `description{18}` | localized; name ≤ 40, description ≤ 160 |
| `swatch[]` | up to 4 `#RRGGBB` for placeholders/loading shimmer |
| `cover` | file ref: the card image (theme: schematic `preview.webp`) |
| `gallery[]` | file refs: detail-page images — schematic screens (`screen-*.webp`) and real English device screenshots (`shot-*.webp`) |
| `packageFile` | file ref: what gets downloaded and installed |
| `sizeBytes` | installed size shown on the detail page |

File refs are `{path, sha256, bytes, width?, height?}` relative to the pinned commit.

## Package formats

All packages are ZIP archives with a JSON manifest at the root, validated by the app before anything
is written (allow-listed names only, size caps, no paths). Unknown JSON keys are ignored.

### `theme` — `.ecxtheme`
`theme.json` (+ optional `font.ttf|otf`, `background.webp|jpg|png`). See the theme format in
EmuCoreX `docs/themes/ecxtheme-format.md` (fields: colors, accents, semantic, shape, typography,
effects.decor, effects.style = UI style engine, background.dim).

### `wallpaper` — `.ecxwall`
```json
{ "formatVersion": 1, "id": "aurora-ridge", "version": 1, "type": "image", "dim": 0.35,
  "focus": "center", "rights": { "creator": "…", "sourceUrl": "https://…", "licenseId": "CC0-1.0",
  "licenseUrl": "https://…", "attribution": "…" } }
```
+ `wallpaper.webp` (≤ 2560 px long side, ≤ 4 MB). Applying installs it as the Home background (the
existing user-image path) and sets the dim. Sources: self-generated art, public-domain space imagery,
CC0 photo/texture libraries — rights are recorded per item.

### `sounds` — `.ecxsounds`
```json
{ "formatVersion": 1, "id": "chip-blips", "version": 1,
  "files": { "click": "click.ogg", "menu_open": "menu_open.ogg", "menu_close": "menu_close.ogg",
             "menu_select": "menu_select.ogg", "music": "music.ogg", "achievement": "achievement.ogg" } }
```
Short sounds ≤ 2 s and ≤ 300 KB each (SoundPool), music optional, ≤ 6 MB, loopable. Formats: OGG
Vorbis or WAV PCM. Applying selects the pack for the click and game-menu kinds (and music if present).
`achievement` (optional, 1–3 s, peak ≤ −3 dBFS) is the sound played when an achievement unlocks; clients
that do not know the key ignore it, and a pack without it keeps the app's default unlock sound. Every
pack in the catalog now ships click, menu_open/menu_close/menu_select, a seamless `music` loop and
`achievement`. The builder derives the listing tags `contains:click`, `contains:menu`, `contains:music`
and `contains:achievement` from the real package contents (tag ids may contain `:`).

### `controls` — `.ecxcontrols`
Two flavours in one format:
```json
{ "formatVersion": 1, "id": "candy-buttons", "version": 1, "skin": { … }, "layout": { … } }
```
- `skin` (optional): art for on-screen buttons. `images` maps control ids (`cross`, `circle`,
  `square`, `triangle`, `dpad_up|down|left|right`, `l1`, `l2`, `r1`, `r2`, `l3`, `r3`, `start`,
  `select`, `analog_base`, `analog_stick`, …) to PNG/WebP files in the archive (square, ≥ 192 px,
  transparent background), optional `<id>_pressed`; `tint`, `opacity`, `pressEffect`.
  Unmapped controls fall back to the built-in vector art.
- `layout` (optional):
  `{ "overlay": { "<controlId>": { "x": 0.0, "y": 0.0, "scale": 1.0, "widthScale": 1.0, "opacity": 0.8,
  "visible": true, "surfaceOnly": false, "secondaryActionId": null } }, "custom": <CustomTouchControlLibrary JSON> }`
  — applied through the app's own sanitizers (overlay positions are the app's `OverlayControlLayout` offsets).
- Skin fields: `tint` `#RRGGBB`/`#AARRGGBB`, `opacity` 0..1, `pressEffect` grow|shrink|spring|glow; images
  64–1024 px, aspect ≤ 1.5; `<id>_pressed` entries live in the same `images` map.

### `font` — `.ecxfont`
```json
{ "formatVersion": 1, "id": "font-jetbrains-mono", "family": "JetBrains Mono", "file": "font.ttf",
  "license": "OFL-1.1", "source": "https://raw.githubusercontent.com/google/fonts/main/ofl/jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf",
  "scripts": ["latin", "latin-ext", "cyrillic", "greek", "vietnamese"], "sizeScale": 1.0 }
```
Exactly three entries: `font.json`, one font file (`font.ttf`; `font.otf` only when no TTF exists; a variable font
is fine — the app asks for the regular weight) and `LICENSE.txt` (the font's own license text). Package ≤ 4 MB.
- `license` is `OFL-1.1` or `Apache-2.0` only; `source` is the exact `raw.githubusercontent.com/google/fonts/main/...`
  URL the file was taken from (nothing else is accepted). `family` equals the English item name.
- `scripts` lists the scripts the font actually covers, detected from its cmap: `latin`, `latin-ext`, `cyrillic`
  (full Russian + Ukrainian alphabets), `greek`, `vietnamese`, `arabic`, `devanagari`, `cjk`. The catalog item carries
  the same values as `script:<x>` tags (filter chips) plus `style:<x>` tags; key:value tags are allowed for all kinds.
- `sizeScale` (0.9–1.25, default 1.0) nudges the UI text size for fonts with a small or large x-height.
- Applying selects the font as the app UI font (the same path as the font inside a theme).
- Builder: `python scripts/build-store-fonts.py` (`--fetch` downloads the files into `.workspace/fonts-cache/`).

## App side (EmuCoreX first, then all apps)

- Store destination in the drawer with tabs Themes · Wallpapers · Sounds · Controls · Fonts; featured row,
  filters (Free / Pro / Installed), sorting (Featured, Newest, Name).
- Item detail: gallery (schematic screens + real screenshots), description, size, version, Install /
  Update / Apply / Remove with download progress; Pro items without Pro open the Pro purchase sheet.
- "Get more in Store" buttons next to the theme picker, wallpaper picker, sound presets and
  touch-control styles in Settings, deep-linking to the matching tab.
- Screenshots for the gallery are captured on device with the app language set to English.
