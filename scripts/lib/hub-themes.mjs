import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { config, fileReference, readJson, repositoryRoot } from "./hub-content.mjs";

// Theme packages: see docs in themes/README.md and EmuCoreX docs/themes/ecxtheme-format.md.
const themesRoot = path.join(repositoryRoot, "themes");
const sourceRoot = path.join(themesRoot, "src");
const distRoot = path.join(themesRoot, "dist");
const builderPath = path.join(repositoryRoot, "scripts", "build-themes.py");
const packageFiles = ["theme.json", "font.ttf", "font.otf", "background.webp", "background.jpg", "background.png"];
export const idPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const hexPattern = /^#[0-9A-Fa-f]{6}$/;
const requiredColors = [
  "primary", "onPrimary", "secondary", "tertiary", "background", "surface",
  "surfaceVariant", "onBackground", "onSurface", "onSurfaceVariant", "outline", "error"
];
const shapeStyles = new Set(["rounded", "cut", "cutDiagonal", "square"]);
const fonts = new Set(["system", "rubik", "exo2", "monospace", "serif", "package", "pixel", "spacemono", "lora", "oswald"]);
const decors = new Set(["none", "neon"]);
const maxPackageBytes = 4 * 1024 * 1024;
const maxImageBytes = 600 * 1024;
const screenOrder = ["hub", "settings", "game", "ingame", "store"];

export function hasThemes() {
  return fs.existsSync(sourceRoot);
}

export function collectThemes() {
  const errors = [];
  const categoriesPath = path.join(themesRoot, "categories.json");
  const categories = fs.existsSync(categoriesPath) ? readJson(categoriesPath) : [];
  const categoryIds = new Set();
  for (const category of categories) {
    category.kind = "theme";
    if (!idPattern.test(category.id ?? "")) errors.push(`themes/categories.json: invalid id ${category.id}`);
    if (categoryIds.has(category.id)) errors.push(`themes/categories.json: duplicate ${category.id}`);
    categoryIds.add(category.id);
    requireLocales(category.labels, `themes/categories.json ${category.id}.labels`, 32, errors);
  }

  const themes = [];
  const directories = fs.existsSync(sourceRoot)
    ? fs.readdirSync(sourceRoot, { withFileTypes: true }).filter((entry) => entry.isDirectory()).map((entry) => entry.name).sort()
    : [];
  for (const id of directories) {
    const location = `themes/src/${id}`;
    const themeDir = path.join(sourceRoot, id);
    let theme;
    let listing;
    try {
      theme = readJson(path.join(themeDir, "theme.json"));
      listing = readJson(path.join(themeDir, "listing.json"));
    } catch (error) {
      errors.push(`${location}: unreadable theme.json/listing.json (${error.message})`);
      continue;
    }
    for (const entry of fs.readdirSync(themeDir)) {
      if (!packageFiles.includes(entry) && entry !== "listing.json") errors.push(`${location}: unexpected file ${entry}`);
    }
    validateTheme(theme, id, location, errors);
    validateListing(listing, categoryIds, location, errors);

    const outDir = path.join(distRoot, id);
    const packagePath = path.join(outDir, "theme.ecxtheme");
    const previewPath = path.join(outDir, "preview.webp");
    const stampPath = path.join(outDir, "build.json");
    if (!fs.existsSync(packagePath) || !fs.existsSync(previewPath) || !fs.existsSync(stampPath)) {
      errors.push(`${location}: dist output is missing; run python scripts/build-themes.py`);
      continue;
    }
    if (readJson(stampPath).sourceSha256 !== sourceDigest(themeDir)) {
      errors.push(`${location}: dist output is stale; run python scripts/build-themes.py`);
    }
    const packageBytes = fs.readFileSync(packagePath);
    if (packageBytes.length > maxPackageBytes) errors.push(`${location}: package exceeds 4 MB`);
    const cover = imageReference(previewPath, `${location} preview`, errors);
    const gallery = [];
    const extras = fs.readdirSync(outDir);
    const screens = extras.filter((name) => /^screen-.+\.webp$/.test(name)).sort((a, b) => screenRank(a) - screenRank(b) || a.localeCompare(b));
    const shots = extras.filter((name) => /^shot-.+\.webp$/.test(name)).sort();
    for (const name of [...screens, ...shots]) gallery.push(imageReference(path.join(outDir, name), `${location} ${name}`, errors));

    const colors = theme.colors ?? {};
    themes.push({
      id,
      kind: "theme",
      version: theme.version,
      requiresPro: theme.requiresPro === true,
      dark: typeof theme.dark === "boolean" ? theme.dark : isDark(colors.background ?? "#000000"),
      minAppVersionCode: theme.minAppVersionCode ?? 0,
      featured: listing.featured === true,
      publishedAt: listing.publishedAt,
      categories: listing.categories ?? [],
      tags: listing.tags ?? [],
      name: theme.name,
      description: listing.description,
      swatch: [colors.primary, colors.secondary, colors.tertiary, colors.background].filter(Boolean),
      cover,
      gallery,
      packageFile: fileReference(packagePath),
      sizeBytes: packageBytes.length
    });
  }
  themes.sort((a, b) => (Number(b.featured) - Number(a.featured)) || b.publishedAt.localeCompare(a.publishedAt) || a.id.localeCompare(b.id));
  return { errors, categories, items: themes };
}

function validateTheme(theme, id, location, errors) {
  if (theme.formatVersion !== 1) errors.push(`${location}: formatVersion must be 1`);
  if (theme.id !== id || !idPattern.test(id) || id.length < 3 || id.length > 48) errors.push(`${location}: id must match the directory and [a-z0-9-]{3,48}`);
  if (!Number.isInteger(theme.version) || theme.version < 1) errors.push(`${location}: version must be a positive integer`);
  if (typeof theme.requiresPro !== "boolean") errors.push(`${location}: requiresPro must be true or false`);
  requireLocales(theme.name, `${location} name`, 40, errors);
  for (const key of requiredColors) {
    if (!hexPattern.test(theme.colors?.[key] ?? "")) errors.push(`${location}: colors.${key} must be #RRGGBB`);
  }
  for (const [key, value] of Object.entries(theme.colors ?? {})) {
    if (!hexPattern.test(value)) errors.push(`${location}: colors.${key} must be #RRGGBB`);
  }
  for (const value of theme.accents ?? []) if (!hexPattern.test(value)) errors.push(`${location}: accents must be #RRGGBB`);
  if ((theme.accents ?? []).length > 3) errors.push(`${location}: at most 3 accents`);
  for (const value of Object.values(theme.semantic ?? {})) if (!hexPattern.test(value)) errors.push(`${location}: semantic colors must be #RRGGBB`);
  if (theme.shape && !shapeStyles.has(theme.shape.style ?? "rounded")) errors.push(`${location}: unknown shape.style`);
  if (theme.shape?.scale !== undefined && !(theme.shape.scale >= 0 && theme.shape.scale <= 2)) errors.push(`${location}: shape.scale must be 0..2`);
  if (theme.typography?.font && !fonts.has(theme.typography.font)) errors.push(`${location}: unknown typography.font`);
  if (theme.effects?.decor && !decors.has(theme.effects.decor)) errors.push(`${location}: unknown effects.decor`);
  const hasFont = fs.existsSync(path.join(sourceRoot, id, "font.ttf")) || fs.existsSync(path.join(sourceRoot, id, "font.otf"));
  if (theme.typography?.font === "package" && !hasFont) errors.push(`${location}: typography.font is "package" but no font file`);
}

export function validateListing(listing, categoryIds, location, errors) {
  if (!Array.isArray(listing.categories) || listing.categories.length === 0) errors.push(`${location}: listing.categories is required`);
  for (const category of listing.categories ?? []) {
    if (!categoryIds.has(category)) errors.push(`${location}: unknown category ${category}`);
  }
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/.test(listing.publishedAt ?? "")) errors.push(`${location}: publishedAt must be an ISO UTC instant`);
  requireLocales(listing.description, `${location} description`, 160, errors);
  for (const tag of listing.tags ?? []) if (!idPattern.test(tag)) errors.push(`${location}: invalid tag ${tag}`);
}

export function requireLocales(map, label, maxLength, errors) {
  for (const locale of config.requiredLocales) {
    const value = map?.[locale];
    if (typeof value !== "string" || value.trim() === "") errors.push(`${label}: missing ${locale}`);
    else if (value.length > maxLength) errors.push(`${label}: ${locale} is longer than ${maxLength} characters`);
  }
}

// Must match source_digest() in scripts/build-themes.py.
function sourceDigest(themeDir) {
  const hash = crypto.createHash("sha256");
  for (const name of [...packageFiles, "listing.json"]) {
    const filePath = path.join(themeDir, name);
    if (!fs.existsSync(filePath)) continue;
    let data = fs.readFileSync(filePath);
    if (name.endsWith(".json")) data = Buffer.from(data.toString("latin1").replace(/\r\n/g, "\n"), "latin1");
    hash.update(Buffer.concat([Buffer.from(name), Buffer.from([0]), data, Buffer.from([0])]));
  }
  hash.update(Buffer.from(fs.readFileSync(builderPath).toString("latin1").replace(/\r\n/g, "\n"), "latin1"));
  return hash.digest("hex");
}

function isDark(hex) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return 0.299 * r + 0.587 * g + 0.114 * b < 128;
}

export function webpDimensions(bytes) {
  if (bytes.length < 30 || bytes.toString("ascii", 0, 4) !== "RIFF" || bytes.toString("ascii", 8, 12) !== "WEBP") return null;
  const type = bytes.toString("ascii", 12, 16);
  if (type === "VP8X") return { width: 1 + bytes.readUIntLE(24, 3), height: 1 + bytes.readUIntLE(27, 3) };
  if (type === "VP8 ") return { width: bytes.readUInt16LE(26) & 0x3fff, height: bytes.readUInt16LE(28) & 0x3fff };
  if (type === "VP8L") {
    const b0 = bytes[21], b1 = bytes[22], b2 = bytes[23], b3 = bytes[24];
    return { width: 1 + b0 + ((b1 & 0x3f) << 8), height: 1 + (b1 >> 6) + (b2 << 2) + ((b3 & 0x0f) << 10) };
  }
  return null;
}

function screenRank(name) {
  const index = screenOrder.findIndex((part) => name === `screen-${part}.webp`);
  return index < 0 ? screenOrder.length : index;
}

// WebP file reference with dimensions and the shared 600 KB image cap.
export function imageReference(filePath, label, errors) {
  const bytes = fs.readFileSync(filePath);
  if (bytes.length > maxImageBytes) errors.push(`${label}: exceeds 600 KB`);
  const dimensions = webpDimensions(bytes);
  if (!dimensions || dimensions.width < 1 || dimensions.height < 1) errors.push(`${label}: not a readable WebP image`);
  return fileReference(filePath, { width: dimensions?.width ?? 0, height: dimensions?.height ?? 0 });
}
