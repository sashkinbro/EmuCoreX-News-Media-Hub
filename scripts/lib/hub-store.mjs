import fs from "node:fs";
import path from "node:path";
import { config, fileReference, readJson, repositoryRoot } from "./hub-content.mjs";
import { collectThemes, idPattern, imageReference, requireLocales, validateListing } from "./hub-themes.mjs";

// Store catalog: see store/README.md. Themes come from themes/{src,dist}; every other kind uses
// store/<kind>/src/<id>/listing.json + store/<kind>/dist/<id>/{package, cover.webp, *.webp, build.json}.
export const storeKinds = ["theme", "wallpaper", "sounds", "controls", "font"];
const extraKinds = {
  wallpaper: { packageName: "item.ecxwall", maxPackageBytes: 8 * 1024 * 1024 },
  sounds: { packageName: "item.ecxsounds", maxPackageBytes: 4 * 1024 * 1024 },
  controls: { packageName: "item.ecxcontrols", maxPackageBytes: 4 * 1024 * 1024 },
  font: { packageName: "item.ecxfont", maxPackageBytes: 4 * 1024 * 1024 }
};
const storeRoot = path.join(repositoryRoot, "store");
const maxSwatch = 4;
const hexPattern = /^#[0-9A-Fa-f]{6}$/;
// Store tags may be plain ("pixel") or key:value ("script:cyrillic", "style:mono").
const tagPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*(?::[a-z0-9]+(?:-[a-z0-9]+)*)?$/;

export function collectStore() {
  const errors = [];
  const categories = [];
  const items = [];

  const themes = collectThemes();
  errors.push(...themes.errors);
  categories.push(...themes.categories);
  items.push(...themes.items);

  for (const [kind, spec] of Object.entries(extraKinds)) {
    const kindRoot = path.join(storeRoot, kind);
    // A kind is published only when it is marked ready (store/<kind>/publish.json), so content that
    // is still being generated or translated never breaks a release.
    if (!fs.existsSync(path.join(kindRoot, "publish.json"))) continue;
    const kindCategories = readCategories(kind, path.join(kindRoot, "categories.json"), errors);
    const categoryIds = new Set(kindCategories.map((category) => category.id));
    categories.push(...kindCategories);
    const sourceRoot = path.join(kindRoot, "src");
    const distRoot = path.join(kindRoot, "dist");
    const directories = fs.existsSync(sourceRoot)
      ? fs.readdirSync(sourceRoot, { withFileTypes: true }).filter((entry) => entry.isDirectory()).map((entry) => entry.name).sort()
      : [];
    for (const id of directories) {
      const location = `store/${kind}/src/${id}`;
      let listing;
      try {
        listing = readJson(path.join(sourceRoot, id, "listing.json"));
      } catch (error) {
        errors.push(`${location}: unreadable listing.json (${error.message})`);
        continue;
      }
      if (!idPattern.test(id) || id.length < 3 || id.length > 48) errors.push(`${location}: id must match [a-z0-9-]{3,48}`);
      validateListing({ ...listing, tags: [] }, categoryIds, location, errors);
      for (const tag of listing.tags ?? []) if (!tagPattern.test(tag)) errors.push(`${location}: invalid tag ${tag}`);
      requireLocales(listing.name, `${location} name`, 40, errors);
      if (!Number.isInteger(listing.version) || listing.version < 1) errors.push(`${location}: version must be a positive integer`);
      if (listing.requiresPro !== undefined && typeof listing.requiresPro !== "boolean") errors.push(`${location}: requiresPro must be true or false`);
      if (listing.minAppVersionCode !== undefined && !Number.isInteger(listing.minAppVersionCode)) errors.push(`${location}: minAppVersionCode must be an integer`);
      const swatch = listing.swatch ?? [];
      if (!Array.isArray(swatch) || swatch.length > maxSwatch || swatch.some((value) => !hexPattern.test(value))) errors.push(`${location}: swatch must be up to ${maxSwatch} #RRGGBB values`);

      const outDir = path.join(distRoot, id);
      const packagePath = path.join(outDir, spec.packageName);
      const coverPath = path.join(outDir, "cover.webp");
      if (!fs.existsSync(packagePath) || !fs.existsSync(coverPath)) {
        errors.push(`store/${kind}/dist/${id}: needs ${spec.packageName} and cover.webp`);
        continue;
      }
      const packageSize = fs.statSync(packagePath).size;
      if (packageSize > spec.maxPackageBytes) errors.push(`${location}: package exceeds ${spec.maxPackageBytes / 1024 / 1024} MB`);
      const gallery = fs.readdirSync(outDir)
        .filter((name) => name.endsWith(".webp") && name !== "cover.webp")
        .sort()
        .map((name) => imageReference(path.join(outDir, name), `${location} ${name}`, errors));
      items.push({
        id,
        kind,
        version: listing.version,
        requiresPro: listing.requiresPro === true,
        minAppVersionCode: listing.minAppVersionCode ?? 0,
        featured: listing.featured === true,
        publishedAt: listing.publishedAt,
        categories: listing.categories ?? [],
        tags: listing.tags ?? [],
        name: listing.name,
        description: listing.description,
        swatch,
        cover: imageReference(coverPath, `${location} cover`, errors),
        gallery,
        packageFile: fileReference(packagePath),
        sizeBytes: packageSize
      });
    }
  }

  const seen = new Map();
  for (const item of items) {
    if (seen.has(item.id)) errors.push(`store: id ${item.id} is used by both ${seen.get(item.id)} and ${item.kind}`);
    else seen.set(item.id, item.kind);
  }
  items.sort((a, b) => (Number(b.featured) - Number(a.featured)) || b.publishedAt.localeCompare(a.publishedAt) || a.id.localeCompare(b.id));
  const used = new Set(items.flatMap((item) => item.categories.map((category) => `${item.kind}:${category}`)));
  return {
    errors,
    categories: categories.filter((category) => used.has(`${category.kind}:${category.id}`)),
    items
  };
}

function readCategories(kind, filePath, errors) {
  if (!fs.existsSync(filePath)) return [];
  const categories = readJson(filePath);
  const ids = new Set();
  for (const category of categories) {
    category.kind = kind;
    if (!idPattern.test(category.id ?? "")) errors.push(`store/${kind}/categories.json: invalid id ${category.id}`);
    if (ids.has(category.id)) errors.push(`store/${kind}/categories.json: duplicate ${category.id}`);
    ids.add(category.id);
    requireLocales(category.labels, `store/${kind}/categories.json ${category.id}.labels`, 32, errors);
  }
  return categories;
}

export function storeCatalog(store) {
  return { formatVersion: 1, releaseId: config.releaseId, categories: store.categories, items: store.items };
}
