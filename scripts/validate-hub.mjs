import fs from "node:fs";
import zlib from "node:zlib";
import path from "node:path";
import {
  collectContent,
  config,
  fileReference,
  readJson,
  repositoryRoot,
  sectionNames
} from "./lib/hub-content.mjs";
import { collectStore, storeCatalog } from "./lib/hub-store.mjs";

const release = process.argv.includes("--release");
const collection = collectContent({ release });
const errors = [...collection.errors];
const warnings = [...collection.warnings];
const catalogRoot = path.join(repositoryRoot, "catalog", "v1");
const manifestPath = path.join(catalogRoot, "manifest.json");

if (!fs.existsSync(manifestPath)) {
  errors.push("catalog/v1/manifest.json is missing; run npm run build");
} else {
  try {
    const manifest = readJson(manifestPath);
    if (manifest.formatVersion !== config.catalogSchemaVersion) errors.push("manifest: unsupported formatVersion");
    if (manifest.releaseId !== config.releaseId) errors.push("manifest: releaseId does not match hub.config.json");
    if (manifest.catalogRevision !== config.catalogRevision) errors.push("manifest: catalogRevision does not match hub.config.json");
    if (manifest.defaultLocale !== config.defaultLocale) errors.push("manifest: defaultLocale does not match configuration");
    if (JSON.stringify(manifest.supportedLocales) !== JSON.stringify(config.requiredLocales)) errors.push("manifest: supportedLocales do not match configuration");
    for (const section of sectionNames) {
      for (const locale of config.requiredLocales) {
        const references = manifest.indexes?.[section]?.[locale];
        if (!Array.isArray(references) || references.length === 0) {
          errors.push(`manifest: missing ${section}/${locale} index`);
          continue;
        }
        for (const reference of references) validateReference(reference, errors);
      }
    }
    validateReference(manifest.assets, errors);
    validateReference(manifest.tombstones, errors);
    const store = collectStore();
    errors.push(...store.errors);
    if (!manifest.store) errors.push("manifest: store reference is missing");
    else {
      validateReference(manifest.store, errors);
      const catalog = readJson(path.join(repositoryRoot, manifest.store.path));
      if (manifest.store.itemCount !== (catalog.items ?? []).length) errors.push("manifest: store itemCount mismatch");
      for (const item of catalog.items ?? []) {
        validateReference(item.cover, errors);
        for (const image of item.gallery ?? []) validateReference(image, errors);
        validateReference(item.packageFile, errors);
        if (item.neutralPackageFile) validateReference(item.neutralPackageFile, errors);
        if (item.kind === "font") validateFontItem(item, errors);
      }
      if (JSON.stringify(catalog) !== JSON.stringify(storeCatalog(store))) errors.push("catalog: store.json is stale; run npm run build");
    }
  } catch (error) {
    errors.push(`manifest: invalid JSON (${error.message})`);
  }
}

for (const item of collection.items) {
  for (const { document } of item.localized.values()) {
    for (const block of document.blocks ?? []) {
      if (block.type === "image" && !collection.assets.has(block.assetId)) {
        errors.push(`${item.metadata.id}/${document.locale}: unknown assetId ${block.assetId}`);
      }
      if (block.type === "gallery") {
        for (const assetId of block.assetIds ?? []) {
          if (!collection.assets.has(assetId)) errors.push(`${item.metadata.id}/${document.locale}: unknown assetId ${assetId}`);
        }
      }
    }
  }
}

if (warnings.length > 0) process.stderr.write(`${warnings.map((value) => `warning: ${value}`).join("\n")}\n`);
if (errors.length > 0) {
  process.stderr.write(`${errors.join("\n")}\n`);
  process.exit(1);
}
process.stdout.write(`Validated ${collection.items.length} content items across ${config.requiredLocales.length} locales.\n`);

function validateReference(reference, errors) {
  if (!reference?.path || reference.path.includes("..") || reference.path.includes("\\") || path.isAbsolute(reference.path)) {
    errors.push("catalog: unsafe file reference");
    return;
  }
  const filePath = path.join(repositoryRoot, reference.path);
  if (!fs.existsSync(filePath)) {
    errors.push(`catalog: missing ${reference.path}`);
    return;
  }
  const actual = fileReference(filePath);
  if (actual.sha256 !== reference.sha256) errors.push(`catalog: checksum mismatch for ${reference.path}`);
  if (actual.bytes !== reference.bytes) errors.push(`catalog: byte count mismatch for ${reference.path}`);
  if (actual.contentType !== reference.contentType) errors.push(`catalog: content type mismatch for ${reference.path}`);
}


// Font packages (.ecxfont): font.json + one font file + LICENSE.txt, see store/README.md.
function validateFontItem(item, errors) {
  const label = `font ${item.id}`;
  const scriptTags = (item.tags ?? []).filter((tag) => tag.startsWith("script:")).map((tag) => tag.slice(7));
  if (scriptTags.length === 0) errors.push(`${label}: needs at least one script:<x> tag`);
  if (!(item.tags ?? []).some((tag) => tag.startsWith("style:"))) errors.push(`${label}: needs a style:<x> tag`);
  const archivePath = path.join(repositoryRoot, item.packageFile?.path ?? "");
  if (!fs.existsSync(archivePath)) return;
  let entries;
  try {
    entries = readZip(fs.readFileSync(archivePath));
  } catch (error) {
    errors.push(`${label}: unreadable package (${error.message})`);
    return;
  }
  const names = [...entries.keys()].sort();
  if (!entries.has("font.json") || !entries.has("LICENSE.txt") || names.length !== 3) errors.push(`${label}: package must contain exactly font.json, one font file and LICENSE.txt`);
  let meta;
  try {
    meta = JSON.parse(entries.get("font.json").toString("utf8"));
  } catch {
    errors.push(`${label}: font.json is missing or invalid`);
    return;
  }
  if (meta.formatVersion !== 1) errors.push(`${label}: font.json formatVersion must be 1`);
  if (meta.id !== item.id) errors.push(`${label}: font.json id mismatch`);
  if (meta.family !== item.name?.en) errors.push(`${label}: font.json family must equal the English name`);
  if (!/^font\.(ttf|otf)$/.test(meta.file ?? "") || !entries.has(meta.file)) errors.push(`${label}: font.json file must name font.ttf or font.otf in the package`);
  else {
    const magic = entries.get(meta.file).subarray(0, 4).toString("latin1");
    if (!["\x00\x01\x00\x00", "true", "OTTO"].includes(magic)) errors.push(`${label}: ${meta.file} is not a TrueType/OpenType font`);
  }
  if (!["OFL-1.1", "Apache-2.0"].includes(meta.license)) errors.push(`${label}: license must be OFL-1.1 or Apache-2.0`);
  if (typeof meta.source !== "string" || !meta.source.startsWith("https://raw.githubusercontent.com/google/fonts/main/")) errors.push(`${label}: source must be a github.com/google/fonts raw URL`);
  if (!Array.isArray(meta.scripts) || [...meta.scripts].sort().join() !== [...scriptTags].sort().join()) errors.push(`${label}: font.json scripts must match the script: tags`);
  if (typeof meta.sizeScale !== "number" || meta.sizeScale < 0.5 || meta.sizeScale > 2) errors.push(`${label}: sizeScale must be a number between 0.5 and 2`);
}

// Minimal ZIP reader (stored + deflate) over the central directory; enough to inspect small packages.
function readZip(buffer) {
  let eocd = -1;
  for (let i = buffer.length - 22; i >= Math.max(0, buffer.length - 66000); i -= 1) {
    if (buffer.readUInt32LE(i) === 0x06054b50) { eocd = i; break; }
  }
  if (eocd < 0) throw new Error("no end of central directory");
  const count = buffer.readUInt16LE(eocd + 10);
  let offset = buffer.readUInt32LE(eocd + 16);
  const entries = new Map();
  for (let n = 0; n < count; n += 1) {
    if (buffer.readUInt32LE(offset) !== 0x02014b50) throw new Error("bad central directory");
    const method = buffer.readUInt16LE(offset + 10);
    const compressed = buffer.readUInt32LE(offset + 20);
    const nameLength = buffer.readUInt16LE(offset + 28);
    const extraLength = buffer.readUInt16LE(offset + 30);
    const commentLength = buffer.readUInt16LE(offset + 32);
    const local = buffer.readUInt32LE(offset + 42);
    const name = buffer.toString("utf8", offset + 46, offset + 46 + nameLength);
    if (name.includes("/") || name.includes("\\") || name.includes("..")) throw new Error(`unsafe entry name ${name}`);
    const dataStart = local + 30 + buffer.readUInt16LE(local + 26) + buffer.readUInt16LE(local + 28);
    const raw = buffer.subarray(dataStart, dataStart + compressed);
    entries.set(name, method === 0 ? raw : zlib.inflateRawSync(raw));
    offset += 46 + nameLength + extraLength + commentLength;
  }
  return entries;
}
