import fs from "node:fs";
import path from "node:path";
import {
  collectContent,
  config,
  fileReference,
  readJson,
  repositoryRoot,
  sectionNames
} from "./lib/hub-content.mjs";

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

