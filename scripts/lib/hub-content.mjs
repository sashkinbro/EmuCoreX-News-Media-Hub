import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptsDirectory = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
export const repositoryRoot = path.dirname(scriptsDirectory);
export const config = readJson(path.join(repositoryRoot, "hub.config.json"));
export const sectionNames = ["news", "videos", "history"];

const idPattern = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const youtubeIdPattern = /^[A-Za-z0-9_-]{11}$/;
const supportedBlockTypes = new Set([
  "paragraph",
  "heading",
  "image",
  "gallery",
  "quote",
  "list",
  "callout",
  "link",
  "youtube"
]);

export function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8").replace(/^\uFEFF/, ""));
}

export function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

export function relativePath(filePath) {
  return path.relative(repositoryRoot, filePath).split(path.sep).join("/");
}

export function sha256Bytes(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

export function fileReference(filePath, extra = {}) {
  const bytes = fs.readFileSync(filePath);
  return {
    path: relativePath(filePath),
    sha256: sha256Bytes(bytes),
    bytes: bytes.length,
    contentType: contentTypeFor(filePath),
    ...extra
  };
}

function contentTypeFor(filePath) {
  switch (path.extname(filePath).toLowerCase()) {
    case ".json": return "application/json";
    case ".webp": return "image/webp";
    case ".jpg":
    case ".jpeg": return "image/jpeg";
    case ".png": return "image/png";
    default: return "application/octet-stream";
  }
}

function isHttps(value) {
  try {
    return new URL(value).protocol === "https:";
  } catch {
    return false;
  }
}

export function normalizeCanonicalUrl(value) {
  const url = new URL(value);
  url.hash = "";
  for (const key of [...url.searchParams.keys()]) {
    if (key.startsWith("utm_") || ["fbclid", "gclid", "si", "feature"].includes(key)) {
      url.searchParams.delete(key);
    }
  }
  url.hostname = url.hostname.toLowerCase();
  url.pathname = url.pathname.replace(/\/+$/, "") || "/";
  if (url.hostname === "youtu.be") {
    const id = url.pathname.split("/").filter(Boolean)[0];
    return `https://www.youtube.com/watch?v=${id}`;
  }
  return url.toString();
}

function normalizeText(value) {
  return value
    .normalize("NFKC")
    .toLocaleLowerCase("en-US")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

function listDirectories(directory) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => path.join(directory, entry.name))
    .sort();
}

export function collectContent({ release = false } = {}) {
  const errors = [];
  const warnings = [];
  const items = [];
  const assets = new Map();
  const ids = new Map();
  const canonicalUrls = new Map();
  const youtubeIds = new Map();
  const titleFingerprints = new Map();

  for (const section of sectionNames) {
    const sectionRoot = path.join(repositoryRoot, "content", section);
    for (const itemDirectory of listDirectories(sectionRoot)) {
      const metadataPath = path.join(itemDirectory, "metadata.json");
      if (!fs.existsSync(metadataPath)) {
        errors.push(`${relativePath(itemDirectory)}: metadata.json is missing`);
        continue;
      }

      let metadata;
      try {
        metadata = readJson(metadataPath);
      } catch (error) {
        errors.push(`${relativePath(metadataPath)}: invalid JSON (${error.message})`);
        continue;
      }

      const location = relativePath(metadataPath);
      if (metadata.schemaVersion !== config.schemaVersion) errors.push(`${location}: unsupported schemaVersion`);
      if (!idPattern.test(metadata.id ?? "")) errors.push(`${location}: invalid id`);
      if (path.basename(itemDirectory) !== metadata.id) errors.push(`${location}: directory name must equal id`);
      if (metadata.type !== section) errors.push(`${location}: type must be ${section}`);
      if (!Number.isInteger(metadata.contentVersion) || metadata.contentVersion < 1) errors.push(`${location}: invalid contentVersion`);
      if (metadata.status !== "published" && metadata.status !== "draft") errors.push(`${location}: status must be draft or published`);
      if (release && metadata.status !== "published") errors.push(`${location}: draft content cannot be released`);
      validateInstant(metadata.publishedAt, `${location}: publishedAt`, errors);
      validateInstant(metadata.updatedAt, `${location}: updatedAt`, errors);
      if (section === "history") validateHistoryDate(metadata, location, errors);
      if (!Array.isArray(metadata.categories) || metadata.categories.length === 0) errors.push(`${location}: categories are required`);
      if (!Array.isArray(metadata.tags) || metadata.tags.length === 0) errors.push(`${location}: tags are required`);
      if (!isHttps(metadata.canonicalUrl ?? "")) errors.push(`${location}: canonicalUrl must use HTTPS`);
      if (!Array.isArray(metadata.sources) || metadata.sources.length === 0) {
        errors.push(`${location}: at least one source is required`);
      } else {
        metadata.sources.forEach((source, index) => {
          if (!isHttps(source.url ?? "")) errors.push(`${location}: source ${index + 1} must use HTTPS`);
          if (!source.title || !source.publisher) errors.push(`${location}: source ${index + 1} requires title and publisher`);
        });
      }

      if (metadata.id) registerUnique(ids, metadata.id, location, "id", errors);
      if (isHttps(metadata.canonicalUrl ?? "")) {
        registerUnique(canonicalUrls, normalizeCanonicalUrl(metadata.canonicalUrl), location, "canonical URL", errors);
      }

      if (section === "videos") {
        if (metadata.provider !== "youtube") errors.push(`${location}: video provider must be youtube`);
        if (!youtubeIdPattern.test(metadata.providerId ?? "")) errors.push(`${location}: invalid YouTube ID`);
        else registerUnique(youtubeIds, metadata.providerId, location, "YouTube ID", errors);
        if (!metadata.channelTitle) errors.push(`${location}: channelTitle is required`);
        if (!Array.isArray(metadata.relatedProductIds)) errors.push(`${location}: relatedProductIds must be an array`);
      }

      const localized = new Map();
      for (const locale of config.requiredLocales) {
        const localePath = path.join(itemDirectory, "locales", `${locale}.json`);
        if (!fs.existsSync(localePath)) {
          errors.push(`${relativePath(itemDirectory)}: missing locale ${locale}`);
          continue;
        }
        try {
          const document = readJson(localePath);
          validateLocaleDocument(document, metadata, locale, relativePath(localePath), errors);
          if (fs.statSync(localePath).size > config.limits.articleBytes) {
            errors.push(`${relativePath(localePath)}: localized article is too large`);
          }
          localized.set(locale, { document, path: localePath });
          const fingerprint = `${section}:${normalizeText(document.title ?? "")}:${metadata.eventDate ?? metadata.publishedAt?.slice(0, 10) ?? ""}`;
          if (locale === config.defaultLocale) {
            registerUnique(titleFingerprints, fingerprint, location, "title/date fingerprint", warnings);
          }
        } catch (error) {
          errors.push(`${relativePath(localePath)}: invalid JSON (${error.message})`);
        }
      }

      const englishDocument = localized.get(config.defaultLocale)?.document;
      if (englishDocument) {
        const englishBody = localizedTextFingerprint(englishDocument);
        for (const [locale, value] of localized) {
          if (locale !== config.defaultLocale && englishBody.length > 80 && localizedTextFingerprint(value.document) === englishBody) {
            errors.push(`${relativePath(value.path)}: localized text duplicates the English article`);
          }
        }
      }

      if (!Array.isArray(metadata.assets)) {
        errors.push(`${location}: assets must be an array`);
      } else if (metadata.assets.length > 5) {
        errors.push(`${location}: at most five article images are supported`);
      }
      if (section !== "videos" && (!Array.isArray(metadata.assets) || metadata.assets.length === 0)) {
        errors.push(`${location}: news and history articles require at least one local image`);
      }
      for (const asset of metadata.assets ?? []) {
        validateAsset(asset, metadata, location, assets, errors);
      }
      if (section !== "videos") {
        if (!metadata.heroAssetId) errors.push(`${location}: heroAssetId is required`);
        else if (!metadata.assets?.some((asset) => asset.assetId === metadata.heroAssetId)) {
          errors.push(`${location}: heroAssetId must reference a local article image`);
        }
      }

      items.push({ section, metadata, metadataPath, itemDirectory, localized });
    }
  }

  if (release) {
    for (const section of sectionNames) {
      const count = items.filter((item) => item.section === section && item.metadata.status === "published").length;
      const minimum = config.minimumPublishedEntries[section];
      if (count < minimum) errors.push(`${section}: ${count} published entries, minimum is ${minimum}`);
    }
  }

  return { items, assets, errors, warnings };
}

function validateLocaleDocument(document, metadata, locale, location, errors) {
  if (document.schemaVersion !== config.articleSchemaVersion) errors.push(`${location}: unsupported schemaVersion`);
  if (document.id !== metadata.id) errors.push(`${location}: id does not match metadata`);
  if (document.locale !== locale) errors.push(`${location}: locale does not match filename`);
  if (document.basedOnVersion !== metadata.contentVersion) errors.push(`${location}: basedOnVersion is stale`);
  if (document.status !== "reviewed") errors.push(`${location}: status must be reviewed`);
  if (!document.title?.trim()) errors.push(`${location}: title is required`);
  if (!document.summary?.trim()) errors.push(`${location}: summary is required`);
  if (!document.author?.trim()) errors.push(`${location}: author is required`);
  if (!Array.isArray(document.searchKeywords)) errors.push(`${location}: searchKeywords must be an array`);
  if (!Array.isArray(document.blocks) || (metadata.type !== "videos" && document.blocks.length === 0)) {
    errors.push(`${location}: article blocks are required`);
    return;
  }
  if (document.blocks.length > 200) errors.push(`${location}: too many blocks`);
  document.blocks.forEach((block, index) => {
    if (!supportedBlockTypes.has(block.type)) errors.push(`${location}: unsupported block type at ${index}`);
    if (["paragraph", "heading", "quote", "callout"].includes(block.type) && !block.text?.trim()) {
      errors.push(`${location}: block ${index} requires text`);
    }
    if (block.type === "image" && !block.assetId) errors.push(`${location}: image block ${index} requires assetId`);
    if (block.type === "link" && !isHttps(block.url ?? "")) errors.push(`${location}: link block ${index} must use HTTPS`);
    if (block.type === "youtube" && !youtubeIdPattern.test(block.providerId ?? "")) {
      errors.push(`${location}: youtube block ${index} has invalid providerId`);
    }
  });
}

function validateHistoryDate(metadata, location, errors) {
  if (!/^\d{4}(-\d{2})?(-\d{2})?$/.test(metadata.eventDate ?? "")) errors.push(`${location}: invalid eventDate`);
  if (!["year", "month", "day"].includes(metadata.datePrecision)) errors.push(`${location}: invalid datePrecision`);
}

function validateInstant(value, label, errors) {
  if (typeof value !== "string" || Number.isNaN(Date.parse(value))) errors.push(`${label} is invalid`);
}

function validateAsset(asset, metadata, location, assets, errors) {
  if (!idPattern.test(asset.assetId ?? "")) {
    errors.push(`${location}: invalid assetId`);
    return;
  }
  if (assets.has(asset.assetId)) {
    errors.push(`${location}: duplicate assetId ${asset.assetId}`);
    return;
  }
  if (!asset.rights?.creator || !asset.rights?.sourceUrl || !asset.rights?.licenseId) {
    errors.push(`${location}: asset ${asset.assetId} requires complete rights metadata`);
  }
  if (typeof asset.rights?.downloadAllowed !== "boolean") errors.push(`${location}: asset ${asset.assetId} requires downloadAllowed`);
  if (!isHttps(asset.rights?.sourceUrl ?? "")) errors.push(`${location}: asset sourceUrl must use HTTPS`);
  const builtVariants = {};
  for (const [name, variant] of Object.entries(asset.variants ?? {})) {
    if (!/^(thumbnail|display|original)$/.test(name)) errors.push(`${location}: unsupported asset variant ${name}`);
    const filePath = path.join(repositoryRoot, variant.path ?? "");
    if (!isSafeRepositoryPath(variant.path) || !fs.existsSync(filePath)) {
      errors.push(`${location}: missing asset file ${variant.path}`);
      continue;
    }
    if (!Number.isInteger(variant.width) || variant.width < 1 || !Number.isInteger(variant.height) || variant.height < 1) {
      errors.push(`${location}: asset ${asset.assetId}/${name} requires dimensions`);
    }
    const maxBytes = name === "thumbnail" ? config.limits.thumbnailBytes : config.limits.imageBytes;
    if (fs.statSync(filePath).size > maxBytes) errors.push(`${location}: asset ${asset.assetId}/${name} is too large`);
    builtVariants[name] = fileReference(filePath, { width: variant.width, height: variant.height });
  }
  if (!builtVariants.thumbnail || !builtVariants.display) errors.push(`${location}: asset ${asset.assetId} requires thumbnail and display variants`);
  assets.set(asset.assetId, {
    assetId: asset.assetId,
    kind: "image",
    variants: builtVariants,
    rights: asset.rights,
    ownerContentId: metadata.id
  });
}

function localizedTextFingerprint(document) {
  const blockText = (document.blocks ?? []).flatMap((block) => [
    block.text ?? "",
    ...(Array.isArray(block.items) ? block.items : [])
  ]).join(" ");
  return normalizeText(`${document.summary ?? ""} ${blockText}`);
}

function isSafeRepositoryPath(value) {
  return typeof value === "string" &&
    /^[a-zA-Z0-9._/-]+$/.test(value) &&
    !value.startsWith("/") &&
    !value.includes("..") &&
    !value.includes("\\");
}

function registerUnique(map, key, location, label, messages) {
  const previous = map.get(key);
  if (previous) messages.push(`${location}: duplicate ${label}; first seen in ${previous}`);
  else map.set(key, location);
}

export function publishedItems(collection) {
  return collection.items.filter((item) => item.metadata.status === "published");
}

export function sortItems(items) {
  return [...items].sort((left, right) => {
    const priority = (right.metadata.priority ?? 0) - (left.metadata.priority ?? 0);
    if (priority !== 0) return priority;
    const leftDate = left.section === "history" ? left.metadata.eventDate : left.metadata.publishedAt;
    const rightDate = right.section === "history" ? right.metadata.eventDate : right.metadata.publishedAt;
    const dateOrder = rightDate.localeCompare(leftDate);
    return dateOrder !== 0 ? dateOrder : left.metadata.id.localeCompare(right.metadata.id);
  });
}
