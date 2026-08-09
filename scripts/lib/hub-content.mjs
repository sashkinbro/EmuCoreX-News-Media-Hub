import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptsDirectory = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
export const repositoryRoot = path.dirname(scriptsDirectory);
export const config = readJson(path.join(repositoryRoot, "hub.config.json"));
export const sectionNames = ["news", "videos", "history", "manuals"];

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
  const imageHashes = new Map();
  const referencedAssetPaths = new Set();
  const defaultLocaleBodies = new Map(sectionNames.map((section) => [section, []]));

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
      if (hasLikelyMojibake(metadata)) errors.push(`${location}: text contains replacement characters or likely mojibake`);
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
          if (source.publishedAt !== undefined) {
            validateInstant(source.publishedAt, `${location}: source ${index + 1} publishedAt`, errors);
          }
        });
      }

      if (metadata.id) registerUnique(ids, metadata.id, location, "id", errors);
      // A repeated canonical URL is a reliable duplicate signal for news and
      // videos. History features and manuals may legitimately build distinct
      // editorial articles from the same primary documentation, so their body,
      // title and block-level duplicate checks remain authoritative instead.
      if (["news", "videos"].includes(section) && isHttps(metadata.canonicalUrl ?? "")) {
        registerUnique(canonicalUrls, normalizeCanonicalUrl(metadata.canonicalUrl), location, "canonical URL", errors);
      }

      if (section === "videos") {
        if (metadata.provider !== "youtube") errors.push(`${location}: video provider must be youtube`);
        if (!youtubeIdPattern.test(metadata.providerId ?? "")) errors.push(`${location}: invalid YouTube ID`);
        else registerUnique(youtubeIds, metadata.providerId, location, "YouTube ID", errors);
        if (!metadata.channelTitle) errors.push(`${location}: channelTitle is required`);
        if (!Array.isArray(metadata.relatedProductIds)) errors.push(`${location}: relatedProductIds must be an array`);
      }
      if (section === "news") {
        if (metadata.releaseChannel !== "editorial") {
          errors.push(`${location}: News must be an editorial development story, not a release channel entry`);
        }
        const releaseLikeTags = (metadata.tags ?? []).some((tag) =>
          /^(release|update|version|build|nightly|pre-release)$/i.test(String(tag))
        );
        if (releaseLikeTags) errors.push(`${location}: release/update announcement tags are not allowed in News`);
      }

      const localized = new Map();
      const localizedFingerprints = new Map();
      for (const locale of config.requiredLocales) {
        const localePath = path.join(itemDirectory, "locales", `${locale}.json`);
        if (!fs.existsSync(localePath)) {
          errors.push(`${relativePath(itemDirectory)}: missing locale ${locale}`);
          continue;
        }
        try {
          const document = readJson(localePath);
          validateLocaleDocument(document, metadata, locale, relativePath(localePath), errors);
          if (fs.statSync(localePath).size > config.limits.documentBytes) {
            errors.push(`${relativePath(localePath)}: localized article is too large`);
          }
          localized.set(locale, { document, path: localePath });
          const localizedFingerprint = localizedTextFingerprint(document);
          if (localizedFingerprint.length > 80) {
            registerUnique(
              localizedFingerprints,
              localizedFingerprint,
              relativePath(localePath),
              "localized article body",
              errors
            );
          }
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
        if (section === "news" && /\b(?:released?|update[sd]?|version|pre-release|nightly build)\b/i.test(englishDocument.title ?? "")) {
          errors.push(`${relativePath(localized.get(config.defaultLocale).path)}: release/update announcements are not allowed in News`);
        }
        if (englishBody.length > 80) {
          defaultLocaleBodies.get(section).push({ location, text: englishBody });
        }
        for (const [locale, value] of localized) {
          if (locale !== config.defaultLocale) {
            validateLocalizedStructure(englishDocument, value.document, relativePath(value.path), errors);
            if (englishBody.length > 80 && localizedTextFingerprint(value.document) === englishBody) {
              errors.push(`${relativePath(value.path)}: localized text duplicates the English article`);
            }
          }
        }
      }

      if (!Array.isArray(metadata.assets)) {
        errors.push(`${location}: assets must be an array`);
      } else if (metadata.assets.length > 5) {
        errors.push(`${location}: at most five article images are supported`);
      }
      if (section !== "videos" && (!Array.isArray(metadata.assets) || metadata.assets.length === 0)) {
        errors.push(`${location}: news, history, and manual articles require at least one local image`);
      }
      for (const asset of metadata.assets ?? []) {
        validateAsset(asset, metadata, location, assets, imageHashes, referencedAssetPaths, errors);
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

  validateNearDuplicateArticles(defaultLocaleBodies, errors);
  validateNoOrphanedMedia(referencedAssetPaths, errors);

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
  if (hasLikelyMojibake(document)) {
    errors.push(`${location}: text contains replacement characters or likely mojibake`);
  }
  if (!Array.isArray(document.searchKeywords)) errors.push(`${location}: searchKeywords must be an array`);
  if (!Array.isArray(document.blocks) || (metadata.type !== "videos" && document.blocks.length === 0)) {
    errors.push(`${location}: article blocks are required`);
    return;
  }
  validateLocaleScript(document, locale, location, errors);
  if (document.blocks.length > 200) errors.push(`${location}: too many blocks`);
  const repeatedBlockText = new Map();
  document.blocks.forEach((block, index) => {
    const fingerprint = normalizeText([block.text ?? "", ...(block.items ?? [])].join(" "));
    if (fingerprint.length < 80) return;
    if (repeatedBlockText.has(fingerprint)) {
      errors.push(`${location}: block ${index} duplicates block ${repeatedBlockText.get(fingerprint)}`);
    } else {
      repeatedBlockText.set(fingerprint, index);
    }
  });
  const paragraphs = document.blocks.filter((block) => block.type === "paragraph");
  if (metadata.type === "videos") {
    if (paragraphs.length < 2) errors.push(`${location}: video descriptions require at least two paragraphs`);
  } else {
    const headings = document.blocks.filter((block) => block.type === "heading");
    if (document.blocks.length < 7) errors.push(`${location}: full articles require at least seven content blocks`);
    if (paragraphs.length < 4) errors.push(`${location}: full articles require at least four paragraphs`);
    if (headings.length < 2) errors.push(`${location}: full articles require at least two sections`);
    const textLength = localizedTextFingerprint(document).length;
    const minimumLength = ["ja", "ko", "zh"].includes(locale) ? 600 : 1200;
    if (textLength < minimumLength) errors.push(`${location}: article text is too short (${textLength}/${minimumLength})`);
  }
  document.blocks.forEach((block, index) => {
    if (!supportedBlockTypes.has(block.type)) errors.push(`${location}: unsupported block type at ${index}`);
    if (["paragraph", "heading", "quote", "callout"].includes(block.type) && !block.text?.trim()) {
      errors.push(`${location}: block ${index} requires text`);
    }
    if (block.type === "image") {
      if (!block.assetId) errors.push(`${location}: image block ${index} requires assetId`);
      else if (!metadata.assets?.some((asset) => asset.assetId === block.assetId)) {
        errors.push(`${location}: image block ${index} references an unknown article asset`);
      }
    }
    if (block.type === "gallery") {
      if (!Array.isArray(block.assetIds) || block.assetIds.length < 2 || block.assetIds.length > 5) {
        errors.push(`${location}: gallery block ${index} requires two to five assetIds`);
      } else {
        for (const assetId of block.assetIds) {
          if (!metadata.assets?.some((asset) => asset.assetId === assetId)) {
            errors.push(`${location}: gallery block ${index} references an unknown article asset`);
          }
        }
      }
    }
    if (block.type === "link" && !isHttps(block.url ?? "")) errors.push(`${location}: link block ${index} must use HTTPS`);
    if (block.type === "youtube" && !youtubeIdPattern.test(block.providerId ?? "")) {
      errors.push(`${location}: youtube block ${index} has invalid providerId`);
    }
  });
  if (metadata.type !== "videos") {
    const referencedAssets = new Set(document.blocks.flatMap((block) => {
      if (block.type === "image" && block.assetId) return [block.assetId];
      if (block.type === "gallery" && Array.isArray(block.assetIds)) return block.assetIds;
      return [];
    }));
    for (const asset of metadata.assets ?? []) {
      if (asset.assetId !== metadata.heroAssetId && !referencedAssets.has(asset.assetId)) {
        errors.push(`${location}: additional asset ${asset.assetId} is not used by the article`);
      }
    }
  }
}

function validateHistoryDate(metadata, location, errors) {
  if (!/^\d{4}(-\d{2})?(-\d{2})?$/.test(metadata.eventDate ?? "")) errors.push(`${location}: invalid eventDate`);
  if (!["year", "month", "day"].includes(metadata.datePrecision)) errors.push(`${location}: invalid datePrecision`);
}

function validateInstant(value, label, errors) {
  const rfc3339 = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/;
  if (typeof value !== "string" || !rfc3339.test(value) || Number.isNaN(Date.parse(value))) {
    errors.push(`${label} must be an RFC 3339 date-time`);
  }
}

function validateAsset(asset, metadata, location, assets, imageHashes, referencedAssetPaths, errors) {
  if (!idPattern.test(asset.assetId ?? "")) {
    errors.push(`${location}: invalid assetId`);
    return;
  }
  if (assets.has(asset.assetId)) {
    errors.push(`${location}: duplicate assetId ${asset.assetId}`);
    return;
  }
  if (
    !asset.rights?.creator ||
    !asset.rights?.sourceUrl ||
    !asset.rights?.licenseId ||
    !asset.rights?.licenseUrl ||
    !asset.rights?.attribution
  ) {
    errors.push(`${location}: asset ${asset.assetId} requires complete rights metadata`);
  }
  if (typeof asset.rights?.modified !== "boolean") errors.push(`${location}: asset ${asset.assetId} requires modified`);
  if (typeof asset.rights?.downloadAllowed !== "boolean") errors.push(`${location}: asset ${asset.assetId} requires downloadAllowed`);
  if (!isHttps(asset.rights?.sourceUrl ?? "")) errors.push(`${location}: asset sourceUrl must use HTTPS`);
  if (!isHttps(asset.rights?.licenseUrl ?? "")) errors.push(`${location}: asset licenseUrl must use HTTPS`);
  const builtVariants = {};
  for (const [name, variant] of Object.entries(asset.variants ?? {})) {
    if (!/^(thumbnail|display|original)$/.test(name)) errors.push(`${location}: unsupported asset variant ${name}`);
    const filePath = path.join(repositoryRoot, variant.path ?? "");
    if (!isSafeRepositoryPath(variant.path) || !fs.existsSync(filePath)) {
      errors.push(`${location}: missing asset file ${variant.path}`);
      continue;
    }
    referencedAssetPaths.add(relativePath(filePath));
    if (!Number.isInteger(variant.width) || variant.width < 1 || !Number.isInteger(variant.height) || variant.height < 1) {
      errors.push(`${location}: asset ${asset.assetId}/${name} requires dimensions`);
    } else {
      const actualDimensions = readImageDimensions(filePath);
      if (!actualDimensions) {
        errors.push(`${location}: asset ${asset.assetId}/${name} has an unreadable image header`);
      } else if (actualDimensions.width !== variant.width || actualDimensions.height !== variant.height) {
        errors.push(
          `${location}: asset ${asset.assetId}/${name} dimensions are ` +
          `${actualDimensions.width}x${actualDimensions.height}, metadata declares ${variant.width}x${variant.height}`
        );
      }
    }
    const maxBytes = name === "thumbnail" ? config.limits.thumbnailBytes : config.limits.imageBytes;
    if (fs.statSync(filePath).size > maxBytes) errors.push(`${location}: asset ${asset.assetId}/${name} is too large`);
    builtVariants[name] = fileReference(filePath, { width: variant.width, height: variant.height });
  }
  if (!builtVariants.thumbnail || !builtVariants.display) errors.push(`${location}: asset ${asset.assetId} requires thumbnail and display variants`);
  if (builtVariants.display) {
    registerUnique(imageHashes, builtVariants.display.sha256, location, "display image hash", errors);
  }
  assets.set(asset.assetId, {
    assetId: asset.assetId,
    kind: "image",
    variants: builtVariants,
    rights: asset.rights,
    ownerContentId: metadata.id
  });
}

function validateNoOrphanedMedia(referencedAssetPaths, errors) {
  const assetsRoot = path.join(repositoryRoot, "assets");
  if (!fs.existsSync(assetsRoot)) return;
  const pending = [assetsRoot];
  while (pending.length > 0) {
    const current = pending.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const entryPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        pending.push(entryPath);
      } else if (/\.(?:webp|png|jpe?g)$/i.test(entry.name)) {
        const assetPath = relativePath(entryPath);
        if (!referencedAssetPaths.has(assetPath)) {
          errors.push(`${assetPath}: orphaned media is not referenced by any content item`);
        }
      }
    }
  }
}

function hasLikelyMojibake(value) {
  return /\uFFFD|(?:QXZ|ZXQ)|(?:Ã[‚ƒ„…†‡ˆ‰Š‹ŒŽ‘’“”•–—˜™š›œžŸ©®±¼½¾¿]|Â[©®±¼½¾¿]|В©|вЂ)/u.test(JSON.stringify(value));
}

function validateLocaleScript(document, locale, location, errors) {
  const patterns = {
    ar: /\p{Script=Arabic}/gu,
    fa: /\p{Script=Arabic}/gu,
    hi: /\p{Script=Devanagari}/gu,
    ja: /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]/gu,
    ko: /\p{Script=Hangul}/gu,
    ru: /\p{Script=Cyrillic}/gu,
    uk: /\p{Script=Cyrillic}/gu,
    zh: /\p{Script=Han}/gu
  };
  const pattern = patterns[locale];
  if (!pattern) return;
  const text = [
    document.title ?? "",
    document.summary ?? "",
    document.heroAlt ?? "",
    ...(document.blocks ?? []).flatMap((block) => [block.text ?? "", ...(block.items ?? [])])
  ].join(" ");
  const scriptCharacters = text.match(pattern)?.length ?? 0;
  if (scriptCharacters < 25) errors.push(`${location}: localized text does not contain enough ${locale} script characters`);
}

function validateLocalizedStructure(englishDocument, document, location, errors) {
  const englishBlocks = englishDocument.blocks ?? [];
  const localizedBlocks = document.blocks ?? [];
  if (localizedBlocks.length !== englishBlocks.length) {
    errors.push(`${location}: localized block count does not match English`);
    return;
  }
  for (let index = 0; index < englishBlocks.length; index += 1) {
    const englishBlock = englishBlocks[index];
    const localizedBlock = localizedBlocks[index];
    if (localizedBlock.type !== englishBlock.type) {
      errors.push(`${location}: block ${index} type does not match English`);
      continue;
    }
    if (englishBlock.type === "youtube" && localizedBlock.providerId !== englishBlock.providerId) {
      errors.push(`${location}: block ${index} providerId does not match English`);
    }
    if (englishBlock.type === "image" && localizedBlock.assetId !== englishBlock.assetId) {
      errors.push(`${location}: block ${index} assetId does not match English`);
    }
    if (englishBlock.type === "gallery" && JSON.stringify(localizedBlock.assetIds) !== JSON.stringify(englishBlock.assetIds)) {
      errors.push(`${location}: block ${index} gallery assets do not match English`);
    }
    if (englishBlock.type === "link" && localizedBlock.url !== englishBlock.url) {
      errors.push(`${location}: block ${index} URL does not match English`);
    }
  }
}

function localizedTextFingerprint(document) {
  const blockText = (document.blocks ?? []).flatMap((block) => [
    block.text ?? "",
    ...(Array.isArray(block.items) ? block.items : [])
  ]).join(" ");
  return normalizeText(`${document.summary ?? ""} ${blockText}`);
}

function validateNearDuplicateArticles(defaultLocaleBodies, errors) {
  for (const [section, articles] of defaultLocaleBodies) {
    const shingles = articles.map((article) => ({
      ...article,
      shingles: wordShingles(article.text, 5)
    }));
    for (let leftIndex = 0; leftIndex < shingles.length; leftIndex += 1) {
      for (let rightIndex = leftIndex + 1; rightIndex < shingles.length; rightIndex += 1) {
        const left = shingles[leftIndex];
        const right = shingles[rightIndex];
        const similarity = jaccardSimilarity(left.shingles, right.shingles);
        if (similarity >= 0.72) {
          errors.push(
            `${right.location}: near-duplicate ${section} article body (${Math.round(similarity * 100)}%); ` +
            `too similar to ${left.location}`
          );
        }
      }
    }
  }
}

function wordShingles(text, size) {
  const words = text.split(" ").filter(Boolean);
  const result = new Set();
  for (let index = 0; index <= words.length - size; index += 1) {
    result.add(words.slice(index, index + size).join(" "));
  }
  return result;
}

function jaccardSimilarity(left, right) {
  if (left.size === 0 || right.size === 0) return 0;
  let intersection = 0;
  const smaller = left.size <= right.size ? left : right;
  const larger = smaller === left ? right : left;
  for (const value of smaller) if (larger.has(value)) intersection += 1;
  return intersection / (left.size + right.size - intersection);
}

function isSafeRepositoryPath(value) {
  return typeof value === "string" &&
    /^[a-zA-Z0-9._/-]+$/.test(value) &&
    !value.startsWith("/") &&
    !value.includes("..") &&
    !value.includes("\\");
}

function readImageDimensions(filePath) {
  const bytes = fs.readFileSync(filePath);
  if (bytes.length < 30 || bytes.toString("ascii", 0, 4) !== "RIFF" || bytes.toString("ascii", 8, 12) !== "WEBP") {
    return null;
  }
  let offset = 12;
  while (offset + 8 <= bytes.length) {
    const chunkType = bytes.toString("ascii", offset, offset + 4);
    const chunkSize = bytes.readUInt32LE(offset + 4);
    const dataOffset = offset + 8;
    if (dataOffset + chunkSize > bytes.length) return null;
    if (chunkType === "VP8X" && chunkSize >= 10) {
      return {
        width: 1 + bytes.readUIntLE(dataOffset + 4, 3),
        height: 1 + bytes.readUIntLE(dataOffset + 7, 3)
      };
    }
    if (chunkType === "VP8 " && chunkSize >= 10 && bytes.readUIntLE(dataOffset + 3, 3) === 0x2a019d) {
      return {
        width: bytes.readUInt16LE(dataOffset + 6) & 0x3fff,
        height: bytes.readUInt16LE(dataOffset + 8) & 0x3fff
      };
    }
    if (chunkType === "VP8L" && chunkSize >= 5 && bytes[dataOffset] === 0x2f) {
      const b0 = bytes[dataOffset + 1];
      const b1 = bytes[dataOffset + 2];
      const b2 = bytes[dataOffset + 3];
      const b3 = bytes[dataOffset + 4];
      return {
        width: 1 + b0 + ((b1 & 0x3f) << 8),
        height: 1 + (b1 >> 6) + (b2 << 2) + ((b3 & 0x0f) << 10)
      };
    }
    offset = dataOffset + chunkSize + (chunkSize % 2);
  }
  return null;
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
    const leftDate = left.section === "history" ? left.metadata.eventDate : left.metadata.publishedAt;
    const rightDate = right.section === "history" ? right.metadata.eventDate : right.metadata.publishedAt;
    const dateOrder = rightDate.localeCompare(leftDate);
    if (dateOrder !== 0) return dateOrder;
    const priority = (right.metadata.priority ?? 0) - (left.metadata.priority ?? 0);
    return priority !== 0 ? priority : left.metadata.id.localeCompare(right.metadata.id);
  });
}
