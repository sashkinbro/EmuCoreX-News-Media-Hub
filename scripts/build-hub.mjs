import fs from "node:fs";
import path from "node:path";
import {
  collectContent,
  config,
  fileReference,
  publishedItems,
  readJson,
  repositoryRoot,
  sectionNames,
  sortItems,
  writeJson
} from "./lib/hub-content.mjs";

const collection = collectContent({ release: false });
if (collection.errors.length > 0) {
  process.stderr.write(`${collection.errors.join("\n")}\n`);
  process.exit(1);
}

const catalogRoot = path.join(repositoryRoot, "catalog", "v1");
fs.rmSync(catalogRoot, { recursive: true, force: true });
fs.mkdirSync(catalogRoot, { recursive: true });

const releasedItems = publishedItems(collection);
const indexes = Object.fromEntries(sectionNames.map((section) => [section, {}]));

for (const locale of config.requiredLocales) {
  for (const section of sectionNames) {
    const items = sortItems(releasedItems.filter((item) => item.section === section));
    const pageSize = Math.max(1, config.indexPageSize ?? 100);
    const chunks = items.length === 0
      ? [[]]
      : Array.from({ length: Math.ceil(items.length / pageSize) }, (_, index) => items.slice(index * pageSize, (index + 1) * pageSize));
    indexes[section][locale] = chunks.map((chunk, index) => {
      const pageNumber = index + 1;
      const pagePath = path.join(catalogRoot, "indexes", locale, `${section}-${String(pageNumber).padStart(4, "0")}.json`);
      const page = {
        formatVersion: config.catalogSchemaVersion,
        releaseId: config.releaseId,
        catalogRevision: config.catalogRevision,
        kind: section,
        locale,
        page: pageNumber,
        next: null,
        items: chunk.map((item) => toIndexItem(item, locale))
      };
      writeJson(pagePath, page);
      return fileReference(pagePath, { itemCount: page.items.length });
    });
  }
}

const assetsPath = path.join(catalogRoot, "assets.json");
writeJson(assetsPath, {
  formatVersion: 1,
  releaseId: config.releaseId,
  assets: [...collection.assets.values()].sort((left, right) => left.assetId.localeCompare(right.assetId))
});

const sourceTombstones = readJson(path.join(repositoryRoot, "tombstones.json"));
const tombstonesPath = path.join(catalogRoot, "tombstones.json");
writeJson(tombstonesPath, {
  ...sourceTombstones,
  releaseId: config.releaseId
});

const manifestPath = path.join(catalogRoot, "manifest.json");
writeJson(manifestPath, {
  formatVersion: config.catalogSchemaVersion,
  schemaRevision: 1,
  releaseId: config.releaseId,
  catalogRevision: config.catalogRevision,
  generatedAt: config.generatedAt,
  defaultLocale: config.defaultLocale,
  supportedLocales: config.requiredLocales,
  localeFallbacks: Object.fromEntries(config.requiredLocales.map((locale) => [locale, locale === "en" ? ["en"] : [locale, "en"]])),
  minimumClientVersionCode: config.minimumClientVersionCode,
  counts: Object.fromEntries(sectionNames.map((section) => [section, releasedItems.filter((item) => item.section === section).length])),
  indexes,
  assets: fileReference(assetsPath),
  tombstones: fileReference(tombstonesPath)
});

for (const warning of collection.warnings) process.stderr.write(`warning: ${warning}\n`);
process.stdout.write(`Built ${releasedItems.length} published items for ${config.requiredLocales.length} locales.\n`);

function toIndexItem(item, locale) {
  const metadata = item.metadata;
  const localized = item.localized.get(locale);
  if (!localized) throw new Error(`Missing ${locale} locale for ${metadata.id}`);
  const common = {
    id: metadata.id,
    kind: item.section,
    contentVersion: metadata.contentVersion,
    localeVersion: localized.document.localeVersion,
    title: localized.document.title,
    summary: localized.document.summary,
    publishedAt: metadata.publishedAt,
    updatedAt: metadata.updatedAt,
    categoryIds: metadata.categories,
    tagIds: metadata.tags,
    relatedProductIds: metadata.relatedProductIds ?? [],
    featured: metadata.featured === true,
    priority: metadata.priority ?? 0,
    canonicalUrl: metadata.canonicalUrl,
    heroAssetId: metadata.heroAssetId ?? null,
    document: fileReference(localized.path),
    sourceCount: metadata.sources.length,
    sources: metadata.sources
  };
  if (item.section === "history") {
    return {
      ...common,
      eventDate: metadata.eventDate,
      datePrecision: metadata.datePrecision,
      year: Number(metadata.eventDate.slice(0, 4))
    };
  }
  if (item.section === "videos") {
    return {
      ...common,
      provider: metadata.provider,
      providerId: metadata.providerId,
      channelTitle: metadata.channelTitle,
      sourceLanguage: metadata.sourceLanguage,
      thumbnailUrl: `https://i.ytimg.com/vi/${metadata.providerId}/hqdefault.jpg`
    };
  }
  return common;
}
