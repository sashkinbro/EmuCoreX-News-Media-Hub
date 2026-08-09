import fs from "node:fs";
import path from "node:path";
import { repositoryRoot, sectionNames } from "./lib/hub-content.mjs";

const references = new Map();

for (const section of sectionNames) {
  const sectionRoot = path.join(repositoryRoot, "content", section);
  if (!fs.existsSync(sectionRoot)) continue;

  for (const entry of fs.readdirSync(sectionRoot, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const metadataPath = path.join(sectionRoot, entry.name, "metadata.json");
    if (!fs.existsSync(metadataPath)) continue;

    const metadata = JSON.parse(fs.readFileSync(metadataPath, "utf8"));
    addReference(metadata.canonicalUrl, `${section}/${entry.name}: canonical URL`);
    for (const source of metadata.sources ?? []) {
      addReference(source.url, `${section}/${entry.name}: source ${source.id ?? source.title ?? "unnamed"}`);
    }
    for (const asset of metadata.assets ?? []) {
      addReference(asset.rights?.sourceUrl, `${section}/${entry.name}: asset ${asset.assetId} source`);
      addReference(asset.rights?.licenseUrl, `${section}/${entry.name}: asset ${asset.assetId} license`);
    }
  }
}

const queue = [...references.entries()];
const failures = [];
let verified = 0;
let cursor = 0;

await Promise.all(Array.from({ length: Math.min(8, queue.length) }, async () => {
  while (cursor < queue.length) {
    const index = cursor++;
    const [url, locations] = queue[index];
    const result = await checkUrl(url);
    if (result.ok) {
      verified += 1;
    } else {
      failures.push(`${url} (${result.reason})\n  ${locations.join("\n  ")}`);
    }
  }
}));

if (failures.length > 0) {
  process.stderr.write(`Source audit failed for ${failures.length} URL(s):\n${failures.join("\n")}\n`);
  process.exit(1);
}

process.stdout.write(`Verified ${verified} unique catalog source URLs.\n`);

function addReference(url, location) {
  if (!url) return;
  const locations = references.get(url) ?? [];
  locations.push(location);
  references.set(url, locations);
}

async function checkUrl(url) {
  for (let attempt = 1; attempt <= 2; attempt += 1) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20_000);
    try {
      const response = await fetch(url, {
        redirect: "follow",
        signal: controller.signal,
        headers: {
          "User-Agent": "EmuCore-Hub-source-audit/1.0",
          Range: "bytes=0-0"
        }
      });
      await response.body?.cancel();
      if (response.status === 404 || response.status === 410) {
        return { ok: false, reason: `HTTP ${response.status}` };
      }
      if (response.status >= 200 && response.status < 500) return { ok: true };
      if (attempt === 2) return { ok: false, reason: `HTTP ${response.status}` };
    } catch (error) {
      if (attempt === 2) return { ok: false, reason: error.name === "AbortError" ? "timeout" : error.message };
    } finally {
      clearTimeout(timeout);
    }
  }
  return { ok: false, reason: "unknown error" };
}
