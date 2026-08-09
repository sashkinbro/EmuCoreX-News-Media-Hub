import fs from "node:fs";
import path from "node:path";
import { readJson, repositoryRoot } from "./lib/hub-content.mjs";

const videosRoot = path.join(repositoryRoot, "content", "videos");
const directories = fs.existsSync(videosRoot)
  ? fs.readdirSync(videosRoot, { withFileTypes: true }).filter((entry) => entry.isDirectory())
  : [];

const videos = directories.map((entry) => readJson(path.join(videosRoot, entry.name, "metadata.json")));
const results = await Promise.all(videos.map(auditVideo));
const failures = results.filter((result) => result.error);

for (const result of failures) process.stderr.write(`${result.id}: ${result.error}\n`);
if (failures.length > 0) process.exit(1);
process.stdout.write(`Verified ${results.length} YouTube videos, embed metadata, and preview images.\n`);

async function auditVideo(metadata) {
  const id = metadata.providerId ?? metadata.id;
  try {
    const endpoint = new URL("https://www.youtube.com/oembed");
    endpoint.searchParams.set("url", metadata.canonicalUrl);
    endpoint.searchParams.set("format", "json");
    const response = await fetch(endpoint, { signal: AbortSignal.timeout(15_000) });
    if (!response.ok) return { id, error: `oEmbed returned HTTP ${response.status}` };
    const provider = await response.json();
    const expectedTitle = metadata.sources?.[0]?.title?.trim();
    const expectedChannel = metadata.channelTitle?.trim();
    if (provider.title?.trim() !== expectedTitle) return { id, error: "stored title does not match YouTube" };
    if (provider.author_name?.trim() !== expectedChannel) return { id, error: "stored channel does not match YouTube" };
    if (!provider.html?.includes(metadata.providerId)) return { id, error: "oEmbed player does not contain the stored provider ID" };
    if (!provider.thumbnail_url) return { id, error: "YouTube did not return a preview image" };

    const thumbnail = await fetch(provider.thumbnail_url, { signal: AbortSignal.timeout(15_000) });
    const thumbnailType = thumbnail.headers.get("content-type") ?? "";
    await thumbnail.body?.cancel();
    if (!thumbnail.ok) return { id, error: `preview image returned HTTP ${thumbnail.status}` };
    if (!thumbnailType.startsWith("image/")) return { id, error: `preview returned ${thumbnailType || "an unknown content type"}` };
    return { id };
  } catch (error) {
    return { id, error: error.message };
  }
}
