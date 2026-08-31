import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import {
  config,
  fileReference,
  repositoryRoot,
  writeJson
} from "./lib/hub-content.mjs";

const commit = process.argv[2];
if (!/^[0-9a-f]{40}$/.test(commit ?? "")) throw new Error("A full 40-character content commit SHA is required");

const manifestPath = path.join(repositoryRoot, "catalog", "v1", "manifest.json");
const privateKeyPath = path.join(repositoryRoot, "channel-private.pem");
if (!fs.existsSync(manifestPath)) throw new Error("Build the catalog before publishing the channel");
if (!fs.existsSync(privateKeyPath)) throw new Error("The local channel signing key is missing");

const privateKey = crypto.createPrivateKey(fs.readFileSync(privateKeyPath));
const publicKey = crypto.createPublicKey(fs.readFileSync(path.join(repositoryRoot, "channels", "channel-public.pem")));
const keyFormat = { type: "spki", format: "der" };
if (!crypto.createPublicKey(privateKey).export(keyFormat).equals(publicKey.export(keyFormat))) {
  throw new Error("The signing key does not match this channel's public key");
}

const manifestReference = fileReference(manifestPath);
const channelPath = path.join(repositoryRoot, "channels", "stable-v1.json");
writeJson(channelPath, {
  formatVersion: config.channelFormatVersion,
  channel: config.channel,
  releaseId: config.releaseId,
  catalogRevision: config.catalogRevision,
  commit,
  issuedAt: config.generatedAt,
  minimumClientVersionCode: config.minimumClientVersionCode,
  manifest: {
    url: `https://raw.githubusercontent.com/sashkinbro/EmuCoreX-News-Media-Hub/${commit}/catalog/v1/manifest.json`,
    sha256: manifestReference.sha256,
    bytes: manifestReference.bytes,
    contentType: manifestReference.contentType
  }
});

const channelBytes = fs.readFileSync(channelPath);
const signature = crypto.sign("sha256", channelBytes, privateKey);
fs.writeFileSync(path.join(repositoryRoot, "channels", "stable-v1.sig"), `${signature.toString("base64")}\n`, "utf8");
process.stdout.write(`Published signed channel metadata for ${commit}.\n`);
