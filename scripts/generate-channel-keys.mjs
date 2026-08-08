import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { repositoryRoot } from "./lib/hub-content.mjs";

const keyDirectory = path.join(repositoryRoot, ".workspace", "keys");
const privateKeyPath = path.join(keyDirectory, "channel-private.pem");
const publicKeyPath = path.join(repositoryRoot, "channels", "channel-public.pem");

if (fs.existsSync(privateKeyPath) || fs.existsSync(publicKeyPath)) {
  throw new Error("Channel keys already exist");
}

const { privateKey, publicKey } = crypto.generateKeyPairSync("ec", {
  namedCurve: "prime256v1",
  privateKeyEncoding: { type: "pkcs8", format: "pem" },
  publicKeyEncoding: { type: "spki", format: "pem" }
});
fs.mkdirSync(keyDirectory, { recursive: true });
fs.mkdirSync(path.dirname(publicKeyPath), { recursive: true });
fs.writeFileSync(privateKeyPath, privateKey, { encoding: "utf8", mode: 0o600 });
fs.writeFileSync(publicKeyPath, publicKey, "utf8");
process.stdout.write("Generated the local channel signing key and public verification key.\n");

