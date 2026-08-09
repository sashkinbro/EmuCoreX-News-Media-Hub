# EmuCore - Hub

This repository is the public content source for EmuCore - Hub. It contains localized news, curated videos, PlayStation 2 history articles, PS2 emulation manuals, shared media, generated locale indexes, and the root manifest consumed by supported EmuCore applications.

Repository updates are published only by SBRO. Pull requests may be proposed and are reviewed individually; submission does not guarantee publication.

## Content layout

```text
channels/stable-v1.json
channels/stable-v1.sig
catalog/v1/
  manifest.json
  indexes/<locale>/<section>-NNNN.json
  assets.json
  tombstones.json
content/
  news/<id>/
  videos/<id>/
  history/<id>/
  manuals/<id>/
assets/<type>/<id>/
schemas/v1/
scripts/
```

Every content directory contains one shared `metadata.json` file and a `locales` directory with a complete localized document for every locale listed in `hub.config.json`. Images are shared by all locales and must include source, attribution, and license information in metadata.

Generated files in `catalog/` must not be edited manually. They are rebuilt from `content/` by the repository scripts. The stable channel points to a catalog at an immutable Git commit and is published only after that commit passes validation.

## Finding existing content

Before adding an item, search by all of the following:

- canonical source URL;
- normalized title;
- YouTube video ID for video entries;
- event date and subject for history entries;
- aliases, product names, emulator names, and platform tags.

The validator rejects duplicate IDs, canonical URLs, YouTube IDs, and strong title/date matches.

## Adding content

1. Copy the appropriate template from the local editorial workspace.
2. Choose a stable lowercase ID that will never be reused.
3. Add shared metadata, verified sources, and legally usable media.
4. Add complete locale files for every required language.
5. Run `npm run build` to rebuild locale indexes, the asset catalog, and the manifest.
6. Run `npm run validate` before publication.

Articles must be original summaries based on cited sources. Full copies of third-party articles, magazines, or transcripts are not accepted. Facts, dates, names, and source URLs must be verified. Article images are stored in this repository; media without clear redistribution permission must not be copied and must be replaced with a relevant, appropriately licensed asset.

## Editorial requirements

- News covers development, architecture, testing, performance, rendering, compatibility, research, tooling, or a meaningful event in emulation. Version-number announcements, release notes, nightly-build notices, and changelog rewrites are not News.
- News and PlayStation 2 History entries are complete features with multiple sections and substantive paragraphs. A date, a short fact, or a padded template is not publishable.
- Manuals include prerequisites, numbered actions, a verification checkpoint, rollback guidance, and common failure modes. They do not provide copyrighted firmware or game data and do not direct readers to unauthorized downloads.
- Video entries are watched before publication. The provider ID, exact title, channel, date, duration, subject, and availability are verified; misleading, unrelated, unavailable, duplicated, or piracy-promoting videos are rejected.
- Every article hero must be locally stored, directly relevant, visibly distinct from other catalog media, and accompanied by source, creator, license, attribution, modification, and download-permission metadata. Additional images are used only when they add evidence or useful context.
- Every required locale is a complete translation of the same article. Machine output or placeholder text remains a draft until terminology, protected names, code identifiers, meaning, and natural phrasing have been reviewed.

## Validation

```bash
npm run build
npm run validate
npm run audit:sources
npm run audit:youtube
```

Validation checks schema versions, required locales, file references, checksums, dates, HTTPS URLs, media limits, duplicate detection, and the minimum published catalog size for each main section. The source audit additionally verifies that canonical references, editorial sources, and media-rights pages have not disappeared.

The YouTube audit uses the public oEmbed endpoint to confirm that every curated video is still available, its stored title and channel match the provider, its embed metadata contains the expected video ID, and its preview image is reachable. It does not download video or audio.
