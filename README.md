# EmuCore - Hub

This repository is the public content source for EmuCore - Hub. It contains localized news, curated videos, PlayStation 2 history articles, shared media, generated locale indexes, and the root manifest consumed by supported EmuCore applications.

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

## Validation

```bash
npm run build
npm run validate
```

Validation checks schema versions, required locales, file references, checksums, dates, HTTPS URLs, media limits, duplicate detection, and the minimum published catalog size for each main section.
