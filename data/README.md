# Image pairs and captions

The full benchmark has **1,000 source–target pairs**, with local IDs `0`–`999`.
The default configuration uses the original **100-pair G3 subset**, IDs `0`–`99`.
Each image has five provided captions used by the text attack objective.

| Bank | Source images | Target images | Captions | Manifest |
|---|---|---|---|---|
| Full 1,000 pairs | `images/bigscale_1000/nips17/` | `images/target_images_1000/1/` | `text/bigscale_1000/nips17/caption.json`, `text/target_images_1000/1/caption.json` | `full_manifest.json` |
| Default 100 pairs | `images/bigscale/nips17/` | `images/target_images/1/` | `text/bigscale/nips17/caption.json`, `text/target_images/1/caption.json` | `manifest.json` |

The 100 subset images are byte-for-byte identical to IDs `0`–`99` in the full
bank. Their caption banks are different: all 100 source entries and all 100
target entries differ between the two versions. Both original banks are kept
to preserve the G3 selection experiment and the historical examples. There are
2,200 dataset image files representing 1,000 unique pairs, and 11,000 caption
strings across the two banks. The 100 adversarial example files are additional.

Pair a source and target by the same numeric filename stem; `i.png` matches
`i.jpg`. The original loader uses **lexical filename order**, rather than numeric
order. Thus the first 100 items of the full folder are a different subset from
the 100 files in the default folder. Each manifest records the loader order,
file paths and SHA-256 hashes. `--sample-start` indexes that lexical order.

Captions use the original JSON schema:

```json
[{"image": "0.png", "caption": ["description 1", "description 2", "description 3", "description 4", "description 5"]}]
```

Target caption keys end in `.jpg`. Captions supplied here train the attack
objective; evaluation generates fresh captions for adversarial and target
images using each victim model, then compares those outputs with GLM-5.

The project paper describes source images from the NIPS 2017 Adversarial
Learning Development Set, targets from the MS-COCO validation set, and five
LLM-generated captions for each image. The original selection seed, mappings
back to upstream image IDs, exact caption generator/version/prompts and
redistribution terms were not supplied with these renamed files. This release
preserves the available files without inventing that metadata or assigning a
new license. See [the release audit](../docs/RELEASE_AUDIT.zh-CN.md).

For the full-data command, see the [main README](../README.md). Keep source,
target and both caption paths from the same bank when running an attack.
