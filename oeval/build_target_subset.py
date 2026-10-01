import argparse
from pathlib import Path
from typing import Dict, List

from oattack.utils import ensure_dir

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def collect_images(root: Path) -> List[Path]:
    return [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def build_target_index(target_root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in collect_images(target_root):
        stem = path.stem
        if stem in mapping:
            raise ValueError(f"Duplicate target image ID: {stem}")
        mapping[stem] = path
    return mapping


def link_target_subset(adv_root: Path, target_root: Path, output_dir: Path, overwrite: bool) -> None:
    ensure_dir(str(output_dir))
    target_index = build_target_index(target_root)
    adv_paths = collect_images(adv_root)

    if not adv_paths:
        raise ValueError(f"No adversarial images found: {adv_root}")
    adv_ids = [path.stem for path in adv_paths]
    if len(set(adv_ids)) != len(adv_ids):
        raise ValueError("Adversarial image IDs must be unique")
    missing = sorted(set(adv_ids) - set(target_index))
    if missing:
        raise ValueError(f"Missing target images for IDs: {missing[:10]}")
    expected_names = {target_index[stem].name for stem in adv_ids}
    stale = [path for path in collect_images(output_dir) if path.name not in expected_names]
    # Include broken image symlinks, which are not returned by collect_images.
    stale.extend(path for path in output_dir.rglob("*") if path.is_symlink() and not path.exists() and path.name not in expected_names)
    if stale and not overwrite:
        raise ValueError(f"Target subset contains stale files: {stale[:10]}")
    for path in stale:
        path.unlink()
    linked = 0
    for adv_path in adv_paths:
        stem = adv_path.stem
        target_path = target_index.get(stem)
        dest = output_dir / target_path.name
        if dest.exists() or dest.is_symlink():
            if dest.is_symlink() and dest.resolve() == target_path.resolve():
                linked += 1
                continue
            if overwrite:
                dest.unlink()
            else:
                raise FileExistsError(f"Target subset contains a conflicting file: {dest}")
        dest.symlink_to(target_path.resolve())
        linked += 1

    print(f"[subset] linked {linked} / {len(adv_paths)} targets into {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a target-image subset matching adversarial image stems.")
    parser.add_argument("--adv-root", type=str, required=True, help="Folder containing adversarial images.")
    parser.add_argument(
        "--target-root",
        type=str,
        required=True,
        help="Root folder containing target images (searched recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write symlinks for the target subset.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing links if needed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adv_root = Path(args.adv_root)
    target_root = Path(args.target_root)
    output_dir = Path(args.output_dir)

    if not adv_root.exists():
        raise FileNotFoundError(f"adv-root does not exist: {adv_root}")
    if not target_root.exists():
        raise FileNotFoundError(f"target-root does not exist: {target_root}")

    link_target_subset(adv_root, target_root, output_dir, overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
