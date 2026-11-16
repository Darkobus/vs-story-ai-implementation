"""
Ingest Vintage Story game asset JSONs (items, blocks, recipes) into the knowledge base.

Assets are read either from the running game container (one `docker exec tar` per
directory) or from a local copy of the game's `assets/` folder — so the KB can be
rebuilt on any machine, e.g. a PC with a GPU, and chroma_db/ copied to the server.

Usage:
    python -m ingest.game_assets [--reset] [--container NAME | --assets-dir PATH] [--dry-run]

Options:
    --reset            Drop and recreate the collection first (full rebuild)
    --container NAME   Docker container to read from (default: VS_CONTAINER or vs-ai-server)
    --assets-dir PATH  Read from a local game `assets` folder instead (the one containing survival/)
    --dry-run          Print generated text chunks without embedding or writing
"""

import argparse
import io
import subprocess
import tarfile
from collections.abc import Iterator
from pathlib import Path

import pyjson5

from vsai import config, kb

CONTAINER_ASSETS_ROOT = "/game/assets"

# (path relative to the assets root, asset type)
ASSET_DIRS = [
    ("survival/recipes/alloy",    "recipe_alloy"),
    ("survival/recipes/grid",     "recipe_grid"),
    ("survival/recipes/smithing", "recipe_smithing"),
    ("survival/recipes/cooking",  "recipe_cooking"),
    ("survival/recipes/barrel",   "recipe_barrel"),
    ("survival/itemtypes",        "item"),
    ("survival/blocktypes",       "block"),
]

# Fields that are purely visual/rendering — strip them to save tokens
VISUAL_KEYS = {
    "shape", "shapebytype", "textures", "texturesByType", "tintIndex",
    "sounds", "soundsByType", "ambientSound",
    "groundStorageTransform", "groundStorageTransformByType",
    "fpHandTransform", "tpHandTransform", "tpOffHandTransform",
    "toolrackTransformByType", "onDisplayTransform",
    "guiTransform", "guiTransformByType",
    "particleCollisionBoxes", "particleProperties",
    "selectionBoxes", "collisionBoxes",
    "randomDrawOffset", "randomSizeAdjust",
    "heldLeftReadyAnimation", "heldRightReadyAnimation",
    "heldLeftTpIdleAnimation", "heldRightTpIdleAnimation",
    "heldLeftHitAnimation", "heldRightHitAnimation",
    "heldLeftAttackAnimation", "heldRightAttackAnimation",
    "fpHandTransformByType", "tpHandTransformByType",
    "lod0shape", "lod1shape", "lod2shape",
    "sideSolid", "sideOpaque", "sideSolidByType", "sideAltitude",
    "climbable", "snowCoverage",
    "emitSurfaces", "frostable",
    "shaderincludes", "shaders",
}

# ── Reading assets ────────────────────────────────────────────────────────────


def read_from_container(container: str, asset_dir: str) -> Iterator[tuple[str, str]]:
    """Yield (relative path, text) for every .json under asset_dir in the container.
    One `tar` stream per directory instead of one `docker exec cat` per file."""
    path = f"{CONTAINER_ASSETS_ROOT}/{asset_dir}"
    proc = subprocess.run(["docker", "exec", container, "tar", "-C", path, "-cf", "-", "."],
                          capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"docker exec tar failed for {path}: {proc.stderr.decode(errors='replace').strip()}")
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
        for member in tar.getmembers():
            if not (member.isfile() and member.name.endswith(".json")):
                continue
            handle = tar.extractfile(member)
            if handle is not None:
                yield member.name.removeprefix("./"), handle.read().decode("utf-8", errors="replace")


def read_from_dir(assets_root: Path, asset_dir: str) -> Iterator[tuple[str, str]]:
    base = assets_root / asset_dir
    if not base.is_dir():
        print(f"  [warn] {base} does not exist")
        return
    for file in sorted(base.rglob("*.json")):
        yield file.relative_to(base).as_posix(), file.read_text(encoding="utf-8", errors="replace")

# ── JSON parsing (VS uses non-standard JSON5-like syntax) ─────────────────────


def parse_vs_json(raw: str) -> dict | list | None:
    try:
        return pyjson5.loads(raw)
    except Exception:
        return None

# ── Stripping visual data ─────────────────────────────────────────────────────


def strip_visual(obj):
    """Recursively remove visual-only keys from a dict/list."""
    if isinstance(obj, dict):
        return {k: strip_visual(v) for k, v in obj.items() if k not in VISUAL_KEYS}
    if isinstance(obj, list):
        return [strip_visual(i) for i in obj]
    return obj

# ── Text converters ───────────────────────────────────────────────────────────


def _code(obj: dict, key="code") -> str:
    return obj.get(key, "unknown")


def _variants(obj: dict) -> str:
    groups = obj.get("variantgroups", [])
    parts = []
    for g in groups:
        states = g.get("states", [])
        if states:
            parts.append(f"{g.get('code','?')}: {', '.join(str(s) for s in states[:20])}")
    return "; ".join(parts) if parts else ""


def _nutrition(attrs: dict) -> str:
    food = attrs.get("nutritionProps") or attrs.get("nutritionPropsbyType", {})
    if isinstance(food, dict) and food:
        parts = [f"{k}={v}" for k, v in food.items() if isinstance(v, (int, float))]
        return ", ".join(parts)
    return ""


def _by_type(value, limit=12) -> str:
    """Format a plain number or a {variant pattern: number} map (e.g. durabilityByType)."""
    if isinstance(value, (int, float)):
        return str(int(value)) if float(value).is_integer() else str(value)
    if isinstance(value, dict):
        items = [(k, v) for k, v in value.items() if isinstance(v, (int, float))][:limit]
        return ", ".join(f"{k}={_by_type(v)}" for k, v in items)
    return ""


def convert_item(data: dict, filename: str) -> str:
    data = strip_visual(data)
    lines = [f"Item: {_code(data)}"]

    if data.get("tool"):
        lines.append(f"Tool type: {data['tool']}")
    if data.get("class"):
        lines.append(f"Class: {data['class']}")

    dur = _by_type(data.get("durability") or data.get("durabilityByType"))
    if dur:
        lines.append(f"Durability: {dur}")

    if data.get("maxstacksize"):
        lines.append(f"Max stack: {data['maxstacksize']}")

    variants = _variants(data)
    if variants:
        lines.append(f"Variants — {variants}")

    attrs = data.get("attributes", {})
    if isinstance(attrs, dict):
        mspeed = attrs.get("miningSpeed") or attrs.get("miningSpeedMultiplier")
        if mspeed and isinstance(mspeed, dict):
            top = sorted(mspeed.items(), key=lambda x: -x[1] if isinstance(x[1], (int, float)) else 0)[:6]
            lines.append("Mining speed: " + ", ".join(f"{k}={v}" for k, v in top))
        atk = attrs.get("attackPower") or attrs.get("damage")
        if atk:
            lines.append(f"Attack damage: {atk}")
        nutr = _nutrition(attrs)
        if nutr:
            lines.append(f"Nutrition: {nutr}")

    perish = data.get("perishableProps")
    if perish and isinstance(perish, dict):
        fresh = perish.get("freshHours", {})
        if isinstance(fresh, dict):
            lines.append(f"Fresh hours: {fresh.get('avg', '?')}")

    behaviors = data.get("behaviors", [])
    bnames = [b.get("name", "") for b in behaviors if isinstance(b, dict) and b.get("name")]
    if bnames:
        lines.append(f"Behaviors: {', '.join(bnames)}")

    return "\n".join(lines)


def convert_block(data: dict, filename: str) -> str:
    data = strip_visual(data)
    lines = [f"Block: {_code(data)}"]

    material = data.get("blockmaterial") or data.get("blockMaterial")
    if material:
        lines.append(f"Material: {material}")

    hardness = _by_type(data.get("resistance") or data.get("resistanceByType"))
    if hardness:
        lines.append(f"Hardness: {hardness}")

    light = data.get("lightAbsorption") or data.get("lightHsv")
    if light:
        lines.append(f"Light: {light}")

    variants = _variants(data)
    if variants:
        lines.append(f"Variants — {variants}")

    drops = data.get("drops") or data.get("dropsByType", [])
    if isinstance(drops, list) and drops:
        drop_strs = [f"{d.get('type','?')}:{d.get('code','?')}" for d in drops[:5] if isinstance(d, dict)]
        if drop_strs:
            lines.append(f"Drops: {', '.join(drop_strs)}")

    behaviors = data.get("behaviors", [])
    bnames = [b.get("name", "") for b in behaviors if isinstance(b, dict) and b.get("name")]
    if bnames:
        lines.append(f"Behaviors: {', '.join(bnames)}")

    return "\n".join(lines)


def convert_alloy(data: dict, filename: str) -> str:
    name = filename.replace(".json", "")
    parts = []
    for ing in data.get("ingredients", []):
        lo, hi = ing.get("minratio", 0), ing.get("maxratio", 0)
        parts.append(f"{ing.get('code', '?')} {round(float(lo) * 100)}-{round(float(hi) * 100)}%")
    output_code = data.get("output", {}).get("code", "?")
    return f"Alloy recipe: {name}\nIngredients: {', '.join(parts)}\nOutput: {output_code}"


def convert_grid(data: dict, filename: str) -> str:
    name = data.get("name") or filename.replace(".json", "")
    pattern = data.get("ingredientPattern", "")
    ingredients = data.get("ingredients", {})
    output = data.get("output", {})
    lines = [f"Grid recipe: {name}"]
    if pattern:
        lines.append(f"Pattern: {pattern}")
    ing_parts = [
        f"{key}={val.get('code','?')} (qty={val.get('quantity',1)})"
        for key, val in (ingredients.items() if isinstance(ingredients, dict) else [])
        if isinstance(val, dict)
    ]
    if ing_parts:
        lines.append(f"Ingredients: {', '.join(ing_parts)}")
    if isinstance(output, dict):
        lines.append(f"Output: {output.get('code','?')} x{output.get('quantity', 1)}")
    return "\n".join(lines)


def convert_smithing(data: dict, filename: str) -> str:
    name = data.get("name") or filename.replace(".json", "")
    ing = data.get("ingredient", {})
    output = data.get("output", {})
    lines = [f"Smithing recipe: {name}"]
    if isinstance(ing, dict):
        lines.append(f"Ingredient: {ing.get('code', '?')}")
        allowed = ing.get("allowedVariants", [])
        if allowed:
            lines.append(f"Allowed metals: {', '.join(allowed)}")
    if isinstance(output, dict):
        qty = output.get("stacksize") or output.get("quantity", 1)
        lines.append(f"Output: {output.get('code','?')} x{qty}")
    return "\n".join(lines)


def convert_cooking(data: dict, filename: str) -> str:
    name = data.get("code") or filename.replace(".json", "")
    lines = [f"Cooking recipe: {name}"]
    for ing in data.get("ingredients", []):
        if not isinstance(ing, dict):
            continue
        stack_codes = [s.get("code", "?") for s in ing.get("validStacks", [])[:3] if isinstance(s, dict)]
        lines.append(f"  - {ing.get('code', '?')}: {', '.join(stack_codes)} "
                     f"qty {ing.get('minQuantity', '?')}-{ing.get('maxQuantity', '?')}")
    cooks_into = data.get("cooksInto", {})
    if isinstance(cooks_into, dict):
        lines.append(f"Cooks into: {cooks_into.get('code','?')} x{cooks_into.get('quantity','?')}")
    return "\n".join(lines)


def convert_barrel(data: dict, filename: str) -> str:
    name = data.get("name") or data.get("code") or filename.replace(".json", "")
    lines = [f"Barrel recipe: {name}"]
    ingredients = data.get("ingredients", [])
    for ing in ingredients if isinstance(ingredients, list) else []:
        if isinstance(ing, dict):
            qty = ing.get("quantity") or ing.get("litres", "?")
            lines.append(f"  - {ing.get('code', '?')} x{qty}")
    output = data.get("output", {})
    if isinstance(output, dict):
        lines.append(f"Output: {output.get('code','?')}")
    hours = data.get("sealHours", data.get("matureDays"))
    if hours:
        lines.append(f"Duration: {hours}")
    return "\n".join(lines)


CONVERTERS = {
    "recipe_alloy":    convert_alloy,
    "recipe_grid":     convert_grid,
    "recipe_smithing": convert_smithing,
    "recipe_cooking":  convert_cooking,
    "recipe_barrel":   convert_barrel,
    "item":            convert_item,
    "block":           convert_block,
}

# ── Chunking ──────────────────────────────────────────────────────────────────


def chunk_text(text: str, max_words=200, overlap=30) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + max_words]))
        i += max_words - overlap
    return chunks


def file_to_chunks(asset_type: str, rel_path: str, raw: str):
    """Turn one asset file into (ids, texts, metadatas). Raises ValueError on bad JSON.

    IDs use the path relative to the asset directory — file names alone are not unique
    (e.g. several subfolders contain a `generic.json`), and duplicate IDs would make
    upsert silently overwrite earlier entries."""
    data = parse_vs_json(raw)
    if data is None:
        raise ValueError("unparseable JSON")
    filename = rel_path.rsplit("/", 1)[-1]
    converter = CONVERTERS[asset_type]
    ids, texts, metas = [], [], []
    for idx, entry in enumerate(data if isinstance(data, list) else [data]):
        if not isinstance(entry, dict):
            continue
        text = converter(entry, filename)
        if not text.strip():
            continue
        for c_idx, chunk in enumerate(chunk_text(text)):
            ids.append(f"{asset_type}:{rel_path}:{idx}:{c_idx}")
            texts.append(chunk)
            metas.append({"source": f"game_asset/{asset_type}/{rel_path}", "type": asset_type})
    return ids, texts, metas

# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="drop and recreate the collection first")
    parser.add_argument("--container", default=config.CONTAINER_NAME)
    parser.add_argument("--assets-dir", type=Path, help="local game assets folder (contains survival/)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    collection = None
    if args.dry_run:
        print("[*] Dry run — nothing is embedded or written")
    else:
        collection = kb.open_collection(create=True, reset=args.reset)
        print(f"[*] Collection '{config.COLLECTION_NAME}' ({config.EMBED_MODEL}), "
              f"{collection.count()} chunks{' after reset' if args.reset else ''}")

    source = f"local {args.assets_dir}" if args.assets_dir else f"container {args.container}"
    print(f"[*] Reading assets from {source}")

    total_files = total_chunks = failed_chunks = errors = 0
    for asset_dir, asset_type in ASSET_DIRS:
        if args.assets_dir:
            files = read_from_dir(args.assets_dir, asset_dir)
        else:
            files = read_from_container(args.container, asset_dir)

        ids, texts, metas, n_files = [], [], [], 0
        for rel_path, raw in files:
            if not raw.strip():
                continue
            try:
                f_ids, f_texts, f_metas = file_to_chunks(asset_type, rel_path, raw)
            except Exception as e:
                errors += 1
                print(f"  [error] {asset_dir}/{rel_path}: {e}")
                continue
            n_files += 1
            ids += f_ids
            texts += f_texts
            metas += f_metas
            if args.dry_run and f_texts and n_files <= 3:
                print(f"  [{rel_path}] {f_texts[0][:200]}")

        print(f"[{asset_type}] {n_files} files, {len(texts)} chunks")
        if collection is not None and texts:
            # replace this asset type wholesale — removed/renamed files leave no stale chunks
            collection.delete(where={"type": asset_type})
            stored, failed = kb.add_documents(collection, ids, texts, metas)
            failed_chunks += failed
            total_chunks += stored
        else:
            total_chunks += len(texts)
        total_files += n_files

    print(f"\n[DONE] {total_files} files, {total_chunks} chunks stored, "
          f"{failed_chunks} chunks failed to embed, {errors} file errors")
    if collection is not None:
        print(f"[*] Collection now holds {collection.count()} chunks")


if __name__ == "__main__":
    main()
