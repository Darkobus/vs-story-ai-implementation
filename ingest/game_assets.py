"""
Ingest Vintage Story game asset JSONs (items, blocks, recipes) into ChromaDB.
Reads files directly from the running Docker container via `docker exec`.
Appends to the existing 'vintage_story_wiki' collection.

Usage:
    python ingest/game_assets.py [--container NAME] [--dry-run]

Options:
    --container NAME   Docker container name (default: vs-ai-server)
    --dry-run          Print generated text chunks without writing to DB
"""

import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
import pyjson5
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

# ── Config ────────────────────────────────────────────────────────────────────

CONTAINER_NAME = "vs-ai-server"
OLLAMA_EMBED_URL = "http://192.168.0.24:11434/api/embeddings"
CHROMA_PATH = str(Path(__file__).resolve().parent.parent / "chroma_db")
COLLECTION_NAME = "vintage_story_wiki"

ASSET_DIRS = [
    ("/game/assets/survival/recipes/alloy",    "recipe_alloy"),
    ("/game/assets/survival/recipes/grid",     "recipe_grid"),
    ("/game/assets/survival/recipes/smithing", "recipe_smithing"),
    ("/game/assets/survival/recipes/cooking",  "recipe_cooking"),
    ("/game/assets/survival/recipes/barrel",   "recipe_barrel"),
    ("/game/assets/survival/itemtypes",        "item"),
    ("/game/assets/survival/blocktypes",       "block"),
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

# ── Embedding ─────────────────────────────────────────────────────────────────

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, url: str):
        self.url = url

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            data = json.dumps({"model": "nomic-embed-text", "prompt": text}).encode()
            req = urllib.request.Request(self.url, data=data,
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=15) as r:
                    embeddings.append(json.loads(r.read())["embedding"])
            except Exception as e:
                print(f"  [embed error] {e}")
                embeddings.append([0.0] * 768)
        return embeddings

# ── Docker helpers ────────────────────────────────────────────────────────────

def docker_ls(container: str, path: str) -> list[str]:
    """Return list of file paths under `path` in container (recursive)."""
    result = subprocess.run(
        ["docker", "exec", container, "find", path, "-name", "*.json", "-type", "f"],
        capture_output=True, text=True
    )
    return [p.strip() for p in result.stdout.splitlines() if p.strip()]


def docker_read(container: str, path: str) -> str:
    result = subprocess.run(
        ["docker", "exec", container, "cat", path],
        capture_output=True, text=True
    )
    return result.stdout

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
        return {
            k: strip_visual(v)
            for k, v in obj.items()
            if k not in VISUAL_KEYS
        }
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
        parts = []
        for k, v in food.items():
            if isinstance(v, (int, float)):
                parts.append(f"{k}={v}")
        return ", ".join(parts) if parts else ""
    return ""

def _durability(obj: dict) -> str:
    d = obj.get("durability") or obj.get("durabilityByType", {})
    if isinstance(d, (int, float)):
        return str(int(d))
    return ""

def convert_item(data: dict, filename: str) -> str:
    data = strip_visual(data)
    code = _code(data)
    lines = [f"Item: {code}"]

    tool = data.get("tool")
    if tool:
        lines.append(f"Tool type: {tool}")

    cls = data.get("class")
    if cls:
        lines.append(f"Class: {cls}")

    dur = _durability(data)
    if dur:
        lines.append(f"Durability: {dur}")

    maxstack = data.get("maxstacksize")
    if maxstack:
        lines.append(f"Max stack: {maxstack}")

    variants = _variants(data)
    if variants:
        lines.append(f"Variants — {variants}")

    attrs = data.get("attributes", {})
    if isinstance(attrs, dict):
        # Mining speed
        mspeed = attrs.get("miningSpeed") or attrs.get("miningSpeedMultiplier")
        if mspeed and isinstance(mspeed, dict):
            top = sorted(mspeed.items(), key=lambda x: -x[1] if isinstance(x[1], (int,float)) else 0)[:6]
            lines.append("Mining speed: " + ", ".join(f"{k}={v}" for k, v in top))
        # Attack damage
        atk = attrs.get("attackPower") or attrs.get("damage")
        if atk:
            lines.append(f"Attack damage: {atk}")
        # Nutrition
        nutr = _nutrition(attrs)
        if nutr:
            lines.append(f"Nutrition: {nutr}")

    # Perishable
    perish = data.get("perishableProps")
    if perish and isinstance(perish, dict):
        fresh = perish.get("freshHours", {})
        if isinstance(fresh, dict):
            lines.append(f"Fresh hours: {fresh.get('avg', '?')}")

    behaviors = data.get("behaviors", [])
    bnames = [b.get("name","") for b in behaviors if isinstance(b, dict) and b.get("name")]
    if bnames:
        lines.append(f"Behaviors: {', '.join(bnames)}")

    return "\n".join(lines)


def convert_block(data: dict, filename: str) -> str:
    data = strip_visual(data)
    code = _code(data)
    lines = [f"Block: {code}"]

    material = data.get("blockmaterial") or data.get("blockMaterial")
    if material:
        lines.append(f"Material: {material}")

    hardness = data.get("resistance") or data.get("resistanceByType")
    if isinstance(hardness, (int, float)):
        lines.append(f"Hardness: {hardness}")

    light = data.get("lightAbsorption") or data.get("lightHsv")
    if light:
        lines.append(f"Light: {light}")

    variants = _variants(data)
    if variants:
        lines.append(f"Variants — {variants}")

    drops = data.get("drops") or data.get("dropsByType", [])
    if isinstance(drops, list) and drops:
        drop_strs = []
        for d in drops[:5]:
            if isinstance(d, dict):
                drop_strs.append(f"{d.get('type','?')}:{d.get('code','?')}")
        if drop_strs:
            lines.append(f"Drops: {', '.join(drop_strs)}")

    behaviors = data.get("behaviors", [])
    bnames = [b.get("name","") for b in behaviors if isinstance(b, dict) and b.get("name")]
    if bnames:
        lines.append(f"Behaviors: {', '.join(bnames)}")

    return "\n".join(lines)


def convert_alloy(data: dict, filename: str) -> str:
    name = filename.replace(".json", "")
    ingredients = data.get("ingredients", [])
    output = data.get("output", {})
    parts = []
    for ing in ingredients:
        code = ing.get("code", "?")
        lo = ing.get("minratio", "?")
        hi = ing.get("maxratio", "?")
        parts.append(f"{code} {int(float(lo)*100)}-{int(float(hi)*100)}%")
    output_code = output.get("code", "?")
    return f"Alloy recipe: {name}\nIngredients: {', '.join(parts)}\nOutput: {output_code}"


def convert_grid(data: dict, filename: str) -> str:
    name = data.get("name") or filename.replace(".json", "")
    pattern = data.get("ingredientPattern", "")
    ingredients = data.get("ingredients", {})
    output = data.get("output", {})
    lines = [f"Grid recipe: {name}"]
    if pattern:
        lines.append(f"Pattern: {pattern}")
    ing_parts = []
    for key, val in (ingredients.items() if isinstance(ingredients, dict) else []):
        if isinstance(val, dict):
            ing_parts.append(f"{key}={val.get('code','?')} (qty={val.get('quantity',1)})")
    if ing_parts:
        lines.append(f"Ingredients: {', '.join(ing_parts)}")
    if isinstance(output, dict):
        qty = output.get("quantity", 1)
        lines.append(f"Output: {output.get('code','?')} x{qty}")
    return "\n".join(lines)


def convert_smithing(data: dict, filename: str) -> str:
    name = data.get("name") or filename.replace(".json", "")
    ing = data.get("ingredient", {})
    output = data.get("output", {})
    lines = [f"Smithing recipe: {name}"]
    if isinstance(ing, dict):
        allowed = ing.get("allowedVariants", [])
        code = ing.get("code", "?")
        lines.append(f"Ingredient: {code}")
        if allowed:
            lines.append(f"Allowed metals: {', '.join(allowed)}")
    if isinstance(output, dict):
        qty = output.get("stacksize") or output.get("quantity", 1)
        lines.append(f"Output: {output.get('code','?')} x{qty}")
    return "\n".join(lines)


def convert_cooking(data: dict, filename: str) -> str:
    name = data.get("code") or filename.replace(".json", "")
    lines = [f"Cooking recipe: {name}"]
    ingredients = data.get("ingredients", [])
    for ing in ingredients:
        if not isinstance(ing, dict):
            continue
        code = ing.get("code", "?")
        stacks = ing.get("validStacks", [])
        stack_codes = [s.get("code","?") for s in stacks[:3] if isinstance(s, dict)]
        qty_min = ing.get("minQuantity", "?")
        qty_max = ing.get("maxQuantity", "?")
        lines.append(f"  - {code}: {', '.join(stack_codes)} qty {qty_min}-{qty_max}")
    cooks_into = data.get("cooksInto", {})
    if isinstance(cooks_into, dict):
        lines.append(f"Cooks into: {cooks_into.get('code','?')} x{cooks_into.get('quantity','?')}")
    return "\n".join(lines)


def convert_barrel(data: dict, filename: str) -> str:
    name = data.get("name") or filename.replace(".json", "")
    lines = [f"Barrel recipe: {name}"]
    ingredients = data.get("ingredients", [])
    for ing in ingredients if isinstance(ingredients, list) else []:
        if isinstance(ing, dict):
            code = ing.get("code","?")
            qty = ing.get("quantity") or ing.get("litres", "?")
            lines.append(f"  - {code} x{qty}")
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
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + max_words]))
        i += max_words - overlap
    return chunks

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dry_run = "--dry-run" in sys.argv
    container = CONTAINER_NAME
    for i, arg in enumerate(sys.argv):
        if arg == "--container" and i + 1 < len(sys.argv):
            container = sys.argv[i + 1]

    if not dry_run:
        db = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = db.get_collection(
            name=COLLECTION_NAME,
            embedding_function=OllamaEmbeddingFunction(OLLAMA_EMBED_URL)
        )
        print(f"[*] Połączono z kolekcją '{COLLECTION_NAME}'")
    else:
        collection = None
        print("[*] Tryb dry-run — nie zapisuję do bazy")

    total_chunks = 0
    total_files = 0
    errors = 0

    for asset_dir, asset_type in ASSET_DIRS:
        converter = CONVERTERS[asset_type]
        files = docker_ls(container, asset_dir)
        print(f"\n[{asset_type}] {len(files)} plików w {asset_dir}")

        for filepath in files:
            raw = docker_read(container, filepath)
            if not raw.strip():
                continue

            data = parse_vs_json(raw)
            if data is None:
                errors += 1
                print(f"  PARSE ERROR: {filepath}")
                continue

            filename = filepath.split("/")[-1]

            # Some files are arrays of recipes
            entries = data if isinstance(data, list) else [data]

            file_chunks = []
            file_ids = []
            file_metas = []

            for idx, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    continue
                try:
                    text = converter(entry, filename)
                except Exception as e:
                    errors += 1
                    print(f"  CONVERT ERROR {filepath}[{idx}]: {e}")
                    continue

                if not text.strip():
                    continue

                chunks = chunk_text(text)
                base_id = f"{asset_type}_{filename}_{idx}"
                for c_idx, chunk in enumerate(chunks):
                    chunk_id = f"{base_id}_c{c_idx}"
                    file_chunks.append(chunk)
                    file_ids.append(chunk_id)
                    file_metas.append({
                        "source": f"game_asset/{asset_type}",
                        "file": filename,
                        "type": asset_type
                    })

            if not file_chunks:
                continue

            if dry_run:
                for chunk in file_chunks[:1]:
                    print(f"  [{filename}] {chunk[:200]}")
            else:
                # Upsert in batches of 50
                for i in range(0, len(file_chunks), 50):
                    try:
                        collection.upsert(
                            documents=file_chunks[i:i+50],
                            ids=file_ids[i:i+50],
                            metadatas=file_metas[i:i+50]
                        )
                    except Exception as e:
                        errors += 1
                        print(f"  DB ERROR {filename}: {e}")

            total_chunks += len(file_chunks)
            total_files += 1
            if total_files % 50 == 0:
                print(f"  ... {total_files} plików, {total_chunks} chunków")

    print(f"\n[DONE] {total_files} plików, {total_chunks} chunków, {errors} błędów")


if __name__ == "__main__":
    main()
