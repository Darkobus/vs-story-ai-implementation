import pytest

pytest.importorskip("pyjson5")

from ingest import game_assets as ga  # noqa: E402
from ingest import wiki  # noqa: E402

# --- game assets ---


def test_alloy_ratios_are_rounded_not_truncated():
    data = {"ingredients": [{"code": "copper", "minratio": 0.88, "maxratio": 0.92},
                            {"code": "tin", "minratio": 0.08, "maxratio": 0.12}],
            "output": {"code": "tinbronze"}}
    text = ga.convert_alloy(data, "tinbronze.json")
    assert "copper 88-92%" in text  # int(0.29 * 100) == 28 was the old bug class
    assert "tin 8-12%" in text


def test_durability_by_type_is_kept():
    text = ga.convert_item({"code": "pickaxe", "durabilityByType": {"*-copper": 250, "*-iron": 600}}, "pickaxe.json")
    assert "Durability: *-copper=250, *-iron=600" in text


def test_ids_are_unique_across_subfolders():
    raw = '{ code: "generic", maxstacksize: 64 }'  # VS JSON5: unquoted keys
    ids_a, *_ = ga.file_to_chunks("item", "resource/generic.json", raw)
    ids_b, *_ = ga.file_to_chunks("item", "tool/generic.json", raw)
    assert ids_a and ids_b and set(ids_a).isdisjoint(ids_b)


def test_bad_json_raises():
    with pytest.raises(ValueError):
        ga.file_to_chunks("item", "x.json", "{ not json")


def test_read_from_local_assets_dir(tmp_path):
    base = tmp_path / "survival" / "itemtypes" / "tool"
    base.mkdir(parents=True)
    (base / "axe.json").write_text('{ code: "axe" }', encoding="utf-8")
    (base / "notes.txt").write_text("ignored", encoding="utf-8")
    assert list(ga.read_from_dir(tmp_path, "survival/itemtypes")) == [("tool/axe.json", '{ code: "axe" }')]


# --- wiki ---


def test_templates_and_links_keep_their_words():
    raw = "Made from {{ll|Charcoal}} and [[Copper ore|copper]] in a '''bloomery'''."
    assert wiki.clean_mediawiki_text(raw) == "Made from Charcoal and copper in a bloomery."


def test_infobox_becomes_key_value_lines():
    raw = ("{{PT}}<languages />\n{{Item\n|Name=<translate><!--T:18--> Black bronze ingot</translate>\n"
           "|Image=File:ingot-blackbronze.png\n|SmeltTemp=1020\n|OutputLink=Ingot mold\n}}")
    text = wiki.clean_mediawiki_text(raw)
    assert "Name: Black bronze ingot" in text
    assert "SmeltTemp: 1020" in text
    assert "PT" not in text and "File:" not in text and "OutputLink" not in text


def test_image_links_leave_no_size_noise():
    raw = "* [[File:Axehead-steel.png|25px]] <translate><!--T:22--> {{ll|Axe|Steel axe head}}</translate>"
    assert wiki.clean_mediawiki_text(raw) == "* Steel axe head"


def test_gallery_refs_and_categories_are_dropped():
    raw = ("Bronze tablet.<ref>assets/survival/recipes/grid/x.json</ref>\n"
           "<gallery>\nFile:Metalbit-tinbronze.png| Tin bronze bit\n</gallery>\n[[Category:Metals]]")
    assert wiki.clean_mediawiki_text(raw) == "Bronze tablet."


def test_table_cells_survive():
    raw = '{| class="wikitable"\n! Metal !! Ratio\n|-\n| Copper || 88-92%\n|}'
    text = wiki.clean_mediawiki_text(raw)
    assert "Copper | 88-92%" in text


def test_chunks_carry_page_title():
    chunks = wiki.create_chunks(" ".join(f"w{i}" for i in range(200)), "Helve_hammer")
    assert len(chunks) == 2
    assert all(c.startswith("Wiki page: Helve hammer\n") for c in chunks)


def test_redirects_and_translations_are_detected():
    assert wiki.is_redirect("  #redirect [[Steel]]")
    assert not wiki.is_redirect("Steel is an alloy")
    assert wiki.is_english_title("Steel")
    assert not wiki.is_english_title("Steel/ru")
    assert not wiki.is_english_title("Сталь")
