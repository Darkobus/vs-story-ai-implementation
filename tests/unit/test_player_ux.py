"""Citations, the help command, event switches, the join hint and tips."""
import pytest

from vsai.kb import Hit

BRONZE = Hit("Bronze: copper 88-92%, tin 8-12%", 0.31, {"source": "wiki/Bronze"})
TINBRONZE_FILE = "game_asset/recipe_alloy/tinbronze.json"


def _dispatch(watchdog, line):
    watchdog.cooldowns.clear()
    calls = []
    watchdog.handle_line(line, lambda fn, *a: calls.append((fn.__name__, a)))
    return calls


# --- citations ---

def test_source_labels(watchdog):
    assert watchdog.source_label("wiki/Temporal storm") == "wiki „Temporal storm”"
    assert watchdog.source_label(TINBRONZE_FILE) == "dane gry „tinbronze”"
    assert watchdog.source_label("game_asset/item/tool/pickaxe.json") == "dane gry „pickaxe”"
    assert watchdog.source_label("something/else") == ""


def test_format_sources_dedupes_and_limits(watchdog):
    hits = [BRONZE, Hit("b", 0.2, {"source": "wiki/Bronze"}),
            Hit("c", 0.3, {"source": TINBRONZE_FILE}), Hit("d", 0.4, {"source": "wiki/Copper"})]
    assert watchdog.format_sources(hits) == "(źródła: wiki „Bronze”, dane gry „tinbronze”)"
    assert watchdog.format_sources([BRONZE]) == "(źródło: wiki „Bronze”)"
    assert watchdog.format_sources([]) == ""


def test_answer_cites_sources_but_memory_does_not(watchdog, monkeypatch):
    watchdog.player_memory.clear()
    monkeypatch.setattr(watchdog, "CITE_SOURCES", True)
    monkeypatch.setattr(watchdog, "fetch_game_knowledge", lambda q: [BRONZE])
    monkeypatch.setattr(watchdog, "generate_text", lambda prompt, **kw: "Miedź 88-92%, cyna 8-12%.")
    assert watchdog.ask_ai("Ola", "jak zrobić brąz") == "Miedź 88-92%, cyna 8-12%. (źródło: wiki „Bronze”)"
    assert watchdog.player_memory["Ola"] == [("jak zrobić brąz", "Miedź 88-92%, cyna 8-12%.")]


def test_no_citation_without_knowledge_or_when_disabled(watchdog, monkeypatch):
    monkeypatch.setattr(watchdog, "CITE_SOURCES", True)
    monkeypatch.setattr(watchdog, "generate_text", lambda prompt, **kw: "Nie mam pewnych informacji.")
    monkeypatch.setattr(watchdog, "fetch_game_knowledge", lambda q: [])
    assert watchdog.ask_ai("Ola", "kto wygra mecz") == "Nie mam pewnych informacji."
    monkeypatch.setattr(watchdog, "fetch_game_knowledge", lambda q: [BRONZE])
    monkeypatch.setattr(watchdog, "CITE_SOURCES", False)
    assert watchdog.ask_ai("Ola", "brąz") == "Nie mam pewnych informacji."


def test_citation_goes_through_send_to_game_intact(watchdog, monkeypatch):
    sent = []
    monkeypatch.setattr(watchdog.rcon, "run", lambda *cmds: sent.append(cmds))
    watchdog.send_to_game("Jest dzień. [CMD: time] (źródło: wiki „Day”)")
    assert sent == [("announce AI: Jest dzień. (źródło: wiki „Day”)", "time")]


# --- help command ---

@pytest.mark.parametrize("msg", ["AI help", "ai pomoc", "AI pomocy!", "@ai ?", "AI POMOC?"])
def test_help_is_answered_without_llm(watchdog, msg):
    calls = _dispatch(watchdog, f"12:01 [Server Chat] 0 | Ola: {msg}")
    assert calls == [("send_to_game", (watchdog.HELP_TEXT,))]


def test_question_mentioning_help_is_a_question(watchdog):
    assert not watchdog.is_help_request("pomoc przy budowie pieca")
    calls = _dispatch(watchdog, "12:01 [Server Chat] 0 | Ola: AI pomoc przy budowie pieca")
    assert calls == [("handle_chat", ("Ola", "pomoc przy budowie pieca"))]


# --- event switches + join hint ---

STORM = "[Server Notification] Message to all in group 0: A temporal storm is approaching"


def test_disabled_events_are_ignored(watchdog, monkeypatch):
    monkeypatch.setattr(watchdog, "EVENT_REACTIONS", {"join"})
    assert _dispatch(watchdog, "[Server Event] [WATCHDOG] PLAYER_DEATH Ola by wolf") == []
    assert _dispatch(watchdog, "[Server Event] [WATCHDOG] ENTITY_KILL Ola killed drifter") == []
    assert _dispatch(watchdog, "[Server Event] Ola [::ffff:1.2.3.4] left.") == []
    assert _dispatch(watchdog, STORM) == []
    assert _dispatch(watchdog, "[Server Event] Ola [::ffff:1.2.3.4] joins.") == [("react_to_event", ("join", "Ola"))]


def test_enabled_death_and_storm_are_dispatched(watchdog, monkeypatch):
    monkeypatch.setattr(watchdog, "EVENT_REACTIONS", {"death", "storm"})
    assert _dispatch(watchdog, "[Server Event] [WATCHDOG] PLAYER_DEATH Ola by wolf") == [
        ("react_to_event", ("death", "Ola was killed by: wolf"))]
    assert _dispatch(watchdog, STORM) == [("react_to_event", ("storm_start", "A temporal storm is approaching"))]


def test_default_reactions_skip_deaths_and_kills(watchdog):
    assert watchdog.EVENT_REACTIONS == {"join", "leave", "storm"}


def test_join_greeting_gets_hint_even_if_llm_fails(watchdog, monkeypatch):
    sent = []
    monkeypatch.setattr(watchdog, "send_to_game", sent.append)
    monkeypatch.setattr(watchdog, "JOIN_HINT_ENABLED", True)
    monkeypatch.setattr(watchdog, "generate_text", lambda prompt, **kw: "Cześć Ola!")
    watchdog.react_to_event("join", "Ola")

    def down(prompt, **kw):
        raise watchdog.GenerationError("quota")

    monkeypatch.setattr(watchdog, "generate_text", down)
    watchdog.react_to_event("join", "Ola")
    watchdog.react_to_event("leave", "Ola")  # no hint, nothing generated -> nothing sent
    assert sent == [f"Cześć Ola! {watchdog.JOIN_HINT}", watchdog.JOIN_HINT]


# --- tips ---

def test_tip_is_built_from_a_random_chunk_and_cited(watchdog, monkeypatch):
    sent, prompts = [], []
    monkeypatch.setattr(watchdog, "wiki_collection", object())
    monkeypatch.setattr(watchdog, "CITE_SOURCES", True)
    monkeypatch.setattr(watchdog, "send_to_game", sent.append)
    monkeypatch.setattr(watchdog.kb, "random_chunk",
                        lambda col: Hit("Alloy recipe: tinbronze", 0.0, {"source": TINBRONZE_FILE}))
    monkeypatch.setattr(watchdog, "generate_text",
                        lambda prompt, **kw: prompts.append(prompt) or "Brąz to miedź i cyna.")
    watchdog.post_tip()
    assert "Data: Alloy recipe: tinbronze" in prompts[0]
    assert sent == ["Brąz to miedź i cyna. (źródło: dane gry „tinbronze”)"]


@pytest.mark.parametrize("reply", ["SKIP", "skip.", "", "[CMD: time]"])
def test_unusable_tip_is_not_sent(watchdog, monkeypatch, reply):
    sent = []
    monkeypatch.setattr(watchdog, "wiki_collection", object())
    monkeypatch.setattr(watchdog, "send_to_game", sent.append)
    monkeypatch.setattr(watchdog.kb, "random_chunk", lambda col: Hit("Block: log", 0.0, {"source": "x"}))
    monkeypatch.setattr(watchdog, "generate_text", lambda prompt, **kw: reply)
    watchdog.post_tip()
    assert sent == []


def test_no_tip_without_knowledge_base(watchdog, monkeypatch):
    monkeypatch.setattr(watchdog, "wiki_collection", None)
    monkeypatch.setattr(watchdog, "generate_text", lambda *a, **kw: pytest.fail("no LLM call expected"))
    watchdog.post_tip()


def test_fixed_chat_texts_have_no_markup(watchdog):
    # Vintage Story renders chat as VTML; "<...>" would be parsed as a tag
    for text in (watchdog.HELP_TEXT, watchdog.JOIN_HINT, watchdog.format_sources([BRONZE])):
        assert "<" not in text and ">" not in text
