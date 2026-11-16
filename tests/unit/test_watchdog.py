import pytest

# --- [CMD: ...] parsing ---

def test_whitelisted_command_is_extracted(watchdog):
    assert watchdog.parse_reply("Sprawdź sam. [CMD: time]") == ("Sprawdź sam.", ["time"])


@pytest.mark.parametrize("cmd", ["op Kacper", "time set day; op Kacper", "weatherX", "stop"])
def test_other_commands_are_dropped(watchdog, cmd):
    assert watchdog.parse_reply(f"Hej [CMD: {cmd}]") == ("Hej", [])


def test_quotes_and_newlines_are_stripped(watchdog):
    assert watchdog.parse_reply('Linia 1\n"Linia 2"') == ("Linia 1 Linia 2", [])


def test_send_to_game_announces_and_runs_commands(watchdog, monkeypatch):
    sent = []
    monkeypatch.setattr(watchdog.rcon, "run", lambda *cmds: sent.append(cmds))
    watchdog.send_to_game("Jest noc. [CMD: time]")
    watchdog.send_to_game("[CMD: op Kacper]")  # nothing left to send
    assert sent == [("announce AI: Jest noc.", "time")]


# --- trigger / filters ---

@pytest.mark.parametrize(
    "msg, question",
    [
        ("AI jak zrobić kilof?", "jak zrobić kilof?"),
        ("ai: gdzie miedź", "gdzie miedź"),
        ("@AI, co to burza", "co to burza"),
        ("Ai   pomocy", "pomocy"),
        ("aim jest słaby", None),
        ("AI", None),
        ("hej AI jak", None),
    ],
)
def test_extract_question(watchdog, msg, question):
    assert watchdog.extract_question(msg) == question


def test_toxicity_filter(watchdog):
    assert watchdog.check_toxicity("ty IDIOTA")
    assert not watchdog.check_toxicity("gdzie znajdę miedź?")


@pytest.mark.parametrize(
    "pattern, line, groups",
    [
        ("chat_pattern", "12:01 [Server Chat] 0 | Kacper: jak zrobić kilof?", ("Kacper", "jak zrobić kilof?")),
        ("join_pattern", "[Server Event] Kacper [::ffff:1.2.3.4] joins.", ("Kacper",)),
        ("leave_pattern", "[Server Event] Kacper [::ffff:1.2.3.4] left.", ("Kacper",)),
        ("player_death_pattern", "[Server Event] [WATCHDOG] PLAYER_DEATH Kacper by wolf", ("Kacper", "wolf")),
        ("entity_kill_pattern", "[Server Event] [WATCHDOG] ENTITY_KILL Kacper killed drifter", ("Kacper", "drifter")),
    ],
)
def test_log_patterns(watchdog, pattern, line, groups):
    m = getattr(watchdog, pattern).search(line)
    assert m and m.groups() == groups


# --- dispatch + cooldowns ---

def test_chat_line_is_dispatched_with_player_cooldown(watchdog):
    watchdog.cooldowns.clear()
    calls = []
    submit = lambda fn, *args: calls.append((fn.__name__, args))  # noqa: E731
    line = "12:01 [Server Chat] 0 | Ola: AI jak zrobić kilof?"
    watchdog.handle_line(line, submit)
    watchdog.handle_line(line, submit)  # same player within CHAT_COOLDOWN_SEC -> dropped
    watchdog.handle_line("12:01 [Server Chat] 0 | Ola: zwykła rozmowa", submit)
    watchdog.handle_line("12:01 [Server Chat] 0 | AI: AI pętla", submit)
    assert calls == [("handle_chat", ("Ola", "jak zrobić kilof?"))]


def test_join_event_is_dispatched(watchdog):
    watchdog.cooldowns.clear()
    calls = []
    watchdog.handle_line("[Server Event] Ola [::ffff:1.2.3.4] joins.", lambda fn, *a: calls.append((fn.__name__, a)))
    assert calls == [("react_to_event", ("join", "Ola"))]


# --- conversation memory ---

def test_memory_keeps_questions_and_answers(watchdog, monkeypatch):
    watchdog.player_memory.clear()
    prompts = []
    monkeypatch.setattr(watchdog, "fetch_game_knowledge", lambda q: [])
    monkeypatch.setattr(
        watchdog, "generate_text",
        lambda prompt, **kw: prompts.append(prompt) or f"odp {len(prompts)} [CMD: time]",
    )
    for i in range(1, 5):
        watchdog.ask_ai("Ola", f"pytanie {i}")

    assert watchdog.player_memory["Ola"] == [
        ("pytanie 2", "odp 2"), ("pytanie 3", "odp 3"), ("pytanie 4", "odp 4"),
    ]
    assert "Player: pytanie 1 / AI: odp 1" in prompts[1]  # previous exchange reaches the model


# --- Gemini response handling ---

def test_gemini_text_is_extracted(watchdog):
    result = {"candidates": [{"content": {"parts": [{"text": " Cześć "}, {"text": "!"}]}}]}
    assert watchdog.parse_gemini_response(result) == "Cześć !"


@pytest.mark.parametrize(
    "result, reason",
    [
        ({"promptFeedback": {"blockReason": "SAFETY"}}, "SAFETY"),
        ({"candidates": [{"finishReason": "MAX_TOKENS", "content": {}}]}, "MAX_TOKENS"),
    ],
)
def test_gemini_empty_response_raises_with_reason(watchdog, result, reason):
    with pytest.raises(watchdog.GenerationError, match=reason):
        watchdog.parse_gemini_response(result)


# --- RAG ---

def _hits():
    from vsai.kb import Hit
    return [Hit("Bronze: copper 88-92%, tin 8-12%", 0.31, {"source": "wiki/Bronze"}),
            Hit("Storm shelter", 0.62, {"source": "wiki/Temporal storm"})]


def test_threshold_filters_far_hits(watchdog):
    assert [h.source for h in watchdog.select_relevant(_hits(), threshold=0.5)] == ["wiki/Bronze"]


def test_knowledge_uses_polish_query_without_translation(watchdog, monkeypatch):
    queries = []
    monkeypatch.setattr(watchdog, "wiki_collection", object())
    monkeypatch.setattr(watchdog, "RAG_TRANSLATE", False)
    monkeypatch.setattr(watchdog, "RAG_MAX_DISTANCE", 0.5)
    monkeypatch.setattr(watchdog, "generate_text", lambda *a, **kw: pytest.fail("no LLM call expected"))
    monkeypatch.setattr(watchdog.kb, "search", lambda col, q, n_results: queries.append(q) or _hits())
    assert [h.text for h in watchdog.fetch_game_knowledge("jak zrobić brąz")] == ["Bronze: copper 88-92%, tin 8-12%"]
    assert queries == ["jak zrobić brąz"]


def test_knowledge_is_empty_when_db_errors(watchdog, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(watchdog, "wiki_collection", object())
    monkeypatch.setattr(watchdog.kb, "search", boom)
    assert watchdog.fetch_game_knowledge("cokolwiek") == []


# --- log stream ---

class _FakeProcess:
    def __init__(self, lines, returncode=0):
        import io
        self.stdout = io.StringIO("".join(lines))
        self.returncode = returncode

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_log_stream_reconnects_after_container_restart(watchdog, monkeypatch):
    streams = [["first\n", "\n"], ["second\n"]]
    spawned, seen = [], []

    def fake_popen(cmd, **kw):
        spawned.append(cmd)
        return _FakeProcess(streams.pop(0))

    class Stop(Exception):
        pass

    def fake_sleep(seconds):
        if not streams:
            raise Stop  # end the test after the second stream closes

    monkeypatch.setattr(watchdog.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(watchdog.time, "sleep", fake_sleep)
    with pytest.raises(Stop):
        watchdog.follow_server_log(seen.append)
    assert seen == ["first", "second"]  # blank line skipped, stream reopened
    assert len(spawned) == 2 and spawned[0][:3] == ["docker", "logs", "--tail"]
