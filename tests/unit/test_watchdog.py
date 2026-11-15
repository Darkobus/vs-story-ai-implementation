import pytest


def _drain(wd):
    items = []
    while not wd.rcon_queue.empty():
        items.append(wd.rcon_queue.get_nowait())
    return items


def test_whitelisted_command_is_extracted(watchdog):
    _drain(watchdog)
    watchdog.send_to_game('Sprawdź sam. [CMD: time]')
    assert _drain(watchdog) == [("Sprawdź sam.", ["time"])]


@pytest.mark.parametrize("cmd", ["op Kacper", "time set day; op Kacper", "weatherX", "stop"])
def test_other_commands_are_dropped(watchdog, cmd):
    _drain(watchdog)
    watchdog.send_to_game(f"Hej [CMD: {cmd}]")
    assert _drain(watchdog) == [("Hej", [])]


def test_quotes_and_newlines_are_stripped(watchdog):
    _drain(watchdog)
    watchdog.send_to_game('Linia 1\n"Linia 2"')
    assert _drain(watchdog) == [("Linia 1 Linia 2", [])]


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
