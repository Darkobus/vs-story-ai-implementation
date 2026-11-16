import json
import os
import random
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from vsai import config, kb
from vsai.rcon import RconClient

CONTAINER_NAME = config.CONTAINER_NAME
PC_IP = os.environ.get("OLLAMA_PC_HOST", "192.168.0.24")  # PC running Ollama — generation when Gemini is not used
OLLAMA_API_URL = f"http://{PC_IP}:11434/api/generate"
# Embeddings (RAG) come from a local Ollama ON THE SERVER (see vsai/config.py), so the PC can stay OFF.

# RCON (Vintage Story server) — password only from .env / environment (checked in main()).
RCON_PASSWORD = config.RCON_PASSWORD

MAIN_MODEL = "gemma3:12b"  # generation model; 12b FITS in 16 GB VRAM (27b does not)
# Note: gemma3:27b (17 GB) does not fit in 16 GB VRAM (RTX 5070 Ti) -> CPU offload ->
# timeouts. gemma3:12b (8 GB) is fast and stable. Tone is controlled by the prompts below.

# --- Gemini (optional, so the PC does not have to generate) ---
# Set GEMINI_API_KEY and the bot GENERATES with Gemini (cloud) instead of local Ollama.
# Embeddings (RAG) STAY on local Ollama, because chroma_db is built with those vectors.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"                  # main answer model
GEMINI_TRANSLATE_MODEL = "gemini-2.5-flash-lite"   # lighter/faster model just for PL->EN translation
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
USE_GEMINI = bool(GEMINI_API_KEY)
# If Gemini fails (quota, outage), try Ollama on the PC. OFF by default: with the PC
# switched off every question would additionally wait for a timeout.
OLLAMA_FALLBACK = os.environ.get("OLLAMA_FALLBACK", "0") == "1"

FORBIDDEN_WORDS = ["kys", "idiota", "debil"]  # Polish chat — keep the words in Polish
ALLOWED_COMMANDS = ["time", "weather"]

# Chat trigger: "AI ...", "ai: ...", "@ai, ..." (case-insensitive)
TRIGGER_PATTERN = re.compile(r"^@?ai[\s:,]+(.+)$", re.IGNORECASE | re.DOTALL)
BOT_NAMES = {"AI", "SYSTEM"}

HISTORY_EXCHANGES = 3     # how many recent question/answer pairs to remember per player
player_memory: dict[str, list[tuple[str, str]]] = {}  # player -> [(question, answer), ...]
memory_lock = threading.Lock()

cooldowns: dict[str, float] = {}  # rate limiting (key -> time of last trigger)
cooldown_lock = threading.Lock()
EVENT_COOLDOWN_SEC = 30   # minimum gap between reactions to the same event type
# minimum gap between questions from one player (spam and Gemini cost protection)
CHAT_COOLDOWN_SEC = int(os.environ.get("CHAT_COOLDOWN_SEC", "10"))

wiki_collection = None    # set by init_knowledge_base()
rcon = RconClient(config.RCON_HOST, config.RCON_PORT, RCON_PASSWORD)

# --- RAG settings (hallucination guard) ---
RAG_N_RESULTS = 5  # how many chunks to fetch for scoring (more = better grounding)
#
# RAG_MAX_DISTANCE = cosine distance threshold (0 = identical, 1 = unrelated, 2 = opposite).
# Chunks FARTHER than this are discarded as misses; if nothing passes, the knowledge is
# empty and the model is told not to make things up. Set RAG_MAX_DISTANCE=off to disable.
#
# The default is a starting point, NOT calibrated on this KB yet. Calibrate on real data:
#   1. run `python -m tools.dbcheck "jak zrobić brąz" "gdzie znaleźć miedź" ...`
#   2. set RAG_MAX_DISTANCE slightly ABOVE the distances of good hits (and below the misses)
# Diagnosis: bot says "I don't know" to sensible questions -> threshold too low (raise it);
#            still injects junk -> threshold too high (lower it).
_threshold = os.environ.get("RAG_MAX_DISTANCE", "0.5").strip().lower()
RAG_MAX_DISTANCE = None if _threshold in ("off", "none", "") else float(_threshold)
# The multilingual embedding model matches Polish questions to English text directly.
# RAG_TRANSLATE=1 re-enables the PL->EN translation call (one extra LLM request per question).
RAG_TRANSLATE = os.environ.get("RAG_TRANSLATE", "0") == "1"


def init_knowledge_base():
    global wiki_collection
    try:
        wiki_collection = kb.open_collection()
        print(f"[*] Knowledge base loaded: '{config.COLLECTION_NAME}' "
              f"({wiki_collection.count()} chunks, {config.EMBED_MODEL})")
    except Exception as e:
        print(f"[*] WARNING: knowledge base unavailable — answering without it: {e}")
        wiki_collection = None


# --- Text generation: Gemini (cloud) or Ollama (local) ---
class GenerationError(Exception):
    pass


def _http_json(req, timeout):
    """POST JSON; on an HTTP error include the response body (the API puts the reason there)."""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        raise GenerationError(f"HTTP {e.code}: {body}") from e


def parse_gemini_response(result):
    """Extract the text from a Gemini response, or raise GenerationError with the reason
    (blocked prompt, safety filter, token limit hit before any text)."""
    candidates = result.get("candidates") or []
    if not candidates:
        reason = (result.get("promptFeedback") or {}).get("blockReason", "no candidates")
        raise GenerationError(f"Gemini returned no answer: {reason}")
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        raise GenerationError(f"Gemini returned an empty answer: {candidate.get('finishReason', '?')}")
    return text


def _gemini_generate(prompt, temperature, timeout, model):
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 400,
            "thinkingConfig": {"thinkingBudget": 0},  # no "thinking" -> faster and cheaper
        },
    }
    req = urllib.request.Request(
        f"{GEMINI_API_BASE}/{model}:generateContent",
        data=json.dumps(data).encode("utf-8"),
        # key in a header, not the URL — so it cannot leak into logs or error messages
        headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
    )
    return parse_gemini_response(_http_json(req, timeout))


def _ollama_generate(prompt, temperature, timeout):
    data = {
        "model": MAIN_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    req = urllib.request.Request(
        OLLAMA_API_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    return _http_json(req, timeout).get("response", "").strip()


def generate_text(prompt, temperature=0.1, timeout=90, model=None):
    """Generate with Gemini if GEMINI_API_KEY is set, otherwise Ollama.
    With OLLAMA_FALLBACK=1 a Gemini failure falls back to Ollama."""
    if not USE_GEMINI:
        return _ollama_generate(prompt, temperature, timeout)
    try:
        return _gemini_generate(prompt, temperature, timeout, model or GEMINI_MODEL)
    except Exception as e:
        if not OLLAMA_FALLBACK:
            raise
        print(f"--> [GEMINI ERROR] {e} — falling back to Ollama")
        return _ollama_generate(prompt, temperature, timeout)


def translate_query_for_rag(prompt):
    """Translate the player's question to English for RAG (Gemini or Ollama)."""
    try:
        translate_prompt = (
            "Translate this game-related question to English, "
            f"return ONLY the translation, no extra words: {prompt}"
        )
        return generate_text(translate_prompt, temperature=0.1, timeout=45, model=GEMINI_TRANSLATE_MODEL)
    except Exception as e:
        print(f"--> [TRANSLATE ERROR]: {e}")
        return prompt  # fall back to the original text


def fetch_game_knowledge(prompt):
    """Search the vector DB with a RELEVANCE THRESHOLD.

    Returns only chunks closer than RAG_MAX_DISTANCE. If none is relevant enough it
    returns "" and the prompt tells the model to admit it does not know, instead of
    making things up from unrelated chunks.
    """
    if not wiki_collection:
        return ""
    try:
        t0 = time.time()
        query = prompt
        if RAG_TRANSLATE:
            query = translate_query_for_rag(prompt)
            print(f"--> [RAG] Translation: {query}")
        t1 = time.time()
        hits = kb.search(wiki_collection, query, n_results=RAG_N_RESULTS)
        print(f"--> [TIME] translation: {t1-t0:.1f}s | embed+search: {time.time()-t1:.1f}s")
    except Exception as e:
        print(f"--> [VECTOR DB ERROR]: {e}")
        return ""

    if not hits:
        print("--> [RAG] No results from the DB.")
        return ""
    # Log every distance — they are what you tune RAG_MAX_DISTANCE with
    print(f"--> [RAG] Distances: {[(round(h.distance, 3), h.source) for h in hits]} "
          f"(threshold={RAG_MAX_DISTANCE})")

    relevant = select_relevant(hits)
    if not relevant:
        print(f"--> [RAG] No relevant chunks (best distance={hits[0].distance:.3f}, "
              f"threshold={RAG_MAX_DISTANCE}). The model will be told not to guess.")
        return ""
    knowledge = "\n".join(h.text for h in relevant)
    print(f"--> [RAG] Accepted {len(relevant)}/{len(hits)} chunks:\n{knowledge[:200]}\n")
    return knowledge


def select_relevant(hits, threshold=None):
    threshold = RAG_MAX_DISTANCE if threshold is None else threshold
    if threshold is None:
        return list(hits)
    return [h for h in hits if h.distance < threshold]


# --- Server log line patterns ---
chat_pattern = re.compile(r"\[Server Chat\].*?\| (.*?): (.*)")
join_pattern = re.compile(r"\[Server Event\] (.+?) \[.+\] joins\.")
leave_pattern = re.compile(r"\[Server Event\] (.+?) \[.+\] (?:left|disconnects)\.")
# "Message to all" — storms and other world events
msg_all_pattern = re.compile(r"\[Server Notification\] Message to all in group \d+: (.+)")
# EventLogger mod
player_death_pattern = re.compile(r"\[Server Event\] \[WATCHDOG\] PLAYER_DEATH (.+?) by (.+)")
entity_kill_pattern = re.compile(r"\[Server Event\] \[WATCHDOG\] ENTITY_KILL (.+?) killed (.+)")
CMD_PATTERN = re.compile(r"\[CMD:\s*(.*?)\]")


def check_toxicity(message):
    msg_lower = message.lower()
    return any(word in msg_lower for word in FORBIDDEN_WORDS)


def extract_question(message):
    """Return the question if the message addresses the bot ("AI ...", "ai: ...", "@ai ..."), else None."""
    match = TRIGGER_PATTERN.match(message.strip())
    return match.group(1).strip() if match else None


def parse_reply(message):
    """Split a model reply into chat text and whitelisted commands."""
    commands = []
    clean_msg = message.replace('"', '').replace('\n', ' ')
    match = CMD_PATTERN.search(clean_msg)
    if match:
        cmd = match.group(1).strip()
        # only an exact whitelisted command, no arguments (the LLM cannot append anything)
        if cmd in ALLOWED_COMMANDS:
            commands.append(cmd)
    clean_msg = CMD_PATTERN.sub('', clean_msg).strip()
    return clean_msg, commands


def send_to_game(message):
    """Post the reply to chat (/announce) and run allowed commands. Safe from any thread."""
    clean_msg, commands = parse_reply(message)
    to_run = ([f"announce AI: {clean_msg}"] if clean_msg else []) + commands
    if not to_run:
        return
    try:
        rcon.run(*to_run)
    except Exception as e:
        print(f"--> [RCON ERROR]: {e}")


def format_history(player):
    with memory_lock:
        exchanges = list(player_memory.get(player, []))
    return " | ".join(f"Player: {q} / AI: {a}" for q, a in exchanges)


def remember(player, question, answer):
    with memory_lock:
        history = player_memory.setdefault(player, [])
        history.append((question, answer))
        del history[:-HISTORY_EXCHANGES]


# Prompts are written in English; the bot always ANSWERS players in Polish.
NO_KNOWLEDGE = ("No matching knowledge in the database - tell the player plainly that you "
                "do not have reliable information on this, instead of guessing.")


def ask_ai(player, prompt, use_memory=True):
    knowledge = fetch_game_knowledge(prompt)
    history = format_history(player) if use_memory else ""

    system_context = (
        "You are a helpful assistant on a Vintage Story game server. "
        "Rules: "
        "1. Always reply in Polish, naturally and in a friendly way. "
        "2. Give the concrete answer first, concisely, at most 4 sentences. "
        "3. Base your answer PRIMARILY on the 'Wiki knowledge' below - it is hard data from this game; "
        "if it contains numbers or recipes, give EXACTLY those values, not your own. "
        "4. Vintage Story is NOT Minecraft - it has its own, different mechanics. Do not carry over "
        "recipes, ratios, names or rules from other games. "
        "5. If the knowledge below does not contain the answer, say plainly that you do not have "
        "reliable information - do NOT guess numbers, recipes or names from memory. "
        "6. No asterisks, stage directions or pathos. "
        "7. At the end you may add the command [CMD: time] or [CMD: weather] when it fits. "
        f"Wiki knowledge: {knowledge or NO_KNOWLEDGE} "
        f"Conversation context: {history}"
    )

    full_prompt = f"{system_context}\nPlayer {player}: {prompt}\nAnswer:"

    try:
        t_gen = time.time()
        ai_reply = generate_text(full_prompt, temperature=0.1, timeout=90)
        print(f"--> [TIME] answer generation: {time.time()-t_gen:.1f}s")

        clean_reply = CMD_PATTERN.sub('', ai_reply).strip()
        if use_memory and clean_reply:
            remember(player, prompt, clean_reply)
        return ai_reply
    except Exception as e:
        print(f"--> [AI ERROR]: {e}")
        return ""


def is_on_cooldown(key, seconds=EVENT_COOLDOWN_SEC):
    """Return True if this key fired too recently; otherwise record it and return False."""
    now = time.time()
    with cooldown_lock:
        if now - cooldowns.get(key, 0) < seconds:
            return True
        cooldowns[key] = now
        return False


def handle_chat(player, question):
    ai_response = ask_ai(player, question)
    if ai_response:
        send_to_game(ai_response)


def react_to_event(event_type, context):
    """Generate and send a friendly assistant reaction to a world event."""
    prompts = {
        "join": f"Player {context} just joined the server. Greet them briefly, warmly and in a friendly way. Max 1-2 sentences.",
        "leave": f"Player {context} just left the server. Say goodbye briefly and nicely. Max 1 sentence.",
        "death": f"Event: {context}. Comment on it with sympathy and add one short, practical tip for the future. Max 2 sentences.",
        "storm_start": "A temporal storm is coming. Warn the players briefly and to the point, remind them to take shelter. Max 2 sentences.",
        "storm_end": "The temporal storm has just passed. Say briefly that it is safe again. Max 1 sentence.",
        "entity_kill": f"Event: {context}. Comment briefly and positively, e.g. praise the player. Max 1 sentence.",
        "world_event": f"World event: {context}. Comment on it briefly and to the point. Max 1-2 sentences.",
    }

    prompt = prompts.get(event_type, f"Event: {context}. Comment briefly and in a friendly way. Max 1 sentence.")
    print(f"[EVENT] Reacting to: {event_type} | {context}")

    full_prompt = (
        "You are a helpful, friendly assistant on a Vintage Story server. "
        "Reply ONLY in Polish, briefly and to the point. "
        "No asterisks, stage directions, brackets or pathos.\n"
        f"{prompt}\nAnswer:"
    )
    try:
        ai_reply = generate_text(full_prompt, temperature=0.7, timeout=90)
        if ai_reply:
            send_to_game(ai_reply)
    except Exception as e:
        print(f"--> [EVENT AI ERROR]: {e}")


def autonomous_loop():
    while True:
        time.sleep(random.randint(300, 900))
        prompt = ("Write one short, friendly tip for Vintage Story players "
                  "(a fact about crafting, survival or a game mechanic). Max 1-2 sentences.")
        ai_response = ask_ai("SYSTEM", prompt, use_memory=False)
        if ai_response:
            send_to_game(ai_response)


def handle_line(line, submit):
    """Classify a log line and hand the work to `submit(fn, *args)` (the thread pool)."""
    # --- Player chat ---
    match = chat_pattern.search(line)
    if match:
        player, msg = match.group(1).strip(), match.group(2).strip()
        if player in BOT_NAMES:
            return
        question = extract_question(msg)
        if not question or check_toxicity(question):
            return
        if is_on_cooldown(f"chat_{player}", CHAT_COOLDOWN_SEC):
            print(f"[CHAT] {player}: skipped (cooldown {CHAT_COOLDOWN_SEC}s)")
            return
        print(f"[CHAT] {player}: {question}")
        submit(handle_chat, player, question)
        return

    # --- Player joins ---
    match = join_pattern.search(line)
    if match:
        player = match.group(1).strip()
        if not is_on_cooldown(f"join_{player}"):
            submit(react_to_event, "join", player)
        return

    # --- Player leaves ---
    match = leave_pattern.search(line)
    if match:
        player = match.group(1).strip()
        if not is_on_cooldown(f"leave_{player}"):
            submit(react_to_event, "leave", player)
        return

    # --- Player death (EventLogger mod) ---
    match = player_death_pattern.search(line)
    if match:
        player, cause = match.group(1).strip(), match.group(2).strip()
        print(f"[EVENT] Player death: {player} by {cause}")
        if not is_on_cooldown(f"player_death_{player}"):
            submit(react_to_event, "death", f"{player} was killed by: {cause}")
        return

    # --- Player killed a creature (EventLogger mod) ---
    match = entity_kill_pattern.search(line)
    if match:
        player, entity = match.group(1).strip(), match.group(2).strip()
        print(f"[EVENT] Kill: {player} -> {entity}")
        if not is_on_cooldown(f"kill_{player}"):
            submit(react_to_event, "entity_kill", f"{player} killed: {entity}")
        return

    # --- World events ("Message to all") ---
    match = msg_all_pattern.search(line)
    if match:
        world_msg = match.group(1).strip()
        if world_msg.startswith("AI:"):  # the bot's own announcements
            return
        msg_lower = world_msg.lower()
        if "temporal storm" in msg_lower and "approach" in msg_lower:
            if not is_on_cooldown("storm_start"):
                submit(react_to_event, "storm_start", world_msg)
        elif "temporal storm" in msg_lower and ("wan" in msg_lower or "pass" in msg_lower or "over" in msg_lower):
            if not is_on_cooldown("storm_end"):
                submit(react_to_event, "storm_end", world_msg)
        # add other world events here


def main():
    if not RCON_PASSWORD:
        raise SystemExit("VS_RCON_PASSWORD is not set — add it to .env (see .env.example)")
    print(f"[*] Generation: {('Gemini ' + GEMINI_MODEL) if USE_GEMINI else ('Ollama ' + MAIN_MODEL)}"
          f"{' (fallback: Ollama)' if USE_GEMINI and OLLAMA_FALLBACK else ''}"
          f" | Embeddings: {config.EMBED_MODEL} @ {config.EMBED_HOST}"
          f" | RAG threshold: {RAG_MAX_DISTANCE}{', translate' if RAG_TRANSLATE else ''}")
    init_knowledge_base()

    threading.Thread(target=autonomous_loop, daemon=True).start()
    # Questions and reactions run in a thread pool — the log loop never waits for the LLM.
    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ai")
    print("[*] Bot started. Listening to logs...")
    follow_server_log(lambda line: handle_line(line, pool.submit))


def follow_server_log(on_line, retry_delay=5.0):
    """Feed every non-empty server log line to `on_line`, forever.

    `docker logs -f` ends whenever the game container restarts or stops. Exiting then
    would be a clean exit (code 0), which systemd's Restart=on-failure does NOT restart —
    the bot would silently stay down. So reconnect instead. Lines logged while
    disconnected are skipped (--tail 0), so old chat is never answered twice."""
    # No sudo — the user must be in the 'docker' group (the systemd service needs that too).
    log_cmd = ["docker", "logs", "--tail", "0", "-f", CONTAINER_NAME]
    while True:
        with subprocess.Popen(log_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            assert process.stdout is not None  # guaranteed by stdout=PIPE
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:
                    on_line(line)
        print(f"[*] Log stream ended (docker exit code {process.returncode}) — reconnecting in {retry_delay:.0f}s")
        time.sleep(retry_delay)


if __name__ == "__main__":
    main()
