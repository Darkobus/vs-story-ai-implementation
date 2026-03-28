import re
import subprocess
import urllib.request
import urllib.error
import json
import threading
import time
import random
import queue
from pathlib import Path
import chromadb

from mcrcon import MCRcon

CONTAINER_NAME = "vs-ai-server"
PC_IP = "192.168.0.24"
OLLAMA_API_URL = f"http://{PC_IP}:11434/api/generate"
OLLAMA_EMBED_URL = f"http://{PC_IP}:11434/api/embeddings"
MAIN_MODEL = "vs-god"          # po fine-tuningu; fallback: "gemma3:27b"
TRANSLATE_MODEL = "gemma3:27b" # tłumaczenie PL->EN dla RAG (zostawiamy duży model)

FORBIDDEN_WORDS = ["kys", "idiota", "debil"]
ALLOWED_COMMANDS = ["time", "weather"]

player_memory = {}
event_cooldowns = {}   # rate-limiting zdarzeń (event_key -> last_trigger_time)
EVENT_COOLDOWN_SEC = 30  # minimum przerwa między reakcjami na ten sam typ zdarzenia
rcon_queue = queue.Queue()  # kolejka do obsługi RCON z głównego wątku

CHROMA_PATH = str(Path(__file__).resolve().parent.parent / "chroma_db")

from chromadb import EmbeddingFunction, Embeddings

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name

    def __call__(self, input: list[str]) -> Embeddings:
        embeddings = []
        for text in input:
            data = json.dumps({"model": self.model_name, "prompt": text}).encode('utf-8')
            req = urllib.request.Request(
                OLLAMA_EMBED_URL,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    res = json.loads(response.read().decode('utf-8'))
                    embeddings.append(res["embedding"])
            except Exception as e:
                print(f"Embedding error: {e}")
                embeddings.append([0.0] * 768)
        return embeddings

try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    wiki_collection = chroma_client.get_collection(
        name="vintage_story_wiki",
        embedding_function=OllamaEmbeddingFunction()
    )
    print("[*] Baza wektorowa zaladowana pomyslnie.")
except Exception as e:
    print(f"[*] OSTRZEZENIE: Brak bazy wektorowej lub blad: {e}")
    wiki_collection = None

def translate_query_for_rag(prompt):
    """Wysyła zapytanie do AI żeby przetłumaczyć na angielski dla RAG"""
    data = {
        "model": TRANSLATE_MODEL,
        "prompt": f"Translate this game-related question to English, return ONLY the translation, no extra words: {prompt}",
        "stream": False,
        "options": {"temperature": 0.1}
    }
    try:
        req = urllib.request.Request(
            OLLAMA_API_URL,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("response", "").strip()
    except Exception as e:
        print(f"--> [TRANSLATE ERROR]: {e}")
        return prompt  # fallback - użyj oryginału

def fetch_game_knowledge(prompt):
    if not wiki_collection:
        return ""
    try:
        translated = translate_query_for_rag(prompt)
        print(f"--> [RAG] Tłumaczenie: {translated}")
        results = wiki_collection.query(
            query_texts=[translated],
            n_results=2
        )
        if results and results['documents'] and results['documents'][0]:
            knowledge = "\n".join(results['documents'][0])
            print(f"--> [RAG] Znaleziono wiedzę:\n{knowledge[:200]}\n")
            return knowledge
    except Exception as e:
        print(f"--> [VECTOR DB ERROR]: {e}")
    print("--> [RAG] Brak wyników.")
    return ""

def check_toxicity(message):
    msg_lower = message.lower()
    return any(word in msg_lower for word in FORBIDDEN_WORDS)

def send_to_game(message):
    """Parsuje wiadomość i wrzuca do kolejki RCON (bezpieczne z wątków tła)."""
    commands_to_run = []
    clean_msg = message.replace('"', '').replace('\n', ' ')

    match = re.search(r'\[CMD:\s*(.*?)\]', clean_msg)
    if match:
        cmd = match.group(1)
        if any(cmd.startswith(allowed) for allowed in ALLOWED_COMMANDS):
            commands_to_run.append(cmd)
        clean_msg = re.sub(r'\[CMD:\s*.*?\]', '', clean_msg).strip()

    rcon_queue.put((clean_msg, commands_to_run))

def flush_rcon_queue():
    """Wywołuj tylko z głównego wątku — przetwarza kolejkę RCON."""
    while not rcon_queue.empty():
        try:
            clean_msg, commands_to_run = rcon_queue.get_nowait()
        except queue.Empty:
            break
        try:
            with MCRcon("127.0.0.1", "klawisz", port=42421, timeout=5) as mcr:
                if clean_msg:
                    mcr.command(f"announce AI: {clean_msg}")
                for cmd in commands_to_run:
                    mcr.command(cmd)
        except Exception as e:
            print(f"--> [RCON ERROR]: {e}")

def ask_ai(player, prompt):
    knowledge = fetch_game_knowledge(prompt)

    if player not in player_memory:
        player_memory[player] = []

    history = " | ".join(player_memory[player][-3:])

    system_context = (
        "Jesteś Bogiem - pomocnikiem w grze Vintage Story. "
        "Zasady: 1. Odpowiadaj tylko po polsku. 2. Wytłumacz wątek najdokładniej jak się da w max 4 zdaniach"
        "3. Zero gwiazdek, nawiasów i didaskaliów. 4. Na końcu możesz dodać komendy [CMD: ...]. "
        f"Wiedza z Wiki: {knowledge if knowledge else 'Brak dodatkowych danych.'} "
        f"Kontekst rozmowy: {history}"
    )

    data = {
        "model": MAIN_MODEL,
        "prompt": f"{system_context}\nGracz {player}: {prompt}\nOdpowiedź:",
        "stream": False,
        "options": {"temperature": 0.1}
    }

    try:
        req = urllib.request.Request(
            OLLAMA_API_URL,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            ai_reply = result.get("response", "").strip()

            clean_reply = re.sub(r'\[CMD:\s*.*?\]', '', ai_reply).strip()
            if clean_reply:
                player_memory[player].append(f"AI: {clean_reply}")
                if len(player_memory[player]) > 5:
                    player_memory[player].pop(0)

            return ai_reply
    except Exception as e:
        print(f"--> [AI ERROR]: {e}")
        return ""

def is_on_cooldown(event_key):
    """Zwraca True jeśli zdarzenie jest na cooldownie (za wcześnie by reagować)."""
    last = event_cooldowns.get(event_key, 0)
    if time.time() - last < EVENT_COOLDOWN_SEC:
        return True
    event_cooldowns[event_key] = time.time()
    return False

def react_to_event(event_type, context):
    """Generuje i wysyła mroczną reakcję AI na zdarzenie w świecie."""
    prompts = {
        "join": f"Gracz {context} właśnie dołączył do serwera. Przywitaj go krótko, mrocznym, enigmatycznym tonem jako Bóg Vintage Story. Max 1-2 zdania.",
        "leave": f"Gracz {context} właśnie opuścił serwer. Skomentuj to krótko, mrocznym tonem jako Bóg. Max 1 zdanie.",
        "death": f"Zdarzenie w świecie: {context}. Skomentuj tę śmierć lub tragedię krótko, mrocznym, sarkastycznym tonem jako Bóg Vintage Story. Max 1-2 zdania.",
        "storm_start": "Burza temporalna nadchodzi na serwer. Ogłoś to mroczno i dramatycznie jako Bóg, nawiązując do chaosu temporalnego. Max 2 zdania.",
        "storm_end": "Burza temporalna właśnie ustąpiła. Skomentuj chwilowy spokój mrocznym tonem jako Bóg. Max 1 zdanie.",
        "entity_kill": f"Zdarzenie: {context}. Skomentuj krótko zabicie jako mroczny Bóg — może pogratulować sarkazm em lub ironicznie. Max 1 zdanie.",
        "world_event": f"Zdarzenie w świecie gry: {context}. Skomentuj to krótko jako mroczny Bóg Vintage Story. Max 1-2 zdania.",
    }

    prompt = prompts.get(event_type, f"Zdarzenie: {context}. Skomentuj mroczno. Max 1 zdanie.")
    print(f"[EVENT] Reaguję na: {event_type} | {context}")

    data = {
        "model": MAIN_MODEL,
        "prompt": (
            "Jesteś mrocznym Bogiem serwera Vintage Story. Odpowiadaj TYLKO po polsku. "
            "Zero gwiazdek, didaskaliów i nawiasów. Bądź enigmatyczny i mroczny.\n"
            f"{prompt}\nOdpowiedź:"
        ),
        "stream": False,
        "options": {"temperature": 0.7}
    }
    try:
        req = urllib.request.Request(
            OLLAMA_API_URL,
            data=json.dumps(data).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=20) as response:
            result = json.loads(response.read().decode('utf-8'))
            ai_reply = result.get("response", "").strip()
            if ai_reply:
                send_to_game(ai_reply)
    except Exception as e:
        print(f"--> [EVENT AI ERROR]: {e}")

def autonomous_loop():
    while True:
        time.sleep(random.randint(300, 900))
        prompt = "Wyślij krótką, mroczną wiadomość do śmiertelników na serwerze."
        ai_response = ask_ai("SYSTEM", prompt)
        if ai_response:
            send_to_game(ai_response)

def main():
    autonomy_thread = threading.Thread(target=autonomous_loop, daemon=True)
    autonomy_thread.start()

    log_cmd = ["sudo", "docker", "logs", "--tail", "0", "-f", CONTAINER_NAME]
    process = subprocess.Popen(log_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    chat_pattern = re.compile(r"\[Server Chat\].*?\| (.*?): (.*)")
    join_pattern = re.compile(r"\[Server Event\] (.+?) \[.+\] joins\.")
    leave_pattern = re.compile(r"\[Server Event\] (.+?) \[.+\] (?:left|disconnects)\.")
    # "Message to all" — burze, inne world eventy
    msg_all_pattern = re.compile(r"\[Server Notification\] Message to all in group \d+: (.+)")
    # EventLogger mod
    player_death_pattern = re.compile(r"\[Server Event\] \[WATCHDOG\] PLAYER_DEATH (.+?) by (.+)")
    entity_kill_pattern = re.compile(r"\[Server Event\] \[WATCHDOG\] ENTITY_KILL (.+?) killed (.+)")

    print("[*] Bot uruchomiony. Nasłuchiwanie logów...")

    for line in iter(process.stdout.readline, ''):
        flush_rcon_queue()
        line = line.strip()
        if not line: continue

        # --- Chat gracza ---
        match = chat_pattern.search(line)
        if match:
            player, msg = match.group(1).strip(), match.group(2).strip()
            if player in ["AI", "SYSTEM"] or not msg.startswith("AI "):
                continue

            clean_msg = msg[3:].strip()
            if check_toxicity(clean_msg):
                continue

            print(f"[CHAT] {player}: {clean_msg}")
            ai_response = ask_ai(player, clean_msg)
            if ai_response:
                send_to_game(ai_response)
            continue

        # --- Gracz dołącza ---
        match = join_pattern.search(line)
        if match:
            player = match.group(1).strip()
            if not is_on_cooldown(f"join_{player}"):
                threading.Thread(target=react_to_event, args=("join", player), daemon=True).start()
            continue

        # --- Gracz wychodzi ---
        match = leave_pattern.search(line)
        if match:
            player = match.group(1).strip()
            if not is_on_cooldown(f"leave_{player}"):
                threading.Thread(target=react_to_event, args=("leave", player), daemon=True).start()
            continue

        # --- Śmierć gracza (EventLogger mod) ---
        match = player_death_pattern.search(line)
        if match:
            player, cause = match.group(1).strip(), match.group(2).strip()
            print(f"[EVENT] Śmierć gracza: {player} przez {cause}")
            if not is_on_cooldown(f"player_death_{player}"):
                ctx = f"{player} zginął od: {cause}"
                threading.Thread(target=react_to_event, args=("death", ctx), daemon=True).start()
            continue

        # --- Gracz zabił stwora (EventLogger mod) ---
        match = entity_kill_pattern.search(line)
        if match:
            player, entity = match.group(1).strip(), match.group(2).strip()
            print(f"[EVENT] Kill: {player} -> {entity}")
            if not is_on_cooldown(f"kill_{player}"):
                ctx = f"{player} zabił: {entity}"
                threading.Thread(target=react_to_event, args=("entity_kill", ctx), daemon=True).start()
            continue

        # --- World events ("Message to all") ---
        match = msg_all_pattern.search(line)
        if match:
            world_msg = match.group(1).strip()
            # Ignoruj własne wiadomości AI
            if world_msg.startswith("AI:"):
                continue

            msg_lower = world_msg.lower()

            if "temporal storm" in msg_lower and "approach" in msg_lower:
                if not is_on_cooldown("storm_start"):
                    threading.Thread(target=react_to_event, args=("storm_start", world_msg), daemon=True).start()
            elif "temporal storm" in msg_lower and ("wan" in msg_lower or "pass" in msg_lower or "over" in msg_lower):
                if not is_on_cooldown("storm_end"):
                    threading.Thread(target=react_to_event, args=("storm_end", world_msg), daemon=True).start()
            # inne world eventy możesz dodać tutaj

if __name__ == "__main__":
    main()
