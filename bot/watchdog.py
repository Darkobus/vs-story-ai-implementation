import re
import subprocess
import urllib.request
import urllib.error
import json
import threading
import time
import random
from pathlib import Path
import chromadb

from mcrcon import MCRcon

CONTAINER_NAME = "vs-ai-server"
PC_IP = "192.168.0.24"
OLLAMA_API_URL = f"http://{PC_IP}:11434/api/generate"
OLLAMA_EMBED_URL = f"http://{PC_IP}:11434/api/embeddings"

FORBIDDEN_WORDS = ["kys", "idiota", "debil"]
ALLOWED_COMMANDS = ["time", "weather"]

player_memory = {}

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
        "model": "gemma3:27b",
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
    commands_to_run = []
    clean_msg = message.replace('"', '').replace('\n', ' ')

    match = re.search(r'\[CMD:\s*(.*?)\]', clean_msg)
    if match:
        cmd = match.group(1)
        if any(cmd.startswith(allowed) for allowed in ALLOWED_COMMANDS):
            commands_to_run.append(cmd)
        clean_msg = re.sub(r'\[CMD:\s*.*?\]', '', clean_msg).strip()

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
        "Zasady: 1. Odpowiadaj tylko po polsku. 2. Maksymalnie dwa krótkie zdania. "
        "3. Zero gwiazdek, nawiasów i didaskaliów. 4. Na końcu możesz dodać komendy [CMD: ...]. "
        f"Wiedza z Wiki: {knowledge if knowledge else 'Brak dodatkowych danych.'} "
        f"Kontekst rozmowy: {history}"
    )

    data = {
        "model": "gemma3:27b",
        "prompt": f"{system_context}\nGracz {player}: {prompt}\nOdpowiedź:",
        "stream": False,
        "options": {"temperature": 0.6}
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

    print("[*] Bot uruchomiony. Nasłuchiwanie logów...")

    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if not line: continue

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

if __name__ == "__main__":
    main()
