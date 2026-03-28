import re
import subprocess
import urllib.request
import urllib.error
import json
import sys
import threading
import time
import random
import chromadb
from mcrcon import MCRcon

CONTAINER_NAME = "vs-ai-server"
PC_IP = "192.168.0.24"

OLLAMA_API_URL = f"http://{PC_IP}:11434/api/generate"
OLLAMA_EMBED_URL = f"http://{PC_IP}:11434/api/embeddings"

FORBIDDEN_WORDS = ["kys", "idiota", "debil"]
ALLOWED_COMMANDS = ["time", "weather"]

player_memory = {}
import asyncio
from typing import List
import mcrcon

class RconAdapter:
    def __init__(self, host: str, port: int, password: str):
        self.host = host
        self.port = port
        self.password = password

    async def execute_command(self, command: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._send, command)

    def _send(self, command: str) -> str:
        with mcrcon.MCRcon(self.host, self.password, self.port) as mcr:
            return mcr.command(command)

class OllamaEmbeddingWrapper:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.client.embeddings(model=self.model, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
class OllamaEmbeddingFunction:
    def name(self):
        return "ollama-embed-text"

    def __call__(self, input):
        embeddings = []
        for text in input:
            req = urllib.request.Request(
                OLLAMA_EMBED_URL,
                data=json.dumps({"model": "nomic-embed-text", "prompt": text}).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            try:
                with urllib.request.urlopen(req) as response:
                    res = json.loads(response.read().decode('utf-8'))
                    embeddings.append(res["embedding"])
            except Exception as e:
                print(f"Embedding error: {e}")
        return embeddings

try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    wiki_collection = chroma_client.get_collection(
        name="vintage_story_wiki",
        embedding_function=OllamaEmbeddingFunction()
    )
    print("[*] Baza wektorowa zaladowana pomyslnie.")
except Exception as e:
    print(f"[*] OSTRZEZENIE: Brak bazy wektorowej lub blad: {e}")
    wiki_collection = None

def fetch_game_knowledge(prompt):
    if not wiki_collection:
        return ""
    try:
        results = wiki_collection.query(
            query_texts=[prompt],
            n_results=2
        )
        if results and results['documents'] and results['documents'][0]:
            return " ".join(results['documents'][0])
    except Exception as e:
        print(f"--> [VECTOR DB ERROR]: {e}")
    return ""

def check_toxicity(message):
    msg_lower = message.lower()
    for word in FORBIDDEN_WORDS:
        if word in msg_lower:
            return True
    return False

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
                response = mcr.command(f"announce AI: {clean_msg}")
                print(f"--> [RCON] Sent msg. Reply: '{response}'")
            for cmd in commands_to_run:
                response = mcr.command(cmd)
                print(f"--> [RCON] Executed cmd: {cmd}. Reply: '{response}'")
    except Exception as e:
        print(f"--> [RCON ERROR]: {e}")

def ask_ai(player, prompt):
    print("--> [AI] Generating response via network...")
    
    knowledge = fetch_game_knowledge(prompt)
    if knowledge:
        print(f"--> [RAG] Wstrzyknieto kontekst z Wiki: {knowledge[:100]}...")
    
    if player not in player_memory:
        player_memory[player] = []
        
    player_memory[player].append(f"Gracz: {prompt}")
    if len(player_memory[player]) > 5:
        player_memory[player].pop(0)
        
    history = " ".join(player_memory[player])
    
    system_context = (
        "Jesteś Bogiem - pomocnikiem w grze Vintage Story "
        "Twoje zasady: "
        "1. Odpowiadaj ZAWSZE i WYŁĄCZNIE w języku polskim. "
        "2. Bądź skrajnie zwięzły. Odpowiedź może mieć MAKSYMALNIE dwa krótkie zdania. "
        "3. NIE używaj żadnych didaskaliów, nawiasów ani gwiazdek. "
        "4. Jeśli chcesz zmienić stan świata, na samym końcu odpowiedzi dodaj [CMD: time set day], [CMD: time set night] lub [CMD: weather clear]. "
        f"Bezwzględna wiedza o mechanikach gry (wykorzystaj ją do merytorycznego wyjaśniania): {knowledge} "
        f"Historia rozmowy z {player}: {history} "
    )
    
    full_prompt = f"{system_context} Odpowiedz na to: {prompt}"
    
    data = {
        "model": "gemma3:27b",
        "prompt": full_prompt,
        "stream": False
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
                
            return ai_reply
    except Exception as e:
        print(f"--> [AI ERROR]: {e}")
        return ""

def autonomous_loop():
    while True:
        sleep_time = random.randint(300, 900)
        time.sleep(sleep_time)
        
        prompt = "Zignoruj historię. Z własnej woli wyślij krótką, złowrogą wiadomość do wszystkich graczy na serwerze, grożąc im lub komentując ich marny żywot."
        ai_response = ask_ai("SYSTEM", prompt)
        
        if ai_response:
            print(f"[AI AUTONOMY] Response: {ai_response}")
            send_to_game(ai_response)

def main():
    print(f"[*] Initializing Watchdog for {CONTAINER_NAME}...")
    
    autonomy_thread = threading.Thread(target=autonomous_loop, daemon=True)
    autonomy_thread.start()

    log_cmd = ["sudo", "docker", "logs", "--tail", "0", "-f", CONTAINER_NAME]
    
    process = subprocess.Popen(log_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    chat_pattern = re.compile(r"\[Server Chat\].*?\| (.*?): (.*)")

    print("[*] Gotowy. Oczekiwanie na wiadomości graczy...")

    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if not line:
            continue
        
        match = chat_pattern.search(line)
        if match:
            player = match.group(1).strip()
            msg = match.group(2).strip()
            
            if player == "AI" or player == "SYSTEM":
                continue
            
            if not msg.startswith("AI"):
                continue
                
            clean_msg = msg[2:].strip()
            
            if check_toxicity(clean_msg):
                print(f"\n[FILTER] Zablokowano wulgarną wiadomość od {player}.")
                continue
            
            print(f"\n[CHAT] {player}: {clean_msg}")
            
            ai_response = ask_ai(player, clean_msg)
            
            if ai_response:
                print(f"[AI] Response: {ai_response}")
                send_to_game(ai_response)

if __name__ == "__main__":
    main()
