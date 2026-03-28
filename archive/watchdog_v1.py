import re
import subprocess
import urllib.request
import urllib.error
import json
import sys
import threading
import time
import random
from mcrcon import MCRcon

CONTAINER_NAME = "vs-ai-server"
OLLAMA_API_URL = "http://192.168.0.24:11434/api/generate"
FORBIDDEN_WORDS = ["kys", "idiota", "debil"]
ALLOWED_COMMANDS = ["time", "weather"]

player_memory = {}

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
    
    if player not in player_memory:
        player_memory[player] = []
        
    player_memory[player].append(f"Gracz: {prompt}")
    if len(player_memory[player]) > 5:
        player_memory[player].pop(0)
        
    history = " ".join(player_memory[player])
    
    system_context = (
        "Jesteś bezlitosnym, aroganckim Bogiem AI na serwerze gry Vintage Story. "
        "Twoje zasady: "
        "1. Odpowiadaj ZAWSZE i WYŁĄCZNIE w języku polskim. "
        "2. Bądź skrajnie zwięzły. Odpowiedź może mieć MAKSYMALNIE dwa krótkie zdania. "
        "3. NIE używaj żadnych didaskaliów, nawiasów ani gwiazdek. "
        "4. Masz wiedzę o mechanikach gry: burze temporalne, płukanie rudy, driftery, kowalstwo. "
        "5. Jeśli chcesz zmienić stan świata, na samym końcu odpowiedzi dodaj [CMD: time set day], [CMD: time set night] lub [CMD: weather clear]. "
        f"Historia rozmowy z {player}: {history} "
    )
    
    full_prompt = f"{system_context} Odpowiedz na to: {prompt}"
    
    data = {
        "model": "gemma3:12b",
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
        sleep_time = random.randint(60, 180)
        print(f"\n[*] Autonomous mode sleeping for {sleep_time} seconds...")
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

    print("[*] Ready. Waiting for chat events...")

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
                print(f"\n[FILTER] Blocked toxic message from {player}.")
                continue
            
            print(f"\n[CHAT] {player}: {clean_msg}")
            
            ai_response = ask_ai(player, clean_msg)
            
            if ai_response:
                print(f"[AI] Response: {ai_response}")
                send_to_game(ai_response)

if __name__ == "__main__":
    main()
