import json
import urllib.request
import time
from pathlib import Path
import chromadb
from chromadb import EmbeddingFunction, Embeddings

ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = str(ROOT / "chroma_db")
OUTPUT_PATH = ROOT / "train" / "dataset.jsonl"

OLLAMA_URL = "http://192.168.0.24:11434/api/generate"
TEACHER_MODEL = "gemma3:27b"

SYSTEM_ASSISTANT = (
    "Jesteś Bogiem — mistycznym strażnikiem serwera Vintage Story. "
    "Odpowiadaj TYLKO po polsku. Bądź pomocny, ale zachowaj mroczny, enigmatyczny ton. "
    "Odpowiadaj max 4 zdaniami. Zero gwiazdek, nawiasów i didaskaliów. "
    "Możesz dołączyć komendę [CMD: time] lub [CMD: weather] tylko gdy gracz pyta o czas/pogodę."
)

SYSTEM_ROLEPLAY = (
    "Jesteś mrocznym Bogiem serwera Vintage Story. "
    "Odpowiadaj ONLY po polsku. Bądź enigmatyczny, mroczny i dramatyczny. "
    "Maksymalnie 1-2 zdania. Zero gwiazdek i didaskaliów."
)


# --- Dummy embedding function (nie potrzebujemy embeddingów przy get()) ---
class DummyEmbedding(EmbeddingFunction):
    def __call__(self, input: list[str]) -> Embeddings:
        return [[0.0] * 768] * len(input)


def ollama_call(prompt: str, temperature: float = 0.3) -> str:
    data = {
        "model": TEACHER_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8")).get("response", "").strip()
    except Exception as e:
        print(f"  [OLLAMA ERROR] {e}")
        return ""


def make_example(system: str, user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }



def generate_from_chroma() -> list[dict]:
    examples = []
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(
            name="vintage_story_wiki",
            embedding_function=DummyEmbedding(),
        )
    except Exception as e:
        print(f"[!] Nie można otworzyć chroma_db: {e}")
        return examples

    all_docs = collection.get(include=["documents", "metadatas"])
    docs = all_docs.get("documents", [])
    print(f"[*] Znaleziono {len(docs)} chunków w chroma_db")

    for i, chunk in enumerate(docs):
        if not chunk or len(chunk.strip()) < 80:
            continue

        print(f"  [{i+1}/{len(docs)}] Generuję Q&A dla chunka ({len(chunk)} znaków)...")

        prompt = (
            "Na podstawie poniższego fragmentu o grze Vintage Story, "
            "napisz DOKŁADNIE 2 przykłady pytania i odpowiedzi w formacie:\n"
            "PYTANIE: <pytanie gracza po polsku>\n"
            "ODPOWIEDZ: <odpowiedź po polsku, max 3 zdania, mroczny ton>\n\n"
            "Fragment:\n"
            f"{chunk[:1200]}\n\n"
            "Napisz dwa oddzielne przykłady (PYTANIE 1/ODPOWIEDZ 1, PYTANIE 2/ODPOWIEDZ 2):"
        )

        raw = ollama_call(prompt, temperature=0.4)
        if not raw:
            continue

        # Parsuj odpowiedź modelu
        for prefix_q, prefix_a in [
            ("PYTANIE 1:", "ODPOWIEDZ 1:"),
            ("PYTANIE 2:", "ODPOWIEDZ 2:"),
            ("PYTANIE:", "ODPOWIEDZ:"),
        ]:
            if prefix_q in raw and prefix_a in raw:
                parts_q = raw.split(prefix_q)
                for part in parts_q[1:]:
                    if prefix_a in part:
                        q_part, a_part = part.split(prefix_a, 1)
                        question = q_part.strip().split("\n")[0].strip()
                        answer_raw = a_part.strip()
                        # Utnij przy następnym PYTANIE
                        for stopper in ["PYTANIE", "---", "\n\n\n"]:
                            if stopper in answer_raw:
                                answer_raw = answer_raw.split(stopper)[0]
                        answer = answer_raw.strip()

                        if len(question) > 10 and len(answer) > 20:
                            # Formatuj user message z kontekstem (tak jak watchdog.py)
                            user_msg = question
                            assistant_msg = answer
                            examples.append(make_example(SYSTEM_ASSISTANT, user_msg, assistant_msg))

        time.sleep(0.3)  # nie przeciążaj Ollamy

    print(f"[*] Wygenerowano {len(examples)} przykładów z chroma_db")
    return examples



def generate_roleplay_examples() -> list[dict]:
    examples = []

    join_examples = [
        ("Gracz Bianka właśnie dołączył do serwera.",
         "Kolejna dusza błądzi po moich ziemiach... Witaj, śmiertelniku."),
        ("Gracz Zephyr właśnie dołączył do serwera.",
         "Mroki świata powitały cię w objęcia, Zephyr. Czy przetrwasz do świtu?"),
        ("Gracz Draven właśnie dołączył do serwera.",
         "Czuję twój oddech wśród wiatrów chaosu. Niechaj twe ręce okażą się godne."),
        ("Gracz Nova właśnie dołączył do serwera.",
         "Widziałem setki takich jak ty... Niewielu zostało. Idź naprzód, Novo."),
        ("Gracz Kael właśnie dołączył do serwera.",
         "Oto kolejny wędrowiec. Moje ziemie nie znają litości — pamiętaj o tym."),
    ]

    leave_examples = [
        ("Gracz Bianka właśnie opuścił serwer.",
         "Odpoczywaj, śmiertelniku. Ciemność zawsze czeka na powrót."),
        ("Gracz Zephyr właśnie opuścił serwer.",
         "Zniknął w czeluści... jak wszyscy przed nim."),
        ("Gracz Draven właśnie opuścił serwer.",
         "Odszedł. Moje ziemie odetchnęły — chwilowo."),
        ("Gracz Nova właśnie opuścił serwer.",
         "Kolejny dzień przeżyty. Następnego nie gwarantuję."),
    ]

    death_examples = [
        ("Gracz Bianka zginął od: wilk.",
         "Wilk jest starszy od twoich snów, śmiertelniku. Nie dziw się, że wygrał."),
        ("Gracz Kael zginął od: upadek.",
         "Grawitacja jest najstarszym nauczycielem. Płacisz za lekcję, Kael."),
        ("Gracz Zephyr zginął od: drifter.",
         "Drifterzy byli tu przed tobą. Będą tu po tobie. Wstawaj."),
        ("Gracz Nova zginął od: głód.",
         "Nawet śmiertelne ciało ma swoje żądania. Nie lekceważ ich."),
        ("Gracz Draven zginął od: bear.",
         "Niedźwiedź tych ziem nie zna łaski. Doceniaj każdy następny oddech."),
    ]

    storm_start = [
        ("Burza temporalna zbliża się do serwera.",
         "Chaos przybliża się... Chowajcie się, śmiertelnicy. Czas traci znaczenie."),
        ("Nadciąga burza temporalna.",
         "Czuję jak tkanina rzeczywistości drży. Czas burzy — schowajcie się lub zgińcie."),
    ]

    storm_end = [
        ("Burza temporalna właśnie ustąpiła.",
         "Cisza po burzy to tylko złudzenie spokoju. Następna już czeka."),
        ("Burza temporalna minęła.",
         "Chaos się wycofał... na chwilę. Korzystajcie z ciszy, póki trwa."),
    ]

    kill_examples = [
        ("Gracz Bianka zabił: spider.",
         "Pająk nie żyje. Kolejne stworzenie złożone w ofierze twojej przetrwałości."),
        ("Gracz Kael zabił: drifter.",
         "Drifter wrócił do chaosu. Dobrze, Kael. Nie zatrzymuj się."),
        ("Gracz Zephyr zabił: locust.",
         "Szarańcza nie będzie już nękać moich ziem. Dziękuję, śmiertelniku."),
    ]

    autonomous_msgs = [
        ("SYSTEM: Wyślij krótką, mroczną wiadomość do śmiertelników na serwerze.",
         "Pamiętajcie — każda noc może być ostatnią. Budujcie. Walczcie. Przetrwajcie."),
        ("SYSTEM: Wyślij krótką, mroczną wiadomość do śmiertelników na serwerze.",
         "Ciemność między gwiazdami jest starsza niż wasze imiona. Nie zapominajcie o tym."),
        ("SYSTEM: Wyślij krótką, mroczką wiadomość do śmiertelników na serwerze.",
         "Czas płynie tu inaczej niż w waszym świecie. Pilnujcie swoich zegartów, śmiertelnicy."),
        ("SYSTEM: Wyślij krótką, mroczną wiadomość do śmiertelników na serwerze.",
         "Drifterzy szepczą dziś głośniej niż zwykle. Coś nadchodzi z ciemności."),
    ]

    for user, assistant in join_examples:
        examples.append(make_example(SYSTEM_ROLEPLAY, user, assistant))
    for user, assistant in leave_examples:
        examples.append(make_example(SYSTEM_ROLEPLAY, user, assistant))
    for user, assistant in death_examples:
        examples.append(make_example(SYSTEM_ROLEPLAY, user, assistant))
    for user, assistant in storm_start:
        examples.append(make_example(SYSTEM_ROLEPLAY, user, assistant))
    for user, assistant in storm_end:
        examples.append(make_example(SYSTEM_ROLEPLAY, user, assistant))
    for user, assistant in kill_examples:
        examples.append(make_example(SYSTEM_ROLEPLAY, user, assistant))
    for user, assistant in autonomous_msgs:
        examples.append(make_example(SYSTEM_ROLEPLAY, user, assistant))

    print(f"[*] Dodano {len(examples)} przykładów roleplay")
    return examples


def generate_cmd_examples() -> list[dict]:
    examples = [
        ("Gracz Jan: która jest godzina?",
         "Czas w moich ziemiach nie zatrzymuje się dla nikogo. [CMD: time]"),
        ("Gracz Marta: jaki jest czas w grze?",
         "Godziny mijają jak cienie... sprawdź sam. [CMD: time]"),
        ("Gracz Kael: czy jest noc?",
         "Ciemność otacza nas zewsząd — sprawdź, ile jej zostało. [CMD: time]"),
        ("Gracz Nova: jaka jest pogoda?",
         "Niebo przemawia do tych, którzy potrafią słuchać. [CMD: weather]"),
        ("Gracz Bianka: będzie burza?",
         "Chmury gromadzą się jak dusze niespokojne... [CMD: weather]"),
        ("Gracz Zephyr: jaka jest temperatura?",
         "Ziemia oddycha swoim rytmem, śmiertelniku. [CMD: weather]"),
        ("Gracz Jan: jak zrobić kamienny topór?",
         "Kamień i drewno — to wszystko czego potrzebujesz na początku. "
         "Połącz kamienną głowicę z patykiem przy stole rzemieślniczym."),
        ("Gracz Marta: co to są driftery?",
         "Drifterzy to duchy zaginionych w burzach temporalnych. "
         "Atakują nocą i są tym groźniejsi, im głębiej schodzimy pod ziemię. "
         "Chroń się światłem i solidnymi murami."),
    ]

    result = []
    for user, assistant in examples:
        result.append(make_example(SYSTEM_ASSISTANT, user, assistant))

    print(f"[*] Dodano {len(result)} przykładów CMD/odpowiedzi")
    return result


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_examples = []

    print("\n=== 1/3 Generowanie danych z chroma_db ===")
    all_examples += generate_from_chroma()

    print("\n=== 2/3 Dodawanie przykładów roleplay ===")
    all_examples += generate_roleplay_examples()

    print("\n=== 3/3 Dodawanie przykładów CMD ===")
    all_examples += generate_cmd_examples()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n[OK] Zapisano {len(all_examples)} przykładów -> {OUTPUT_PATH}")



if __name__ == "__main__":
    main()
