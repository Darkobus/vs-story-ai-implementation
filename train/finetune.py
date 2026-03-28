"""
Fine-tuning Qwen2.5-7B-Instruct na danych VS (bot asystent + roleplay).
Używa Unsloth + QLoRA 4-bit.

Wymagania (poza projektem):
    pip install unsloth trl peft accelerate bitsandbytes datasets

Uruchom: python train/finetune.py
Wynik:
    train/vs-god-model/       <- adapter LoRA (HuggingFace format)
    train/vs-god-gguf/        <- model GGUF gotowy do Ollamy

RTX 5070 Ti (16GB VRAM): QLoRA 4-bit zużyje ~10-12GB podczas treningu.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "train" / "dataset.jsonl"
OUTPUT_ADAPTER = ROOT / "train" / "vs-god-model"
OUTPUT_GGUF = ROOT / "train" / "vs-god-gguf"

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 2048

# --- Parametry LoRA (pod 16GB VRAM) ---
LORA_R = 32          # wyższy rank = lepsza jakość, więcej VRAM
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# --- Parametry treningu ---
BATCH_SIZE = 4
GRAD_ACCUM = 2       # efektywny batch = 8
EPOCHS = 3
LR = 2e-4


def load_dataset_from_jsonl(path: Path):
    from datasets import Dataset
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[*] Wczytano {len(records)} przykładów z {path}")
    return Dataset.from_list(records)


def main():
    # --- Import Unsloth ---
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ImportError:
        raise ImportError(
            "Zainstaluj Unsloth: pip install unsloth trl peft accelerate bitsandbytes"
        )

    from trl import SFTTrainer, SFTConfig

    # --- 1. Załaduj model z kwantyzacją 4-bit ---
    print(f"[*] Ładowanie modelu {BASE_MODEL} (QLoRA 4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,  # auto-detect (bf16 na Blackwell/Ampere)
    )

    # Qwen2.5 używa ChatML — Unsloth obsługuje to automatycznie
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # --- 2. Dodaj adaptery LoRA ---
    print("[*] Dodawanie adapterów LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # oszczędność VRAM
        random_state=42,
    )

    # --- 3. Wczytaj dataset i zastosuj chat template ---
    dataset = load_dataset_from_jsonl(DATASET_PATH)

    def apply_template(examples):
        texts = tokenizer.apply_chat_template(
            examples["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": texts}

    dataset = dataset.map(apply_template, batched=True, remove_columns=["messages"])
    print(f"[*] Dataset po formatowaniu: {len(dataset)} przykładów")
    print(f"    Przykład:\n{dataset[0]['text'][:300]}\n...")

    # --- 4. Trening ---
    print("[*] Rozpoczynam trening...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            fp16=False,
            bf16=True,              # Blackwell/Ampere obsługuje bf16
            logging_steps=10,
            save_strategy="epoch",
            output_dir=str(OUTPUT_ADAPTER),
            report_to="none",       # zmień na "wandb" jeśli chcesz monitoring
            optim="adamw_8bit",     # oszczędność VRAM optymalizatora
            seed=42,
        ),
    )
    trainer.train()

    # --- 5. Zapisz adapter LoRA ---
    print(f"[*] Zapisuję adapter LoRA -> {OUTPUT_ADAPTER}")
    model.save_pretrained(str(OUTPUT_ADAPTER))
    tokenizer.save_pretrained(str(OUTPUT_ADAPTER))

    # --- 6. Eksport do GGUF (Q4_K_M — dobry balans jakość/rozmiar) ---
    print(f"[*] Eksportuję do GGUF (Q4_K_M) -> {OUTPUT_GGUF}")
    OUTPUT_GGUF.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(
        str(OUTPUT_GGUF),
        tokenizer,
        quantization_method="q4_k_m",
    )

    print("\n[OK] Gotowe!")
    print(f"  Adapter LoRA:  {OUTPUT_ADAPTER}")
    print(f"  Model GGUF:    {OUTPUT_GGUF}")
    print("\nNastępny krok:")
    print("  ollama create vs-god -f train/Modelfile")
    print("  ollama run vs-god 'test'")


if __name__ == "__main__":
    main()
