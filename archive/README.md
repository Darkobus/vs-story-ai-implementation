# archive/

Retired code, kept for reference. Not maintained, not tested, excluded from lint.

## train/ — fine-tuning experiment

Synthetic Q&A generation from the knowledge base (`generate_data.py`, teacher model via
Ollama), QLoRA fine-tuning of Qwen2.5-7B with Unsloth (`finetune.py`), and an Ollama
`Modelfile` for the GGUF export.

Retired because RAG with a relevance threshold turned out to be the better fit: the game's
facts change with updates and the wiki, and retrieval keeps answers grounded without
retraining. The prompts here still use the early "dark god" persona that the bot dropped in
favour of a plain helpful assistant. The scripts also read the old `vintage_story_wiki`
collection (nomic-embed-text, L2), not the current knowledge base.
