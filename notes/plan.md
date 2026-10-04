# План решения ДЗ «Enterprise Private GPT»

## Цель
Локальный защищённый корпоративный чат-бот (RAG): LLM + векторная БД,
отвечает строго по загруженным документам, без утечки данных в интернет.

## Принятые решения
- **LLM:** Ollama + `qwen2.5:3b-instruct` (основная, быстро на CPU),
  опционально `qwen2.5:7b-instruct` (качественнее, медленнее).
- **Эмбеддинги:** `intfloat/multilingual-e5-small` (sentence-transformers, CPU, RU).
- **Векторная БД:** ChromaDB (persistent, локально).
- **Документы:** датасет RuLegalNER (русские юр. документы), подвыборка.
- **Интерфейс:** OpenWebUI, подключённый к собственному OpenAI-совместимому
  FastAPI-эндпоинту RAG-пайплайна.

## Архитектура
```
Ollama (Qwen2.5)  <--  LangChain RAG (retriever -> prompt -> LLM)
                            ^
                       ChromaDB (эмбеддинги multilingual-e5-small)
                            ^
                    FastAPI /v1/chat/completions
                            ^
                       OpenWebUI (браузер)
```

## Структура
```
hw4/
  Scripts/
    config.py
    01_download_data.py
    02_prepare_docs.py
    03_ingest.py
    04_rag.py
    05_api.py
    06_evaluate.py
  notes/
  data/raw, data/processed
  storage/chroma/
  requirements.txt
  README.md
  .gitignore
  .env.example
```

## Этапы
1. Структура папок + план. [готово]
2. Окружение: Ollama, Git, Python 3.12 venv, зависимости.
3. Загрузка модели в Ollama, проверка API.
4. Загрузка RuLegalNER (gdown), извлечение чистого текста.
5. Индексация: `RecursiveCharacterTextSplitter(chunk_size=800, overlap=150)`,
   эмбеддинги, persistent Chroma с метаданными `source`.
6. RAG-ядро: top-K (по умолчанию 4), строгий системный промпт.
7. FastAPI OpenAI-совместимый шлюз.
8. OpenWebUI: connection на RAG API.
9. Тесты: по базе / вне базы / конфликт фактов + Top-K и latency.
10. README, requirements.txt, .gitignore.

## Риски
- RuLegalNER на Google Drive: формат и объём уточняются на шаге 4.
- `open-webui` несовместим с Python 3.13 -> отдельный venv на 3.12.
- CPU-инференс медленный -> основная модель 3B.
