# Запуск OpenWebUI (порт 8080), подключённого к RAG-шлюзу.
# Требуется, чтобы Scripts/run_api.ps1 уже был запущен.
$root = Split-Path -Parent $PSScriptRoot
$env:DATA_DIR = "$root\storage\open-webui"
$env:WEBUI_SECRET_KEY = "dev-secret"
$env:ENABLE_OLLAMA_API = "false"
$env:OPENAI_API_BASE_URL = "http://127.0.0.1:8000/v1"
$env:OPENAI_API_KEY = "rag-local"
& "$root\.venv-webui\Scripts\open-webui.exe" serve --port 8080
