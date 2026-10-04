# Запуск OpenAI-совместимого RAG-шлюза (порт 8000).
$root = Split-Path -Parent $PSScriptRoot
& "$root\.venv\Scripts\python.exe" -X utf8 "$PSScriptRoot\05_api.py"
