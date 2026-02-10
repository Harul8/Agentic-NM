"""
Ollama model configuration — single source for all LLM operations.

RTX 4060 8GB — Qwen2.5 7B Instruct: best for this app (JSON, structured output, instructions).
Install: ollama pull qwen2.5:7b-instruct
"""

# Chat/reasoning model used everywhere (direct API + CrewAI)
OLLAMA_MODEL = "qwen2.5:7b-instruct"

# CrewAI agents use this (format: ollama/model_name) — must match OLLAMA_MODEL
CREWAI_LLM = "ollama/qwen2.5:7b-instruct"
