"""
Ollama model configuration — single source for all LLM operations.

Mistral 7B - RTX 4060 8GB compatible:
- Q4_K_M quantization: ~4-5GB VRAM (model weights)
- Total with KV cache: ~6-7GB - fits comfortably in 8GB
- Install: ollama pull mistral:7b
"""

# Chat/reasoning model used everywhere (direct API + CrewAI)
OLLAMA_MODEL = "mistral:7b"

# CrewAI agents use this (format: ollama/model_name) — must match OLLAMA_MODEL
CREWAI_LLM = "ollama/mistral:7b"
