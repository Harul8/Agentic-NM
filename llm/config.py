"""
Ollama model configuration.

Mistral 7B - RTX 4060 8GB compatible:
- Q4_K_M quantization: ~4-5GB VRAM (model weights)
- Total with KV cache: ~6-7GB - fits comfortably in 8GB
- Install: ollama pull mistral:7b
"""

# Model for direct Ollama API calls (fact_collector, response_generator, etc.)
OLLAMA_MODEL = "mistral:7b"

# Model for CrewAI agents (format: ollama/model_name)
CREWAI_LLM = "ollama/mistral:7b"
