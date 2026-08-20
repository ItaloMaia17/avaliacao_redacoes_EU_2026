"""
Modelos selecionados na Seção 3.3 da metodologia, com seus identificadores
no OpenRouter.
"""

MODELS: dict[str, str] = {
    "llama-3.2-3b": "meta-llama/llama-3.2-3b-instruct",
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct:free",
    "ministral-3b": "mistralai/ministral-3b-2512",
    "ministral-14b-2512": "mistralai/ministral-14b-2512",
    "gemma-4-31b": "google/gemma-4-31b-it:free",
    "gemma-4-26b-a4b": "google/gemma-4-26b-a4b-it:free",
    "qwen3-next-80b-a3b-instruct": "qwen/qwen3-next-80b-a3b-instruct:free",
    "qwen3.6-27b": "qwen/qwen3.6-27b:free",
    "nvidia/nemotron-3-ultra-550b-a55b": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b": "nvidia/nemotron-3-super-120b-a12b:free",
    "microsoft/phi-4": "microsoft/phi-4",
    "microsoft/phi-4-mini-instruct": "microsoft/phi-4-mini-instruct",
    "openai/gpt-oss-20b": "openai/gpt-oss-20b:free",
    "openai/gpt-oss-120b": "openai/gpt-oss-120b:free",
    
}
PILOT_MODEL = "openai/gpt-oss-120b"
