"""NEXUS RAG — LLM Integration Layer
Production-grade wrapper for local Ollama and optional Groq cloud fallback.
"""

import time
from typing import Dict, Any, List, Optional
from rag.prompts import build_llm_prompt, SYSTEM_PROMPT, INSUFFICIENT_EVIDENCE_REPLY

# ──────────────────────────────────────────────
# OLLAMA CLIENT
# ──────────────────────────────────────────────
class OllamaClient:
    """Wrapper for local Ollama daemon."""

    @staticmethod
    def is_available() -> bool:
        try:
            import ollama
            ollama.list()
            return True
        except Exception:
            return False

    @staticmethod
    def get_available_models() -> List[str]:
        try:
            import ollama
            res = ollama.list()
            models = [m.get("name", m.get("model", "")) for m in res.get("models", [])]
            return [m for m in models if m] or ["mistral", "llama3", "phi3"]
        except Exception:
            return ["mistral", "llama3", "phi3", "gemma2"]

    @staticmethod
    def generate(
        model: str,
        context: str,
        question: str,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """Generates answer using local Ollama model."""
        import ollama

        start_time = time.perf_counter()
        prompt = build_llm_prompt(context, question)

        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature}
            )
            elapsed = time.perf_counter() - start_time
            answer = response["message"]["content"].strip()

            return {
                "success": True,
                "answer": answer,
                "latency_ms": round(elapsed * 1000, 2),
                "model": model,
                "error": None
            }
        except Exception as e:
            err_msg = str(e)
            if "connection refused" in err_msg.lower() or "not found" in err_msg.lower():
                guidance = "Ollama daemon unreachable. Please start it in a terminal using: 'ollama serve'"
            else:
                guidance = f"Ollama execution error: {err_msg}"
            return {
                "success": False,
                "answer": f"SYSTEM ERROR: {guidance}",
                "latency_ms": 0.0,
                "model": model,
                "error": guidance
            }

# ──────────────────────────────────────────────
# GROQ CLOUD CLIENT (FREE TIER CLOUD BACKEND)
# ──────────────────────────────────────────────
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

class GroqClient:
    """Free tier cloud client with dynamic model discovery and self-healing fallback."""

    @staticmethod
    def get_available_models(api_key: str = "") -> List[str]:
        if not api_key:
            return GROQ_MODELS
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            m_list = client.models.list()
            chat_models = [
                m.id for m in m_list.data
                if not any(x in m.id.lower() for x in ["whisper", "guard", "vision", "embed"])
            ]
            return chat_models or GROQ_MODELS
        except Exception:
            return GROQ_MODELS

    @staticmethod
    def generate(
        api_key: str,
        model: str,
        context: str,
        question: str,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        from groq import Groq

        if not api_key:
            return {
                "success": False,
                "answer": "Groq API key required. Obtain a free key at https://console.groq.com (no credit card required).",
                "latency_ms": 0.0,
                "model": model,
                "error": "Missing API Key"
            }

        start_time = time.perf_counter()
        prompt = build_llm_prompt(context, question)

        try:
            client = Groq(api_key=api_key)

            # 1. Primary Attempt with requested model
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1024
                )
                elapsed = time.perf_counter() - start_time
                answer = resp.choices[0].message.content.strip()

                return {
                    "success": True,
                    "answer": answer,
                    "latency_ms": round(elapsed * 1000, 2),
                    "model": model,
                    "error": None
                }
            except Exception as initial_err:
                err_str = str(initial_err)
                is_model_issue = any(k in err_str.lower() for k in [
                    "model_not_found", "does not exist", "decommissioned", "not have access", "invalid_request_error"
                ])
                if not is_model_issue:
                    raise initial_err

                # 2. Dynamic Discovery: query active models directly from user's Groq account
                m_list = client.models.list()
                candidates = [
                    m.id for m in m_list.data
                    if not any(x in m.id.lower() for x in ["whisper", "guard", "vision", "embed"])
                    and m.id != model
                ]

                last_err = err_str
                for candidate in candidates:
                    try:
                        resp = client.chat.completions.create(
                            model=candidate,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=temperature,
                            max_tokens=1024
                        )
                        elapsed = time.perf_counter() - start_time
                        answer = resp.choices[0].message.content.strip()
                        return {
                            "success": True,
                            "answer": answer,
                            "latency_ms": round(elapsed * 1000, 2),
                            "model": candidate,
                            "error": None
                        }
                    except Exception as try_err:
                        last_err = str(try_err)
                        continue

                raise Exception(f"All available Groq models failed. Last error: {last_err}")

        except Exception as e:
            return {
                "success": False,
                "answer": f"Groq API Error: {str(e)}",
                "latency_ms": 0.0,
                "model": model,
                "error": str(e)
            }

