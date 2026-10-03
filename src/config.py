from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared configuration for the lab.

    - Paths: repo root, dataset dir, state dir.
    - Compact-memory settings: threshold + messages to keep.
    - Provider settings for main model and judge model.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _getenv(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load environment variables and return a LabConfig.

    1. Resolve the repo root (default: parent of this file's directory).
    2. Optionally load values from `.env`.
    3. Create `state/` if it does not exist.
    4. Return a populated LabConfig instance.
    """
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    # Load .env if present (best-effort, never fail the lab if missing).
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except Exception:
        pass

    data_dir = root / "data"
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Compact-memory defaults: small enough to trigger on the stress test,
    # large enough to avoid compacting every short conversation.
    compact_threshold_tokens = int(_getenv("COMPACT_THRESHOLD_TOKENS", "1200") or "1200")
    compact_keep_messages = int(_getenv("COMPACT_KEEP_MESSAGES", "6") or "6")

    # Provider settings for main + judge models.
    provider = normalize_provider(_getenv("LLM_PROVIDER", "openai") or "openai")
    model_name = _getenv("LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
    temperature = float(_getenv("LLM_TEMPERATURE", "0") or "0")

    judge_provider_raw = _getenv("JUDGE_PROVIDER", provider) or provider
    judge_provider = normalize_provider(judge_provider_raw)
    judge_model_name = _getenv("JUDGE_MODEL", model_name) or model_name

    # Collect credentials / endpoints for every supported provider.
    openai_key = _getenv("OPENAI_API_KEY")
    gemini_key = _getenv("GEMINI_API_KEY") or _getenv("GOOGLE_API_KEY")
    anthropic_key = _getenv("ANTHROPIC_API_KEY")
    openrouter_key = _getenv("OPENROUTER_API_KEY")
    custom_key = _getenv("CUSTOM_API_KEY") or _getenv("OPENAI_API_KEY")
    custom_base = _getenv("CUSTOM_BASE_URL")
    ollama_base = _getenv("OLLAMA_BASE_URL") or _getenv("OLLAMA_HOST")

    def _creds_for(prov: str) -> tuple[str | None, str | None]:
        if prov == "openai":
            return openai_key, None
        if prov == "custom":
            return custom_key, custom_base
        if prov == "gemini":
            return gemini_key, None
        if prov == "anthropic":
            return anthropic_key, None
        if prov == "ollama":
            return None, ollama_base
        if prov == "openrouter":
            return openrouter_key, _getenv("OPENROUTER_BASE_URL")
        return None, None

    api_key, base_url = _creds_for(provider)
    judge_key, judge_base = _creds_for(judge_provider)

    model = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
    )
    judge_model = ProviderConfig(
        provider=judge_provider,
        model_name=judge_model_name,
        temperature=0.0,
        api_key=judge_key,
        base_url=judge_base,
    )

    return LabConfig(
        base_dir=root,
        data_dir=data_dir,
        state_dir=state_dir,
        compact_threshold_tokens=compact_threshold_tokens,
        compact_keep_messages=compact_keep_messages,
        model=model,
        judge_model=judge_model,
    )
