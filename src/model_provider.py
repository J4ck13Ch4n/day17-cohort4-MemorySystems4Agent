from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

_ALIASES: dict[str, str] = {
    # common typos / variants
    "anthorpic": "anthropic",
    "anthrophic": "anthropic",
    "claude": "anthropic",
    "gemni": "gemini",
    "gemini-pro": "gemini",
    "google": "gemini",
    "google-genai": "gemini",
    "gpt": "openai",
    "open-ai": "openai",
    "custom-openai": "custom",
    "openai-compatible": "custom",
    "ollama-local": "ollama",
    "open-router": "openrouter",
    "open_router": "openrouter",
}


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None

    def __post_init__(self) -> None:
        self.provider = normalize_provider(self.provider)


def normalize_provider(value: str) -> str:
    """Normalize a provider name, mapping aliases/typos to canonical names."""
    v = (value or "").strip().lower().replace(" ", "").replace("_", "-")
    if v in _ALIASES:
        v = _ALIASES[v]
    if v not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider {value!r}. Supported: {sorted(SUPPORTED_PROVIDERS)}")
    return v


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    - `openai` -> `ChatOpenAI`
    - `custom` -> `ChatOpenAI` with `base_url`
    - `gemini` -> `ChatGoogleGenerativeAI`
    - `anthropic` -> `ChatAnthropic`
    - `ollama` -> `ChatOllama`
    - `openrouter` -> `ChatOpenRouter` (via langchain-openrouter) or
      OpenAI-compatible `ChatOpenAI` pointed at https://openrouter.ai/api/v1
    """
    provider = normalize_provider(config.provider)
    kwargs: dict = {
        "model": config.model_name,
        "temperature": config.temperature,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if provider == "custom":
        from langchain_openai import ChatOpenAI

        if not config.base_url:
            raise ValueError("`custom` provider requires base_url (OpenAI-compatible endpoint).")
        kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Need `langchain-google-genai` for gemini provider.") from exc
        gm_kwargs: dict = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            gm_kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**gm_kwargs)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Need `langchain-anthropic` for anthropic provider.") from exc
        an_kwargs: dict = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            an_kwargs["api_key"] = config.api_key
        if config.base_url:
            an_kwargs["base_url"] = config.base_url
        return ChatAnthropic(**an_kwargs)

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOllama
            except ImportError as exc:
                raise ImportError("Need `langchain-ollama` for ollama provider.") from exc
        ol_kwargs: dict = {"model": config.model_name, "temperature": config.temperature}
        if config.base_url:
            ol_kwargs["base_url"] = config.base_url
        return ChatOllama(**ol_kwargs)

    if provider == "openrouter":
        # Prefer native integration, fallback to OpenAI-compatible endpoint.
        try:
            from langchain_openrouter import ChatOpenRouter  # type: ignore

            or_kwargs: dict = {"model": config.model_name, "temperature": config.temperature}
            if config.api_key:
                or_kwargs["api_key"] = config.api_key
            return ChatOpenRouter(**or_kwargs)
        except ImportError:
            from langchain_openai import ChatOpenAI

            or_kwargs2: dict = {
                "model": config.model_name,
                "temperature": config.temperature,
                "base_url": config.base_url or "https://openrouter.ai/api/v1",
            }
            if config.api_key:
                or_kwargs2["api_key"] = config.api_key
            return ChatOpenAI(**or_kwargs2)

    raise ValueError(f"Unsupported provider: {provider}")
