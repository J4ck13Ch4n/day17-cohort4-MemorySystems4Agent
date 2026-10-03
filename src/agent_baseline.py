from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-session memory only, no persistent User.md.

    - Remembers messages inside the same `thread_id`.
    - Forgets everything when a new thread starts (fair baseline).
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        if not self.force_offline:
            try:
                self._maybe_build_langchain_agent()
            except Exception:
                self.langchain_agent = None

    # -- public API --

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if not self.force_offline:
            try:
                return self._reply_live(thread_id, message)
            except Exception:
                pass
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        st = self.sessions.get(thread_id)
        return st.token_usage if st else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        st = self.sessions.get(thread_id)
        return st.prompt_tokens_processed if st else 0

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    # -- offline path (deterministic, used by benchmark/tests) --

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        st = self.sessions.get(thread_id)
        if st is None:
            st = SessionState()
            self.sessions[thread_id] = st

        # Prompt load = full history carried into this turn (before appending).
        history_tokens = sum(estimate_tokens(m.get("content", "")) for m in st.messages)
        incoming = estimate_tokens(message)
        st.prompt_tokens_processed += history_tokens + incoming

        st.messages.append({"role": "user", "content": message})

        # Baseline is intentionally naive: it never recalls facts from another
        # thread, so its answer must NOT contain long-term facts. It can only
        # echo the current message generically.
        text = (
            "Mình đã ghi nhận ý của bạn trong phiên này. "
            "Bạn có thể nhắc lại thông tin cụ thể để mình xử lý tiếp nhé."
        )
        st.messages.append({"role": "assistant", "content": text})
        reply_tokens = estimate_tokens(text)
        st.token_usage += incoming + reply_tokens

        return {
            "text": text,
            "agent_tokens": incoming + reply_tokens,
            "prompt_tokens": history_tokens + incoming,
            "thread_id": thread_id,
        }

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        # Minimal live path: keep thread history, call the chat model once.
        st = self.sessions.get(thread_id)
        if st is None:
            st = SessionState()
            self.sessions[thread_id] = st
        history_tokens = sum(estimate_tokens(m.get("content", "")) for m in st.messages)
        incoming = estimate_tokens(message)
        st.prompt_tokens_processed += history_tokens + incoming
        st.messages.append({"role": "user", "content": message})
        try:
            model = build_chat_model(self.config.model)
            from langchain_core.messages import HumanMessage

            res = model.invoke([HumanMessage(content=message)])
            text = getattr(res, "content", str(res))
        except Exception as exc:
            text = f"[live-error] {exc}"
        st.messages.append({"role": "assistant", "content": text})
        reply_tokens = estimate_tokens(text)
        st.token_usage += incoming + reply_tokens
        return {"text": text, "agent_tokens": incoming + reply_tokens,
                "prompt_tokens": history_tokens + incoming, "thread_id": thread_id}

    def _maybe_build_langchain_agent(self):
        """Optionally wire `create_agent` + `InMemorySaver` here."""
        if self.force_offline:
            self.langchain_agent = None
            return None
        try:
            model = build_chat_model(self.config.model)
        except Exception:
            self.langchain_agent = None
            return None
        try:
            try:
                from langchain.agents import create_agent as _create  # type: ignore
            except ImportError:
                try:
                    from langgraph.prebuilt import create_react_agent as _create  # type: ignore
                except ImportError:
                    from langgraph.prebuilt import create_agent as _create  # type: ignore

            try:
                from langgraph.checkpoint.memory import InMemorySaver  # type: ignore

                checkpointer = InMemorySaver()
            except Exception:
                checkpointer = None
            if checkpointer is not None:
                try:
                    self.langchain_agent = _create(model, tools=[], checkpointer=checkpointer)
                except TypeError:
                    self.langchain_agent = _create(model, tools=[])
            else:
                self.langchain_agent = _create(model, tools=[])
        except Exception:
            # Live agent is optional; offline path always works.
            self.langchain_agent = None
        return self.langchain_agent
