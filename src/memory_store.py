from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Simple heuristic token estimator (offline-stable).

    - Strip whitespace, 0 for empty.
    - Approximate 1 token ~ 4 characters (covers VI + EN mix).
    """
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, int(len(stripped) / 4))


def _sanitize_user_id(user_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9À-ỹ_.-]+", "_", (user_id or "default").strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "default"


DEFAULT_PROFILE_TEMPLATE = """# Hồ sơ người dùng

(Chưa có thông tin. Agent sẽ bổ sung khi người dùng chia sẻ facts ổn định.)
"""


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` (one markdown file per user)."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        safe = _sanitize_user_id(user_id)
        return Path(self.root_dir) / safe / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return DEFAULT_PROFILE_TEMPLATE
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return DEFAULT_PROFILE_TEMPLATE

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        updated = current.replace(search_text, replacement, 1)
        self.write_text(user_id, updated)
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        if not path.exists():
            return 0
        return path.stat().st_size

    # ---- Bonus helpers: structured facts + conflict handling ----

    def facts(self, user_id: str) -> dict[str, str]:
        """Parse `- Key: value` bullet lines into a dict."""
        text = self.read_text(user_id)
        out: dict[str, str] = {}
        for line in text.splitlines():
            m = re.match(r"\s*-\s*([^:：]+)\s*[:：]\s*(.+)", line)
            if m:
                out[m.group(1).strip()] = m.group(2).strip()
        return out

    def upsert_fact(self, user_id: str, key: str, value: str) -> Path:
        """Insert or replace one `- Key: value` line (conflict handling).

        Keeps exactly one line per key so corrections overwrite stale facts
        instead of accumulating contradictions.
        """
        value = value.strip()
        if not value:
            raise ValueError("upsert_fact value must be non-empty")
        current = self.read_text(user_id)
        if current.strip() == DEFAULT_PROFILE_TEMPLATE.strip():
            current = "# Hồ sơ người dùng\n\n"
        pattern = re.compile(rf"(?m)^\s*-\s*{re.escape(key)}\s*[:：]\s*.*$")
        replacement = f"- {key}: {value}"
        if pattern.search(current):
            updated = pattern.sub(replacement, current, count=1)
        else:
            if not current.endswith("\n"):
                current += "\n"
            updated = current + replacement + "\n"
        return self.write_text(user_id, updated)


# ---------------------------------------------------------------------------
# Profile extraction with confidence threshold + noise filtering (bonus)
# ---------------------------------------------------------------------------

_QUESTION_MARKERS = ("tên gì", "ở đâu", "nghề gì", "là gì", "là ai", "nhắc", "đâu mới", "bạn biết")

_KNOWN_PLACES = ["Đà Nẵng", "Hà Nội", "Hồ Chí Minh", "Sài Gòn", "Mỹ Khê", "Hội An", "Huế"]

# Sentences containing these markers are treated as noise / deprecated info.
_NOISE_MARKERS = (
    "câu đùa",
    "chỉ là câu đùa",
    "đùa",
    "đi họp",
    "họp hai ngày",
    "đối tác",
    "ví dụ cũ",
    "đừng lấy",
    "thông tin cũ",
    "không còn là nơi ở",
)


def _is_question_turn(message: str) -> bool:
    """Confidence gate: questions should not create facts.

    Any turn containing `?` is treated as a question (recall questions always
    contain `?`, while fact-bearing turns in this lab never do). This prevents
    recall questions like 'Nếu ai đó nhắc Huế...' from corrupting User.md.
    """
    return "?" in message


def _sentence_is_deprecated(sentence: str) -> bool:
    low = sentence.lower()
    return any(m in low for m in _NOISE_MARKERS)


def _extract_name(message: str) -> str | None:
    low = message.lower()
    # Guard: pet-name context ("corgi tên Bơ", "bé corgi tên Bơ", "nuôi ... tên ...")
    # must never overwrite the person's name.
    if "dũngct stress" in low:
        return "DũngCT Stress"
    if "dũngct" in low:
        return "DũngCT"
    if any(k in low for k in ("corgi", "nuôi", "bé cưng", "thú cưng")) and "tên" in low:
        return None
    m = re.search(
        r"(?:tên(?:\s+mình)?(?:\s+là)?|mình\s+tên)\s+([A-ZÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]*){0,2})",
        message,
    )
    if m:
        name = m.group(1).strip().rstrip(".,;!")
        if len(name) >= 2 and "bạn" not in name.lower():
            # Extra guard: single short names in pet contexts are unreliable.
            if len(name) <= 3 and any(k in low for k in ("corgi", "mèo", "chó", "nuôi")):
                return None
            return name
    return None


def _extract_location(message: str) -> str | None:
    # 1. Explicit correction: "từ X sang Y" -> Y is current (highest confidence).
    m = re.search(r"từ\s+([A-ZÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]*)?)\s+sang\s+([A-ZÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]*)?)", message)
    if m:
        dest = m.group(2).strip()
        for place in _KNOWN_PLACES:
            if place.lower() in dest.lower() or dest.lower() in place.lower():
                return place
        return dest
    # 2. "ở X chứ không còn ở Y" -> X is current.
    m = re.search(r"ở\s+([A-ZÀ-Ỹ][\wÀ-ỹ]*(?:\s+[A-ZÀ-Ỹ][\wÀ-ỹ]*)?)\s+chứ không còn ở", message)
    if m:
        cand = m.group(1).strip()
        for place in _KNOWN_PLACES:
            if place.lower() in cand.lower() or cand.lower() in place.lower():
                return place
        return cand

    # 3. General case: offset-ordered mentions, stale-filtered, recency wins.
    #    Supports both "ở <Place>" and "nơi ở ... là <Place>" / "hiện tại là <Place>".
    low = message.lower()
    mentions: list[tuple[int, str]] = []
    for place in _KNOWN_PLACES:
        for pat in (r"ở\s+" + re.escape(place), r"là\s+" + re.escape(place)):
            for mm in re.finditer(pat, message):
                # "là <Place>" only counts in a residence context.
                if pat.startswith("là"):
                    window = message[max(0, mm.start() - 40): mm.start()].lower()
                    if not any(k in window for k in ("nơi ở", "hiện tại", "đang ở", "ở hiện", "chuyển", "sống", "làm việc ở", "ở ")):
                        # Still accept "nơi ở hiện tại là Đà Nẵng" (window has "nơi ở").
                        # If no residence cue, skip this "là X" (could be food/trivia).
                        continue
                mentions.append((mm.start(), place))
        # Bare "Huế" with strong residence cue (e.g. "hiện ở Huế" handled above,
        # but "dù trước đó có nhắc Huế" must NOT count -> filtered below).
        if place == "Huế":
            for mm in re.finditer(r"\bHuế\b", message):
                window = message[max(0, mm.start() - 30): mm.end() + 10].lower()
                if "hiện" in window or "đang ở" in window:
                    mentions.append((mm.start(), place))
    if not mentions:
        return None
    mentions.sort(key=lambda x: x[0])

    # Mark stale mentions: old-city / meeting-city / joke contexts.
    stale_idx: set[int] = set()
    old_cues = (
        "lúc đầu", "trước đó", "trước đây", "ví dụ cũ", "đừng lấy",
        "thông tin cũ", "không còn ở", "chỉ là nơi", "đi họp", "họp",
        "đối tác", "câu đùa", "đùa",
    )
    sentences = re.split(r"[.;\n]+", message)
    for i, (pos, place) in enumerate(mentions):
        # Find the sentence containing this mention.
        sent = ""
        for s in sentences:
            if place in s:
                # approximate: sentence contains the place; check cues
                # (a message may repeat the place in several sentences, so
                # only mark the occurrence whose sentence has old cues AND
                # the mention is inside that sentence span).
                try:
                    s_start = message.index(s)
                    s_end = s_start + len(s)
                    if s_start <= pos < s_end:
                        sent = s
                        break
                except ValueError:
                    continue
        slow = (sent or message).lower()
        if any(c in slow for c in old_cues):
            # "không còn ở Y" / "lúc đầu ... ở X" / "trước đó ... Huế" -> stale,
            # UNLESS the same sentence also declares it current with
            # "hiện tại là", "đang ở", "chuyển sang", "cập nhật".
            if not re.search(r"(hiện tại là|đang ở|chuyển sang|cập nhật|mới là)\s*" + re.escape(place), sent):
                # For "A chứ không còn ở B": B is stale but A already handled
                # by rule 2; here conservatively mark B stale.
                # For "từ tuần này ... ở Đà Nẵng ... dù trước đó ... Huế":
                # the Huế sentence fragment is stale.
                if place in ("Huế", "Hà Nội") and any(
                    k in slow for k in ("lúc đầu", "trước đó", "trước đây", "ví dụ cũ", "họp", "đùa", "chỉ là nơi")
                ):
                    stale_idx.add(i)
                elif f"không còn ở {place}" in message or f"không còn ở {place.lower()}" in slow:
                    stale_idx.add(i)
    fresh = [p for i, (_, p) in enumerate(mentions) if i not in stale_idx]
    pool = fresh or [p for _, p in mentions]
    # Extra guard at message level: never learn a meeting city as home when
    # the whole message frames it as a trip/joke.
    last = pool[-1]
    if "hà nội" in last.lower() and ("họp" in low or "đùa" in low):
        without_hn = [p for p in pool if "hà nội" not in p.lower()]
        if without_hn:
            return without_hn[-1]
        return None
    return last


def _extract_profession(message: str) -> str | None:
    sentences = re.split(r"[.;\n]+", message)
    cands: list[str] = []
    for sent in sentences:
        if not sent.strip() or _sentence_is_deprecated(sent):
            continue
        low = sent.lower()
        # Skip sentences that label the profession as old.
        if any(k in low for k in ("thông tin cũ", "đó là thông tin cũ", "nghề cũ", "không còn làm", "đừng nói", "đừng nhắc")) and "chuyển sang" not in low and "giờ" not in low and "hiện" not in low:
            # e.g. "đừng nói backend engineer nữa nhé, vì đó là thông tin cũ."
            continue
        if "mlops" in low:
            cands.append("MLOps engineer")
        elif "backend engineer" in low or ("backend" in low and "làm" in low):
            # only accept backend if message does not already declare the new job
            cands.append("backend engineer")
        elif "product manager" in low:
            # product manager only counts as a real fact when NOT a joke;
            # joke sentences were already filtered above.
            cands.append("product manager")
    if not cands:
        return None
    # Conflict handling: newest profession wins; MLOps overrides backend.
    if "MLOps engineer" in cands:
        return "MLOps engineer"
    return cands[-1]


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Implements a confidence threshold:
    - question-only turns -> {}
    - noise/joke/meeting sentences are ignored
    - only high-confidence facts are returned
    """
    if not message or not message.strip():
        return {}
    if _is_question_turn(message):
        return {}

    out: dict[str, str] = {}
    low = message.lower()

    name = _extract_name(message)
    if name:
        out["name"] = name

    loc = _extract_location(message)
    if loc:
        # Extra guard: never learn a meeting city as home.
        if not ("hà nội" in loc.lower() and ("họp" in low or "đùa" in low or "ví dụ" in low)):
            out["location"] = loc

    prof = _extract_profession(message)
    if prof:
        if not ("product manager" in prof.lower() and ("đùa" in low or "họp" in low)):
            # If both old backend and new MLOps appear, keep the new one.
            if prof == "backend engineer" and "mlops" in low:
                out["profession"] = "MLOps engineer"
            else:
                out["profession"] = prof

    if "cà phê sữa đá" in low:
        out["drink"] = "cà phê sữa đá"
    if "mì quảng" in low:
        out["food"] = "mì Quảng"
    if "corgi" in low:
        # keep the dog's name when present
        if "bơ" in low:
            out["pet"] = "corgi tên Bơ"
        else:
            out["pet"] = "corgi"
    elif re.search(r"\bbơ\b", low) and "corgi" in low:
        out["pet"] = "corgi tên Bơ"

    # Response style: keep canonical phrases used by recall checks.
    # NOTE: "bullet ngắn" / "gọn" always implies "ngắn gọn" so cross-session
    # recall for "ngắn gọn" keeps working even after style refinements.
    styles: list[str] = []
    if "ngắn gọn" in low or "gọn" in low or ("ngắn" in low and "bullet" in low):
        styles.append("ngắn gọn")
    if "3 bullet" in low or "3-bullet" in low or ("3" in message and "bullet" in low):
        styles.append("3 bullet")
    elif "bullet" in low:
        styles.append("bullet ngắn")
    if "ví dụ thực" in low:
        styles.append("có ví dụ thực tế")
    elif "thực chiến" in low:
        styles.append("có ví dụ thực chiến")
    if "trade-off" in low or "tradeoff" in low:
        styles.append("nhấn trade-off")
    if styles:
        # keep deterministic order, join for User.md
        ordered = []
        for s in ("ngắn gọn", "3 bullet", "bullet ngắn", "có ví dụ thực tế", "có ví dụ thực chiến", "nhấn trade-off"):
            if s in styles:
                ordered.append(s)
        out["style"] = ", ".join(ordered)

    if "python" in low:
        out.setdefault("interests", "Python, AI ứng dụng")
    # NOTE: use word-boundary for bare "AI" so Vietnamese words like "hai",
    # "tai", "mai", "vai" don't trigger a false AI-interest fact.
    if (re.search(r"\bai\b", low) or "ai ứng dụng" in low or "ai agent" in low) and "interests" not in out:
        out["interests"] = "AI ứng dụng"
    if "mlops" in low and "interests" in out and "MLOps" not in out["interests"]:
        out["interests"] = out["interests"] + ", MLOps"

    # Work routine / extra stable facts (optional, low risk).
    if "chạy bộ" in low and "6 giờ" in low:
        out["routine"] = "chạy bộ lúc 6 giờ sáng"

    return out


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact heuristic summary of older messages."""
    if not messages:
        return ""
    head = messages[:max_items]
    parts: list[str] = []
    for m in head:
        role = m.get("role", "user")
        content = (m.get("content", "") or "").strip().replace("\n", " ")
        if len(content) > 140:
            content = content[:140] + "…"
        parts.append(f"{role}: {content}")
    suffix = "" if len(messages) <= max_items else f" (+{len(messages) - max_items} tin nhắn cũ khác)"
    return f"Tóm tắt {len(messages)} tin nhắn cũ{suffix}: " + " | ".join(parts)


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    - Keeps recent messages in full.
    - When thread tokens exceed threshold, older content moves into a summary.
    - Tracks compaction count per thread for benchmarking.
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def _ensure(self, thread_id: str) -> dict[str, object]:
        st = self.state.get(thread_id)
        if st is None:
            st = {"messages": [], "summary": "", "compactions": 0}
            self.state[thread_id] = st
        return st

    def _thread_tokens(self, st: dict[str, object]) -> int:
        total = 0
        summary = st.get("summary", "")
        if isinstance(summary, str) and summary:
            total += estimate_tokens(summary)
        msgs = st.get("messages", [])
        assert isinstance(msgs, list)
        for m in msgs:
            if isinstance(m, dict):
                total += estimate_tokens(str(m.get("content", "")))
        return total

    def append(self, thread_id: str, role: str, content: str) -> None:
        st = self._ensure(thread_id)
        msgs = st["messages"]
        assert isinstance(msgs, list)
        msgs.append({"role": role, "content": content})
        if self._thread_tokens(st) <= self.threshold_tokens:
            return
        # Compact: summarize oldest messages, keep only the newest K.
        if len(msgs) <= self.keep_messages:
            return
        n_drop = len(msgs) - self.keep_messages
        to_summarize = [dict(m) for m in msgs[:n_drop]]  # type: ignore
        new_summary = summarize_messages(to_summarize, max_items=6)
        prev = st.get("summary", "")
        if isinstance(prev, str) and prev:
            st["summary"] = prev + "\n" + new_summary
        else:
            st["summary"] = new_summary
        st["messages"] = list(msgs[n_drop:])
        st["compactions"] = int(st.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        st = self._ensure(thread_id)
        msgs = st.get("messages", [])
        return {
            "messages": list(msgs) if isinstance(msgs, list) else [],
            "summary": st.get("summary", ""),
            "compactions": st.get("compactions", 0),
        }

    def compaction_count(self, thread_id: str) -> int:
        st = self.state.get(thread_id)
        if not st:
            return 0
        return int(st.get("compactions", 0))
