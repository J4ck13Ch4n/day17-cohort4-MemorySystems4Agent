from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import estimate_tokens


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of conversations in {path}")
    return data


def _norm(text: str) -> str:
    return (text or "").lower()


def recall_points(answer: str, expected: list[str]) -> float:
    """Fraction of expected facts present in the answer (0..1)."""
    if not expected:
        return 1.0
    ans = _norm(answer)
    hits = sum(1 for e in expected if _norm(e) in ans)
    return hits / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Lightweight offline quality score (0..1).

    Combines recall coverage with structure/length signals so a fact-correct,
    concise, well-structured answer scores higher than a generic one.
    """
    if not answer or not answer.strip():
        return 0.0
    recall = recall_points(answer, expected)
    score = 0.25 + 0.55 * recall
    low = _norm(answer)
    # Structure bonus: bullets / short structured answers read better.
    if any(m in answer for m in ("-", "•", "1.", "2.", ":")):
        score += 0.05
    # Conciseness bonus: prefer 20..600 chars; penalize empty or rambling.
    n = len(answer.strip())
    if 20 <= n <= 600:
        score += 0.05
    elif n > 1500:
        score -= 0.05
    # Vietnamese politeness/structure cue.
    if "bạn" in low:
        score += 0.02
    return max(0.0, min(1.0, round(score, 3)))


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    1. Feed all turns to the agent (per-conversation thread).
    2. Track agent tokens only + prompt tokens processed.
    3. Ask recall questions in a FRESH thread (cross-session recall).
    4. Average recall + quality; record memory growth + compactions.
    """
    total_agent_tokens = 0
    total_prompt_tokens = 0
    total_compactions = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    # Memory growth: measure User.md bytes for advanced agents.
    users = {c.get("user_id", "default") for c in conversations}
    size_before: dict[str, int] = {}
    for u in users:
        if hasattr(agent, "memory_file_size"):
            try:
                size_before[u] = agent.memory_file_size(u)
            except Exception:
                size_before[u] = 0

    for conv in conversations:
        user_id = conv.get("user_id", "default")
        conv_id = conv.get("id", "conv")
        thread_id = f"{agent_name}::{conv_id}"
        for turn in conv.get("turns", []):
            res = agent.reply(user_id, thread_id, turn)
            total_agent_tokens += int(res.get("agent_tokens", 0))
            total_prompt_tokens += int(res.get("prompt_tokens", 0))
        try:
            total_compactions += int(agent.compaction_count(thread_id))
        except Exception:
            pass
        # Recall in a fresh thread -> cross-session memory only.
        recall_thread = f"{agent_name}::{conv_id}::recall"
        for q in conv.get("recall_questions", []):
            q_text = q.get("question", "")
            expected = q.get("expected_contains", [])
            res = agent.reply(user_id, recall_thread, q_text)
            total_agent_tokens += int(res.get("agent_tokens", 0))
            total_prompt_tokens += int(res.get("prompt_tokens", 0))
            ans = res.get("text", "")
            recall_scores.append(recall_points(ans, expected))
            quality_scores.append(heuristic_quality(ans, expected))
        try:
            total_compactions += int(agent.compaction_count(recall_thread))
        except Exception:
            pass

    memory_growth = 0
    for u in users:
        if hasattr(agent, "memory_file_size"):
            try:
                memory_growth += max(0, agent.memory_file_size(u) - size_before.get(u, 0))
            except Exception:
                pass

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=round(avg_recall, 3),
        response_quality=round(avg_quality, 3),
        memory_growth_bytes=memory_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = ["Agent", "Agent tokens only", "Prompt tokens processed",
               "Cross-session recall", "Response quality", "Memory growth (bytes)", "Compactions"]
    data = [
        [r.agent_name, r.agent_tokens_only, r.prompt_tokens_processed,
         f"{r.recall_score:.3f}", f"{r.response_quality:.3f}",
         r.memory_growth_bytes, r.compactions]
        for r in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(data, headers=headers, tablefmt="github")
    except Exception:
        lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
        for row in data:
            lines.append(" | ".join(str(x) for x in row))
        return "\n".join(lines)


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    std_path = config.data_dir / "conversations.json"
    stress_path = config.data_dir / "advanced_long_context.json"
    std_convs = load_conversations(std_path)
    stress_convs = load_conversations(stress_path)

    # Fresh agents per suite so state does not leak between tables.
    baseline_std = BaselineAgent(config=config, force_offline=False)
    advanced_std = AdvancedAgent(config=config, force_offline=False)
    rows_std = [
        run_agent_benchmark("Baseline", baseline_std, std_convs, config),
        run_agent_benchmark("Advanced", advanced_std, std_convs, config),
    ]
    print("### Standard Benchmark (data/conversations.json)")
    print(format_rows(rows_std))
    print()

    baseline_stress = BaselineAgent(config=config, force_offline=False)
    advanced_stress = AdvancedAgent(config=config, force_offline=False)
    rows_stress = [
        run_agent_benchmark("Baseline", baseline_stress, stress_convs, config),
        run_agent_benchmark("Advanced", advanced_stress, stress_convs, config),
    ]
    print("### Long-Context Stress Benchmark (data/advanced_long_context.json)")
    print(format_rows(rows_stress))
    print()
    print("Ghi chú:")
    print("- Advanced có User.md nên recall cao hơn hẳn Baseline (vốn quên qua thread mới).")
    print("- Hội thoại ngắn: Advanced có thể tốn hơn vì luôn mang theo User.md.")
    print("- Hội thoại dài: compact memory giữ prompt tokens của Advanced bounded,")
    print("  trong khi Baseline mang nguyên lịch sử nên prompt tokens đội lên mạnh.")


if __name__ == "__main__":
    main()
