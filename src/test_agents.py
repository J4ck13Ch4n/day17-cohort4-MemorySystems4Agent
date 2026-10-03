from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests."""
    base = load_config(Path(__file__).resolve().parent.parent)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=base.base_dir,
        data_dir=base.data_dir,
        state_dir=state_dir,
        # Small threshold so compaction triggers quickly in tests.
        compact_threshold_tokens=200,
        compact_keep_messages=4,
        model=base.model,
        judge_model=base.judge_model,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, updated, and edited."""
    cfg = make_config(tmp_path)
    store = UserProfileStore(cfg.state_dir / "profiles")
    user = "test_user"

    p = store.write_text(user, "# Hồ sơ người dùng\n\n- Tên: DũngCT\n")
    assert p.exists()
    assert "DũngCT" in store.read_text(user)

    store.upsert_fact(user, "Tên", "DũngCT Stress")
    assert "DũngCT Stress" in store.read_text(user)

    changed = store.edit_text(user, "DũngCT Stress", "DũngCT")
    assert changed is True
    assert "DũngCT" in store.read_text(user)
    assert store.file_size(user) > 0
    assert store.edit_text(user, "không-tồn-tại-xyz", "y") is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    mgr = CompactMemoryManager(threshold_tokens=200, keep_messages=4)
    tid = "t1"
    for i in range(20):
        mgr.append(tid, "user", f"Tin nhắn dài số {i} " + ("nội dung benchmark memory " * 10))
    assert mgr.compaction_count(tid) >= 1
    ctx = mgr.context(tid)
    assert ctx["summary"]
    assert len(ctx["messages"]) <= 4


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    cfg = make_config(tmp_path)
    adv = AdvancedAgent(config=cfg, force_offline=False)
    base = BaselineAgent(config=cfg, force_offline=False)

    adv.reply("dungct", "thread-1", "Mình tên là DũngCT, ở Huế và làm MLOps engineer.")
    adv.reply("dungct", "thread-1", "Đồ uống yêu thích là cà phê sữa đá, trả lời ngắn gọn nhé.")
    base.reply("dungct", "thread-1", "Mình tên là DũngCT, ở Huế và làm MLOps engineer.")
    base.reply("dungct", "thread-1", "Đồ uống yêu thích là cà phê sữa đá, trả lời ngắn gọn nhé.")

    # Fresh thread = new session: only persistent memory survives.
    a = adv.reply("dungct", "thread-2", "Mình tên gì và đồ uống yêu thích là gì?")
    b = base.reply("dungct", "thread-2", "Mình tên gì và đồ uống yêu thích là gì?")
    assert "DũngCT" in a["text"]
    assert "cà phê sữa đá" in a["text"]
    assert "DũngCT" not in b["text"]


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    cfg = make_config(tmp_path)
    adv = AdvancedAgent(config=cfg, force_offline=False)
    base = BaselineAgent(config=cfg, force_offline=False)

    long_turn = "Tin tức benchmark dài " + ("Artemis X-59 WMO energy " * 30)
    for i in range(15):
        adv.reply("u1", "long", f"{long_turn} lượt {i}")
        base.reply("u1", "long", f"{long_turn} lượt {i}")

    assert adv.compaction_count("long") >= 1
    assert base.compaction_count("long") == 0
    assert adv.prompt_token_usage("long") < base.prompt_token_usage("long")
    # estimate_tokens sanity
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world test") > 0
