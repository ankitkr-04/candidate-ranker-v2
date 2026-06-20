"""One-shot repair of Chinese code-switching in the SLM `evidence` column.

The SLM (Qwen3) occasionally emitted Chinese for an English word inside the free-text
evidence span. That field is display-only -- the scorer never reads it -- so the fix is a
pure data edit, no model re-run: translate the frequent, cleanly-mappable phrases to
English, then cut the span at the first leftover CJK/Hangul character (rare mid-word
fragments with no clean equivalent). After the pass no leaked character remains.

This is a standalone utility, NOT a precompute stage. Run it once after copying a fresh
features.parquet from the GPU box; the ranker then reads clean evidence with no code in
its hot path:

    python -m src.features.repair_evidence --parquet artifacts/100k/features.parquet
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import polars as pl

# CJK ideographs (+ Ext-A), CJK symbols, compatibility ideographs, fullwidth forms, and
# Hangul -- every script the SLM leaked into the evidence spans.
_LEAK = re.compile(r"[　-〿㐀-䶿一-鿿가-힣豈-﫿＀-￯]")

# Frequent, cleanly-translatable phrases the SLM emitted in Chinese instead of English.
# Applied longest-key-first so compounds win over their parts (性能营销 before 营销).
_RESTORE = {
    "隐式反馈库": "implicit feedback",
    "性能营销": "performance marketing",
    "性能优化": "performance optimization",
    "我们的": "our",
    "管理者": "manager",
    "科学家": "scientist",
    "自动化": "automation",
    "机械": "mechanical",
    "职责": "responsibilities",
    "提及": "mentions",
    "模型": "model",
    "支持": "support",
    "谈判": "negotiation",
    "会计": "accounting",
    "涉及": "involving",
    "排名": "ranking",
    "服务": "service",
    "阶段": "stage",
    "控制": "control",
    "品牌": "brand",
    "工程": "engineering",
    "分析": "analysis",
    "检索": "retrieval",
    "开发": "development",
    "写作": "writing",
    "部署": "deployment",
    "文档": "documentation",
    "成本": "cost",
    "内容": "content",
    "设计": "design",
    "销售": "sales",
    "遗留": "legacy",
    "聚焦": "focused on",
    "编辑": "editing",
    "职能": "function",
    "物流": "logistics",
    "咨询": "consulting",
    "建设": "building",
    "资产": "assets",
    "交付": "delivery",
    "运维": "operations",
    "架构": "architecture",
    "测试": "testing",
    "前端": "frontend",
    "岗位": "role",
    "系统": "system",
    "安卓": "Android",
    "营销": "marketing",
    "学习": "learning",
    "质量": "quality",
    "我的": "my",
    "云": "cloud",
    "邻": "neighbor",
    "嵌": "embedding",
    "栈": "stack",
    "和": "and",
    "或": "or",
    "到": "to",
}
_RESTORE_ORDERED = sorted(_RESTORE.items(), key=lambda kv: len(kv[0]), reverse=True)

# Function words left dangling when a span is truncated at a leak.
_TRAILING_WORDS = ("and", "or", "to", "in", "of", "for", "with", "the", "an", "a", "my", "our")
_TRAILING_FRAGMENT = re.compile(r" [a-z]{1,2}$")


def _trim_tail(text: str) -> str:
    """Drop punctuation, dangling connectors, and 1-2 char cut-off fragments at the tail."""
    previous = None
    while text != previous:
        previous = text
        text = text.rstrip(" \t\r\n,;:&/-—–(·")
        lowered = text.lower()
        if any(lowered.endswith(" " + word) for word in _TRAILING_WORDS):
            word = next(w for w in _TRAILING_WORDS if lowered.endswith(" " + w))
            text = text[: -(len(word) + 1)]
            continue
        text = _TRAILING_FRAGMENT.sub("", text)
    return text


def _normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text)               # collapse whitespace (incl. padding)
    text = re.sub(r"\s+([,.;:%)\]!?])", r"\1", text)  # no space before closing punctuation
    text = re.sub(r"([(\[])\s+", r"\1", text)      # no space after opening bracket
    return text


def repair_evidence(text: str | None) -> str | None:
    if text is None or not _LEAK.search(text):
        return text
    # Pad each substitution with spaces: the model glued the Chinese to adjacent English
    # (e.g. "in隐式反馈库"), so a bare replace would fuse words. _normalize_spacing then
    # collapses the padding and re-tightens punctuation.
    for chinese, english in _RESTORE_ORDERED:
        text = text.replace(chinese, f" {english} ")
    leak = _LEAK.search(text)
    if leak:  # rare fragment with no clean equivalent: cut at the leak
        text = text[: leak.start()]
    return _trim_tail(_normalize_spacing(text)).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Replace SLM Chinese leakage in the evidence column with English (no SLM re-run).")
    ap.add_argument("--parquet", type=Path, default=Path("artifacts/100k/features.parquet"))
    ap.add_argument("--no-backup", action="store_true", help="Skip writing the .bak copy.")
    ap.add_argument("--dry-run", action="store_true", help="Report counts and samples without writing.")
    args = ap.parse_args()

    df = pl.read_parquet(args.parquet)
    original = df["evidence"].to_list()
    repaired = [repair_evidence(v) for v in original]

    leaked = [(o, r) for o, r in zip(original, repaired) if o and _LEAK.search(o)]
    after = sum(1 for r in repaired if r and _LEAK.search(r))
    print(f"rows={df.height}  evidence non-null={sum(v is not None for v in original)}")
    print(f"spans with leakage: {len(leaked)} -> {after} still leaking after repair\n")
    for o, r in leaked[:8]:
        print(f"  - {o!r}\n  + {r!r}\n")

    if args.dry_run:
        print("dry-run: parquet not modified.")
        return

    if not args.no_backup:
        backup = args.parquet.with_name(args.parquet.name + ".bak")
        shutil.copy2(args.parquet, backup)
        print(f"backup -> {backup}")

    df.with_columns(pl.Series("evidence", repaired, dtype=pl.Utf8)).write_parquet(args.parquet)
    print(f"wrote repaired evidence to {args.parquet}")


if __name__ == "__main__":
    main()
