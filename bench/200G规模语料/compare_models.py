#!/usr/bin/env python3
"""比较两个 bbpe 模型：词表/合并必须逐条一致，meta 的易变字段单独列出。"""
import hashlib
import json
import sys

VOLATILE_META = {"created_at", "git_commit", "bbpe_rs_version", "sampling", "corpus"}


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def canon(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def h(obj):
    return hashlib.sha256(canon(obj).encode()).hexdigest()[:16]


a, b = load(sys.argv[1]), load(sys.argv[2])
print(f"A: {sys.argv[1]}")
print(f"B: {sys.argv[2]}")

for field in ("format", "format_version", "pre_tokenizer_pattern"):
    same = a.get(field) == b.get(field)
    print(f"  {field:22s} {'一致' if same else '❌ 不同'}")

for field in ("vocab", "merges"):
    va, vb = a.get(field, {}), b.get(field, {})
    same = va == vb
    detail = ""
    if not same:
        ka, kb = set(va), set(vb)
        diff_keys = len(ka ^ kb)
        diff_val = sum(1 for k in ka & kb if va[k] != vb[k])
        detail = f" | 键差异 {diff_keys} | 值差异 {diff_val}"
    print(f"  {field:22s} {'一致' if same else '❌ 不同'} | {len(va)} 项 | "
          f"sha256 {h(va)}/{h(vb)}{detail}")

print("  meta 对比：")
ma, mb = a.get("meta", {}), b.get("meta", {})
for k in sorted(set(ma) | set(mb)):
    if k in VOLATILE_META:
        continue
    same = ma.get(k) == mb.get(k)
    print(f"    {k:20s} {'一致' if same else '❌ 不同'}")

same_all = (a.get("vocab") == b.get("vocab") and a.get("merges") == b.get("merges")
            and a.get("pre_tokenizer_pattern") == b.get("pre_tokenizer_pattern"))
print("\n结论:", "✅ 词表与 merges 逐条一致" if same_all else "❌ 存在实质差异")
sys.exit(0 if same_all else 1)
