#!/usr/bin/env python3
"""调查「同语料训练下 bbpe 与 HF tokenizers 的压缩率差异」的根源。

背景：同一份 1GB 语料、同为 150256 词表时，两者压缩率差 0.14%
（bbpe 4.991 vs HF 4.998 B/token）。要看这个差从哪来，光比总量不够，
需要拆到词表与文本类别两层。

四步：
  1. 词表重叠：把 HF 的 ByteLevel token 还原成字节串，与 bbpe 的 vocab 比对
  2. 差异样本：各自独有的 token 例子（按长度排序，看是否集中在某类字符）
  3. 分类别 token 数：汉字 / 拉丁 / 数字 / 标点 / 空白 / 混合
  4. 汇总

用法：
  python analyze_gap.py --bbpe output/bbpe_1gb_150256.json \
      --hf output/hf_bpe_150256.json --input data/heldout.jsonl --max-mb 200
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_bbpe_vocab(path):
    """返回 {字节串: id}（bbpe 的 vocab 是 id -> list[int] 字节）。"""
    m = json.load(open(path, encoding="utf-8"))
    out = {}
    for k, v in m["vocab"].items():
        out[bytes(v)] = int(k)
    return out, m.get("pre_tokenizer_pattern", "")


def _bytes_to_unicode() -> dict:
    """GPT-2 的 byte↔char 映射（ByteLevel 内部用的就是这张表）。"""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def load_hf_vocab(path):
    """把 HF 的 ByteLevel token 还原成原始字节串。"""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(path)
    # ⚠️ 不能用 alphabet() 的下标当字节值：alphabet() 只是"字符集合"，
    # 顺序不保证等于 byte 值。必须用 GPT-2 的 bytes_to_unicode 反表。
    char2byte = {c: b for b, c in _bytes_to_unicode().items()}
    out = {}
    skipped = 0
    for token, tid in tok.get_vocab().items():
        try:
            out[bytes(char2byte[c] for c in token)] = tid
        except KeyError:  # <|endoftext|> 这类 special 不在 ByteLevel 字母表里
            skipped += 1
    return out, skipped, tok


def classify(bs: bytes) -> str:
    """按字节内容给 token 归类，用于看差异集中在哪种字符上。"""
    try:
        s = bs.decode("utf-8")
    except UnicodeDecodeError:
        return "非UTF8片段"

    def kind(ch):
        if "\u4e00" <= ch <= "\u9fff":
            return "han"
        if ch.isalpha():
            return "latin"
        if ch.isdigit():
            return "digit"
        if ch.isspace():
            return "space"
        return "punct"

    kinds = {kind(c) for c in s}
    if not kinds:
        return "空"
    return kinds.pop() if len(kinds) == 1 else "混合"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbpe", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--input", default=None, help="held-out jsonl，用于分类别统计")
    ap.add_argument("--max-mb", type=float, default=200.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    bb, pattern = load_bbpe_vocab(a.bbpe)
    hf, skipped, hf_tok = load_hf_vocab(a.hf)
    print(f"bbpe 词表 {len(bb)} 条 | HF 词表 {len(hf)} 条（跳过 special {skipped}）")
    print(f"bbpe 预分词正则：{pattern.strip().splitlines()[0] if pattern else '(未记录)'}")
    print(f"HF  预分词：{hf_tok.pre_tokenizer}")

    # ---------- 1. 词表重叠 ----------
    sb, sh = set(bb), set(hf)
    inter = sb & sh
    print("\n=== 1. 词表重叠（按字节串精确相等） ===")
    print(f"  交集 {len(inter)} | bbpe 独有 {len(sb - sh)} | HF 独有 {len(sh - sb)}")
    print(f"  Jaccard = {len(inter) / len(sb | sh):.4f}")
    print(f"  bbpe 侧覆盖率 = {len(inter) / len(sb):.4f} | HF 侧覆盖率 = {len(inter) / len(sh):.4f}")

    # ---------- 2. 差异样本 ----------
    print("\n=== 2. 独有 token 的样子（按字节长度降序，取最长 5 个） ===")
    for tag, only in (("bbpe 独有", sb - sh), ("HF 独有", sh - sb)):
        longest = sorted(only, key=len, reverse=True)[:5]
        shown = []
        for b in longest:
            try:
                shown.append(f"{b.decode('utf-8')!r}({len(b)}B)")
            except UnicodeDecodeError:
                shown.append(f"{b!r}")
        print(f"  {tag}: {' | '.join(shown)}")

    # 独有 token 的字符类别分布
    for tag, only in (("bbpe 独有", sb - sh), ("HF 独有", sh - sb)):
        cnt = collections.Counter(classify(b) for b in only)
        top = ", ".join(f"{k} {v}" for k, v in cnt.most_common(4))
        print(f"  {tag} 的字符类别分布: {top}")

    # ---------- 3. 分类别 token 数 ----------
    result = {
        "bbpe_vocab": len(bb), "hf_vocab": len(hf),
        "intersection": len(inter),
        "bbpe_only": len(sb - sh), "hf_only": len(sh - sb),
        "jaccard": round(len(inter) / len(sb | sh), 4),
    }

    if a.input:
        import re
        sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
        from bbpe_tokenizer import bbpe_rs

        rng = {k: [] for k in ("han", "latin", "digit", "punct", "space")}
        budget = a.max_mb * (1024 ** 2)
        used = 0
        with open(a.input, encoding="utf-8") as f:
            for line in f:
                try:
                    t = json.loads(line)["text"]
                except (json.JSONDecodeError, KeyError):
                    continue
                used += len(t.encode())
                for name, pat in (("han", r"\u4e00-\u9fff"), ("latin", r"A-Za-z"),
                                  ("digit", r"0-9"),
                                  ("punct", r"!-/:-@\[-`{-~，。！？；：、（）【】“”‘’—…"),
                                  ("space", r" \t\n")):
                    rng[name].append("".join(re.findall(f"[{pat}]", t)))
                if used >= budget:
                    break
        src_bytes = used

        print(f"\n=== 3. 分类别压缩（held-out 前 {src_bytes/1e6:.0f} MB 中抽取同类字符）===")
        print(f"{'类别':8s}{'字符数':>12s}{'bbpe tok':>12s}{'HF tok':>12s}"
              f"{'bbpe B/tok':>12s}{'HF B/tok':>10s}{'差':>8s}")
        bb_tok = bbpe_rs.BBPETokenizer(a.bbpe)

        per_cat = {}
        for name, chunks in rng.items():
            text = "\n".join(chunks)
            nb = len(text.encode())
            if nb == 0:
                continue
            out_bb = bb_tok.encode_ordinary(text)
            n_bb = len(getattr(out_bb, "ids", out_bb))
            n_hf = len(hf_tok.encode(text, add_special_tokens=False).ids)
            bpt_bb = nb / n_bb if n_bb else 0
            bpt_hf = nb / n_hf if n_hf else 0
            delta = (n_hf - n_bb) / n_bb * 100 if n_bb else 0
            per_cat[name] = {"bytes": nb, "bbpe_tokens": n_bb, "hf_tokens": n_hf,
                             "bbpe_bpt": round(bpt_bb, 3), "hf_bpt": round(bpt_hf, 3),
                             "hf_vs_bbpe_pct": round(delta, 2)}
            print(f"{name:8s}{nb:12,d}{n_bb:12,d}{n_hf:12,d}"
                  f"{bpt_bb:12.3f}{bpt_hf:10.3f}{delta:+7.2f}%")
        result["per_category"] = per_cat

    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(result, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\n报告: {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
