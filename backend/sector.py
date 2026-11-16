# -*- coding: utf-8 -*-
"""QuantPilot · 行业分类（申万口径，用于行业中性化）"""
from __future__ import annotations
import sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)
IND_CSV = CACHE / "industries.csv"


def fetch_one(code: str) -> tuple[str, str]:
    import akshare as ak
    try:
        df = ak.stock_individual_info_em(symbol=code)
        d = dict(zip(df["item"], df["value"]))
        return code, str(d.get("行业", "")).strip() or "未知"
    except Exception:
        return code, "未知"


def build_industries(codes: list[str], workers: int = 10,
                     verbose: bool = True) -> pd.DataFrame:
    old = pd.read_csv(IND_CSV, dtype={"code": str}) if IND_CSV.exists() else pd.DataFrame(columns=["code", "industry"])
    known = dict(zip(old["code"], old["industry"])) if len(old) else {}
    todo = [c for c in codes if c not in known or known[c] == "未知"]
    if verbose:
        print(f"[sector] 已有 {len(known)} 条，需补 {len(todo)} 条")
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, c): c for c in todo}
        for f in as_completed(futs):
            c, ind = f.result()
            known[c] = ind
            done += 1
            if verbose and done % 100 == 0:
                print(f"  {done}/{len(todo)}  {done/max(time.time()-t0,1e-9):.2f}/s", flush=True)
    df = pd.DataFrame({"code": list(known), "industry": list(known.values())})
    df.to_csv(IND_CSV, index=False, encoding="utf-8")
    return df


def load_industries() -> dict[str, str]:
    if not IND_CSV.exists():
        return {}
    df = pd.read_csv(IND_CSV, dtype={"code": str})
    return dict(zip(df["code"], df["industry"]))


if __name__ == "__main__":
    import re
    codes = []
    for f in CACHE.glob("*.csv"):
        m = re.search(r"_cn_(\d{6})$", f.stem)
        if m:
            codes.append(m.group(1))
    codes = sorted(set(codes))
    print(f"[sector] 目标 {len(codes)} 只")
    df = build_industries(codes, workers=10)
    known = df[df["industry"] != "未知"]
    print(f"[sector] 完成: {len(known)}/{len(df)} 有行业")
    print(known["industry"].value_counts().head(15).to_string())
