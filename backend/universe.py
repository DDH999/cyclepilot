# -*- coding: utf-8 -*-
"""QuantPilot · 票池定义 + 批量下载"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datafeed

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)

INDEXES = {
    "csi300":  ("000300", "沪深300"),
    "csi500":  ("000905", "中证500"),
    "csi1000": ("000852", "中证1000"),
}


def get_index_cons(key: str) -> pd.DataFrame:
    """从中证指数官方获取成分券。缓存 1 天。"""
    code, name = INDEXES[key]
    cp = CACHE / f"universe_{key}.csv"
    if cp.exists() and (time.time() - cp.stat().st_mtime) < 86400:
        return pd.read_csv(cp, dtype={"code": str})
    import akshare as ak
    raw = ak.index_stock_cons_csindex(symbol=code)
    df = pd.DataFrame({
        "code": raw["成分券代码"].astype(str).str.zfill(6),
        "name": raw["成分券名称"].astype(str).str.strip(),
        "exchange": raw["交易所"].astype(str),
        "index": key,
    }).drop_duplicates("code")
    df.to_csv(cp, index=False, encoding="utf-8")
    return df


def get_universe(which: str = "csi800") -> pd.DataFrame:
    """csi300 / csi500 / csi1000 / csi800 / csi1800"""
    keys = {"csi300": ["csi300"], "csi500": ["csi500"], "csi1000": ["csi1000"],
            "csi800": ["csi300", "csi500"],
            "csi1800": ["csi300", "csi500", "csi1000"]}[which]
    parts = []
    for k in keys:
        try:
            parts.append(get_index_cons(k))
        except Exception as e:
            print(f"[universe] {k} failed: {e}")
    df = pd.concat(parts, ignore_index=True).drop_duplicates("code")
    return df.reset_index(drop=True)


def bulk_download(codes: list[str], workers: int = 10, refresh: bool = False,
                  verbose: bool = True) -> dict:
    ok, fail, skipped = [], [], []
    t0 = time.time()

    def one(c):
        try:
            d = datafeed.load_ashare(c, refresh=refresh)
            return c, len(d)
        except Exception as e:
            return c, -1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, c): c for c in codes}
        for i, f in enumerate(as_completed(futs), 1):
            c, n = f.result()
            if n > 250:
                ok.append(c)
            else:
                fail.append(c)
            if verbose and i % 50 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(codes)}  ok={len(ok)} fail={len(fail)}  "
                      f"{i/el:.2f}/s  剩余{(len(codes)-i)/max(i/el,1e-9)/60:.1f}min",
                      flush=True)
    return {"ok": ok, "fail": fail, "elapsed": round(time.time() - t0, 1),
            "total": len(codes)}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "csi800"
    uni = get_universe(which)
    print(f"[universe] {which}: {len(uni)} 只")
    print(uni.head(5).to_string(index=False))
    r = bulk_download(uni["code"].tolist(), workers=10)
    print(f"\n[done] 成功 {len(r['ok'])} / 失败 {len(r['fail'])} / 用时 {r['elapsed']}s")
    (CACHE / f"downloaded_{which}.json").write_text(
        json.dumps({"index": which, "ok": r["ok"], "fail": r["fail"],
                    "elapsed": r["elapsed"], "ts": time.strftime("%Y-%m-%d %H:%M:%S")},
                   ensure_ascii=False, indent=1), encoding="utf-8")
