# -*- coding: utf-8 -*-
"""
QuantPilot · 退市股数据（修复幸存者偏差）
=========================================
只用存活股票回测 = 只交易了活下来的公司 = 收益系统性虚高。
本模块把沪/深已退市公司拉进票池，让策略有机会买到"后来死掉"的股票。
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
import datafeed

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
DELIST_CSV = CACHE / "delisted.csv"


def get_delisted_list(refresh: bool = False) -> pd.DataFrame:
    """沪 + 深 已退市公司清单"""
    if DELIST_CSV.exists() and not refresh:
        return pd.read_csv(DELIST_CSV, dtype={"code": str})
    import akshare as ak
    rows = []
    try:
        sh = ak.stock_info_sh_delist()
        for _, r in sh.iterrows():
            rows.append({"code": str(r["公司代码"]).zfill(6), "name": r["公司简称"],
                         "list_date": r.get("上市日期"), "delist_date": r.get("暂停上市日期"),
                         "market": "SH"})
    except Exception as e:
        print(f"  沪退市清单失败: {e}")
    try:
        sz = ak.stock_info_sz_delist(symbol="终止上市公司")
        for _, r in sz.iterrows():
            rows.append({"code": str(r["证券代码"]).zfill(6), "name": r["证券简称"],
                         "list_date": r.get("上市日期"), "delist_date": r.get("终止上市日期"),
                         "market": "SZ"})
    except Exception as e:
        print(f"  深退市清单失败: {e}")
    df = pd.DataFrame(rows).drop_duplicates("code").reset_index(drop=True)
    df.to_csv(DELIST_CSV, index=False, encoding="utf-8")
    return df


def _download_one(code: str, start: str = "20100101") -> tuple[str, int]:
    """优先腾讯（退市股仍可取，单次请求，快）"""
    import akshare as ak
    sk = ("sh" if code[0] in "65" else "sz") + code
    try:
        raw = ak.stock_zh_a_hist_tx(symbol=sk, adjust="qfq")
        if raw is not None and len(raw) >= 60:
            df = datafeed._norm(raw)
            datafeed._cpath(f"cn_{code}")  # 确保目录存在
            df.to_csv(datafeed._cpath(f"cn_{code}"), index=False, encoding="utf-8")
            return code, len(df)
    except Exception:
        pass
    return code, -1


def download_delisted(codes: list[str], workers: int = 8) -> dict:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    ok, fail = [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_download_one, c): c for c in codes}
        for i, f in enumerate(as_completed(futs), 1):
            c, n = f.result()
            (ok if n > 0 else fail).append(c)
            if i % 50 == 0:
                print(f"  {i}/{len(codes)} ok={len(ok)} fail={len(fail)} "
                      f"{i/max(time.time()-t0,1e-9):.2f}/s", flush=True)
    return {"ok": ok, "fail": fail, "elapsed": round(time.time() - t0, 1)}


def load_delisted_panel(start: str = "2016-01-01") -> tuple[dict, pd.DataFrame]:
    """返回 (panel, 退市信息表)。panel 结构与其他票一致。"""
    dl = get_delisted_list()
    dl["delist_date"] = pd.to_datetime(dl["delist_date"], errors="coerce")
    dl = dl[dl["delist_date"].notna()]
    codes = [c for c in dl["code"] if not datafeed._cpath(f"cn_{c}").exists()]
    if codes:
        print(f"[delisted] 需下载 {len(codes)} 只")
        download_delisted(codes)
    panel = {}
    for c in dl["code"]:
        cp = datafeed._cpath(f"cn_{c}")
        if not cp.exists():
            continue
        try:
            d = pd.read_csv(cp, parse_dates=["date"])
            d = d[d["date"] >= pd.Timestamp(start)]
            if len(d) > 120:
                panel[c] = d.reset_index(drop=True)
        except Exception:
            continue
    return panel, dl


if __name__ == "__main__":
    dl = get_delisted_list(refresh=True)
    print(f"[delisted] 总量 {len(dl)}")
    dl["delist_date"] = pd.to_datetime(dl["delist_date"], errors="coerce")
    recent = dl[dl["delist_date"] >= "2016-01-01"]
    print(f"[delisted] 2016 后退市: {len(recent)}")
    todo = [c for c in recent["code"] if not datafeed._cpath(f"cn_{c}").exists()]
    print(f"[delisted] 待下载 {len(todo)}")
    r = download_delisted(todo, workers=8)
    print(f"[done] ok={len(r['ok'])} fail={len(r['fail'])} 用时 {r['elapsed']}s")
