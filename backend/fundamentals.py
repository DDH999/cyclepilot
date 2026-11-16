# -*- coding: utf-8 -*-
"""
QuantPilot · 基本面数据层（时点对齐）
=====================================================
数据源：akshare stock_yjbb_em(季度) —— 东财业绩报表，含「最新公告日期」
关键：所有因子在调仓日 T 只能使用 announce_date <= T 的报告，杜绝未来函数。
"""
from __future__ import annotations
import sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)
FUND_DIR = CACHE / "fund"
FUND_DIR.mkdir(exist_ok=True)

# 2015Q1 ~ 2026Q3（回测 2016 起，多留 1 年做同比/环比基准）
QUARTERS = [f"{y}{q}" for y in range(2015, 2027) for q in ("0331", "0630", "0930", "1231")]
QUARTERS = [q for q in QUARTERS if q <= "20260930"]

RENAME = {
    "股票代码": "code", "每股收益": "eps", "营业总收入-营业总收入": "revenue",
    "营业总收入-同比增长": "rev_yoy", "营业总收入-季度环比增长": "rev_qoq",
    "净利润-净利润": "np", "净利润-同比增长": "np_yoy", "净利润-季度环比增长": "np_qoq",
    "每股净资产": "bps", "净资产收益率": "roe", "每股经营现金流量": "ocfps",
    "销售毛利率": "gross_margin", "所处行业": "ind_em", "最新公告日期": "announce",
}


def fetch_quarter(q: str) -> tuple[str, int]:
    cp = FUND_DIR / f"yjbb_{q}.csv"
    if cp.exists():
        try:
            return q, len(pd.read_csv(cp))
        except Exception:
            pass
    import akshare as ak
    for attempt in range(3):
        try:
            raw = ak.stock_yjbb_em(date=q)
            if raw is None or not len(raw):
                return q, 0
            df = raw.rename(columns={k: v for k, v in RENAME.items() if k in raw.columns})
            keep = [c for c in RENAME.values() if c in df.columns]
            df = df[keep].copy()
            df["code"] = df["code"].astype(str).str.zfill(6)
            df["report_period"] = pd.to_datetime(q, format="%Y%m%d")
            if "announce" in df.columns:
                df["announce"] = pd.to_datetime(df["announce"], errors="coerce")
            for c in keep:
                if c not in ("code", "announce", "ind_em"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.drop_duplicates("code")
            df.to_csv(cp, index=False, encoding="utf-8")
            return q, len(df)
        except Exception:
            time.sleep(3 + attempt * 4)
    return q, -1


def download_all(quarters: list[str] | None = None, workers: int = 4) -> dict:
    qs = quarters or QUARTERS
    todo = [q for q in qs if not (FUND_DIR / f"yjbb_{q}.csv").exists()]
    print(f"[fund] 总 {len(qs)} 季，待下载 {len(todo)}")
    ok, fail = [], []
    t0 = time.time()
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fetch_quarter, q): q for q in todo}
            for i, f in enumerate(as_completed(futs), 1):
                q, n = f.result()
                (ok if n > 0 else fail).append(q)
                if i % 5 == 0:
                    print(f"  {i}/{len(todo)} ok={len(ok)} fail={len(fail)} "
                          f"{(time.time()-t0)/i:.1f}s/季", flush=True)
    return {"ok": ok, "fail": fail, "elapsed": round(time.time() - t0, 1)}


def load_all() -> pd.DataFrame:
    files = sorted(FUND_DIR.glob("yjbb_*.csv"))
    if not files:
        return pd.DataFrame()
    parts = []
    for f in files:
        try:
            d = pd.read_csv(f, dtype={"code": str}, parse_dates=["report_period", "announce"])
            parts.append(d)
        except Exception:
            continue
    df = pd.concat(parts, ignore_index=True)
    # 公告日缺失的用报告期 + 60 天兜底（保守，宁可晚用不可早用）
    miss = df["announce"].isna()
    df.loc[miss, "announce"] = df.loc[miss, "report_period"] + pd.Timedelta(days=60)
    return df.sort_values(["code", "report_period"]).reset_index(drop=True)


# ───────────────────────────── 因子构造
def _ttm_eps(df: pd.DataFrame) -> pd.DataFrame:
    """向量化 TTM 每股收益：本期累计 + 上年年报 - 上年同期累计"""
    p = df.pivot_table(index="code", columns="report_period", values="eps", aggfunc="last")
    cols = list(p.columns)
    out = pd.DataFrame(index=p.index, columns=cols, dtype=float)
    for rp in cols:
        y, m = rp.year, rp.month
        if m == 12:
            out[rp] = p[rp]
            continue
        q = m // 3
        try:
            ann = p[pd.Timestamp(year=y - 1, month=12, day=31)]
            same = p[pd.Timestamp(year=y - 1, month=q * 3, day=1) +
                     pd.offsets.MonthEnd(0)]
        except KeyError:
            out[rp] = np.nan
            continue
        out[rp] = p[rp] + ann - same
    long = out.reset_index().melt(id_vars="code", var_name="report_period", value_name="eps_ttm")
    long["report_period"] = pd.to_datetime(long["report_period"])
    return long


def build_factor_panel() -> pd.DataFrame:
    """
    为每只股票每个报告期计算基本面因子（只用到当期及历史报告）。
    """
    df = load_all()
    if df.empty:
        return df
    df = df.sort_values(["code", "report_period"]).copy()
    g = df.groupby("code", group_keys=False)

    df["np_yoy_prev"] = g["np_yoy"].shift(1)
    df["gm_4q"] = g["gross_margin"].shift(4)
    df["np_yoy_4q"] = g["np_yoy"].shift(4)

    # TTM 每股收益（向量化）
    ttm = _ttm_eps(df)
    df = df.merge(ttm, on=["code", "report_period"], how="left")

    # ── 因子
    df["f_roe"] = df["roe"]
    df["f_np_yoy"] = df["np_yoy"]
    df["f_rev_yoy"] = df["rev_yoy"]
    df["f_gm"] = df["gross_margin"]
    df["f_gm_chg"] = df["gross_margin"] - df["gm_4q"]
    df["f_accel"] = df["np_yoy"] - df["np_yoy_4q"]
    df["f_cash_q"] = np.where(df["eps"].abs() > 1e-6,
                              df["ocfps"] / df["eps"].abs(), np.nan)
    df["f_cash_q"] = df["f_cash_q"].clip(-5, 5)
    return df


FACTORS = {
    "f_roe": "ROE（净资产收益率）",
    "f_np_yoy": "净利润同比增速",
    "f_rev_yoy": "营收同比增速",
    "f_gm": "销售毛利率",
    "f_gm_chg": "毛利率同比变化",
    "f_accel": "盈利加速（同比增速二阶）",
    "f_cash_q": "现金流质量（OCF/EPS）",
}


if __name__ == "__main__":
    r = download_all(workers=4)
    print(f"[fund] 完成 ok={len(r['ok'])} fail={len(r['fail'])} 用时 {r['elapsed']}s")
    fp = build_factor_panel()
    print(f"[fund] 因子面板 {len(fp):,} 行, {fp['code'].nunique()} 只")
    if len(fp):
        for c, name in FACTORS.items():
            if c in fp.columns:
                print(f"   {name:26} 覆盖 {fp[c].notna().mean():.1%}  中位 {fp[c].median():.3f}")
