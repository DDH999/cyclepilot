# -*- coding: utf-8 -*-
"""诊断：子集等权 1.26%（只读）"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import backtest_strict as BS
import universe_pit as UP

START, END = "2019-01-02", "2021-12-31"


def ann(series):
    s = pd.Series(series).dropna()
    if len(s) < 2:
        return float("nan")
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1)


def main():
    basic = UP.load_basic()
    cal = UP.trading_days(START, END)
    idx = UP.build_index(START, END, basic=basic, cal=cal)["index"]
    delisted = basic[basic["out_date"].notna()]
    pool_any = set().union(*[idx[str(pd.Timestamp(d).date())] for d in cal])
    dl_in = [c for c in delisted["code"] if c in pool_any
             and pd.Timestamp(basic.loc[basic["code"] == c, "out_date"].iloc[0])
             <= pd.Timestamp(END)]
    surv = [c for c in basic["code"] if c not in set(delisted["code"])]
    rng = np.random.default_rng(42)
    surv_s = list(rng.choice(sorted(surv), size=300, replace=False))
    codes = sorted(set(dl_in) | set(surv_s))
    panel = BS.fetch_panel(codes, "2018-06-01", END, with_flags=True)["panel"]
    print(f"面板 {len(panel)} 只（请求 {len(codes)}）")

    CL = pd.DataFrame({c: d.set_index("date")["close"]
                       for c, d in panel.items()}).sort_index().loc[START:END]
    R = CL.pct_change(fill_method=None)

    print("\n=== a) 复权方式 ===")
    try:
        import baostock as bs
        import baostock_source as BSrc
        BSrc._login()
        q = BS.fetch_panel(["600519"], START, END, with_flags=False)["panel"]
        rs = bs.query_history_k_data_plus(
            "sh.600519", "date,close", start_date=START, end_date=END,
            frequency="d", adjustflag="3")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        raw = pd.DataFrame(rows, columns=["date", "close"])
        raw["date"] = pd.to_datetime(raw["date"])
        raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
        m = q["600519"].merge(raw, on="date", suffixes=("_q", "_r"))
        m["rq"] = m["close_q"].pct_change()
        m["rr"] = m["close_r"].pct_change()
        d = (m["rq"] - m["rr"]).abs()
        print(f"  600519 qfq vs 不复权：单日收益最大差 {d.max()*100:.2f}pp")
        print(f"  不复权最大单日跌 {m['rr'].min()*100:.2f}%（除权跳空）")
        print(f"  qfq    最大单日跌 {m['rq'].min()*100:.2f}%")
    except Exception as ex:
        print(f"  复权对照失败（不影响后续）：{str(ex)[:60]}")

    print("\n=== b) 收益聚合 ===")
    ew = R.mean(axis=1, skipna=True).add(1).cumprod()
    tots = pd.Series({c: (CL[c].dropna().iloc[-1] / CL[c].dropna().iloc[0] - 1)
                      for c in CL.columns if CL[c].notna().sum() >= 200})
    print(f"  现用口径（横截面算术 + 时序几何）年化 {ann(ew)*100:.2f}%")
    print(f"  个股全期总收益 均值 {tots.mean()*100:.2f}%  中位 {tots.median()*100:.2f}%")
    print(f"  分布 p5 {tots.quantile(.05)*100:.1f}% / p25 {tots.quantile(.25)*100:.1f}%"
          f" / p75 {tots.quantile(.75)*100:.1f}% / p95 {tots.quantile(.95)*100:.1f}%")

    print("\n=== c) 异常值 ===")
    st = R.stack().reset_index()
    st.columns = ["date", "code", "ret"]
    st = st.dropna()
    print(st.nsmallest(10, "ret").to_string(
        index=False, formatters={"ret": lambda x: f"{x*100:.2f}%"}))
    print(f"  单日 <-21%：{int((st['ret'] < -0.21).sum())} 条；"
          f"单日 >+21%：{int((st['ret'] > 0.21).sum())} 条")

    print("\n=== d) 停牌处理 ===")
    nan_ratio = float(R.isna().sum().sum()) / (R.shape[0] * R.shape[1])
    print(f"  收益矩阵 NaN 占比 {nan_ratio*100:.2f}%，mean(skipna=True) → 跳过不算 0")

    print("\n=== e) 样本构成（关键）===")
    dl_set = set(dl_in) & set(panel)
    print(f"  子集 {len(panel)} 只，退市股 {len(dl_set)} 只 = "
          f"{len(dl_set)/len(panel)*100:.1f}%（真实市场约 0.5~1%）")
    surv_only = [c for c in CL.columns if c not in dl_set]
    ew_s = R[surv_only].mean(axis=1, skipna=True).add(1).cumprod()
    print(f"  剔除退市股后等权年化 {ann(ew_s)*100:.2f}%"
          f"（含退市股 {ann(ew)*100:.2f}%）")
    print(f"  → 退市股拖低 {(ann(ew_s)-ann(ew))*100:.2f}pp")
    dl_tots = tots[[c for c in dl_set if c in tots.index]]
    if len(dl_tots):
        print(f"  退市股全期总收益中位 {dl_tots.median()*100:.1f}%，"
              f"最差 {dl_tots.min()*100:.1f}%")


if __name__ == "__main__":
    main()
