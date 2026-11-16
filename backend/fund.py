# -*- coding: utf-8 -*-
"""
基金数据与指标（M19）
====================================================
你只有基金，那就按基金的方式来：

* **场外基金**：只有每日单位净值（T 日盘后披露，T+1 才看到），没有盘中价
* **ETF / LOF**：有二级市场价格与净值，但场外申赎仍按净值
* **货币基金**：净值恒为 1，收益体现为万份收益 / 七日年化

模块能力
--------
1. 基金名录（代码 → 简称 / 类型），本地缓存
2. 抓历史单位净值
3. 净值指标：年化、波动、Sharpe、最大回撤、近 1/3 年收益、恢复天数
4. 类型归类（股票/混合/债券/货币/QDII/指数/FOF）

注意
----
* 基金代码与股票代码会撞号（`000001` 既是平安银行也是华夏成长）
  → 必须用 `type` 字段区分，不能靠代码猜
* 场外基金净值有 T+1 延迟，**不能当实时价格用**
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

__version__ = "0.1.0"
__all__ = ["CATALOG_CACHE", "NAV_CACHE", "load_catalog", "lookup_fund",
           "category_bucket", "fetch_nav", "latest_nav", "nav_metrics",
           "format_report"]

_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _ROOT / ".cache" / "fund"
CATALOG_CACHE = CACHE_DIR / "catalog.json"
NAV_CACHE = CACHE_DIR / "nav"
CATALOG_MAX_AGE_DAYS = 7
NAV_MAX_AGE_HOURS = 12
TD = 252


# ══════════════════════════════════════════════════════════
# 名录与分类
# ══════════════════════════════════════════════════════════
def load_catalog(refresh: bool = False,
                 cache: Path | str | None = None) -> pd.DataFrame:
    """
    基金名录：code / name / type / bucket。

    首次约 5 秒，之后走本地缓存（默认 7 天有效）。
    cache 传 None 时读模块级 CATALOG_CACHE，便于测试替换。
    """
    cache = Path(cache) if cache is not None else Path(CATALOG_CACHE)
    if cache.exists() and not refresh:
        try:
            age_d = (time.time() - cache.stat().st_mtime) / 86400
            if age_d <= CATALOG_MAX_AGE_DAYS:
                df = pd.read_csv(cache, dtype={"code": str})
                if len(df) > 0:
                    return df
        except Exception:
            pass

    try:
        import akshare as ak
        raw = ak.fund_name_em()
    except Exception:
        return pd.DataFrame(columns=["code", "name", "type", "bucket"])

    df = pd.DataFrame({
        "code": raw["基金代码"].astype(str).str.zfill(6),
        "name": raw["基金简称"].astype(str),
        "type": raw["基金类型"].astype(str),
    })
    df["bucket"] = df["type"].map(category_bucket)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False, encoding="utf-8")
    except Exception:
        pass
    return df


def _strip_paren(text: str) -> str:
    """去掉括号内的说明。例如「商品（不含QDII）」→「商品」。"""
    out, depth = [], 0
    for ch in str(text):
        if ch in "（(":
            depth += 1
            continue
        if ch in "）)":
            depth = max(depth - 1, 0)
            continue
        if depth == 0:
            out.append(ch)
    return "".join(out)


def category_bucket(fund_type: str) -> str:
    """
    把基金类型归到 9 个大类（用于风格暴露与配置分析）。

    先去掉括号说明再匹配——否则「商品（不含QDII）」会被误判成 QDII。
    """
    t = _strip_paren(fund_type).strip()
    if "商品" in t or "黄金" in t or "原油" in t:
        return "商品型"
    if "货币" in t:
        return "货币型"
    if "债券" in t or "纯债" in t or "短债" in t:
        return "债券型"
    if "QDII" in t.upper() or "海外" in t or "全球" in t:
        return "QDII"
    if "指数" in t or "ETF" in t.upper() or "联接" in t:
        return "指数型"
    if "股票" in t:
        return "股票型"
    if "混合" in t or "灵活" in t or "平衡" in t:
        return "混合型"
    if "FOF" in t.upper():
        return "FOF"
    return "其他"


def lookup_fund(code: str, catalog: pd.DataFrame | None = None) -> dict | None:
    """按代码查基金信息；查不到返回 None。"""
    df = load_catalog() if catalog is None else catalog
    if df is None or len(df) == 0:
        return None
    hit = df[df["code"] == str(code).zfill(6)]
    if len(hit) == 0:
        return None
    r = hit.iloc[0]
    return {"code": r["code"], "name": r["name"],
            "type": r["type"], "bucket": r["bucket"]}


# ══════════════════════════════════════════════════════════
# 净值
# ══════════════════════════════════════════════════════════
def _nav_cache_path(code: str, kind: str = "open") -> Path:
    safe = "".join(ch for ch in str(code) if ch.isalnum())
    return NAV_CACHE / f"{kind}_{safe}.csv"


def fetch_nav(code: str, kind: str = "open", start: str | None = None,
              refresh: bool = False,
              cache_dir: Path | str | None = None) -> pd.Series:
    """
    抓基金历史单位净值。

    kind: "open"（场外，默认）或 "etf"
    返回 index=日期 的 Series；失败抛 RuntimeError。
    """
    code = str(code).zfill(6)
    fp = _nav_cache_path(code, kind)
    if cache_dir is not None:
        fp = Path(cache_dir) / f"{kind}_{code}.csv"

    if fp.exists() and not refresh:
        try:
            age_h = (time.time() - fp.stat().st_mtime) / 3600
            if age_h <= NAV_MAX_AGE_HOURS:
                df = pd.read_csv(fp, parse_dates=["date"])
                s = df.set_index("date")["nav"].astype("float64").dropna()
                if len(s) > 5:
                    return s.sort_index()
        except Exception:
            pass

    import akshare as ak
    errs: list[str] = []

    def _open():
        raw = ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        return raw, "净值日期"

    def _etf():
        st = start or "20150101"
        raw = ak.fund_etf_fund_info_em(
            fund=code, start_date=st,
            end_date=pd.Timestamp.today().strftime("%Y%m%d"))
        return raw, "净值日期"

    # 多源：场外接口优先，失败再试 ETF 接口（同一只基金两个入口都有）
    order = [_etf, _open] if kind == "etf" else [_open, _etf]
    raw = None
    dcol = "净值日期"
    for fn in order:
        try:
            raw, dcol = fn()
            if raw is None or len(raw) == 0:
                raise RuntimeError("返回空")
            break
        except Exception as ex:
            errs.append(f"{fn.__name__}: {str(ex)[:100]}")
            raw = None
    if raw is None:
        raise RuntimeError(
            f"取净值失败（{code}/{kind}）：" + "；".join(errs))

    try:
        col = "单位净值" if "单位净值" in raw.columns else raw.columns[-1]
        s = pd.Series(
            pd.to_numeric(raw[col], errors="coerce").to_numpy(),
            index=pd.to_datetime(raw[dcol], errors="coerce"),
        ).dropna().sort_index()
        if len(s) <= 5:
            raise RuntimeError(f"净值点太少（{len(s)}）")
    except Exception as ex:
        raise RuntimeError(f"取净值失败（{code}/{kind}）：{ex}") from None

    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        s.rename("nav").rename_axis("date").to_csv(fp, float_format="%.6f")
    except Exception:
        pass
    if start:
        s = s.loc[pd.Timestamp(start):]
    return s


def latest_nav(code: str, kind: str = "open") -> float:
    """最新单位净值。取不到抛 RuntimeError（不返回假值）。"""
    s = fetch_nav(code, kind)
    return float(s.iloc[-1])


# ══════════════════════════════════════════════════════════
# 指标
# ══════════════════════════════════════════════════════════
def nav_metrics(nav, rf_annual: float = 0.0, bench=None) -> dict:
    """
    净值指标。

    * 年化收益 / 年化波动 / 夏普 / 最大回撤 / 恢复天数
    * 近 1 年、近 3 年收益
    * 可选：与基准的 beta 与 R2
    """
    s = pd.Series(nav).replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    if len(s) < 30:
        return {"ok": False, "reason": f"净值仅 {len(s)} 天，至少需要 30 天"}
    r = s.pct_change(fill_method=None).dropna()
    if len(r) < 20:
        return {"ok": False, "reason": "收益样本不足"}

    years = len(r) / TD
    total = float(s.iloc[-1] / s.iloc[0] - 1)
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan")
    sd = float(np.std(r.to_numpy(), ddof=0))
    vol = sd * np.sqrt(TD)
    rfd = (1 + rf_annual) ** (1 / TD) - 1
    sharpe = ((float(np.mean(r.to_numpy()) - rfd) / sd * np.sqrt(TD))
              if sd > 0 else float("nan"))

    arr = s.to_numpy()
    peak = np.maximum.accumulate(arr)
    dd = arr / np.where(peak <= 0, np.nan, peak) - 1
    mdd = float(np.nanmin(dd))
    trough_i = int(np.nanargmin(dd))
    recover = None
    for i in range(trough_i, len(arr)):
        if arr[i] >= peak[trough_i]:
            recover = i - trough_i
            break

    out = {
        "ok": True, "n_days": int(len(r)),
        "start": str(s.index[0].date()) if hasattr(s.index[0], "date") else str(s.index[0]),
        "end": str(s.index[-1].date()) if hasattr(s.index[-1], "date") else str(s.index[-1]),
        "nav_latest": round(float(s.iloc[-1]), 4),
        "total_return": round(total, 6),
        "annual_return": round(float(cagr), 6) if np.isfinite(cagr) else None,
        "annual_vol": round(float(vol), 6),
        "sharpe": round(float(sharpe), 4) if np.isfinite(sharpe) else None,
        "max_drawdown": round(mdd, 6),
        "recovery_days": recover,
    }
    for label, days in (("1y", 252), ("3y", 756)):
        if len(s) > days:
            out["return_" + label] = round(float(s.iloc[-1] / s.iloc[-days - 1] - 1), 6)
        else:
            out["return_" + label] = None

    if bench is not None:
        b = pd.Series(bench).replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
        idx = s.index.intersection(b.index)
        if len(idx) >= 30:
            rs = s.loc[idx].pct_change(fill_method=None).dropna()
            rb = b.loc[idx].pct_change(fill_method=None).dropna()
            common = rs.index.intersection(rb.index)
            x = rb.loc[common].to_numpy()
            y = rs.loc[common].to_numpy()
            if len(x) > 20 and float(np.var(x, ddof=1)) > 0:
                beta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
                yhat = beta * (x - x.mean())
                ss_res = float(np.sum((y - y.mean() - yhat) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                out["bench_beta"] = round(beta, 4)
                out["bench_r2"] = (round(1 - ss_res / ss_tot, 4)
                                   if ss_tot > 0 else None)
    return out


def format_report(info: dict | None, m: dict) -> str:
    L = ["=" * 66, "基金体检", "=" * 66]
    if info:
        L.append(f"  {info['code']}  {info['name']}  [{info['bucket']}]")
        L.append(f"  基金类型: {info['type']}")
    if not m.get("ok"):
        L.append(f"  无法评估：{m.get('reason')}")
        L.append("=" * 66)
        return chr(10).join(L)
    L.append("-" * 66)
    L.append(f"  区间 {m['start']} ~ {m['end']}（{m['n_days']} 天）")
    L.append(f"  最新净值 {m['nav_latest']:<10}   累计 {m['total_return'] * 100:+.2f}%")
    ar = ("—" if m.get("annual_return") is None
          else format(m["annual_return"] * 100, "+.2f") + "%")
    L.append(f"  年化收益 {ar}   年化波动 {m['annual_vol'] * 100:.2f}%"
             f"   夏普 {m['sharpe']}")
    rec = "尚未恢复" if m["recovery_days"] is None else str(m["recovery_days"]) + " 天"
    L.append(f"  最大回撤 {m['max_drawdown'] * 100:.2f}%   恢复 {rec}")
    r1 = ("—" if m.get("return_1y") is None
          else format(m["return_1y"] * 100, "+.2f") + "%")
    r3 = ("—" if m.get("return_3y") is None
          else format(m["return_3y"] * 100, "+.2f") + "%")
    L.append("  近 1 年 " + r1 + "   近 3 年 " + r3)
    if m.get("bench_r2") is not None:
        L.append(f"  与基准 beta {m.get('bench_beta')}   R2 {m.get('bench_r2')}")
    L.append("=" * 66)
    return chr(10).join(L)


if __name__ == "__main__":
    # 分类逻辑（离线）
    cases = {
        "混合型-灵活": "混合型", "股票型": "股票型", "债券型-长债": "债券型",
        "货币型": "货币型", "QDII": "QDII", "指数型-股票": "指数型",
        "FOF": "FOF", "": "其他",
    }
    for t, want in cases.items():
        got = category_bucket(t)
        assert got == want, (t, got, want)
    print("分类逻辑:", {t: category_bucket(t) for t in cases})

    # 净值指标（合成）
    idx = pd.bdate_range("2020-01-02", periods=1000)
    rng = np.random.default_rng(7)
    nav = pd.Series(np.cumprod(1 + rng.normal(0.0004, 0.01, 1000)), index=idx)
    m = nav_metrics(nav)
    assert m["ok"] and m["max_drawdown"] < 0
    print(f"指标: 年化 {m['annual_return']*100:.2f}%  回撤 {m['max_drawdown']*100:.2f}%  "
          f"近1年 {m['return_1y']}")
    print(format_report({"code": "000001", "name": "示例基金",
                         "type": "混合型-灵活", "bucket": "混合型"}, m))
    print(f"OK v{__version__}")
