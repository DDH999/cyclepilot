# -*- coding: utf-8 -*-
"""
QuantPilot 看板（M51）—— Streamlit
====================================================
启动：
    streamlit run frontend/dashboard.py
或：
    python backend/quantpilot.py dash

页面内容
--------
1. 顶部指标：仓位状态 / 最大仓位 / 沪深300 / 均线
2. 今日清单：次日买卖信号 + CSV 下载
3. 净值曲线：策略 vs 沪深300（含回撤子图）
4. 仓位上限历史：什么时候降过仓
5. 数据体检：最近的数据清洗报告

⚠ 免责
------
本页面只展示研究结果，不构成投资建议。清单里的因子在 2022-2024
只有弱正 IC（0.035），请自行判断后再执行。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import dash_data as DD  # noqa: E402

st.set_page_config(page_title="QuantPilot 看板", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")


# ══════════════════════════════════════════════════════════
@st.cache_data(ttl=300, show_spinner=False)
def _trade():
    t = DD.latest_trade_list()
    if t.get("ok"):
        t = dict(t)
        t["df"] = t["df"].to_dict("records")
    return t


@st.cache_data(ttl=300, show_spinner=False)
def _position(band: float, window: int):
    return DD.position_now(band=band, window=window)


@st.cache_data(ttl=1800, show_spinner="首次计算净值曲线约 10 秒…")
def _nav(factor: str, top_n: int, use_cap: bool, band: float, force: bool):
    r = DD.nav_history(factor=factor, top_n=top_n, use_cap=use_cap,
                       band=band, force=force)
    if r.get("ok"):
        df = r["df"].copy()
        df["date"] = df["date"].astype(str)
        r = dict(r)
        r["df"] = df.to_dict("records")
    return r


@st.cache_data(ttl=600, show_spinner=False)
def _clean():
    return DD.clean_reports(3)


# ══════════════════════════════════════════════════════════
st.sidebar.title("参数")
factor = st.sidebar.selectbox(
    "因子", ["low_vol", "reversal", "momentum", "lt_reversal"],
    index=0,
    help="low_vol 在 2022-2024 衰减最小且 IC 为正")
top_n = st.sidebar.slider("持仓只数", 5, 30, 10, 5)
use_cap = st.sidebar.checkbox("启用仓位上限", value=True,
                              help="沪深300 均线上方 80%，下方 30%")
band = st.sidebar.slider("缓冲带 %", 0.0, 5.0, 2.0, 0.5,
                         help="抑制来回切换，2% 实测最优") / 100.0
window = st.sidebar.slider("均线天数", 5, 120, 20, 5)
force = st.sidebar.button("重算净值曲线", width="stretch")
st.sidebar.caption("清单与仓位每 5 分钟刷新一次")

# ══════════════════════════════════════════════════════════
t = _trade()
p = _position(band, window)

st.title("📈 QuantPilot 看板")
st.caption(f"数据时间：{p.get('summary', {}).get('as_of', '—')}"
           if p.get("ok") else "数据未就绪")

# ── 顶部指标
c1, c2, c3, c4 = st.columns(4)
if p.get("ok"):
    state = p["summary"]["current_state"]
    cap = p["cap"]
    c1.metric("仓位状态", state,
              delta=f"偏离 {p['diff_pct']:+.2f}%" if p.get("diff_pct") else None,
              delta_color="normal")
    c2.metric("最大仓位", f"{cap*100:.0f}%",
              delta=f"阈值 {p['cap_low']*100:.0f}% / {p['cap_high']*100:.0f}%",
              delta_color="off")
    c3.metric("沪深300", f"{p['close']:,.2f}" if p.get("close") else "—")
    c4.metric(f"{window} 日均线",
              f"{p['ma']:,.2f}" if p.get("ma") else "—")
else:
    st.error(f"仓位数据不可用：{p.get('reason')}")

st.divider()

# ── 今日清单
st.subheader("📋 今日交易清单")
if t.get("ok"):
    left, right = st.columns([3, 2])
    with left:
        df = pd.DataFrame(t["df"])
        show = df.copy()
        if "target_weight" in show.columns:
            show["target_weight"] = (show["target_weight"] * 100).round(1)
            show = show.rename(columns={"target_weight": "目标权重%"})
        if "cap" in show.columns:
            show["cap"] = (show["cap"] * 100).round(0)
            show = show.rename(columns={"cap": "仓位上限%"})
        st.dataframe(show, width="stretch", hide_index=True)
        st.caption(f"信号日 {df['signal_date'].iloc[0] if 'signal_date' in df else '—'}"
                   f" ｜ 文件 {Path(t['path']).name} ｜ 更新 {t['mtime']}")
        st.download_button("下载 CSV", Path(t["path"]).read_bytes(),
                           file_name=Path(t["path"]).name,
                           mime="text/csv", width="stretch")
    with right:
        if "target_weight" in df.columns:
            total = float(df["target_weight"].sum())
            st.metric("清单合计仓位", f"{total*100:.0f}%",
                      delta=f"剩余现金 {(1-total)*100:.0f}%",
                      delta_color="off")
        if "factor" in df.columns:
            st.info(f"因子：**{df['factor'].iloc[0]}** ｜ "
                    f"等权 {len(df)} 只")
else:
    st.warning(t.get("reason", "没有清单"))
    st.code("python backend/daily_scheduler.py --job trade", language="powershell")

st.divider()

# ── 净值曲线
st.subheader("📊 历史净值（策略 vs 沪深300）")
n = _nav(factor, top_n, use_cap, band, force)
if n.get("ok"):
    df = pd.DataFrame(n["df"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    if n.get("metrics"):
        m = n["metrics"]
        a, b, c = st.columns(3)
        a.metric("年化", f"{m['annual']*100:.2f}%",
                 delta=f"基准 {(df['bench'].iloc[-1]**(1/((df.index[-1]-df.index[0]).days/365.25))-1)*100:.2f}%",
                 delta_color="off")
        b.metric("夏普", f"{m['sharpe']:.2f}")
        c.metric("最大回撤", f"{m['max_drawdown']*100:.2f}%")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.72, 0.28], vertical_spacing=0.05,
                        subplot_titles=("净值", "策略回撤"))
    fig.add_trace(go.Scatter(x=df.index, y=df["nav"], name="策略",
                             line=dict(color="#d62728", width=2)), row=1, col=1)
    if "bench" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["bench"], name="沪深300",
                                 line=dict(color="#7f7f7f", width=1.4,
                                           dash="dot")), row=1, col=1)
    dd = df["nav"] / df["nav"].cummax() - 1
    fig.add_trace(go.Scatter(x=df.index, y=dd, name="回撤",
                             fill="tozeroy", line=dict(color="#9467bd",
                                                       width=0.8)), row=2, col=1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10),
                      hovermode="x unified",
                      legend=dict(orientation="h", y=1.08))
    fig.update_yaxes(tickformat=".2f", row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"{len(df)} 个交易日 ｜ "
               f"{df.index[0].date()} ~ {df.index[-1].date()} ｜ "
               f"缓存={'是' if n.get('cached') else '否'}"
               + (f" ｜ 耗时 {n.get('elapsed')}s" if n.get("elapsed") else ""))

    # ── 仓位上限
    if "cap" in df.columns and df["cap"].notna().any():
        st.subheader("🎚️ 仓位上限历史")
        capdf = df["cap"].dropna()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=capdf.index, y=capdf, name="仓位上限",
                                  fill="tozeroy", line=dict(color="#1f77b4")))
        fig2.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                           yaxis=dict(tickformat=".0%", range=[0, 1]),
                           showlegend=False)
        st.plotly_chart(fig2, width="stretch")
        low_ratio = float((capdf <= 0.35).mean())
        st.caption(f"低于 35% 的时间占 {low_ratio*100:.0f}%｜"
                   f"平均上限 {capdf.mean()*100:.0f}%")
else:
    st.error(f"净值曲线不可用：{n.get('reason')}")
    st.code("python backend/dash_data.py", language="powershell")

st.divider()

# ── 数据体检
st.subheader("🩺 数据体检")
cl = _clean()
if cl:
    for c in cl:
        with st.expander(f"{c['date']}（{len(c['text'])} 字符）"):
            st.markdown(c["text"])
else:
    st.info("还没有清洗报告，跑 `python backend/daily_scheduler.py --job clean`")

st.divider()
st.caption("⚠ 本页面仅供研究参考，不构成投资建议。"
           "清单中的因子在 2022-2024 只有弱正 IC（0.035），"
           "三年样本无法区分能力与运气。")
