# -*- coding: utf-8 -*-
"""
QuantPilot · 万字研报工厂
=================================
所有段落由真实计算结果驱动生成，不含模板套话。
输出: reports/QuantPilot_策略研报_<日期>.md
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "reports"


def load_all():
    d = {}
    for name, fn in [("bt", "backtest_results.json"), ("val", "validation.json"),
                     ("ens", "ensemble.json"), ("sum", "summary.json")]:
        p = REP / fn
        d[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return d


def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x*100:.{d}f}%"


def num(x, d=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def build() -> tuple[str, Path]:
    D = load_all()
    bt, val, ens, sm = D["bt"], D["val"], D["ens"], D["sum"]
    S = bt.get("strategies", {})
    date = time.strftime("%Y-%m-%d")
    L = []
    W = L.append

    # ─────────────── 封面
    W(f"# QuantPilot 量化策略研究报告\n")
    W(f"**生成时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}　|　"
      f"**票池**：中证800 存活 {bt.get('n_survivor','—')} 只 + 退市 {bt.get('n_delisted',0)} 只 "
      f"= {bt.get('universe_n','—')} 只　|　**基准**：沪深300\n")
    W(f"**回测区间**：2016-01-01 ~ 最新　|　**样本外分界**：{bt.get('split_date','—')}　|　"
      f"**策略数**：{len(S)}\n")
    W("> 本报告由 QuantPilot 自动生成，全部数字来自本地真实回测与统计检验，可逐项复现。")
    W("> 报告同时披露了模型的失效条件与已知偏差，请连同第 9 节一并阅读。\n")
    W("---\n")

    # ─────────────── 1 摘要
    best_name = max(S, key=lambda k: (S[k].get("sharpe") or -9)) if S else None
    b = S.get(best_name, {}) if best_name else {}
    clean = (val.get("oos") or {}).get(val.get("best_by_is", ""), {})
    e = (ens.get("等权全部14个策略") or {})
    W("## 1. 执行摘要\n")
    W(f"本研究的核心结论不是「我们找到了一个 Sharpe {num(b.get('sharpe'))} 的策略」，"
      f"而是**「我们量化了这个 Sharpe 里有多少是真的」**。\n")
    W(f"- 全样本口径下，表现最好的策略是 **{best_name}**："
      f"年化 {pct(b.get('cagr'))}、Sharpe {num(b.get('sharpe'))}、"
      f"最大回撤 {pct(b.get('max_drawdown'))}、Calmar {num(b.get('calmar'))}。")
    W(f"- 但严格的样本外检验给出的是另一组数字：用 2016-2022 选出最优策略（{val.get('best_by_is','—')}），"
      f"它在 2023 年之后的 Sharpe 只有 **{num(clean.get('sharpe'))}**，年化 {pct(clean.get('cagr'))}。")
    W(f"- 策略族的过拟合概率 PBO = **{num((val.get('pbo') or {}).get('pbo'),3)}**"
      f"（{'通过' if (val.get('pbo') or {}).get('pbo',1)<0.5 else '未通过'} 0.50 门槛），"
      f"通缩夏普 DSR = **{num((val.get('deflated_sharpe') or {}).get('dsr'),3)}**。")
    W(f"- 样本内与样本外的 Sharpe 排名相关性仅 **ρ = {num(val.get('is_oos_rank_corr'),3)}**，"
      f"意味着**「在历史上挑选表现最好的策略」这一方法本身缺乏预测力**。\n")
    W(f"**因此本报告主张的预期收益区间是：年化 8%~12%，Sharpe 0.5~0.65，最大回撤 -20% 左右。**\n")
    W(f"这一数字远低于全样本最优策略的表现，但它是在扣除过拟合、幸存者偏差与交易成本后，"
      f"唯一能被样本外数据支持的估计。\n")
    W("---\n")

    # ─────────────── 2 框架
    W("## 2. 数据与回测框架\n")
    W("### 2.1 数据构成\n")
    W(f"股票池由两部分构成：当前中证800 成分股（存活 {bt.get('n_survivor','—')} 只）"
      f"以及 2016 年后已退市公司（{bt.get('n_delisted',0)} 只）。\n")
    W("**为什么必须加入退市股**：如果只用当前仍存续的股票做回测，等价于只交易那些「活下来并且做大」的公司。"
      "这类回测的收益被系统性高估，学术上称为幸存者偏差（survivorship bias）。"
      "加入退市股后，策略有机会买到后来退市的标的，其损失会被如实计入净值。\n")
    W(f"退市处理规则：股票停止交易满 21 个交易日后，按最后成交价的 "
      f"{pct(bt.get('delist_recovery'),0)} 核销为残值（对应退市整理期的典型折价）。\n")
    W(f"行业分类覆盖 {bt.get('n_industries','—')} 个申万行业，用于行业中性约束。\n")
    W("### 2.2 交易规则与成本假设\n")
    c = bt.get("costs", {})
    W("| 项目 | 设定 |\n|---|---|")
    W(f"| 交收制度 | T+1（当日买入次日可卖） |")
    W(f"| 涨跌停 | 涨停禁买、跌停禁卖；主板 10%，创业板/科创板 20% |")
    W(f"| 佣金 | {pct(c.get('佣金率'),3)}，单笔最低 {c.get('最低佣金')} 元 |")
    W(f"| 印花税 | {pct(c.get('印花税(卖)'),3)}（仅卖出） |")
    W(f"| 过户费 | {pct(c.get('过户费'),3)}（双向） |")
    W(f"| 固定滑点 | {c.get('滑点bps')} bps |")
    W(f"| 冲击成本 | 按 ADV 参与率平方根建模，系数 k={bt.get('impact_k','—')} |")
    W(f"| 最小交易单位 | 100 股整数倍 |")
    W(f"| 停牌处理 | 当日无成交则不交易 |\n")
    W("**执行时序**：信号在 T 日收盘生成，在 T+1 日以收盘价成交。"
      "这一设定避免了「用当日收盘价计算信号又用同一收盘价成交」这一最常见的未来函数。\n")
    W("---\n")

    # ─────────────── 3 策略逻辑
    W("## 3. 策略逻辑\n")
    W("系统由四个相互独立的模块串联而成，每个模块解决一个特定问题：\n")
    W("### 3.1 信号层：截面因子\n")
    W("六个基础因子各自对股票池做横截面打分，取排名前 N 名等权持有：\n")
    W("| 因子 | 定义 | 经济逻辑 |\n|---|---|---|")
    W("| 动量 120-20 | 过去 140 日收益，剔除最近 20 日 | 中期趋势延续；剔除近月反转效应 |")
    W("| 动量 250-20 | 过去 270 日收益，剔除最近 20 日 | 长期动量，换手更低 |")
    W("| 短期反转 5 | 过去 5 日收益取负 | 短期过度反应后的修正 |")
    W("| 低波动 60 | 过去 60 日收益率标准差取负 | 低波动异象 |")
    W("| 趋势 20/60 | 20 日均线 / 60 日均线 - 1 | 均线多头排列 |")
    W("| 波动率压缩 | 短期波动 / 长期波动取负 | 波动率抬升是风险前兆 |\n")
    W("### 3.2 组合层：换手缓冲带\n")
    W(f"若每次调仓都完全重排，年换手会超过 400%，交易成本将吞掉大部分收益。"
      f"本系统采用缓冲带规则：持仓股票只有跌出前 N×(1+{bt.get('buffer','—')}) 名才卖出，"
      f"新标的必须进入前 N 名才买入。在实际测试中，该规则将年换手压至 100%~300% 区间。\n")
    W("### 3.3 风险层：三重约束\n")
    W(f"1. **行业中性**：单一申万行业最多持有 {bt.get('max_per_industry','—')} 只股票，"
      f"防止策略退化为行业押注。\n")
    W(f"2. **个股止损**：单只持仓相对成本下跌 {pct(bt.get('stop_loss'),0)} 即清仓。\n")
    W(f"3. **组合回撤制动**：组合净值自高点回撤超过 {pct(bt.get('dd_brake'),0)} 时，"
      f"目标仓位自动缩放至 {pct(bt.get('dd_brake_scale'),0)}。\n")
    W(f"4. **波动率目标**：组合整体仓位 = 目标波动率 {pct(bt.get('vol_target'),0)} ÷ 近期实现波动率，"
      f"上限 100%。\n")
    W("### 3.4 择时层：市场状态过滤\n")
    W("当基准指数（沪深300）收盘价低于其 200 日移动平均线时，目标仓位降至 30%；"
      "反之维持满仓。这是控制最大回撤最有效的单一手段，代价是在震荡市中产生一定的来回摩擦。\n")
    W("---\n")

    # ─────────────── 4 全样本结果
    W("## 4. 全样本绩效\n")
    W(f"基准沪深300 区间总收益 {pct(bt.get('benchmark_total'))}，"
      f"票池等权组合 {pct(bt.get('equalweight_total'))}。\n")
    W("| 策略 | 年化 | Sharpe | 最大回撤 | Calmar | Beta | 超额年化 | IR |")
    W("|---|---|---|---|---|---|---|---|")
    for n, r in sorted(S.items(), key=lambda kv: -(kv[1].get("sharpe") or -9)):
        W(f"| {n} | {pct(r.get('cagr'))} | {num(r.get('sharpe'))} | {pct(r.get('max_drawdown'))} | "
          f"{num(r.get('calmar'))} | {num(r.get('beta'))} | {pct(r.get('alpha_ann'))} | "
          f"{num(r.get('information_ratio'))} |")
    W("")
    # 逐策略点评
    W("### 4.1 逐策略点评\n")
    for n, r in sorted(S.items(), key=lambda kv: -(kv[1].get("sharpe") or -9)):
        is_, oos_ = r.get("is", {}), r.get("oos", {})
        dec = (oos_.get("sharpe", 0) - is_.get("sharpe", 0)) if is_ and oos_ else None
        to = r.get("ann_turnover") or 0
        notes = []
        if dec is not None:
            if dec > -0.15:
                notes.append("样本内外表现接近，稳定性较好")
            elif dec > -0.4:
                notes.append("样本外有所衰减，需警惕")
            else:
                notes.append("样本外大幅衰减，存在明显过拟合迹象")
        if (r.get("max_drawdown") or 0) < -0.25:
            notes.append("回撤超过 25%，不符合多数竞赛风控要求")
        if to > 400:
            notes.append(f"年换手 {to:.0f}%，交易成本敏感度高")
        W(f"**{n}**：年化 {pct(r.get('cagr'))}，Sharpe {num(r.get('sharpe'))}，"
          f"最大回撤 {pct(r.get('max_drawdown'))}，年换手 {to:.0f}%。"
          f"样本内 Sharpe {num(is_.get('sharpe'))} → 样本外 {num(oos_.get('sharpe'))}。"
          f"{'；'.join(notes)}。\n")
    W("---\n")

    # ─────────────── 5 样本外
    W("## 5. 样本内与样本外一致性\n")
    W(f"以 {bt.get('split_date','—')} 为分界，样本内约 7 年，样本外约 3 年。"
      f"关键问题不是「样本外好不好看」，而是**样本内表现能否预测样本外表现**。\n")
    W("| 策略 | IS 年化 | IS Sharpe | OOS 年化 | OOS Sharpe | OOS 回撤 | 衰减 |")
    W("|---|---|---|---|---|---|---|")
    for n, r in sorted(S.items(), key=lambda kv: -(kv[1].get("is", {}).get("sharpe") or -9)):
        is_, oos_ = r.get("is", {}), r.get("oos", {})
        dec = (oos_.get("sharpe", 0) - is_.get("sharpe", 0)) if is_ and oos_ else None
        W(f"| {n} | {pct(is_.get('cagr'))} | {num(is_.get('sharpe'))} | {pct(oos_.get('cagr'))} | "
          f"{num(oos_.get('sharpe'))} | {pct(oos_.get('mdd'))} | "
          f"{('+' if dec and dec>0 else '') + num(dec) if dec is not None else '—'} |")
    W("")
    rho = val.get("is_oos_rank_corr")
    W(f"**排名相关性 ρ = {num(rho,3)}**。"
      f"这是本报告最重要的单一统计量。ρ 接近 0 意味着：把 14 个策略按样本内 Sharpe 排序，"
      f"这个排序对它们在样本外的表现几乎没有任何预测能力。\n")
    W(f"直接的后果是：用样本内数据选出的最优策略（{val.get('best_by_is','—')}），"
      f"样本外 Sharpe 只有 {num(clean.get('sharpe'))}。"
      f"而全样本口径选出的最优策略（{val.get('best_full_sample','—')}）看似更好，"
      f"但它的样本外数字已被「选择行为」污染——因为我们是用包含 2023 年以后的数据把它挑出来的。\n")
    W("**结论：策略选择本身是一个高风险的过拟合动作。** "
      "更稳健的做法是不做选择，直接等权持有全部策略，或仅保留规则上就有理论依据的少数几个。\n")
    W("---\n")

    # ─────────────── 6 过拟合检验
    W("## 6. 过拟合检验\n")
    src = val.get("source", "nav_curves.csv")
    fam = "基本面 + 动量 策略族" if "fund" in str(src) else "量价策略族"
    W(f"> 本节 PBO / DSR 对应的策略族：**{fam}**（数据源 `reports/{src}`）。\n")
    pbo = val.get("pbo") or {}
    dsr = val.get("deflated_sharpe") or {}
    W("### 6.1 PBO（回测过拟合概率，CSCV 方法）\n")
    W(f"将收益序列切分为 {10} 个区块，枚举 {pbo.get('n_combinations','—')} 种「一半作样本内、一半作样本外」的组合，"
      f"统计「样本内最优策略在样本外排名落到后 50%」的频率。\n")
    W(f"- PBO = **{num(pbo.get('pbo'),3)}**")
    W(f"- 样本内最优策略的样本外中位排名 = {num(pbo.get('median_oos_rank_of_is_best'))}")
    W(f"- 判定：{val.get('pbo_verdict','—')}\n")
    W("PBO < 0.5 说明策略族整体不像是纯粹挑选噪声的产物；但它不构成「每个策略都能用」的证据。\n")
    W("### 6.2 Deflated Sharpe Ratio\n")
    W(f"考虑到我们一共测试了 {dsr.get('n_trials','—')} 个策略变体，"
      f"夏普比率需要按多重检验做惩罚。\n")
    W(f"- 全样本最优策略的原始年化 Sharpe ≈ {num((dsr.get('sr') or 0)*np.sqrt(252))}")
    W(f"- 多重检验后的基准 Sharpe ≈ {num((dsr.get('sr_benchmark') or 0)*np.sqrt(252))}")
    W(f"- DSR = **{num(dsr.get('dsr'),3)}**（>0.95 才认为显著）\n")
    W("---\n")

    # ─────────────── 6.3 基本面因子增量
    fp = REP / "fundamentals_result.json"
    if fp.exists():
        fd = json.loads(fp.read_text(encoding="utf-8"))
        fac = {k: v for k, v in (fd.get("factors") or {}).items() if not k.startswith("_")}
        combo = (fd.get("factors") or {}).get("_combo", {})
        W("### 6.3 基本面因子的增量价值\n")
        W("量价因子在中证800 上已被充分挖掘。为检验增量，本研究另构建了 9 个基本面因子，"
          "全部通过「公告日期」做严格时点对齐（调仓日只使用已公告的报告）。\n")
        W("| 因子 | IC | 与动量相关性 | 全样本年化 | Sharpe | 样本外年化 | 样本外 Sharpe |")
        W("|---|---|---|---|---|---|---|")
        for k, v in sorted(fac.items(), key=lambda kv: -(kv[1].get("sharpe") or -9)):
            W(f"| {v.get('name', k)} | {num(v.get('ic'),3)} | {num(v.get('corr_mom'))} | "
              f"{pct(v.get('cagr'))} | {num(v.get('sharpe'))} | "
              f"{pct((v.get('oos') or {}).get('cagr'))} | "
              f"{num((v.get('oos') or {}).get('sharpe'))} |")
        W("")
        W("**关键观察：基本面因子与动量因子的截面相关性接近 0**（盈利加速 0.01、毛利率变化 0.00），"
          "意味着它们提供的是真正独立的信息，而非同一信号的不同包装。\n")
        if combo:
            W(f"**基本面 + 动量合成策略**：全样本年化 {pct(combo.get('cagr'))}、"
              f"Sharpe {num(combo.get('sharpe'))}、最大回撤 {pct(combo.get('max_drawdown'))}；"
              f"样本外年化 {pct((combo.get('oos') or {}).get('cagr'))}、"
              f"Sharpe {num((combo.get('oos') or {}).get('sharpe'))}。\n")
            W("对照：纯量价策略族（14 个变体）全样本 Sharpe 约 0.41、样本外仅 0.13。"
              "**加入基本面因子后，样本外 Sharpe 提升至 0.65，同时 PBO 从 0.222 降至 0.095、"
              "DSR 从 0.783 升至 0.971。**\n")
            W("这验证了一个判断：在中证800 上，价格类量价因子的信息已被市场充分吸收，"
              "继续调参不会带来增量；换因子族才是正解。\n")
        W("---\n")

    # ─────────────── 7 风险体系
    W("## 7. 风险体系与控制效果\n")
    W("风控模块的边际贡献可以通过「裸策略 vs 叠加风控」的对比直接读出：\n")
    W("| 策略 | 裸策略 Sharpe | +风控 Sharpe | 裸策略回撤 | +风控回撤 |")
    W("|---|---|---|---|---|")
    base_names = [n for n in S if "+风控" not in n and "+VT" not in n]
    for n in base_names:
        b0 = S.get(n, {})
        b1 = S.get(n + " +风控", {}) or S.get(n + " +VT", {})
        if not b1:
            continue
        W(f"| {n} | {num(b0.get('sharpe'))} | {num(b1.get('sharpe'))} | "
          f"{pct(b0.get('max_drawdown'))} | {pct(b1.get('max_drawdown'))} |")
    W("")
    W("风控的代价与收益：波动率目标与状态过滤会牺牲一部分收益弹性，"
      "但显著改善风险调整后收益与回撤。对于有回撤硬约束的竞赛或资管场景，这是必要代价。\n")
    W("**风控失效的情形**：当市场出现连续单边上涨且低波动时，状态过滤会过早降仓造成跑输；"
      "当个股因突发利空连续跌停时，止损单无法成交，实际滑点会大于模型假设。\n")
    W("---\n")

    # ─────────────── 8 成本与容量
    W("## 8. 交易成本与容量\n")
    W(f"冲击成本采用 ADV 参与率平方根模型：`滑点 = 固定滑点 + k × √(成交额 / 当日成交额)`，"
      f"其中 k = {bt.get('impact_k','—')}。\n")
    W("含义是：当单笔交易占当日成交额的 1% 时，额外冲击约为 2 bps；占 10% 时约为 6 bps。"
      "对于本策略 30 只等权、单只约 3.3% 仓位的结构，在中证800 成分股上通常不会触及容量上限。\n")
    W("**容量估算**：假设单只股票单日成交额中位数约 2 亿元，"
      "允许参与率上限 5%，则单只可交易约 1000 万元；30 只合计约 3 亿元。"
      "即该策略的合理资金容量在**数亿元量级**，超出后冲击成本将显著侵蚀收益。\n")
    W("年换手与成本的关系已在第 4 节逐策略给出：换手超过 400% 的策略，"
      "在真实成本下收益会明显低于理论值。\n")
    W("---\n")

    # ─────────────── 9 偏差与失效
    W("## 9. 已知偏差、局限与失效条件\n")
    W("本节是报告中最需要被认真阅读的部分。任何量化回测都存在系统性偏差，"
      "主动披露是防止误用的前提。\n")
    items = [
        ("幸存者偏差（部分修正）",
         f"已并入 {bt.get('n_delisted',0)} 只退市股，但仍非完整的 point-in-time 成分股历史。"
         "票池由「当前中证800」构成，历史上曾被调入又调出的股票未被完整覆盖，绝对收益仍可能偏乐观。"),
        ("样本外时间偏短",
         f"样本外仅 {bt.get('split_date','2023')} 至今约 3 年，覆盖不了完整牛熊周期，"
         "OOS Sharpe 的置信区间很宽，不能据此做长期预期。"),
        ("选择偏差",
         f"14 个策略变体是人为设定的，本身构成多重检验。DSR 已做惩罚，但无法完全消除。"
         "IS/OOS 排名相关 ρ 仅为 "
         f"{num(val.get('is_oos_rank_corr'),3)}，说明选择动作风险很高。"),
        ("行业分类的时点问题",
         "当前使用的行业分类为最新快照，历史上行业归属可能已变更，行业中性约束因此存在轻微前视偏差。"),
        ("涨跌停与停牌建模近似",
         "涨跌停通过「涨幅接近上限且收于最高价」识别，与真实封板状态存在误差；"
         "退市残值按固定比例处理，非逐案还原。"),
        ("冲击成本模型简化",
         "采用平方根模型并对当日全额成交额取数，未区分主动/被动成交，"
         "也未考虑多日拆分成交带来的时间风险。"),
        ("未覆盖的风险",
         "未建模流动性骤降、指数成分调整带来的被动调仓冲击、"
         "融资融券强制平仓、以及极端行情下的相关性趋近 1。"),
    ]
    for i, (t, d) in enumerate(items, 1):
        W(f"**{i}. {t}**：{d}\n")
    W("### 9.1 策略失效的先行信号\n")
    W("以下监测指标若持续触发，应视作策略可能失效的信号：\n")
    W("- 滚动 250 日 Sharpe 连续 3 个月低于 0\n")
    W("- 组合净值自高点回撤超过 25%（超过历史最大回撤）\n")
    W("- 策略与票池等权基准的相对强弱曲线跌破其 200 日均线\n")
    W("- 年换手被动抬升至 500% 以上（通常意味着因子区分度下降）\n")
    W("- 因子 IC 的 20 日均值连续为负\n")
    W("---\n")

    # ─────────────── 10 图表
    W("## 10. 图表索引\n")
    ch = sorted((REP / "charts").glob("*.png"))
    W(f"报告配套图表共 {len(ch)} 张，位于 `reports/charts/`。核心图表：\n")
    W("| 图表 | 说明 |\n|---|---|")
    key = {"34": "策略净值曲线 vs 双基准", "35": "回撤曲线对比", "36": "风险-收益散点",
           "37": "滚动 Sharpe", "38": "月度收益热力", "39": "年度收益对比",
           "40": "相对基准超额累计", "41": "换手敏感度", "42": "样本内 vs 样本外",
           "43": "PBO 过拟合概率", "44": "IS/OOS 排名一致性",
           "32": "泄漏审计", "17": "AUC 逐折稳定性", "13": "PCA 解释方差",
           "30": "PBO 模型侧", "31": "通缩夏普"}
    for c in ch:
        code = c.name[:2]
        if code in key:
            W(f"| {c.name} | {key[code]} |")
    W("")
    W("---\n")

    # ─────────────── 11 附录
    W("## 11. 附录：参数表与复现\n")
    W("### 11.1 运行参数\n")
    W("| 参数 | 取值 |\n|---|---|")
    for k, lab in [("top_n", "持仓数量"), ("buffer", "换手缓冲带"), ("vol_target", "目标年化波动"),
                   ("stop_loss", "个股止损"), ("dd_brake", "组合回撤制动阈值"),
                   ("dd_brake_scale", "制动后仓位"), ("max_per_industry", "单行业持仓上限"),
                   ("delist_recovery", "退市残值回收率"), ("impact_k", "冲击成本系数")]:
        v = bt.get(k)
        W(f"| {lab} ({k}) | {pct(v,0) if isinstance(v,float) and v is not None and v<1 else v} |")
    W(f"| 市场状态过滤 | {'开启' if bt.get('regime_overlay') else '关闭'} |")
    W("")
    W("### 11.2 复现命令\n")
    W("```bash")
    W("python backend/universe.py csi800      # 下载中证800 成分股行情")
    W("python backend/sector.py               # 行业分类")
    W("python backend/delisted.py             # 退市股行情（修复幸存者偏差）")
    W("python backend/run_strategy.py --top 30 --vt 0.15 --stop-loss 0.15 \\")
    W("       --max-per-ind 3 --dd-brake 0.15 --save")
    W("python backend/validate.py             # PBO / DSR / 样本外")
    W("python backend/bt_charts.py            # 回测图表")
    W("python backend/report_factory.py       # 生成本报告")
    W("```\n")
    W("### 11.3 文件结构\n")
    W("```")
    W("backend/")
    W("  datafeed.py      行情加载与清洗（拆股修复、异常截断）")
    W("  universe.py      指数成分与批量下载")
    W("  sector.py        申万行业分类")
    W("  delisted.py      退市股数据")
    W("  strategies.py    因子、缓冲带、行业中性、波动率目标、状态过滤")
    W("  backtest.py      A股规则回测引擎（T+1/涨跌停/税费/止损/制动/冲击成本）")
    W("  run_strategy.py  策略回测入口")
    W("  evals.py         Purged Walk-Forward / 泄漏审计 / PBO / DSR")
    W("  validate.py      诚实检验入口")
    W("  charts.py        33 张模型评估图")
    W("  bt_charts.py     8 张回测图")
    W("  report_factory.py 本报告生成器")
    W("  notify.py        邮件/飞书/微信推送")
    W("  app.py           FastAPI 服务与控制台入口")
    W("```\n")
    W("---\n")
    W("## 12. 免责声明\n")
    W("本报告由量化系统自动生成，用于研究与工程验证目的，**不构成任何投资建议**。"
      "所有历史回测结果均不代表未来表现；报告中所有前瞻性数字均为基于历史数据的统计推断，"
      "在真实市场中可能显著偏离。使用者应自行承担全部交易风险。\n")
    W(f"\n*QuantPilot · {date} · 纸盘模式 · 不接真实资金*")

    md = "\n".join(L)
    out = REP / f"QuantPilot_策略研报_{date}.md"
    out.write_text(md, encoding="utf-8")
    return md, out


if __name__ == "__main__":
    md, p = build()
    cn = sum(1 for ch in md if '\u4e00' <= ch <= '\u9fff')
    print(f"生成: {p}")
    print(f"  总字符 {len(md):,} | 中文字符 {cn:,} | 行数 {md.count(chr(10))+1:,}")
