# QuantPilot 架构说明

**项目定位**：量化策略真实性与风险体检平台 —— 不追求选股收益，只做策略打假与风控质检
**当前进度**：**M1–M29 全部完成** ｜ 测试 1514 / 1514
**契约版本**：`1.0`（`backend/schemas.py`）
**更新**：2026-11-08

---

## 一、总体数据流

```
                     ┌──────────────────────────────────────┐
                     │ 输入：策略代码 + 收益序列 CSV          │
                     │      （可选）持仓权重 CSV / 基准 CSV   │
                     └──────────────────┬───────────────────┘
                                        │  api_v2.run_checkup
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
┌───────────────┐             ┌──────────────────┐            ┌────────────────┐
│ validator.py  │             │ stress_tester.py │            │ purity_tester  │
│               │             │                  │            │                │
│ 未来函数扫描   │             │ 滑点压力          │            │ 等效持仓数      │
│ DSR / PBO     │             │ 极端行情切片      │            │ 仓位暴露度      │
│ PBO 分级诊断   │             │ 崩盘场景套件      │            │ 与基准 R²       │
└───────┬───────┘             └────────┬─────────┘            │ 收益集中度      │
        │    dict（schemas.py 约束）    │                      └───────┬────────┘
        └───────────────┬──────────────┴──────────────────────────────┘
                        ▼
             ┌────────────────────────┐
             │  adjudicator.py        │  ← 方案 D 分级裁决
             │                        │
             │  CORE      → REJECTED  │
             │  DATA_NA   → N/A       │
             │  MATH_ABN  → WARN 封顶C │
             │  EXEMPT    → N/A 标注  │
             │  人工复核 override      │
             │  审计日志 AUDIT_LOG     │
             └───────────┬────────────┘
                         ▼
             ┌────────────────────────┐
             │  scoring.py            │  ← 六维聚合
             │  防作弊 / 抗过拟合      │
             │  抗压 / 纯度 / 成本     │
             │  稳健性                 │
             └───────────┬────────────┘
                         ▼
             六维雷达数据 + 总评级（A/B/C/D/REJECTED）
                         │
        ┌────────────────┴─────────────────┐
        ▼                                  ▼
┌──────────────────┐            ┌──────────────────────┐
│ ai_diagnostician │            │ frontend/checkup.html│
│                  │            │                      │
│ 主：DeepSeek API │            │ Canvas 雷达图         │
│ 兜：template_eng │            │ 维度表 + 三段诊断      │
│     + 知识库 RAG │            │ 代码命中明细          │
└────────┬─────────┘            └──────────────────────┘
         ▼
  三段诊断（综合定性 / 核心致命伤 / 整改建议）
  source = "llm" 或 "template"，降级时如实标注
```

---

## 二、模块清单

| 模块 | 版本 | 职责 | 对外接口 |
|---|---|---|---|
| `validator.py` | 0.4.0 | 代码真实性、过拟合评估 | `check_future_function` `scan_code` `calculate_overfitting` `grade_pbo` |
| `stress_tester.py` | 0.3.0 | 滑点与极端行情 | `apply_slippage` `test_extreme_crash` `run_crash_suite` |
| `purity_tester.py` | 0.1.0 | 策略纯度四指标 | `purity_report` |
| `adjudicator.py` | 0.1.0 | 方案 D 裁决 + 人工复核 | `adjudicate` `apply_override` |
| `scoring.py` | 0.1.0 | 六维评分与总报告 | `full_report` `build_radar` |
| `schemas.py` | 0.1.0 | 模块间数据契约 | `validate_payload` |
| `ai_diagnostician.py` | 0.1.0 | 大模型诊断 | `diagnose` `build_prompt` `call_llm` |
| `template_engine.py` | 0.1.0 | 本地规则模板（降级） | `render` `summarize` `find_fatal` `build_advice` |
| `api_v2.py` | 0.1.0 | 体检服务 | `run_checkup` `parse_series_csv` `app` |
| `daily_job.py` | 0.1.0 | 日频自动体检 + 变化提醒 | `run_daily` `detect_changes` `notify` `run_loop` |
| `attribution.py` | 0.1.0 | 收益归因 + 成本核算 | `performance_attribution` `cost_analysis` `load_bench` |
| `compound.py` | 0.1.0 | 复利测算 | `compound_growth` `required_rate` `fee_drag_effect` |
| `valuation.py` | 0.1.0 | 估值分位（指数/个股） | `valuation_report` `percentile_rank` |
| `blackswan.py` | 0.1.0 | 事件日历 + 情景回放 + 尾部风险 | `event_calendar` `stress_windows` `fragility_score` |
| `datacheck.py` | 0.1.0 | 数据质量体检 | `check_series` |
| `walkforward.py` | 0.1.0 | 滚动样本外（purge/embargo） | `purged_walk_forward` |
| `execution.py` | 0.1.0 | 冲击成本与容量 | `cost_breakdown` `max_capacity` |
| `riskcontrol.py` | 0.1.0 | 四道闸门动态风控 | `kill_switch` `crowding_score` `style_exposure` |
| `portfolio_theory.py` | 0.1.0 | EMH / MPT / CAPM / APT / 主动管理法则 | `emh_test` `efficient_frontier` `capm_regression` `apt_regression` `fundamental_law` |
| `allocation.py` | 0.1.0 | 资产配置 · 风险平价 · 凯利 | `risk_parity` `compare_allocation` `kelly_fraction` |
| `timeseries.py` | 0.1.0 | 时间序列诊断 | `ts_report` `hurst` `adf_proxy` `half_life` |
| `pitfalls.py` | 0.1.0 | 回测陷阱十条清单 | `check_pitfalls` |
| `portfolio.py` | 0.1.0 | 真实持仓台账（读 CSV → 估值 → 权重） | `load_holdings` `mark_to_market` `fetch_last_prices` |
| `fund.py` | 0.1.0 | 基金名录/净值/指标/分类 | `load_catalog` `fetch_nav` `nav_metrics` |
| `dca.py` | 0.1.0 | 定投台账 / XIRR / 定投vs一次性 | `load_ledger` `xirr` `dca_analysis` |
| `calibration.py` | 0.1.0 | 评分效度校准（敏感性/区分度/盲区） | `calibrate` `run_profile` |
| `risk_state.py` | 0.1.0 | 风控闭环（落盘/执行率） | `record` `followthrough` |
| `quantpilot.py` | 0.1.0 | 统一入口 CLI | `main` |
| `strategy_runner.py` | 0.1.0 | 策略代码 → 回测 → 六维 | `run_from_code` `full_checkup` |
| `datasource.py` | 0.1.0 | 多源降级 + 熔断 + 列名归一 | `get_stock_daily` `get_fund_nav` |
| `robustness.py` | 0.1.0 | 扰动测试 + 压力测试 | `perturbation_suite` `stress_suite` |
| `cache_manager.py` | 0.1.0 | 缓存统计/按龄清理/全清 | `stats` `trim` `clean_all` |
| `zz_source.py` | 0.1.0 | zzshare 适配（日线/估值/行业） | `stock_daily` `valuation_snapshot` `industry_map` |
| `overseas.py` | 0.1.0 | 海外指数与 QDII 基准 | `index_history` `match_fund_index` `fund_vs_benchmark` |
| `baostock_source.py` | 0.1.0 | Baostock 适配（日线/行业） | `stock_daily` `industry_map` |
| `credentials.py` | 0.1.0 | API 凭据（掩码/环境变量优先） | `load_key` `save_key` `status` |
| `us_source.py` | 0.1.0 | 美股数据（ROIC/xfinlink，含节流） | `roic_profile` `roic_latest` `xfin_prices` |
| `quality.py` | 0.1.0 | 第七维：基本面质量（美股） | `quality_report` `score_quality` |
| `advisor.py` | 0.1.0 | 实盘决策引擎 | `advise` `dca_multiplier` `rebalance_plan` |
| `ml_risk.py` | 0.1.0 | 回撤风险预测（含自我否决） | `validate_market` `current_risk` `risk_multiplier` |
| `optimizer.py` | 0.1.0 | 马科维茨稳定性 + 凯利上限 | `markowitz_stability` `kelly_sizing` `optimize_report` |
| `journal.py` | 0.1.0 | 决策台账 + 到期结算 + 复盘 | `record` `settle_pending` `review` |
| `cb_new.py` | 0.1.0 | 可转债打新扫描 | `scan` `assess_bond` |
| `fee_check.py` | 0.1.0 | 费率核对 A/C 类 | `breakeven_months` `compare_ac` |
| `paper_validate.py` | 0.1.0 | 主动交易准入五道闸门 | `validate` |
| `cb_double_low.py` | 0.1.0 | 可转债双低轮动（含回测） | `screener` `redeem_risk` `backtest_double_low` |
| `cb_paper.py` | 0.1.0 | 滑点/冲击成本 + 模拟盘 | `order_cost` `simulate_double_low` `advance` |
| `backtest_strict.py` | 0.1.0 | 严格回测（次日开盘/涨跌停/样本外） | `StrictBacktest` `tune_oos` `audit_panel` |
| `strategy_lab.py` | 0.1.0 | 多策略族横评（IC + OOS） | `run_matrix` `ic_series` `build_scores` |
| `position_manager.py` | 0.1.0 | 大盘趋势 → 仓位上限 | `position_cap` `apply_cap` `cap_summary` |
| `daily_scheduler.py` | 0.1.0 | 每日定时任务（schedule 库） | `clean_data` `make_trade_list` `run_forever` |
| `mailer.py` | 0.1.0 | 邮件推送（SMTP + HTML + 附件） | `send_mail` `render_daily` `load_config` |
| `dash_data.py` | 0.1.0 | 看板数据层（清单/仓位/净值缓存） | `nav_history` `latest_trade_list` `position_now` |
| `frontend/dashboard.py` | 0.1.0 | Streamlit 看板页面 | — |
| `preflight.py` | 0.1.0 | 一键自检（环境/依赖/测试/产物/路由） | `main` |

---

## 三、裁决规则（方案 D）

| NaN 类别 | 触发条件 | 处理 | 对评级的影响 |
|---|---|---|---|
| **CORE** | 净值为空、年化算不出、**代码含未来函数** | REJECTED | 一票否决，得分 0 |
| **DATA_UNAVAILABLE** | 数据区间未覆盖 | N/A | 不进分母，不降级 |
| **MATH_ABNORMAL** | 零波动、样本不足、分母为 0 | WARN | 该项 0 分，**整体封顶 C** |
| **EXEMPT** | 指标对该策略不适用（全程空仓） | N/A | 特别标注，不扣分 |

**REJECTED 的两种文案必须区分**：

| 成因 | 文案 |
|---|---|
| 数据异常 | `数据异常，无法评估，判定为不合格` |
| 未来函数 | `检测到未来函数（数据穿越）嫌疑，策略存在主观作弊，直接一票否决，严禁上盘` |

结论优先级：`REJECTED > FAIL > WARNING > PASS`

---

## 四、六维评分

| 维度 | key | 权重 | 取数来源 |
|---|---|---|---|
| 防作弊 | `anti_cheat` | 0.25 | `validator.check_future_function` |
| 抗过拟合 | `anti_overfit` | 0.20 | `validator.calculate_overfitting` |
| 抗压能力 | `stress` | 0.20 | 滑点 + 极端行情切片 |
| 策略纯度 | `purity` | 0.15 | `purity_tester.purity_report` |
| 成本可控 | `cost` | 0.10 | 年化成本拖累 |
| 稳健性 | `stability` | 0.10 | 参数敏感性（可豁免） |

评级门槛：`≥90 → A`，`≥75 → B`，`≥60 → C`，其余 `D`；存在 WARN 项时封顶 C。
雷达图中 N/A 维度为 `None`，前端画灰色轴，**不画成 0 分**。

---

## 五、数据契约

```
StressResult · CrashResult · OverfitResult · PurityResult · ReportModel · ItemDict
```

原则：

- `extra="allow"` 保证向后兼容（下游加字段不崩）
- 校验失败**返回错误清单**，不抛异常
- 所有模型带 `contract: "1.0"`

```python
from schemas import ReportModel, validate_payload
ok, errs = validate_payload(ReportModel, report_dict)
```

---

## 六、人工复核（override）

```python
adjudicator.adjudicate(items, override={
    "decision": "FAIL",
    "reason": "人工复核发现数据源错误",
    "operator": "zhang",
    "timestamp": "2026-11-04T09:00:00Z",
})
```

安全边界：

- 缺 `reason` → 拒绝
- `reject_kind == "cheat"`（未来函数）→ **拒绝，不可覆盖**
- 所有尝试（含拒绝）写入 `result["audit"]` 与模块级 `AUDIT_LOG`

---

## 七、大模型诊断与降级链

```
diagnose(report)
  ├─ use_llm=False / QP_LLM_DISABLE=1 → 模板 llm_disabled
  ├─ 无 QP_LLM_KEY                    → 模板 no_api_key（不发请求）
  ├─ 网络不通 / DNS 失败               → 模板 network_error
  ├─ 超时                              → 模板 timeout
  ├─ 401 / 403                        → 模板 auth_failed
  ├─ 402 额度用尽                      → 模板 quota_exhausted
  ├─ 429 限流                          → 模板 rate_limited
  ├─ 返回非 JSON / 缺三段               → 模板 bad_response
  └─ 任何未预料异常                     → 模板 unexpected_error
```

**`diagnose()` 永不抛异常。** 降级结果里 `degraded=True`、`fallback_reason` 如实标注。

诊断输出强制三段：`综合定性` / `核心致命伤` / `整改建议`。
人格设定为**客观严谨的金融风控合规专家**，禁止情绪化表述。

环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `QP_LLM_KEY` | 空 | 缺失则直接走模板 |
| `QP_LLM_BASE` | `https://api.deepseek.com` | API 地址 |
| `QP_LLM_MODEL` | `deepseek-chat` | 模型名 |
| `QP_LLM_TIMEOUT` | `20` | 秒 |
| `QP_LLM_DISABLE` | 空 | 设 `1` 强制走模板（断网演练） |
| `QP_PORT` | `8421` | 服务端口 |

---

## 八、HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 体检页面 |
| GET | `/api/health` | 健康检查（含 LLM 是否已配置） |
| POST | `/api/checkup` | 提交体检，JSON body |
| GET | `/api/daily/latest` | 最近一次日更结果 |
| GET | `/api/daily/history?limit=30` | 历史日更摘要 |
| GET | `/api/paper/nav?limit=500` | 纸盘净值曲线 |
| POST | `/api/daily/run` | 手动触发一次日更（约 5 秒） |
| GET | `/api/attribution` | 收益归因 + 成本核算（对标沪深300） |
| GET | `/api/risk` | 风控仪表盘聚合（数据/样本外/成本/四道闸门） |
| POST | `/api/checkup_code` | 策略代码 → 回测 → 六维体检 |
| GET | `/api/dca` | 定投台账分析（XIRR，净值自动补齐） |
| GET | `/api/sources` | 数据源健康（可用/熔断/降级次数） |
| GET | `/api/perturbation` | 扰动测试（结论稳定性） |
| GET | `/api/stress` | 压力测试（极端情景存活率） |
| GET | `/api/zz` | zzshare 可用性 |
| GET | `/api/valuation` | 全市场估值快照 |
| GET | `/api/industry_map` | 股票→行业映射 |
| GET | `/api/overseas` | 海外指数历史 |
| GET | `/api/overseas/diagnose` | 数据源连通性诊断 |
| GET | `/api/baostock` | Baostock 可用性 |
| GET | `/api/us` | 美股数据（profile/latest/prices） |
| GET | `/api/keys` | 凭据状态（仅掩码） |
| GET | `/api/quality` | 基本面质量评分 |
| GET | `/api/advice` | 今日决策（定投/再平衡/回撤） |
| GET | `/api/ml_risk` | 回撤风险预测（含验证结果） |
| GET | `/api/optimizer` | 马科维茨/凯利实测 |
| GET | `/api/journal` | 决策台账（结算/复盘） |
| GET | `/api/cb_new` | 可转债打新扫描 |
| GET | `/api/fee` | 费率核对 |
| GET | `/api/admit` | 交易准入判定 |
| GET | `/api/cb_double_low` | 双低候选池 + 强赎 |
| GET | `/api/cb_paper` | 可转债模拟盘（查看/初始化/推进） |
| GET | `/api/position` | 仓位管理器（沪深300 趋势 → 上限） |

请求体：

```json
{
  "code": "策略源码文本",
  "returns_csv": "date,ret\n2023-01-03,0.0012\n...",
  "weights_csv": "date,600519,000001\n...",
  "bench_csv": "date,ret\n...",
  "use_llm": false,
  "is_split": "2023-01-01"
}
```

只有 `returns_csv` 必填，其余可选。解析失败返回 400 + 中文原因；其余异常兜底返回 500 说明，前端不会白屏。

---

## 九、前端

单文件 `frontend/checkup.html`（约 26 KB，零依赖、零 CDN），两个标签页：

| 页签 | 数据来源 | 渲染 |
|---|---|---|
| 策略体检 | `POST /api/checkup` | Canvas 六维雷达 + 维度表 + 三段诊断 + 代码命中 |
| 每日监控 | `/api/daily/latest` `/api/daily/history` `/api/paper/nav` | KPI + Canvas 净值曲线 + 变化清单 + 历史表 |
| 风控仪表盘 | `/api/risk` | 大号动作徽章 + 触发原因 + 四道闸门表 + 成本容量 + 滚动样本外 |

雷达图中 N/A 维度画灰色轴（不画成 0 分），净值曲线用渐变面积图。

## 十、日频任务（M5）

```
watchlist.txt → 拉日线(0.5s/只) → 更新纸盘净值 → 重算六维
                                              ↓
                              与昨日结果比对 → 变化清单 → 落盘 + 通知
```

| 环节 | 说明 |
|---|---|
| 数据 | `akshare.stock_zh_a_hist` 前复权日线，实测 0.5 秒/只 |
| 净值 | 等权买入持有，首次用历史回补，之后逐日追加（不重复日期） |
| 体检 | 复用 `api_v2.run_checkup`，保证与页面口径一致 |
| 变化检测 | 结论 / 评级 / 得分(±5) / 单维(±10) / 回撤(±5pp) |
| 通知 | 邮件 + 飞书，**默认 dry-run 只打印不发** |
| 定时 | `--loop` 常驻，每天指定时刻执行 |

告警阈值常量：`SCORE_DELTA=5`、`DIM_DELTA=10`、`MDD_DELTA=0.05`。

**设计注意**：诊断降级（走模板）是常态，不进入变化提醒，否则每天都是噪音。

## 十一、测试

| 文件 | 覆盖 | 项数 |
|---|---|---|
| `tests/test_m1.py` | 夏普口径、比率保护、代码扫描、过拟合、滑点、切片、评审回归、PBO 诊断 | 155 |
| `tests/test_adjudicator.py` | 四类 NaN 裁决、评级门槛、封顶、适配器、鲁棒性 | 70 |
| `tests/test_m2.py` | 纯度指标、六维打分、雷达图、总报告、M1 联调 | 91 |
| `tests/test_contracts.py` | 数据契约、人工复核、极低仓位、端到端黄金样本 | 44 |
| `tests/test_m3.py` | 模板三段、定性分档、降级链、输出解析、提示词、永不抛异常 | 88 |
| `tests/test_m4.py` | CSV 解析、体检主流程、HTTP 接口、日更接口、风控接口、前端静态检查 | 117 |
| `tests/test_m5.py` | 自选池、净值增量、评估、变化检测、通知、定时、端到端 | 57 |
| `tests/test_m6_attribution.py` | 回本、成本、换手敏感性、归因、基准缓存、渲染 | 55 |
| `tests/test_m7_compound.py` | 增长、目标反推、通胀、成本代价、回撤代价 | 41 |
| `tests/test_m8_valuation.py` | 分位、统计、定性、缓存、报告 | 41 |
| `tests/test_m9_blackswan.py` | 选举日期、日历、尾部风险、情景回放、脆弱度 | 55 |
| `tests/test_m10_risk.py` | 数据体检、滚动样本外、成本容量、动态风控、端到端 | 79 |
| `tests/test_m11_theory.py` | EMH、MPT、CAPM、APT、法则、配置、凯利、时序、陷阱 | 86 |
| `tests/test_m12_portfolio.py` | 持仓读取、估值、权重、价格缓存、接口 | 47 |
| `tests/test_m13_fund.py` | 基金分类、净值指标、名录、基金持仓、接口 | 46 |
| `tests/test_m14_dca.py` | XIRR、台账、份额、定投vs一次性 | 44 |
| `tests/test_m15_calibration.py` | 换手口径、风险封顶、校准检出率 | 42 |
| `tests/test_m16_workflow.py` | 风控闭环、拥挤度覆盖率、CLI、定投接口、页面 | 65 |
| `tests/test_m17_strategy_runner.py` | 权重校验、代码执行、一体化体检、接口 | 40 |
| `tests/test_m18_datasource.py` | 列名归一、熔断、降级、接线、接口 | 41 |
| `tests/test_m19_robustness.py` | 扰动可复现、稳健区分、压力情景、接口 | 56 |
| `tests/test_m20_deploy.py` | 缓存统计/清理/安全边界、部署产物、CLI | 73 |
| `tests/test_m21_zzshare.py` | 代码转换、日线归一、估值、行业映射、降级、接口 | 59 |
| `tests/test_backtest.py` | 回测引擎（T+1/涨跌停/税费/退市核销） | 22 |

一键回归：

```bash
python tests/test_m1.py && python tests/test_adjudicator.py && \
python tests/test_m2.py && python tests/test_contracts.py && \
python tests/test_m3.py && python tests/test_m4.py && \
python tests/test_m5.py && python tests/test_backtest.py
```

---

## 十二、待建设

| 优先级 | 事项 | 前置条件 |
|---|---|---|
| P0 | 用真实策略批量体检，验证区分度 | 需要有 ≥20 个待检策略 |
| P1 | 六维权重敏感性校准 | 同上，需要策略样本 |
| P2 | 未来函数检测升级（AST 轻量数据流） | — |
| P2 | 知识库语义检索（向量库） | 知识库条目膨胀到数百条时 |

---

*本文件随模块更新同步维护。*
