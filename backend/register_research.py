# -*- coding: utf-8 -*-
"""把 low_vol 因子与策略及其 Walk-forward 结果登记进版本库。"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import versions as V

FACTOR_CODE = '''# 低波动因子（strategy_lab.build_scores 摘录）
# 约定：得分越高越好，组合层永远 nlargest
r = CL.pct_change(fill_method=None)          # 日收益矩阵 date×code
s = -r.rolling(lookback).std(ddof=0)         # 波动越低分越高
# CL 只含 <= 当日 的收盘价，无前视
'''

STRATEGY_CODE = '''# 低波动 topN 月调仓 —— 严格引擎 + 参数级 Walk-forward
import param_walkforward as PW, backtest_strict as BS, strategy_lab as SL

panel = PW.load_panel("cn", min_rows=1500)   # 63 只，历史 >= 1500 行
res = PW.run_walkforward(
    panel, factor="low_vol",
    n_folds=4, test_days=250, min_train_days=750,
    start_after="2016-01-01",
    grid={"lookback": [10, 20, 40], "top_n": [5, 10, 20]},
)
# 每折：IS 选参(夏普最大) → 紧邻 OOS 验证，OOS 曲线独立起算后拼接
'''

PARAMS = {
    "factor": "low_vol",
    "lookback_grid": [10, 20, 40],
    "top_n_grid": [5, 10, 20],
    "rebalance_days": 20,
    "n_folds": 4,
    "test_days": 250,
    "min_train_days": 750,
    "start_after": "2016-01-01",
    "capital": 1_000_000,
    "T_plus_1": True,
    "max_part": 0.05,
    "slippage_bps": 5.0,
    "commission": 0.00025,
    "stamp_tax_sell": 0.0005,
}

METRICS = {
    "oos_annual": -0.0811,
    "oos_sharpe": -0.499,
    "oos_max_dd": -0.3969,
    "n_folds": 4,
    "is_oos_decay_pp": 9.99,
    "param_stability": 0.25,
    "fold_details": [
        {"fold": 1, "test": "2019-01-29~2020-02-13",
         "params": "lb40/n20", "is_annual": -3.96, "oos_annual": -15.61},
        {"fold": 2, "test": "2020-02-14~2021-02-23",
         "params": "lb10/n5", "is_annual": -0.69, "oos_annual": -5.22},
        {"fold": 3, "test": "2021-02-24~2022-03-04",
         "params": "lb40/n5", "is_annual": 2.16, "oos_annual": 1.69},
        {"fold": 4, "test": "2022-03-07~2023-03-15",
         "params": "lb20/n5", "is_annual": 10.99, "oos_annual": -12.31},
    ],
}

RULES = {
    "市场": "A股",
    "交易制度": "T+1，当日买入次日可卖",
    "涨跌停": "逐日 isST 判定 5%，创业板/科创板 20%，北交所 30%",
    "成交价": "次日开盘价",
    "成本": "佣金 2.5bp(最低5元) + 印花税 5bp(卖) + 过户 0.1bp "
            "+ 滑点 5bp + 冲击 0.002*sqrt(part)",
    "流动性": "单日成交不超过当日成交额 5%",
    "退市": "分档核销（连跌停/无成交→0；否则 min(30日均价,最后收盘)）",
    "样本外": "参数只用训练段选，OOS 紧邻其后且独立起算",
}


def main() -> int:
    print("=== 因子入库 ===")
    f = V.create("factor", "低波动因子 low_vol（N 日已实现波动率取负）",
                 "低波动异象：波动越低、风险调整后收益越高；"
                 "杠杆约束与彩票偏好解释", "low_vol")
    print(" ", f)
    fid = f["id"]
    r = V.save("factor", fid, FACTOR_CODE, {"lookback": 20},
               "v1 首版：-rolling_std(20)，仅用当日及之前数据", parent=0)
    print(" ", r)
    V.record_run("factor", fid, 1, METRICS, PARAMS,
                 "2019-01-29", "2023-03-15", "A股 63 只（历史>=1500行）",
                 "Walk-forward 样本外")

    print("\n=== 策略入库 ===")
    s = V.create("strategy", "低波动 topN 月调仓（严格引擎 + 参数级 WF）",
                 "低波动选股在严格交易约束下的样本外表现", "low_vol")
    print(" ", s)
    sid = s["id"]
    r2 = V.save("strategy", sid, STRATEGY_CODE, PARAMS,
                "v1 首版：low_vol topN 20日调仓，3年训练/1年测试滚动", parent=0)
    print(" ", r2)
    V.record_run("strategy", sid, 1, METRICS,
                 {"config": PARAMS, "rules": RULES},
                 "2019-01-29", "2023-03-15", "A股 63 只",
                 "Walk-forward 样本外，结论：无正 alpha")

    print("\n" + V.format_table("factor"))
    print(V.format_table("strategy"))
    lt = V.latest("strategy", sid)
    print(f"策略最新版本 v{lt['version']}  sha1={lt['code_sha1']}")
    print(f"运行归档 {len(V.runs_of('strategy', sid))} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
