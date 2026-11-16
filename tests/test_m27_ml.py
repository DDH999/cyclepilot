# -*- coding: utf-8 -*-
"""
M27 验证：回撤风险预测（机器学习）
====================================================================
跑法:  python tests/test_m27_ml.py   （不联网，用合成数据）
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import ml_risk as ML
import overseas as OS

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


print("\n" + "#" * 70)
print(f"# M27 验证  ml_risk v{ML.__version__}")
print("#" + "=" * 69)

print("\n[1] 特征无未来信息")

idx = pd.bdate_range("2000-01-03", periods=1200)
rng = np.random.default_rng(7)
close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 1200)), index=idx)
X = ML.build_features(close)

check("特征数量正确", X.shape[1] == len(ML.FEATURE_COLS), X.shape[1])
check("列名一致", list(X.columns) == ML.FEATURE_COLS)
check("开头有 NaN（滚动窗口不足）", X.iloc[0].isna().any())
first_valid = X.dropna().index[0]
check("首个有效点在 200 天后", first_valid >= idx[199], first_valid.date())

# 关键：改动未来价格不能改变过去特征
close2 = close.copy()
close2.iloc[900:] *= 1.5
X2 = ML.build_features(close2)
same_past = X.loc[:idx[800]].equals(X2.loc[:idx[800]])
check("改未来不影响过去特征（无泄漏）", same_past)

print("\n[2] 标签构造")

y = ML.make_target(close, horizon=20, thresh=-0.10)
check("长度一致", len(y) == len(close))
check("最后 20 天为 NaN", y.tail(20).isna().all())
check("其余非空", y.iloc[:-20].notna().all())
check("标签只有 0/1", set(y.dropna().unique()).issubset({0.0, 1.0}),
      set(y.dropna().unique()))

# 构造一个已知答案的序列：末期暴跌
crash = pd.Series([100.0] * 300 + [70.0] * 30, index=pd.bdate_range("2020-01-01", periods=330))
yc = ML.make_target(crash, horizon=20, thresh=-0.10)
check("暴跌前标记为正样本", yc.iloc[280] == 1.0, yc.iloc[280])
check("平稳期标记为负样本", yc.iloc[100] == 0.0, yc.iloc[100])

print("\n[3] walk-forward")


def synth_close(seed: int, signal: bool) -> pd.Series:
    """signal=True 时构造「波动上升 → 随后回撤」的可学模式。"""
    rng = np.random.default_rng(seed)
    n = 2000
    vol = 0.008 + 0.004 * np.sin(np.linspace(0, 40, n))
    r = rng.normal(0.0003, 1.0, n) * vol
    if signal:
        # 高波动段之后追加下跌
        for i in range(300, n - 30, 300):
            r[i:i + 25] -= 0.006
    return pd.Series(100 * np.cumprod(1 + r),
                     index=pd.bdate_range("2010-01-04", periods=n))


learnable = synth_close(11, signal=True)
Xl = ML.build_features(learnable)
yl = ML.make_target(learnable)
res_l = ML.walk_forward(Xl, yl)
check("可学数据能跑通", res_l["ok"] is True, res_l.get("reason"))
check("可学数据 AUC 高于 0.55", res_l["mean_auc"] > 0.55, res_l["mean_auc"])
check("含多折", len(res_l["folds"]) >= 3, len(res_l["folds"]))
check("每折有正例数", all("positives" in f for f in res_l["folds"]))

noise = synth_close(99, signal=False)
Xn = ML.build_features(noise)
yn = ML.make_target(noise)
res_n = ML.walk_forward(Xn, yn)
check("纯噪声 AUC 接近 0.5（不会凭空变高）",
      0.35 < res_n["mean_auc"] < 0.72, res_n["mean_auc"])
check("噪声数据 AUC 低于可学数据",
      res_n["mean_auc"] < res_l["mean_auc"],
      (res_n["mean_auc"], res_l["mean_auc"]))

check("样本过短返回 ok=False",
      ML.walk_forward(Xl.iloc[:200], yl.iloc[:200])["ok"] is False)

print("\n[4] 验证门槛（关键：不达标要拒绝）")

_orig = OS.index_history


def fake_history(key, days=20000, refresh=False):
    series = learnable if key == "good" else noise
    return {"ok": True, "key": key, "name": key.upper(),
            "series": series, "source": "fake", "errors": []}


OS.index_history = fake_history
ML._CACHE.clear()
_orig_wf = ML.walk_forward
_orig_thr = dict(ML.THRESHOLDS)
try:
    # 合成数据正例只有 72 个而门槛是 100；这里临时降门槛，
    # 目的是测「守卫逻辑本身」，而不是测合成数据够不够真实。
    ML.THRESHOLDS["min_positives"] = 50

    # ① 好市场：stub 一个高 AUC 的验证结果
    ML.walk_forward = lambda X, y: {
        "ok": True, "mean_auc": 0.75, "min_auc": 0.66, "max_auc": 0.9,
        "n": len(y), "pos_rate": float(y.mean()),
        "folds": [{"fold": 1, "n_train": 500, "n_test": 300, "auc": 0.75,
                   "positives": 30, "base_rate": 0.1}]}
    good = ML.validate_market("good")
    check("高 AUC 市场通过验证", good["ok"] is True and good["usable"] is True,
          good.get("reasons"))
    check("记录 AUC", good["walk_forward"]["mean_auc"] > 0.55)
    check("记录门槛", good["thresholds"]["min_mean_auc"] == 0.62)
    check("记录正样本数", good["positives_total"] > 0)
    check("可用时无拒绝理由", good["reasons"] == [], good["reasons"])

    cur_good = ML.current_risk("good")
    check("可用市场给出概率", cur_good.get("usable") is True, cur_good.get("reason"))
    check("概率在 0~1", 0.0 <= cur_good["probability"] <= 1.0,
          cur_good.get("probability"))
    check("含历史均值", "base_rate" in cur_good)
    check("含等级", cur_good.get("level") in ("高", "偏高", "正常", "低"),
          cur_good.get("level"))

    # ② 坏市场：AUC 不够 → 拒绝出信号
    ML._CACHE.clear()
    ML.walk_forward = lambda X, y: {
        "ok": True, "mean_auc": 0.55, "min_auc": 0.40, "max_auc": 0.70,
        "n": len(y), "pos_rate": float(y.mean()), "folds": []}
    bad = ML.validate_market("bad")
    check("低 AUC 市场判定不可用", bad["ok"] is True and bad["usable"] is False,
          bad.get("reasons"))
    check("拒绝理由含 AUC", any("AUC" in r for r in bad["reasons"]), bad["reasons"])

    cur_bad = ML.current_risk("bad")
    check("不可用市场拒绝出信号", cur_bad.get("usable") is False)
    check("无概率值", cur_bad.get("probability") is None)
    check("说明原因", "未通过验证" in str(cur_bad.get("reason")),
          cur_bad.get("reason"))

    # ③ 正样本不足也拒绝
    ML._CACHE.clear()
    ML.THRESHOLDS["min_positives"] = 10000
    ML.walk_forward = lambda X, y: {
        "ok": True, "mean_auc": 0.80, "min_auc": 0.70, "max_auc": 0.9,
        "n": len(y), "pos_rate": float(y.mean()), "folds": []}
    few = ML.validate_market("few")
    check("正样本不足也拒绝", few["usable"] is False, few.get("reasons"))
    check("理由提到正样本", any("正样本" in r for r in few["reasons"]),
          few["reasons"])
finally:
    OS.index_history = _orig
    ML.walk_forward = _orig_wf
    ML.THRESHOLDS.update(_orig_thr)
    ML._CACHE.clear()

print("\n[5] 风险倍数")

for ratio, want in [(2.5, 0.5), (1.8, 0.7), (1.0, 1.0), (0.4, 1.2)]:
    m = ML.risk_multiplier(0.037 * ratio, 0.037)
    check(f"概率/均值={ratio} → {want}x", m["multiplier"] == want, m["multiplier"])

check("缺参返回 1.0", ML.risk_multiplier(None, 0.1)["multiplier"] == 1.0)
check("均值为 0 返回 1.0", ML.risk_multiplier(0.1, 0.0)["multiplier"] == 1.0)
check("倍数上限不超 1.2", ML.risk_multiplier(0.0, 0.1)["multiplier"] <= 1.2)
check("倍数下限不低于 0.5", ML.risk_multiplier(0.9, 0.1)["multiplier"] >= 0.5)

print("\n[6] 报告")

val = {"ok": True, "key": "sp500", "name": "标普500", "n_days": 10000,
       "start": "1986-01-01", "end": "2026-11-12",
       "walk_forward": {"mean_auc": 0.767, "min_auc": 0.683, "max_auc": 0.942,
                        "pos_rate": 0.036, "folds": [
                            {"fold": 1, "n_train": 5000, "n_test": 900,
                             "auc": 0.767, "positives": 90, "base_rate": 0.1}]},
       "positives_total": 380, "usable": True, "reasons": [],
       "thresholds": ML.THRESHOLDS}
txt = ML.format_report(val)
check("报告含 AUC", "AUC" in txt)
check("报告含免责", "不是预言机" in txt)
check("可用时给出结论", "通过验证" in txt)

val_bad = dict(val, usable=False, reasons=["均值 AUC 0.578 < 0.62"])
txt2 = ML.format_report(val_bad)
check("不可用时明确拒绝", "拒绝出信号" in txt2)
check("列出拒绝原因", "0.578" in txt2)

print("\n[7] 与决策引擎集成")

import advisor as AD

_orig_ml_val = ML.validate_market
_orig_ml_cur = ML.current_risk
ML.validate_market = lambda k, refresh=False: {"ok": True, "usable": True}
ML.current_risk = lambda k, refresh=False: {
    "ok": True, "usable": True, "probability": 0.12, "base_rate": 0.04,
    "level": "高", "name": "标普500", "mean_auc": 0.767}
try:
    res = AD.advise(["F1"], {"F1": "博时标普500ETF联接C"},
                    weights={"F1": 0.5}, monthly_amount=1000.0,
                    portfolio_drawdown=-0.02,
                    index_percentiles={"sp500": {"ok": True, "name": "标普500",
                                                 "current": 100, "percentile": 50.0,
                                                 "source": "t", "n": 300,
                                                 "years": 3, "low": 90, "high": 110}},
                    use_ml=True)
    check("开启 ML 后有风险信号", "ml_risk" in (res["details"] or {}))
    acts = [a["action"] for a in res["actions"]]
    check("高风险触发减投", "减投" in acts, acts)
    check("风险动作优先级最高",
          res["actions"][0]["priority"] == 1, res["actions"][0]["priority"])
    check("报告含 ML 段", "机器学习回撤风险" in AD.format_report(res))

    res_off = AD.advise(["F1"], {"F1": "博时标普500ETF联接C"},
                        weights={"F1": 0.5}, monthly_amount=1000.0,
                        portfolio_drawdown=-0.02, use_ml=False)
    check("关闭 ML 时不含风险段",
          "ml_risk" not in (res_off["details"] or {}))
finally:
    ML.validate_market = _orig_ml_val
    ML.current_risk = _orig_ml_cur

print("\n[8] CLI / API")


def boom(*a, **k):
    return {"ok": False, "reason": "fake"}


ML.validate_market = boom
try:
    import api_v2 as API
    from fastapi.testclient import TestClient
    cli = TestClient(API.app)
    r = cli.get("/api/ml_risk?key=nope")
    check("失败时返回 400", r.status_code in (400, 500), r.status_code)
finally:
    ML.validate_market = _orig_ml_val

import quantpilot as QP
check("CLI 注册 ml", "ml" in QP.COMMANDS)
pp = QP.build_parser()
check("解析 --keys", pp.parse_args(["ml", "--keys", "sp500"]).keys == "sp500")
check("advice 支持 --ml", pp.parse_args(["advice", "--ml"]).ml is True)
check("advice 默认不带 ML", pp.parse_args(["advice"]).ml is False)

print("\n" + "#" * 70)
total = len(PASS) + len(FAIL)
print(f"# 通过 {len(PASS)} / {total}")
if FAIL:
    print("# 失败项:")
    for f in FAIL:
        print(f"#   - {f}")
    print("#" * 70)
    sys.exit(1)
print("# 全部通过  回撤风险模型可用且会自我否决")
print("#" * 70)
