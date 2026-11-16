# QuantPilot 部署手册

> **想快速本地部署？** 直接双击 `start.bat`，或看 [docs/部署手册.md](docs/部署手册.md)（一键脚本 / 缓存治理 / 备份 / 安全边界）。
> 本文是完整技术手册，内容较旧，保留备查。

配套文档：[STATUS.md](STATUS.md)（能信什么/不能信什么）、[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)（架构与数据流）

本手册分两部分：

- **第一部分｜体检平台（当前主推，M1–M4）** ← 只看这一节也能跑起来
- **第二部分｜策略研究系统（早期版本，保留可用）**

---

# 第一部分｜体检平台部署

## 1. 环境要求

| 项 | 要求 | 校验命令 |
|---|---|---|
| Python | 3.10+（实测 3.14.3） | `python --version` |
| 依赖 | fastapi / uvicorn / pydantic / numpy / pandas / scipy | 见下 |
| 磁盘 | < 50 MB（纯代码，无数据缓存依赖） | — |
| 网络 | **非必需**（大模型可选，缺失自动降级） | — |

```powershell
python -m pip install fastapi uvicorn pydantic numpy pandas scipy
```

## 2. 启动

```powershell
cd C:\Users\张泽华\Documents\deepseek-harness\default-workspace\quantpilot
python backend/api_v2.py
```

打开 **http://127.0.0.1:8421**

看到 `Uvicorn running on http://127.0.0.1:8421` 即成功。

## 3. 使用

1. 上传 **收益序列 CSV**（必填）：列名 `date,ret` 或 `date,nav`
2. 上传 **策略代码文件**（可选）：用于检测未来函数
3. 上传 **持仓权重 CSV** / **基准 CSV**（可选）：用于纯度与 beta 评估
4. 点「开始体检」

页面顶部有两个页签：

- **策略体检**：上传材料做单次体检
- **每日监控**：查看日更结果、净值曲线、变化记录，并可点「立即更新」触发一次真实日更

**答辩演示建议**：直接点「载入示例数据」，示例里含故意写的未来函数，
系统会当场判 REJECTED 并列出命中行号。

## 4. 大模型配置（可选）

```powershell
# 当前会话
$env:QP_LLM_KEY = "sk-你的key"

# 永久生效
[Environment]::SetEnvironmentVariable("QP_LLM_KEY", "sk-你的key", "User")
```

其他可选变量：

| 变量 | 默认 | 用途 |
|---|---|---|
| `QP_LLM_BASE` | `https://api.deepseek.com` | API 地址 |
| `QP_LLM_MODEL` | `deepseek-chat` | 模型名 |
| `QP_LLM_TIMEOUT` | `20` | 超时秒数 |
| `QP_PORT` | `8421` | 服务端口 |

未配置 Key 时**自动使用本地规则模板**，报告结构完全一致，只在右上角标注「本地模板（降级）」。

## 5. 断网演练（答辩前必做）

```powershell
$env:QP_LLM_DISABLE = "1"
python backend/api_v2.py
```

确认页面右上角显示「本地模板（降级）」、三段诊断完整。
**这是设计好的降级链路，不是故障**，答辩时可主动说明。

## 6. 一键自检

```powershell
python backend/preflight.py
```

会依次检查：Python 版本、依赖、模块文件、导入、纸盘产物、大模型配置、HTTP 路由，
最后跑完全部 8 套测试并汇总。**答辩前跑一次，全绿再上台。**

## 7. 测试

```powershell
python tests/test_m1.py
python tests/test_adjudicator.py
python tests/test_m2.py
python tests/test_contracts.py
python tests/test_m3.py
python tests/test_m4.py
```

全部应通过（合计 503 项；另有 `test_backtest.py` 22 项属于第二部分）。

## 8. 日频自动体检（M5）

```powershell
# 编辑自选池
notepad watchlist.txt        # 每行一个 6 位代码

# 跑一次
python backend/daily_job.py --once

# 每天 15:30 自动跑（窗口别关）
python backend/daily_job.py --loop

# 真正发送通知
$env:QP_NOTIFY_DRYRUN = "0"
python backend/daily_job.py --once --notify
```

产物：

| 路径 | 内容 |
|---|---|
| `reports/paper_nav.csv` | 纸盘净值曲线（逐日追加） |
| `reports/daily/YYYY-MM-DD.json` | 当日完整体检结果 |
| `reports/daily/latest.json` | 最近一次结果（供比对） |

通知渠道（可选，不配就只打印）：

| 变量 | 说明 |
|---|---|
| `QP_SMTP_HOST` / `QP_SMTP_PORT` / `QP_SMTP_USER` / `QP_SMTP_PASS` / `QP_MAIL_TO` | 邮件 |
| `QP_FEISHU_WEBHOOK` | 飞书机器人 |
| `QP_NOTIFY_DRYRUN` | 默认 `1`（不发），设 `0` 才真发 |

**注意**：默认 dry-run。想真发必须显式把 `QP_NOTIFY_DRYRUN` 设为 `0`。

## 9. 接口速查

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 体检页面 |
| GET | `/api/health` | 健康检查 |
| GET | `/api/attribution` | 收益归因 + 成本核算 |
| POST | `/api/checkup` | 提交体检（JSON） |

```powershell
# 命令行验证
curl http://127.0.0.1:8421/api/health
```

## 10. 常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| 页面提示「未连接到后端」 | 服务没启动 | 先跑 `python backend/api_v2.py` |
| 提示「收益序列有效样本仅 N 条」 | CSV 行数不足 30 | 提供更长区间 |
| 诊断显示「本地模板（降级）」 | 无 Key 或网络不通 | 正常降级，可继续使用 |
| 纯度维度显示 N/A | 没传权重文件 | 上传持仓权重 CSV |
| 抗压显示 N/A | 收益区间未覆盖 2024 年初 | 正常豁免，报告已标注 |
| 端口被占用 | 8421 已被占用 | `$env:QP_PORT="8422"` |
| 日频任务拉取失败 | 网络或代码停牌 | 日志会列出失败标的，其余继续 |
| 日频任务报「净值仅 N 天」 | 首次运行历史不足 | 等积累，或调大 `--days` |

---

# 第二部分｜策略研究系统（早期版本）

**冻结版 2026-11-03** ｜ 研究/回测系统 ｜ 纸盘模式，不接真实资金
配套文档：`STATUS.md`（能信什么/不能信什么）、`README.md`（技术细节）

---

## 0. 先选场景

| 场景 | 你的情况 | 跳到 |
|---|---|---|
| **A** | 就在这台机器、这个目录（数据已齐：2150 只 / 47 季度 / 788 MB 缓存） | 第 2 节，**3 条命令** |
| **B** | 同机但要换个干净目录 | 第 3 节（复制代码 + 重跑数据） |
| **C** | 全新机器（没装过 Python 依赖） | 第 1 节 → 第 3 节 |

---

## 1. 环境要求（场景 C 必读）

| 项 | 要求 | 校验命令 |
|---|---|---|
| Python | **3.10+**（实测 3.14.3） | `python --version` |
| 内存 | ≥ 4 GB（2066 只全量回测峰值约 2 GB） | — |
| 磁盘 | **≥ 2 GB**（行情缓存 788 MB + 产物） | 见第 6 节 |
| 网络 | 可访问东财 / 新浪 / 腾讯 / 中证指数 | `python -c "import akshare;print(akshare.__version__)"` |

### 1.1 安装依赖

```powershell
python -m pip install fastapi uvicorn scikit-learn pandas numpy scipy matplotlib akshare
```

### 1.2 校验（必须通过再往下）

```powershell
python -c "import fastapi,sklearn,pandas,numpy,scipy,matplotlib,akshare;print('依赖 OK')"
```

预期输出：`依赖 OK`

> **Windows 提示**：PowerShell 里中文乱码时，先执行 `$env:PYTHONIOENCODING="utf-8"`。

---

## 2. 场景 A：现有目录，3 条命令启动

```powershell
cd C:\Users\张泽华\Documents\deepseek-harness\default-workspace\quantpilot

# ① 跑一遍全部计算（33 秒）
python backend/run_all.py

# ② 验证引擎规则（22 项测试，2 秒）
python tests/test_backtest.py

# ③ 启动控制台
python backend/app.py
```

**③ 执行后打开浏览器** → http://127.0.0.1:8420

### 预期输出（照此比对）

① 应该看到：

```
[✓] 基本面回测       24.1s
[✓] 诚实检验         4.4s
[✓] 回测图表         3.4s
[✓] 研报生成         1.1s

完成 4/4  总耗时 33.0s
```

② 最后一行应该是 `通过 22 / 22` 和 `全部通过 ✓`

③ 看到 `Uvicorn running on http://127.0.0.1:8420` 即成功。

---

## 3. 场景 B/C：从零完整部署

**总耗时约 40~50 分钟，绝大部分是下载数据。** 按顺序执行，每步都有校验点。

### 步骤 1｜复制代码（仅场景 B）

```powershell
robocopy "C:\Users\张泽华\Documents\deepseek-harness\default-workspace\quantpilot" `
         "D:\quantpilot" /E /XD .cache reports .tmp /XF *.pkl *.png
cd D:\quantpilot
```

> 场景 C（全新机器）跳过这步。

### 步骤 2｜下载行情（约 5 分钟）

```powershell
python backend/universe.py csi800
```

**校验点**：输出类似 `成功 795 / 失败 5 / 用时 265.9s`

失败几只（退市、改名）属正常。缓存落在 `.cache/`。

### 步骤 3｜扩展票池到中证1000（约 7 分钟，可选）

```powershell
python backend/universe.py csi1000
```

**校验点**：`成功 991 / 失败 5`

> 不做也能跑，只是票池小一半；做了才覆盖中小盘。

### 步骤 4｜行业分类（约 3 分钟）

```powershell
python backend/sector.py
```

**校验点**：最后打印 `完成: N/N 有行业` + 行业分布表

### 步骤 5｜退市股（10~15 分钟，**不建议跳过**）

```powershell
python backend/delisted.py
```

**校验点**：`待下载 267` → 最终 `完成 ok=253 fail=14`

> 这步修幸存者偏差。跳过会让回测收益虚高约 **9 个百分点**。
> 中断了直接重跑，已下载的会跳过。

### 步骤 6｜业绩报表（约 13 分钟）

```powershell
python backend/fundamentals.py
```

**校验点**：`因子面板 358,679 行 11724 只`

### 步骤 7｜第一次完整计算（约 2 分钟）

```powershell
python backend/run_strategy.py --top 30 --vt 0.15 --stop-loss 0.15 --max-per-ind 3 --save
python backend/combo_strategy.py
python backend/validate.py --nav combo_nav.csv --trials 36
```

**校验点**：
- 第 1 条末尾出现 `[保存] backtest_results.json + nav_curves.csv`
- 第 2 条出现 `全样本 年化 7.4% Sharpe 0.76 ...`
- 第 3 条出现 `DSR=0.793`

### 步骤 8｜因子诊断与敏感性（约 8 分钟）

```powershell
python backend/factor_lab.py        # 451 秒
python backend/sensitivity.py       # 14 秒
```

**校验点**：`保存 reports/factor_lab.json (36 因子, FDR 通过 7)`

### 步骤 9｜图表与研报

```powershell
python backend/bt_charts.py
python backend/report_factory.py
```

**校验点**：`reports/charts/` 出现 50+ 张 PNG；研报在 `reports/` 下

### 步骤 10｜验证 + 启动

```powershell
python tests/test_backtest.py
python backend/app.py
```

---

## 4. 验证清单（部署完逐项打勾）

| # | 检查 | 命令 | 通过标准 |
|---|---|---|---|
| 1 | 依赖完整 | `python -c "import fastapi,sklearn,akshare;print('OK')"` | 打印 OK |
| 2 | 行情缓存 | 看 `.cache/*_cn_*.csv` 数量 | ≥ 795 |
| 3 | 行业分类 | `.cache/industries.csv` 行数 | ≥ 795 |
| 4 | 业绩报表 | `.cache/fund/` 下 csv 数量 | = 47 |
| 5 | 引擎规则 | `python tests/test_backtest.py` | `22 / 22` |
| 6 | 全流程 | `python backend/run_all.py` | `完成 4/4` |
| 7 | 控制台 | 浏览器打开 `:8420` | 页面有 6 个区块 |
| 8 | API | 打开 `/api/gate` | 返回 JSON |

---

## 5. 日常怎么用

### 5.1 只看结果

```powershell
python backend/app.py
```

打开 http://127.0.0.1:8420，重点三块：

| 区块 | 看什么 |
|---|---|
| **门禁判定** | 是否允许进入纸盘（不过会明确说"拒绝部署"） |
| **策略回测** | 14 个策略的收益/Sharpe/回撤 |
| **诚实检验** | PBO / DSR / 样本外，**这块最重要** |

### 5.2 改参数重跑

```powershell
# 更保守（降回撤）
python backend/run_strategy.py --top 20 --vt 0.10 --stop-loss 0.10 --max-per-ind 2 --save

# 每改一次必须重跑检验，否则你就是在过拟合
python backend/validate.py --nav nav_curves.csv
```

### 5.3 参数含义

| 参数 | 默认 | 调大的后果 |
|---|---|---|
| `--top 30` | 30 | 越分散，波动越小，收益越平庸 |
| `--vt 0.15` | 0.15 | 目标年化波动，越高仓位越重、回撤越大 |
| `--stop-loss 0.15` | 0.15 | 个股止损线 |
| `--max-per-ind 3` | 3 | 单行业最多几只 |
| `--dd-brake 0.15` | 0.15 | 组合回撤制动阈值 |

### 5.4 加新因子

1. 在 `backend/factors.py` 加一个函数，返回 `date × code` 的 DataFrame
2. 跑 `python backend/factor_lab.py`，看 IC / FDR 是否通过
3. **只有 FDR<0.05 的因子才能进组合**，否则就是在挖噪声

---

## 6. 数据与磁盘

| 内容 | 大小 | 能否删 |
|---|---|---|
| `.cache/*_cn_*.csv` | 约 700 MB | 删了要重下载（5~7 分钟） |
| `.cache/fund/` | 51 MB | 删了要重下载（13 分钟） |
| `.cache/panel_*.pkl` | 约 150 MB | **可随时删**，下次自动重建（35 秒） |
| `.cache/industries.csv` | < 1 MB | 可删，重下载 3 分钟 |
| `reports/charts/*.png` | 约 5 MB | 可删，重生成 4 秒 |
| `.tmp/` | 很小 | 调试脚本，可随时删 |

**安全清理（释放约 150 MB）**：

```powershell
Remove-Item .cache\panel_*.pkl -Force
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

---

## 7. 故障排查（都是本项目实测遇到过的）

| 现象 | 原因 | 解决 |
|---|---|---|
| `JSONDecodeError` / `RemoteDisconnected` | 东财源限流 | 自动降级到新浪/腾讯；全失败就等 10 分钟重试 |
| `OpenBLAS error: Memory allocation failed` | 同时持有多份全历史矩阵 | 已内置 `OPENBLAS_NUM_THREADS=1`；仍报错则减小票池 `--limit 500` |
| `ModuleNotFoundError: sklearn.inference` | 应为 `sklearn.inspection` | 已修正 |
| `PermissionError: [WinError 5]`（joblib） | 沙箱禁止命名管道 | `evals.py` 已内置串行补丁 |
| 控制台打开是空白 | 后端未启动 / 端口占用 | 确认 `app.py` 在跑；改端口 |
| 中文图表出现方块 | 字体缺失 | 确认有"微软雅黑"；Linux 装 `fonts-wqy-zenhei` |
| `strategy family empty` | 票池太小或信号全空 | 增大票池，检查 `.cache` |
| 研报里数字是 `—` | 对应 JSON 未生成 | 按第 3 节顺序重跑 |
| 推送返回 `ok:false` | 环境变量未生效 | PowerShell 用 `$env:QP_SMTP_HOST="..."` |

---

## 8. 后台常驻（可选）

```powershell
# 最小化窗口运行
Start-Process python -ArgumentList "backend\app.py" -WindowStyle Hidden
```

开机自启：把下面存成 `start_quantpilot.bat`，放进
`C:\Users\<你>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\`

```bat
@echo off
cd /d C:\Users\张泽华\Documents\deepseek-harness\default-workspace\quantpilot
start /min python backend\app.py
```

**停止**：任务管理器结束 python 进程。

---

## 9. 边界（部署完务必读）

这套系统**能**告诉你：

- 你的策略在 A 股真实规则下（T+1、涨跌停、税费、手数）能不能活
- 你的因子是不是噪声（Newey-West t + FDR 校正）
- 你的策略是不是过拟合（PBO / DSR / 样本外）

这套系统**不能**告诉你：

- 明天买什么（不产生交易信号）
- 能赚多少（已验证组合 DSR 0.793，不显著）
- 跨周期是否有效（**这项没验证完，见 STATUS.md 3.1**）

**红线**：不要因为回测好看就上真钱。本系统没有实现任何实盘下单能力。

---

## 10. 一句话总结

```powershell
cd C:\Users\张泽华\Documents\deepseek-harness\default-workspace\quantpilot
python backend/run_all.py     # 33 秒跑完
python backend/app.py         # 打开 http://127.0.0.1:8420
```

就这两条，其余都是可选。

---

*QuantPilot · 2026-11-03 · 纸盘模式 · 不构成投资建议*
