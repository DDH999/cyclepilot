# QuantPilot 运维手册

> 系统已挂上 Windows 计划任务，每工作日自动运行。
> 本文档是你**几个月后回来看**时需要的全部信息。

---

## 一、它每天在做什么

```
15:30（工作日）
  ingest   补数据（最多 300 只/次，从上次中断处继续）
  health   6 项体检（数据新鲜度/入库失败率/审计错误/磁盘/存储/版本库）
  research 因子 IC + 参数级 Walk-forward（最多 200 只）
  report   写 reports/daily_YYYYMMDD.md
  notify   邮件发送日报（已配置 QQ 邮箱）
```

周日 20:00 额外跑一次**备份**（打包 `.data/` 到 `.data/backups/`，保留 7 份）。

---

## 二、计划任务

| 任务名 | 时间 | 命令 |
|---|---|---|
| `QuantPilot-Daily` | 工作日 15:30 | `daily_pipeline.py --once --notify` |
| `QuantPilot-Backup` | 周日 20:00 | `quantpilot.py ops --action backup --keep 7` |
| `QuantPilot-Smoke` | 手动（明年才到期） | `daily_pipeline.py --once --limit 0 --no-wf`（17 秒冒烟） |

**常用命令**：

```powershell
schtasks /query /tn "QuantPilot-Daily" /v /fo LIST   # 看状态
schtasks /run   /tn "QuantPilot-Smoke"               # 立即冒烟测试
schtasks /run   /tn "QuantPilot-Daily"               # 立即跑完整日更
schtasks /end   /tn "QuantPilot-Daily"               # 中止
schtasks /delete /tn "QuantPilot-Daily" /f           # 删除
```

> ⚠ 注册任务必须用 **PowerShell 脚本文件（UTF-8 BOM）**，不要用 `schtasks /create` 直接传中文路径——
> 中文用户名经命令行传递会变成 `????`，任务静默失效。参考 `.tmp/setup_tasks.ps1`。

---

## 三、日常检查（三条命令）

```powershell
# 1. 体检（6 项）
python backend\quantpilot.py ops --action health

# 2. 最近运行记录
python backend\quantpilot.py ops --action audit --limit 20

# 3. 数据现状
python backend\quantpilot.py status
```

**看今天的日报**：`reports\daily_YYYYMMDD.md`

---

## 四、出问题怎么查

### 症状：日报没生成

```powershell
schtasks /query /tn "QuantPilot-Daily" /v /fo LIST   # 看 Last Result（0=成功）
python backend\quantpilot.py ops --action audit --limit 30
```

常见原因：

| `Last Result` | 含义 | 处理 |
|---|---|---|
| `0` | 成功 | 检查报告目录写权限 |
| `1` | 脚本异常 | 看审计日志里的 `error` 字段 |
| `0x1` | 路径错误 | 检查 python 路径、脚本路径 |
| `0x41303` | 从未运行 | 正常（新建任务尚未到点） |

### 症状：数据不更新

```powershell
python backend\quantpilot.py ops --action health
```

`data_freshness` 报滞后 → 数据源限流。手动补：

```powershell
python backend\bulk_ingest.py --threaded --workers 3 --limit 200
```

### 症状：邮件收不到

配置文件 `config/mail.local.json`（QQ 邮箱授权码）。
测试：`python backend\quantpilot.py mail --test`

---

## 五、数据与备份

| 内容 | 位置 |
|---|---|
| 行情数据（Parquet） | `.data/parquet/<市场>/<年>/<代码>.parquet` |
| 审计日志 | `.data/logs/audit.jsonl` |
| 策略/因子版本 | `.data/versions/` |
| 备份 | `.data/backups/backup_YYYYmmdd_HHMMSS.zip` |
| 日报 | `reports/daily_YYYYMMDD.md` |

**恢复备份**：

```python
import opshealth as OH
OH.verify_backup(r".data\backups\backup_xxx.zip")   # 先校验
OH.restore(r".data\backups\backup_xxx.zip", r".data_restore")
```

---

## 六、关键结论（免得你重新踩一遍）

1. **策略没有正 alpha**。严格样本外 Walk-forward 年化 -8.11%、夏普 -0.499。
   换时点票池后策略跑输全市场等权 17.21pp。
2. **风险预算降低回撤但不改善风险调整后收益**（等权 9.21%/0.93 → 逆波动 6.18%/0.87）。
3. **前复权在长历史上会产生负价**（AAPL 1984 年 close=-10.75），已截断到首个正价区间。
4. **空仓看起来像低波动**：投资率 < 50% 的结果不可信，引擎会标记 `coverage_warning`。
5. **免费数据源给不了全市场全历史**：efinance/adata 限流、zzshare 429、baostock 慢且 800 行。
   Tushare 需要 120 积分才能用 `daily`/`stock_basic`。

---

## 七、系统边界（哪些还不是真的）

| 项 | 状态 |
|---|---|
| 交易执行（OMS/拆单/券商 API） | ❌ 未做 |
| 实盘下单 | ❌ 从未接过券商 |
| 多账户资金调度 | ❌ 未做（单账户纸盘） |
| 税务报表 | ❌ 未做（你没有使用场景） |
| 全市场数据 | ⚠️ 301 只（目标 5566） |
| 指数历史权重 | ❌ 缺（需 Tushare 2000 积分） |

**所有输出仅供研究参考，不构成投资建议。系统从未、也不会自行下单。**

---

## 八、想扩展时从哪改

| 想做什么 | 改哪里 |
|---|---|
| 换因子 | `backend/strategy_lab.py` 的 `build_scores` |
| 加交易日更步骤 | `backend/daily_pipeline.py` 的 `STEPS` + 新增 `step_xxx` |
| 加体检项 | `backend/opshealth.py` 加 `@check("name")` 装饰的函数 |
| 改调度时间 | 重跑 `.tmp/setup_tasks.ps1`（改里面的 `-At`） |
| 加数据源 | `backend/feed.py` 新增 `_src_xxx` 并注册到 `SOURCES` |
| 接入券商 | 新建 `backend/broker.py`，参考第 4 层设计（未实现） |

**改完务必跑**：

```powershell
python backend\preflight.py     # 2663 项测试，必须全绿
```
---

## 九、推送渠道（邮件 / 飞书 / 企业微信 / 个人微信）

配置文件：`config/notify.local.json`（也可用环境变量覆盖）

| 渠道 | 怎么配 | 推到哪 | 门槛 |
|---|---|---|---|
| `email` | 复用 `config/mail.local.json` | 邮箱 | 已配 |
| `feishu` | 飞书群 → 设置 → 群机器人 → 添加自定义机器人 → 复制 Webhook | 飞书群 | 1 分钟 |
| `wechat` | 企业微信群 → 右键 → 添加群机器人 → 复制 Webhook | 企业微信群 | 需企业微信 |
| `wechat_personal` | ✅ **已配置**（SCT47268****） | **个人微信** | 免费 5 条/天 |

```powershell
python backend
otify.py --status          # 看哪些渠道已配置
python backend
otify.py --test feishu     # 单独测一个渠道
QP_DRY_RUN=1 python backend
otify.py --all  # 干跑全部（不真发）
```

⚠ **`QP_DRY_RUN` 默认 0（真发）**。旧版本默认 1，导致"配好了却从没收到"。
想演练就显式设 `QP_DRY_RUN=1`。

⚠ 个人微信：微信没有开放个人号消息接口。合规路径只有 Server酱（公众号推送）
或企业微信应用消息（需管理员）。第三方库走逆向协议，封号风险高，不提供。
