# AI 量化『知识与信号工坊』

**范式**：规则进模型、事实进 RAG、证据先行  
**架构**：学生-老师 / Bronze→Silver→Gold→Signals  
**KPI**：CCR≥92%、ESR≥88%、RAR≥97%、Hit@5≥0.85、P95(热/冷)≤3s/6s

## 1. 目录与节拍
- `raw_docs → canon → chunks → gold → signals`
- 学生：白天检索/抽取/索引/健康面板（见 Actions: student-daily）
- 老师：夜间蒸馏/难例回放/蓝绿发布（Actions: teacher-nightly）

## 2. Schema & 口径
- 事件-因子统一 Schema：`schemas/event_factor.schema.json`
- 时间四元组：occur/visible/ingested/factorized（禁止时间穿越）

## 3. 质量与闸门
- 检索门：Hit@5、QPS、延迟、来源多样性
- 抽取门：CCR/ESR/RAR、字段缺失率、单位/数值合理性
- 证据门：KP-ID + URL + 摘要片段 + 时间四元组
- 可解释率 ≥ 90%

## 4. 本地手动
```powershell
# 学生健康面板
pwsh -File ops/health/alignment/alignment_daily_check.ps1 `
  -Root $PWD -EnvName repo311 -ActiveGlob 'chunks\*.active.ndjson' `
  -LatestLink -Diag
