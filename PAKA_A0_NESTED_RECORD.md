# R10 PAKA $a_0$ nested 敏感性分析记录（投稿前硬门 2）

日期：2026-10-31。脚本：`rounds/R10_RESS_reframe_2026-10-30/paka_a0_nested_r10.py`（modes: smoke / run / check）。
输出：`outputs/paka_a0_per_draw.csv.gz`（811,800 行）、`paka_a0_info.csv`、`paka_a0_regret_summary.csv`、`paka_a0_kshot_summary.csv`。
动机：R09 正文称 "$a_0$ 在 {0,3,6} 上 immaterial（worst shortfall 变化 <0.01）"，但原 $a_0$ 比较在同目标子集上完成，存在选择偏差；本分析把 $a_0$ 敏感性放入 programme-level 框架重测，并新增只用训练库选 $a_0$ 的 nested 变体。

## 设计

- 网格：15 rung × 全 target × S∈{3,6,12,24}（各 10 个 subset 种子）+ all，共 14,760 任务，~20 min on 24 cores。
- 每个任务复现 R06 `sc_task`（同 subset/draw 种子）⇒ `PAKA-S3` 必须逐行等于冻结 `PAKA-S`。
- 变体：`PAKA-S0/S3/S6`（固定 $a_0$）、`PAKA-NESTS`（用训练库内层 LOPO 联合对数密度选 $a_0$）、`PAKA-NESTR`（用内层 LOPO 逐点 RMSE 选 $a_0$）。Nested 变体的 $a_0$ 选择**只接触训练库**，不接触留出 programme。
- 基线 regret 分母：冻结 `ladder_per_draw.csv.gz` 中三臂（KA-FIXED / KA-RE / DATA-RE）逐 task 最优。

## 校验

- **身份校验通过**：`PAKA-S3` vs 冻结 `PAKA-S`，162,360 个共享行 max|Δ|=4.4e-16（机器精度）。
- Nested 选择分布：score 规则 91.5% 任务选 $a_0=0$；RMSE 规则 0/3/6 = 44/24/32%。

## 结果

| 变体 | worst shortfall (k=0, 跨 rung) | mean shortfall |
|---|---|---|
| PAKA-S3（默认） | **0.112** | **0.048** |
| PAKA-S0 | 0.122 | 0.054 |
| PAKA-S6 | 0.140 | 0.054 |
| PAKA-NESTR（nested RMSE 选） | 0.115 | 0.050 |
| PAKA-NESTS（nested score 选） | 0.122 | 0.054 |

- $a_0\{0,3,6\}$ 的 worst shortfall 极差 0.028（S6−S3）；逐 rung spread max 0.062、mean 0.034。
- **原 "<0.01 immaterial" 主张不成立**，需要按上表改写；改写后主张变为"有界但可见"（bounded but visible）。
- $a_0=3$ 在全局 worst-case 上最优；但 BEAM rung 上 $a_0=0$ 更优（0.077–0.078 vs 0.090–0.112），说明最优 $a_0$ 依 rung 而异 → nested 逐任务选择的合理性。
- **k-shot 更新增益（全文 headline）基本不敏感**：S=all 时各域 k=0→3 RMSE 降幅在 $a_0$ 间极差 ≤1.85pp（CARB 0.28 / DEEP 0.42 / DEEP-WB 1.24 / FRP 0.34 / SFRC 1.85）→ 正文 "under 2 percentage points" 有据。
- **NEST-RMSE 只用训练库即可逼近最优固定值**（0.115 vs 0.112，差 0.003）；NEST-SCORE（对数分数选）偏向 $a_0=0$，worst-case 反而差——对数分数与 RMSE regret 不对齐，正文默认仍用 $a_0=3$ 固定值，nested 作为证据说明选择无需留出 programme 信息。

## 判定

- **PAKA 保留为主线组件，不降级**。headline k-shot 增益对 $a_0$ 稳定；nested 分析把"选择用了留出结果"的质疑正面消掉。
- 正文五处同步改写（abstract / intro / methods / results / limitations），新增 `generated/numbers_r10.tex` 注入上述核验数值。
- R09/CACAIE 包保持冻结，未改。
