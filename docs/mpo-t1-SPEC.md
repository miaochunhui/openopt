# MPO-T1 实现规格书（Iteration 37 · 冻结版 v1）

> 阅读版(HTML 渲染): [mpo-t1.html 附录 A](mpo-t1.html#appendix-a) — 本 md 为唯一权威源。

多周期信号（return_1_30 / return_1_240 / return_1_2400）联合优化下单策略，参考 Axioma 051
《Multi-Period Portfolio Optimization with Alpha Decay》，显式建模 A 股 T+1 与非对称成本。
本文档是并行实现的唯一权威接口来源；`src/openpo/mpo/config.py` 与 `types.py` 已冻结，实现者
**只读不改**这两个文件。所有新代码位于 `src/openpo/mpo/`，测试位于 `tests/mpo/`，不得修改仓库
现有文件。纯 numpy/scipy/osqp 实现（不依赖 torch），float64。

## 0. 时间与日历约定

- 交易日 240 分钟，分钟索引 `0..239`；全局分钟 `tau = day*240 + minute`。
- 决策纪元：每天 8 个，分钟偏移 `{5,35,65,95,125,155,185,215}`（跳过开盘竞价余波）。
- 所有 horizon/衰减按**交易分钟**计（隔夜计 0 分钟），文档与代码注释显式声明。
- 一周 5 天，无节假日（合成日历）。

## 1. 阶段网格（stages.py, calendar.py）

`build_stage_grid(tau_now, cal, cfg) -> StageGrid`，J≤4 个阶段（绝对 τ 半开区间 `(start,end]`）：

- S1 执行段：`(tau_now, min(tau_now+30, 当日收盘)]`
- S2 日内剩余：`(S1.end, 当日收盘]`，长度可为 0（为 0 时丢弃，J 退化为 3）
- S3 次日全天：240 分钟
- S4 长尾：到 `tau_now+2400`（交易分钟），即约 8 个交易日聚合块；日历末尾截断。

不变量（必须测试）：阶段互不重叠、连续、S2 之后所有边界都在日收盘；`sum(L_j) ≤ 2400`。

## 2. Alpha 期限结构（term_structure.py, decay.py）

不重叠分段 `SEG = [(0,30], (30,240], (240,2400]]`（相对当前时刻的交易分钟）。

**标定**（训练窗 = 前 `calib_days` 日，之后冻结不再重估——本轮不滚动重拟合，留 P1）：
1. 信号横截面 z-score（每个 tau 对全体股票，均值 0 方差 1，nan 安全）。
2. 每个分段 s 的已实现分段收益 `r_seg[s](tau,i)`（由价格算：`P[tau+b]/P[tau+a]-1`，
   区间端点为交易分钟偏移；越界样本丢弃）。
3. 池化面板 ridge 回归（目标 winsorize ±3σ 截面）：
   `r_seg[s] ~ B[s,:] @ z`，先验收缩到 telescoping B0（`B0 = [[1,0,0],[-1,1,0],[0,-1,1]]`
   逐行乘以对应分段真实收益尺度）：`min ||r-ZB||^2 + lam*||B-B0||^2`，
   `lam = cfg.ridge_lambda * n_samples`。B 形状 `[3seg, 3sig]`。
4. AR(1) 自相关：逐信号在 30min 纪元网格上估计 lag-1 自相关（逐股 OLS 后取截面中位数），
   换算分钟 `rho_k = rho_epoch^(1/30)`；Kendall 偏差修正 `rho += (1+3*rho)/T_eff`，clip 到 (0,1)。

**输出** `TermStructureModel(B, rho, seg_bounds, diagnostics)`，方法
`stage_alphas(z[N,3], grid, tau_now, cal) -> alpha[J,N]`：

```
alpha[j,i] = Σ_s overlap_minutes(SEG_s_shifted, stage_j)/len(SEG_s) *
             Σ_k B[s,k] * rho_k^{delta_j} * z[i,k]
```
`SEG_s_shifted` = 以 tau_now 为原点的分段映射到绝对 τ；`delta_j` = tau_now 到 stage_j 中点的
交易分钟数（stage 1 的 delta=0）。

**单测**：合成数据已知真值 B 时恢复精度（相关>0.9）；lookahead 守卫（信号前移 1 bar 后
分段 IC 显著坍缩）。

## 3. 多阶段 QP（qp.py, solve.py, closed_form.py）

变量 `x = [w^1..w^J, b^1..b^J, s^1..s^J] ∈ R^{3JN}`（权重占 NAV，b/s ≥ 0 买卖拆分）。

**目标（min 形式）**：
```
min Σ_j [ -alpha[j]·w^j + (gamma/2)*(L_j/240)*w^j'Σ_d w^j
          + c_buy·1'b^j + c_sell·1's^j + (eta/2)*(b^j+s^j)' D (b^j+s^j) ]
```
- `Σ_d` 日频协方差，因子结构 `Σ = X F X' + diag(spec)`，QP 中用低秩展开变量 `y^j = X' w^j`
  （附加等式 `y^j = X'w^j`）或直接稠密（N=300 可稠密，实现者二选一，稠密优先，保正确）。
- `D = diag(impact_coef * sigma_d / adv_frac)`，`adv_frac_i = ADV_i / NAV`。
- 闭式解 parity 路径：`cfg.cost_mode='axioma'` 时改用 `(delta/2)*Δw'ΣΔw`（无 b/s 线性/冲击成本）。

**约束（全线性）**：
1. 动力学：`w^j - w^{j-1} - b^j + s^j = 0`（`w^0` = 当前持仓权重常数）
2. 预算：`0 ≤ 1'w^j ≤ 1`（留现金；现金零收益）
3. box：`0 ≤ w^j ≤ w_max`；`b^j, s^j ≥ 0`
4. **T+1**：今日阶段集合 `D0`（S1、S2）：`Σ_{j∈D0} s^j ≤ sellable_now`（常数向量，权重单位）；
   `s^{S3} ≤ w^{S2 或 w^0 若无S2}`；`s^{S4} ≤ w^{S3}`
5. 涨跌停（仅 S1）：涨停 `b^1_i = 0`，跌停 `s^1_i = 0`（box 上界置 0）
6. `cfg.t1_enabled=False` 时去掉约束 4（消融轴）。

> **脚注（box 可达性松弛，0822 修复）**：约束 3 的 box 上界不是死板的 `w_max`，而是**逐票取"合法可达的最紧上界"** ——
> `ub_i^j = max(w_max, 该阶段在 T+1 与涨跌停下 w_i 能被压到的最小值)`。原因：价格漂移会把既有持仓的权重
> 顶过 `w_max`，而 T+1 当日禁售该仓 ⇒ 若仍按 `w_max` 设上界，约束集**真不可行**（OSQP 报
> `primal infeasible` 是正确判决，不是求解器抽风），整批基线会退化为 fail-soft 不交易（实测曾达 82%）。
> 松弛只**容忍继承来的超配**：抬高后的上界 `w0 - max_cumulative_sell_j ≤ w0`，因此上界永远不超过该票**当前已有**
> 的权重 —— **买入无法制造出新的超配**（可 trim 的票一分不放松；目标函数里的风险与 alpha 项仍会在股份变可卖的
> 第一时间把仓位压回 `w_max`）。遥测：`meta['w_box_relaxed']`（被抬高的 (stage, name) 数）与
> `meta['w_box_relax_max']`（最大抬高幅度）供第一现场核查。对应单测见 `tests/mpo/test_qp_analytic.py`
> （漂移超配票不可 trim 时仍解得出、其余票仍受 `w_max` 约束）。

**求解** `solve_mpo_qp(qp, backend='osqp')`：OSQP（eps 1e-6, max_iter 20000, polish）主路，
`scipy.optimize` 或 Clarabel 备选；返回 `MpoSolution(weights[J,N], buys, sells, status,
objective, solve_time)`；非 optimal 时 fail-soft：返回 `status!='optimal'`，调用方回退为不交易。

**闭式解单测**（test_qp_analytic.py）：
- `closed_form_two_stage(alpha1, alpha2_exp, Sigma, gamma, delta, w0) -> w1`
  （Axioma 051 eq29-30：`w1 = a*inv(Σ)α¹ + a*δ*w0 + a*(δ/(δ+γ))*inv(Σ)E[α²]`，
  `a=(γ+δ)/((γ+δ)²+δγ)`）。J=2、无约束（box/budget/T+1 全放开为 ±1e6）、
  `cost_mode='axioma'` 时 QP 数值解 vs 闭式解 `max|diff| < 1e-6`（N=10 随机 SPD Σ）。
- 涌现性单测：强正短 alpha+零中长 alpha ⇒ S1 买入≈0（T+1+往返成本使纯短 alpha 买入无利）；
  强中长 alpha ⇒ 买入>0。

## 4. T+1 台账与执行（ledger.py, execution.py）

`T1Ledger`（股数单位，独立于 QP 再做一次硬校验）：
- 状态：`prev_close_shares[N]`（昨收持仓）、`sold_today[N]`、`bought_today[N]`
- `sellable() = max(prev_close_shares - sold_today, 0)`
- `on_sell_fill(i, shares)`：断言 `shares ≤ sellable()[i] + 1e-9`，违规 raise `T1ViolationError`
- `on_buy_fill(i, shares)`；`on_day_close(holdings)`：`prev_close_shares=holdings`，清零今日累计

`plan_orders(target_w, snapshot, prices, cfg) -> list[Order]`：
- 股数 = `(target_w - w_now)*NAV/price`；买单向下取整 100 股手；卖单 min(可卖, 取整)，
  清仓尾零股允许；金额 < `max(100股, cfg.min_order_nav_frac*NAV)` 丢弃（no-trade band 2bp）。
- 现金检查：买单总额 ≤ 现金 +本纪元卖单预计回款（A 股当日回款可用）；不足按 alpha 边际排序截断。

`match_orders(orders, bar_volume, prices, limit_flags, ledger, cfg) -> list[Fill]`：
- 成交价 = 窗口中点分钟收盘价 × (1±slippage_bp)（决策后 30 分钟 VWAP 的近似，
  与真 VWAP 差 <0.15bp／实测；索引 = `(tau + tau_end) // 2`，收盘截断的末纪元取靠前中点）
  - 修订 2026-08-23（Iteration 43）：原窗末（bar 收盘价）打标与 VWAP 语义偏差 44.5%，已订正。
- 成交量 ≤ `cfg.participation_rate * bar 成交量`（30min 聚合），残量**撤单**（不结转）
- 涨停买单 / 跌停卖单拒单并计数；卖出按**成交时刻**扣台账。

## 5. 模拟器（simulator.py）

`Simulator(market, cfg).run(strategy, signals, day_range) -> BacktestResult`：
- 每纪元：构造 `DecisionContext`（tau, z 信号切片, PortfolioSnapshot, 价格, 涨跌停旗标）
  → `strategy.decide(ctx) -> target_w[N] 或 None(不交易)` → plan → match → 记账。
- 记账恒等式（每日断言）：`NAV_t = NAV_{t-1} + 持仓盈亏 - 佣金 - 印花税 - 滑点成本`；
  现金 ≥ 0；T+1 违规数=0（由台账 raise 保证）。
- 停牌（合成 0.5%/日）：信号与交易双冻结，价格冻结。
- 产出 `BacktestResult`：日 NAV 序列、日换手、成本四分解（佣金/印花税/滑点/冲击 est）、
  拒单/截断计数、QP 墙钟 p50/p95、状态计数。

## 6. 合成数据（synth.py）

`generate_market(scfg, seed) -> (MarketPanel, SignalPanel)`，numpy `default_rng(seed)` spawn 子流：
- N=300 股 × D=100 日 ×240 分钟。分钟收益
  `r = Σ_k c_k*m_k + beta_i*f_mkt + 0.3*(风格因子项) + idio`：
  - 潜在 alpha `m_k`：AR(1)，半衰期 {30,240,2400} 交易分钟，跨信号相关≈0.3；
  - 市场因子年化 vol 20%，beta∈[0.8,1.2]；8 个 CN4 风格样式因子（简化：随机载荷、
    年化 vol 4-8%）；特质使截面日波动≈2.5%（年化~30-45%），日内 U 形季节性（首尾 ×1.5）。
- `c_k` 标定：使真条件均值对各 horizon 已实现收益的截面 IC 上限约 2×目标 IC。
- 观测信号 = 真 horizon 条件均值（AR(1) 几何和解析式）+ 高斯噪声，噪声幅度二分搜索使
  实测截面 Spearman IC ≈ `{0.05, 0.04, 0.03}`（burn-in 后窗口）。
- 微结构：价格从 20±10 CNY 起 GBM 累积；ADV lognormal（中位 8e7 CNY/日）分钟摊 U 形；
  涨跌停 ±10%（±0.1% 触发率/股日，触发日冻结越界方向）；停牌 0.5%/股日。
- `MarketPanel` 含 oracle：真分段 alpha `true_seg_alpha[T_epoch, N, 3]`（供 oracle-MPO 与
  标定恢复单测）。

## 7. 策略（strategies.py）

统一协议 `Strategy.decide(ctx: DecisionContext) -> np.ndarray | None`（目标权重 [N]，
执行段目标 = MPO 的 w^1；单期策略 = 其单期解）：
1. `MpoStrategy(ts_model, risk, cfg)` — 本体（J≤4）
2. `SingleSignalStrategy(k, ...)` — 单信号单期（J=1，alpha = 该信号分段贡献聚合到 S1∪S2 当日）
3. `FixedWeightStrategy(lambdas, ...)` — 固定权重复合单期（信号按截面波动归一后加权），
   `lambdas` 由 tuning seeds 网格选出（stats.py 提供 `tune_fixed_weights`）
4. `InformedTradingStrategy(...)` — Israelov-Katz：中长 alpha 单期目标 + 短 alpha 门控
   （仅当 `sign(z_short) 与 sign(trade)` 一致或 |z_short|<0.1 才放行该票交易；精确公式写死）
5. `OracleMpoStrategy` — 用真分段 alpha 的 MPO（增益分解上界，非竞争者）
6. `BuyAndHoldStrategy` — 等权锚（模拟器漏钱检测）
所有策略同 γ、同约束、同执行层；单期策略同样受 T+1 台账约束（QP 层加 `s ≤ sellable`）。

## 8. 回测编排与统计（backtest.py, stats.py）

- `run_seed(seed, cfg) -> dict`：生成市场 → calib 窗拟合 TS → 全部策略同一路径回测 eval 窗
  → 每策略指标 dict（净 Sharpe(√240 年化日收益)、净/毛年化、日均换手、成本分解、MaxDD、
  拒单率、T+1 违规=0、QP 墙钟）。
- `paired_stats(per_seed_metrics, strategy_a, strategy_b)`：配对差 t 检验 +
  seed 层 bootstrap(1e4) 95% CI + 胜率 + Hedges g。主检验预注册：MPO vs FixedWeight。
- 多策略比较 Holm 校正。

## 9. 跑批（benchmarks/mpo_t1_benchmark.py）

argparse CLI：`--seeds 24 --mode {smoke,main,sensitivity} --jobs N --output ...`
- smoke：2 seeds × N=100 × 20 日（calib 10 + eval 10），全部策略，<5 分钟，硬校验全绿。
- main：24 seeds 全默认。sensitivity：轴 `cost_mult {0.5,1,2}`、`ic_mult {0.5,1,1.5}`、
  `decay_mult {0.5,1,2}`、`stages {1,2,4}`、`t1 {on,off}`，每格 8 seeds。
- multiprocessing 按 seed 并行（`--jobs`），逐 seed JSON checkpoint（断点续跑），
  聚合 artifact 含 config hash、seed 列表、env（python/numpy/scipy/osqp 版本、主机）、
  git rev、结果 sha256。
- 产物：`benchmark-results/mpo-t1-{mode}.json` + `docs/mpo-t1-report.html`（report.py 渲染）。

## 10. 验收清单

- `pytest tests/mpo/ -q` 全绿；闭式解 parity < 1e-6；T+1 违规 0；现金恒等式；
  lookahead 守卫；标定恢复相关 >0.9。
- smoke 跑批全绿后才允许 main/sensitivity。
- 报告须含：各策略配对差 CI、成本敏感性、T+1 on/off 对照（MPO 相对优势来源的直接证据）、
  oracle 增益分解（优化结构增益 vs 标定损耗）、cold/hot 墙钟 p50/p95。
