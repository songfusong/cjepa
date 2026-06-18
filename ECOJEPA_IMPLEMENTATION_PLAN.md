# ECO-JEPA Implementation Plan

Last updated: 2026-06-18

## 1. Goal

ECO-JEPA 的目标不是证明平均 rollout MSE 更低，而是证明模型在机器人操作中最关键的
contact-mode transition 附近更可靠，并且这种提升能转化为 planning reliability。

核心假设：

- H1: world model 的主要预测误差集中在 contact onset、release、post-contact 等接触模式切换窗口。
- H2: event-conditioned object-history masking 比 C-JEPA random object masking 更能提升 contact-induced transition prediction。
- H3: 更好的 contact-transition prediction 能提升 MPC / CEM planning 的可靠性。

本项目必须紧贴 C-JEPA 的前提：使用 object-level latent masking 迫使模型从其他对象、动作和 proprioception 上下文中恢复被遮蔽对象，从而学习 interaction reasoning。ECO-JEPA 的改动应围绕 masking intervention、contact transition supervision 和 event-centric evaluation 展开，不应退化成一个泛泛的 dynamics model。

## 2. Current Baseline And ECO-JEPA Direction

当前 codebase 的基础是 C-JEPA PushT slot-latent pipeline：

- 输入为 pre-extracted object slots，加上 proprioception / action context。
- C-JEPA baseline 使用 random object-history masking。
- PushT planning 使用 learned latent world model + CEM / MPC。

ECO-JEPA 的增量方向：

- 将 object masking 从纯随机改为 contact-event-conditioned masking。
- 显式评估 contact onset、persistent contact、release、post-contact 阶段的误差。
- 增加 contact graph prediction、relative transition loss 和 planner reliability 指标。
- 从 PushT 扩展到 PushCube，再扩展到一个 articulated/contact-constrained task。

当前实现状态以 `ECOJEPA_IMPLEMENTATION_PROGRESS.md` 为准。本文件只定义目标和计划，不作为完成情况记录。

## 3. Implementation Roadmap

### Stage 1: PushT Masking Ablation

目标：确认 event-conditioned masking 是否值得继续。

实现内容：

- 保留 C-JEPA random object masking 作为主 baseline。
- 对比 `random`、`event_window`、`contact`、`distance` 四种 masking strategy。
- 所有策略使用同一 PushT 数据、同一训练预算、同一 validation / planning protocol。

关键判断：

- `event_window > random` 表明 event-conditioned masking 可能有效。
- `event_window > distance` 才能支持它不是简单 proximity heuristic。
- 如果 `event_window ~= distance`，下一步应先改进 event definition / masking schedule，而不是直接扩展任务。

验收：

- 至少 seed 0/1/2。
- 每个模型有 one-step event-centric prediction eval。
- 每个模型有 PushT planning eval。
- 汇总 mean / std，并明确标注 smoke、sanity 或 formal evidence。

### Stage 2: Multi-Horizon Event-Centric Rollout Evaluation

目标：评估接触误差是否会在 longer rollout 中累积。

实现内容：

- 支持 horizon `H = 1, 5, 10, 20`。
- 按 phase 汇总 error：free、onset、persistent、release、post-contact。
- 使用 event-balanced sampling，避免 free-motion 样本淹没 contact samples。

验收：

- 输出 per-horizon、per-phase CSV / JSON。
- 主观察点是 onset-window 和 post-contact rollout error。
- 不只报告 average MSE。

### Stage 3: Contact Graph Prediction Head

目标：让模型直接预测 future contact graph，并给出可解释的 contact transition 指标。

实现内容：

- 在 predictor 表征上增加 contact graph head。
- 输出未来 `t+1:t+H` 的 binary contact graph logits。
- 使用 BCE loss 训练 contact prediction。
- 报告 contact F1、onset F1、release F1、precision、recall。

注意：

- contact accuracy 不能作为主指标，因为 contact event 稀疏，accuracy 容易虚高。
- distance-threshold contact 是主实验需要报告的非 oracle 版本。
- simulator contact 可以作为 oracle upper bound，但不能作为唯一证据。

验收：

- C-JEPA random masking 和 ECO-JEPA event masking 都能跑 contact eval。
- ECO-JEPA 在 onset / release F1 上相对 random 有稳定收益，才可作为有效证据。

### Stage 4: Relative Transition Loss

目标：让模型更关注 contact-induced state change，而不是只拟合绝对 latent state。

实现内容：

- 增加 relative transition target：`Delta x_t = x_{t+1} - x_t`。
- 增加 event-window weighting：`w_t = 1 + alpha * I[t in event window]`。
- 默认 `alpha = 2.0`。
- loss 形式为 weighted MSE on predicted transition。

验收：

- 与 `L_future only`、`+ masked_history`、`+ contact_graph`、`+ relative_transition` 做 loss ablation。
- 重点看 post-contact error、onset F1、planning success 和 action-ranking correlation。

### Stage 5: Planner Reliability

目标：证明 prediction gain 能转化为 control / planning gain。

实现内容：

- 保留官方 PushT MPC / CEM planning eval。
- 新增 action-ranking correlation：
  - 在同一个 validation state 采样多条 action sequence。
  - 用 learned model rollout 得到 predicted cost。
  - 用 simulator rollout 得到 actual cost。
  - 计算 Spearman correlation / Kendall tau。

验收：

- 不只看 MPC success rate，还要看 predicted cost vs actual cost correlation。
- 如果 MSE 提升但 action ranking 和 MPC success 不提升，只能作为弱 evidence。
- 如果 event-conditioned model 在 ranking 和 MPC 都更好，才支持 H3。

### Stage 6: PushCube Transfer

目标：证明方法不是 PushT 几何结构特化。

实现内容：

- 复用 PushT 的 data logging、contact event parser、masking strategy 和 eval pipeline。
- 先做 state / slot-level MVP，不引入 RGB-D、tactile 或 learned contact detector。
- 对比 random、event_window、distance masking。

验收：

- PushCube 上 event-window 至少应稳定优于 random。
- 如果 distance baseline 和 event-window 接近，需要分析 contact event 定义是否过于接近 distance heuristic。

### Stage 7: Articulated / Contact-Constrained Task

目标：验证 ECO-JEPA 不只适用于平面 pushing。

优先任务标准：

- 有明确 contact onset。
- 有 contact 后 constrained motion。
- 不需要复杂 grasp-insertion pipeline。
- 可以稳定采集 state、action、contact 数据。

候选：

- Drawer / cabinet drawer task。
- PullCubeTool。
- PokeCube。
- TurnFaucet。

PegInsertionSide 只作为 appendix diagnostic，不作为主 claim。

验收：

- 至少完成 prediction-side event-centric metrics。
- 如果 planning pipeline 成本过高，可以先做 proxy planning / action-ranking。
- 若环境调试成本过高，应及时 fallback，不把项目卡在环境工程上。

## 4. Evaluation Requirements

### Prediction Metrics

必须报告：

- one-step prediction error。
- multi-step rollout error。
- final displacement / latent displacement error。
- phase-specific error：free、onset、persistent、release、post-contact。
- horizon-specific error：`H = 1, 5, 10, 20`。

### Contact Metrics

必须报告：

- contact graph F1。
- onset F1。
- release F1。
- precision / recall。
- time-to-contact error。
- contact duration error。
- false contact rate。
- missed contact rate。

### Planning Metrics

必须报告：

- MPC / CEM success rate。
- final goal distance。
- normalized task score if available。
- predicted cost vs actual cost correlation。
- action-ranking Spearman / Kendall tau。
- contact violation rate if available。

## 5. Experiment Integrity Rules

正式结论必须满足：

- 同一数据集。
- 同一 train / validation / test split。
- 同一训练预算。
- 同一 evaluation protocol。
- 至少 3 random seeds。
- 同一 evaluation initial states 上做 paired comparison。
- 结果记录包含命令、checkpoint、输出文件、seed、metrics。

不能将以下结果描述为 formal evidence：

- 单 seed smoke test。
- debug run。
- 使用不同 batch size / optimizer / dataset split 但未明确说明的结果。
- 只看 average MSE 的结果。
- 只在 oracle contact label 上有效、没有 noisy / distance contact 对照的结果。

## 6. Minimum Success Criteria

最低可继续推进标准：

- PushT 上 `event_window` 相比 `random` 有更好的 onset / post-contact prediction。
- PushT planning success 或 action-ranking correlation 至少一个指标优于 random baseline。
- `event_window` 与 `distance` baseline 的差异被明确分析，不能混淆为 proximity heuristic。
- 多 seed 结果趋势一致，或失败原因可解释。

最低可投稿标准：

- PushT 和 PushCube 都有 event-centric prediction gain。
- PushT 至少有 planning reliability gain。
- 有 masking ablation、loss ablation、noisy contact ablation。
- 有 influence / dependency visualization 或等价分析支持 interaction reasoning claim。

强结果标准：

- PushT + PushCube + one articulated/contact-constrained task。
- onset / post-contact rollout error 明显下降。
- contact onset / release F1 明显提升。
- action-ranking correlation 和 MPC success 都提升。
- OOD friction / pose / goal 中至少一个维度保持优势。

## 7. Out Of Scope For MVP

MVP 阶段不做：

- RGB image end-to-end training。
- RGB-D。
- tactile input。
- slot attention retraining。
- learned contact detector as main method。
- full PegInsertionSide success claim。

这些可以作为后续扩展，但不应阻塞当前 ECO-JEPA contact-transition 主线。
