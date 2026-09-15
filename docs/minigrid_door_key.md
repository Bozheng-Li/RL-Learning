# MiniGrid 实验

配置文件是 [`../config/minigrid.yaml`](../config/minigrid.yaml)。默认环境是 `MiniGrid-Empty-8x8-v0`，通过 `minigrid_flat` 包装器把图像和朝向编码为一维向量，便于使用 MLP。默认算法为 PPO，也可切换 A2C 或 DQN。

```bash
python train.py     --config minigrid
python visualize.py --config minigrid
python play.py      --config minigrid
```

默认结果目录为 `outputs/minigrid/`。轨迹图使用图像统计摘要，视频保留网格世界画面。需要安装 `minigrid`。

## 进阶：DoorKey（稀疏奖励）

智能体必须找到钥匙、捡起、开门、到达终点。除了成功那一刻，每步奖励都是 0。Empty-8x8 每步都给回报，所以这里的进阶点是**稀疏奖励下的信用分配**：回报要回溯到几十步前的“捡钥匙”动作。

共有两个配置，对应两级难度：

| 配置文件 | 环境 | 结果 |
| --- | --- | --- |
| [`../config/minigrid_doorkey.yaml`](../config/minigrid_doorkey.yaml) | `MiniGrid-DoorKey-5x5-v0` | ~40k 步达到 100% |
| [`../config/minigrid_doorkey_hard.yaml`](../config/minigrid_doorkey_hard.yaml) | `MiniGrid-DoorKey-6x6-v0` | ~20k 步达到 95-100% |

```bash
python train.py --config minigrid_doorkey        # -> outputs/minigrid_doorkey/
python train.py --config minigrid_doorkey_hard   # -> outputs/minigrid_doorkey_hard/
python play.py  --config minigrid_doorkey_hard
```

### 尺寸决定这个任务是否可解

实测结果，不是理论推测：

| 环境 | 无 shaping |
| --- | --- |
| `MiniGrid-DoorKey-5x5-v0` | ~40k 步达到 100% |
| `MiniGrid-DoorKey-6x6-v0` | 160k 步仍为 0% |
| `MiniGrid-DoorKey-8x8-v0` | 600k 步仍为 0% |

原因是 7x7 的视野。5x5 房间比视野还小，钥匙基本总在视野内；6x6 以上钥匙经常落在视野外。

### 6x6 为什么卡住，以及怎么解开

这是 `minigrid_doorkey_hard.yaml` 想演示的核心。诊断过程值得复现：

**第一步，定位瓶颈。** 用随机策略统计 200 个回合：

| 指标 | 随机策略 |
| --- | --- |
| 平均访问格数 | 14.3 / 64（22%） |
| 拾取过钥匙 | **93%** |
| 开过门 | **14%** |
| 到达目标 | 4% |

瓶颈**不是找不到钥匙**，而是开门。`toggle` 要求智能体同时满足「持钥匙 + 站在门旁 + 面向门」这个精确的 (位置, 朝向) 组合，随机撞上的概率只有 14%。

**第二步，试 count-based novelty —— 失败。** `minigrid_novelty` 包装器把状态离散成 `(x, y, direction)` 并给出 `scale / sqrt(count)` 的探索奖励（MiniGrid 状态空间小到可以精确计数，比 RND 更合适：无需梯度更新、奖励可解释）。结果三种配置（8x8 scale=0.5 / 8x8 scale=0.1 / 6x6 scale=0.5）跑满 300k 步，成功率**全是 0%**。训练回报从 83.8 衰减到 6.25，说明探索预算被用光了，门却始终没开。

失败原因是结构性的：**状态 novelty 只奖励“访问新状态”，不奖励“在该状态下按对动作”**。智能体可以无数次站在门旁面朝门，却一直按“前进”而不是 `toggle`——因为 `pickup` / `toggle` 的价值要在几十步之后才兑现，而 novelty 对此无能为力。

**第三步，给两个不可逆事件加中间奖励 —— 成功。** `minigrid_progress` 对「拾到钥匙」和「打开门」各给 0.5 的奖金。这两件事出现在每条成功轨迹上，所以 shaping 不改变最优策略是什么。加上它之后，6x6 在约 20k 步内达到 95-100%（此前 160k 步仍是 0%）。

这个对比就是这一节的全部价值：**novelty 解决“没去过的地方”，shaping 解决“去过但不知道按什么”**，两者针对的问题不同。

### 8x8：从「未解」到「接近解开」

即使加上 progress shaping，8x8 在 900k 步内仍是 0%。深挖之后发现，原因和「探索不足」完全不是一回事。

**第一步，读训练日志的回报。** shaping 满分是 1.9（拾钥匙 0.5 + 开门 0.5 + 成功 0.9）：

| 环境 | ep_rew_mean（末段） | ep_len_mean |
| --- | --- | --- |
| 6x6 | 1.964 | 14.2 步 |
| 8x8（900k） | 0.252 | 640（打满） |

8x8 的峰值只有 0.578，从未接近 1.0 —— **开门始终没学会**。

**第二步，看训练后的策略到底在做什么。** 用模型跑 100 回合确定性评估：

| 指标 | 随机策略 | 训练后策略 |
| --- | --- | --- |
| 拾钥匙 | 93% | **11%** |
| 开门 | 14% | 0% |

动作分布：`forward` 11.1%、`pickup` **37.9%**、`drop` **31.9%**、`toggle` **0.0%**、`done` 15.0%。

**这不是探索不足，是策略坍缩**：智能体收敛到「按 pickup / drop」这类局部无害动作（在空地上是 no-op，不会撞墙），几乎不前进，也从不按 toggle。PPO 不但没学会，还把策略训得**比随机更差**（拾钥匙 93% → 11%）。

**第三步，对症修复：提高熵正则。** 把 `ent_coef` 从 0.01 提到 0.05：

| 配置 | 训练回报（末段） | 峰值成功率 |
| --- | --- | --- |
| ent_coef=0.01 | 0.252 | 0% |
| ent_coef=0.05 | **1.786** | 80% |
| ent_coef=0.10 | 1.857 | 45% |
| ent_coef=0.05 + max_episode_steps=250 | 1.835 | 55% |

训练回报逼近上限 1.9；训练后策略的 `toggle` 使用率从 **0.0% 升到 19.4%**、到达目标从 0% 升到 24%。因果链闭合。

**当前状态**：8x8 已经能完成整个任务，但确定性评估成功率极不稳定。跑到 1M 步时：

| 指标 | 数值 |
| --- | --- |
| 训练回报（末段） | 1.839 / 上限 1.9 |
| 确定性评估成功率 | 在 **0% ~ 85%** 之间震荡，最近 6 次均值 46% |

训练回报逼近满分说明智能体在采样策略下确实会做；但确定性执行时（`argmax`）波动巨大，且**加大训练步数并没有让它收敛**。这更像"策略不稳定"而非"学不会"——`ent_coef=0.05` 维持了较高的动作熵，确定性执行时容易选偏。**因此它不适合作为默认配置**，6x6 是稳定基线，8x8 留作开放问题。

可能的方向（都还没验证）：熵系数退火（先高后低）、从 6x6 预训练迁移的课程学习、或对重复的空操作加惩罚。

要复现这组实验，把 `minigrid_doorkey_hard.yaml` 的 `environment.id` 改成 `MiniGrid-DoorKey-8x8-v0`、`ent_coef` 改成 `0.05` 即可。

### 两条排查经验

这次踩的两个坑值得记下来，它们比结论本身更通用：

1. **先看行为的分布，再猜原因。**「8x8 学不会」的第一直觉是探索不足，但动作分布直接显示 agent 根本不按 `toggle` —— 那是策略坍缩，不是探索。诊断顺序反了会浪费大量算力（本次在 novelty 上白跑了 120 万步）。
2. **稀疏奖励下，熵正则不是可有可无的装饰。** `ent_coef=0.01` 在 6x6 上够用，在 8x8 上直接导致策略坍缩到比随机还差。同一套超参跨环境尺寸并不成立。

### 关于记忆

一个反直觉的结论，有实验支持：**DoorKey 不是记忆任务**。在 DoorKey-5x5 上把视野压到 3x3（`environment.kwargs: {agent_view_size: 3}`），无记忆的 `PPO` 仍然 100% 成功，而且比 LSTM 学得更快。

原因是 DoorKey 是**反应式可解**的：“看到钥匙就 pickup、看到门就 toggle”只依赖当前视野，不需要回忆钥匙在哪。所以默认算法的选择不必纠结记忆能力。配置里保留了 `RecurrentPPO` 与 `PPO` 两个 profile，可以直接对比。

如果你要真正的部分可观测（POMDP）挑战，`MiniGrid-MemoryS7-v0` 更合适：智能体必须先看到一个物体、记住它，再走到走廊尽头两个房间中匹配的那个。不过实测下来 `RecurrentPPO` 与无记忆 `PPO` 在 300k 步内表现相当（训练回报 0.47 对 0.49，成功率都在 50-60%），都还没真正解决任务——随机基线是 22%，两者都高于随机，但远低于“真记住”应有的水平。

## 包装器

| 名称 | 作用 |
| --- | --- |
| `minigrid_flat` | 把图像和朝向编码为一维向量，移除文本 mission |
| `minigrid_novelty` | `(x, y, direction)` 计数式内在奖励，参数 `novelty_scale` |
| `minigrid_progress` | DoorKey 进度 shaping，参数 `shaping_pickup_bonus` / `shaping_door_bonus` |

两个奖励包装器**只在训练时生效**：评估和轨迹录制会自动禁用它们，所以 `evaluation.json` 里的回报和录制的视频始终是未塑形的任务奖励。

## 其他

`experiment.torch_threads: 1` 限制 torch 线程数。小策略在单线程下约快一倍（本机 256 核 CPU 实测：默认 128 线程 224 FPS，单线程 462 FPS），因为线程同步开销超过了并行收益。

`RecurrentPPO` 来自 `sb3-contrib`。
