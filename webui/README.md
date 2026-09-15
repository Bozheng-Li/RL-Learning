# Reinforce 实验台

一个本地 WebUI，用来查看实验的中间过程和结果，以及从网页发起新训练。

它是**纯新增的一层视图**：只读写 `outputs/` 下的产物、以子进程方式调用既有的
`train.py` / `visualize.py`，不改动 `rl_common/` 与 `algorithms/` 的任何行为。

## 安装

WebUI 的依赖是独立的，跑实验不需要它们：

```bash
python -m pip install -r webui/requirements.txt
```

国内直连 PyPI 不稳时加镜像：

```bash
python -m pip install -r webui/requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 启动

服务默认只监听 `127.0.0.1`（这个服务没有任何鉴权，不要暴露到局域网）。在无桌面的
服务器上，用 SSH 端口转发访问：

```bash
# 在服务器上启动
streamlit run webui/app.py

# 在你自己的机器上开隧道
ssh -N -L 8501:127.0.0.1:8501 <user>@<服务器>

# 然后浏览器打开
# http://127.0.0.1:8501
```

换端口：`streamlit run webui/app.py --server.port 8600`，隧道那侧端口要跟着改。

## 五个页面

| 页面 | 用途 |
| --- | --- |
| **运行总览** | 一屏看完 `outputs/` 下所有实验的状态、算法、种子、最终回报；进行中的运行带进度条和剩余时间。支持通配符/环境/算法筛选，可导出 CSV。 |
| **运行详情** | 单个运行的 7 个 tab：训练曲线、训练信号、动作诊断、周期评估、轨迹回放（视频）、产物清单、配置与日志。还能从这里**重新出图 / 重录轨迹**。 |
| **跨运行对比** | 选若干运行，曲线叠一张图、最终评估并排看。选中不同环境时会提示奖励尺度不可比。 |
| **发起训练** | 表单从 YAML 动态生成，页面实时显示**将要执行的完整命令行**，可直接复制去终端。 |
| **训练监控** | 每个任务一张卡片：进度条、剩余时间、实时信号曲线、日志尾部。定时刷新可关。 |

## 关于「发起训练」的几件事

**训练独立于页面运行。** 进程用 `start_new_session=True` 启动，关掉浏览器、
甚至重启 Streamlit 都不会打断训练。服务重启后靠 `webui/.state/jobs.json`
重新接管仍在运行的任务。

**不做终止功能。** 页面上会显示 PID，需要停止请手动 `kill <pid>`。

**输出目录默认平铺**成 `outputs/<配置>_<算法>_s<种子>_<时间戳>`，而不是嵌套的
`outputs/<配置>/<时间戳>/`。原因是 `compare.py` 只扫 `outputs/` 的一层
（`iterdir`），嵌套目录它永远发现不了——跑出来的实验会在终端对比里消失。
名字里带上算法和种子，顺带支持「同环境换算法/换种子」的对照实验。

**表单里的算法下拉来自当前配置的 `algorithm.profiles`**，不是全部可用算法。
选到没有对应 profile 的算法，训练会直接抛
`ValueError: No algorithm.profiles.X section exists`。同理，DQN 只支持离散动作、
SAC/TD3 只支持连续，这两类不匹配也会在点击之前就提示。

**「高级：任意配置覆盖」只列出这份配置里已经存在的标量键。** `apply_overrides`
刻意不允许新建键——拼错的键名如果被静默接受，训练会照常用默认值跑完，而你以为
改过了。所以拼错就直接报错。

## 实时进度是怎么来的

读 `logs/progress.csv`——SB3 的 CSV logger 每个 rollout 就 flush 一次。

**不能**用 `logs/episodes.csv`：那个文件是块缓冲的，训练结束才落盘，实时读永远是空的。

剩余时间三级回退：`time/fps` → `time/time_elapsed` → 用 `resolved_config.yaml`
的修改时间当墙钟起点。前两列只有 SB3 系的算法会写，自研 REINFORCE/A2C/PPO 没有。

## 常见问题

**页面报「没有发现任何运行」。**
「运行」的判据是目录里有 `evaluation.json`（跑完的标志）或 `resolved_config.yaml`
（已开工）。检查侧边栏的 `outputs 目录` 是否指对了。

**列表里少了某些运行。**
`output.timestamped: true` 会产生嵌套目录，本 UI 会扫两层，但 `compare.py`
不会。见上面关于平铺命名的说明。

**「运行详情」里不能重新出图。**
说明这次运行的 `output.directory` 是 `--set` 指定的，反查不到对应的配置名。
回命令行执行 `python visualize.py --config <配置> --set output.directory=...`。

**训练信号图是空的。**
`progress.csv` 的列集合每个算法都不一样，UI 只画这个运行真的有的列。
DQN 没有近似 KL，自研算法没有 FPS，这是正常的。

**改了绘图代码但图没变。**
Streamlit 的缓存没失效。点侧边栏的「清空缓存并刷新」。

**启动时报 `missing ScriptRunContext`。**
这是 `streamlit.testing` 或裸模式下运行时的提示，正常使用不受影响。
