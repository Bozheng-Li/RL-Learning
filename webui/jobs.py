"""以子进程方式发起训练与重新出图。

**为什么必须是子进程**：``train_from_config`` 完全阻塞，而且有全局副作用——
``apply_torch_threads`` 会改全局 torch 线程数，``OnPolicyAlgorithm.__init__`` 会设
全局 ``torch.manual_seed`` / ``np.random.seed``。在 Streamlit 进程里直接调用，既会把
脚本重跑模型卡死，也会让同一进程里的多次训练互相污染。

**为什么日志落磁盘而不是 PIPE**：PIPE 的缓冲区写满（约 64KB）后子进程会阻塞，
而我们只在页面重跑时才读。落文件还顺带解决了「断线重连 / 服务重启后日志还在」。

本模块不依赖 Streamlit，可以脱离界面验证。
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from . import paths

JobKind = Literal["train", "visualize"]


@dataclass
class Job:
    """一个由 WebUI 启动的后台任务。"""

    job_id: str
    label: str
    kind: JobKind
    argv: list[str]
    log_path: str
    started_at: float
    run_dir: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    total_timesteps: int | None = None
    device: str | None = None
    #: 传给子进程的 ``CUDA_VISIBLE_DEVICES``（形如 ``GPU-<uuid>``）。用 UUID 而不是序号，
    #: 因为序号是 CUDA 的枚举顺序（默认 FASTEST_FIRST），与 nvidia-smi 的物理 index
    #: 不一致——按序号绑卡会静默绑到另一张物理卡上。见 ``webui/gpus.py`` 的模块注释。
    cuda_visible: str | None = None
    #: 只在当前进程内有意义，不写进注册表。
    _process: subprocess.Popen | None = field(default=None, repr=False, compare=False)

    # ---- 状态 ----

    @property
    def alive(self) -> bool:
        if self._process is not None:
            return self._process.poll() is None
        return self.pid is not None and pid_alive(self.pid)

    @property
    def finished(self) -> bool:
        if self._process is not None:
            return self._process.poll() is not None
        return self.exit_code is not None

    def refresh(self) -> None:
        """同步一次退出状态。页面每次重跑都该调一次（``poll()`` 是非阻塞的）。"""
        if self._process is None:
            return
        code = self._process.poll()
        if code is not None:
            self.exit_code = code
            self._process = None

    @property
    def command(self) -> str:
        return shlex.join(self.argv)

    def to_record(self) -> dict[str, Any]:
        """序列化成可写进 JSON 的形式（丢掉进程句柄）。

        刻意不用 ``dataclasses.asdict()``：它会 ``deepcopy`` 每个字段，而
        ``_process`` 里的 ``Popen`` 持有线程锁，深拷贝会直接抛
        ``TypeError: cannot pickle '_thread.lock' object``——在能 pop 掉它之前就炸了。
        """
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "_process"
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "Job":
        known = {f for f in cls.__dataclass_fields__ if f != "_process"}
        return cls(**{key: value for key, value in record.items() if key in known})


# --------------------------------------------------------------------------- #
# 进程探活与注册表
# --------------------------------------------------------------------------- #


def pid_alive(pid: int) -> bool:
    """``kill(pid, 0)`` 探活，不引入 psutil。"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def load_registry() -> list[Job]:
    """读磁盘上的任务注册表。

    用途是服务重启后重新接管还在跑的任务——``st.session_state`` 会随进程消失，
    但训练进程还活着。
    """
    path = paths.REGISTRY_PATH
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as handle:
            records = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(records, list):
        return []
    jobs: list[Job] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            jobs.append(Job.from_record(record))
        except TypeError:
            continue
    return jobs


def save_registry(jobs: list[Job]) -> None:
    """原子写入注册表：先写临时文件再 ``os.replace``，避免读到半截 JSON。"""
    path = paths.REGISTRY_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump([job.to_record() for job in jobs], handle, ensure_ascii=False, indent=1)
        os.replace(temporary, path)
    except OSError:
        pass


def upsert(job: Job) -> None:
    """把一个任务写进注册表（存在则更新）。"""
    jobs = [existing for existing in load_registry() if existing.job_id != job.job_id]
    jobs.append(job)
    save_registry(jobs)


# --------------------------------------------------------------------------- #
# 启动
# --------------------------------------------------------------------------- #


def launch(
    argv: list[str],
    *,
    label: str,
    kind: JobKind,
    run_dir: Path | None = None,
    total_timesteps: int | None = None,
    device: str | None = None,
    cuda_visible: str | None = None,
) -> Job:
    """启动一个后台子进程，stdout/stderr 落到 ``webui/.state/logs/<job_id>.log``。

    ``cuda_visible`` 是形如 ``GPU-<uuid>`` 的物理卡标识（见 ``webui/gpus.py``），
    ``device`` 只是给人看的显示名。**只有明确给了 ``cuda_visible`` 才设
    ``CUDA_VISIBLE_DEVICES``**：无条件覆盖会静默破坏用户在启动 WebUI 前设好的显卡
    过滤，导致界面显示的卡与实际用到的卡不是同一张。
    """
    job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}"
    log_path = paths.LOG_ROOT / f"{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    # 用 UUID 做物理卡级隔离：子进程里那张卡被映射成本地的 cuda:0，
    # 所以 argv 里的 algorithm.device 恒为 cuda:0（见 build_train_argv）。
    if cuda_visible:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible

    handle = log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(  # noqa: S603  —— argv 由本模块构造，无 shell
            argv,
            cwd=str(paths.REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            # 重定向到文件时 Python 默认块缓冲，「实时日志」会变成「批量日志」。
            env=env,
            # 新进程组：SSH 断开或 Streamlit 重启都不会给训练发 SIGHUP。
            start_new_session=True,
            close_fds=True,
        )
    finally:
        # 子进程已经继承了 fd，父进程这份要关掉，否则会一直占着。
        handle.close()

    job = Job(
        job_id=job_id,
        label=label,
        kind=kind,
        argv=argv,
        log_path=str(log_path),
        started_at=time.time(),
        run_dir=str(run_dir) if run_dir else None,
        pid=process.pid,
        total_timesteps=total_timesteps,
        device=device,
        cuda_visible=cuda_visible,
        _process=process,
    )
    upsert(job)
    return job


# --------------------------------------------------------------------------- #
# 构造命令行
# --------------------------------------------------------------------------- #


def yaml_scalar(value: Any) -> str:
    """把 Python 值序列化成 ``--set`` 能安全 round-trip 的 YAML 标量。

    ``cli.collect_overrides`` 对 ``--set key=value`` 的右值调 ``yaml.safe_load``，
    所以字符串必须按 YAML 规则引号化——一个含 ``": "`` 的字符串不加引号会被解析成
    dict，``apply_overrides`` 就会拿到一个 dict 而报错。
    """
    text = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True).strip()
    if text.endswith("\n..."):
        text = text[:-4]
    return text.strip()


def has_path(config: dict[str, Any], dotted: str) -> bool:
    """判断点号路径在配置里是否存在。

    ``apply_overrides`` 刻意只允许覆盖已存在的键，所以传之前必须先确认，
    否则会抛 ``KeyError`` 把训练打挂。
    """
    target: Any = config
    for key in dotted.split("."):
        if not isinstance(target, dict) or key not in target:
            return False
        target = target[key]
    return True


def build_train_argv(
    config_name: str,
    *,
    algorithm: str | None = None,
    seed: int | None = None,
    timesteps: int | None = None,
    output_directory: str | None = None,
    timestamped: bool | None = None,
    device: str | None = None,
    config: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> list[str]:
    """构造 ``train.py`` 的完整 argv。

    优先使用 ``--algorithm`` / ``--seed`` / ``--timesteps`` 这三个一等公民参数
    （``cli.py`` 对它们有专门处理，不经过 YAML 解析，没有引号风险）；只有输出目录
    与高级覆盖才走 ``--set``。
    """
    argv = [sys.executable, "-u", str(paths.TRAIN_ENTRY), "--config", config_name]
    if algorithm:
        argv += ["--algorithm", algorithm]
    if seed is not None:
        argv += ["--seed", str(int(seed))]
    if timesteps is not None:
        argv += ["--timesteps", str(int(timesteps))]

    sets: dict[str, Any] = dict(overrides or {})
    if output_directory:
        sets["output.directory"] = output_directory
        # 只有配置里真有这个键才覆盖，否则 apply_overrides 会 KeyError。
        if timestamped is not None and (config is None or has_path(config, "output.timestamped")):
            sets["output.timestamped"] = bool(timestamped)

    # 计算设备调度：``device`` 是 gpus.DeviceChoice.device_arg，绑定由 launch() 的
    # CUDA_VISIBLE_DEVICES 完成。那张物理卡在子进程里被映射成本地的 cuda:0，所以这里
    # 统一落到 cuda:0——把任意 ``cuda:N`` 归一到 0，避免调用方误传物理序号时静默用错卡。
    if device:
        if device.startswith("cuda:"):
            sets["algorithm.device"] = "cuda:0"
        else:
            sets["algorithm.device"] = device

    # SAC/TD3 需要连续动作空间。配置默认是离散的（LunarLander 的 continuous 键
    # 默认 false），所以选到这两个算法时自动翻成连续——否则训练启动后会直接报错。
    if algorithm and algorithm.upper().endswith(("SAC", "TD3")):
        if config is None or has_path(config, "environment.kwargs.continuous"):
            sets.setdefault("environment.kwargs.continuous", True)

    for key, value in sets.items():
        argv += ["--set", f"{key}={yaml_scalar(value)}"]
    return argv


def build_visualize_argv(
    config_name: str,
    run_dir: Path,
    *,
    trajectories: bool,
    model_path: Path | None = None,
) -> list[str]:
    """构造重新出图/重录轨迹的 argv。

    必须显式带上 ``--set output.directory=<run_dir>``：``visualize_from_config``
    用 ``run_directory(config)`` 定位结果，对开了 ``output.timestamped`` 的运行，
    只有显式覆盖才能指回那一次的结果。
    """
    argv = [
        sys.executable,
        "-u",
        str(paths.VISUALIZE_ENTRY),
        "--config",
        config_name,
        "--set",
        f"output.directory={yaml_scalar(str(run_dir))}",
    ]
    # visualize.py 里这两个开关互斥，只能给一个。
    argv.append("--trajectory-only" if trajectories else "--training-only")
    if model_path is not None:
        argv += ["--model", str(model_path)]
    return argv


def suggest_output_directory(
    config_name: str,
    algorithm: str,
    seed: int,
    *,
    unique: bool,
    stamp: str | None = None,
    tag: str | None = None,
) -> str:
    """给新训练建议一个输出目录。

    默认是**平铺**的 ``outputs/<配置>_<算法>_s<种子>_[<变体>_]<时间戳>``，而不是
    ``outputs/<配置>/<时间戳>/``：``compare.discover_runs`` 只扫 ``outputs/`` 的一层
    （``iterdir``），嵌套目录它永远发现不了，跑出来的实验在 ``compare.py`` 里会消失。
    平铺的名字里带上算法和种子，正好支持「同环境换算法/换种子」的对照实验。

    ``stamp`` 由调用方传入时，同一批对照实验会共用同一个时间戳，目录名只在算法和
    种子上不同，事后一眼就能把它们归到同一组。

    ``tag`` 是变体标识（见 ``webui/variants.py``）。**没有变体时逐字保持原样**——
    目录名是历史产物的一部分，改格式会让旧运行看起来像另一类东西。有变体时必须带，
    否则同算法同种子的两个变体会算出一模一样的目录，后启动的默默覆盖先启动的。
    """
    if not unique:
        return f"outputs/{config_name}"
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    slug = algorithm.replace("-", "").lower()
    parts = [f"outputs/{config_name}_{slug}_s{seed}"]
    if tag:
        parts.append(tag)
    parts.append(stamp)
    return "_".join(parts)
