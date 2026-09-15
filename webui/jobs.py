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
) -> Job:
    """启动一个后台子进程，stdout/stderr 落到 ``webui/.state/logs/<job_id>.log``。"""
    job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}"
    log_path = paths.LOG_ROOT / f"{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    handle = log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(  # noqa: S603  —— argv 由本模块构造，无 shell
            argv,
            cwd=str(paths.REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            # 重定向到文件时 Python 默认块缓冲，「实时日志」会变成「批量日志」。
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
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
    config_name: str, algorithm: str, seed: int, *, unique: bool, stamp: str | None = None
) -> str:
    """给新训练建议一个输出目录。

    默认是**平铺**的 ``outputs/<配置>_<算法>_s<种子>_<时间戳>``，而不是
    ``outputs/<配置>/<时间戳>/``：``compare.discover_runs`` 只扫 ``outputs/`` 的一层
    （``iterdir``），嵌套目录它永远发现不了，跑出来的实验在 ``compare.py`` 里会消失。
    平铺的名字里带上算法和种子，正好支持「同环境换算法/换种子」的对照实验。
    """
    if not unique:
        return f"outputs/{config_name}"
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    slug = algorithm.replace("-", "").lower()
    return f"outputs/{config_name}_{slug}_s{seed}_{stamp}"
