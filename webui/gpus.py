"""显卡探测与训练任务分流。

这一层回答两个问题：**这台机器上有哪些卡能用**，以及**这一批任务该怎么分到卡上**。
它刻意不依赖 Streamlit（可以脱离界面验证）、不依赖新的第三方包（只用标准库 +
可选 torch + `nvidia-smi`），因为在别人的机器上装不上的依赖等于没有。

## 为什么不能用「第几张卡」来绑卡

`CUDA_VISIBLE_DEVICES=<序号>` 里的序号**不是** `nvidia-smi` 的物理 index，而是 CUDA
自己的枚举顺序。未设 `CUDA_DEVICE_ORDER` 时 CUDA 默认 `FASTEST_FIRST`，本机实测：
torch 枚举是 0=4090D / 1=4090D / 2=5880 Ada / 3=3090 / 4=3090，而 `nvidia-smi` 的
index 是 0=5880 Ada / 1=4090D / 2=4090D / 3=3090 / 4=3090——**5880 Ada 正好错位**。
两套顺序还随驱动、卡型、`CUDA_DEVICE_ORDER` 变化，所以按序号绑卡等于随机绑卡。

因此这里一律用 UUID 绑定：`CUDA_VISIBLE_DEVICES=GPU-<uuid>`（实测可行），并且
**不带的 `GPU-` 前缀的裸 uuid 会让 torch 报 `No CUDA GPUs are available`**，前缀不能
省。UUID 读不到的卡**不参与分流**，宁可退到 CPU 也不猜——静默绑错卡是最坏的失败模式：
任务会跑起来、日志正常、结果全部对不上号。

## 为什么可以直接覆盖子进程的 CUDA_VISIBLE_DEVICES

实测父进程 `CUDA_VISIBLE_DEVICES=3,4` 时，给子进程设被隐藏那张卡的 UUID，子进程仍能
看到它。所以「按探测结果用 UUID 绑卡」和「尊重父进程的过滤意图」不冲突：探测阶段
已经先把范围收在父进程允许的集合内了。
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

#: 利用率高于这个百分比就认为卡正被别人用着。
BUSY_UTILIZATION_PCT = 20

#: 剩余显存低于这个值就认为放不下一次训练。本项目的网络很小（几十 MB 量级），
#: 2 GB 是给 CUDA context 与框架开销留的余量。
BUSY_FREE_MEMORY_MB = 2048

#: `nvidia-smi` 查询超时（秒）。本机 5 张卡实测 81 ms，2 s 已经非常宽松。
_SMI_TIMEOUT = 2.0

#: `nvidia-smi --query-gpu` 要的字段。尾部字段顺序与 `parse_smi_output` 一一对应。
#:
#: 逐个列出 ``memory.used`` 而不是用 ``total - free`` 反推：本机实测两者差 ~630 MB
#: （驱动/ECC 预留），反推出来的「占用」会让界面上出现「已用 0.6 GB」配「没有进程占用」
#: 这种自相矛盾的画面。
_SMI_FIELDS = (
    "index", "uuid", "name",
    "memory.total", "memory.used", "memory.free", "utilization.gpu",
)

#: `parse_smi_output` 从行尾倒着取的字段个数（total / used / free / utilization）。
_SMI_NUMERIC_FIELDS = 4

#: `nvidia-smi --query-compute-apps` 要的字段，与 `parse_smi_processes` 一一对应。
_SMI_APP_FIELDS = ("gpu_uuid", "pid", "process_name", "used_memory")

#: 单次探测最多跑几条 `nvidia-smi`。GPU 状态面板要按卡列出占用进程，常规探测则
#: 不需要——见 `detect_devices(with_processes=...)`。
_SMI_CALLS_WITH_PROCESSES = 2

#: 探测结果的进程内缓存有效期（秒）。首次 CUDA 初始化要 1475 ms，之后 0.03 ms，
#: 所以缓存主要是为了躲开 CUDA init，而不是为了躲开 nvidia-smi。
_CACHE_TTL = 5.0

#: `CUDA_VISIBLE_DEVICES` 里表示「禁用全部 GPU」的取值。
#: 注意：**未设置**（``None``）表示不限制，与「设成空串」语义相反，必须区分。
_DISABLED_TOKENS = {"", "-1", "none", "void", "nodevfiles"}


# --------------------------------------------------------------------------- #
# 设备
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GpuDevice:
    """一张物理显卡。

    刻意是纯数据：不持有任何句柄，所以测试里可以凭空构造，不需要真的有卡。
    """

    index: int
    """`nvidia-smi` 的物理 index，**只用于展示**，不用于绑定。"""

    uuid: str | None
    """可直接用作 `CUDA_VISIBLE_DEVICES` 的值，形如 ``GPU-xxxx``。``None`` = 读不到。"""

    name: str
    memory_total_mb: int | None = None
    memory_free_mb: int | None = None
    utilization_pct: int | None = None
    memory_used_mb: int | None = None
    """`nvidia-smi` 直接给的已用显存。比 ``total - free`` 精确（后者含驱动预留）。"""

    source: str = "nvidia-smi"
    """探测来源：``nvidia-smi`` / ``torch``。torch 拿不到利用率。"""

    # ---- 派生 ----

    @property
    def key(self) -> str:
        """稳定的展示/比较用标识。UUID 优先，因为它是唯一不随枚举顺序变的东西。"""
        return self.uuid or f"index:{self.index}"

    @property
    def cuda_visible(self) -> str | None:
        """传给子进程的 ``CUDA_VISIBLE_DEVICES`` 值；``None`` 表示这张卡不能安全绑定。"""
        return self.uuid or None

    @property
    def memory_used_fallback_mb(self) -> int | None:
        """`nvidia-smi` 没给已用量时的兜底估算（``total - free``，含驱动预留）。"""
        if self.memory_total_mb is None or self.memory_free_mb is None:
            return None
        return max(0, self.memory_total_mb - self.memory_free_mb)

    @property
    def used_mb(self) -> int | None:
        """界面用的「已用显存」：优先用实测值，拿不到才退回估算。"""
        return self.memory_used_mb if self.memory_used_mb is not None else self.memory_used_fallback_mb

    @property
    def is_busy(self) -> bool:
        """是否正被占用。

        两个条件任一命中即算忙。信息缺失（两个都读不到）时**不算忙**——宁可去挤一张
        看不清的卡，也不要因为读不到数据就把所有卡都判死。
        """
        if self.memory_free_mb is not None and self.memory_free_mb < BUSY_FREE_MEMORY_MB:
            return True
        return (
            self.utilization_pct is not None
            and self.utilization_pct >= BUSY_UTILIZATION_PCT
        )

    @property
    def label(self) -> str:
        """界面上的一行文字。"""
        text = f"GPU {self.index}: {self.name}"
        if self.memory_free_mb is not None:
            text += f" · {self.memory_free_mb / 1024:.1f} GB 空闲"
        if self.utilization_pct:
            text += f" · 占用 {self.utilization_pct}%"
        return text

    @property
    def short_label(self) -> str:
        """窄地方用的短名，例如 ``GPU 2: RTX 5880 Ada``。"""
        return f"GPU {self.index}: {self.name.replace('NVIDIA ', '')}"


@dataclass(frozen=True)
class GpuProcess:
    """一张卡上的一个进程。``gpu_uuid`` 用来把它挂回对应的 ``GpuDevice``。"""

    gpu_uuid: str
    pid: int | None
    name: str
    used_memory_mb: int | None


@dataclass(frozen=True)
class Inventory:
    """一次探测的完整结果，连带「为什么只有这些卡」的说明。"""

    devices: list[GpuDevice] = field(default_factory=list)
    """过滤之后真正可以分流的卡（按物理 index 排）。"""

    physical_count: int = 0
    """探测看到的物理卡总数，在 `CUDA_VISIBLE_DEVICES` 过滤之前。"""

    visible: set[str] | None = None
    """父进程的 ``CUDA_VISIBLE_DEVICES`` 约束；``None`` = 未设置（不限制）。"""

    source: str = "none"
    """``nvidia-smi`` / ``torch`` / ``none``。"""

    notes: list[str] = field(default_factory=list)
    """给界面用的一句句说明，已按重要性排好序。"""

    processes: list[GpuProcess] = field(default_factory=list)
    """每张卡上正在跑的进程。torch 路径拿不到，恒为空。"""

    def __bool__(self) -> bool:
        return bool(self.devices)

    @property
    def note(self) -> str:
        return "；".join(self.notes)

    @property
    def total_free_mb(self) -> int:
        """所有可用卡的剩余显存之和；读不到显存的卡不计入。"""
        return sum(device.memory_free_mb or 0 for device in self.devices)


# --------------------------------------------------------------------------- #
# 解析（纯函数，测试的主要目标）
# --------------------------------------------------------------------------- #


def parse_visible_devices(raw: str | None) -> set[str] | None:
    """解析 ``CUDA_VISIBLE_DEVICES``。

    返回值语义：
    - ``None``：这个变量没设置，**不限制**。
    - ``set()``：显式禁用全部 GPU（空串、``-1``、``none``、``void``）。
    - 非空集合：允许的卡，元素是序号（``"2"``）或 UUID（``"GPU-xxxx"``）混排。
    """
    if raw is None:
        return None
    text = raw.strip()
    if text.lower() in _DISABLED_TOKENS:
        return set()
    return {token.strip() for token in text.split(",") if token.strip()}


def parse_smi_output(text: str) -> list[GpuDevice]:
    """解析 ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits`` 的输出。

    一行一张卡、逗号分隔。前两个字段是序号与 UUID，**后四个字段从行尾倒着取**
    （total / used / free / utilization），中间剩下的全部拼成名字——有些卡名自带逗号
    （``GeForce GTX 1080, rev. 2`` 这类），从左往右按位置取会把显存读成名字的一部分。

    脏行（字段不足、序号不是整数）直接跳过，不让它带崩整个探测——`nvidia-smi` 在不同
    驱动版本下会吐出格式略有差异的东西。
    """
    devices: list[GpuDevice] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        uuid = _normalize_uuid(parts[1])
        numeric = parts[-_SMI_NUMERIC_FIELDS:] if len(parts) >= 3 + _SMI_NUMERIC_FIELDS else []
        # 名字重新用 ", " 拼回去：`nvidia-smi` 的逗号后面本来带空格，逐段 strip 之后
        # 直接用 "," 拼会把空格吃掉（``GeForce GTX 1080,rev. 2``）。
        name = (
            ", ".join(parts[2:-_SMI_NUMERIC_FIELDS]).strip()
            if numeric else ", ".join(parts[2:]).strip()
        )
        devices.append(
            GpuDevice(
                index=index,
                uuid=uuid,
                # 没有名字时不要留空串，否则界面上一行会变成「GPU 3: 」
                name=name or uuid or f"GPU {index}",
                memory_total_mb=_int_or_none(numeric[0]) if numeric else None,
                memory_used_mb=_int_or_none(numeric[1]) if numeric else None,
                memory_free_mb=_int_or_none(numeric[2]) if numeric else None,
                utilization_pct=_int_or_none(numeric[3]) if numeric else None,
                source="nvidia-smi",
            )
        )
    return devices


def _int_or_none(raw: str) -> int | None:
    """``"24064"`` → 24064；``"N/A"`` / ``""`` → ``None``。"""
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def parse_smi_processes(text: str) -> list[GpuProcess]:
    """解析 ``nvidia-smi --query-compute-apps=...`` 的输出。

    字段顺序是 ``gpu_uuid, pid, process_name, used_memory``。**进程名从两个已知字段中间
    取出**（``parts[2:-1]``）：进程名里可能带逗号（路径、参数都会出现在这一列），
    从左往右按位置切会把参数读成别的字段。没有进程时 `nvidia-smi` 输出空行，这里自然
    得到空列表。
    """
    processes: list[GpuProcess] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        uuid = _normalize_uuid(parts[0])
        if uuid is None:
            continue
        processes.append(
            GpuProcess(
                gpu_uuid=uuid,
                pid=_int_or_none(parts[1]),
                name=", ".join(parts[2:-1]).strip() or "?",
                used_memory_mb=_int_or_none(parts[-1]),
            )
        )
    return processes


def _normalize_uuid(raw: object) -> str | None:
    """把各路来的 UUID 统一成可直接用于 ``CUDA_VISIBLE_DEVICES`` 的形式。

    `nvidia-smi` 给的已经带前缀；torch 的 ``props.uuid`` 是裸 uuid，必须补上，
    否则 torch 会报 ``No CUDA GPUs are available``。MIG 实例的 UUID 形如
    ``MIG-GPU-xxxx/1/0``，原样保留。
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.upper() in {"N/A", "[N/A]", "NONE", "NO DEVICES"}:
        return None
    if text.upper().startswith(("GPU-", "MIG-")):
        return text
    return f"GPU-{text}"


def _is_visible(device: GpuDevice, visible: set[str]) -> bool:
    """这张卡是否落在 ``CUDA_VISIBLE_DEVICES`` 允许的集合里。

    序号和 UUID 都认——两种写法都合法，而且用户可能写 ``2``、也可能写
    ``GPU-xxxx``。UUID 比较不区分大小写。
    """
    upper = device.uuid.upper() if device.uuid else None
    for token in visible:
        if token == str(device.index):
            return True
        if upper is not None and token.upper() == upper:
            return True
    return False


# --------------------------------------------------------------------------- #
# 探测
# --------------------------------------------------------------------------- #

_CACHE: tuple[float, Inventory] | None = None


def detect_devices(
    *, ttl: float = _CACHE_TTL, force: bool = False, with_processes: bool = False
) -> Inventory:
    """探测可用显卡，带进程内缓存。

    首次调用要付 CUDA 初始化的代价（实测 1475 ms），而列表页的按钮文案里就要用一次，
    每次页面重跑都付不起，所以缓存。``force=True`` 或短 ``ttl`` 用于 GPU 状态面板这类
    需要反映实时显存的地方。

    ``with_processes=True`` 时多跑一次 `nvidia-smi` 查每张卡上的占用进程，给 GPU 状态
    面板用；常规探测不需要，省一次进程调用。

    刻意不用 ``st.cache_data``：这一层要能脱离 Streamlit 验证，而缓存本身就是纯函数
    的缓存，没有副作用。
    """
    global _CACHE

    now = time.monotonic()
    if (
        not force
        and not with_processes
        and _CACHE is not None
        and now - _CACHE[0] < ttl
    ):
        return _CACHE[1]

    inventory = _probe(with_processes=with_processes)
    _CACHE = (now, inventory)
    return inventory


def clear_cache() -> None:
    """丢掉探测缓存。测试与「清空缓存并刷新」用。"""
    global _CACHE
    _CACHE = None


def _probe(*, with_processes: bool = False) -> Inventory:
    devices, source, processes = _raw_devices(with_processes=with_processes)
    physical_count = len(devices)
    notes: list[str] = []

    # 1) 尊重父进程的 CUDA_VISIBLE_DEVICES。用户设它就是为限制 WebUI 用哪些卡，
    #    不理它会导致实际占用被封的卡。nvidia-smi 看得到全部物理卡、不受子进程的
    #    CVD 影响，所以必须自己按序号/UUID 过滤。
    visible = parse_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if visible is not None:
        allowed = [device for device in devices if _is_visible(device, visible)]
        if len(allowed) < physical_count:
            notes.append(
                f"探测到 {physical_count} 张物理显卡，受启动时的 "
                f"CUDA_VISIBLE_DEVICES 限制，其中 {len(allowed)} 张可用"
            )
        devices = allowed

    # 2) 读不到 UUID 的卡不能安全绑定，留给手动命令行使用，不参与分流。
    bindable = [device for device in devices if device.cuda_visible]
    if len(bindable) < len(devices):
        notes.append(
            f"{len(devices) - len(bindable)} 张显卡读不到 UUID，无法稳定绑定，已跳过"
        )
    devices = bindable

    # 3) 兜底说明。空列表不是错误——没有 GPU 的机器上就该走 CPU。
    if physical_count == 0:
        notes.append("没有检测到 NVIDIA 显卡，将使用 CPU")
    elif not devices:
        notes.append("没有可用的显卡（被限制、占用或读不到 UUID），将使用 CPU")

    kept = {device.uuid for device in devices if device.uuid}
    return Inventory(
        devices=devices,
        physical_count=physical_count,
        visible=visible,
        source=source,
        notes=notes,
        # 只保留还留在候选池里的卡上的进程——被 CUDA_VISIBLE_DEVICES 挡掉的卡上
        # 有什么在跑，与这个 WebUI 能不能用无关。
        processes=[item for item in processes if item.gpu_uuid in kept],
    )


def _raw_devices(with_processes: bool) -> tuple[list[GpuDevice], str, list[GpuProcess]]:
    """按可靠性回退：`nvidia-smi` → torch → 空。"""
    text = _run_smi()
    if text is not None:
        devices = parse_smi_output(text)
        if devices:
            # 第二次 nvidia-smi 只有 GPU 状态面板需要。实测两次共 170 ms，而列表页
            # 的按钮文案里就要探测一次，能省则省。
            apps = parse_smi_processes(_run_smi_apps() or "") if with_processes else []
            return devices, "nvidia-smi", apps

    devices = _probe_torch()
    if devices:
        return devices, "torch", []

    return [], "none", []


def _run_smi() -> str | None:
    """跑一次 `nvidia-smi --query-gpu`。任何失败都返回 ``None``。"""
    return _run_smi_query(f"--query-gpu={','.join(_SMI_FIELDS)}")


def _run_smi_apps() -> str | None:
    """跑一次 `nvidia-smi --query-compute-apps`，看每张卡上是谁在占。"""
    return _run_smi_query(f"--query-compute-apps={','.join(_SMI_APP_FIELDS)}")


def _run_smi_query(query: str) -> str | None:
    """跑一条 `nvidia-smi` 查询。任何失败都返回 ``None``，由调用方降级。"""
    try:
        result = subprocess.run(  # noqa: S603  —— 固定 argv，无 shell，无用户输入
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=_SMI_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        # 没装 nvidia-smi / 不是 NVIDIA 机器 / 驱动挂了 / 超时，全部静默降级。
        return None
    return result.stdout if result.returncode == 0 else None


def _probe_torch() -> list[GpuDevice]:
    """torch 回退路径。拿不到利用率，显存与名字都有。"""
    try:
        import torch  # noqa: PLC0415  —— 重依赖，只在需要时导入

        if not torch.cuda.is_available():
            return []
        devices: list[GpuDevice] = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            total_mb = free_mb = None
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                total_mb = int(total_bytes // (1024 * 1024))
                free_mb = int(free_bytes // (1024 * 1024))
            except Exception:  # noqa: BLE001  —— 拿不到显存不该让整张卡消失
                pass
            devices.append(
                GpuDevice(
                    index=index,
                    uuid=_normalize_uuid(getattr(properties, "uuid", None)),
                    name=getattr(properties, "name", f"GPU {index}"),
                    memory_total_mb=total_mb,
                    memory_free_mb=free_mb,
                    utilization_pct=None,
                    source="torch",
                )
            )
        return devices
    except Exception:  # noqa: BLE001  —— 没有 CUDA 支持的 torch、驱动不匹配等
        return []


# --------------------------------------------------------------------------- #
# 分流
# --------------------------------------------------------------------------- #


def _sort_by_free_memory(devices: Iterable[GpuDevice]) -> list[GpuDevice]:
    """按剩余显存降序；读不到显存的排最后，但仍然留在池子里。

    本机是异构卡（1×5880 Ada 47 GB + 2×4090D 24 GB + 2×3090 24 GB），大显存优先能
    把最大的那张先填满。
    """
    return sorted(
        devices,
        key=lambda device: (
            device.memory_free_mb is None,
            -(device.memory_free_mb or 0),
            device.index,
        ),
    )


def select_devices(
    devices: Sequence[GpuDevice], *, excluded: Iterable[str] = ()
) -> tuple[list[GpuDevice], list[str]]:
    """从探测结果里挑出真正要用的卡：排除指定的、跳过忙的、按剩余显存降序排。

    返回 ``(候选卡, 说明文案)``。**绝不会返回空池导致调用方分不出任务**：如果所有卡
    都被判为忙，就退回未过滤的池子并给出警告——挤一张忙卡好过静默退到 CPU。
    """
    notes: list[str] = []
    excluded_keys = set(excluded)

    pool = [device for device in devices if device.key not in excluded_keys]
    if len(pool) < len(devices):
        notes.append(f"已按你的选择排除 {len(devices) - len(pool)} 张显卡")

    idle = [device for device in pool if not device.is_busy]
    if pool and not idle:
        notes.append("所有显卡当前都在忙，仍会按剩余显存顺序分配（可能变慢）")
        return _sort_by_free_memory(pool), notes
    if len(idle) < len(pool):
        notes.append(f"跳过 {len(pool) - len(idle)} 张忙碌中的显卡")

    return _sort_by_free_memory(idle), notes


def plan_assignments(devices: Sequence[GpuDevice], count: int) -> list[GpuDevice]:
    """把 ``count`` 个任务轮询铺到候选卡上。

    纯轮询：候选卡已按剩余显存降序排好，所以「轮流用一遍最大的那张、再回头」。
    候选卡为空或 ``count <= 0`` 时返回空列表，由调用方决定退到 CPU。
    """
    if not devices or count <= 0:
        return []
    return [devices[index % len(devices)] for index in range(count)]


def auto_assignments(
    devices: Sequence[GpuDevice], count: int, *, excluded: Iterable[str] = ()
) -> tuple[list[GpuDevice], list[str]]:
    """``select_devices`` + ``plan_assignments``，供两个调用点共用。

    这是「动态分流」的唯一入口：探测到的卡数决定分流到几张卡，不需要任何地方写死
    「5 卡」。
    """
    pool, notes = select_devices(devices, excluded=excluded)
    return plan_assignments(pool, count), notes


# --------------------------------------------------------------------------- #
# 交接给子进程
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DeviceChoice:
    """一次「用哪张卡跑」的完整决定，从界面一路传到 ``jobs.launch``。

    把「显示名」和「绑卡方式」分开，是因为它们不该混：``cuda_visible`` 是给操作系统的
    物理标识，``device_arg`` 是给子进程里那个只剩一张卡的 CUDA 的序号。
    """

    key: str
    """稳定标识，写进任务注册表。GPU 用 ``GPU-<uuid>``，CPU 用 ``cpu``。"""

    label: str
    """界面上显示的文字（下拉框选项、预览卡片）。"""

    short_label: str = ""
    """窄地方用的短名，例如 ``GPU 2: RTX 5880 Ada``。

    任务注册表里存的是它——监控页把它塞进一个 metric 里，长名字会把卡片撑破。
    """

    cuda_visible: str | None = None
    """传给子进程的 ``CUDA_VISIBLE_DEVICES``；``None`` = 不设置（CPU 或无法绑定时）。"""

    device_arg: str = "cpu"
    """子进程内 ``algorithm.device`` 的取值。"""

    @property
    def is_gpu(self) -> bool:
        return self.device_arg.startswith("cuda:")

    @classmethod
    def from_device(cls, device: GpuDevice) -> "DeviceChoice":
        """一张可绑定的卡 → 选择项。

        ``device_arg`` 恒为 ``cuda:0``：用 ``CUDA_VISIBLE_DEVICES`` 隔离之后，子进程里
        那张卡就是本地的第 0 张——这正是用 UUID 绑定的意义，子进程不需要知道物理序号。
        """
        return cls(
            key=device.key,
            label=f"🚀 {device.label}",
            short_label=device.short_label,
            cuda_visible=device.cuda_visible,
            device_arg="cuda:0",
        )


#: 「自动分流」这个抽象选项的 key。它不是物理卡，交给 ``build_choices`` 展开。
AUTO_KEY = "auto"

#: CPU 选项。
CPU_CHOICE = DeviceChoice(
    key="cpu",
    label="🖥️ CPU (不用 GPU)",
    short_label="CPU",
    cuda_visible=None,
    device_arg="cpu",
)


def device_options(inventory: Inventory) -> list[DeviceChoice]:
    """界面用的设备选项：自动分流（>=2 张卡时）+ 逐卡 + CPU。

    文案里的卡数来自实测探测，不写死任何数字。
    """
    options: list[DeviceChoice] = []
    if len(inventory.devices) >= 2:
        options.append(
            DeviceChoice(
                key=AUTO_KEY,
                label=f"⚡ 自动分流到全部 {len(inventory.devices)} 张显卡（按剩余显存均衡）",
                short_label=f"{len(inventory.devices)} 卡自动分流",
                cuda_visible=None,
                device_arg="auto",
            )
        )
    options.extend(DeviceChoice.from_device(device) for device in inventory.devices)
    options.append(CPU_CHOICE)
    return options


def build_choices(
    option: DeviceChoice,
    count: int,
    inventory: Inventory,
    *,
    excluded: Iterable[str] = (),
) -> tuple[list[DeviceChoice], list[str]]:
    """把界面上选的那一项展开成 ``count`` 个任务各自的实际用卡。

    - 选中某张具体的卡：全部任务都用它。
    - 选中 CPU：全部任务都用 CPU。
    - 选中自动分流：交给 ``auto_assignments``，**探测到几张卡就分流到几张卡**，
      没有可用卡时全部退到 CPU——这是「0 张卡也能正常跑」的落点。

    返回 ``(每任务一个用卡, 说明文案)``，长度恒等于 ``count``。
    """
    if option.key != AUTO_KEY:
        return [option] * count, []

    devices, notes = auto_assignments(inventory.devices, count, excluded=excluded)
    if not devices:
        return [CPU_CHOICE] * count, notes
    choices = [DeviceChoice.from_device(device) for device in devices]
    return choices, notes


def short_label(choice: DeviceChoice) -> str:
    """界面与注册表里用的短名。``DeviceChoice`` 自带短名时用它，否则退回 key。"""
    return choice.short_label or choice.key
