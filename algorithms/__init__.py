"""算法注册表与工厂函数。

本项目有两类算法，通过名字前缀区分：

- **自研**（``REINFORCE`` / ``A2C`` / ``PPO``）—— 见 ``reinforce.py`` /
  ``a2c.py`` / ``ppo.py``，从零实现，中文注释写明了每个公式的来历。
- **SB3**（``SB3-PPO`` / ``SB3-DQN`` / ``SB3-SAC`` ...）—— 通过
  ``sb3_wrapper.SB3Algorithm`` 适配，用作对照基线与自研尚未覆盖的算法。

配置里改一行 ``algorithm.name`` 就能在两套实现之间切换，这是做对照实验的基础。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseAlgorithm
from .sb3_wrapper import SB3_ALGORITHMS, SB3Algorithm

#: 自研算法：算法名 -> 类。这些类直接实现 ``BaseAlgorithm``。
#: 由各算法模块在文件末尾注册，避免在这里硬编码导入顺序。
NATIVE_ALGORITHMS: dict[str, type] = {}


def register(cls: type) -> type:
    """把自研算法登记进注册表的装饰器。

    用法::

        @register
        class PPO(BaseAlgorithm):
            ...
    """
    NATIVE_ALGORITHMS[cls.__name__.upper()] = cls
    return cls


def _normalize(name: str) -> str:
    return str(name).upper()


def resolve_name(raw: str) -> str:
    """把配置里写的算法名解析成注册表中的规范名。

    规则是**自研优先，缺失时回退到 SB3**：

    - ``PPO`` 在自研 PPO 存在时指向自研实现；不存在时回退为 ``SB3-PPO``。
      这样现有配置不用改就能继续跑，而自研算法一旦落地就自动接管这个名字。
    - ``SB3-PPO`` 始终指向 SB3 实现，用于明确要求对照基线时。

    想要固定用某一种实现，就在配置里写全名（``SB3-PPO``）；只写 ``PPO``
    意味着「用本项目的 PPO 实现」。
    """
    upper = str(raw).upper()
    if upper in NATIVE_ALGORITHMS:
        return upper
    if upper in SB3_ALGORITHMS:
        return upper
    prefixed = f"SB3-{upper}"
    if prefixed in SB3_ALGORITHMS:
        return prefixed
    return upper


def is_sb3(name: str) -> bool:
    """判断某个算法名是否来自 SB3（而不是自研实现）。"""
    return resolve_name(name) in SB3_ALGORITHMS


def resolve_model_file(stem: str | Path, algorithm_name: str) -> Path | None:
    """推断模型文件的实际路径。

    SB3 用 ``torch.save`` 存成 ``.zip``，自研算法存成 ``.pt``。调用方通常只知道
    文件名主干（比如 ``final_model``），这里按算法来源补上扩展名，并兼容
    「路径本身已经带扩展名」的情况。

    Returns:
        存在的文件路径；若都不存在则返回首选路径（供调用方报错时展示）。
    """
    stem = Path(stem)
    if stem.suffix in {".zip", ".pt"}:
        return stem
    ext = ".zip" if is_sb3(algorithm_name) else ".pt"
    preferred = stem.with_suffix(ext)
    if preferred.exists():
        return preferred
    # 兜底：算法来源与文件名不一致时（比如配置改过），另一个扩展名也试一下。
    fallback = stem.with_suffix(".pt" if ext == ".zip" else ".zip")
    return fallback if fallback.exists() else preferred


def available_algorithms() -> list[str]:
    """返回所有可用算法名，供报错提示使用。"""
    return sorted({*NATIVE_ALGORITHMS, *SB3_ALGORITHMS})


def build_algorithm(env: Any, config: dict[str, Any]) -> BaseAlgorithm:
    """按配置构造算法实例。

    这是训练编排层唯一的算法入口——它不关心拿到的实例是自研还是 SB3 的。

    Args:
        env: 已包装好的训练环境。
        config: 完整实验配置，算法名取自 ``config["algorithm"]["name"]``。

    Raises:
        ImportError: 配置要求 sb3-contrib 的算法但该包未安装。
        ValueError: 算法名不在注册表中。
    """
    raw_name = str(config["algorithm"]["name"])
    name = resolve_name(raw_name)

    if name in NATIVE_ALGORITHMS:
        return NATIVE_ALGORITHMS[name](env, config)
    if name in SB3_ALGORITHMS:
        return SB3Algorithm(env, config, SB3_ALGORITHMS[name])

    # sb3-contrib 未安装时，给出比「不支持该算法」更有用的提示。
    if _normalize(raw_name) in {"RECURRENTPPO", "SB3-RECURRENTPPO"}:
        raise ImportError(
            "RecurrentPPO 需要 `sb3-contrib` 包，请用 "
            "`python -m pip install sb3-contrib` 安装；"
            "或者改用配置里的 `name: SB3-PPO`。"
        )

    raise ValueError(
        f"Unsupported algorithm {raw_name!r}; choose one of: "
        f"{', '.join(available_algorithms())}"
    )


def load_algorithm(
    path: str | Path,
    config: dict[str, Any],
    env: Any = None,
    device: str = "cpu",
) -> BaseAlgorithm:
    """从磁盘加载模型，算法类型由配置决定。

    可视化脚本用它重新载入训练好的策略，因此必须和训练时用到同一份配置。

    Args:
        path: 模型文件路径（SB3 存的是 ``.zip``）。
        config: 与训练时相同的实验配置。
        env: 可选环境，自研算法恢复空间信息时可能需要。
        device: 计算设备。
    """
    name = resolve_name(config["algorithm"]["name"])

    if name in SB3_ALGORITHMS:
        model = SB3_ALGORITHMS[name].load(str(path), device=device)
        # 已经载入的模型不需要再构造，这里用一个轻量壳把它包成统一接口。
        return _LoadedSB3Wrapper(model, config, env)

    if name in NATIVE_ALGORITHMS:
        return NATIVE_ALGORITHMS[name].load(path, env=env, device=device)

    raise ValueError(f"Unsupported algorithm in configuration: {name}")


class _LoadedSB3Wrapper(BaseAlgorithm):
    """包住一个已经加载好的 SB3 模型，让它满足 ``BaseAlgorithm`` 接口。

    与 ``SB3Algorithm`` 的区别：那个负责*构造并训练*，这个只负责*推理*，
    所以不需要 profile 解析，也不能再调用 ``learn``。
    """

    def __init__(self, model: Any, config: dict[str, Any], env: Any) -> None:
        super().__init__(env, config)
        self.model = model

    def learn(self, *args: Any, **kwargs: Any) -> "BaseAlgorithm":
        raise NotImplementedError("加载得到的模型不可继续训练")

    def predict(
        self,
        observation: Any,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = False,
    ) -> tuple[Any, Any]:
        return self.model.predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )

    def save(self, path: str | Path) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str | Path, env: Any = None, device: str = "cpu") -> Any:
        raise NotImplementedError("请使用 algorithms.load_algorithm()")


# 导入自研算法模块，触发里面的 @register 注册。
# 放在文件末尾是必须的：这些模块要 `from . import register`，
# 而本模块此时已经执行到末尾，register 已经定义好了。
from . import a2c, ppo, reinforce  # noqa: E402,F401
