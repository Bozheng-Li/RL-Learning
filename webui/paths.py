"""路径常量与 ``sys.path`` 引导。

这是整个 ``webui`` 包里唯一允许改 ``sys.path`` 的地方，而且必须 ``insert(0)``：
仓库根下的 ``compare.py`` / ``train.py`` 是顶层模块（不是包），``import compare``
靠 ``sys.path`` 命中；如果 ``webui/`` 下出现同名文件，它会把这行 import 抢走。

这个模块只在首次被 import 时执行一次（Streamlit 每次交互会重跑入口脚本，但
已经 import 过的模块不会重新执行），所以放在这里做一次性的路径引导最稳。
"""

from __future__ import annotations

import sys
from pathlib import Path

#: 本目录（``<root>/webui``）。
WEBUI_ROOT = Path(__file__).resolve().parent

#: 仓库根目录。
REPO_ROOT = WEBUI_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: 训练产物的根目录。
OUTPUTS_ROOT = REPO_ROOT / "outputs"

#: 实验配置所在目录。
CONFIG_ROOT = REPO_ROOT / "config"

#: WebUI 自己的运行时状态（任务注册表、日志），不入库。
STATE_ROOT = WEBUI_ROOT / ".state"

#: 子进程日志。
LOG_ROOT = STATE_ROOT / "logs"

#: 任务注册表。
REGISTRY_PATH = STATE_ROOT / "jobs.json"

#: 三个入口脚本，一律以子进程方式调用。
TRAIN_ENTRY = REPO_ROOT / "train.py"
VISUALIZE_ENTRY = REPO_ROOT / "visualize.py"
PLAY_ENTRY = REPO_ROOT / "play.py"

#: 判定「一个运行目录」的判据，与 ``compare.discover_runs`` 保持一致。
RUN_MARKER = "evaluation.json"

#: 运行目录里最早落盘的文件，用作「这次训练什么时候开始的」的墙钟起点。
START_MARKER = "resolved_config.yaml"


def resolve_repo_path(value: str | Path) -> Path:
    """把配置里的相对路径解析成相对仓库根的绝对路径。

    与 ``rl_common.config.resolve_repo_path`` 语义一致，但**不导入那个模块**——
    它顶层会 ``import torch``，为了算一个路径付两三秒的导入代价不值得。
    """
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()
