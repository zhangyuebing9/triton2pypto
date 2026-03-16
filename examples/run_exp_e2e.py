#!/usr/bin/env python3
"""exp kernel 端到端验证便捷脚本。

等价于: python examples/run_e2e.py --kernel exp

1. 从 Triton exp kernel 源码提取 TTIR（含 pid 和 mask）
2. 转换为 PyPTO IR
3. PyPTO + simpler a2a3sim CPU 仿真执行
4. 与参考 (torch.exp) 对比

需要: pypto, torch, triton, SIMPLER_ROOT=third_party/simpler
"""

import os
import sys

# 复用统一入口
_script_dir = os.path.dirname(os.path.abspath(__file__))
_run_e2e = os.path.join(_script_dir, "run_e2e.py")
os.execv(sys.executable, [sys.executable, _run_e2e, "--kernel", "exp"])
