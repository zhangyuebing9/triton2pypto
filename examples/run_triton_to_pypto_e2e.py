#!/usr/bin/env python3
"""add kernel 端到端验证便捷脚本。

等价于: python examples/run_e2e.py --kernel add --triton-compare

1. 从 Triton add kernel 源码提取 TTIR（compile-only，无需 GPU）
2. 转换为 PyPTO IR
3. PyPTO 编译 + simpler a2a3sim CPU 仿真执行
4. Triton TRITON_INTERPRET=1 CPU 执行
5. 对比两者结果一致

需要: pypto, torch, triton, SIMPLER_ROOT=third_party/simpler
"""

import os
import sys

# 复用统一入口
_script_dir = os.path.dirname(os.path.abspath(__file__))
_run_e2e = os.path.join(_script_dir, "run_e2e.py")
os.execv(sys.executable, [sys.executable, _run_e2e, "--kernel", "add", "--triton-compare"])
