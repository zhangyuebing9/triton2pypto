# Triton to PyPTO Adapter

将 Triton IR 适配到 PyPTO 后端，实现在华为 AI 加速器上运行 Triton 内核。

## 项目结构

```
triton2pypto/
├── third_party/
│   ├── triton/          # submodule: triton-lang/triton
│   └── pypto/           # submodule: hw-native-sys/pypto
├── src/
│   ├── triton_adapter/  # Triton IR 提取/转换
│   ├── pypto_backend/   # PyPTO 后端接入
│   └── passes/          # IR 转换 pass
├── tests/
├── scripts/
└── tasks/
```

## 快速开始

### 1. 初始化 Submodules

```bash
bash scripts/init_submodules.sh
```

### 2. 安装依赖

```bash
pip install -e ".[dev]"
```

### 3. 运行测试

```bash
pytest tests/
```

## 开发

详见 [AGENTS.md](AGENTS.md)