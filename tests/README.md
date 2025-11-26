# 测试说明

本文档说明如何运行项目的测试套件。

## 安装测试依赖

项目使用 `uv` 作为包管理器。要安装测试依赖，请运行：

```bash
uv add --optional test
```

或者直接安装 pytest：

```bash
uv add pytest pytest-cov
```

## 运行测试

### 运行所有测试

```bash
uv run pytest
```

### 运行特定测试文件

```bash
uv run pytest tests/test_basetype.py
```

### 运行特定测试类

```bash
uv run pytest tests/test_basetype.py::TestTransform
```

### 运行特定测试方法

```bash
uv run pytest tests/test_basetype.py::TestTransform::test_transform_initialization
```

### 带覆盖率报告

```bash
uv run pytest --cov=base --cov-report=html
```

## 测试文件结构

- `test_basetype.py`: 测试基础数据类型（Transform, PoseSeries, DataCheck）
- `test_interpolate.py`: 测试插值函数（slerp_rotation, interpolate_vector3d, get_time_series）

## 编写新测试

1. 在 `tests/` 目录下创建新的测试文件，命名格式为 `test_*.py`
2. 测试类名应以 `Test` 开头
3. 测试方法名应以 `test_` 开头
4. 使用 `pytest` 的 fixture 功能进行测试设置和清理

## 测试最佳实践

- 每个测试应该独立运行
- 使用 `tmp_path` fixture 处理临时文件
- 使用 `pytest.raises` 测试异常情况
- 为测试提供清晰的描述性名称
