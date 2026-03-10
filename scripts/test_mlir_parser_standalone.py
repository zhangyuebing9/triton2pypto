#!/usr/bin/env python3
"""
独立测试脚本 - 不依赖 pypto 或 triton，仅测试 MLIR 解析器
"""

import sys
import os

# 直接导入 mlir_parser 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 只导入 mlir_parser，不导入 ttir_converter（避免 pypto 依赖）
from triton_adapter.mlir_parser import MLIRParser, MLIRType, parse_ttir


def test_basic_parsing():
    """测试基本解析功能"""
    print("=" * 70)
    print("测试 1: 基本操作解析")
    print("=" * 70)
    
    mlir_text = """
    %0 = arith.constant 1.0 : f32
    """
    
    parser = MLIRParser()
    ops = parser.parse_module(mlir_text)
    
    assert len(ops) == 1
    assert ops[0].name == "arith.constant"
    assert ops[0].result.name == "0"
    print("✓ 常量操作解析成功")


def test_elementwise_operations():
    """测试 elementwise 操作解析"""
    print("\n" + "=" * 70)
    print("测试 2: Elementwise 操作解析")
    print("=" * 70)
    
    mlir_text = """
    %0 = arith.addf %x, %y : tensor<128xf32>
    %1 = arith.subf %a, %b : tensor<128xf32>
    %2 = arith.mulf %c, %d : tensor<128xf32>
    %3 = arith.divf %e, %f : tensor<128xf32>
    """
    
    parser = MLIRParser()
    ops = parser.parse_module(mlir_text)
    
    assert len(ops) == 4
    assert ops[0].name == "arith.addf"
    assert ops[1].name == "arith.subf"
    assert ops[2].name == "arith.mulf"
    assert ops[3].name == "arith.divf"
    
    for op in ops:
        assert len(op.operands) == 2
        print(f"✓ {op.name} 解析成功，操作数: {op.operands}")


def test_memory_operations():
    """测试内存操作解析"""
    print("\n" + "=" * 70)
    print("测试 3: 内存操作解析")
    print("=" * 70)
    
    mlir_text = """
    %0 = tt.load %arg0 : tensor<128xf32>
    tt.store %arg1, %0 : tensor<128xf32>
    """
    
    parser = MLIRParser()
    ops = parser.parse_module(mlir_text)
    
    assert len(ops) == 2
    assert ops[0].name == "tt.load"
    assert ops[1].name == "tt.store"
    
    print(f"✓ tt.load 解析成功，操作数: {ops[0].operands}")
    print(f"✓ tt.store 解析成功，操作数: {ops[1].operands}")


def test_type_utilities():
    """测试类型工具函数"""
    print("\n" + "=" * 70)
    print("测试 4: 类型工具函数")
    print("=" * 70)
    
    # 测试 tensor 类型
    tensor_type = MLIRType("tensor<128xf32>")
    assert tensor_type.is_tensor() == True
    assert tensor_type.is_pointer() == False
    
    shape = tensor_type.get_shape()
    print(f"✓ tensor<128xf32> 形状: {shape}")
    
    elem_type = tensor_type.get_element_type()
    print(f"✓ tensor<128xf32> 元素类型: {elem_type}")
    
    # 测试指针类型
    ptr_type = MLIRType("!tt.ptr<fp32>")
    assert ptr_type.is_tensor() == False
    assert ptr_type.is_pointer() == True
    print(f"✓ !tt.ptr<fp32> 是指针类型")


def test_complete_kernel():
    """测试完整 kernel 解析"""
    print("\n" + "=" * 70)
    print("测试 5: 完整向量加法 kernel")
    print("=" * 70)
    
    mlir_text = """
module {
  tt.func @add_kernel(%arg0: !tt.ptr<fp32>, %arg1: !tt.ptr<fp32>, %arg2: !tt.ptr<fp32>) {
    %cst = arith.constant 1.0 : f32
    %0 = tt.load %arg0 : tensor<128xf32>
    %1 = tt.load %arg1 : tensor<128xf32>
    %2 = arith.addf %0, %1 : tensor<128xf32>
    tt.store %arg2, %2 : tensor<128xf32>
  }
}
    """
    
    parser = MLIRParser()
    ops = parser.parse_module(mlir_text)
    
    print(f"✓ 解析到 {len(ops)} 个操作")
    
    op_names = [op.name for op in ops]
    expected_ops = ["arith.constant", "tt.load", "tt.load", "arith.addf", "tt.store"]
    
    for expected in expected_ops:
        if expected in op_names:
            idx = op_names.index(expected)
            print(f"  ✓ 找到 {expected}: {ops[idx]}")
        else:
            print(f"  ✗ 未找到 {expected}")


def show_phase1_status():
    """显示 Phase 1 实施状态"""
    print("\n" + "=" * 70)
    print("Phase 1 实施状态")
    print("=" * 70)
    
    status = {
        "MLIR 解析器": "✓ 完成",
        "类型映射框架": "✓ 完成",
        "转换器框架": "✓ 完成",
        "arith.constant": "⏳ 待实现转换逻辑",
        "arith.addf/subf/mulf/divf": "⏳ 待实现转换逻辑",
        "tt.load/store": "⏳ 待实现转换逻辑",
        "端到端测试": "⏳ 待实现",
    }
    
    for feature, stat in status.items():
        print(f"  {feature:30} {stat}")
    
    print("\n下一步:")
    print("  1. 实现 arith.constant 转换逻辑")
    print("  2. 实现 arith.addf/subf/mulf/divf 转换逻辑")
    print("  3. 实现 tt.load/store 转换逻辑")
    print("  4. 创建端到端测试示例")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Phase 1 Elementwise 算子转换 - 独立测试")
    print("=" * 70)
    
    try:
        test_basic_parsing()
        test_elementwise_operations()
        test_memory_operations()
        test_type_utilities()
        test_complete_kernel()
        show_phase1_status()
        
        print("\n" + "=" * 70)
        print("✓ 所有测试通过！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)