#!/usr/bin/env python3
"""
测试SAM3节点是否能正常导入
"""

import sys
import os

# 添加到路径
nodes_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, nodes_path)

print("=" * 60)
print("测试节点导入")
print("=" * 60)

try:
    from sam3_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print(f"\n✓ SAM3节点导入成功")
    print(f"✓ 节点数量: {len(NODE_CLASS_MAPPINGS)}")
    print(f"✓ 节点列表:")
    for key, value in NODE_CLASS_MAPPINGS.items():
        display_name = NODE_DISPLAY_NAME_MAPPINGS.get(key, key)
        print(f"  - {key} ({display_name}): {value}")

except Exception as e:
    print(f"\n✗ SAM3节点导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
