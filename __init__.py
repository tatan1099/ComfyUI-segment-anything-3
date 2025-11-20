"""
ComfyUI Segment Anything 2 & 3 Package
统一导出SAM2和SAM3节点
"""

import importlib.util
import os
import sys

# 导入SAM2节点（使用动态导入处理连字符）
SAM2_NODES = {}
SAM2_NAMES = {}

try:
    sam2_path = os.path.join(os.path.dirname(__file__), "ComfyUI-segment-anything-2", "__init__.py")
    spec = importlib.util.spec_from_file_location("sam2_module", sam2_path)
    sam2_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sam2_module)

    SAM2_NODES = sam2_module.NODE_CLASS_MAPPINGS
    SAM2_NAMES = sam2_module.NODE_DISPLAY_NAME_MAPPINGS
except Exception as e:
    print(f"⚠️  Warning: Failed to import SAM2 nodes: {e}")
    SAM2_NODES = {}
    SAM2_NAMES = {}

# 导入SAM3节点
try:
    from .sam3_nodes import (
        NODE_CLASS_MAPPINGS as SAM3_NODES,
        NODE_DISPLAY_NAME_MAPPINGS as SAM3_NAMES
    )
except ImportError as e:
    print(f"⚠️  Warning: Failed to import SAM3 nodes: {e}")
    SAM3_NODES = {}
    SAM3_NAMES = {}

# 合并节点映射
NODE_CLASS_MAPPINGS = {**SAM2_NODES, **SAM3_NODES}
NODE_DISPLAY_NAME_MAPPINGS = {**SAM2_NAMES, **SAM3_NAMES}

# 打印加载的节点
print(f"✓ Loaded {len(SAM2_NODES)} SAM2 nodes")
print(f"✓ Loaded {len(SAM3_NODES)} SAM3 nodes")
print(f"✓ Total {len(NODE_CLASS_MAPPINGS)} segment-anything nodes")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
