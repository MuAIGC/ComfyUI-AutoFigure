import os
import sys
from pathlib import Path

# 确保能找到 autofigure2.py
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from .nodes import *

NODE_CLASS_MAPPINGS = {
    "AF_LLM_ImageGenerator": AF_LLM_ImageGenerator,
    "AF_SAM3_Segment": AF_SAM3_Segment,
    "AF_IconExtractor": AF_IconExtractor,
    "AF_SVG_TemplateGenerator": AF_SVG_TemplateGenerator,
    "AF_SVG_IconReplacer": AF_SVG_IconReplacer,
    "AF_SVG_Saver": AF_SVG_Saver,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AF_LLM_ImageGenerator": "🎨 AF 文生图 (Stage 1)",
    "AF_SAM3_Segment": "✂️ AF SAM3 分割 (Stage 2)",
    "AF_IconExtractor": "🧩 AF 图标提取 (Stage 3)",
    "AF_SVG_TemplateGenerator": "📐 AF SVG 生成 (Stage 4)",
    "AF_SVG_IconReplacer": "🔄 AF 图标替换 (Stage 5)",
    "AF_SVG_Saver": "💾 AF 保存 SVG",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("✅ AutoFigure-Edit for ComfyUI loaded successfully")
