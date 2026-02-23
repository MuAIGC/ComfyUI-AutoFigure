import os
import sys
import re
import tempfile
import uuid
from pathlib import Path

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from autofigure2 import replace_icons_in_svg
from ..utils.adapters import TypeAdapter

class AF_SVG_IconReplacer:
    """AutoFigure 步骤五：图标替换到 SVG 占位符"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "svg_template": ("SVG_CODE",),    # 字符串类型的 SVG
                "icons_rgba": ("IMAGE",),         # [N, H, W, 4] batch
                "boxlib": ("JSON",),              # box 信息，用于坐标
                "scale_factors": ("VEC2",),       # (scale_x, scale_y)
            },
            "optional": {
                "match_by_label": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("SVG_CODE", "IMAGE")
    RETURN_NAMES = ("final_svg", "final_preview")
    FUNCTION = "replace"
    CATEGORY = "AutoFigure/Stage5"
    
    def replace(self, svg_template, icons_rgba, boxlib, scale_factors, match_by_label=True):
        
        if isinstance(boxlib, str):
            boxlib = json.loads(boxlib)
        
        boxes = boxlib.get("boxes", [])
        
        if len(boxes) == 0:
            # 无图标，直接返回模板
            preview = TypeAdapter.svg_to_tensor(svg_template)
            return (svg_template, preview)
        
        # 准备 icon_infos
        icon_infos = []
        icons_np = icons_rgba.cpu().numpy()
        
        for i, box in enumerate(boxes):
            if i >= len(icons_np):
                break
            
            # 获取第 i 个图标
            icon_np = icons_np[i]  # [H, W, 4]
            
            # 去除 padding（找到非零区域）
            alpha = icon_np[:, :, 3]
            coords = np.where(alpha > 0)
            if len(coords[0]) > 0:
                y1, x1 = coords[0].min(), coords[1].min()
                y2, x2 = coords[0].max() + 1, coords[1].max() + 1
                icon_crop = icon_np[y1:y2, x1:x2]
            else:
                icon_crop = icon_np
            
            # 保存为临时文件
            icon_pil = Image.fromarray((icon_crop * 255).astype(np.uint8), 'RGBA')
            temp_icon = os.path.join(tempfile.gettempdir(), f"af_icon_{i:02d}_{uuid.uuid4().hex}.png")
            icon_pil.save(temp_icon)
            
            label = box.get("label", f"<AF>{i+1:02d}")
            label_clean = label.replace("<", "").replace(">", "")
            
            icon_infos.append({
                "id": box.get("id", i),
                "label": label,
                "label_clean": label_clean,
                "x1": box["x1"],
                "y1": box["y1"],
                "x2": box["x2"],
                "y2": box["y2"],
                "width": box["x2"] - box["x1"],
                "height": box["y2"] - box["y1"],
                "nobg_path": temp_icon
            })
        
        # 保存模板 SVG
        temp_svg = os.path.join(tempfile.gettempdir(), f"af_template_{uuid.uuid4().hex}.svg")
        with open(temp_svg, 'w', encoding='utf-8') as f:
            f.write(svg_template)
        
        output_path = os.path.join(tempfile.gettempdir(), f"af_final_{uuid.uuid4().hex}.svg")
        
        # 调用原函数
        final_path = replace_icons_in_svg(
            template_svg_path=temp_svg,
            icon_infos=icon_infos,
            output_path=output_path,
            scale_factors=scale_factors,
            match_by_label=match_by_label
        )
        
        # 读取结果
        with open(final_path, 'r', encoding='utf-8') as f:
            final_svg = f.read()
        
        final_preview = TypeAdapter.svg_to_tensor(final_svg)
        
        return (final_svg, final_preview)
