import os
import sys
import json
import tempfile
import uuid
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from autofigure2 import (
    generate_svg_template, 
    optimize_svg_with_llm,
    get_svg_dimensions,
    calculate_scale_factors,
    PROVIDER_CONFIGS
)
from ..utils.adapters import TypeAdapter
from ..utils.constants import DEFAULT_PLACEHOLDER_MODE, DEFAULT_OPTIMIZE_ITERATIONS

class AF_SVG_TemplateGenerator:
    """AutoFigure 步骤四：SVG 生成 + 验证 + 优化 + 坐标对齐"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "figure_image": ("IMAGE",),      # 原图
                "samed_image": ("IMAGE",),       # 带标记的图
                "boxlib": ("JSON",),             # box 信息
                "provider": (["bianxie", "openrouter"], {"default": "bianxie"}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "base_url": ("STRING", {"default": ""}),
                "svg_model": ("STRING", {"default": ""}),
                "placeholder_mode": (["none", "box", "label"], {"default": DEFAULT_PLACEHOLDER_MODE}),
                "optimize_iterations": ("INT", {"default": DEFAULT_OPTIMIZE_ITERATIONS, "min": 0, "max": 5}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0}),
            }
        }
    
    RETURN_TYPES = ("SVG_CODE", "IMAGE", "VEC2")
    RETURN_NAMES = ("svg_template", "preview", "scale_factors")
    FUNCTION = "generate"
    CATEGORY = "AutoFigure/Stage4"
    
    def generate(self, figure_image, samed_image, boxlib, provider, api_key,
                base_url="", svg_model="", placeholder_mode=DEFAULT_PLACEHOLDER_MODE,
                optimize_iterations=DEFAULT_OPTIMIZE_ITERATIONS, temperature=0.3):
        
        if not api_key:
            raise ValueError("API Key is required for SVG generation")
        
        config = PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS["bianxie"])
        base_url = base_url or config["base_url"]
        model = svg_model or config["default_svg_model"]
        
        # 保存图片
        paths = {}
        for name, img in [("figure", figure_image), ("samed", samed_image)]:
            pil = TypeAdapter.tensor_to_pil(img)
            path = os.path.join(tempfile.gettempdir(), f"af_svg_{name}_{uuid.uuid4().hex}.png")
            pil.save(path)
            paths[name] = path
        
        # 保存 boxlib
        boxlib_path = os.path.join(tempfile.gettempdir(), f"af_svg_box_{uuid.uuid4().hex}.json")
        with open(boxlib_path, 'w', encoding='utf-8') as f:
            json.dump(boxlib, f, ensure_ascii=False, indent=2)
        
        output_dir = os.path.join(tempfile.gettempdir(), f"af_svg_{uuid.uuid4().hex}")
        os.makedirs(output_dir, exist_ok=True)
        
        template_path = os.path.join(output_dir, "template.svg")
        
        # 步骤四：生成 SVG（含 4.5 自动验证修复）
        svg_path = generate_svg_template(
            figure_path=paths["figure"],
            samed_path=paths["samed"],
            boxlib_path=boxlib_path,
            output_path=template_path,
            api_key=api_key,
            model=model,
            base_url=base_url,
            provider=provider,
            placeholder_mode=placeholder_mode
        )
        
        # 步骤 4.6：LLM 优化（迭代次数可配置，0 则跳过）
        optimized_path = os.path.join(output_dir, "optimized.svg")
        if optimize_iterations > 0:
            optimize_svg_with_llm(
                figure_path=paths["figure"],
                samed_path=paths["samed"],
                final_svg_path=svg_path,
                output_path=optimized_path,
                api_key=api_key,
                model=model,
                base_url=base_url,
                provider=provider,
                max_iterations=optimize_iterations,
                skip_base64_validation=True  # 模板阶段无 base64 图片
            )
            final_svg_path = optimized_path
        else:
            final_svg_path = svg_path
        
        # 读取 SVG 代码
        with open(final_svg_path, 'r', encoding='utf-8') as f:
            svg_code = f.read()
        
        # 生成预览图
        preview_tensor = TypeAdapter.svg_to_tensor(svg_code)
        
        # 步骤 4.7：坐标系对齐计算
        figure_pil = Image.open(paths["figure"])
        svg_w, svg_h = get_svg_dimensions(svg_code)
        
        if svg_w and svg_h:
            scale_x, scale_y = calculate_scale_factors(
                figure_pil.width, figure_pil.height, svg_w, svg_h
            )
        else:
            scale_x, scale_y = 1.0, 1.0
        
        return (svg_code, preview_tensor, (scale_x, scale_y))
