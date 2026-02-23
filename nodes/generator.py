import os
import sys
import json
import tempfile
import uuid
from pathlib import Path

import torch
from PIL import Image

# 引入原项目
sys.path.insert(0, str(Path(__file__).parent.parent))
from autofigure2 import generate_figure_from_method, PROVIDER_CONFIGS
from ..utils.adapters import TypeAdapter

class AF_LLM_ImageGenerator:
    """AutoFigure 步骤一：Paper Method -> Figure PNG"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "method_text": ("STRING", {"multiline": True, "default": "Enter paper method here..."}),
                "provider": (["bianxie", "openrouter"], {"default": "bianxie"}),
                "api_key": ("STRING", {"default": ""}),
            },
            "optional": {
                "base_url": ("STRING", {"default": ""}),
                "image_model": ("STRING", {"default": ""}),  # 空则使用 provider 默认
                "use_reference": ("BOOLEAN", {"default": False}),
                "reference_image": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "JSON")
    RETURN_NAMES = ("figure_image", "metadata")
    FUNCTION = "generate"
    CATEGORY = "AutoFigure/Stage1"
    
    def generate(self, method_text, provider, api_key, base_url="", 
                image_model="", use_reference=False, reference_image=None, temperature=0.7):
        
        if not api_key:
            raise ValueError("API Key is required")
        
        # 获取配置
        config = PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS["bianxie"])
        base_url = base_url or config["base_url"]
        model = image_model or config["default_image_model"]
        
        # 处理参考图
        ref_path = None
        if use_reference and reference_image is not None:
            ref_pil = TypeAdapter.tensor_to_pil(reference_image)
            ref_path = os.path.join(tempfile.gettempdir(), f"af_ref_{uuid.uuid4().hex}.png")
            ref_pil.save(ref_path)
        
        # 临时输出路径
        output_dir = os.path.join(tempfile.gettempdir(), f"af_gen_{uuid.uuid4().hex}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "figure.png")
        
        # 调用原函数
        result_path = generate_figure_from_method(
            method_text=method_text,
            output_path=output_path,
            api_key=api_key,
            model=model,
            base_url=base_url,
            provider=provider,
            use_reference_image=use_reference,
            reference_image_path=ref_path
        )
        
        # 加载结果
        img = Image.open(result_path)
        tensor = TypeAdapter.pil_to_tensor(img)
        
        metadata = {
            "path": result_path,
            "provider": provider,
            "model": model,
            "has_reference": use_reference
        }
        
        return (tensor, json.dumps(metadata))
