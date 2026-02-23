import os
import sys
import json
import tempfile
import uuid
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from autofigure2 import segment_with_sam3
from ..utils.adapters import TypeAdapter
from ..utils.constants import DEFAULT_SAM_PROMPT, DEFAULT_MERGE_THRESHOLD

class AF_SAM3_Segment:
    """AutoFigure 步骤二：SAM3 分割 + Box 合并"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "sam_prompt": ("STRING", {"default": DEFAULT_SAM_PROMPT}),  # 支持逗号分隔多 prompt
            },
            "optional": {
                "sam_backend": (["local", "fal", "roboflow"], {"default": "local"}),
                "min_score": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "merge_threshold": ("FLOAT", {"default": DEFAULT_MERGE_THRESHOLD, "min": 0.0, "max": 1.0}),
                "sam_api_key": ("STRING", {"default": ""}),  # fal/roboflow 需要
                "sam_max_masks": ("INT", {"default": 32, "min": 1, "max": 100}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "JSON", "MASK")
    RETURN_NAMES = ("marked_image", "boxlib", "combined_mask")
    FUNCTION = "segment"
    CATEGORY = "AutoFigure/Stage2"
    
    def segment(self, image, sam_prompt, sam_backend="local", min_score=0.5, 
               merge_threshold=DEFAULT_MERGE_THRESHOLD, sam_api_key="", sam_max_masks=32):
        
        # 保存输入图
        pil_img = TypeAdapter.tensor_to_pil(image)
        temp_input = os.path.join(tempfile.gettempdir(), f"af_seg_input_{uuid.uuid4().hex}.png")
        pil_img.save(temp_input)
        
        # 输出目录
        output_dir = os.path.join(tempfile.gettempdir(), f"af_seg_{uuid.uuid4().hex}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 调用原函数（支持多 prompt 逗号分隔）
        samed_path, boxlib_path, boxes = segment_with_sam3(
            image_path=temp_input,
            output_dir=output_dir,
            text_prompts=sam_prompt,
            min_score=min_score,
            merge_threshold=merge_threshold,
            sam_backend=sam_backend,
            sam_api_key=sam_api_key if sam_api_key else None,
            sam_max_masks=sam_max_masks
        )
        
        # 加载 samed.png
        samed_img = Image.open(samed_path)
        samed_tensor = TypeAdapter.pil_to_tensor(samed_img)
        
        # 读取 boxlib
        with open(boxlib_path, 'r', encoding='utf-8') as f:
            boxlib_data = json.load(f)
        
        # 生成 mask tensor（所有 box 的合并 mask）
        mask = torch.zeros((1, pil_img.height, pil_img.width), dtype=torch.float32)
        for box in boxes:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            mask[0, y1:y2, x1:x2] = 1.0
        
        return (samed_tensor, boxlib_data, mask)
