import os
import sys
import json
import tempfile
import uuid
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from autofigure2 import crop_and_remove_background
from ..utils.adapters import TypeAdapter

class AF_IconExtractor:
    """AutoFigure 步骤三：裁切 + RMBG2 去背景"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "boxlib": ("JSON",),  # 来自 SAM3 节点的 JSON
            },
            "optional": {
                "rmbg_model_path": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "JSON")
    RETURN_NAMES = ("icons_rgba", "icon_masks", "icon_infos")
    FUNCTION = "extract"
    CATEGORY = "AutoFigure/Stage3"
    
    def extract(self, original_image, boxlib, rmbg_model_path=""):
        # 保存原图
        pil_img = TypeAdapter.tensor_to_pil(original_image)
        temp_img = os.path.join(tempfile.gettempdir(), f"af_ext_{uuid.uuid4().hex}.png")
        pil_img.save(temp_img)
        
        # 保存 boxlib
        temp_boxlib = os.path.join(tempfile.gettempdir(), f"af_box_{uuid.uuid4().hex}.json")
        with open(temp_boxlib, 'w', encoding='utf-8') as f:
            json.dump(boxlib, f, ensure_ascii=False, indent=2)
        
        # 输出目录
        output_dir = os.path.join(tempfile.gettempdir(), f"af_icons_{uuid.uuid4().hex}")
        
        # 调用原函数
        icon_infos = crop_and_remove_background(
            image_path=temp_img,
            boxlib_path=temp_boxlib,
            output_dir=output_dir,
            rmbg_model_path=rmbg_model_path if rmbg_model_path else None
        )
        
        if not icon_infos:
            # 返回空
            empty_img = torch.zeros((1, 64, 64, 4))  # RGBA
            empty_mask = torch.zeros((1, 64, 64))
            return (empty_img, empty_mask, [])
        
        # 加载所有图标为 batch [N, H, W, C]
        icons = []
        masks = []
        valid_infos = []
        
        for info in icon_infos:
            try:
                icon = Image.open(info["nobg_path"])
                # 转换为 RGBA tensor [H, W, 4]
                icon_np = np.array(icon.convert("RGBA")).astype(np.float32) / 255.0
                icon_tensor = torch.from_numpy(icon_np)
                
                # 从 alpha 通道生成 mask
                alpha = icon_np[:, :, 3]
                mask_tensor = torch.from_numpy(alpha).unsqueeze(0)  # [1, H, W]
                
                icons.append(icon_tensor)
                masks.append(mask_tensor)
                valid_infos.append(info)
            except Exception as e:
                print(f"[AutoFigure] Skip invalid icon: {e}")
                continue
        
        if not icons:
            empty_img = torch.zeros((1, 64, 64, 4))
            empty_mask = torch.zeros((1, 64, 64))
            return (empty_img, empty_mask, [])
        
        # 统一尺寸（取最大尺寸 padding 或保持原样，这里直接 stack，ComfyUI 的 batch 需要相同尺寸）
        # 实际使用时应保持原始尺寸列表，通过 JSON 传递
        max_h = max([i.shape[0] for i in icons])
        max_w = max([i.shape[1] for i in icons])
        
        padded_icons = []
        padded_masks = []
        for icon, mask in zip(icons, masks):
            h, w = icon.shape[0], icon.shape[1]
            if h != max_h or w != max_w:
                # 创建 padding
                new_icon = torch.zeros((max_h, max_w, 4))
                new_mask = torch.zeros((1, max_h, max_w))
                new_icon[:h, :w, :] = icon
                new_mask[:, :h, :w] = mask
                padded_icons.append(new_icon)
                padded_masks.append(new_mask)
            else:
                padded_icons.append(icon)
                padded_masks.append(mask)
        
        icons_batch = torch.stack(padded_icons, dim=0)  # [N, H, W, 4]
        masks_batch = torch.stack(padded_masks, dim=0)  # [N, 1, H, W]
        
        return (icons_batch, masks_batch.squeeze(1), valid_infos)
