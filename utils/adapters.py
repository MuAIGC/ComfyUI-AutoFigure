import torch
import numpy as np
from PIL import Image
import io

class TypeAdapter:
    """AutoFigure (PIL) <=> ComfyUI (Tensor) 适配器"""
    
    @staticmethod
    def tensor_to_pil(t_image: torch.Tensor) -> Image.Image:
        """ComfyUI [B,H,W,C] 0-1 -> PIL RGB"""
        if t_image is None:
            return None
        
        # 处理 batch
        if len(t_image.shape) == 4:
            t_image = t_image[0]
        
        # 转为 numpy 0-255
        np_image = (t_image.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(np_image)
    
    @staticmethod
    def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
        """PIL RGB -> ComfyUI [B,H,W,C] 0-1"""
        if pil_image is None:
            return torch.zeros((1, 512, 512, 3))
        
        np_image = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
        return torch.from_numpy(np_image).unsqueeze(0)
    
    @staticmethod
    def svg_to_tensor(svg_code: str, width: int = 1024, height: int = 1024) -> torch.Tensor:
        """SVG 代码渲染为预览图"""
        try:
            import cairosvg
            png_data = cairosvg.svg2png(
                bytestring=svg_code.encode(), 
                output_width=width, 
                output_height=height
            )
            pil_image = Image.open(io.BytesIO(png_data)).convert('RGB')
            return TypeAdapter.pil_to_tensor(pil_image)
        except Exception:
            # 失败返回空白图
            return torch.zeros((1, height, width, 3))
