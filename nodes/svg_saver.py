import os
from pathlib import Path

class AF_SVG_Saver:
    """保存 SVG 到指定路径"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "svg_code": ("SVG_CODE",),
                "filename_prefix": ("STRING", {"default": "AutoFigure"}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "./output"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    CATEGORY = "AutoFigure/IO"
    OUTPUT_NODE = True
    
    def save(self, svg_code, filename_prefix, output_dir="./output"):
        # 确保目录存在
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.svg"
        filepath = out_path / filename
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(svg_code)
        
        print(f"[AutoFigure] SVG saved to: {filepath}")
        return (str(filepath),)
