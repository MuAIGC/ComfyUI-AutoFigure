"""从 server.py 提取的默认配置"""
DEFAULT_SAM_PROMPT = "icon,person,animal,robot"
DEFAULT_PLACEHOLDER_MODE = "label"
DEFAULT_MERGE_THRESHOLD = 0.01
DEFAULT_OPTIMIZE_ITERATIONS = 2

PROVIDER_CONFIGS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_image_model": "google/gemini-3-pro-image-preview",
        "default_svg_model": "google/gemini-3-pro-preview",
    },
    "bianxie": {
        "base_url": "https://api.bianxie.ai/v1",
        "default_image_model": "gemini-3-pro-image-preview",
        "default_svg_model": "gemini-3-pro-preview",
    },
}
