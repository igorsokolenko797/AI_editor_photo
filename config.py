# config.py
import os
from dataclasses import dataclass

@dataclass
class BotConfig:
    token: str = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
    admin_id: int = int(os.getenv("ADMIN_ID", 0))
    
@dataclass
class ImageProcessingConfig:
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    supported_formats: tuple = ('.jpg', '.jpeg', '.png', '.webp')
    output_quality: int = 95  # ← Качество по умолчанию
    clothes_scale_factor: float = 0.8
    clothes_position_offset: int = 50
    target_width: int = 600   # ← Оптимальный размер для скорости и качества
    target_height: int = 800
    
@dataclass
class APIConfig:
    remove_bg_api_key: str = os.getenv("REMOVE_BG_API_KEY", "")
    request_timeout: int = 60
    download_timeout: int = 30

class Config:
    bot = BotConfig()
    image = ImageProcessingConfig()
    api = APIConfig()

config = Config()