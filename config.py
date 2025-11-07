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
    output_quality: int = 95
    clothes_scale_factor: float = 0.7
    clothes_position_offset: int = 50
    
@dataclass
class APIConfig:
    remove_bg_api_key: str = os.getenv("REMOVE_BG_API_KEY", "")
    request_timeout: int = 30

class Config:
    bot = BotConfig()
    image = ImageProcessingConfig()
    api = APIConfig()

config = Config()
