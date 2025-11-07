import aiohttp
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from typing import Optional, Tuple
from config import config

class FileHandler:
    @staticmethod
    async def download_telegram_file(bot, file_id: str) -> Optional[bytes]:
        """Скачивание файла из Telegram"""
        try:
            file = await bot.get_file(file_id)
            file_path = file.file_path
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f'https://api.telegram.org/file/bot{config.bot.token}/{file_path}',
                    timeout=aiohttp.ClientTimeout(total=config.api.request_timeout)
                ) as resp:
                    if resp.status == 200:
                        return await resp.read()
        except Exception as e:
            print(f"Error downloading file: {e}")
            return None
    
    @staticmethod
    def bytes_to_pil_image(image_data: bytes) -> Optional[Image.Image]:
        """Конвертация bytes в PIL Image"""
        try:
            return Image.open(BytesIO(image_data)).convert('RGB')
        except Exception as e:
            print(f"Error converting bytes to PIL image: {e}")
            return None
    
    @staticmethod
    def pil_to_bytes(image: Image.Image, format: str = 'JPEG') -> Optional[bytes]:
        """Конвертация PIL Image в bytes"""
        try:
            output_buffer = BytesIO()
            image.save(output_buffer, format=format, quality=config.image.output_quality)
            output_buffer.seek(0)
            return output_buffer.getvalue()
        except Exception as e:
            print(f"Error converting PIL to bytes: {e}")
            return None
    
    @staticmethod
    def pil_to_cv2(image: Image.Image) -> np.ndarray:
        """Конвертация PIL Image в OpenCV format"""
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def cv2_to_pil(image: np.ndarray) -> Image.Image:
        """Конвертация OpenCV image в PIL"""
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    @staticmethod
    def validate_image_size(image_data: bytes) -> bool:
        """Проверка размера изображения"""
        return len(image_data) <= config.image.max_file_size
