from PIL import Image
from io import BytesIO
from typing import Tuple, Optional
from config import config

class ImageValidator:
    @staticmethod
    def validate_image_format(image_data: bytes) -> bool:
        """Проверка формата изображения"""
        try:
            image = Image.open(BytesIO(image_data))
            format = image.format.lower()
            return format in ['jpeg', 'jpg', 'png', 'webp']
        except:
            return False
    
    @staticmethod
    def get_image_dimensions(image_data: bytes) -> Optional[Tuple[int, int]]:
        """Получение размеров изображения"""
        try:
            image = Image.open(BytesIO(image_data))
            return image.size
        except:
            return None
    
    @staticmethod
    def is_image_too_small(width: int, height: int, min_size: int = 200) -> bool:
        """Проверка минимального размера изображения"""
        return width < min_size or height < min_size
    
    @staticmethod
    def is_image_too_large(width: int, height: int, max_size: int = 4000) -> bool:
        """Проверка максимального размера изображения"""
        return width > max_size or height > max_size
