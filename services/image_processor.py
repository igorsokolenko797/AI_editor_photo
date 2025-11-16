from typing import Optional, Tuple
from PIL import Image
import numpy as np
import logging

from services.reliable_placer import ReliableClothesPlacer  # ← НАДЕЖНЫЙ ПЛАСЕР
from utils.file_handlers import FileHandler
from config import config

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.reliable_placer = ReliableClothesPlacer()  # ← НАДЕЖНЫЙ ПЛАСЕР
        self.file_handler = FileHandler()
        logger.info("✅ ImageProcessor использует ReliableClothesPlacer")
    
    async def process_try_on(
        self, 
        human_image_data: bytes, 
        clothes_image_data: bytes
    ) -> Optional[bytes]:
        """ОСНОВНОЙ метод - используем НАДЕЖНЫЙ пласер"""
        try:
            logger.info("🔄 ImageProcessor: начинаем обработку...")
            
            # Конвертируем в PIL Image
            human_image = self.file_handler.bytes_to_pil_image(human_image_data)
            clothes_image = self.file_handler.bytes_to_pil_image(clothes_image_data)
            
            if not human_image or not clothes_image:
                logger.error("❌ Не удалось конвертировать изображения")
                return None
            
            logger.info("🎯 ImageProcessor: передаем в ReliableClothesPlacer...")
            
            # Используем НАДЕЖНЫЙ пласер
            result_image = self.reliable_placer.process_tryon(human_image, clothes_image)
            
            if result_image is None:
                logger.error("❌ ReliableClothesPlacer вернул None")
                return None
            
            # Конвертируем обратно в bytes
            result_bytes = self.file_handler.pil_to_bytes(result_image, quality=95)
            
            if result_bytes:
                logger.info(f"✅ ImageProcessor: обработка завершена. Размер: {len(result_bytes)} байт")
            else:
                logger.error("❌ Не удалось конвертировать результат в bytes")
            
            return result_bytes
            
        except Exception as e:
            logger.error(f"❌ Ошибка в ImageProcessor: {e}")
            return None
    
    def validate_images(
        self, 
        human_image_data: bytes, 
        clothes_image_data: bytes
    ) -> Tuple[bool, str]:
        """Валидация входных изображений"""
        from utils.validators import ImageValidator
        
        # Ваш существующий код валидации
        return True, "OK"