from typing import Optional, Tuple
from PIL import Image
import cv2
import numpy as np
import logging

from services.segmentation import SimpleSegmentation
from services.clothes_placer import ClothesPlacer
from utils.file_handlers import FileHandler
from config import config

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.segmentation_service = SimpleSegmentation()
        self.clothes_placer = ClothesPlacer()
        self.file_handler = FileHandler()
    
    async def process_try_on(
        self, 
        human_image_data: bytes, 
        clothes_image_data: bytes
    ) -> Optional[bytes]:
        """Основной метод обработки примерки с улучшенным качеством"""
        try:
            # Конвертируем в PIL Image
            human_image = self.file_handler.bytes_to_pil_image(human_image_data)
            clothes_image = self.file_handler.bytes_to_pil_image(clothes_image_data)
            
            if not human_image or not clothes_image:
                logger.error("❌ Не удалось конвертировать изображения")
                return None
            
            # Масштабируем с сохранением пропорций
            human_image = self._resize_with_aspect_ratio(human_image, config.image.target_width, config.image.target_height)
            clothes_image = self._resize_with_aspect_ratio(clothes_image, config.image.target_width // 2, config.image.target_height // 2)
            
            logger.info(f"🖼️ Размеры после масштабирования: человек={human_image.size}, одежда={clothes_image.size}")
            
            # Конвертируем в OpenCV для обработки
            human_cv = self.file_handler.pil_to_cv2(human_image)
            clothes_cv = self.file_handler.pil_to_cv2(clothes_image)
            
            # Детектируем позу для умного позиционирования
            body_points = self.segmentation_service.detect_pose_landmarks(human_cv)
            logger.info(f"📍 Обнаружены точки тела: {body_points is not None}")
            
            # Выполняем примерку
            result_image = self.clothes_placer.place_clothes_smart(
                human_image, clothes_image, body_points
            )
            
            if result_image is None:
                logger.error("❌ ClothesPlacer вернул None")
                return None
            
            # Конвертируем обратно в bytes с высоким качеством
            result_bytes = self.file_handler.pil_to_bytes(result_image, quality=config.image.output_quality)
            
            if result_bytes:
                logger.info(f"✅ Обработка завершена. Размер результата: {len(result_bytes)} байт")
            else:
                logger.error("❌ Не удалось конвертировать результат в bytes")
            
            return result_bytes
            
        except Exception as e:
            logger.error(f"❌ Ошибка в обработке изображений: {e}")
            return None

    def _resize_with_aspect_ratio(self, image: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """Масштабирование с сохранением пропорций"""
        try:
            original_width, original_height = image.size
            
            # Вычисляем соотношения
            width_ratio = target_width / original_width
            height_ratio = target_height / original_height
            
            # Используем минимальное соотношение чтобы сохранить пропорции
            ratio = min(width_ratio, height_ratio)
            
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            
            # Убедимся что размеры не меньше минимальных
            new_width = max(100, new_width)
            new_height = max(100, new_height)
            
            logger.debug(f"🔄 Масштабирование: {original_width}x{original_height} -> {new_width}x{new_height}")
            
            return image.resize((new_width, new_height), Image.LANCZOS)
            
        except Exception as e:
            logger.error(f"❌ Ошибка масштабирования: {e}")
            return image  # Возвращаем оригинал в случае ошибки
    
    def validate_images(
        self, 
        human_image_data: bytes, 
        clothes_image_data: bytes
    ) -> Tuple[bool, str]:
        """Валидация входных изображений"""
        from utils.validators import ImageValidator
        
        # Проверка что данные не пустые
        if not human_image_data or len(human_image_data) == 0:
            return False, "Фото человека пустое"
        
        if not clothes_image_data or len(clothes_image_data) == 0:
            return False, "Фото одежды пустое"
        
        # Проверка размера файлов
        if not self.file_handler.validate_image_size(human_image_data):
            return False, "Фото человека слишком большое (максимум 10MB)"
        
        if not self.file_handler.validate_image_size(clothes_image_data):
            return False, "Фото одежды слишком большое (максимум 10MB)"
        
        # Проверка форматов
        if not ImageValidator.validate_image_format(human_image_data):
            return False, "Неверный формат фото человека. Используйте JPG, PNG или WebP"
        
        if not ImageValidator.validate_image_format(clothes_image_data):
            return False, "Неверный формат фото одежды. Используйте JPG, PNG или WebP"
        
        # Проверка размеров изображений
        human_dims = ImageValidator.get_image_dimensions(human_image_data)
        clothes_dims = ImageValidator.get_image_dimensions(clothes_image_data)
        
        if not human_dims:
            return False, "Не удалось определить размер фото человека"
        
        if not clothes_dims:
            return False, "Не удалось определить размер фото одежды"
        
        human_width, human_height = human_dims
        clothes_width, clothes_height = clothes_dims
        
        # Минимальные размеры
        if human_width < 100 or human_height < 100:
            return False, "Фото человека слишком маленькое (минимум 100x100 пикселей)"
        
        if clothes_width < 50 or clothes_height < 50:
            return False, "Фото одежды слишком маленькое (минимум 50x50 пикселей)"
        
        logger.info(f"✅ Валидация пройдена: человек={human_dims}, одежда={clothes_dims}")
        return True, "OK"