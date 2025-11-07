from typing import Optional, Tuple
from PIL import Image
import cv2
import numpy as np

from .segmentation import SegmentationService
from .clothes_placer import ClothesPlacer
from utils.file_handlers import FileHandler
from config import config

class ImageProcessor:
    def __init__(self):
        self.segmentation_service = SegmentationService()
        self.clothes_placer = ClothesPlacer()
        self.file_handler = FileHandler()
    
    async def process_try_on(
        self, 
        human_image_data: bytes, 
        clothes_image_data: bytes
    ) -> Optional[bytes]:
        """Основной метод обработки примерки"""
        try:
            # Конвертируем в PIL Image
            human_image = self.file_handler.bytes_to_pil_image(human_image_data)
            clothes_image = self.file_handler.bytes_to_pil_image(clothes_image_data)
            
            if not human_image or not clothes_image:
                return None
            
            # Конвертируем в OpenCV для обработки
            human_cv = self.file_handler.pil_to_cv2(human_image)
            clothes_cv = self.file_handler.pil_to_cv2(clothes_image)
            
            # Детектируем позу для умного позиционирования
            body_points = self.segmentation_service.detect_pose_landmarks(human_cv)
            
            # Выполняем примерку
            result_image = self.clothes_placer.place_clothes_smart(
                human_image, clothes_image, body_points
            )
            
            # Конвертируем обратно в bytes
            return self.file_handler.pil_to_bytes(result_image)
            
        except Exception as e:
            print(f"Error in image processing: {e}")
            return None
    
    def validate_images(
        self, 
        human_image_data: bytes, 
        clothes_image_data: bytes
    ) -> Tuple[bool, str]:
        """Валидация входных изображений"""
        from utils.validators import ImageValidator
        
        # Проверка размера файлов
        if not self.file_handler.validate_image_size(human_image_data):
            return False, "Фото человека слишком большое"
        
        if not self.file_handler.validate_image_size(clothes_image_data):
            return False, "Фото одежды слишком большое"
        
        # Проверка форматов
        if not ImageValidator.validate_image_format(human_image_data):
            return False, "Неверный формат фото человека"
        
        if not ImageValidator.validate_image_format(clothes_image_data):
            return False, "Неверный формат фото одежды"
        
        # Проверка размеров
        human_dims = ImageValidator.get_image_dimensions(human_image_data)
        clothes_dims = ImageValidator.get_image_dimensions(clothes_image_data)
        
        if not human_dims or not clothes_dims:
            return False, "Не удалось прочитать размеры изображений"
        
        if ImageValidator.is_image_too_small(*human_dims):
            return False, "Фото человека слишком маленькое"
        
        if ImageValidator.is_image_too_small(*clothes_dims):
            return False, "Фото одежды слишком маленькое"
        
        return True, "OK"
