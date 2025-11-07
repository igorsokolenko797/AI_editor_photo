import cv2
import numpy as np
from PIL import Image
from typing import Optional, Dict, Tuple
from config import config

class ClothesPlacer:
    def __init__(self):
        self.segmentation_service = SegmentationService()
    
    def place_clothes_smart(
        self, 
        human_image: Image.Image, 
        clothes_image: Image.Image,
        body_points: Optional[Dict] = None
    ) -> Image.Image:
        """Умное размещение одежды на человеке"""
        human_np = np.array(human_image)
        clothes_np = np.array(clothes_image)
        
        if body_points:
            return self._place_with_body_points(human_np, clothes_np, body_points)
        else:
            return self._place_simple(human_np, clothes_np)
    
    def _place_with_body_points(
        self, 
        human_np: np.ndarray, 
        clothes_np: np.ndarray,
        body_points: Dict
    ) -> Image.Image:
        """Размещение с использованием ключевых точек тела"""
        try:
            # Получаем точки плеч
            left_shoulder = body_points.get('left_shoulder')
            right_shoulder = body_points.get('right_shoulder')
            nose = body_points.get('nose')
            
            if not all([left_shoulder, right_shoulder]):
                return self._place_simple(human_np, clothes_np)
            
            # Вычисляем параметры для размещения
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            chest_level = min(left_shoulder[1], right_shoulder[1])
            
            # Масштабируем одежду
            clothes_height, clothes_width = clothes_np.shape[:2]
            scale_factor = shoulder_width * 1.3 / clothes_width
            
            new_width = int(clothes_width * scale_factor)
            new_height = int(clothes_height * scale_factor)
            
            clothes_resized = cv2.resize(clothes_np, (new_width, new_height))
            
            # Создаем маску для одежды
            clothes_mask = self.segmentation_service.remove_clothes_background(clothes_resized)
            
            # Позиционируем одежду
            start_x = left_shoulder[0] - int(new_width * 0.4)
            start_y = chest_level - int(new_height * 0.2)
            
            # Корректируем позицию если выходит за границы
            start_x = max(0, min(start_x, human_np.shape[1] - new_width))
            start_y = max(0, min(start_y, human_np.shape[0] - new_height))
            
            # Накладываем одежду
            result = self._blend_images(
                human_np, clothes_resized, clothes_mask, start_x, start_y
            )
            
            return Image.fromarray(result)
            
        except Exception as e:
            print(f"Error in smart placement: {e}")
            return self._place_simple(human_np, clothes_np)
    
    def _place_simple(self, human_np: np.ndarray, clothes_np: np.ndarray) -> Image.Image:
        """Простое размещение одежды"""
        human_height, human_width = human_np.shape[:2]
        clothes_height, clothes_width = clothes_np.shape[:2]
        
        # Масштабируем одежду
        scale_factor = min(
            human_width * config.image.clothes_scale_factor / clothes_width,
            human_height * 0.4 / clothes_height
        )
        
        new_width = int(clothes_width * scale_factor)
        new_height = int(clothes_height * scale_factor)
        
        clothes_resized = cv2.resize(clothes_np, (new_width, new_height))
        clothes_mask = self.segmentation_service.remove_clothes_background(clothes_resized)
        
        # Позиционируем по центру
        start_x = (human_width - new_width) // 2
        start_y = human_height // 3
        
        # Накладываем одежду
        result = self._blend_images(
            human_np, clothes_resized, clothes_mask, start_x, start_y
        )
        
        return Image.fromarray(result)
    
    def _blend_images(
        self, 
        background: np.ndarray,
        foreground: np.ndarray,
        mask: np.ndarray,
        x: int, 
        y: int
    ) -> np.ndarray:
        """Смешивание изображений с использованием маски"""
        result = background.copy()
        fg_height, fg_width = foreground.shape[:2]
        bg_height, bg_width = background.shape[:2]
        
        # Обрезаем если выходит за границы
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bg_width, x + fg_width), min(bg_height, y + fg_height)
        
        fg_x1, fg_y1 = max(0, -x), max(0, -y)
        fg_x2, fg_y2 = fg_x1 + (x2 - x1), fg_y1 + (y2 - y1)
        
        # Применяем маску и накладываем
        for channel in range(3):
            result[y1:y2, x1:x2, channel] = (
                background[y1:y2, x1:x2, channel] * 
                (1 - mask[fg_y1:fg_y2, fg_x1:fg_x2] / 255.0) +
                foreground[fg_y1:fg_y2, fg_x1:fg_x2, channel] * 
                (mask[fg_y1:fg_y2, fg_x1:fg_x2] / 255.0)
            )
        
        return result
