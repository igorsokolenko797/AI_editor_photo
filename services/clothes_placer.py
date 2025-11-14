import cv2
import numpy as np
from PIL import Image
from typing import Optional, Dict, Tuple
from config import config
from services.segmentation import SimpleSegmentation

class ClothesPlacer:
    def __init__(self):
        self.segmentation_service = SimpleSegmentation()
    
    def place_clothes_smart(
        self, 
        human_image: Image.Image, 
        clothes_image: Image.Image,
        body_points: Optional[Dict] = None
    ) -> Image.Image:
        """Умное размещение одежды на человеке"""
        # Конвертируем в numpy arrays
        human_np = np.array(human_image)
        clothes_np = np.array(clothes_image)
        
        # Улучшаем качество входных изображений
        human_np = self._enhance_image(human_np)
        clothes_np = self._enhance_image(clothes_np)
        
        if body_points:
            return self._place_with_body_points(human_np, clothes_np, body_points)
        else:
            return self._place_simple(human_np, clothes_np)
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Улучшение качества изображения"""
        try:
            # Увеличиваем резкость
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Немного увеличиваем контраст
            lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except:
            return image
    
    def _place_with_body_points(
        self, 
        human_np: np.ndarray, 
        clothes_np: np.ndarray,
        body_points: Dict
    ) -> Image.Image:
        """Размещение с использованием ключевых точек тела"""
        try:
            # Получаем точки плеч с проверкой на None
            left_shoulder = body_points.get('left_shoulder')
            right_shoulder = body_points.get('right_shoulder')
            
            # Проверяем что точки существуют и являются кортежами
            if not left_shoulder or not right_shoulder:
                return self._place_simple(human_np, clothes_np)
            
            if not isinstance(left_shoulder, (tuple, list)) or not isinstance(right_shoulder, (tuple, list)):
                return self._place_simple(human_np, clothes_np)
            
            if len(left_shoulder) < 2 or len(right_shoulder) < 2:
                return self._place_simple(human_np, clothes_np)
            
            # Безопасно вычисляем
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            chest_level = min(left_shoulder[1], right_shoulder[1])
            
            # Масштабируем одежду с учетом качества
            clothes_height, clothes_width = clothes_np.shape[:2]
            
            # Более точный расчет масштаба
            scale_factor = shoulder_width * 1.1 / clothes_width  # Уменьшил множитель для лучшего fit
            
            # Ограничиваем масштаб
            scale_factor = max(0.3, min(scale_factor, 2.0))
            
            new_width = max(50, int(clothes_width * scale_factor))  # Минимальная ширина
            new_height = max(50, int(clothes_height * scale_factor))
            
            # Используем интерполяцию лучшего качества
            clothes_resized = cv2.resize(clothes_np, (new_width, new_height), 
                                       interpolation=cv2.INTER_LANCZOS4)
            
            # Создаем маску для одежды с лучшим качеством
            clothes_mask = self._create_high_quality_mask(clothes_resized)
            
            # Позиционируем одежду более точно
            start_x = left_shoulder[0] - int(new_width * 0.35)  # Более точное позиционирование
            start_y = chest_level - int(new_height * 0.15)
            
            # Корректируем позицию если выходит за границы
            start_x = max(0, min(start_x, human_np.shape[1] - new_width))
            start_y = max(0, min(start_y, human_np.shape[0] - new_height))
            
            # Накладываем одежду с улучшенным блендингом
            result = self._blend_images_advanced(
                human_np, clothes_resized, clothes_mask, start_x, start_y
            )
            
            # Улучшаем итоговое изображение
            result = self._enhance_image(result)
            
            return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            
        except Exception as e:
            print(f"Error in smart placement: {e}")
            return self._place_simple(human_np, clothes_np)
    
    def _create_high_quality_mask(self, clothes_np: np.ndarray) -> np.ndarray:
        """Создание высококачественной маски для одежды"""
        try:
            # Получаем базовую маску
            base_mask = self.segmentation_service.remove_clothes_background(clothes_np)
            
            # Улучшаем маску
            kernel = np.ones((3, 3), np.uint8)
            
            # Закрываем мелкие дыры
            mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, kernel)
            
            # Открываем для удаления шума
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Размываем края для плавного перехода
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            return mask
            
        except Exception as e:
            print(f"Mask creation error: {e}")
            return np.ones(clothes_np.shape[:2], dtype=np.uint8) * 255
    
    def _place_simple(self, human_np: np.ndarray, clothes_np: np.ndarray) -> Image.Image:
        """Простое размещение одежды с улучшенным качеством"""
        human_height, human_width = human_np.shape[:2]
        clothes_height, clothes_width = clothes_np.shape[:2]
        
        # Масштабируем одежду с лучшей интерполяцией
        scale_factor = min(
            human_width * 0.6 / clothes_width,  # Увеличил масштаб
            human_height * 0.5 / clothes_height
        )
        
        scale_factor = max(0.3, min(scale_factor, 1.5))  # Ограничиваем масштаб
        
        new_width = max(50, int(clothes_width * scale_factor))
        new_height = max(50, int(clothes_height * scale_factor))
        
        # Высококачественное масштабирование
        clothes_resized = cv2.resize(clothes_np, (new_width, new_height),
                                   interpolation=cv2.INTER_LANCZOS4)
        
        # Создаем улучшенную маску
        clothes_mask = self._create_high_quality_mask(clothes_resized)
        
        # Позиционируем по центру
        start_x = (human_width - new_width) // 2
        start_y = human_height // 4  # Подняли выше
        
        # Накладываем с улучшенным блендингом
        result = self._blend_images_advanced(
            human_np, clothes_resized, clothes_mask, start_x, start_y
        )
        
        # Улучшаем итоговое изображение
        result = self._enhance_image(result)
        
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    
    def _blend_images_advanced(
        self, 
        background: np.ndarray,
        foreground: np.ndarray,
        mask: np.ndarray,
        x: int, 
        y: int
    ) -> np.ndarray:
        """Улучшенное смешивание изображений с плавными переходами"""
        result = background.copy()
        fg_height, fg_width = foreground.shape[:2]
        bg_height, bg_width = background.shape[:2]
        
        # Определяем область наложения
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bg_width, x + fg_width), min(bg_height, y + fg_height)
        
        # Если область наложения пустая, возвращаем оригинал
        if x1 >= x2 or y1 >= y2:
            return result
        
        # Вычисляем соответствующие области
        fg_x1, fg_y1 = max(0, -x), max(0, -y)
        fg_x2, fg_y2 = min(fg_width, bg_width - x), min(fg_height, bg_height - y)
        
        # Извлекаем области
        bg_region = result[y1:y2, x1:x2].astype(np.float32)
        fg_region = foreground[fg_y1:fg_y2, fg_x1:fg_x2].astype(np.float32)
        mask_region = mask[fg_y1:fg_y2, fg_x1:fg_x2].astype(np.float32) / 255.0
        
        # Если маска 3-канальная, преобразуем в 1-канальную
        if len(mask_region.shape) == 3:
            mask_region = cv2.cvtColor(mask_region.astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.0
        
        # Создаем 3-канальную маску
        if len(mask_region.shape) == 2:
            mask_region = np.stack([mask_region] * 3, axis=-1)
        
        # Плавное наложение с учетом альфа-канала
        blended_region = bg_region * (1 - mask_region) + fg_region * mask_region
        
        # Вставляем обратно
        result[y1:y2, x1:x2] = blended_region.astype(np.uint8)
        
        return result
    
    def _blend_images(
        self, 
        background: np.ndarray,
        foreground: np.ndarray,
        mask: np.ndarray,
        x: int, 
        y: int
    ) -> np.ndarray:
        """Старая версия для совместимости"""
        return self._blend_images_advanced(background, foreground, mask, x, y)