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
        """ПРОСТОЕ но РАБОЧЕЕ размещение одежды"""
        try:
            print("🎯 Начинаем простое наложение одежды...")
            
            # Конвертируем в numpy
            human_np = np.array(human_image)
            clothes_np = np.array(clothes_image)
            
            print(f"📏 Размеры: человек {human_np.shape}, одежда {clothes_np.shape}")
            
            # Упрощенный подход: наложение по центру с маской
            result_np = self._simple_overlay(human_np, clothes_np)
            
            if result_np is not None:
                print("✅ Наложение завершено успешно")
                # Конвертируем обратно в PIL Image
                return Image.fromarray(result_np)
            else:
                print("❌ Наложение не удалось, возвращаем оригинал")
                return human_image
                
        except Exception as e:
            print(f"❌ Ошибка в place_clothes_smart: {e}")
            return human_image
    
    def _simple_overlay(self, human_np: np.ndarray, clothes_np: np.ndarray) -> Optional[np.ndarray]:
        """Простое наложение одежды по центру - возвращает numpy array"""
        try:
            # Создаем копию человека
            result = human_np.copy()
            h_h, h_w = human_np.shape[:2]
            c_h, c_w = clothes_np.shape[:2]
            
            print(f"🎯 Масштабирование: человек {h_w}x{h_h}, одежда {c_w}x{c_h}")
            
            # Масштабируем одежду под размер человека
            scale_factor = min(
                h_w * 0.6 / c_w,  # 60% ширины человека
                h_h * 0.4 / c_h   # 40% высоты человека
            )
            
            # Ограничиваем масштаб
            scale_factor = max(0.3, min(scale_factor, 1.5))
            
            new_width = int(c_w * scale_factor)
            new_height = int(c_h * scale_factor)
            
            print(f"📐 Новый размер одежды: {new_width}x{new_height} (масштаб: {scale_factor:.2f})")
            
            # Масштабируем одежду
            clothes_resized = cv2.resize(clothes_np, (new_width, new_height), 
                                       interpolation=cv2.INTER_LANCZOS4)
            
            # Получаем маску одежды
            clothes_mask = self.segmentation_service.remove_clothes_background(clothes_resized)
            
            print(f"🎭 Маска одежды: {np.unique(clothes_mask)}")
            
            # Позиция по центру (немного выше центра для футболки)
            x = (h_w - new_width) // 2
            y = h_h // 4  # 25% от верха
            
            print(f"📍 Позиция: x={x}, y={y}")
            
            # Простое наложение с маской
            for i in range(new_height):
                for j in range(new_width):
                    if y + i < h_h and x + j < h_w:  # Проверка границ
                        if clothes_mask[i, j] > 128:  # Если пиксель не фон
                            result[y + i, x + j] = clothes_resized[i, j]
            
            return result
            
        except Exception as e:
            print(f"❌ Ошибка в _simple_overlay: {e}")
            return None
    
    def _place_with_body_points(
        self, 
        human_np: np.ndarray, 
        clothes_np: np.ndarray,
        body_points: Dict
    ) -> Image.Image:
        """Размещение с использованием ключевых точек (упрощенное)"""
        try:
            print("🎯 Используем точки тела для позиционирования...")
            
            # Безопасное извлечение координат
            left_shoulder = body_points.get('left_shoulder', (0, 0))
            right_shoulder = body_points.get('right_shoulder', (0, 0))
            
            # Вычисляем параметры
            shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
            chest_level = min(left_shoulder[1], right_shoulder[1])
            
            print(f"📐 Ширина плеч: {shoulder_width}, уровень груди: {chest_level}")
            
            # Масштабируем одежду под ширину плеч
            c_h, c_w = clothes_np.shape[:2]
            scale_factor = shoulder_width * 1.2 / c_w
            
            new_width = int(c_w * scale_factor)
            new_height = int(c_h * scale_factor)
            
            clothes_resized = cv2.resize(clothes_np, (new_width, new_height))
            clothes_mask = self.segmentation_service.remove_clothes_background(clothes_resized)
            
            # Позиционируем
            start_x = left_shoulder[0] - int(new_width * 0.3)
            start_y = chest_level - int(new_height * 0.1)
            
            # Накладываем
            result = human_np.copy()
            for i in range(new_height):
                for j in range(new_width):
                    if (start_y + i < human_np.shape[0] and start_x + j < human_np.shape[1] and
                        clothes_mask[i, j] > 128):
                        result[start_y + i, start_x + j] = clothes_resized[i, j]
            
            return Image.fromarray(result)
            
        except Exception as e:
            print(f"❌ Ошибка в _place_with_body_points: {e}")
            result_np = self._simple_overlay(human_np, clothes_np)
            return Image.fromarray(result_np) if result_np is not None else Image.fromarray(human_np)
    
    def _place_simple(self, human_np: np.ndarray, clothes_np: np.ndarray) -> Image.Image:
        """Простое размещение (fallback)"""
        result_np = self._simple_overlay(human_np, clothes_np)
        return Image.fromarray(result_np) if result_np is not None else Image.fromarray(human_np)