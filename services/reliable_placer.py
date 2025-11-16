import cv2
import numpy as np
from PIL import Image
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ReliableClothesPlacer:
    def __init__(self):
        logger.info("✅ ReliableClothesPlacer инициализирован")
    
    def process_tryon(self, human_image: Image.Image, clothes_image: Image.Image) -> Optional[Image.Image]:
        """НАДЕЖНЫЙ метод наложения - основан на том, что РАБОТАЛО"""
        try:
            logger.info("🎯 Запускаем НАДЕЖНЫЙ метод наложения...")
            
            # Конвертируем в numpy
            human_np = np.array(human_image)
            clothes_np = np.array(clothes_image)
            
            # Конвертируем RGB to BGR для OpenCV
            human_bgr = cv2.cvtColor(human_np, cv2.COLOR_RGB2BGR)
            clothes_bgr = cv2.cvtColor(clothes_np, cv2.COLOR_RGB2BGR)
            
            logger.info(f"📏 Размеры: человек {human_bgr.shape}, одежда {clothes_bgr.shape}")
            
            # Используем ПРОВЕРЕННЫЙ метод
            result = self._reliable_overlay(human_bgr, clothes_bgr)
            
            # Конвертируем обратно
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            
            logger.info("✅ НАДЕЖНЫЙ метод завершен")
            return Image.fromarray(result_rgb)
            
        except Exception as e:
            logger.error(f"❌ Ошибка надежного метода: {e}")
            return None
    
    def _reliable_overlay(self, human: np.ndarray, clothes: np.ndarray) -> np.ndarray:
        """ПРОВЕРЕННЫЙ метод наложения"""
        try:
            # Создаем копию человека
            result = human.copy()
            h_h, h_w = human.shape[:2]
            c_h, c_w = clothes.shape[:2]
            
            logger.info(f"📍 Человек: {h_w}x{h_h}, Одежда: {c_w}x{c_h}")
            
            # ПРОСТОЙ масштаб
            scale = min(h_w * 0.6 / c_w, h_h * 0.4 / c_h)
            new_w = int(c_w * scale)
            new_h = int(c_h * scale)
            
            logger.info(f"📐 Масштаб: {scale:.2f}, новый размер: {new_w}x{new_h}")
            
            # Масштабируем одежду
            clothes_resized = cv2.resize(clothes, (new_w, new_h))
            
            # ПРОСТАЯ позиция
            x = (h_w - new_w) // 2
            y = h_h // 4
            
            logger.info(f"🎯 Позиция: x={x}, y={y}")
            
            # ПРОСТОЙ и ЭФФЕКТИВНЫЙ метод наложения
            self._simple_reliable_overlay(result, clothes_resized, x, y)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка надежного наложения: {e}")
            return human
    
    def _simple_reliable_overlay(self, background: np.ndarray, foreground: np.ndarray, x: int, y: int):
        """ПРОСТОЙ и ЭФФЕКТИВНЫЙ метод наложения"""
        try:
            h_fg, w_fg = foreground.shape[:2]
            
            logger.info("🔧 Начинаем простое надежное наложение...")
            
            # СЧИТАЕМ сколько пикселей мы обработали
            processed_pixels = 0
            
            for i in range(h_fg):
                for j in range(w_fg):
                    target_y = y + i
                    target_x = x + j
                    
                    # Проверяем границы
                    if 0 <= target_y < background.shape[0] and 0 <= target_x < background.shape[1]:
                        # Берем пиксель одежды
                        clothes_pixel = foreground[i, j]
                        
                        # ПРОСТАЯ логика: если пиксель НЕ белый - накладываем
                        is_white = (clothes_pixel[0] > 220 and clothes_pixel[1] > 220 and clothes_pixel[2] > 220)
                        
                        if not is_white:
                            # ПРОСТО ЗАМЕНЯЕМ пиксель
                            background[target_y, target_x] = clothes_pixel
                            processed_pixels += 1
            
            logger.info(f"✅ Обработано пикселей: {processed_pixels}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка в простом наложении: {e}")