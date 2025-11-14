import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class SimpleSegmentation:
    """Упрощенная сегментация без MediaPipe"""
    
    def segment_human(self, image_array: np.ndarray) -> np.ndarray:
        """
        Упрощенная сегментация человека на основе контраста и цветов
        """
        try:
            # Конвертируем в разные цветовые пространства
            hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image_array, cv2.COLOR_BGR2LAB)
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            
            # 1. Детекция кожи в HSV
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([25, 255, 255], dtype=np.uint8)
            skin_mask_hsv = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # 2. Детекция по яркости в LAB
            l_channel = lab[:,:,0]
            _, bright_mask = cv2.threshold(l_channel, 150, 255, cv2.THRESH_BINARY)
            
            # 3. Детекция границ для нахождения контуров
            edges = cv2.Canny(gray, 50, 150)
            
            # Комбинируем маски
            combined_mask = cv2.bitwise_or(skin_mask_hsv, bright_mask)
            combined_mask = cv2.bitwise_or(combined_mask, edges)
            
            # Морфологические операции для улучшения маски
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Заполняем дыры
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                filled_mask = np.zeros_like(combined_mask)
                cv2.drawContours(filled_mask, [largest_contour], -1, 255, -1)
                return filled_mask > 0
            else:
                return np.ones(image_array.shape[:2], dtype=bool)
                
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            # Возвращаем полную маску как fallback
            return np.ones(image_array.shape[:2], dtype=bool)
    
    def detect_pose_landmarks(self, image_array: np.ndarray):
        """
        Упрощенное определение ключевых точек на основе геометрии изображения
        """
        try:
            height, width = image_array.shape[:2]
            
            # Простая эвристика для определения позы
            points = {
                'left_shoulder': (width // 4, height // 3),
                'right_shoulder': (3 * width // 4, height // 3),
                'left_hip': (width // 3, 2 * height // 3),
                'right_hip': (2 * width // 3, 2 * height // 3),
                'nose': (width // 2, height // 4)
            }
            
            return points
            
        except Exception as e:
            logger.error(f"Pose detection error: {e}")
            return None
    
    def remove_clothes_background(self, image_array: np.ndarray) -> np.ndarray:
        """Удаление фона с одежды"""
        try:
            # Конвертируем в разные цветовые пространства
            hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            
            # Создаем маски для разных типов фона
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 55, 255])
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            
            # Маска для светлых тонов
            _, mask_light = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Комбинируем маски фона
            background_mask = cv2.bitwise_or(mask_white, mask_light)
            
            # Инвертируем чтобы получить маску одежды
            clothes_mask = cv2.bitwise_not(background_mask)
            
            # Улучшаем маску
            kernel = np.ones((3, 3), np.uint8)
            clothes_mask = cv2.morphologyEx(clothes_mask, cv2.MORPH_CLOSE, kernel)
            clothes_mask = cv2.morphologyEx(clothes_mask, cv2.MORPH_OPEN, kernel)
            
            return clothes_mask
            
        except Exception as e:
            logger.error(f"Clothes background removal error: {e}")
            return np.ones(image_array.shape[:2], dtype=np.uint8) * 255