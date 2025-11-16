import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class SimpleSegmentation:
    def __init__(self):
        pass
    
    def segment_human(self, image_array: np.ndarray) -> np.ndarray:
        """
        Улучшенная сегментация человека
        """
        try:
            # Создаем копию для работы
            img = image_array.copy()
            h, w = img.shape[:2]
            
            # 1. Детекция по цвету кожи (в HSV)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Диапазон цветов кожи в HSV
            lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
            lower_skin2 = np.array([0, 20, 70], dtype=np.uint8) 
            upper_skin2 = np.array([20, 255, 255], dtype=np.uint8)
            
            skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
            skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
            
            # 2. Детекция по градиентам (края)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # 3. Детекция по движению (разность с фоном)
            # Предполагаем что человек контрастирует с фоном
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel = lab[:,:,0]
            
            # Бинаризация по яркости
            _, bright_mask = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 4. Комбинируем все маски
            combined_mask = cv2.bitwise_or(skin_mask, edges)
            combined_mask = cv2.bitwise_or(combined_mask, bright_mask)
            
            # 5. Морфологические операции для улучшения маски
            kernel = np.ones((5, 5), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # 6. Находим самый большой контур (предполагаем что это человек)
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Находим самый большой контур
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Создаем маску только с этим контуром
                final_mask = np.zeros_like(combined_mask)
                cv2.drawContours(final_mask, [largest_contour], -1, 255, -1)
                
                # Увеличиваем маску на 10% чтобы захватить больше области
                kernel_large = np.ones((15, 15), np.uint8)
                final_mask = cv2.dilate(final_mask, kernel_large, iterations=1)
                
                return final_mask > 0
            else:
                # Если контуров нет, возвращаем центральную область
                logger.warning("Не найдены контуры, использую центральную область")
                center_mask = np.zeros((h, w), dtype=bool)
                center_h, center_w = h//2, w//2
                center_mask[center_h-100:center_h+100, center_w-50:center_w+50] = True
                return center_mask
                
        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            # Возвращаем центральную область как fallback
            h, w = image_array.shape[:2]
            fallback_mask = np.zeros((h, w), dtype=bool)
            fallback_mask[h//4:3*h//4, w//4:3*w//4] = True
            return fallback_mask
    
    def detect_pose_landmarks(self, image_array: np.ndarray):
        """
        Улучшенное определение ключевых точек
        """
        try:
            height, width = image_array.shape[:2]
            
            # Получаем маску для лучшего определения позы
            mask = self.segment_human(image_array)
            
            if mask is not None and np.any(mask):
                # Находим bounding box человека
                y_coords, x_coords = np.where(mask)
                
                if len(x_coords) > 0 and len(y_coords) > 0:
                    x_min, x_max = np.min(x_coords), np.max(x_coords)
                    y_min, y_max = np.min(y_coords), np.max(y_coords)
                    
                    # Вычисляем точки на основе bounding box
                    human_width = x_max - x_min
                    human_height = y_max - y_min
                    
                    points = {
                        'left_shoulder': (x_min + human_width * 0.2, y_min + human_height * 0.25),
                        'right_shoulder': (x_min + human_width * 0.8, y_min + human_height * 0.25),
                        'left_hip': (x_min + human_width * 0.3, y_min + human_height * 0.7),
                        'right_hip': (x_min + human_width * 0.7, y_min + human_height * 0.7),
                        'nose': (x_min + human_width * 0.5, y_min + human_height * 0.1)
                    }
                    
                    # Преобразуем в целые числа
                    points = {k: (int(v[0]), int(v[1])) for k, v in points.items()}
                    return points
            
            # Fallback: геометрические точки
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
        """Улучшенное удаление фона с одежды"""
        try:
            # Множественные подходы к удалению фона
            hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            lab = cv2.cvtColor(image_array, cv2.COLOR_BGR2LAB)
            
            # 1. Удаление белого фона
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            
            # 2. Удаление светлых тонов
            _, mask_light_gray = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            
            # 3. Удаление по яркости в LAB
            l_channel = lab[:,:,0]
            _, mask_light_lab = cv2.threshold(l_channel, 220, 255, cv2.THRESH_BINARY)
            
            # Комбинируем все маски фона
            background_mask = cv2.bitwise_or(mask_white, mask_light_gray)
            background_mask = cv2.bitwise_or(background_mask, mask_light_lab)
            
            # Инвертируем чтобы получить маску одежды
            clothes_mask = cv2.bitwise_not(background_mask)
            
            # Улучшаем маску
            kernel = np.ones((3, 3), np.uint8)
            clothes_mask = cv2.morphologyEx(clothes_mask, cv2.MORPH_CLOSE, kernel)
            clothes_mask = cv2.morphologyEx(clothes_mask, cv2.MORPH_OPEN, kernel)
            
            # Заполняем мелкие дыры
            contours, _ = cv2.findContours(clothes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                filled_mask = np.zeros_like(clothes_mask)
                cv2.drawContours(filled_mask, [largest_contour], -1, 255, -1)
                clothes_mask = filled_mask
            
            return clothes_mask
            
        except Exception as e:
            logger.error(f"Clothes background removal error: {e}")
            return np.ones(image_array.shape[:2], dtype=np.uint8) * 255