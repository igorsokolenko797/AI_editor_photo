import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, Dict
from config import config

class SegmentationService:
    def __init__(self):
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.mp_pose = mp.solutions.pose
        
    def segment_human(self, image_array: np.ndarray) -> Optional[np.ndarray]:
        """Сегментация человека с помощью MediaPipe"""
        try:
            with self.mp_selfie_segmentation.SelfieSegmentation(
                model_selection=1
            ) as selfie_segmentation:
                
                results = selfie_segmentation.process(
                    cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                )
                
                if results.segmentation_mask is not None:
                    condition = np.stack(
                        (results.segmentation_mask,) * 3, axis=-1
                    ) > 0.5
                    return condition
                return None
                
        except Exception as e:
            print(f"Error in human segmentation: {e}")
            return None
    
    def detect_pose_landmarks(self, image_array: np.ndarray) -> Optional[Dict]:
        """Детекция ключевых точек позы"""
        try:
            with self.mp_pose.Pose(
                static_image_mode=True,
                min_detection_confidence=0.5,
                model_complexity=1
            ) as pose:
                
                results = pose.process(
                    cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                )
                
                if not results.pose_landmarks:
                    return None
                
                landmarks = results.pose_landmarks.landmark
                h, w = image_array.shape[:2]
                
                points = {}
                point_indices = {
                    'left_shoulder': mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                    'right_shoulder': mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                    'left_hip': mp.solutions.pose.PoseLandmark.LEFT_HIP,
                    'right_hip': mp.solutions.pose.PoseLandmark.RIGHT_HIP,
                    'nose': mp.solutions.pose.PoseLandmark.NOSE
                }
                
                for name, landmark_idx in point_indices.items():
                    landmark = landmarks[landmark_idx]
                    points[name] = (int(landmark.x * w), int(landmark.y * h))
                
                return points
                
        except Exception as e:
            print(f"Error in pose detection: {e}")
            return None
    
    def remove_clothes_background(self, image_array: np.ndarray) -> np.ndarray:
        """Удаление фона с одежды"""
        try:
            # Конвертируем в разные цветовые пространства для лучшего выделения
            hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image_array, cv2.COLOR_BGR2LAB)
            
            # Создаем маски для разных цветов фона
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            
            # Маска для светлых тонов в LAB
            lower_light = np.array([0, 128, 128])
            upper_light = np.array([255, 255, 255])
            mask_light = cv2.inRange(lab, lower_light, upper_light)
            
            # Комбинируем маски
            combined_mask = cv2.bitwise_or(mask_white, mask_light)
            
            # Инвертируем маску (одежда становится белой)
            mask = cv2.bitwise_not(combined_mask)
            
            # Улучшаем маску морфологическими операциями
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            return mask
            
        except Exception as e:
            print(f"Error in clothes background removal: {e}")
            # Возвращаем маску по умолчанию
            return np.ones(image_array.shape[:2], dtype=np.uint8) * 255
