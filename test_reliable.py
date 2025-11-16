import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from services.reliable_placer import ReliableClothesPlacer
from PIL import Image
import numpy as np
import cv2

def create_basic_test_images():
    """Создание БАЗОВЫХ тестовых изображений"""
    print("🎨 Создаем БАЗОВЫЕ тестовые изображения...")
    
    # 1. Человек - простой силуэт
    human_img = Image.new('RGB', (400, 600), color='lightgray')
    human_array = np.array(human_img)
    # Простой силуэт
    cv2.rectangle(human_array, (150, 100), (250, 400), (100, 100, 100), -1)
    
    # 2. Одежда - цветной прямоугольник на белом фоне
    clothes_img = Image.new('RGB', (200, 150), color='white')
    clothes_array = np.array(clothes_img)
    # Цветной прямоугольник
    cv2.rectangle(clothes_array, (50, 25), (150, 125), (0, 100, 200), -1)  # Цветной
    
    human_pil = Image.fromarray(human_array)
    clothes_pil = Image.fromarray(clothes_array)
    
    human_pil.save("debug_human_basic.jpg")
    clothes_pil.save("debug_clothes_basic.jpg")
    
    print("✅ Базовые изображения созданы")
    return human_pil, clothes_pil

def test_reliable_method():
    """ТЕСТ НАДЕЖНОГО МЕТОДА"""
    print("🧪 ТЕСТ НАДЕЖНОГО МЕТОДА")
    
    human_img, clothes_img = create_basic_test_images()
    placer = ReliableClothesPlacer()
    
    print("🎯 Запускаем надежный метод...")
    result = placer.process_tryon(human_img, clothes_img)
    
    if result:
        result.save("debug_reliable_result.jpg")
        print("✅ Надежный метод завершен")
        
        # Простой анализ
        result_array = np.array(result)
        original_array = np.array(human_img)
        
        # Сравниваем с оригиналом
        difference = np.sum(result_array != original_array)
        print(f"📊 Разница с оригиналом: {difference} пикселей")
        
        if difference > 500:
            print("🎉 УСПЕХ! Изображение изменилось - наложение работает!")
            return True
        else:
            print("❌ Изображение почти не изменилось")
            return False
    else:
        print("❌ Метод не сработал")
        return False

if __name__ == "__main__":
    print("=" * 50)
    success = test_reliable_method()
    print("=" * 50)
    
    if success:
        print("🎉 НАДЕЖНЫЙ МЕТОД РАБОТАЕТ!")
    else:
        print("💥 НАДЕЖНЫЙ МЕТОД НЕ РАБОТАЕТ!")
    
    print("=" * 50)