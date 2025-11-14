import aiohttp
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from typing import Optional, Tuple
from config import config
import asyncio

class FileHandler:
    @staticmethod
    async def download_telegram_file(bot, file_id: str) -> Optional[bytes]:
        """Скачивание файла из Telegram с увеличенным таймаутом"""
        try:
            file = await bot.get_file(file_id)
            file_path = file.file_path
            
            # Создаем сессию с увеличенным таймаутом
            timeout = aiohttp.ClientTimeout(total=config.api.download_timeout)
            connector = aiohttp.TCPConnector(limit=10)
            
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                async with session.get(
                    f'https://api.telegram.org/file/bot{config.bot.token}/{file_path}'
                ) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        print(f"✅ Файл загружен, размер: {len(data)} байт")
                        return data
                    else:
                        print(f"❌ Ошибка загрузки: {resp.status}")
                        return None
                        
        except asyncio.TimeoutError:
            print("❌ Таймаут при загрузке файла")
            return None
        except Exception as e:
            print(f"❌ Ошибка загрузки файла: {e}")
            return None
    
    @staticmethod
    def bytes_to_pil_image(image_data: bytes) -> Optional[Image.Image]:
        """Конвертация bytes в PIL Image"""
        try:
            return Image.open(BytesIO(image_data)).convert('RGB')
        except Exception as e:
            print(f"❌ Ошибка конвертации в PIL: {e}")
            return None
    
    @staticmethod
    def pil_to_bytes(image: Image.Image, format: str = 'JPEG', quality: int = None) -> Optional[bytes]:
        """Конвертация PIL Image в bytes с поддержкой качества"""
        try:
            output_buffer = BytesIO()
            
            # Устанавливаем качество если передано, иначе используем настройки по умолчанию
            if quality is not None:
                image.save(output_buffer, format=format, quality=quality, optimize=True)
            else:
                image.save(output_buffer, format=format, quality=config.image.output_quality, optimize=True)
                
            output_buffer.seek(0)
            return output_buffer.getvalue()
        except Exception as e:
            print(f"❌ Ошибка конвертации в bytes: {e}")
            return None
    
    @staticmethod
    def pil_to_cv2(image: Image.Image) -> np.ndarray:
        """Конвертация PIL Image в OpenCV format"""
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    @staticmethod
    def cv2_to_pil(image: np.ndarray) -> Image.Image:
        """Конвертация OpenCV image в PIL"""
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    @staticmethod
    def validate_image_size(image_data: bytes) -> bool:
        """Проверка размера изображения"""
        return len(image_data) <= config.image.max_file_size