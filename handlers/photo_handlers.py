import logging
import asyncio
from aiogram import types
from aiogram.dispatcher import FSMContext

from states.user_states import UserStates
from services.image_processor import ImageProcessor
from utils.file_handlers import FileHandler
from config import config

# Инициализация сервисов
image_processor = ImageProcessor()
file_handler = FileHandler()

async def handle_human_photo(message: types.Message, state: FSMContext):
    """Обработчик фото человека"""
    try:
        await message.answer("✅ Фото получено! Теперь отправьте фото одежды 👕")
        
        # Сохраняем фото человека
        photo = message.photo[-1]
        file_id = photo.file_id
        
        image_data = await file_handler.download_telegram_file(message.bot, file_id)
        if not image_data:
            await message.answer("❌ Не удалось загрузить фото. Попробуйте еще раз.")
            return
        
        await state.update_data(human_photo=image_data)
        await UserStates.waiting_for_clothes_photo.set()
        
    except Exception as e:
        logging.error(f"Error handling human photo: {e}")
        await message.answer("❌ Произошла ошибка. Попробуйте еще раз.")
        await UserStates.waiting_for_human_photo.set()

async def handle_clothes_photo(message: types.Message, state: FSMContext):
    """Обработчик фото одежды"""
    try:
        await message.answer("⏳ Обрабатываю фото... Это займет несколько секунд.")
        
        # Получаем сохраненное фото человека
        user_data = await state.get_data()
        human_photo_data = user_data.get('human_photo')
        
        if not human_photo_data:
            await message.answer("❌ Не найдено фото человека. Начните заново.")
            await UserStates.waiting_for_human_photo.set()
            return
        
        # Сохраняем фото одежды с обработкой таймаута
        photo = message.photo[-1]
        file_id = photo.file_id
        
        print(f"📥 Начинаем загрузку файла {file_id}...")
        clothes_photo_data = await file_handler.download_telegram_file(message.bot, file_id)
        
        if not clothes_photo_data:
            await message.answer("❌ Не удалось загрузить фото одежды. Файл слишком большой или проблема с интернетом.")
            return
        
        print(f"✅ Файл загружен, размер: {len(clothes_photo_data)} байт")
        
        # Валидация изображений
        is_valid, error_message = image_processor.validate_images(
            human_photo_data, clothes_photo_data
        )
        
        if not is_valid:
            await message.answer(f"❌ {error_message}")
            await UserStates.waiting_for_human_photo.set()
            return
        
        print("🔄 Начинаем обработку изображений...")
        # Обработка изображений
        result_image_data = await image_processor.process_try_on(
            human_photo_data, clothes_photo_data
        )
        
        if result_image_data:
            await message.answer_photo(
                photo=result_image_data,
                caption="🎉 Вот результат примерки!\n\nХотите попробовать еще? Отправьте новое фото человека."
            )
            print("✅ Обработка завершена успешно")
        else:
            await message.answer("❌ Не удалось обработать фото. Попробуйте с другими изображениями.")
        
        await UserStates.waiting_for_human_photo.set()
        
    except asyncio.TimeoutError:
        await message.answer("❌ Превышено время обработки. Попробуйте еще раз.")
        await UserStates.waiting_for_human_photo.set()
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        await message.answer("❌ Произошла ошибка при обработке. Попробуйте еще раз.")
        await UserStates.waiting_for_human_photo.set()