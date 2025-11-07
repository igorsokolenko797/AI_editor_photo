import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.utils import executor

from config import config
from states.user_states import UserStates

# Импорт хендлеров
from handlers.start import (
    send_welcome, 
    handle_wrong_input_human, 
    handle_wrong_input_clothes
)
from handlers.photo_handlers import handle_human_photo, handle_clothes_photo
from handlers.errors import handle_telegram_error, handle_other_errors

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Инициализация бота
bot = Bot(token=config.bot.token)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

# Регистрация хендлеров
def register_handlers():
    # Команды
    dp.register_message_handler(send_welcome, commands=['start'], state='*')
    
    # Фото хендлеры
    dp.register_message_handler(
        handle_human_photo, 
        state=UserStates.waiting_for_human_photo, 
        content_types=types.ContentType.PHOTO
    )
    dp.register_message_handler(
        handle_clothes_photo, 
        state=UserStates.waiting_for_clothes_photo, 
        content_types=types.ContentType.PHOTO
    )
    
    # Неправильный ввод
    dp.register_message_handler(
        handle_wrong_input_human, 
        state=UserStates.waiting_for_human_photo
    )
    dp.register_message_handler(
        handle_wrong_input_clothes, 
        state=UserStates.waiting_for_clothes_photo
    )
    
    # Ошибки
    dp.register_errors_handler(handle_telegram_error, exception=TelegramAPIError)
    dp.register_errors_handler(handle_other_errors, exception=Exception)

async def on_startup(dp):
    """Действия при запуске бота"""
    logging.info("Бот запущен")
    register_handlers()

async def on_shutdown(dp):
    """Действия при остановке бота"""
    logging.info("Бот остановлен")
    await bot.close()

if __name__ == '__main__':
    executor.start_polling(
        dp,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        skip_updates=True
    )
