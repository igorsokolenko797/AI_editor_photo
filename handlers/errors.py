import logging
from aiogram import types
from aiogram.utils.exceptions import TelegramAPIError

async def handle_telegram_error(update: types.Update, exception: TelegramAPIError):
    """Обработчик ошибок Telegram API"""
    logging.error(f"Telegram API error: {exception}")
    return True

async def handle_other_errors(update: types.Update, exception: Exception):
    """Обработчик прочих ошибок"""
    logging.error(f"Unexpected error: {exception}")
    
    if update and update.message:
        try:
            await update.message.answer(
                "❌ Произошла непредвиденная ошибка. "
                "Попробуйте еще раз или обратитесь к администратору."
            )
        except:
            pass
    
    return True
