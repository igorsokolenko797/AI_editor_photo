# handlers/__init__.py
from .start import send_welcome, handle_wrong_input_human, handle_wrong_input_clothes
from .photo_handlers import handle_human_photo, handle_clothes_photo

__all__ = [
    'send_welcome', 
    'handle_wrong_input_human', 
    'handle_wrong_input_clothes',
    'handle_human_photo', 
    'handle_clothes_photo'
]