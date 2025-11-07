from aiogram.dispatcher.filters.state import State, StatesGroup

class UserStates(StatesGroup):
    waiting_for_human_photo = State()
    waiting_for_clothes_photo = State()
    processing = State()
