import os
import asyncio
import httpx
import logging
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

user_filters = {}  # {user_id: set(selected_sounds)}


class SoundStates(StatesGroup):
    waiting_for_name = State()
    waiting_for_sound = State()


load_dotenv()
logging.basicConfig(level=logging.INFO)
BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode="Markdown"))
dp = Dispatcher()

API_URL = os.getenv("API_BASE_URL", "http://localhost:8085")
HEADERS = {"ngrok-skip-browser-warning": "true"}

BASE_SOUNDS = ["Тишина", "Собака", "Музыка", "Кошка", "Стекло", "Птица", "Детский плач", "Сирена", "Жарка еды", "Речь",
               "Щелчок", "Хлопок", "Транспорт"]
AVAILABLE_SOUNDS = BASE_SOUNDS.copy()
streaming_active = {}


def get_main_menu():
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="Настроить мониторинг")],
            [KeyboardButton(text="Запустить стрим"), KeyboardButton(text="➕ Добавить звук")],
            [KeyboardButton(text="Список звуков")]
        ],
        resize_keyboard=True
    )


def get_stop_kb():
    return ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="Остановить стрим")]], resize_keyboard=True)


def get_cancel_kb():
    return ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text="Отмена")]], resize_keyboard=True)


def get_sounds_markup(user_id: int):
    builder = InlineKeyboardBuilder()
    selected = user_filters.get(user_id, set())
    for sound in AVAILABLE_SOUNDS:
        status = "✅ " if sound in selected else ""
        builder.button(text=f"{status}{sound}", callback_data=f"toggle_{sound}")
    builder.button(text="📥 Применить фильтры", callback_data="apply_filter")
    builder.adjust(2)
    return builder.as_markup()


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("🎙 **Система готова.**\nИспользуйте меню ниже.", reply_markup=get_main_menu())


@dp.message(F.text == "Настроить мониторинг")
async def monitor_settings(message: types.Message):
    user_id = message.from_user.id
    await message.answer("Выберите звуки для отслеживания (алерты придут аудиофайлом):",
                         reply_markup=get_sounds_markup(user_id))


@dp.message(F.text == "➕ Добавить звук")
async def add_sound_start(message: types.Message, state: FSMContext):
    await state.set_state(SoundStates.waiting_for_name)
    await message.answer("📝 Введите **название** для нового звука (например: `стук_в_дверь`):",
                         reply_markup=get_cancel_kb())


@dp.message(SoundStates.waiting_for_name, F.text != "Отмена")
async def process_name(message: types.Message, state: FSMContext):
    await state.update_data(custom_name=message.text)
    await state.set_state(SoundStates.waiting_for_sound)
    await message.answer(f"🎙 Отлично! Отправьте **голосовое сообщение** для звука `{message.text}`:",
                         reply_markup=get_cancel_kb())


@dp.message(SoundStates.waiting_for_sound, F.voice | F.audio)
async def process_sound(message: types.Message, state: FSMContext):
    data = await state.get_data()
    sound_name = data.get("custom_name", "unknown")

    file_id = message.voice.file_id if message.voice else message.audio.file_id
    file = await bot.get_file(file_id)
    file_data = await bot.download_file(file.file_path)

    status_msg = await message.answer("⏳ Анализирую и сохраняю на плате...", reply_markup=ReplyKeyboardRemove())

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            files = {'file': (f'{sound_name}.ogg', file_data, 'audio/ogg')}
            res = await client.post(f"{API_URL}/add_sound?name={sound_name}", files=files, headers=HEADERS)
            await status_msg.delete()
            if res.status_code == 200:
                clean_name = "".join(x for x in sound_name if x.isalnum() or x in "._- ")
                if clean_name not in AVAILABLE_SOUNDS: AVAILABLE_SOUNDS.append(clean_name)
                await message.answer(f"✅ Звук **{clean_name}** добавлен! Выберите его в настройках.",
                                     reply_markup=get_main_menu())
            else:
                await message.answer(f"❌ Ошибка сервера: {res.status_code}", reply_markup=get_main_menu())
        except Exception as e:
            await status_msg.delete()
            await message.answer(f"💥 Ошибка связи: {e}", reply_markup=get_main_menu())
    await state.clear()


@dp.message(F.text == "Запустить стрим")
async def stream_logs_handler(message: types.Message):
    user_id = message.from_user.id
    streaming_active[user_id] = True
    await message.answer("▶️ Поток логов запущен.", reply_markup=get_stop_kb())

    from collections import deque
    seen_events = set()
    seen_queue = deque(maxlen=2000)

    async with httpx.AsyncClient(headers=HEADERS, timeout=10.0) as client:
        while streaming_active.get(user_id):
            user_filter = user_filters.get(user_id, set())
            try:
                response = await client.get(f"{API_URL}/logs")
                if response.status_code != 200:
                    await asyncio.sleep(1)
                    continue
                data = response.json()
                events = data.get("events", [])

                for event in events:
                    msg, ts = event.get("message", ""), event.get("timestamp", "")
                    event_key = f"{ts}_{msg}"
                    if event_key in seen_events: continue
                    seen_events.add(event_key)
                    seen_queue.append(event_key)
                    if len(seen_queue) == seen_queue.maxlen: seen_events.discard(seen_queue.popleft())

                    if user_filter and not any(s.lower() in msg.lower() for s in user_filter): continue
                    await message.answer(f"🔔 `[{ts}]` **{msg}**")
                await asyncio.sleep(1.2)
            except Exception:
                await asyncio.sleep(2)
                continue
    await message.answer("⏹️ Стрим остановлен.", reply_markup=get_main_menu())


@dp.message(F.text == "Отмена")
async def cancel_action(message: types.Message, state: FSMContext):
    await state.clear()
    await message.answer("Действие отменено.", reply_markup=get_main_menu())


@dp.callback_query(F.data.startswith("toggle_"))
async def toggle_sound_callback(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    sound = callback.data.replace("toggle_", "")
    current = user_filters.setdefault(user_id, set())
    if sound in current:
        current.remove(sound)
    else:
        current.add(sound)
    await callback.message.edit_reply_markup(reply_markup=get_sounds_markup(user_id))
    await callback.answer()


@dp.callback_query(F.data == "apply_filter")
async def apply_filter_callback(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    selected = list(user_filters.get(user_id, set()))

    payload = {"chat_id": user_id, "filters": selected}

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            res = await client.post(f"{API_URL}/set_active_sounds", json=payload, headers=HEADERS)
            if res.status_code == 200:
                text = ", ".join(selected) if selected else "Отключены"
                await callback.message.edit_text(f"🎯 **Мониторинг запущен для:**\n{text}")
            else:
                await callback.message.answer(f"⚠️ Ошибка сервера: {res.text}")
        except Exception as e:
            await callback.message.answer(f"💥 Ошибка связи с платой: {e}")
    await callback.answer()


@dp.message(F.text == "Остановить стрим")
async def stop_stream(message: types.Message):
    streaming_active[message.from_user.id] = False


@dp.message(F.text == "Список звуков")
async def cmd_list(message: types.Message):
    await message.answer("🔎 **Доступные звуки:**\n" + "\n".join([f"• {s}" for s in AVAILABLE_SOUNDS]))


async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())