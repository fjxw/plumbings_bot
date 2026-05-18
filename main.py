# --- Импорт библиотек ---
import os
import json
import re
import random
import wave
import subprocess
import uuid
import telebot
from PIL import Image, ImageDraw
import nltk
from vosk import Model, KaldiRecognizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsNERTagger, Doc
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import database

# --- Параметры проекта ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PRODUCTS_PATH = os.path.join(PROJECT_ROOT, 'products.json')
DIALOGUES_PATH = os.path.join(PROJECT_ROOT, 'dialogues.txt')
CATEGORY_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'category_images')
VOSK_MODEL_DIR = os.path.join(PROJECT_ROOT, 'vosk-model-small-ru-0.22')

# Токен теперь берется из переменных окружения (безопасность для GitHub)
TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_TELEGRAM_BOT_TOKEN')
ADMIN_IDS = list(map(int, os.environ.get('TELEGRAM_ADMIN_IDS', '').split(','))) if os.environ.get('TELEGRAM_ADMIN_IDS') else []

# Инициализация БД
database.init_db()
try:
    with open(PRODUCTS_PATH, 'r', encoding='utf-8') as f:
        database.migrate_from_json(json.load(f))
except:
    pass

# --- NLP и ML инициализация ---
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
ner_tagger = NewsNERTagger(emb)

CATEGORY_NAME_MAP = {
    'smesiteli': 'смесители', 'vanny': 'ванны', 'unitazy': 'унитазы',
    'rakoviny': 'раковины', 'dushevye': 'душевые',
    'смесители': 'смесители', 'ванны': 'ванны', 'унитазы': 'унитазы',
    'раковины': 'раковины', 'душевые': 'душевые'
}

def clear_phrase(phrase):
    if not phrase: return ''
    return re.sub(r'[^а-яё0-9\- ]', '', phrase.lower()).strip()

def lemmatize(text):
    text = clear_phrase(text)
    if not text: return ''
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens: token.lemmatize(morph_vocab)
    return ' '.join([_.lemma for _ in doc.tokens])

SENTIMENT_DICT = {
    'плохо': -1, 'ужасно': -1, 'дорого': -0.3, 'сломано': -1, 'проблема': -0.5,
    'хорошо': 0.5, 'отлично': 1, 'супер': 1, 'нравится': 0.8, 'спасибо': 0.5
}

def get_sentiment(text):
    lemmas = lemmatize(text).split()
    return max(-1.0, min(1.0, sum([SENTIMENT_DICT.get(l, 0) for l in lemmas])))

def extract_budget(text):
    """ДОП. ФУНКЦИЯ: Извлекает бюджет из текста (например: 'до 10000 рублей')"""
    match = re.search(r'(до|дешевле|менее)\s*(\d+)', text.lower())
    if match:
        return int(match.group(2))
    return None

def generate_receipt_image(items, order_id):
    """ДОП. ФУНКЦИЯ: Рисует красивый чек заказа"""
    img = Image.new('RGB', (400, 400), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    
    # Пытаемся загрузить шрифт с поддержкой кириллицы
    try:
        from PIL import ImageFont
        # Пути к шрифтам на Windows и Linux
        font_paths = ["arial.ttf", "C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        font = None
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, 18)
                font_bold = ImageFont.truetype(path, 22)
                break
            except:
                continue
        if not font:
            font = ImageFont.load_default()
            font_bold = font
    except:
        font = None
        font_bold = None

    y_text = 20
    d.text((120, y_text), "ЧЕК ОПЛАТЫ", fill=(0, 0, 0), font=font_bold)
    y_text += 40
    d.text((20, y_text), f"Заказ №: {order_id}", fill=(50, 50, 50), font=font)
    y_text += 30
    total = 0
    for item in items:
        text = f"- {item['name']} : {item['price']} руб."
        d.text((20, y_text), text, fill=(0, 0, 0), font=font)
        total += item['price']
        y_text += 25
    y_text += 20
    d.line([(20, y_text), (380, y_text)], fill=(0, 0, 0), width=2)
    y_text += 10
    d.text((20, y_text), f"ИТОГО К ОПЛАТЕ: {total} руб.", fill=(200, 0, 0), font=font_bold)
    
    filename = f"receipt_{order_id}.jpg"
    img.save(filename)
    return filename

# --- Конфиг интентов и обучение ML ---
BOT_CONFIG = {
    'intents': {
        'hello': {'examples': ['привет', 'добрый день', 'здравствуйте', 'начать'], 'responses': ['Здравствуйте! Чем могу помочь? Ищете сантехнику?']},
        'bye': {'examples': ['пока', 'до свидания', 'закончить'], 'responses': ['До свидания! Обращайтесь.']},
        'order': {'examples': ['заказать', 'купить', 'оформить'], 'responses': ['Оформляем заказ. Оставьте ваш номер телефона.']},
        'confirm': {'examples': ['подтверждаю', 'верно', 'да'], 'responses': ['Подтверждено!']}
    },
    'failure_phrases': ['Не совсем понял вас. Попробуйте переформулировать или воспользуйтесь кнопками.']
}

# --- Дополнительная функция: Поиск ---
def find_by_name(text):
    text = text.lower().replace("найди", "").replace("поиск", "").strip()
    if len(text) < 3: return None
    return database.search_products(text)

X_text, y = [], []
for intent, data in BOT_CONFIG['intents'].items():
    for example in data['examples']:
        X_text.append(lemmatize(example))
        y.append(intent)

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
X = vectorizer.fit_transform(X_text)
clf = LinearSVC(C=1.0)
clf.fit(X, y)

# --- Заглушка на случай пустого файла ---
def generate_answer(replica): return None

# --- Логика Бота с поддержкой пользователей ---
class PlumbingBot:
    def __init__(self):
        self.stats = {'requests': 0}
        self.users_state = {}

    def get_user_state(self, user_id):
        if user_id not in self.users_state:
            self.users_state[user_id] = {'context': None, 'cart': [], 'last_cat': None}
        return self.users_state[user_id]

    def get_response(self, text, user_id):
        self.stats['requests'] += 1
        state = self.get_user_state(user_id)
        
        if get_sentiment(text) <= -0.7:
            return 'Вижу, вы расстроены. Хотите я позову живого оператора?'

        text_lem = lemmatize(text)
        vector = vectorizer.transform([text_lem])
        intent = clf.predict(vector)[0]
        budget = extract_budget(text)

        # Поиск по названию
        search_res = find_by_name(text)
        if search_res:
            res = "Вот что я нашел по вашему запросу:\n"
            for item in search_res[:5]:
                res += f"🔹 {item['name']} ({item['category']}) — {item['price']} руб.\n"
            return res

        # Извлечение сущностей из БД
        all_cats = database.get_categories()
        entities = [cat for cat in all_cats if nltk.edit_distance(text_lem, cat[:-1]) <= 2 or cat in text_lem]

        # Контекст заказа
        if state['context'] == 'confirm':
            if intent in ['confirm']:
                state['context'] = None
                return '!CONFIRM!'
            elif 'нет' in text.lower():
                state['context'] = None
                state['cart'] = []
                return 'Заказ отменен.'

        if state['context'] == 'order' and any(c.isdigit() for c in text):
            state['context'] = 'confirm'
            return f'Проверьте данные. Оформляем заявку. Все верно? (Да/Нет)'

        if intent == 'order':
            if entities:
                items = database.get_products_by_category(entities[0])
                if items:
                    state['cart'].append(items[0])
            elif state.get('last_cat'):
                items = database.get_products_by_category(state['last_cat'])
                if items:
                    state['cart'].append(items[0])
            
            if not state['cart']:
                return 'Сначала выберите категорию товаров, например "ванны".'
                
            state['context'] = 'order'
            return 'Отлично! Для оформления заказа введите ваш номер телефона (10 цифр).'

        # Поиск в БД
        if entities:
            cat = entities[0]
            state['last_cat'] = cat
            items = database.get_products_by_category(cat, max_price=budget)
            
            if not items:
                if budget:
                    return f'К сожалению, {cat} до {budget} руб. сейчас нет в наличии.'
                return f'К сожалению, товаров в категории {cat} сейчас нет.'
            
            res = f'Категория "{cat.capitalize()}"{" до " + str(budget) + " руб" if budget else ""}:\n'
            for item in items[:5]:
                res += f'🔹 {item["name"]} — {item["price"]} руб. (в наличии: {item["stock"]} шт.)\n'
            res += '\nХотите что-то из этого заказать? Напишите "заказать".'
            return res

        if intent in BOT_CONFIG['intents']:
            return random.choice(BOT_CONFIG['intents'][intent]['responses'])

        return random.choice(BOT_CONFIG['failure_phrases'])

# Загружаем модель MobileNetV2 один раз
model_ai = MobileNetV2(weights='imagenet')

def classify_photo(img_path):
    try:
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model_ai.predict(img_array)
        results = decode_predictions(preds, top=5)[0]

        # Расширенный маппинг (ImageNet -> Твои категории)
        mapping = {
            'faucet': 'смесители', 'tap': 'смесители', 'cock': 'смесители',
            'bathtub': 'ванны', 'tub': 'ванны', 'bathing_tub': 'ванны',
            'toilet_seat': 'унитазы', 'toilet_tissue': 'унитазы',
            'washbasin': 'смесители', 'sink': 'раковины', 'basin': 'раковины',
            'shower_curtain': 'душевые', 'shower_cap': 'душевые', 'enclosure': 'душевые'
        }

        for (id, label, prob) in results:
            label = label.lower()
            if label in mapping:
                found_cat = mapping[label]
                print(f"DEBUG: ИИ увидел {label}, маппинг в категорию: {found_cat}")
                return found_cat
        
        return None
    except Exception as e:
        print(f"ОШИБКА НЕЙРОСЕТИ: {e}")
        return None

def recognize_voice(wav_path):
    wf = wave.open(wav_path, 'rb')
    model = Model(VOSK_MODEL_DIR)
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(False)
    text_parts = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0: break
        if rec.AcceptWaveform(data): text_parts.append(json.loads(rec.Result()).get('text', ''))
    text_parts.append(json.loads(rec.FinalResult()).get('text', ''))
    return ' '.join(text_parts).strip()


bot = telebot.TeleBot(TOKEN)
bot_logic = PlumbingBot()

@bot.message_handler(commands=['start'])
def start_message(message):
    markup = telebot.types.InlineKeyboardMarkup()
    cats = database.get_categories()
    for cat in cats:
        markup.add(telebot.types.InlineKeyboardButton(cat.capitalize(), callback_data=f"cat_{cat}"))
    
    bot.send_message(
        message.chat.id, 
        'Привет! Я помощник по сантехнике. Выберите категорию или напишите, что ищете (например, "найди Grohe").',
        reply_markup=markup
    )

@bot.callback_query_handler(func=lambda call: call.data.startswith('cat_'))
def category_callback(call):
    cat = call.data.split('_')[1]
    items = database.get_products_by_category(cat)
    if not items:
        bot.answer_callback_query(call.id, "Товаров в этой категории пока нет.")
        return
    
    res = f'Категория "{cat.capitalize()}":\n'
    for item in items[:5]:
        res += f'🔹 {item["name"]} — {item["price"]} руб. (в наличии: {item["stock"]} шт.)\n'
    res += '\nХотите что-то из этого заказать? Напишите "заказать".'
    
    bot.send_message(call.message.chat.id, res)
    bot.answer_callback_query(call.id)

@bot.message_handler(commands=['top'])
def top_products(message):
    top = database.get_top_products()
    res = "🔥 Популярные товары (по наличию):\n"
    for item in top:
        res += f"🔹 {item['name']} — {item['price']} руб.\n"
    bot.send_message(message.chat.id, res)

@bot.message_handler(commands=['feedback'])
def feedback_start(message):
    bot.send_message(message.chat.id, "Напишите ваш отзыв о нашей работе:")
    bot.register_next_step_handler(message, feedback_finish)

def feedback_finish(message):
    # В реальности можно сохранять в БД или слать админу
    for admin_id in ADMIN_IDS:
        bot.send_message(admin_id, f"📣 Новый отзыв от @{message.from_user.username}: {message.text}")
    bot.send_message(message.chat.id, "Спасибо за ваш отзыв!")

@bot.message_handler(content_types=['text'])
def text_message(message):
    response = bot_logic.get_response(message.text, message.chat.id)
    if response == '!CONFIRM!':
        state = bot_logic.get_user_state(message.chat.id)
        cart = state.get('cart', [])
        if not cart:
            bot.send_message(message.chat.id, "Ваша корзина пуста.")
            return
            
        order_id = str(uuid.uuid4())[:8].upper()
        total_price = sum(item['price'] for item in cart)
        
        # Сохраняем в БД и уменьшаем остаток
        database.save_order(order_id, message.from_user.id, [i['name'] for i in cart], total_price)
        for item in cart:
            database.update_stock(item['name'], 1)
            
        receipt_file = generate_receipt_image(cart, order_id)
        with open(receipt_file, 'rb') as photo:
            bot.send_photo(message.chat.id, photo, caption=f'Заказ {order_id} подтвержден! Сумма: {total_price} руб.\nОжидайте звонка менеджера.')
        os.remove(receipt_file)
        state['cart'] = []
    else:
        bot.reply_to(message, response)

@bot.message_handler(content_types=['voice'])
def voice_message(message):
    uid = str(uuid.uuid4())
    ogg_path, wav_path = f'voice_{uid}.ogg', f'voice_{uid}.wav'
    try:
        downloaded_file = bot.download_file(bot.get_file(message.voice.file_id).file_path)
        with open(ogg_path, 'wb') as f: f.write(downloaded_file)
        subprocess.run(['ffmpeg', '-y', '-i', ogg_path, '-ar', '16000', '-ac', '1', wav_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        text = recognize_voice(wav_path)
        if not text:
            bot.reply_to(message, 'Не удалось распознать речь')
            return
            
        response = bot_logic.get_response(text, message.chat.id)
        bot.reply_to(message, f'{response}')
    except Exception as e:
        bot.reply_to(message, f'Ошибка обработки аудио: {e}')
    finally:
        for p in [ogg_path, wav_path]:
            if os.path.exists(p): os.remove(p)

@bot.message_handler(content_types=['photo'])
def photo_message(message):
    img_path = f'image_{uuid.uuid4()}.jpg'
    try:
        # 1. Качаем фото
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        with open(img_path, 'wb') as f:
            f.write(downloaded_file)

        # 2. Запускаем ИИ из Keras
        category = classify_photo(img_path)

        # Проверка по БД
        all_cats = database.get_categories()
        print(f"DEBUG: В базе доступны категории: {all_cats}")

        if category:
            # Если категория есть в базе товаров
            if category in all_cats:
                items = database.get_products_by_category(category)
                offer = f'Я узнал этот товар, это {category[:-1]}! Вот лучшие варианты для вас:\n\n'
                for item in items[:3]:
                    offer += f'🔹 {item["name"]} — {item["price"]} руб. (осталось {item["stock"]} шт.)\n'
                bot.reply_to(message, offer)
            else:
                # ИИ узнал товар, но в БД нет такой категории
                bot.reply_to(message, f'Я вижу на фото {category}, но сейчас их нет в нашем прайс-листе. Посмотрите наши ванны или смесители!')
        else:
            bot.reply_to(message, 'Не уверен, что это за сантехника. Попробуйте сфотографировать под другим углом или напишите название текстом.')

    except Exception as e:
        print(f"Ошибка в photo_message: {e}")
        bot.reply_to(message, 'Произошла техническая ошибка при распознавании.')
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

@bot.message_handler(commands=['admin'])
def admin_menu(message):
    if message.from_user.id not in ADMIN_IDS:
        bot.reply_to(message, "У вас нет прав администратора.")
        return
    
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add("Просмотр заказов", "Добавить товар")
    markup.add("Удалить товар", "Изменить остаток")
    markup.add("Выход из админки")
    bot.send_message(message.chat.id, "Меню администратора:", reply_markup=markup)

@bot.message_handler(func=lambda m: m.from_user.id in ADMIN_IDS and m.text == "Просмотр заказов")
def view_orders(message):
    orders = database.get_all_orders()
    if not orders:
        bot.send_message(message.chat.id, "Заказов пока нет.")
        return
    
    res = "Последние заказы:\n"
    for order in orders[:10]:
        res += f"📦 ID: {order[0]} | User: {order[1]} | Сумма: {order[3]} руб. | Дата: {order[4]}\n"
    bot.send_message(message.chat.id, res)

@bot.message_handler(func=lambda m: m.from_user.id in ADMIN_IDS and m.text == "Добавить товар")
def add_product_start(message):
    bot.send_message(message.chat.id, "Введите данные в формате: категория;название;цена;количество")
    bot.register_next_step_handler(message, add_product_finish)

def add_product_finish(message):
    try:
        cat, name, price, stock = message.text.split(';')
        database.add_product(cat.strip().lower(), name.strip(), int(price), int(stock))
        bot.send_message(message.chat.id, "Товар успешно добавлен!")
    except:
        bot.send_message(message.chat.id, "Ошибка формата. Попробуйте еще раз.")

@bot.message_handler(func=lambda m: m.from_user.id in ADMIN_IDS and m.text == "Удалить товар")
def delete_product_start(message):
    bot.send_message(message.chat.id, "Введите точное название товара для удаления:")
    bot.register_next_step_handler(message, delete_product_finish)

def delete_product_finish(message):
    database.delete_product(message.text.strip())
    bot.send_message(message.chat.id, "Товар удален (если он существовал).")

@bot.message_handler(func=lambda m: m.from_user.id in ADMIN_IDS and m.text == "Изменить остаток")
def change_stock_start(message):
    bot.send_message(message.chat.id, "Введите: название товара;новое количество")
    bot.register_next_step_handler(message, change_stock_finish)

def change_stock_finish(message):
    try:
        name, stock = message.text.split(';')
        database.set_stock(name.strip(), int(stock))
        bot.send_message(message.chat.id, "Остаток обновлен!")
    except:
        bot.send_message(message.chat.id, "Ошибка формата.")

@bot.message_handler(func=lambda m: m.from_user.id in ADMIN_IDS and m.text == "Выход из админки")
def exit_admin(message):
    bot.send_message(message.chat.id, "Вы вышли из админ-меню.", reply_markup=telebot.types.ReplyKeyboardRemove())

@bot.message_handler(content_types=['text'])
def text_message(message):
    response = bot_logic.get_response(message.text, message.chat.id)
    if response == '!CONFIRM!':
        state = bot_logic.get_user_state(message.chat.id)
        cart = state.get('cart', [])
        if not cart:
            bot.send_message(message.chat.id, "Ваша корзина пуста.")
            return
            
        order_id = str(uuid.uuid4())[:8].upper()
        total_price = sum(item['price'] for item in cart)
        
        # Сохраняем в БД и уменьшаем остаток
        database.save_order(order_id, message.from_user.id, [i['name'] for i in cart], total_price)
        for item in cart:
            database.update_stock(item['name'], 1)
            
        receipt_file = generate_receipt_image(cart, order_id)
        with open(receipt_file, 'rb') as photo:
            bot.send_photo(message.chat.id, photo, caption=f'Заказ {order_id} подтвержден! Сумма: {total_price} руб.\nОжидайте звонка менеджера.')
        os.remove(receipt_file)
        state['cart'] = []
    else:
        bot.reply_to(message, response)