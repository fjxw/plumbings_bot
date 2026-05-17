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

# --- Параметры проекта ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PRODUCTS_PATH = os.path.join(PROJECT_ROOT, 'products.json')
DIALOGUES_PATH = os.path.join(PROJECT_ROOT, 'dialogues.txt')
CATEGORY_IMAGES_DIR = os.path.join(PROJECT_ROOT, 'category_images')
VOSK_MODEL_DIR = os.path.join(PROJECT_ROOT, 'vosk-model-small-ru-0.22')

# Токен теперь берется из переменных окружения (безопасность для GitHub)
TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', 'YOUR_TELEGRAM_BOT_TOKEN')

# Создаем базовые файлы, если их нет (чтобы не падало в GitHub Actions)
if not os.path.exists(PRODUCTS_PATH):
    with open(PRODUCTS_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            "ванны": [{"name": "Ванна акриловая", "price": 12000}, {"name": "Ванна чугунная", "price": 25000}],
            "смесители": [{"name": "Смеситель Grohe", "price": 4500}, {"name": "Смеситель эконом", "price": 1500}]
        }, f, ensure_ascii=False)

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
    img = Image.new('RGB', (400, 300), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    y_text = 20
    d.text((120, y_text), "ЧЕК ОПЛАТЫ", fill=(0, 0, 0))
    y_text += 40
    d.text((20, y_text), f"Заказ №: {order_id}", fill=(50, 50, 50))
    y_text += 30
    total = 0
    for item in items:
        d.text((20, y_text), f"- {item['name']} : {item['price']} руб.", fill=(0, 0, 0))
        total += item['price']
        y_text += 25
    y_text += 20
    d.line([(20, y_text), (380, y_text)], fill=(0, 0, 0), width=2)
    y_text += 10
    d.text((20, y_text), f"ИТОГО К ОПЛАТЕ: {total} руб.", fill=(200, 0, 0))
    
    filename = f"receipt_{order_id}.jpg"
    img.save(filename)
    return filename

# --- Загрузка данных ---
try:
    with open(PRODUCTS_PATH, 'r', encoding='utf-8') as f:
        PRODUCTS = json.load(f)
except FileNotFoundError:
    PRODUCTS = {}

# --- Конфиг интентов и обучение ML ---
BOT_CONFIG = {
    'intents': {
        'hello': {'examples': ['привет', 'добрый день', 'здравствуйте', 'начать'], 'responses': ['Здравствуйте! Чем могу помочь? Ищете сантехнику?']},
        'bye': {'examples': ['пока', 'до свидания', 'закончить'], 'responses': ['До свидания! Обращайтесь.']},
        'order': {'examples': ['заказать', 'купить', 'оформить'], 'responses': ['Оформляем заказ. Оставьте ваш номер телефона.']},
        'confirm': {'examples': ['подтверждаю', 'верно', 'да'], 'responses': ['Подтверждено!']}
    },
    'failure_phrases': ['Не совсем понял вас. Попробуйте переформулировать.']
}

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
        self.users_state = {} # ДОП. ФУНКЦИЯ: Индивидуальный контекст для каждого чата

    def get_user_state(self, user_id):
        if user_id not in self.users_state:
            self.users_state[user_id] = {'context': None, 'cart': []}
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

        # Извлечение сущностей (категорий товара)
        entities = [cat for cat in PRODUCTS.keys() if nltk.edit_distance(text_lem, cat[:-1]) <= 2 or cat in text_lem]

        # Контекст заказа
        if state['context'] == 'confirm':
            if intent in ['confirm']:
                state['context'] = None
                return '!CONFIRM!' # Специальный маркер для генерации чека
            elif intent == 'no':
                state['context'] = None
                state['cart'] = []
                return 'Заказ отменен.'

        if state['context'] == 'order' and any(c.isdigit() for c in text):
            state['context'] = 'confirm'
            return f'Проверьте данные. Оформляем заявку. Все верно? (Да/Нет)'

        if intent == 'order':
            if entities:
                state['cart'].append(PRODUCTS[entities[0]][0]) # Добавляем первый товар из категории
            state['context'] = 'order'
            return 'Отлично! Для оформления заказа введите ваш номер телефона (10 цифр).'

        # Поиск с учетом бюджета
        if entities:
            cat = entities[0]
            items = PRODUCTS[cat]
            if budget:
                items = [i for i in items if i['price'] <= budget]
                if not items:
                    return f'К сожалению, {cat} до {budget} руб. сейчас нет в наличии.'
            
            res = f'Категория "{cat.capitalize()}"{" до " + str(budget) + " руб" if budget else ""}:\n'
            for item in items[:3]:
                res += f'🔹 {item["name"]} — {item["price"]} руб.\n'
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
            'washbasin': 'сместители', 'sink': 'раковины', 'basin': 'раковины',
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
    bot.send_message(message.chat.id, 'Привет! Я помощник по сантехнике. Напишите, что ищете (например, "ванна до 15000"), или отправьте фото/голосовое сообщение.')

@bot.message_handler(content_types=['text'])
def text_message(message):
    response = bot_logic.get_response(message.text, message.chat.id)
    if response == '!CONFIRM!':
        cart = bot_logic.get_user_state(message.chat.id).get('cart', [])
        if not cart: cart = [{"name": "Товар по акции", "price": 1000}]
        order_id = str(uuid.uuid4())[:8].upper()
        receipt_file = generate_receipt_image(cart, order_id)
        with open(receipt_file, 'rb') as photo:
            bot.send_photo(message.chat.id, photo, caption=f'Заказ успешно подтвержден! Ожидайте звонка менеджера.')
        os.remove(receipt_file)
        bot_logic.users_state[message.chat.id]['cart'] = [] # очищаем корзину
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

        # Проверка: что у нас вообще есть в базе?
        available_categories = list(PRODUCTS.keys())
        print(f"DEBUG: В базе доступны категории: {available_categories}")

        if category:
            # Если категория есть в базе товаров
            if category in PRODUCTS:
                items = PRODUCTS[category]
                offer = f'Я узнал этот товар, это {category[:-1]}! Вот лучшие варианты для вас:\n\n'
                for item in items[:3]:
                    offer += f'🔹 {item["name"]} — {item["price"]} руб.\n'
                bot.reply_to(message, offer)
            else:
                # ИИ узнал товар, но в json-файле нет такой категории
                bot.reply_to(message, f'Я вижу на фото {category[:-1]}, но сейчас их нет в нашем прайс-листе. Посмотрите наши ванны или смесители!')
        else:
            bot.reply_to(message, 'Не уверен, что это за сантехника. Попробуйте сфотографировать под другим углом или напишите название текстом.')

    except Exception as e:
        print(f"Ошибка в photo_message: {e}")
        bot.reply_to(message, 'Произошла техническая ошибка при распознавании.')
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

if __name__ == '__main__':
    print('Бот запущен...')
    bot.polling(none_stop=True)