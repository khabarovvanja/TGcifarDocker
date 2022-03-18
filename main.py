
import numpy as np
from telegram.ext import *
import tensorflow as tf
import cv2
from io import BytesIO

TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
model = tf.keras.models.load_model('modelCIFAR.h5')
classes = ['самолет',
           'автомобиль',
           'птица',
           'кот',
           'олень',
           'собака',
           'лягушка',
           'лошадь',
           'корабль',
           'грузовик']


def start(update, context):
    update.message.reply_text('Добро пожаловать!')


def help(update, context):
    update.message.reply_text('''
     /start - Старт
     /help  - Помощь''')

def handle_message(update, context):
    update.message.reply_text('Этот бот предназначен для распознавания картинок')


def handle_photo(update, context):
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

    prediction = model.predict(np.array([img / 255]))
    update.message.reply_text(f'На картинке распознано: {classes[np.argmax(prediction)]}')


if __name__ == '__main__':
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(CommandHandler('help', help))
    dp.add_handler(MessageHandler(Filters.text, handle_message))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))

    updater.start_polling()
    updater.idle()
