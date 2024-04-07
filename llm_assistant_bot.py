import telebot
from telebot import types
import random
from random import choice
from config import BOT_TOKEN
from model_wrapper import ModelWrapper
from datetime import datetime
from logger import BotLogger

"""
get_text_messages - обработка любого текстового сообщения, в том числе того, что отправился при нажатии кнопки.

Методы, реализующие одноименные команды телеграм-боту:
start
help
generate
checkmodel
model
"""

logger = BotLogger('logs/bot.log')

bot = telebot.TeleBot(BOT_TOKEN)

model_wrapper = ModelWrapper() # внутри класса описание

@bot.message_handler(commands=['help'])
def help(message):
    user_id = message.from_user.id
    logger.log('info', f"Пользователь {user_id} запросил помощь /help")
    help_message = """Доступны следующие команды:
/start старт бота
/model выбор модели
/checkmodel посмотреть, как модель сейчас загружена
/generate сгенерировать текст по контексту (можно использовать без введения команды)
"""
    bot.send_message(message.from_user.id, help_message)


@bot.message_handler(commands=['start'])
def start(message):
    user_id = message.from_user.id
    logger.log('info', f"Пользователь {user_id} запустил бота с командой /start")
    bot.send_message(message.from_user.id, "Привет! Для знакомства с доступными командами введите /help")


@bot.message_handler(commands=['model'])
def model(message):
    user_id = message.from_user.id
    logger.log('info', f"Пользователь {user_id} запросил доступные модели с помощью /models")
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("StatLM")
    btn2 = types.KeyboardButton("GPT")
    btn3 = types.KeyboardButton("Llama")
    markup.add(btn1, btn2, btn3)
    bot.send_message(message.from_user.id, "Выберите модель для генерации", reply_markup=markup)


@bot.message_handler(commands=['checkmodel'])
def checkmodel(message):
    user_id = message.from_user.id
    logger.log('info', f"Пользователь {user_id} запросил проверку текущей модели с помощью команды /checkmodel")
    bot.send_message(message.from_user.id, f"Текущая модель: {str(model_wrapper.current_model_name)}")


@bot.message_handler(commands=['generate'])
def generate(message):
    user_id = message.from_user.id
    logger.log('info', f"Пользователь {user_id} запросил генерацию текста командой /generate")
    bot.send_message(message.from_user.id,
                     "Введите текст (вопрос, на который нужно ответить, либо текст, который нужно продолжить)")


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    print(f'<{message.text}>')
    if message.text in ['StatLM', 'GPT', 'Llama']:
        user_id = message.from_user.id
        logger.log('debug', f"Пользователь {user_id} выбрал модель {message.text}")
        print(f'@{message.text}@')
        
        status, result = model_wrapper.load(message.text, test_inference=True)

        if status:
            bot.send_message(message.from_user.id, "Подгружено")
            logger.log('debug', 'Модель подгружена')
        else:
            bot.send_message(message.from_user.id, f"Проблемы с загрузкой модели, ниже описаны ошибки.\n{result}")
            logger.log('debug', f'Проблемы с загрузкой модели, ниже описаны ошибки.\n{result}')
    else:
        status, result = model_wrapper.generate(message.text)
        logger.log('debug', f'{result} - результат генерации с помощью загруженной модели')
        if status:
            bot.send_message(message.from_user.id, result)
            logger.log('debug', f'Результат генерации с помощью выбранной модели: {result}')
        else:
            bot.send_message(message.from_user.id, f"Проблемы с генерацией, ниже описаны ошибки.\n{result}")
            logger.log('debug', f"Проблемы с генерацией, ниже описаны ошибки.\n{result}")
    # a_log = open(f'/logs/log_{message.chat.id}.txt', 'a')
    # a_log.write(f'{datetime.now()}: {message.text}\n')

bot.polling(none_stop=True, interval=0) #обязательная для работы бота часть
# TODO: сделайте логирование запросов с указанием модели и параметров - это полезно