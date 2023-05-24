import pickle

import pandas as pd
import telebot
from telebot.types import ReplyKeyboardMarkup
import config
from util.parser import *

bot = telebot.TeleBot(config.TOKEN)
data_sec_met = parser_sec_met()
data_met_num = parser_metro_num()
data_sec_num = parser_sec_num()
data_way_num = {'Transport': 0, 'Foot': 1}
data_room_num = {'studio apartment': 0, 'one-room apartment': 1, 'two-rooms apartment': 2, 'three-rooms apartment': 3,
                 'four-rooms apartment': 4}
data_minutes_num = {'up to 5 minutes': 5,
                    'up to 10 minutes': 10,
                    'up to 15 minutes': 15,
                    'up t0 20 minutes': 20}

data_area_num = {'25-30': 27, '30-35': 32, '35-40': 37,
                 '40-45': 42, '45-50': 47, '50-55': 52,
                 '55-60': 57, '60-65': 62, '65-70': 67,
                 '70-75': 72, '75-80': 77, '80-85': 82,
                 '85-90': 87, '90-95': 92, '95-100': 97,
                 '100-105': 102, '105-110': 107, '110-115': 112,
                 '115-120': 117}

selected_values = {'munites': None,
                   'way': None,
                   'rooms': None,
                   'total_area': None,
                   'numsect': None,
                   'nummetro': None}


file = open('model/model.pkl', 'rb')
tree = pickle.load(file)


def create_keyboard(keys) -> ReplyKeyboardMarkup:
    keyboard = ReplyKeyboardMarkup(row_width=1, one_time_keyboard=True)
    keyboard.add(*keys)
    return keyboard


def rent_predict(numsect, nummetro, total_area, rooms, way,
                 minutes, tree) -> int:  # получает входные параметры(все целые числа) и выдает цену
    parametres = pd.DataFrame({
        'minutes': minutes,
        'way': way,
        'rooms': rooms,
        'total_area': total_area,
        'numsect': numsect,
        'nummetro': nummetro,
    }, index=[0]
    )
    predict = tree.predict(parametres)
    return int(predict)


keyboard_sec = create_keyboard(data_sec_num.keys())
keyboard_way = create_keyboard(data_way_num.keys())
keyboard_minutes = create_keyboard(data_minutes_num.keys())
keyboard_rooms = create_keyboard(data_room_num.keys())
keyboard_area = create_keyboard(data_area_num.keys())


@bot.message_handler(commands=['start'])
def welcome(message):
    sti = open('static/welcome.webp', 'rb')
    bot.send_sticker(message.chat.id, sti)
    bot.send_message(message.chat.id, "Добро пожаловать, {0.first_name}."
                                      "\nМеня зовут {1.first_name} я бот по предсказанию стоимости квартиры в районах "
                                      "Москвы, чтобы узнать подробнее введи /help"
                     .format(message.from_user, bot.get_me()),
                     parse_mode='html')


@bot.message_handler(commands=['help'])
def help(message):
    bot.send_message(message.chat.id, "Смотри для того чтобы предсказать стоимость аренды жилья тебе необходимо всего "
                                      "лишь ввести комаду /predict, далее я тебя попрошу выбрать некоторые параметры,"
                                      " которые буду учитывать твои предпочтения и на основе этого мы предскажем"
                                      " стоимость, чтобы ты смог ориентироваться в ценах. \nКлассно не правда ли",
                     parse_mode='html')


@bot.message_handler(commands=['predict'])
def predict(message):
    bot.send_message(message.chat.id, text="Select a sector", reply_markup=keyboard_sec)


@bot.message_handler(content_types=['text'])
def lalala(message):
    selected_key = message.text
    if selected_key in data_sec_num.keys():
        keyboard_met = create_keyboard(data_sec_met[selected_key])
        selected_values['numsect'] = data_sec_num[selected_key]
        bot.send_message(message.chat.id, text="Select a metro", reply_markup=keyboard_met)
    elif selected_key in data_met_num.keys():
        selected_values['nummetro'] = data_met_num[selected_key]
        bot.send_message(message.chat.id, text="Choose how you want to get to the metro by transport or by foot",
                         reply_markup=keyboard_way)
    elif selected_key in data_way_num.keys():
        selected_values['way'] = data_way_num[selected_key]
        bot.send_message(message.chat.id, text="Select a minutes to metro",
                         reply_markup=keyboard_minutes)
    elif selected_key in data_minutes_num.keys():
        selected_values['minutes'] = data_minutes_num[selected_key]
        bot.send_message(message.chat.id, text="Select number of rooms",
                         reply_markup=keyboard_rooms)
    elif selected_key in data_room_num.keys():
        selected_values['rooms'] = data_room_num[selected_key]
        bot.send_message(message.chat.id, text="Select total area",
                         reply_markup=keyboard_area)
    elif selected_key in data_area_num.keys():
        selected_values['total_area'] = data_area_num[selected_key]
        price = rent_predict(selected_values['numsect'], selected_values['nummetro'],
                             selected_values['total_area'], selected_values['rooms'],
                             selected_values['way'], selected_values['minutes'],
                             tree)
        bot.send_message(message.chat.id, text=f'predicted price: {price}')


bot.polling(none_stop=True)
