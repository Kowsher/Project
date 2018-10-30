# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 13:50:13 2018

@author: karigor
"""

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import os

bot = ChatBot('bot')
bot.set_trainer(ListTrainer)

for files in os.listdir('C:/Users/karigor/Downloads/Compressed\chatterbot-corpus-master/chatterbot_corpus/data/bangla/'):
    data = open('C:/Users/karigor/Downloads/Compressed\chatterbot-corpus-master/chatterbot_corpus/data/bangla/' + files, 'r', encoding="utf8").readlines()
    bot.train(data)

while True:
    message = input('You:')
    if message.strip != 'Bye':
        
        reply = bot.get_response(message)
        print('ChatBot: ', reply)
    if message.strip()=='Bye':
        print('ChatBot: Bye')
        break
    