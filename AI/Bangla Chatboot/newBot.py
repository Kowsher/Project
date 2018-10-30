# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 13:50:13 2018

@author: karigor
"""

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import speech_recognition as sr
import os

bot = ChatBot('bot')
bot.set_trainer(ListTrainer)

for files in os.listdir('C:/Users/karigor/Downloads/Compressed\chatterbot-corpus-master/chatterbot_corpus/data/english/'):
    data = open('C:/Users/karigor/Downloads/Compressed\chatterbot-corpus-master/chatterbot_corpus/data/english/' + files, 'r', encoding="utf8").readlines()
    bot.train(data)

while True:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Speak  :")
        audio = r.listen(source)
        try:
            message = r.recognize_google(audio, language="bn-BD")
        
            print("You  : {}".format(message))
            if  message.strip != 'Bye':
        
                reply = bot.get_response(message)
                print('Doly: ', reply)
            if message.strip()=='Bye':
                print('Doly: Bye')
                break
           
        except:
            print("Sorry could not recognize what you said")
            
