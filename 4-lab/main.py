import pyttsx3
import time
import webbrowser as wb
import speech_recognition as sr
import random

from data import data

recognizer = sr.Recognizer()
engine = pyttsx3.init()
  
def speech_to_text():
  with sr.Microphone() as source:
    print("Говоріть...")
    audio = recognizer.listen(source)

  text = ""
  if audio:
    try:
      text = recognizer.recognize_google(audio, language="uk-UA")
      print("Ви сказали: " + text)
    except BaseException:
      print("Не розумію вас. Можливо, стался помилка")
  return text

def speak(text: str):
  print(text)
  engine.say(text)
  engine.runAndWait()
  engine.stop()

def redirect(link: str):
  wb.open(link)

def process_voice_command(text):
  if "бувай" in text.lower():
    speak("До побачення! Гарного дня!")
    return True
  
  if "щось" in text.lower():
    index = random.randint(0, len(data.keys()) - 1)
    speak(data[list(data.keys())[index]][0])
    
  for key in data.keys():
    if key in text.lower():
      redirect(data[key][1])
      speak(data[key][0])
      break

if __name__ == "__main__":
  turned_off = False
  while not turned_off:
    text = speech_to_text()
    time.sleep(2)
    turned_off = process_voice_command(text)
