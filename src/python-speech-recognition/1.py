import speech_recognition as sr
from guessing_game import recognize_speech_from_mic
r = sr.Recognizer()
m = sr.Microphone()
recognize_speech_from_mic(r, m)  # speak after running this line
#{'success': True, 'error': None, 'transcription': 'hello'}
