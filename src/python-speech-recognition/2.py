#coding:utf-8
import speech_recognition as sr 
import difflib
text1="take a photo "
text2="now practice"
text3="oKay wait"
text4="apple"
# obtain audio from the microphone 
r = sr.Recognizer() 
#harvard = sr.AudioFile('audio_files/harvard.wav')#harvard.wav') jackhammer.wav
#harvard = sr.AudioFile('audio_files/harvard.wav')

#with harvard as source: 
##    audio = r.record(source) # recognize speech using Sphinx 

with sr.Microphone() as source:
    print("Say something!")
    r.adjust_for_ambient_noise(source, duration=0.3)
    audio = r.listen(source)
    #res=
    ans1=difflib.SequenceMatcher(None,text1,r.recognize_sphinx(audio))
    ans2=difflib.SequenceMatcher(None,text2,r.recognize_sphinx(audio))
    ans3=difflib.SequenceMatcher(None,text3,r.recognize_sphinx(audio))
    ans4=difflib.SequenceMatcher(None,text4,r.recognize_sphinx(audio))
try: 
    print("Sphinx thinks you said : " + r.recognize_sphinx(audio)) 
    print(ans1.ratio())
    print(ans2.ratio())
    print(ans3.ratio())
    print(ans4.ratio())
    c = [ans1.ratio(),ans2.ratio(),ans3.ratio()]
    print (c.index(max(c))+1)  # 返回最小值
except sr.UnknownValueError: 
    print("Sphinx could not understand audio") 
except sr.RequestError as e: 
    print("Sphinx error; {0}".format(e))

