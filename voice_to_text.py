import speech_recognition as sr

def record_and_transcribe():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Speak now...")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError:
        print("Could not request results; check your internet")
        return ""
