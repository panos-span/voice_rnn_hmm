import librosa
import os
import re

'''
Φτιάξτε μία συνάρτηση (data parser) που να διαβάζει όλα τα αρχεία ήχου που δίνονται μέσα στο φάκελο digits/
και να επιστρέφει 3 λίστες Python, που να περιέχουν: Το wav που διαβάστηκε με librosa, τον αντίστοιχο ομιλητή
και το ψηφίο

'''


def data_parser(directory):
    wavs = []
    speakers = []
    digits = []
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        # Sr = 16000 γιατί τα αρχεία ήχου έχουν δείγματα στα 16kHz
        wav , sr = librosa.load(filepath, sr=16000)
        wavs.append(wav)
        
        # Εξαγωγή ψηφίου και ομιλητή από το όνομα του αρχείου
        name_part = filename.split('.')[0]
        match = re.match(r"([a-zA-Z]+)(\d+)", name_part)
        if match:
            digits.append(match.group(1))
            speakers.append(int(match.group(2)))
        else:
            digits.append(None)
            speakers.append(None)
            
    return wavs, speakers, digits
