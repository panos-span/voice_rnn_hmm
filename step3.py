'''
Εξάγετε με το librosa τα Mel-Frequency Cepstral Coefficients (MFCCs) για κάθε αρχείο ήχου. Εξάγετε 13
χαρακτηριστικά ανά αρχείο. Χρησιμοποιήστε μήκος παραθύρου 25 ms και βήμα 10 ms. Επίσης, υπολογίστε και
την πρώτη και δεύτερη τοπική παράγωγο των χαρακτηριστικών, τις λεγόμενες deltas και delta-deltas (hint:
υπάρχει έτοιμη υλοποίηση στο librosa).
'''

import librosa

def extract_mfccs(wavs):
    mfccs = []
    
    for wav in wavs:
        mfcc = librosa.feature.mfcc(y=wav, sr=16000, n_mfcc=13, n_fft=400, hop_length=160)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        
        mfccs.append((mfcc, delta, delta2))
        
    return mfccs