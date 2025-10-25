import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

filename = librosa.example('nutcracker')

y, sr = librosa.load(filename)

# get a 2D array[row: amplitudes at various frequencies, col: timeframes]
# using short-time fourier transform
D = librosa.stft(y)

# convert to decibels
S_db = librosa.amplitude_to_db(abs(D), ref=np.max)

print(f"Sample rate: {sr}")
print(f"Duration: {len(y) / sr} second")

# make a waveform
librosa.display.waveshow(y, sr=sr)
plt.title("Waveform of Example Audio")

# make a spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
plt.colorbar(format='%+2.0f db')
plt.title("Spectrogram of Example Audio")
plt.show()