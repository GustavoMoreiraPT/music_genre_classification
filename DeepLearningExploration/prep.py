import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

genre = "rock"
genreCapital = "Rock"
file = "{}.00000.wav".format(genre)

# waveforms
signal, sr = librosa.load(file, sr=22050) # sr * T (duration) -> 22050 * 30 seconds
librosa.display.waveplot(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("{} waveforms".format(genreCapital))
plt.show()

# fft -> spectrum
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

left_frequency = frequency[:int(len(frequency) / 2)]
left_magnitude = magnitude[:int(len(frequency) / 2)]

plt.plot(left_frequency, left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("{} FFT".format(genreCapital))
plt.show()

# stft -> spectogram

n_fft = 2048
hop_length = 512

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectogram = np.abs(stft)

log_spectogram = librosa.amplitude_to_db(spectogram) # taking original spectorgram and convert to decibal, more perceptive to humans

librosa.display.specshow(log_spectogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.title("{} STFT".format(genreCapital))
plt.show()

# MFCCs
MFFCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFFCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.title("{} MFCC".format(genreCapital))
plt.colorbar()
plt.show()