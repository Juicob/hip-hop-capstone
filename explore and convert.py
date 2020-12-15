#%%
import tensorflow as tf
import tensorflow_io as tfio

audio = tfio.audio.decode_wav(r"./audio/UGK (Underground Kingz) - Int'l Players Anthem (I Choose You) (Director's Cut) ft. OutKast-awMIbA34MT8.wav")
# audio = tfio.audio.AudioIOTensor('gs://cloud-samples-tests/speech/brooklyn.flac')


# %%
print(audio)
# %%
import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np



# %%
file = "./audio/UGK (Underground Kingz) - Int'l Players Anthem (I Choose You) (Director's Cut) ft. OutKast-awMIbA34MT8.wav"

signal, sr = librosa.load(file, sr = 22050)
# %%
# short way - idk if the other way is better or not
y, sr = librosa.load(file, sr=32000, mono=True)
melspec = librosa.feature.melspectrogram(y, sr=sr, n_mels = 128)
melspec = librosa.power_to_db(melspec).astype(np.float32)
librosa.display.specshow(melspec, x_axis='time', y_axis='mel',sr=sr, fmax=16000)
#%%
librosa.display.waveplot(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitutde")
plt.show()
# %%
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(frequency)/2)]

plt.plot(left_frequency, left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()
# %%
n_fft = 2048
hop_length = 512

stft = librosa.core.stft(signal,hop_length=hop_length, n_fft=n_fft)
spectogram=np.abs(stft)

log_spectogram = librosa.amplitude_to_db(spectogram)

librosa.display.specshow(log_spectogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()
# %%
MFCCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()
# %%

#%%
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
spectrogram_extractor = Spectrogram()
logmel_extractor = LogmelFilterBank()
y = spectrogram_extractor(y)
y = self.logmel_extractor(y)