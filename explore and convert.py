#%%
import tensorflow as tf
import tensorflow_io as tfio

audio = tfio.audio.decode_wav(r"./audio/UGK (Underground Kingz) - Int'l Players Anthem (I Choose You) (Director's Cut) ft. OutKast-awMIbA34MT8.wav")
# audio = tfio.audio.AudioIOTensor('gs://cloud-samples-tests/speech/brooklyn.flac')


# %%
print(audio)
# %%
import os
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import librosa, librosa.display
import skimage.io

# %%
album_path = r"D:\Python_Projects\flatiron\class-materials\capstone-data\music\Jay-Z\Discography@320kbps{gotfw}\{1996~2013}\01. Studio Albums\(1996)Reasonable Doubt\\"
files = os.listdir(album_path)

artist_album = "jayz_reasonable_doubt"
img_num = 0

for file in tqdm(files[:1]):
    img_num += 1
    file_joined = os.path.join(album_path, file)
    signal, sr = librosa.load(file_joined, sr = 22050)
    y, sr = librosa.load(file_joined, sr=32000, mono=True)
    melspec = librosa.feature.melspectrogram(y, sr=sr, n_mels = 128)
    melspec = librosa.power_to_db(melspec).astype(np.float32)

    fig = plt.Figure()
    # canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    img = librosa.display.specshow(melspec, ax=ax, x_axis='time', y_axis='mel',sr=sr, fmax=16000)
    fig.savefig(fr'D:\Python_Projects\flatiron\class-materials\capstone_data\mel-specs\{artist_album}\{file.replace(".mp3","")}.png')
    # skimage.io.imsave("melspec.png",img)
# file = r"D:\Python_Projects\flatiron\class-materials\capstone-data\music\Jay-Z\Discography@320kbps{gotfw}\{1996~2013}\01. Studio Albums\(1996)Reasonable Doubt\01 Can't Knock The Hustle.mp3"
# file = r"..\capstone-audio\jayz\reasonable-doubt\Cashmere Thoughts-Vf2LNPFLPAE.m4a"

# %%
# short way - idk if the other way is better or not
#%%
# visualizing waveform to see time vs amplitude - don't need for CNN
librosa.display.waveplot(signal, sr=sr)
plt.xlabel("Time")
plt.ylabel("Amplitutde")
plt.show()
# %%
# processing fast fourier transform (fft) and creating a power spectrum to view frequency vs magnitude
fft = np.fft.fft(signal)

magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

# only getting the first half becasue the second is a reflection of the same information
left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(frequency)/2)]

plt.plot(left_frequency, left_magnitude)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.show()
# %%
# processing short time fourier tranform (stft) and creating a spectogram to view time vs frequency

# number of samples per fft - creating the size of the window of sound to gather data from
n_fft = 2048
# how far to shift this window for each interval to gather data and represent it over time
hop_length = 512

stft = librosa.core.stft(signal,hop_length=hop_length, n_fft=n_fft)
spectogram=np.abs(stft)
# applying a logto convert amplitude to decibles to show a better spectogram
log_spectogram = librosa.amplitude_to_db(spectogram)

librosa.display.specshow(log_spectogram, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()
# %%
# processing mel frequency cepstrum coefficient (MFCC) to better represent how humans process sound and 
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