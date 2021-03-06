
# %%
import os
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import librosa, librosa.display
import skimage.io

# %%
%%time
album_path = r"D:\Python_Projects\flatiron\class-materials\capstone-data\music\Jay-Z\Discography@320kbps{gotfw}\{1996~2013}\01. Studio Albums\(1996)Reasonable Doubt\\"
files = os.listdir(album_path)

artist_album = "jayz_reasonable_doubt"
img_num = 0

for file in tqdm(files):
    img_num += 1
    file_joined = os.path.join(album_path, file)
    signal, sr = librosa.load(file_joined, sr = 22050)
    y, sr = librosa.load(file_joined, sr=32000, mono=True)
    melspec = librosa.feature.melspectrogram(y, sr=sr, n_mels = 128)
    melspec = librosa.power_to_db(melspec).astype(np.float32)

    fig = plt.Figure()
    # canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    img = librosa.display.specshow(melspec, ax=ax, sr=sr, fmax=16000, cmap='bone') # x_axis='time', y_axis='mel'
    fig.savefig(fr'../capstone-data/mel-specs/{artist_album}/{file.replace(".mp3","")}.png')


# %%
top_dir_list = os.listdir(top_dir_path)

# for folder in top_dir_list:
    # print(os.path.join(top_dir_path, folder))
artist_folders = [os.path.join(top_dir_path, folder) for folder in top_dir_list]
# print(artist_folders)
for folder in artist_folders:
    print(os.listdir(folder))

# ! Reorganize folder structure in regions > artists > albums > track files
# %%
top_dir_path = "D:/Python_Projects/flatiron/class-materials/capstone-data/music - Copy"
train_path = os.path.join(top_dir_path, 'TRAIN')
# !                                                                                 ============== Gate
# while True:
for region in os.listdir(train_path)[:]:
    print(region)
    region_path = os.path.join(train_path, region)
    # print(region_path)
    for artist in tqdm(os.listdir(region_path)[:]): #sub region path for train path
        print(artist)
        artist_path = os.path.join(region_path, artist) #sub region path for train path
        print('\n',artist_path)
        if os.path.isdir(artist_path):
            for album in tqdm(os.listdir(artist_path)[:]):
                album_path = os.path.join(artist_path, album)
                print(region, artist, album)
                if os.path.isdir(album_path):
                    for track in tqdm(os.listdir(album_path)[:]):
                        track_path = os.path.join(album_path, track)
                        if track.endswith('.mp3'):
                            # print(track)
                            try:
                                print(track_path)
                                signal, sr = librosa.load(track_path, sr = 22050)
                                y, sr = librosa.load(track_path, sr=32000, mono=True)
                                melspec = librosa.feature.melspectrogram(y, sr=sr, n_mels = 128)
                                melspec = librosa.power_to_db(melspec).astype(np.float32)

                                fig = plt.Figure()
                                # canvas = FigureCanvas(fig)
                                ax = fig.add_subplot(111)
                                img = librosa.display.specshow(melspec, ax=ax, sr=sr, fmax=16000, cmap='gist_ncar') # x_axis='time', y_axis='mel'
                                fig.savefig(track_path.replace(".mp3",".png"))
                            except:
                                print('CORRUPTED FILE')
                # ! remove file if .jpg




        # print(region_path)
# %%
# waveplot sample
plt.figure(figsize=(10,4))
y, sr = librosa.load(r'D:\Python_Projects\flatiron\class-materials\capstone-data\music\TRAIN\East\Jay-Z\(1998)Vol. 2 ~ Hard Knock Life\02 Hard Knock Life (The Ghetto Anthem).mp3', mono=True)
librosa.display.waveplot(y[:500000].flatten(), sr=sr)


# %%
# mel spec sample - maybe rerun on all files in the future to classify snippets of songs instead of the entire thing
y, sr = librosa.load(r'D:\Python_Projects\flatiron\class-materials\capstone-data\music\TRAIN\East\Jay-Z\(1998)Vol. 2 ~ Hard Knock Life\02 Hard Knock Life (The Ghetto Anthem).mp3', sr=32000, mono=True)
melspec = librosa.feature.melspectrogram(y[500000:1500000], sr=sr, n_mels = 128)
melspec = librosa.power_to_db(melspec).astype(np.float32)

fig = plt.figure(figsize=(20,4))
fig2 = plt.figure()
# canvas = FigureCanvas(fig)
ax = fig.add_subplot(111)
ax2 = fig2.add_subplot(111)
img = librosa.display.specshow(melspec, ax=ax, sr=sr, fmax=16000, cmap='gist_ncar') # x_axis='time', y_axis='mel'
img2 = librosa.display.specshow(melspec, ax=ax2, sr=sr, fmax=16000, cmap='gist_ncar') # x_axis='time', y_axis='mel'
fig.savefig(fr'../capstone-data/mel-specs/testttt.png')