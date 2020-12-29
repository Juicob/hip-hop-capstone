
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
top_dir_path = r"D:\Python_Projects\flatiron\class-materials\capstone-data\music"
top_dir_list = os.listdir(top_dir_path)

# for folder in top_dir_list:
    # print(os.path.join(top_dir_path, folder))
artist_folders = [os.path.join(top_dir_path, folder) for folder in top_dir_list]
print(artist_folders)
for folder in artist_folders:
    print(os.listdir(folder))

# ! Reorganize folder structure in regions > artists > albums > song files