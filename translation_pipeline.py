from scipy.io import wavfile
import numpy as np
import os
import audacity_pipe
import timbre_transfer
from utils import (get_model, to_file, QuantileTransformer)


def translate(curr_dir, song_dir, fname, from_genre, to_genre):
    song_path = f"{song_dir}{fname}"
    print(song_path)

    print("\nPerforming source separation on the input song")
    os.system(f"python3 -m demucs.separate {song_path} -d cpu")
    tracks_path = f"{curr_dir}separated/demucs_quantized/{fname[:-4]}/"


    print("\nConverting separated tracks to mono")
    mono_tracks = []
    for filename in os.listdir(tracks_path):
        print(f" {filename}")
        filename = f"{tracks_path}{filename}"
        samplerate, audiodata = wavfile.read(filename)

        newaudiodata = []

        for i in range(len(audiodata)):
            d = (audiodata[i][0]/2) + (audiodata[i][1]/2)
            newaudiodata.append(d)

        out = np.array(newaudiodata, dtype='int16')
        mono_tracks.append((filename, samplerate, out))

    # write output files in one go, otherwise the previous loop would misbehave
    for filename, samplerate, data in mono_tracks:
        wavfile.write(filename.split('.')[0]+"mono.wav", samplerate, data)
    mono_tracks = []


    print("\nPerform timbre transfer on all mono files")
    timbre_transfered = []
    for filename in os.listdir(tracks_path):
        if not filename.endswith("mono.wav"):
            continue
        model = get_model(filename, from_genre, to_genre)
        if model == None:
            continue
        print(f"Filename = {filename}\nModel = {model}")
        data, out_file, sr = timbre_transfer.process(curr_dir, tracks_path, filename, model)
        timbre_transfered.append((data, out_file, sr))

    # write output files in one go, otherwise the previous loop would misbehave
    for data, out_file, sr in timbre_transfered:
        to_file(data, out_file, sample_rate=sr)


    print("\nCombining all tracks and necessary editing through Audacity")
    audacity_pipe.process(f"{song_dir}{fname}", tracks_path, from_genre, to_genre)
