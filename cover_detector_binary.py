import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import scipy.io.wavfile
import librosa
import beat_aligned_feats_binary as bafb
from pydub import AudioSegment
from pydub.utils import mediainfo



def binary_cover_classification(path, wav_file, cluster_number):
    #input must be the cover song mp3 file and the cluster number of the songs
    #it belongs to which can be found in the file  "all_tracks_clusters15_titles.txt"

    #method: the method will be a 2 category knn clustering of all the cover songs of
    #that title and another title chosen at random

    #process the wav song in the same manner that was processesed for the core analysis
    loc  = path+wav_file
    sound = AudioSegment.from_wav(loc)
    y,sr=librosa.load(loc,sr=None)
    sound = sound.set_channels(1)
    sound = sound.get_array_of_samples()
    if len(np.shape(sound))>1:
        sound = np.squeeze(sound)
    test_audio = np.array(sound,dtype=float)
    chroma = librosa.feature.chroma_stft(test_audio, sr =sr).T

    tempo,btstarts = librosa.beat.beat_track(test_audio, sr=sr)
    segstarts = librosa.onset.onset_detect(test_audio, sr=sr)
    segstarts2 = np.arange(0,segstarts[-1],segstarts[-1]/np.shape(chroma)[0])
    duration =len(test_audio)*1/sr
    #features, beat alignment
    btchromas = bafb.get_btchromas(segstarts2, btstarts,chroma,duration)

    if sum(np.shape(btchromas))==0: #some file tracks are empty
        print('File appears to be empty')
    else:

        #increase chroma contrast
        chroma_contrast = btchromas.T**1.96

        #create patches of chroma
        patches = []
        for j in range(len(chroma_contrast)-75):
            patch = chroma_contrast[j:j+75]
            patches.append(patch)


        #fft
        fft = np.fft.fft2(patches)
        fft_r = np.reshape(fft, (np.shape(patches)[0],900,))
        # song_ffts.append(fft_patch)
        #take the median
        wav_fft_med = np.absolute(np.median(fft_r, axis=0))
        # song_ffts_med.append(fft_med[:].real)



    #load all the cluster numbers of songs with more than 15 covers
    cluster_nums = np.load('data/cluster_nums.npy')
    ffts = np.load('data/fft_meds.npy',allow_pickle=True)

    cover_songs_inds = cluster_nums==cluster_number
    ffts_cover = ffts[cover_songs_inds,:]
    y_cover = np.empty(np.shape(ffts_cover)[0])
    y_cover.fill(0)

    #generate a random number to have as the "other" song
    cluster_range = list (range(0,29))
    cluster_range.pop(cluster_number)
    random_cluster = np.random.choice(cluster_range)

    #extract "other" songs
    other_song_ids = cluster_nums==random_cluster
    ffts_other = ffts[other_song_ids,:]
    y_other = np.empty(np.shape(ffts_other)[0])
    y_other.fill(1)

    #concatenate the data from covers and others for the model
    ffts_model = np.concatenate((ffts_cover,ffts_other))
    y_model = np.concatenate((y_cover, y_other))

    #create a knn clustering model
    #model parameters are from the grid search for the best parameters in the core analysis
    model = KNeighborsClassifier(metric='euclidean',n_neighbors=26, weights = 'uniform')
    model.fit(ffts_model,y_model)

    #predict the song cluster
    wav_cover_binary = model.predict(np.reshape(wav_fft_med,(-1,1)).T)
    if wav_cover_binary ==0:
        print('Song is a cover song.')
    else:
        print('Song is not a cover song')

    return 1 - wav_cover_binary


if __name__ == "__main__":
    binary_cover_classification('HYAMLC_cover.wav', 28)



