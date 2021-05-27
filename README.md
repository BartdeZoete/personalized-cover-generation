# Personalized Cover Generation
This package contains code for a final project in the course of Audio Processing
and Indexing at Leiden University. The code can be used to perform cover song
detection, genre classification, and automatically create jazz covers from pop
songs.

## Getting Started
Several packages have to be installed to succesfully to run the code. The list below might not be complete but it should cover most of the work and worked with all our tested standard setups:
```bash
#for song creation
pip3 install tensorflow
pip3 install crepe
pip3 install ddsp==1.0.1
pip3 install librosa
pip3 install demucs
pip3 install tables
#for genre classification
pip3 install disarray
pip3 install fastai==1.0.61
pip3 install pywaffle
pip3 install seaborn
pip3 install h5py
pip3 install pandas
```
We recommend installing with conda to avoid conflicts.

When creating covers, [Audacity](https://www.audacityteam.org/) is required. Before creating your first cover, set ```mod-script-pipe``` to ```Enabled``` under Edit->Preferences->Modules. This setting needs to be changed just once and is active after re-opening Audacity. Make sure to keep Audacity open at any time when creating covers.

## Cover Song Creation
To create a cover for a single song, open the file ```process_single_song.py``` and **change the paths and filenames** at the top of the file to the appropriate values. Then run the following to create the cover:
```bash
python3 process_single_song.py
```
The cover will be stored in the same directory as where the original file is
located. Be sure to have audacity open during all of this process! No further steps should be needed, the program will take care of the rest.

Note that this program is quite demanding to run, especially for songs of more
than a few minutes long (people running the program for the first time might also first have to wait until the download of all the pretrained models is complete, which can take up to several minutes).

###Additional datasets
To copy our experiments and carry out further exploration please first install the required datasets which we can not provide within this github repo:
1) The MSD
Installation is not trivial and requires some knowledge about aws services. For the most direct step by step guid follow the outline on: http://millionsongdataset.com/pages/getting-dataset/

2) GTZAN: either download the orginal version from http://marsyas.info/downloads/datasets.html or download an already process version from kaggle: https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification (preferred).

### Cover Song Classification
To create a batch of covers and apply cover song classification on them, open
```classify_covers.py``` and set the paths and filenames to appropriate values.
When this is done, run:
```bash
python3 classify_covers.py
```
This script reads from a file which songs to turn into covers, and after each
translation is complete it classifies if the output is indeed a cover or not of
the original song. This can be seen as an objective metric of the quality of the
created cover. The cover classification is a binary task,  outputting whether the
generated song is a cover of the intended song or not. This is currently designed
to work with tracks found the Million Song Dataset,specifically tracks with 
greater than 15 covers available. A all_tracks_clusters15_titles.txt.txt file 
contains all the available songs (>15 covers) and the cluster number they belong to.
You will need the cluster number for the cover song classification. 

### Genre Classification
To recreate or experiment with our genre classification algorithms one may have a look into the provided genre jupyter notebooks.
One must first run the genre explorer, which will provide several insight into the datasets along side an approach of traditional classification with a decision tree, a random forest model and support vector machines.
In the end the notebook also creates the visual representation of the dataset which is needed to run the CNN models.
To explor the CNN models look into the 3 genre classification notebooks for the respective datasets.