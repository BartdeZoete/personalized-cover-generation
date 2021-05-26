# Personalized Cover Generation
This package contains code for a final project in the course of Audio Processing
and Indexing at Leiden University. The code can be used to perform cover song
detection, genre classification, and automatically create jazz covers from pop
songs.

## Getting Started
Several packages are needed to run the code. The list below might not be complete but it should cover most of the work:
```bash
pip3 install tensorflow
pip3 install crepe
pip3 install ddsp==1.0.1
pip3 install librosa
pip3 install demucs
pip3 install tables
```

When creating covers, [Audacity](https://www.audacityteam.org/) is required. Before creating your first cover, set ```mod-script-pipe``` to ```Enabled``` under Edit->Preferences->Modules. This setting needs to be changed just once and is active after re-opening Audacity. Make sure to keep Audacity open at any time when creating covers.

To create a cover for a single song, open the file ```process_single_song.py``` and change the paths and filenames at the top of the file to the appropriate values. Then run the following to create the cover:
```bash
python3 process_single_song.py
```
Note that this program is quite demanding to run, especially for longer songs.
