# Cytus Project
Creates charts for Cytus II based on music

# How to make charts
To create charts, first run song file through SpleeterGUI (Parts to separate = 4; check Recombine: Vocal, Bass, and Other)

Move [song_name].wav, [song_name]-recombine.wav, and [song_name]/drums.wav to data/songs/[song_name]; 

Rename [song_name]-recombine.wav to [song_name]-recombined.wav
and drums.wav to [song_name]-drums.wav.

This is important for making charts that play on both sides of the screen; using Spleeter is not required as long as you have multiple tracks of the same song. Just be sure to rename them and put them in data/songs.

To create charts from your tracks, run
```
python program/make.py [song_name] (OPTIONAL) [bpm]
```
Example:
```
python program/make.py One-Last-You 122
```
which will produce Final_[song_name].json which can then be opened in a Cytus editor/program to transfer to Cytoid.

Procedural generation does not work properly: try at your own risk
Then if you want procedurally generated note placements, run
```
python Procedural.py [song_name]
```
(Final_[song_name].json must have been made) which will produce Procedural_[song_name].json


# Acknowledgements
Credit for timing models goes to https://gitlab.com/woodyZootopia

BPM detector taken from: https://github.com/scaperot/the-BPM-detector-python

Spleeter is created by Deezer

SpleeterGUI made by Chris Mitchell

# Warning
Copyright of the music does not belong to the owner of this repository