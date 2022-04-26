import pickle
import numpy as np
from musicprocessor import *
from scipy.signal import argrelmax
from librosa.util import peak_pick
from librosa.onset import onset_detect


def by_librosa_detection2(inference, song):
    inference = smooth(inference, 5)
    song.timestampboth = (peak_pick(inference, 1, 2, 4, 5,
                                0.12, 3)+7)
    song.timestamp=song.timestampboth*512/song.samplerate
    print(len(song.timestamp))
    song.synthesize(diff=False)
    song.save("data//inferredmusic.wav")

def create_tja(filename, song, timestampdon, timestampka=None):
    if timestampka is None:
        timestamp=timestampdon*512/song.samplerate
        with open(filename, "w") as f:
            f.write('TITLE: xxx\nSUBTITLE: --\nBPM: 240\nWAVE:xxx.ogg\nOFFSET:0\n#START\n')
            i = 0
            time = 0
            while(i < len(timestamp)):
                if time/100 >= timestamp[i]:
                    f.write('1')
                    i += 1
                else:
                    f.write('0')
                if time % 100 == 99:
                    f.write(',\n')
                time += 1
            f.write('#END')
    else:
        timestampdon=np.rint(timestampdon*512/song.samplerate*100).astype(np.int32)
        timestampka=np.rint(timestampka*512/song.samplerate*100).astype(np.int32)
        with open(filename, "w") as f:
            f.write('TITLE: xxx\nSUBTITLE: --\nBPM: 240\nWAVE:xxx.ogg\nOFFSET:0\n#START\n')
            for time in range(np.max((timestampdon[-1],timestampka[-1]))):
                if np.isin(time,timestampdon) == True:
                    f.write('1')
                elif np.isin(time,timestampka) == True:
                    f.write('2')
                else:
                    f.write('0')
                if time%100==99:
                    f.write(',\n')
            f.write('#END')

    
