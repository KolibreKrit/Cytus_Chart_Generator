import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from scipy import signal
from scipy.fftpack import fft
from librosa.filters import mel
from librosa.display import specshow
from librosa import stft
from librosa.effects import pitch_shift
import pickle
import sys
from numba import jit, prange
from sklearn.preprocessing import normalize


class Audio:
    '''audio class which holds music data and timestamp for notes.'''

    def __init__(self, filename, stereo=True):
        self.data, self.samplerate = sf.read(filename, always_2d=True)
        if stereo is False:
            self.data = (self.data[:, 0]+self.data[:, 1])/2
        self.timestamp = []

    def plotaudio(self, start_t, stop_t):
        plt.plot(np.linspace(start_t, stop_t, stop_t-start_t),
                 self.data[start_t:stop_t, 0])
        plt.show()

    def save(self, filename="./savedmusic.wav", start_t=0, stop_t=None):
        if stop_t is None:
            stop_t = self.data.shape[0]
        sf.write(filename, self.data[start_t:stop_t], self.samplerate)

    def importtja(self, filename, verbose=False, diff=False, difficulty=None):
        '''imports tja file and convert it into timestamp'''
        now = 0.0
        bpm = 100
        measure = [4, 4]
        self.timestamp = []
        skipflag = False
        with open(filename, "rb") as f:
            while True:
                line = f.readline()
                try:
                    line = line.decode('sjis')
                except UnicodeDecodeError:
                    line = line.decode('utf-8')
                if line.find('//') != -1:
                    line = line[:line.find('//')]
                if line[0:5] == "TITLE":
                    if verbose:
                        print("importing: ", line[6:])
                elif line[0:6] == "OFFSET":
                    now = -float(line[7:-2])
                elif line[0:4] == "BPM:":
                    bpm = float(line[4:-2])
                if line[0:6] == "COURSE":
                    if difficulty and difficulty > 0:
                        skipflag = True
                        difficulty -= 1
                elif line == "#START\r\n":
                    if skipflag:
                        skipflag = False
                        continue
                    break
            sound = []
            while True:
                line = f.readline()
                # print(line)
                try:
                    line = line.decode('sjis')
                except UnicodeDecodeError:
                    line = line.decode('utf-8')

                if line.find('//') != -1:
                    line = line[:line.find('//')]
                if line[0] <= '9' and line[0] >= '0':
                    if line.find(',') != -1:
                        sound += line[0:line.find(',')]
                        beat = len(sound)
                        for i in range(beat):
                            if diff:
                                if int(sound[i]) in (1, 3, 5, 6, 7):
                                    self.timestamp.append(
                                        [int(100*(now+i*60*measure[0]/bpm/beat))/100, 1])
                                elif int(sound[i]) in (2, 4):
                                    self.timestamp.append(
                                        [int(100*(now+i*60*measure[0]/bpm/beat))/100, 2])
                            else:
                                if int(sound[i]) != 0:
                                    self.timestamp.append(
                                        [int(100*(now+i*60*measure[0]/bpm/beat))/100, int(sound[i])])
                        now += 60/bpm*measure[0]
                        sound = []
                    else:
                        sound += line[0:-2]
                elif line[0] == ',':
                    now += 60/bpm*measure[0]
                elif line[0:10] == '#BPMCHANGE':
                    bpm = float(line[11:-2])
                elif line[0:8] == '#MEASURE':
                    measure[0] = int(line[line.find('/')-1])
                    measure[1] = int(line[line.find('/')+1])
                elif line[0:6] == '#DELAY':
                    now += float(line[7:-2])
                elif line[0:4] == "#END":
                    if(verbose):
                        print("import complete!")
                    self.timestamp = np.array(self.timestamp)
                    break

    def synthesize(self, diff=True, don="data//don.wav", ka="data//ka.wav"):
        donsound = sf.read(don)[0]
        donsound = donsound[0,:]
        kasound = sf.read(ka)[0]
        kasound = kasound[0,:]
        donlen = len(donsound)
        kalen = len(kasound)
        if diff is True:
            for stamp in self.timestamp:
                timing = int(stamp[0]*self.samplerate)
                try:
                    if stamp[1] in (1, 3, 5, 6, 7):
                        self.data[timing:timing+donlen] += donsound
                    elif stamp[1] in (2, 4):
                        self.data[timing:timing +
                                  kalen] += kasound
                except ValueError:
                    pass
        elif diff == 'don':
            if isinstance(self.timestamp[0], tuple):
                for stamp in self.timestamp:
                    if stamp*self.samplerate+donlen < self.data.shape[0]:
                        self.data[int(stamp[0]*self.samplerate):int(stamp[0]*self.samplerate) + donlen] += donsound
            else:
                for stamp in self.timestamp:
                    if stamp*self.samplerate+donlen < self.data.shape[0]:
                        self.data[int(stamp*self.samplerate):int(stamp*self.samplerate) + donlen] += donsound
        elif diff == 'ka':
            if isinstance(self.timestamp[0], tuple):
                for stamp in self.timestamp:
                    if stamp*self.samplerate+kalen < self.data.shape[0]:
                        self.data[int(stamp[0]*self.samplerate):int(stamp[0]*self.samplerate) + kalen] += kasound
            else:
                for stamp in self.timestamp:
                    if stamp*self.samplerate+kalen < self.data.shape[0]:
                        self.data[int(stamp*self.samplerate):int(stamp*self.samplerate) + kalen] += kasound


def Frame(data, nhop, nfft):
    """helping function for fftandmelscale
    TODO: the sampling should be from the center of the biggest framesize 4096.
    ??????????????????????????????????????????nffts?????????????????????4096?????????????????????????????????????????????????????????"""
    length = data.shape[0]
    framedata = np.concatenate((data, np.zeros(nfft)))
    return np.array([framedata[i*nhop:i*nhop+nfft] for i in range(length//nhop)])


@jit
def fftandmelscaleikkatsu(song, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0, include_zero_cross=False):
    feat_channels = []
    for nfft in nffts:
        feats = []
        window = signal.blackmanharris(nfft)
        filt = mel(song.samplerate, nfft, mel_nband, mel_freqlo, mel_freqhi)
        frame = Frame(song.data, nhop, nfft)
        # frame = Frame2(data, nhop, nfft, nffts[-1])
        print(frame.shape)
        processedframe = fft(window*frame)[:, :nfft//2+1]
        processedframe = np.dot(filt, np.transpose(np.abs(processedframe)**2))
        processedframe = 20*np.log10(processedframe+0.1)
        # processedframe = normalize(processedframe, axis=1, copy=False)
        print(processedframe.shape)
        feat_channels.append(processedframe)
    if include_zero_cross:
        song.zero_crossing = np.where(np.diff(np.sign(song.data)))[0]
        print(song.zero_crossing)
    return np.array(feat_channels)


def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError

    if x.size < window_len:
        raise ValueError

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError

    s = np.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y

def musicfortest(serv, deletemusic=True, verbose=False):
    song = Audio(serv+'.wav', stereo=False)
    # song.importtja(glob(serv+"/*.tja")[-1])
    song.feats = fftandmelscaleikkatsu(song, include_zero_cross=False)
    with open('data/pickles/testdata.pickle', mode='wb') as f:
        pickle.dump(song, f)
