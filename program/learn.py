import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from musicprocessor import *
from tqdm import tqdm
import numpy as np

class convNet(nn.Module):
    """
    copies the neural net used in a paper "Improved musical onset detection with Convolutional Neural Networks".
    https://ieeexplore.ieee.org/document/6854953
    seemingly no-RNN version of rNet.
    """

    def __init__(self):
        super(convNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, (3, 7))
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(1120, 256)
        self.fc2 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120, 1)

    def forward(self, x, istraining=False, minibatch=1):
        x = F.max_pool2d(F.relu(self.conv1(x)), (3, 1))
        x = F.max_pool2d(F.relu(self.conv2(x)), (3, 1))
        x = F.dropout(x.view(minibatch, -1), training=istraining)
        x = F.dropout(F.relu(self.fc1(x)), training=istraining)
        x = F.dropout(F.relu(self.fc2(x)), training=istraining)
        return F.sigmoid(self.fc3(x))

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    @jit
    def learninggenerator(self, feats, answers, samplerate, soundlen=15, minibatch=1, split=1):
        idx = np.random.permutation(
            np.arange(answers.shape[0]-3*soundlen))+soundlen
        X, y = [], []
        j = 0
        for i in range(int(idx.shape[0]*split)):
            if answers[idx[i]] != 0:
                X.append(feats[:, :, idx[i]-soundlen//2:idx[i]+soundlen//2+1])
                y.append(answers[idx[i]])
                j += 1
                if j % minibatch == 0:
                    yield(torch.from_numpy(np.array(X)).float(), torch.from_numpy(np.array(y)).float())
                    X, y = [], []

    # return the data between notes with moderate interval.
    @jit
    def learninggenerator2(self, feats, answers, major_note_index, samplerate, soundlen=15, minibatch=1, split=1):
        # acceptable interval in seconds
        minspace = 0.1
        maxspace = 0.7
        idx = np.random.permutation(
            major_note_index.shape[0]-soundlen)+soundlen//2
        X, y = [], []
        j = 0
        for i in range(int(idx.shape[0]*split)):
            dist = major_note_index[idx[i]+1]-major_note_index[idx[i]]
            if dist < maxspace*samplerate/512 and dist > minspace*samplerate/512:
                for k in range(-1, dist+2):
                    X.append(feats[:, :, major_note_index[idx[i]]-soundlen //
                                   2+k:major_note_index[idx[i]]+soundlen//2+k+1])
                    y.append(answers[major_note_index[idx[i]]+k])
                    j += 1
                    if j % minibatch == 0:
                        yield(torch.from_numpy(np.array(X)).float(), torch.from_numpy(np.array(y)).float())
                        X, y = [], []

    def inferencegenerator(self, feats, soundlen=15, minibatch=1):
        x = []
        for i in range(feats.shape[2]-soundlen):
            x.append(feats[:, :, i:i+soundlen])
            if i % minibatch == minibatch-1:
                yield(torch.from_numpy(np.array(x)).float())
                x = []
        if len(x) != 0:
            yield(torch.from_numpy(np.array(x)).float())

    def train(self, songs, minibatch, epochs, device, soundlen=15, valsong=None, saveplace='../model/convmodel.pth', logplace='./out.txt', usedonka=0):
        """when usedonka is 0 uses both."""
        optimizer = optim.SGD(self.parameters(), lr=0.02)
        # optimizer = optim.Adam(self.parameters(), lr=0.03)

        for song in songs:
            timing = song.timestamp[:, 0]
            sound = song.timestamp[:, 1]
            song.answer = np.zeros((song.feats.shape[2]))
            if usedonka == 0:
                song.major_note_index = np.rint(timing[np.where(sound != 0)]
                                                * song.samplerate/512).astype(np.int32)
            else:
                song.major_note_index = np.rint(timing[np.where(sound == usedonka)]
                                                * song.samplerate/512).astype(np.int32)
                song.minor_note_index = np.rint(timing[np.where(sound == 3-usedonka)]
                                                * song.samplerate/512).astype(np.int32)
            song.major_note_index = np.delete(song.major_note_index, np.where(
                song.major_note_index >= song.feats.shape[2]))
            song.minor_note_index = np.delete(song.minor_note_index, np.where(
                song.minor_note_index >= song.feats.shape[2]))
            song.answer[song.major_note_index] = 1
            song.answer[song.minor_note_index] = 0.26
            song.answer = milden(song.answer)

        running_loss = 0
        val_loss = 0
        criterion = nn.MSELoss()
        # criterion = nn.CrossEntropyLoss()
        # validate if valsong is given
        if valsong:
            timing = valsong.timestamp[:, 0]
            sound = valsong.timestamp[:, 1]
            valsong.answer = np.zeros((valsong.feats.shape[2]))
            if usedonka == 0:
                valsong.major_note_index = np.rint(timing[np.where(sound != 0)]
                                                   * valsong.samplerate/512).astype(np.int32)
            else:
                valsong.major_note_index = np.rint(timing[np.where(sound == usedonka)]
                                                   * valsong.samplerate/512).astype(np.int32)
                valsong.minor_note_index = np.rint(timing[np.where(sound == 3-usedonka)]
                                                   * valsong.samplerate/512).astype(np.int32)
            valsong.major_note_index = np.delete(valsong.major_note_index, np.where(
                valsong.major_note_index >= valsong.feats.shape[2]))
            valsong.minor_note_index = np.delete(valsong.minor_note_index, np.where(
                valsong.minor_note_index >= valsong.feats.shape[2]))
            valsong.answer[valsong.major_note_index] = 1
            valsong.answer[valsong.minor_note_index] = 0.26
            valsong.answer = milden(valsong.answer)

        previous_val_loss = 1

        for i in range(epochs):
            print("epoch:", i)
            for song in songs:
                # for X, y in self.learninggenerator(song.feats, song.answer, song.samplerate, soundlen, minibatch, split=1):
                for X, y in self.learninggenerator2(song.feats, song.answer, song.major_note_indew, song.samplerate, soundlen, minibatch, split=0.2):
                    optimizer.zero_grad()
                    output = self(X.to(device), istraining=True,
                                  minibatch=minibatch)
                    target = y.to(device)
                    loss = criterion(output.squeeze(), target)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.data.item()
            with open(logplace, 'a') as f:
                print("running loss is", running_loss, file=f)
            running_loss = 0
            if valsong:
                inference = torch.from_numpy(self.infer(
                    valsong.feats, device, minibatch=512)).to(device)
                target = torch.from_numpy(
                    valsong.answer[:-soundlen]).float().to(device)
                loss = criterion(inference.squeeze(), target)
                val_loss = loss.data.item()
                with open(logplace, 'a') as f:
                    print("validation loss is \t\t\t\t",
                          val_loss, file=f)
                if previous_val_loss > val_loss:
                    torch.save(self.state_dict(), saveplace)
                    previous_val_loss = val_loss
                # val_loss = 0

    def infer(self, feats, device, minibatch=1):
        with torch.no_grad():
            inference = None
            for x in tqdm(self.inferencegenerator(feats, minibatch=minibatch), total=feats.shape[2]//minibatch):
                output = self(x.to(device), minibatch=x.shape[0])
                if inference is not None:
                    inference = np.concatenate(
                        (inference, output.cpu().numpy().reshape(-1)))
                else:
                    inference = output.cpu().numpy().reshape(-1)
            return np.array(inference).reshape(-1)

