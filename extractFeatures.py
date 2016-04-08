import os
import numpy as np
import scipy.io.wavfile as wav
import scipy 
from scikits.talkbox.features import mfcc

#
dataDirectories = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

data = np.zeros((900,1014))
labels = np.zeros((900,))
count = 0
for i in range(0,len(dataDirectories)):
    dir = "genres/" + dataDirectories[i]
    
    for fileName in os.listdir(dir):
        if fileName.endswith(".wav"):
            sample_rate, X = wav.read(dir + "/" + fileName)
            fft_features = abs(scipy.fft(X)[:1000])
            ceps, mspec, spec = mfcc(X)
            x = np.mean(ceps[int(4135*1/10):int(4135*9/10)], axis=0)
            data[count,0] = i
            data[count,1:] = np.concatenate((fft_features,x)) 
            #print data[count,0], data[count,1], data[count,2], data[count,1010]
            count = count + 1
            if count % 10 == 0:
                print count
#write to file

with open("train.data", "w") as f:
    for r in range (0,900):
        line = ''
        for c in range (0,1014):
            line = line + str(data[r,c])
            if c != 1013:
                line = line + ','
        line = line + '\n'
        f.write(line)
        if r % 10 == 0:
            print r