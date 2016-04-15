import os
import numpy as np
import scipy.io.wavfile as wav
import scipy 
from scikits.talkbox.features import mfcc
import librosa


dataDirectories = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

numberOfSamples = 900
numberOfFeaturesPlusLabel = 1014
data = np.zeros((numberOfSamples,numberOfFeaturesPlusLabel))

numberOfValidationSamples = 100
numberOfValidationFeatures = 1013
validationData = np.zeros((numberOfValidationSamples,numberOfValidationFeatures))


labels = np.zeros((numberOfSamples,))
count = 0
for i in range(0,len(dataDirectories)):
    dir = "genres/" + dataDirectories[i]
    
    for fileName in os.listdir(dir):
        if fileName.endswith(".wav"):
            sample_rate, X = wav.read(dir + "/" + fileName)
            fft_features = abs(scipy.fft(X)[:1000])
            
            ceps, mspec, spec = mfcc(X)
            numCeps = len(ceps)
            x = np.mean(ceps[int(numCeps*1/10):int(numCeps*9/10)], axis=0)
            
            #y, sr = librosa.load(dir + "/" + fileName)
            #energyValues = librosa.feature.rmse(y=y)
            #energy = np.zeros((1,))
            #energy[0] =  np.mean(energyValues[0,:])
            
            data[count,0] = i
            data[count,1:] = np.concatenate((fft_features,x)) 
            #data[count,1:] = x 
            
            #print data[count,0], data[count,1], data[count,2], data[count,1010]
            count = count + 1
            if count % 10 == 0:
                print count
                
#write to file
print data

with open("train-fft-mfcc.data", "w") as f:
    for r in range (0,numberOfSamples):
        line = ''
        for c in range (0,numberOfFeaturesPlusLabel):
            line = line + str(data[r,c])
            if c != (numberOfFeaturesPlusLabel -1):
                line = line + ','
        line = line + '\n'
        f.write(line)
        if r % 10 == 0:
            print r
            
            
validate_dir = 'validation'

count = 0
for fileName in os.listdir(validate_dir):
    if fileName.endswith(".wav"):
        sample_rate, X = wav.read(validate_dir + "/" + fileName)
        fft_features = abs(scipy.fft(X)[:1000])
            
        ceps, mspec, spec = mfcc(X)
        numCeps = len(ceps)
        x = np.mean(ceps[int(numCeps*1/10):int(numCeps*9/10)], axis=0)
            
        #y, sr = librosa.load(dir + "/" + fileName)
        #energyValues = librosa.feature.rmse(y=y)
        #energy = np.zeros((1,))
        #energy[0] =  np.mean(energyValues[0,:])
            
        
        validationData[count] = np.concatenate((fft_features,x)) 
        #data[count,1:] = x 
            
        #print data[count,0], data[count,1], data[count,2], data[count,1010]
        count = count + 1
        if count % 10 == 0:
            print count
            
with open("validation-fft-mfcc.data", "w") as f:
    for r in range (0,numberOfValidationSamples):
        line = ''
        for c in range (0,numberOfValidationFeatures):
            line = line + str(validationData[r,c])
            if c != (numberOfValidationFeatures -1):
                line = line + ','
        line = line + '\n'
        f.write(line)
        if r % 10 == 0:
            print r