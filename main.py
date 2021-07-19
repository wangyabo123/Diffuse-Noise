# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 16:36:11 2021

@author: Dell
"""

#matlab2python
import numpy as np
import scipy.io.wavfile
from scipy.stats import norm
import librosa
import math
from scipy.signal import stft, istft
from scipy.linalg import cholesky
import soundfile
#Init parameters
Fs=8000 #frequency sample
c=340 #Speed of sound transmission 
K=256 #FFT length
M=2 #Numbers of sensors
d=0.1 #Distance between sensors
type_nf='spherical' 
L=20*Fs #The length of the data

#**********The function of mix_signals****************** 
def mix_signals(n, DC, method):
    M=n.shape[1] #numbers of sensors
    K=(DC.shape[2]-1)*2 #numbers of frequency bins
    n=np.vstack([np.zeros([K//2,M]),n,np.zeros([K//2,M])])
    n=n.transpose()
    f,t,N=stft(n,window='hann',nperseg=K,noverlap=0.75*K, nfft=K)
    X=np.zeros(N.shape,dtype=complex)
    for k in range(1,K//2+1):
        C=cholesky(DC[:,:,k])
        X[:,k,:] = np.dot(np.squeeze(N[:,k,:]).transpose(),np.conj(C)).transpose()
    #do istft
    F,x = istft(X,window='hann',nperseg=K,noverlap=0.75*K, nfft=K)
    x=x.transpose()[K//2:-K//2,:]
    return x
#**************end*****************************************

#Generate M mutually 'independent' babble speech input signals 

data,Fs_data = librosa.load('./babble_8KHZ.wav',8000)
data=data-np.mean(data)

babble = np.zeros([L,M])  
for m in range(0,M):
    babble[:,m]=data[m*L:(m+1)*L]

#Generate matrix with desired spatial coherence

ww = 2*math.pi*Fs*np.array([i for i in range(K//2+1)])/K
DC = np.zeros([M,M,K//2+1])
for p in range(0,M):
    for q in range(0,M):
        if p==q:
            DC[p,q,:] = np.ones([1,1,K//2+1])
        elif type_nf=='spherical':
            DC[p,q,:] = np.sinc(ww*np.abs(p-q)*d/(c*math.pi))
        else:
            print('error')
            
#Mix signals

x = mix_signals(babble, DC, 'cholesky').transpose()

#save the file
soundfile.write('./for_test.wav',x.transpose(),Fs)



