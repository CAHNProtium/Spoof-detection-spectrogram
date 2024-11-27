#Autor: Carlos Alberto Hernández Nava
#ASVspoof2017 Spectrogram extractor an convert to csv

import sys, csv, os, time, shutil, cv2
import pydub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import soundfile as sf
from PIL import Image
from pydub import AudioSegment
from os import remove
import librosa

#paths y vars
pathflac1="/ASVspoof2019LA/traindev/"
pathflac2="/ASVspoof2019LA/dev/"
pathflac3="/ASVspoof2019LA/eval/"

path1="/spectro/traindev/"
path2="/spectro/dev/"
path3="/spectro/eval/"

dataclase1 = pd.read_excel("/traindev2019LA.xlsx", header=None ,names=["Nombre", "Clase"])
dataclase2 = pd.read_excel("/dev2019LA.xlsx", header=None ,names=["Nombre", "Clase"])
dataclase3 = pd.read_excel("/eval2019LA.xlsx", header=None ,names=["Nombre", "Clase"])


    
def convertireval():
  cont=0
  for i in dataclase3.index:
    if dataclase3["Clase"][i] == 'spoof':
      archivospo = dataclase3["Nombre"][i]
      #spectrogram
      sig, fs = librosa.load(pathflac3+archivospo+'.wav', sr=16000, duration=3, NFTT=1024)
      Pxx, freqs, bins, im = plt.specgram(sig, Fs=fs)
      im.set_cmap('jet')
      yticks = [100,10000] #yticks line to erase to log3 to 4
      plt.yticks(yticks) #remove this line to log3 to log 4
      plt.axis("off")
      plt.savefig(path3+'spoof/'+archivospo+'.png', bbox_inches='tight', pad_inches = 0)
      plt.close()
      #cut, resize
      image = cv2.imread(path3+'spoof/'+archivospo+'.png')
      crop_img = image[30:217, 0:334]
      image = cv2.resize(crop_img, (70, 70))
      cv2.imwrite(path3+'spoof/'+archivospo+'.png',image)
      del image
      del crop_img
    else:
      archivobon = dataclase3["Nombre"][i]
      #spectrogram
      sig, fs = librosa.load(pathflac3+archivobon+'.wav', sr=16000, duration=3, , NFTT=1024)
      Pxx, freqs, bins, im = plt.specgram(sig, Fs=fs)
      im.set_cmap('jet')
      yticks = [100,10000] #yticks line to erase to log3 to 4
      plt.yticks(yticks) #remove this line to log3 to log 4
      plt.axis("off")
      plt.savefig(path3+'bonafide/'+archivobon+'.png', bbox_inches='tight', pad_inches = 0)
      plt.close()
      #cut, resize
      image = cv2.imread(path3+'bonafide/'+archivobon+'.png')
      crop_img = image[30:217, 0:334]
      image = cv2.resize(crop_img, (70, 70))
      cv2.imwrite(path3+'bonafide/'+archivobon+'.png',image)
      del image
      del crop_img
    cont=cont+1
    print('\rConvertidos: '+str(cont), end='')
    
#To convert the spectrograms to a csv file
#The same process is used to convert all spectrograms to csv for use in models, only need to change the paths

def imgtocsv():
  categories = ['bonafide','spoof']
  cont=0
  conttotal=0
  for category in categories:
    path = os.path.join(path3, category)
    label = categories.index(category)
    print('\n'+path)
    for image_name in os.listdir(path):
      image_path = os.path.join(path, image_name)
      image = cv2.imread(image_path)
      image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      try:
        image = np.array(image, dtype=np.float32)
        image = image.flatten()
        image = np.append(label, image)
        image = pd.DataFrame(image)
        image = image.transpose()
        if not os.path.isfile('/csv/spectro/eval.csv'):
          image.to_csv('/csv/spectro/eval.csv', header = None, index = None)
        else:
          image.to_csv('/csv/spectro/eval.csv', mode="a" ,header = None, index = None)      
      except Exception as e:
        pass
      cont=cont+1
      print('\rCargados: '+str(cont), end='')
    conttotal=conttotal+cont
    cont=0
  print('\nImagenes transformadas: '+str(conttotal), end='')
