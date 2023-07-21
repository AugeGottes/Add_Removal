import numpy as np
import pandas as pd
import os
import librosa
import scipy
import moviepy
import moviepy.editor as mp
import IPython.display as ipd
from pathlib import Path
from tqdm import tqdm, tqdm_pandas
tqdm.pandas()
from scipy.stats import skew
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pickle
def get_mfcc(name, path):
    SAMPLE_RATE = 44100
    data, sr = librosa.core.load(path + name, sr=SAMPLE_RATE)
    try:
        ft1 = librosa.feature.mfcc(y=data, sr=SAMPLE_RATE, n_mfcc=13)
        ft2 = librosa.feature.zero_crossing_rate(y=data)[0]
        ft3 = librosa.feature.spectral_rolloff(y=data)[0]
        ft4 = librosa.feature.spectral_centroid(y=data)[0]
        ft5 = librosa.feature.spectral_contrast(y=data)[0]
        ft6 = librosa.feature.spectral_bandwidth(y=data)[0]
        ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), np.max(ft1, axis=1), np.median(ft1, axis=1), np.min(ft1, axis=1)))
        ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
        ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
        ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
        ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), np.max(ft5), np.median(ft5), np.min(ft5)))
        ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), np.max(ft6), np.median(ft6), np.max(ft6)))
        return pd.Series(np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc)))
    except Exception as e:
        print(f"Error: {str(e)}")
        return pd.Series([0] * 101)

