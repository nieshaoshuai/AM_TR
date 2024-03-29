import torchaudio
import os
import numpy as np
import scipy.io as sio
import scipy.io.wavfile as wav
from scipy import signal

def FileSize(audio_path):
    return os.path.getsize(audio_path)

def load_audio(path, sample_rate = 16000):
    try:
        if os.path.exists(path) and FileSize(path) > 24000:
            rate, sound = wav.read(path)
            if rate != sample_rate:
                sound = signal.resample(sound, int(sound.shape[0] * sample_rate / rate))
        else:
            return None
        ##sound = sound / 65536.0
        return np.float32(sound)
    except:
        print('load_audio %s failed' % (path))
        return None

def load_mat(filename, key='feat'):
    if os.path.exists(filename):
        key = sio.whosmat(filename)[0][0]
        ##print('key = %s' % key)
        data = sio.loadmat(filename)
        if key in data:
            mat = data[key]
        else:
            return None

        if mat.shape[0] > 1:
            mat = mat.reshape(mat.shape[0])
        elif mat.shape[1] > 1:
            mat = mat.reshape(mat.shape[1])
        if np.isnan(mat).sum() > 0:
            return None
        return mat
    else:
        print('load_mat {} failed'.format(filename))
        return None
        
def check_audio(path, file_size = 30000):
    try:
        if os.path.exists(path) and FileSize(path) > file_size:
            rate, sound = wav.read(path)
            return True
        else:
            return False
    except:
        return False  