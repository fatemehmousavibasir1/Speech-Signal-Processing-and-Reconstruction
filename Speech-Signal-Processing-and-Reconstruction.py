import wave
import numpy as np
import librosa

import numpy as np

def remove_silence(input_path, output_path):
    with wave.open(input_path, 'rb') as wav_file:
        params = wav_file.getparams()
        num_frames = params.nframes
        frame_rate = params.framerate
        num_channels = params.nchannels
        sample_width = params.sampwidth

        raw_data = wav_file.readframes(num_frames)

        audio_data = np.frombuffer(raw_data, dtype=np.int16)

        threshold = 1000 
        non_silent_indices = np.where(np.abs(audio_data) > threshold)[0]
        start_index = non_silent_indices[0]
        end_index = non_silent_indices[-1]
        trimmed_audio = audio_data[start_index:end_index]
        with wave.open(output_path, 'wb') as trimmed_wav_file:
            trimmed_wav_file.setparams((num_channels, sample_width, frame_rate, len(trimmed_audio), params.comptype, params.compname))
            trimmed_wav_file.writeframes(trimmed_audio.tobytes())

input_audio_path = "C:/Users/Asus/Desktop/class.wav"
output_audio_path = "C:/Users/Asus/Desktop/newclass.wav"
remove_silence(input_audio_path, output_audio_path)


import librosa
import numpy as np
import matplotlib.pyplot as plt

audio_file = "C:/Users/Asus/Desktop/newclass.wav"
y, sr = librosa.load(audio_file, sr=None)
frame_length = 300 
hop_length = 100  

frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)

for i in range(min(4, frames.shape[1])):
    frame_waveform = frames[:, i]
    time = np.arange(len(frame_waveform)) * hop_length / sr

    plt.figure()
    plt.plot(time, frame_waveform)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform for Frame {} (Before Hamming Window)'.format(i + 1))
    plt.show()
frames = frames.copy()


for i in range(frames.shape[1]):
    frames[:, i] *= np.hamming(frame_length)

for i in range(min(4, frames.shape[1])):
    frame_waveform = frames[:, i]
    time = np.arange(len(frame_waveform)) * hop_length / sr

    plt.figure()
    plt.plot(time, frame_waveform)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform for Frame {}'.format(i + 1))
    plt.show()







import numpy as np
import librosa

def linear_prediction_coefficients(frames, p):
    A = np.zeros((p, frames.shape[1])) 
    
    for i in range(frames.shape[1]):
        # Select the current frame
        frame = frames[:, i]
        
        # Construct the matrix X for linear regression
        X = np.zeros((len(frame) - p, p))
        for j in range(p):
            X[:, j] = frame[p-j-1:len(frame)-j-1]
        
        # Calculate linear regression coefficients
        A[:, i] = np.linalg.lstsq(X, frame[p:], rcond=None)[0] 
    
    return A
p = 20
A = linear_prediction_coefficients(frames, p)
print("Shape of matrix A:", A.shape)



def calculate_gain(frames):
    gain = np.zeros(frames.shape[1])
    for i in range(frames.shape[1]):
        gain[i] = np.sqrt(np.sum(frames[:, i] ** 2))

    return gain

G = calculate_gain(frames)
print("Gain of each frame:")
print(G)





import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import librosa
def compute_frame_energy(frames):
    frame_energies = []
    for frame in frames:
        energy = np.sum(np.square(frame))
        frame_energies.append(energy)
    return frame_energies

def detect_voiced_frames(frame_energies, threshold):
    voiced_frames = [energy > threshold for energy in frame_energies]
    return voiced_frames

filename = "C:/Users/Asus/Desktop/newclass.wav"
signal, sr = librosa.load(filename, sr=None)
frame_length = 300
hop_length = 100
frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length).T
frame_energies = compute_frame_energy(frames)
threshold = 30 
voiced_frames = detect_voiced_frames(frame_energies, threshold)

def plot_frame_energy(frame_energies):
    plt.plot(frame_energies)
    plt.xlabel('Frame Index')
    plt.ylabel('Energy')
    plt.title('Energy of Frames')
    plt.show()


plot_frame_energy(frame_energies)
voiced_frames_indicator = np.ones(len(voiced_frames), dtype=int)
voiced_frames_indicator[~np.array(voiced_frames)] = 0
for i, frame_voiced in enumerate(voiced_frames_indicator):
    if frame_voiced:
        print(f"Frame {i}: voiced")
    else:
        print(f"Frame {i}: unvoiced")
print(voiced_frames_indicator)



def hps(data, fs):
    corr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
    first_min = np.argmin(corr[len(data):]) + len(data)
    f0 = fs / first_min

    return f0

def hps_frames(data, fs, frame_length, hop_length):
    f0_frames = np.zeros(int(np.ceil((len(data) - frame_length) / hop_length) + 1))
    for i in range(f0_frames.shape[0]):
        frame = data[i * hop_length:i * hop_length + frame_length]
        f0_frames[i] = hps(frame, fs)

    return f0_frames


frame_length = 300
hop_length = 100

f0_frames = hps_frames(frames, 16000, frame_length, hop_length)

print("Fundamental frequency of each frame:", f0_frames)
print("frame shape:", f0_frames.shape)
for i in range(541):
    if voiced_frames_indicator[i]==0:
        f0_frames[i]=0
f0_frames[541]=0
print("Fundamental frequency of each frame:", f0_frames)
print("frame shape:", f0_frames.shape)

filename = "C:/Users/Asus/Desktop/newclass.wav"
data, fs = librosa.load(filename, sr=None) 





import numpy as np
import soundfile as sf
from scipy.signal import lfilter

def reconstruct_signal(A, G, f0_frames, frame_length, hop_length, fs):
    reconstructed_signal = np.zeros((len(f0_frames) - 1) * hop_length + frame_length)
    n_samples = 0
    for i in range(len(A)):
        if f0_frames[i] == 0:
            frame_signal = np.random.normal(size=frame_length) * np.sqrt(G[i])
        else:
            AR_coefficients = np.hstack([1, -A[i]])
            frame_signal = lfilter([1], AR_coefficients, np.random.normal(size=frame_length)) * np.sqrt(G[i])
            # Upsample the frame signal to match the sample rate
            frame_signal = np.interp(np.arange(0, frame_length, 1 / fs), np.arange(0, frame_length), frame_signal)

        # Place the frame signal into the reconstructed signal
        start_index = i * hop_length
        end_index = start_index + frame_length
        reconstructed_signal[start_index:end_index] += frame_signal
        n_samples = end_index

    return reconstructed_signal[:n_samples]

reconstructed_signal = reconstruct_signal(A, G, f0_frames, frame_length, hop_length, fs)

sf.write("C:/Users/Asus/Desktop/reconstructed_signal.wav", reconstructed_signal, fs)










#second part



import librosa
import numpy as np
import matplotlib.pyplot as plt

audio_file = "C:/Users/Asus/Desktop/newclass.wav"
y, sr = librosa.load(audio_file, sr=None)
frame_length = 300 
hop_length = 0

frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)

for i in range(min(4, frames.shape[1])):
    frame_waveform = frames[:, i]
    time = np.arange(len(frame_waveform)) * hop_length / sr

    plt.figure()
    plt.plot(time, frame_waveform)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform for Frame {} (Before Hamming Window)'.format(i + 1))
    plt.show()
frames = frames.copy()



for i in range(min(4, frames.shape[1])):
    frame_waveform = frames[:, i]
    time = np.arange(len(frame_waveform)) * hop_length / sr

    plt.figure()
    plt.plot(time, frame_waveform)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform for Frame {}'.format(i + 1))
    plt.show()







import numpy as np
import librosa

def linear_prediction_coefficients(frames, p):
    A = np.zeros((p, frames.shape[1])) 
    
    for i in range(frames.shape[1]):
        # Select the current frame
        frame = frames[:, i]
        
        # Construct the matrix X for linear regression
        X = np.zeros((len(frame) - p, p))
        for j in range(p):
            X[:, j] = frame[p-j-1:len(frame)-j-1]
        
        # Calculate linear regression coefficients
        A[:, i] = np.linalg.lstsq(X, frame[p:], rcond=None)[0] 
    
    return A
p = 20
A = linear_prediction_coefficients(frames, p)
print("Shape of matrix A:", A.shape)



def calculate_gain(frames):
    gain = np.zeros(frames.shape[1])
    for i in range(frames.shape[1]):
        gain[i] = np.sqrt(np.sum(frames[:, i] ** 2))

    return gain

G = calculate_gain(frames)
print("Gain of each frame:")
print(G)





import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import librosa
def compute_frame_energy(frames):
    frame_energies = []
    for frame in frames:
        energy = np.sum(np.square(frame))
        frame_energies.append(energy)
    return frame_energies

def detect_voiced_frames(frame_energies, threshold):
    voiced_frames = [energy > threshold for energy in frame_energies]
    return voiced_frames

filename = "C:/Users/Asus/Desktop/newclass.wav"
signal, sr = librosa.load(filename, sr=None)
frame_length = 300
hop_length = 0
frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length).T
frame_energies = compute_frame_energy(frames)
threshold = 30 
voiced_frames = detect_voiced_frames(frame_energies, threshold)

def plot_frame_energy(frame_energies):
    plt.plot(frame_energies)
    plt.xlabel('Frame Index')
    plt.ylabel('Energy')
    plt.title('Energy of Frames')
    plt.show()


plot_frame_energy(frame_energies)
voiced_frames_indicator = np.ones(len(voiced_frames), dtype=int)
voiced_frames_indicator[~np.array(voiced_frames)] = 0
for i, frame_voiced in enumerate(voiced_frames_indicator):
    if frame_voiced:
        print(f"Frame {i}: voiced")
    else:
        print(f"Frame {i}: unvoiced")
print(voiced_frames_indicator)



def hps(data, fs):
    corr = np.correlate(data - np.mean(data), data - np.mean(data), mode='full')
    first_min = np.argmin(corr[len(data):]) + len(data)
    f0 = fs / first_min

    return f0

def hps_frames(data, fs, frame_length, hop_length):
    f0_frames = np.zeros(int(np.ceil((len(data) - frame_length) / hop_length) + 1))
    for i in range(f0_frames.shape[0]):
        frame = data[i * hop_length:i * hop_length + frame_length]
        f0_frames[i] = hps(frame, fs)

    return f0_frames


frame_length = 300
hop_length = 0

f0_frames = hps_frames(frames, 16000, frame_length, hop_length)

print("Fundamental frequency of each frame:", f0_frames)
print("frame shape:", f0_frames.shape)
for i in range(541):
    if voiced_frames_indicator[i]==0:
        f0_frames[i]=0
f0_frames[541]=0
print("Fundamental frequency of each frame:", f0_frames)
print("frame shape:", f0_frames.shape)

filename = "C:/Users/Asus/Desktop/newclass.wav"
data, fs = librosa.load(filename, sr=None) 





import numpy as np
import soundfile as sf
from scipy.signal import lfilter

def reconstruct_signal(A, G, f0_frames, frame_length, hop_length, fs):
    reconstructed_signal = np.zeros((len(f0_frames) - 1) * hop_length + frame_length)
    n_samples = 0
    for i in range(len(A)):
        if f0_frames[i] == 0:
            frame_signal = np.random.normal(size=frame_length) * np.sqrt(G[i])
        else:
            AR_coefficients = np.hstack([1, -A[i]])
            frame_signal = lfilter([1], AR_coefficients, np.random.normal(size=frame_length)) * np.sqrt(G[i])
            # Upsample the frame signal to match the sample rate
            frame_signal = np.interp(np.arange(0, frame_length, 1 / fs), np.arange(0, frame_length), frame_signal)

        # Place the frame signal into the reconstructed signal
        start_index = i * hop_length
        end_index = start_index + frame_length
        reconstructed_signal[start_index:end_index] += frame_signal
        n_samples = end_index

    return reconstructed_signal[:n_samples]

reconstructed_signal = reconstruct_signal(A, G, f0_frames, frame_length, hop_length, fs)

sf.write("C:/Users/Asus/Desktop/newoutput.wav", reconstructed_signal, fs)
