import numpy as np 
import IPython.display as ipd
import matplotlib.pyplot as plt
from IPython.display import display  # Import display explicitly
    
# complete the function k_harmonics  
def harmonics_arpeggiator(r, k, freq, duration, sample_rate): 
    t = np.arange(0, duration, 1.0 / sample_rate) # time formula
    note_size = t.shape[0]
    audio = np.zeros(r * k * note_size)
    for r in range(r):
        for i in range(k):
            freqi = (i+1)*freq
            haudio = np.sin(2 * np.pi * freqi * t[0:note_size])
            start = (r * k + i)
            audio[start * note_size : (start + 1) * note_size] = haudio
    return audio



# use the code in a notebook cell to plot/listen to the resulting 
# signal of the k_harmonics function 
f0 = 220
sr = 44100 
duration1 = 0.125
duration2 = 0.025
r = 4 
k = 12 

# Some sound
signal1 = harmonics_arpeggiator(r, k, f0, duration1, sr)

# Pac-Man sound
signal2 = harmonics_arpeggiator(r, k, f0, duration2, sr)

# Use IPython.display.display() to show both audio players
display(ipd.Audio(signal1, rate=sr))
display(ipd.Audio(signal2, rate=sr))

