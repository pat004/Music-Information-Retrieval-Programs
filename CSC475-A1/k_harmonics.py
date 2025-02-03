import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd

# Function to create a single sinusoidal wave
def create_sinusoid(amp, freq, duration, sample_rate):
    t = np.arange(0, duration, 1.0 / sample_rate)
    return amp * np.sin(2 * np.pi * freq * t)

# Function to compute k_harmonics
def k_harmonics(k, amp, freq, duration, sample_rate):
    # Total duration of the output signal
    total_duration = k * duration
    num_samples = int(total_duration * sample_rate)
    signal = np.zeros(num_samples)
    
    # Loop to add each harmonic progressively over the k durations
    for i in range(1, k + 1):
        # In each segment, add harmonics from 1 to i
        for h in range(1, i + 1):
            # Calculate the amplitude for the current harmonic
            harmonic_amp = amp / (1 + h - 1)  # Amplitude decreases as h increases
            # Generate the harmonic for frequency h*f
            harmonic = create_sinusoid(harmonic_amp, freq * h, duration, sample_rate)
            
            # Add the harmonic to the appropriate segment
            signal[(i-1)*int(duration*sample_rate): i*int(duration*sample_rate)] += harmonic
    
    return signal

# Parameters
f0 = 220  # Base frequency (fundamental)
sr = 8000  # Sample rate
amp = 0.5  # Amplitude
duration = 1  # Duration of each segment in seconds
k = 5  # Number of harmonics

# Generate the signal
signal = k_harmonics(k, amp, f0, duration, sr)

# Plot the signal
plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, k * duration, len(signal)), signal)
plt.title("k Harmonics Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# Play the audio (uncomment if using a notebook)
ipd.Audio(signal, rate=sr)
