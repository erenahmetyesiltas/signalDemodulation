import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftshift
from scipy.io.wavfile import write

# === Step 1: Read data and time vector ===
data = np.loadtxt('70.txt')
signal = np.array(data)
fs = 96000  # Sampling frequency
t = np.arange(len(signal)) / fs

# === Step 2: FFT of raw signal ===
Signal_f = fftshift(fft(signal))
frqs = fftshift(np.fft.fftfreq(len(signal), 1 / fs))

plt.figure(figsize=(10, 4))
plt.plot(frqs, np.abs(Signal_f))
plt.title("Fourier Transform (FT) of Input Signal")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid()
plt.tight_layout()
plt.show()

# === Step 3: Peak Detection ===
peak_indices, _ = find_peaks(np.abs(Signal_f[len(signal)//2:]), height=160)
peak_frqs = frqs[len(signal)//2:][peak_indices]
print("Tespit edilen taşıyıcılar (Hz):", peak_frqs)

# === Step 4: Modulation Frequencies ===
mod_freqs = [12000, 36000]
print(f"\nSelected modulation frequencies for demodulation: {mod_freqs}")

# === Step 5: Bandpass Filter ===
def bandpass_filter(signal, fs, freq_low, freq_high):
    nyq = fs / 2
    b, a = butter(4, [freq_low / nyq, freq_high / nyq], btype='band')
    return filtfilt(b, a, signal)

# === Step 6: Envelope Demodulation ===
def absolute_demodulation(signal, fs):
    rectified = np.abs(signal)
    b, a = butter(4, 3000 / (fs / 2), btype='low')
    envelope = filtfilt(b, a, rectified)
    return envelope

# === Step 7: Cosine Demodulation ===
def cosine_demodulation(signal, fs, freq):
    t = np.arange(len(signal)) / fs
    demodulated = signal * np.cos(2 * np.pi * freq * t)
    b, a = butter(4, 3000 / (fs / 2), btype='low')
    baseband = filtfilt(b, a, demodulated)
    return baseband

# === Step 8: Plot Time and Frequency Domain ===
def plotTimeFreqGraph(signal, fs, title=''):
    N = len(signal)
    t = np.arange(N) / fs
    freq = np.linspace(-fs / 2, fs / 2, N)
    spectrum = fftshift(fft(signal))
    signal_pos = np.maximum(signal, 0)

    plt.figure(figsize=(14, 5))

    # Time domain (only positive)
    plt.subplot(1, 2, 1)
    plt.plot(t, signal_pos)
    plt.title(f'{title} (Time Domain)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid()

    # Frequency domain
    plt.subplot(1, 2, 2)
    plt.plot(freq, np.abs(spectrum))
    plt.title(f'{title} (Frequency Domain)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('|X(f)|')
    plt.grid()

    plt.tight_layout()
    plt.show()

# === Step 9: Process Each Carrier ===
for freq in mod_freqs:
    print(f"\nProcessing modulation frequency: {freq:.1f} Hz")

    # Bandpass filter
    filtered = bandpass_filter(signal, fs, freq - 4000, freq + 4000)
    plotTimeFreqGraph(filtered, fs, f"Bandpass Filtered around {freq:.1f} Hz")

    # Absolute (envelope) demodulation
    abs_demod = absolute_demodulation(filtered, fs)
    plotTimeFreqGraph(abs_demod, fs, f'Absolute Demodulation - {freq:.1f} Hz')

    # Cosine demodulation
    cos_demod = cosine_demodulation(filtered, fs, freq)
    cos_demod = np.maximum(cos_demod, 0)  # keep only positive
    plotTimeFreqGraph(cos_demod, fs, f'Cosine Demodulation - {freq:.1f} Hz')

    # Save to WAV
    print("Saving WAV files...")
    write(f"absdemod_{int(freq)}Hz.wav", fs, (abs_demod / np.max(abs_demod) * 32767).astype(np.int16))
    write(f"cosdemod_{int(freq)}Hz.wav", fs, (cos_demod / np.max(cos_demod) * 32767).astype(np.int16))