import h5py
import matplotlib.pyplot as plt
from os.path import dirname, join
import numpy as np
import scipy.signal as s

# reading in data
path = dirname(__file__)
def read_data(fname):
    h=h5py.File(fname,"r")
    detector_name=h["meta/Detector"][()]
    start_time = h["meta/UTCstart"][()]
    strain=h["strain/Strain"][()]
    print("Reading %s data starting at %s"%(str(detector_name),str(start_time)))
    h.close()
    return(detector_name,start_time,strain)
# read hanford measurement
h1_name,h1_start_time,h1_strain=read_data(join(path, "H-H1_LOSC_4_V2-1126259446-32.hdf5"))
# read livingston measurement
l1_name,l1_start_time,l1_strain=read_data(join(path, "L-L1_LOSC_4_V2-1126259446-32.hdf5"))
    

# global variables used in multiple functions
fs = 4096 # sampling frequency
samples_per_signal = len(h1_strain)
sec_per_signal = len(h1_strain)/fs
sample_spacing = 1/fs

# steps
n = np.linspace(0, 4096 - 1, fs)
k = np.linspace(0, 4096 - 1, fs)

# Problem 3
f_k = np.fft.fftfreq(len(k), 1/fs)
f_shifted = np.fft.fftshift(f_k)
window = s.hann(samples_per_signal)

# time domain
t = np.arange(0, 32, 1/fs)

# time delay
time_delay = 8.6e-4
time_shift = t + time_delay

# Functions in problems
def windowed_strain(strain):
    return np.fft.rfft(s.hann(len(strain)) * strain)

def whitening_filter(strain):
    return 1/np.abs(windowed_strain(strain))

def whitening(strain):
    return np.fft.irfft(whitening_filter(strain) * windowed_strain(strain))

def filtering(strain):
    return np.convolve(np.repeat(1.0/8, 8), whitening(strain), mode='same')


# Problems
def prob1():
    # Samples per signal
    print("Amount of samples: ", samples_per_signal)
    # seconds of signal 
    print("Seconds of signal: ", sec_per_signal)
    # Sample spacing
    print("Sample spacing: ", sample_spacing)

def prob2():
    # Plot of the H1 and L1 strain data
    time = np.linspace(0, 32, samples_per_signal)
    plt.plot(time, h1_strain)
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.title("H1 Strain")
    plt.show()

    plt.plot(time, l1_strain)
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.title("L1 Strain")
    plt.show()

    # max, min, and average of the H1 data
    maximum_h = np.max(h1_strain)
    minimum_h = np.min(h1_strain)
    mean_h = np.mean(h1_strain)

    # max, min, and average of the L1 data
    maximum_l = np.max(l1_strain)
    minimum_l = np.min(l1_strain)
    mean_l = np.mean(l1_strain)

    print("H1: Max: ", maximum_h, " Min: ", minimum_h, " Mean: ", mean_h)
    print("L1: Max: ", maximum_l, " Min: ", minimum_l, " Mean: ", mean_l)

    return 

def prob3():
    print(f_shifted)
    # values
    f = 31.5
    # sing signal, x[n]
    sin_sig = np.sin(2*np.pi*f*n / fs)
    # Hann window, w[n]
    window = s.hann(len(n))

    # b plot
    plt.plot(n, sin_sig)
    plt.plot(n, sin_sig * window)
    plt.xlabel("n")
    plt.legend(["x[n]", "w[n] * x[n]"])
    plt.show()

    #c
    
    # d
    print("31.5 Hz: ", np.argmin(np.abs(f_k - 31.5)))
    print("-31.5 Hz: ", np.argmin(np.abs(f_k + 31.5)))

    # e
    x_hat = np.fft.fftshift(np.fft.fft(sin_sig))
    x_hat_windowed = np.fft.fftshift(np.fft.fft(sin_sig * window))
    
    plt.plot(f_shifted, 10*np.log10(np.abs(x_hat)**2))
    plt.plot(f_shifted, 10*np.log10(np.abs(x_hat_windowed)**2))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.legend(["x_hat", "x_hat_windowed"])
    plt.show()

    # power spectrum
    power = np.abs(x_hat)**2
    power_windowed = np.abs(x_hat_windowed)**2
    print(power, power_windowed)
    # f
    plt.plot(f_shifted, 10*np.log10(np.abs(x_hat)**2))
    plt.plot(f_shifted, 10*np.log10(np.abs(x_hat_windowed)**2))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.legend(["x_hat", "x_hat_windowed"])
    plt.axvline(x=31.5, color='r')
    plt.axvline(x=-31.5, color='r')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.show()

    # g


    # h
    xh = 10*np.log10(np.abs(x_hat)**2)
    xhw = 10*np.log10(np.abs(x_hat_windowed)**2)
    # db difference
    print("Difference: ", np.min(xh) - np.min(xhw))

    return

def prob4():
    h1_strain_windowed = h1_strain * window
    l1_strain_windowed = l1_strain * window
    fft_h1 = np.fft.fftshift(np.fft.fft(h1_strain_windowed))
    fft_l1 = np.fft.fftshift(np.fft.fft(l1_strain_windowed))

    plt.plot(np.fft.fftshift(np.fft.fftfreq(len(h1_strain), 1/fs)), 10*np.log10(np.abs(fft_h1)**2))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.legend("H1")
    plt.xlim(0, fs/2)
    plt.show()

    plt.plot(np.fft.fftshift(np.fft.fftfreq(len(l1_strain), 1/fs)), 10*np.log10(np.abs(fft_l1)**2))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.legend("L1")
    plt.xlim(0, fs/2)
    plt.show()

    return

def prob5():

    y_h = whitening(h1_strain)
    y_l = whitening(l1_strain)

    # H1
    plt.plot(t,y_h)
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.show()

    # L1
    plt.plot(t,y_l)
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.show()

    # zoomed in to the gravitational wave signal location
    plt.plot(t,y_h)
    plt.plot(t,y_l)
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.xlim(16.2, 16.5)
    plt.show()
    return

def prob6():
    # low pass filter
    f = np.linspace(-np.pi, np.pi, 1000)

    # convert to hertz
    f_hz = f * fs / (2*np.pi)

    # frequency response
    fr = np.zeros(len(f))

    L = 8
    # updating frequency response
    for i in range(0, L):
        fr = fr + (1/L) * np.exp(-1j * f * i)
        
    plt.plot(f_hz, 10*np.log10(np.abs(fr)**2))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.axvline(x=300, color='k', linestyle='--') 
    plt.axhline(y=-6, color='k', linestyle='--')
    plt.show()

    #6d, e and f
    h1_filtered = filtering(h1_strain)
    l1_filtered = filtering(l1_strain)
    
    
    plt.plot(t, h1_filtered)
    plt.plot(t, l1_filtered)
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.xlim(16.1, 16.6)
    plt.show()
    return 

def prob7():
    
    h1_filtered = filtering(h1_strain)
    l1_filtered = filtering(l1_strain)

    time_delay = 4.25e-3
    time_shift = t + time_delay
    # plotting the magnitudes
    plt.plot(t, np.abs(h1_filtered))
    plt.plot(time_shift , np.abs(l1_filtered))
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.legend(["H1", "L1"])
    plt.xlim(16.1, 16.6)
    plt.show()

    return

def prob8():
    h1_filtered = filtering(h1_strain)
    l1_filtered = filtering(l1_strain)

    h_hz, h_time, h1_f = s.spectrogram(h1_filtered, fs=fs, window='hann', nperseg=400, noverlap=380, nfft=4096)
    l_hz, l_time, l1_f = s.spectrogram(l1_filtered, fs=fs, window='hann', nperseg=400, noverlap=380, nfft=4096)

    plt.pcolormesh(h_time, h_hz, 10*np.log10(np.abs(h1_f)**2), cmap='viridis', shading='auto', vmin=-200)
    plt.colorbar( label='Power (dB)')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.xlim(15.5, 17)
    plt.ylim(0, 400)
    plt.show()
    
    # L1
    plt.pcolormesh(l_time, l_hz, 10*np.log10(np.abs(l1_f)**2), cmap='viridis', shading='auto', vmin=-200)
    plt.colorbar( label='Power (dB)')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.xlim(15.5, 17)
    plt.ylim(0, 400)
    plt.show()
    return

# Running the functions
prob1()
prob2()
prob3()
prob4()
prob5()
prob6()
prob7()
prob8()
