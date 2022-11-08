from tarfile import XHDTYPE
from tkinter import N
import h5py
import matplotlib.pyplot as plt
from os.path import dirname, join
import numpy as np
import scipy.signal as s

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
    
# global variables
fs = 4096 # sampling frequency
samples_per_signal = len(h1_strain)
sec_per_signal = len(h1_strain)/fs
sample_spacing = 1/fs
n = np.linspace(0, 4096 - 1, fs)
k = np.linspace(0, 4096 - 1, fs)
f_k = np.fft.fftfreq(len(k), 1/fs)
f_shifted = np.fft.fftshift(f_k)
window = s.hann(samples_per_signal)
window_w = np.fft.fft(s.hann(len(h1_strain)) * h1_strain)
t = np.arange(0, 32, 1/fs)

def plot_spectrum(strain,fs):
    x_hat= np.fft.fftshift(np.fft.fft(s.hann(len(strain))*strain))
    plt.plot(np.fft.fftshift(np.fft.fftfreq(len(strain),1.0/fs)),10.0*n.log10(np.abs(x_hat)**2.0))
    plt.show()

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
    plt.plot(time, l1_strain)
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
    # b
    xhat_h = h1_strain * window_w
    xhat_l = l1_strain * window_w

    hhat_h = 1 / np.abs(xhat_h)
    hhat_l = 1 / np.abs(xhat_l)

    yhat_h = hhat_h * xhat_h
    yhat_l = hhat_l * xhat_l

    y_h = np.fft.ifft(yhat_h)
    y_l = np.fft.ifft(yhat_l)

    plt.plot(t,y_h)
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.show()

    plt.plot(t,y_l)
    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.show()

    # c


    return

prob5()


