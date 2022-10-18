import numpy as np
import matplotlib.pyplot as plt

sample_rate = 10000.0

t = np.arange(4.0 * sample_rate) / sample_rate

sig1 = np.zeros(len(t), dtype=np.complex64)
sig2 = np.zeros(len(t), dtype=np.complex64)

N = 50
T = 0.55

# the delay
tau = 0.2

c_k = 1/T

# signal without delay
for k in range(-N, N + 1):
    sig1 += c_k * np.exp(1j * (2 * np.pi)/(T) * k * (t))

# signal with delay
for k in range(-N, N + 1):
    c_k_marked = c_k * np.exp(-1j * (2 * np.pi * tau)/(T) * k)
    sig2 += c_k_marked * np.exp(1j * (2 * np.pi)/(T) * k * (t))

# plot of signal without delay
plt.plot(t, sig1.real, color="purple")
plt.plot(t, sig1.imag)

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(['Real', 'Imaginary'], loc='upper right')
plt.show()

# plot of signal with delay
plt.plot(t, sig2.real, color="red")
plt.plot(t, sig2.imag)

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(['Real', 'Imaginary'], loc='upper right')
plt.show()

# figure to show the difference between the two signals, and that the delay worked
plt.plot(t, sig1.real, color="purple")
plt.plot(t, sig1.imag)
plt.plot(t, sig2.real, color="red")
plt.plot(t, sig2.imag)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(['Real', 'Imaginary'], loc='upper right')
plt.show()
