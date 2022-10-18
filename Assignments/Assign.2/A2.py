import numpy as n
import matplotlib.pyplot as plt


chirp = n.fromfile("chirp.bin",dtype=n.float32)
m = n.fromfile("sonar_meas.bin",dtype=n.float32)


interpulse_period = 10000
velocity = 343.0 # m/s
sample_rate = 44.1e3
N_p = int(n.floor(len(m) / interpulse_period)) # 194


deconvolution_filter = chirp[::-1]
P = n.zeros([N_p,interpulse_period])

for i in range(N_p):
    echo=m[(i * interpulse_period) : (i * interpulse_period + interpulse_period)]
    deconvolved = n.convolve(echo,deconvolution_filter, mode="same")
    P[i,:] = n.abs(deconvolved) ** 2.0


pulse_length = 1000.0 / sample_rate
dist_per_sample = velocity / sample_rate
meas_length = len(m) / sample_rate

con = n.convolve(chirp,chirp[::-1])

plt.plot(con)
plt.xlim([990,1008])
plt.title("Autocorrelation function")
plt.xlabel("Time")
plt.ylabel("$c[n]$")
plt.show()

print("g) Fairly close, but not exactly \delta[n]. There are significant non-zero values only up to +/- 3 samples around the peak of the autocorrelation function. This will affect the range resolution at which the measurement can be made, as ranges +/- 3 samples around a certain range will still be mixed up and seen as artefacts in the measurement. This is a smearing of echo power as a function of range over around 6 samples. This is still better than 1000 ranges being smeared together, which we would have without the deconvolution operation.")

print("h) plotting.")
tot_range = velocity * n.arange(interpulse_period) / sample_rate
t = n.arange(N_p) * interpulse_period / sample_rate
    

plt.pcolormesh(t, tot_range, 10.0 * n.log10(n.transpose(P)), vmin=-30, vmax=30)
plt.xlabel("Time (s)")
plt.ylabel("Total propagation distance (m)")
plt.title("g) Scattered power as a function of time and range")
plt.ylim([0,10])
cb = plt.colorbar()
cb.set_label("Scattered power (dB) arbitrary reference")
plt.show()
