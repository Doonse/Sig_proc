import numpy as np
import matplotlib.pyplot as plt

chirp = np.fromfile("Assignments/Assign.2/chirp.bin",dtype=np.float32)
m = np.fromfile("Assignments/Assign.2/sonar_meas.bin",dtype=np.float32)

interpulse_period = 10000 
velocity = 343.0 # m/s
sample_rate = 44.1e3 # Hz
N_p = int(np.floor(len(m) / interpulse_period)) # 194


deconvolution_filter = chirp[::-1]
P = np.zeros([N_p,interpulse_period])

for i in range(N_p):
    echo=m[(i * interpulse_period) : (i * interpulse_period + interpulse_period)]
    deconvolved = np.convolve(echo,deconvolution_filter, mode="same")
    P[i,:] = np.abs(deconvolved) ** 2.0

# tasks
pulse_length = 1000.0 / sample_rate
dist_per_sample = velocity / sample_rate
meas_length = len(m) / sample_rate

con = np.convolve(chirp,chirp[::-1])

plt.plot(con)
plt.xlim([990,1008])
plt.title("Autocorrelation function")
plt.xlabel("Time")
plt.ylabel("$c[n]$")
plt.show()

tot_range = velocity * np.arange(interpulse_period) / sample_rate
# sorted values from 0 to length of N_p  *  interpulse_period / sample_rate
t = np.arange(N_p) * interpulse_period / sample_rate

# plot for h
# x axis is time, y axis is range, z axis is power (dB) which is illustrated in the colorbar
plt.pcolormesh(t, tot_range, 10.0 * np.log10(np.transpose(P)), vmin=-30, vmax=30)
plt.xlabel("Time (s)")
plt.ylabel("Propagation distance (m)")
plt.title("Scattered power as function of time and range")
plt.ylim([0,7])
cb = plt.colorbar()
cb.set_label("dB")
plt.show()
