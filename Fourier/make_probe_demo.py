import numpy as np
import matplotlib.pyplot as plt

N = 1024            # サンプル数
dt = 0.001          # サンプリング周期 [s]
f1, f2, f3 = 60, 120, 180   # 周波数 [Hz]

t = np.arange(0, N*dt, dt) # 時間 [s]
x = np.sin(2*np.pi*f1*t) + np.sin(2*np.pi*f2*t) + np.sin(2*np.pi*f3*t) + 3 # 信号

fig, ax = plt.subplots()
ax.plot(t, x)
# ax.set_xlim(0, 0.1)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Signal")
ax.grid()
plt.show()
plt.savefig('/home/oba/programs/Fourier/Probe_demo/test2_1.png')


F = np.fft.fft(x) # 変換結果
freq = np.fft.fftfreq(N, d=dt) # 周波数

fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6,6))
ax[0].plot(F.real, label="Real part")
ax[0].legend()
ax[1].plot(F.imag, label="Imaginary part")
ax[1].legend()
ax[2].plot(freq, label="Frequency")
ax[2].legend()
ax[2].set_xlabel("Number of data")
plt.show()
plt.savefig('/home/oba/programs/Fourier/Probe_demo/test2_2.png')

Amp = np.abs(F/(N/2)) # 振幅

fig, ax = plt.subplots()
ax.plot(freq[1:int(N/2)], Amp[1:int(N/2)])
ax.set_xlabel("Freqency [Hz]")
ax.set_ylabel("Amplitude")
ax.grid()
plt.show()
plt.savefig('/home/oba/programs/Fourier/Probe_demo/test2_3.png')
