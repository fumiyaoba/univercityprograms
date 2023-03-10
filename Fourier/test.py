import graph
import numpy as np
import matplotlib.pyplot as plt
path="/home/oba/programs/Fourier/fourier_result/graph/"
filename="test"
time = graph.array()
Probe_Request = []
for i in range(len(time)):
   Probe_Request.append(time[i, 1]) 
print(Probe_Request)
x_plot = np.linspace(0, 1, 46)
plt.plot(x_plot,Probe_Request)
plt.show()
plt.savefig( path + filename +"1.png")

F = np.fft.fft(Probe_Request) # 変換結果
freq = np.fft.fftfreq(len(Probe_Request)) # 周波数

fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6,6))
ax[0].plot(F.real, label="Real part")
ax[0].legend()
ax[1].plot(F.imag, label="Imaginary part")
ax[1].legend()
ax[2].plot(freq, label="Frequency")
ax[2].legend()
ax[2].set_xlabel("Number of data")
plt.show()
plt.savefig( path + filename +"2.png")
N=len(Probe_Request)
Amp = np.abs(F/(len(Probe_Request)/2)) # 振幅

fig, ax = plt.subplots()
ax.plot(freq[1:int(N/2)], Amp[1:int(N/2)])
ax.set_xlabel("Freqency [Hz]")
ax.set_ylabel("Amplitude")
ax.grid()
plt.show()
plt.savefig( path + filename +"3.png")