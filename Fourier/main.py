import numpy as np
import matplotlib.pyplot as plt
import gen_probe
import list_sum
import FFTmodule

min_time = 60
time = min_time * 60            # time 秒数
f = [10, 60, 120, 180 ]  #　発信間隔
fnum = [4, 6, 3, 1] #台数設定"""

cicleminute = 3
cicletime = cicleminute* 60
t_plot = np.linspace(0, cicleminute, cicletime )
probe = []


#probe作成------------------------------------------
for i in range(len(f)):
    source = gen_probe.source(f[i], time, fnum[i])
    start_time = source.start_probe()
    print(start_time)
    print(1/f[i])
    source = source.gen_probe()
    plt.plot(t_plot, source[:cicletime])
    plt.savefig(f"/home/oba/programs/Fourier/main_result/testf{i+1}_2.png")

    plt.show()
    plt.clf()
    probe.extend([source])

probe = list_sum.list_sum(probe)

plt.plot(t_plot, probe[:cicletime])
 
plt.savefig("/home/oba/programs/Fourier/main_result/testsum_probe_2.png")
plt.show()
plt.clf()


#フーリエ変換-------------------------------------------

F = np.fft.fft(probe) # 変換結果
F = np.abs(F)
freq = np.fft.fftfreq(time,d=90) # 周波数
#freq = 1/60 # 周波数


fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(6,6))
ax[0].plot(F.real, label="Real part")
ax[0].legend()
ax[1].plot(F.imag, label="Imaginary part")
ax[1].legend()
ax[2].plot(freq, label="Frequency")
ax[2].legend()
ax[2].set_xlabel("Number of data")
plt.show()
plt.savefig("/home/oba/programs/Fourier/main_result/testsum_probe_fft1_2.png")



Amp = np.abs(F/(time/2)) # 振幅
print(np.sum(Amp))
fig, ax = plt.subplots()
ax.plot(freq[1:int(time/2)], Amp[1:int(time/2)])
ax.set_xlabel("Freqency [Hz]")
ax.set_ylabel("Amplitude")
ax.grid()
plt.show()
plt.savefig("/home/oba/programs/Fourier/main_result/testsum_probe_fft2_2.png")
plt.clf()

'''
N = 3600      # サンプル数
Fs = 1       # サンプリング周波数
Dt = 1 / Fs  # サンプル間の時間差
Fb = Fs / N  # 基本周波数
Ft = 1/180
#fs = np.linspace(0, Fb * time, time, endpoint=False)
fig, ax2 = plt.subplots()
fs = np.linspace(0, Fb * N, N, endpoint=False)
ax2 = fig.add_subplot()
ax2.bar(fs, F, width=Fb * 0.8)
# tick間隔をFtにしたほうが結果がよりはっきりわかるのですが、そうしてしまうと
# 横軸ラベルが重なってしまい視認できなかったのでFtの倍の間隔にしてあります
ax2.xaxis.set_ticks(np.arange(0, 1, Ft * 2))


plt.show()
plt.savefig("/home/oba/programs/Fourier/main_result/testsum_probe_fft3.png")'''
dt = 180
t = np.arange(0, 3600, 1)
x = probe
#dt = 0.01 #This value should be correct as real.
output_FN = "test1.png"

split_t_r = 0.1 #1つの枠で全体のどの割合のデータを分析するか。
overlap = 0.5 #オーバーラップ率
window_F = "hanning" #窓関数選択: hanning, hamming, blackman
y_label = "amplitude"
y_unit = "V"
FFTmodule.FFT_main(t, x, dt, split_t_r, overlap, window_F, output_FN, y_label, y_unit)