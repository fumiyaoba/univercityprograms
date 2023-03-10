import numpy as np
import matplotlib.pyplot as plt
"""
arr1 = (np.random.randint(1, 10, 6)).tolist()
arr1.extend(np.random.randint(11, 20, 9).tolist())
arr1.extend(np.random.randint(21, 30, 7).tolist())
print(arr1)
plt.hist(arr1,bins=30)
plt.savefig("test1.png")"""
"""
import numpy as np

np.random.seed(100)
dir_path = '/home/oba/programs/bayes/dpmm_result/'
path_name = 'test2'
#---------------------------------------------------------------
N = 1000
mus_sv = np.array([10, 20, 30])
taus_sv = np.array([3., 3., 3.])
pi = np.array([0.6, 0.3, 0.1])
K_true = len(mus_sv)

ss = np.random.choice(K_true, size=1000, replace=True, p=pi)
x1 = np.random.randn(N) * 1.0/np.sqrt(taus_sv[ss]) + mus_sv[ss]

print(len(x1))"""
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
N = 22
x = pd.Series(np.around(np.random.randint(0.0 , 30.0 , N)))
nbins = 30
plt.hist(x , nbins , density = True)
plt.savefig("test2.png")
plt.figure()
print(x)
bins = np.linspace(0,30,31)
print(bins)
freq = x.value_counts(bins=bins,sort=False)
print(freq)

rel_freq = freq / x.count()
print(rel_freq)
x1=rel_freq.values.tolist()
print(x1)
y= [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
plt.bar( y,x1)
plt.savefig("test3.png")
plt.figure()"""
a = np.random.normal(
    loc   = 1,      # 平均
    scale = 0.5,      # 標準偏差
    size  = 30,# 出力配列のサイズ(タプルも可)
)
print(a)