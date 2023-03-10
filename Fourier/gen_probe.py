import numpy as np
import matplotlib.pyplot as plt
import random

class source():
    def __init__(self, f, time, f_num):
        self.inter = f
        self.time = time
        self.num = f_num
        self.f_lists = []
        self.start_time_lists =[] 

    def start_probe(self):
        for l in range(self.num):
            self.start_time_lists.append(random.randint(0, self.inter - 1))
        return self.start_time_lists
        
    def gen_probe(self):
        '''for l in range(self.num):
            self.start_time_lists.append(random.randint(0, self.inter - 1))'''
        self.probe_list = [0] * self.time
        for l in range(self.num):
            f_comp_list = []
            trans_time = self.start_time_lists[l]
            for p in range(self.time):
                if p == trans_time:
                    f_comp_list.append(1)
                    trans_time += self.inter
                else :
                    f_comp_list.append(0)
#                self.f_lists.extend([f_comp_list])
            self.probe_list = [x+y for (x,y) in zip(self.probe_list,f_comp_list)]
            
        return self.probe_list #, self.start_time_lists

if __name__ == "__main__":
    min_time = 5
    time = min_time * 60            # time 秒数
    dt = 0.001          # サンプリング周期 [s]
    f1, f2, f3 = 60, 120, 180   
    t_plot = np.linspace(0, min_time, time)

    f1_num, f2_num, f3_num = 6, 3, 1 
#f1_lists = [[] for i in range(f1_num)]

    p=source(f3, time, f3_num)
    a = p.start_probe()
    p = p.gen_probe()

    print(a,p)
    plt.plot(t_plot, p)
 
    plt.savefig("TEST.png")   # プロットしたグラフをファイルsin.pngに保存する
    plt.show()
'''
list = [0]*time
for i in range(len(p)):
    list = [x+y for (x,y) in zip(list,p[i])]
print(list)
for i in range(f1_num):
    f1_start_time = (random.randint(1, f1))
    count = 0
    for j in range(f1):'''