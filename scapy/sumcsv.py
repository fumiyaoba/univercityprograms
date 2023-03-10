import pandas as pd
import os
import datetime
import glob

R = 14
for i in range(R):
    point = str(i + 1)
#    files = glob.glob("/home/oba/csv_dataset/1112/1112traindata/1112R"+ point +"*.csv")
    files = glob.glob("/home/oba/csv_dataset/1112/1112traindata/1112R*.csv")
    list = []
    for file in files:
        try:
            list.append(pd.read_csv(file))
        except :
            print("ERROR: {} is empty".format(file))

#    df = pd.concat(list,axis=1)
    df = pd.concat(list)
    df.to_csv("/home/oba/csv_dataset/1112/1112traindata/1112sum.csv",index = False)