from scapy.all import *
from scapy.all import Ether
from scapy.all import IP   
from scapy.all import TCP
import pandas as pd
import os
import glob
import datetime


file_name = '/home/oba/dataset/morishita/wificapture_10_2'
ap_number = '1'
date = "2021-10-07-test1"

#os.makedirs("/home/oba/csv_dataset/" + date + "/" ,exist_ok=True)
files = sorted(glob.glob(file_name+'/*.pcap'))
#print(files)

df = []

for file in files:
    packet = rdpcap(file)
    #for i in range(len(packet)):
    for i in range(len(packet)):
        df_line = []
        try :
            a = packet[i]['Dot11'].subtype#subtype取得
        except :
            a = 0
        if  a ==4 :#subtypeが8(beacon)の時のみcsvファイルに書き込む
            packet[i].show()
            break

#df=pd.DataFrame(df)
#df.columns = ["TIME","Mac_Address","RSSI","SSID"]
#df.to_csv("/home/oba/csv_dataset/" + date + "/" + ap_number + ".csv",index = False)