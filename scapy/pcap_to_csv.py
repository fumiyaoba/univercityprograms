from scapy.all import *
from scapy.all import Ether
from scapy.all import IP   
from scapy.all import TCP
import pandas as pd
import os
import glob
import datetime


file_name = '/home/oba/dataset/pi/test15'
ap_number = '3'

#os.makedirs("/home/oba/csv_dataset/" + date + "/" ,exist_ok=True)
files = sorted(glob.glob(file_name+'/*.pcap'))
print(files)
files =["/home/oba/pcapdataset/AP3/20221002AP3_point05_2.pcap"]
print(files)

df = []

for file in files:
    packet = rdpcap(file)
    #for i in range(len(packet)):
    for i in range(len(packet)):
        df_line = []
        try :
            a4 = packet[i]['Dot11'].subtype#subtype取得
        except :
            a4 = 0
        try :
            a5 = (packet[i]['Dot11Elt'].info).decode()#SSID取得後bytes型をstring型へ変換
        except :
            a5 = ("")
        if  a4 == 4 :#and a5 == "TCUWiFi":subtypeが4(ProbeRequest)の時のみcsvファイルに書き込む
            try :
                #a1 = packet[i]['Dot11'].timestamp
                a1 = packet[i].time #時間取得
                print(a1)
                a1 = datetime.datetime.fromtimestamp(a1)#unic時刻を日付表示に
                date = a1.date()
                print(a1)
                a1 = a1.time()
                print(a1)
            except :
                a1 = ("")
            try :
                a2 = packet[i]['Dot11'].addr2#MACアドレス取得
            except :
                a2 = ("") 
            try :
                a3 = packet[i].dBm_AntSignal#RSSI取得
            except :
                a3 = ("")

   
            df_line.extend([a1,a2,a3,a5])
    
            df.append(df_line)

df=pd.DataFrame(df)
df.columns = ["TIME","Mac_Address","RSSI","SSID"]
#df.to_csv("/home/oba/csv_dataset/" + date + "/" + ap_number + ".csv",index = False)
df.to_csv("/home/oba/test1.csv",index = False)