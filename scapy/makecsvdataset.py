from scapy.all import *
from scapy.all import Ether
from scapy.all import IP   
from scapy.all import TCP
import pandas as pd
import os
import glob
import datetime

class source():
    def __init__(self,files): #.pcapを入力

        self.files = files
        self.files = sorted(self.files)
        print(self.files)
        self.df = []
        self.subtype = 4 #デフォルトは4(ProbeRequest)の時のみcsvファイルに書き込むにしておく

    def select_subtype(self,subtype):#全てのpacketをcsvに変換したいときは-1を入力
        self.subtype = subtype 

    def make_csvdf(self,macaddress,R,date,ap):
        #for file in self.files:
        for file in range(len(self.files)):        
        #    packet = rdpcap(file)
            print(type(self.files[file]))
            packet = rdpcap(self.files[file])

            if  self.subtype == -1 :#-1なら全てのパケットをcsvファイルに書き込む
                for i in range(len(packet)):
                    df_line = []
                    try :
                        time = packet[i].time #時間取得
                        time = datetime.datetime.fromtimestamp(time)#unic時刻を日付表示に
                    except :
                        time = ("")
                    try :
                        mac = packet[i]['Dot11'].addr2#MACアドレス取得
                    except :
                        mac = ("") 
                    try :
                        rssi = packet[i].dBm_AntSignal#RSSI取得
                    except :
                        rssi = ("")
                    if mac in macaddress:
                        df_line.extend([time,mac,rssi])
                        self.df.append(df_line)
            else :
                try :
                    sub = packet[i]['Dot11'].subtype#subtype取得
                except :
                    sub = False
                if sub == self.subtype :
                    try :
                        time = packet[i].time
                        time = datetime.datetime.fromtimestamp(time)
                    except :
                        time = ("")
                    try :
                        mac = packet[i]['Dot11'].addr2#MACアドレス取得
                    except :
                        mac = ("") 
                    try :
                        rssi = packet[i].dBm_AntSignal#RSSI取得
                    except :
                        rssi = ("")
                    if mac in macaddress:
                        df_line.extend([time,mac,rssi])
                        self.df.append(df_line)

        df=pd.DataFrame(self.df)
        try :
            df.columns = ["_ws.col.Time","wlan.sa","radiotap.dbm_antsignal"]
        except :
            false = False
        os.makedirs("/home/oba/csvpcapdataset/" + date + "/",exist_ok=True)
        df.to_csv("/home/oba/csvpcapdataset/" + date + "/" "R_" + R + "/"+ date +"AP"+ ap +".csv",index = False)

if __name__ == "__main__":
    macaddress = ["38:f9:d3:3f:49:2e","86:ce:22:4e:44:10","98:46:0a:d7:ba:71","68:ef:43:67:25:bd"]
#    macaddress = ["68:EF:43:67:25:BD"]
    subtype = -1
    R = "14"
    date = "1112"
    ap = "18-01"
    ap1 = "/home/oba/raspi/18-01/raspi18-01_R_14.pcap"
#    ap2 = "/home/oba/pcapdataset/AP2/20221002AP2_point05_test.pcap"
#    ap3 = "/home/oba/pcapdataset/AP3/20221002AP3_point06_test.pcap"
#    ap4 = "/home/oba/pcapdataset/AP4/20221002AP3_point05_test.pcap"
    files = [ap1]
    csv = source(files)
    csv.select_subtype(subtype)
    csv.make_csvdf(macaddress,R,date,ap)