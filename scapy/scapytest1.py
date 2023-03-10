from scapy.all import *
from scapy.all import Ether
from scapy.all import IP   
from scapy.all import TCP
import pandas as pd
import os
import glob
packet=rdpcap("/home/oba/dataset/shiomotolab02/wificapture_5_2/wificap_20211007154300_00005_20211007154903.pcap")
a4 = (packet[4]['Dot11Elt'].info).decode()
#a4 = a4[2:-1]
a = packet[4]['Dot11'].subtype
print(a,a4)
print(type(a4))
print(type(a))
for i in range(20):
    #packet[i].show()
    try :
        a1 = packet[i].time                         #時間取得
        #a1 = packet[i]['Dot11'].timestamp
        print(a1)
a2 = packet[i]['Dot11'].addr2               #MACアドレス取得
        a3 = packet[i].dBm_AntSignal                #RSSI取得
        a4 = packet[i][Dot11Elt].info               #SSID取得
        #print(a1)
        #print(a2)
        #print(a3)
        print(a4)
        #print("----------------------------------------------------")'''
'''    except :
        a4 = ("")
        print(a4)'''




'''print(a1,a2,a3,packet[1]['Dot11'].timestamp,a5)
'''