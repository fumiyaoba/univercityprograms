from scapy.all import *
from scapy.all import Ether
from scapy.all import IP   
from scapy.all import TCP
import matplotlib.pyplot as plt


packet=rdpcap("/home/oba/dataset/pi/test9/wificap_20210923103100_00010_20210923103233.pcap")
RSSI = []
TIME = []
c = 0
for i in range(len(packet)):
    try :
        a = packet[i].time
        c = c + 1
        RSSI.append(packet[i].dBm_AntSignal)
        TIME.append(packet[i].time)
    except :
        a = 0

plt.plot(TIME,RSSI)
plt.savefig('test.png')