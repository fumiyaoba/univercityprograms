from scapy.all import *
from scapy.all import Ether
from scapy.all import IP   
from scapy.all import TCP
packet=rdpcap("/home/oba/dataset/pi/test9/wificap_20210923103100_00010_20210923103233.pcap")
pkt0 = packet[0]
headers = ['802.11-FCS']
a1 = pkt0['Dot11'].addr2
a2 = pkt0.dBm_AntSignal
try :
    a = packet[3]['Dot11'].timestamp
    print(a)
except :
    a = packet[4]['Dot11'].timestamp
    print(a)