import os

for i in range(15):   
    os.makedirs("/home/oba/csvpcapdataset/R_"+str(i) ,exist_ok=True)

macaddress = ["38:f9:d3:3f:49:2e","86:ce:22:4e:44:10","98:46:0a:d7:ba:71","a4:3b:fa:b2:45:6e"]
a2 = "38:f9:d3:3f:49:2e"
if a2 in macaddress:
    print(a2)