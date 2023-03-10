import numpy as np
import sys
import ast
import time
import json
import random
import argparse
import requests as req
from datetime import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas.tseries.offsets as offsets

#jsonのネストを要素ごとに分解
def conv_to_2d(objct, parent=None, num=None): 
    for key, vals in objct.items():
        # keyの設定
        if parent is not None and num is not None:
            abs_key = "{}.{}.{}".format(parent, key, num)
        elif parent is not None:
            abs_key = "{}.{}".format(parent, key)
        else:
            abs_key = key

        # valsのタイプごとに処理を分岐
        if type(vals) is dict:
            yield from conv_to_2d(objct=vals, parent=key)
        elif type(vals) is list:
            val_list = []
            for n, val in enumerate(vals):
                is_target = [type(val) is int, type(val) is float, type(val) is bool]
                if type(val) is str:
                    if val:
                        val_list += [val]
                elif any(is_target):
                    num_str = str(val)
                    if num_str:
                        val_list += [num_str]
                elif type(val) is dict:
                    yield from conv_to_2d(objct=val, parent=abs_key, num=n)
            if val_list:
                yield abs_key, ",".join(val_list)
        else:
            yield abs_key, vals



def main():
    #標準入力(入力ファイルと出力ファイルの指定)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--infile', nargs='?', type=argparse.FileType(),
                        default=sys.stdin)
    parser.add_argument('-o','--outfile',nargs='?',
                        default=sys.stdin)
    args = parser.parse_args()
    
    #jsonファイルの読み込み
    wifi_table = []
    with open('/home/oba/morishita_cp/test/dataspace/20211223_pcaplog_morishita.json','r') as f:
        for line in f:
            s = json.loads(line)
            if len(s['layers']['radiotap_dbm_antsignal']) >= 2:
                for number in range(0,len(s['layers']['radiotap_dbm_antsignal'])):
                    s['layers']['radiotap_dbm_antsignal'][number] = int(s['layers']['radiotap_dbm_antsignal'][number])
                s['layers']['radiotap_dbm_antsignal'] = min(s['layers']['radiotap_dbm_antsignal'])
            else:
                s['layers']['radiotap_dbm_antsignal'][0] = int(s['layers']['radiotap_dbm_antsignal'][0])
            record = {key: val for key, val in conv_to_2d(s)}
            wifi_table.append(record)
    
    #timestampの調整
    df = pd.DataFrame(wifi_table)
    df = df.drop("layers.frame_time",axis=1)
    df["timestamp"] = pd.to_datetime(df["timestamp"],unit="ms")
    df["timestamp"] = df["timestamp"] + offsets.Hour(9)
    df["layers.radiotap_dbm_antsignal"] = df["layers.radiotap_dbm_antsignal"].astype(int)

    #Probe Requestの絞り込み
    #df1 = df[df['layers.radiotap_dbm_antsignal'] > -50]
    #df1 = df[(df["layers.wlan_sa"].str[1] == "2") | (df["layers.wlan_sa"].str[1] == "6") | (df["layers.wlan_sa"].str[1] == "a") | (df["layers.wlan_sa"].str[1] == "e") ]
    #df1 = df[((df["layers.wlan_sa"].str[1] == "2") | (df["layers.wlan_sa"].str[1] == "6") | (df["layers.wlan_sa"].str[1] == "a") | (df["layers.wlan_sa"].str[1] == "e")) & (df['layers.radiotap_dbm_antsignal'] > -40)]
    #print(df1)
    
    #発信間隔
    #new_df = df.groupby(pd.Grouper(key='timestamp', freq='1min')).nunique().reset_index()
    #new_df_1min = df.groupby(pd.Grouper(key='timestamp', freq='1min')).nunique().reset_index()
    #new_df_90S = df.groupby(pd.Grouper(key='timestamp', freq='90S')).nunique().reset_index()
    #new_df_3min = df.groupby(pd.Grouper(key='timestamp', freq='3min')).nunique().reset_index()
    new_df_3min = df.groupby(pd.Grouper(key='timestamp', freq='3min')).count().reset_index()
    #new_df_5min = df.groupby(pd.Grouper(key='timestamp', freq='5min')).nunique().reset_index()
    #new_df_10min = df.groupby(pd.Grouper(key='timestamp', freq='10min')).nunique().reset_index()
    #new_df = df1.groupby(pd.Grouper(key='timestamp', freq='10min')).count().reset_index()

    #ウインドウサイズに分割後1分辺りのprobe request数で割って端末数算出
    #new_df_1min["layers.wlan_sa"] = new_df_1min["layers.wlan_sa"] / 2 
    #new_df_3min["layers.wlan_sa"] = new_df_3min["layers.wlan_sa"] / 3 
    #new_df_5min["layers.wlan_sa"] = new_df_5min["layers.wlan_sa"] / 5 
    #new_df_10min["layers.wlan_sa"] = new_df_10min["layers.wlan_sa"] / 10
    #new_df_3min_span15 = new_df_3min["layers.wlan_sa"] / 12
    #new_df_3min_span30 = new_df_3min["layers.wlan_sa"] / 6
    #new_df_3min_span90 = new_df_3min["layers.wlan_sa"] / 2
    #new_df_3min_span120 = new_df_3min["layers.wlan_sa"] / 3 * 2
    
    #時間と端末数のdataframe作成
    #new_df_1 = pd.concat([new_df_1min["timestamp"],new_df_1min["layers.wlan_sa"]],axis=1)
    #new_df_90 = pd.concat([new_df_90S["timestamp"],new_df_90S["layers.wlan_sa"]],axis=1)
    new_df_3 = pd.concat([new_df_3min["timestamp"],new_df_3min["layers.wlan_sa"]],axis=1)
    #new_df_5 = pd.concat([new_df_5min["timestamp"],new_df_5min["layers.wlan_sa"]],axis=1)
    #new_df_10 = pd.concat([new_df_10min["timestamp"],new_df_10min["layers.wlan_sa"]],axis=1)
    #new_df_3_s15 = pd.concat([new_df_3min["timestamp"],new_df_3min_span15],axis=1)
    #new_df_3_s30 = pd.concat([new_df_3min["timestamp"],new_df_3min_span30],axis=1)
    #new_df_3_s90 = pd.concat([new_df_3min["timestamp"],new_df_3min_span90],axis=1)
    #new_df_3_s120 = pd.concat([new_df_3min["timestamp"],new_df_3min_span120],axis=1)

    #指数平滑移動平均
    #new_df_1["layers.wlan_sa"] = new_df_1["layers.wlan_sa"].ewm(span=20,adjust=False).mean()
    #new_df_90["layers.wlan_sa"] = new_df_90["layers.wlan_sa"].ewm(span=20,adjust=False).mean()
    #new_df_3["layers.wlan_sa"] = new_df_3["layers.wlan_sa"].ewm(span=7,adjust=False).mean()
    #new_df_5["layers.wlan_sa"] = new_df_5["layers.wlan_sa"].ewm(span=20,adjust=False).mean()
    #new_df_10["layers.wlan_sa"] = new_df_10["layers.wlan_sa"].ewm(span=20,adjust=False).mean()
    #new_df_3_s15["layers.wlan_sa"] = new_df_3_s15["layers.wlan_sa"].ewm(span=20,adjust=False).mean()
    #new_df_3_s30["layers.wlan_sa"] = new_df_3_s30["layers.wlan_sa"].ewm(span=20,adjust=False).mean()
    #new_df_3_s90["layers.wlan_sa"] = new_df_3_s90["layers.wlan_sa"].ewm(span=20,adjust=False).mean()
    #new_df_3_s120["layers.wlan_sa"] = new_df_3_s120["layers.wlan_sa"].ewm(span=20,adjust=False).mean()
    #print(new_df_1)
    #print(new_df_90)
    print(new_df_3)
    #print(new_df_5)
    #print(new_df_10)
    #print(new_df_3_s15)
    #print(new_df_3_s30)
    #print(new_df_3_s90)
    #print(new_df_3_s120)
    #print(new_df_1min["layers.radiotap_dbm_antsignal"])
    #print(new_df_3min)
    #print(new_df_5min)
    #print(new_df_10min)
    
    #dfを読み込んで配列に入れたい
    new_df_3_1 = new_df_3.values
    print(new_df_3_1)

    #グラフ
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    #ax.plot("timestamp","layers.wlan_sa",data=new_df_1,label="1min")
    #ax.plot("timestamp","layers.wlan_sa",data=new_df_90,label="90sec")
    ax.plot("timestamp","layers.wlan_sa",data=new_df_3,label="3min")
    #ax.plot("timestamp","layers.wlan_sa",data=new_df_5,label="5min")
    #ax.plot("timestamp","layers.wlan_sa",data=new_df_3_s15,label="15sec")
    #ax.plot("timestamp","layers.wlan_sa",data=new_df_3_s30,label="30sec")
    #ax.plot("timestamp","layers.wlan_sa",data=new_df_3,label="1min")
    #ax.plot("timestamp","layers.wlan_sa",data=new_df_3_s90,label="90sec")
    #ax.plot("timestamp","layers.wlan_sa",data=new_df_3_s120,label="2min")
    ax.legend()
    ax.grid(which = "major", axis = "y", color = "black", alpha = 0.5,
         linewidth = 1)
    plt.xlabel("時間(分)")
    #plt.ylabel("推定端末数")
    plt.ylabel("Probe Request数")

    plt.savefig( "./result" + "TEST.png")
    plt.close('all')
    

'''
    plt.figure(figsize=(40,28))
    new_df.plot()
    #sns.scatterplot(x='timestamp', y='layers.wlan_sa', data=new_df,legend=True)
    #sns.scatterplot(x='timestamp', y='layers.radiotap_dbm_antsignal', hue='layers.wlan_sa', data=new_df,legend=True)
    plt.savefig( "./result" + args.outfile + ".png")
    plt.close('all')
    '''

def array():
        #標準入力(入力ファイルと出力ファイルの指定)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--infile', nargs='?', type=argparse.FileType(),
                        default=sys.stdin)
    parser.add_argument('-o','--outfile',nargs='?',
                        default=sys.stdin)
    args = parser.parse_args()
    
    #jsonファイルの読み込み
    wifi_table = []
    with open('/home/oba/morishita_cp/test/dataspace/20211223_pcaplog_morishita.json','r') as f:
        for line in f:
            s = json.loads(line)
            if len(s['layers']['radiotap_dbm_antsignal']) >= 2:
                for number in range(0,len(s['layers']['radiotap_dbm_antsignal'])):
                    s['layers']['radiotap_dbm_antsignal'][number] = int(s['layers']['radiotap_dbm_antsignal'][number])
                s['layers']['radiotap_dbm_antsignal'] = min(s['layers']['radiotap_dbm_antsignal'])
            else:
                s['layers']['radiotap_dbm_antsignal'][0] = int(s['layers']['radiotap_dbm_antsignal'][0])
            record = {key: val for key, val in conv_to_2d(s)}
            wifi_table.append(record)
    
    #timestampの調整
    df = pd.DataFrame(wifi_table)
    df = df.drop("layers.frame_time",axis=1)
    df["timestamp"] = pd.to_datetime(df["timestamp"],unit="ms")
    df["timestamp"] = df["timestamp"] + offsets.Hour(9)
    df["layers.radiotap_dbm_antsignal"] = df["layers.radiotap_dbm_antsignal"].astype(int)
    
    #発信間隔    
    new_df_3min = df.groupby(pd.Grouper(key='timestamp', freq='3min')).count().reset_index()
    new_df_3 = pd.concat([new_df_3min["timestamp"],new_df_3min["layers.wlan_sa"]],axis=1)


    new_df_3_1 = new_df_3.values
    return new_df_3_1

    



if __name__ == "__main__":
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    array = array()
    time, request  =np.hsplit(array, 2)
    plt.plot(time, request, label="test")
    plt.show()
    plt.savefig( "./result" + "TEST2.png")
    plt.close('all')
    print(type(array))