# ライブラリのインポート
import numpy as np
import pandas as pd
from pandas import plotting
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm

pd.set_option('display.max_rows',10000)

# データセットの読み込み
#df = pd.read_csv("/home/oba/csvpcapdataset/traindataset/sumtest.csv")
df = pd.read_csv("/home/oba/csv_dataset/111217/111217train.csv")

# 特徴量と目的変数の選定
#X = df[["AP9","AP8","AP7","AP6","AP5","AP4", "AP3", "AP2", "AP1"]]
X = df[["AP3","AP4","AP6","AP7"]]
#X = df[["AP1"]]
y  = df["lavel"]

#X_train = df[["AP9","AP8","AP7","AP6","AP5","AP4", "AP3", "AP2", "AP1"]]
#X_train = df[["AP1","AP2","AP3","AP7"]]
#X_train = df[["AP1"]]
#y_train  = df["lavel"]

# テストデータ分割
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
'''
print(X_train)
print(y_train)
'''

# KNNのインスタンス定義
knn = KNeighborsClassifier(n_neighbors=7)

# モデルfit
knn.fit(X,y)

# テストデータセットの読み込み
df2 = pd.read_csv("/home/oba/csv_dataset/111217/111217test.csv")
#X_test = df2[["AP9","AP8","AP7","AP6","AP5","AP4", "AP3", "AP2", "AP1"]]
X_test = df2[["AP3","AP4","AP6","AP7"]]
#X_test = df2[["AP1"]]
y_test  = df2["lavel"]

#スコア計算
score = format(knn.score(X_test, y_test))
print(X_test.columns.values,',正解率:', score)

Y_pred = knn.predict(X_test)
#print(y_test.transpose().values.tolist())
print(Y_pred)
#print(metrics.accuracy_score(y_test, Y_pred))
"""
accuracy_list = []
sns.set()
k_range = range(1, 100)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    Y_pred = knn.predict(X_test)
    accuracy_list.append(metrics.accuracy_score(y_test, Y_pred))

figure = plt.figure()
ax = figure.add_subplot(111)
ax.plot(k_range, accuracy_list)
ax.set_xlabel('k-nn')
ax.set_ylabel('accuracy')
plt.show()
#plt.savefig("/home/oba/programs/knn/111217k=7igoukannR14AP1543.png")
plt.clf()"""
