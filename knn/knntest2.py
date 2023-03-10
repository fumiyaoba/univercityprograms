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

# データセットの読み込み
#df = pd.read_csv("/home/oba/csvpcapdataset/traindataset/sumtest.csv")
df = pd.read_csv("/home/oba/csvpcapdataset/traindataset/sumtestnoAP3.csv")

# 特徴量と目的変数の選定
#X = df[["AP4","AP3", "AP2", "AP1"]]
X = df[["AP4", "AP2", "AP1"]]
y  = df["lavel"]

# テストデータ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train)
print(y_train)


# KNNのインスタンス定義
knn = KNeighborsClassifier(n_neighbors=5)

# モデルfit
knn.fit(X,y)

#スコア計算
score = format(knn.score(X_test, y_test))
print('正解率:', score)

Y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, Y_pred))

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
plt.savefig("/home/oba/programs/knn/test1.png")
plt.clf()

# 正規化
scaler = MinMaxScaler()
df.loc[:,:]  = scaler.fit_transform(df)
df.head()

plotting.scatter_matrix(df.iloc[:, 1:], figsize=(8, 8), c=list(df.iloc[:, 0]), alpha=0.5)
plt.show()
plt.savefig("/home/oba/programs/knn/test3.png")
plt.clf()

dfs = df.iloc[:, 1:].apply(lambda x: (x-x.mean())/x.std(), axis=0)

#主成分分析の実行
pca = PCA()
pca.fit(dfs)
# データを主成分空間に写像
feature = pca.transform(dfs)

# 主成分得点
pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(len(dfs.columns))]).head()


# 第一主成分と第二主成分でプロットする
plt.figure(figsize=(6, 6))
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=list(df.iloc[:, 0]),cmap=cm.seismic)
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.plot(color='#e41a1c')
plt.show()
plt.savefig("/home/oba/programs/knn/test4.png")
plt.clf()

from pandas import plotting 
plotting.scatter_matrix(pd.DataFrame(feature, 
                        columns=["PC{}".format(x + 1) for x in range(len(dfs.columns))]), 
                        figsize=(8, 8), c=list(df.iloc[:, 0]), alpha=0.5) 
plt.show()
plt.savefig("/home/oba/programs/knn/test5.png")
plt.clf()

# 寄与率
pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])

# 累積寄与率を図示する
import matplotlib.ticker as ticker
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution rate")
plt.grid()
plt.show()
plt.savefig("/home/oba/programs/knn/test6.png")
plt.clf()

# PCA の固有値
pd.DataFrame(pca.explained_variance_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])

# PCA の固有ベクトル
pd.DataFrame(pca.components_, columns=df.columns[1:], index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])

# 第一主成分と第二主成分における観測変数の寄与度をプロットする
plt.figure(figsize=(6, 6))
for x, y, name in zip(pca.components_[0], pca.components_[1], df.columns[1:]):
    plt.text(x, y, name)

plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
plt.savefig("/home/oba/programs/knn/test7.png")
plt.clf()

