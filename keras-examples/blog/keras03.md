Kerasによる2クラスロジスティック回帰
2016/10/19

まずはもっとも簡単な2クラスロジスティック回帰モデルをKerasで書いてみる。ロジスティック回帰は、回帰とつくけど分類のアルゴリズムで、隠れ層のないニューラルネットとしてモデル化できる。

データは、PRMLの4章の[ex2data1](https://raw.githubusercontent.com/sylvan5/PRML/master/ch4/ex2data1.txt) を使う。1列目と2列目がデータで3列目がクラスラベル（0または1の2クラス）。

logreg_ex2data1.py

# データのロードと正規化

データを読み込むライブラリにはpandasなどもあるが、ここではnumpy.genfromtxt()を使う。Xがデータで二次元データのリスト、tがラベルで0または1のリスト。

```python
# load training data
data = np.genfromtxt(os.path.join('data', 'ex2data1.txt'), delimiter=',')
X = data[:, (0, 1)]
t = data[:, 2]
```

データの各列の平均が0、標準偏差が1になるようにデータを正規化する。この正規化をしないと学習がまったく進まない（=lossが小さくならない）ケースが多かったのでやった方がよさそう。sklearnのpreprocessingモジュールに正規化のメソッドがあるので使う。

```python
from sklearn import preprocessing
# normalize data
X = preprocessing.scale(X)
```

以下のコードで平均0、標準偏差1であることが確認できる。平均は0っぽくないが、e-17なので限りなく0に近い。

```python
print(np.mean(X, axis=0))
print(np.std(X, axis=0))
```

```
[ -7.66053887e-17   1.11022302e-15]
[ 1.  1.]
```

# データの可視化

どのようなデータかmatplotlibで可視化する。

```python
import matplotlib.pyplot as plt

def plot_data(X, t):
    positive = [i for i in range(len(t)) if t[i] == 1]
    negative = [i for i in range(len(t)) if t[i] == 0]

    plt.scatter(X[positive, 0], X[positive, 1], c='red', marker='o', label='positive')
    plt.scatter(X[negative, 0], X[negative, 1], c='blue', marker='o', label='negative')

# plot training data
plt.figure(1)
plot_data(X, t)

```

赤が正例（positive）で青が負例（negative）。この2クラスを直線で分類するのが目標。

【プロット図】

# ロジスティック回帰モデルの構築

【図】

ロジスティック回帰モデルを組み立てる。空のSequentialモデルを作成し、そこにレイヤ（Dense）や活性化関数（Activation）を順番に追加し、最後にコンパイルする。Sequentialモデルというのは通常のニューラルネットのように層を積み重ねたモデルを指す。コンパイル時に最適化手法（optimizer）、損失関数（loss）、評価指標（metrics）を指定する。

```python
# create the logistic regression model
model = Sequential()
model.add(Dense(1, input_shape=(2, )))
model.add(Activation('sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

- このタスクは0または1を予測する二値分類なので損失関数には`binary_crossentropy`を用いた。
- あとで実験するがMNISTのような多値分類では`categorical_crossentropy`を用いた。
- 最適化アルゴリズムにはAdam、評価指標には精度を用いた。これは訓練データで精度求めているのかな？要検証。

modelの組み立て方は上の方法がスタンダードだけれど、[Functional API](https://keras.io/ja/getting-started/functional-api-guide/)を使うともっと柔軟にモデルが作れるようだ。あとでこのAPIを使った書き方もしてみよう。

# 訓練

モデルの訓練はscikit-learnと同じくfit()という関数にデータとラベルを渡せばよい。

```python
# fit the model
model.fit(X, t, nb_epoch=1000, batch_size=5, verbose=1)
```

- fit()には、固定のエポック数（nb_epoch）、バッチサイズ（batch_size）、経過出力方法（verbose）を指定する。

```
Epoch 1/1000
100/100 [==============================] - 0s - loss: 0.6531 - acc: 0.6600
Epoch 2/1000
100/100 [==============================] - 0s - loss: 0.6455 - acc: 0.6700
Epoch 3/1000
100/100 [==============================] - 0s - loss: 0.6385 - acc: 0.6700
Epoch 4/1000
100/100 [==============================] - 0s - loss: 0.6310 - acc: 0.6800
Epoch 5/1000
100/100 [==============================] - 0s - loss: 0.6243 - acc: 0.6800
```

- verboseを1にしておくと学習経過を棒グラフで表示してくれるので非常に便利！
- あとで紹介するEarly-stoppingを使うと固定エポックだけループを回すのではなく、収束判定して止めてくれるようになる。

# 学習した重みの取得

`model`の`layers`リストに各層の情報が格納されている。好きな層を指定して`get_weights()`で重みが取得できる。重みが多次元リストなのでややこしいが、最初の次元は、[0]が重み、[1]がバイアス項を表すようだ。

【図】

```python
# get the learned weight
weights = model.layers[0].get_weights()
w1 = weights[0][0, 0]
w2 = weights[0][1, 0]
b = weights[1][0]
```

ここでは、決定境界を描画したいので学習した重みw1、w2とバイアスbを取得した。

# 決定境界の描画

ロジスティック回帰の決定境界の条件は

[tex: w1 * x1 + w2 * x2 + b = 0]

なので [x1, x2] 平面上での直線の方程式に直すと

[tex: x2 = - (w1 / w2) * x1 - (b / w2)]

となる。この直線をmatplotlibで先ほどのグラフに追記する。

```python
# draw decision boundary
plt.figure(1)
xmin, xmax = min(X[:, 0]), max(X[:, 0])
ymin, ymax = min(X[:, 1]), max(X[:, 1])
xs = np.linspace(xmin, xmax, 100)
ys = [- (w1 / w2) * x - (b / w2) for x in xs]
plt.plot(xs, ys, 'b-', label='decision boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.legend()
plt.show()
```

結果は、

【図】

となり、分類する直線が学習できていることがわかる。
今回は、テストデータを使った評価などは行っていない。

# 参考

- [Sequentialモデルのガイド](https://keras.io/ja/getting-started/sequential-model-guide/)
- Introduction to Python Deep Learning with Keras
