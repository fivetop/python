from vgg16_linear import VGG16
from keras.layers import Input
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import json
import sys


def deprocess_image(x):
    """正規化された3Dテンソル (row, col, channel) から元の画像を復元"""
    # テンソルを平均0、標準偏差0.1になるように正規化
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # [0, 1]にクリッピング
    x += 0.5
    x = np.clip(x, 0, 1)

    # RGBに変換
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')

    return x


def normalize(img, value):
    return value / np.prod(K.int_shape(img)[1:])


def rmsprop(grads, cache=None, decay_rate=0.95):
    """RMSpropによる最適化"""
    if cache is None:
        cache = np.zeros_like(grads)
    cache = decay_rate * cache + (1 - decay_rate) * grads ** 2
    step = grads / np.sqrt(cache + K.epsilon())

    return step, cache


def visualize_filter(layer_name, filter_index, num_loops=200):
    """指定した層の指定したフィルタの出力を最大化する入力画像を勾配法で求める"""
    if layer_name not in layer_dict:
        print("ERROR: invalid layer name: %s" % layer_name)
        return

    # 指定した層
    layer = layer_dict[layer_name]

    # layer.output_shape[-1]はどの層でもフィルタ数にあたる（tfの場合）
    # predictions層の場合はクラス数になる
    if not (0 <= filter_index < layer.output_shape[-1]):
        print("ERROR: invalid filter index: %d" % filter_index)
        return

    # 指定した層の指定したフィルタの出力の平均
    activation_weight = 1.0
    if layer_name == 'predictions':
        # 出力層だけは2Dテンソル (num_samples, num_classes)
        activation = activation_weight * K.mean(layer.output[:, filter_index])
    else:
        # 隠れ層は4Dテンソル (num_samples, row, col, channel)
        activation = activation_weight * K.mean(layer.output[:, :, :, filter_index])

    # Lpノルム正則化項
    # 今回の設定ではactivationは大きい方がよいため正則化のペナルティ項は引く
    p = 6.0
    lpnorm_weight = 10.0
    if np.isinf(p):
        lp = K.max(input_tensor)
    else:
        lp = K.pow(K.sum(K.pow(K.abs(input_tensor), p)), 1.0 / p)
    activation -= lpnorm_weight * normalize(input_tensor, lp)

    # Total Variationによる正則化
    beta = 2.0
    tv_weight = 10.0
    a = K.square(input_tensor[:, 1:, :-1, :] - input_tensor[:, :-1, :-1, :])
    b = K.square(input_tensor[:, :-1, 1:, :] - input_tensor[:, :-1, :-1, :])
    tv = K.sum(K.pow(a + b, beta / 2.0))
    activation -= tv_weight * normalize(input_tensor, tv)

    # 層の出力の入力画像に対する勾配を求める
    # 入力画像を微小量変化させたときの出力の変化量を意味する
    # 層の出力を最大化したいためこの勾配を画像に足し込む
    grads = K.gradients(activation, input_tensor)[0]

    # 正規化トリック
    # 画像に勾配を足し込んだときにちょうどよい値になる
    grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon())

    # 画像を入力して層の出力と勾配を返す関数を定義
    iterate = K.function([input_tensor], [activation, grads])

    # ノイズを含んだ画像（4Dテンソル）から開始する
    x = np.random.random((1, img_height, img_width, 3))
    x = (x - 0.5) * 20 + 128

    # 初期画像を描画
    # fig, ax = plt.subplots()
    # img = deprocess_image(x[0])
    # ax.imshow(img)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # fig.savefig('initial_image.png')

    # 勾配法で層の出力（activation_value）を最大化するように入力画像を更新する
    cache = None
    for i in range(num_loops):
        activation_value, grads_value = iterate([x])
        # activation_valueを大きくしたいので画像に勾配を加える
        step, cache = rmsprop(grads_value, cache)
        x += step
        print(i, activation_value)

    # 画像に戻す
    img = deprocess_image(x[0])

    return img


if len(sys.argv) != 2:
    print("usage: python visualize_vgg16.py [VGG16 layer name]")
    sys.exit(1)

# 入力画像のサイズ
img_width, img_height, num_channels = 224, 224, 3

# 1000クラスの出力とクラス名の対応
# 画像にクラス名を表示するのに使う
class_index = json.load(open('imagenet_class_index.json'))

# 入力画像を表す3Dテンソル
input_tensor = Input(shape=(img_height, img_width, num_channels))

# VGG16モデルをロード
model = VGG16(include_top=True, weights='imagenet', input_tensor=input_tensor)
layer_dict = dict([(layer.name, layer) for layer in model.layers])
model.summary()

# 可視化する層の名前
# model.summary()で表示される名前と同じ
target_layer = sys.argv[1]
assert target_layer in layer_dict.keys(), 'Layer ' + target_layer + ' not found in model.'

# 指定した層のフィルタの最大数
num_filter = layer_dict[target_layer].output_shape[-1]

# target_layerからランダムにフィルタを選択
nrows, ncols = 4, 4
num_images = nrows * ncols
# np.random.seed(0)

target_index = [np.random.randint(0, num_filter) for x in range(num_images)]
# target_index = [65, 18, 130, 779, 302, 100, 870, 366, 99, 9, 351, 144, 63, 704, 248, 282]

# 可視化した画像を描画
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8))
for i in range(nrows):
    for j in range(ncols):
        idx = nrows * i + j
        img = visualize_filter(target_layer, target_index[idx], num_loops=1000)
        ax = axes[i, j]
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')

        # クラス名を画像に描画
        if target_layer == "predictions":
            ax.text(5, 20, class_index['%d' % target_index[idx]][1])
        else:
            ax.text(5, 20, "filter: %d" % target_index[idx])

fig.subplots_adjust(wspace=0, hspace=0)
fig.tight_layout()
fig.savefig('result.png')
