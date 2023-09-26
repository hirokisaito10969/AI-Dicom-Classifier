# https://www.tensorflow.org/tutorials/keras/classification?hl=ja

import tensorflow as tf
import numpy as np
import pydicom
import os
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import onnxmltools
import datetime
import shutil
import pickle
import time
import sys
import os

def resize_array_bilinear(original_array, new_rows, new_cols):
    original_rows, original_cols = original_array.shape
    resized_array = np.zeros((new_rows, new_cols))
    row_ratio = original_rows / new_rows
    col_ratio = original_cols / new_cols
    for i in range(new_rows):
        for j in range(new_cols):
            row_index = int(i * row_ratio)
            col_index = int(j * col_ratio)

            p1 = original_array[row_index, col_index]
            p2 = original_array[min(row_index + 1, original_rows - 1), col_index]
            p3 = original_array[row_index, min(col_index + 1, original_cols - 1)]
            p4 = original_array[min(row_index + 1, original_rows - 1), min(col_index + 1, original_cols - 1)]

            weight_row = (i * row_ratio) - row_index
            weight_col = (j * col_ratio) - col_index
            resized_array[i, j] = (1 - weight_row) * (1 - weight_col) * p1 + weight_row * (1 - weight_col) * p2 + \
                                (1 - weight_row) * weight_col * p3 + weight_row * weight_col * p4
    return resized_array

def read_dicom_and_get_parent_folder(directory,image_size):
    # print
    print("Start reading dicom")
    # DICOM画像のピクセルデータを格納するリスト
    pixel_array_list = []
    # 一つ上のディレクトリ名を格納するリスト
    parent_folder_names = []
    # count
    count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".dcm"):
                dicom_file_path = os.path.join(root, file)
                ds = pydicom.dcmread(dicom_file_path)
                
                # ピクセルデータをNumpy配列に変換し、リストに追加
                pixel_array = ds.pixel_array
                pixel_array = np.array(pixel_array)
                pixel_array = resize_array_bilinear(pixel_array, image_size, image_size)
                pixel_array_list.append(pixel_array)

                # 一つ上のディレクトリ名を取得し、種類によって整数のラベルを追加
                parent_folder = os.path.basename(os.path.dirname(dicom_file_path))
                if parent_folder == 'cine':
                    parent_folder_names.append(0)
                elif parent_folder == 'mag':
                    parent_folder_names.append(1)
                elif parent_folder == 'x':
                    parent_folder_names.append(2)
                elif parent_folder == 'y':
                    parent_folder_names.append(3)
                elif parent_folder == 'z':
                    parent_folder_names.append(4)
                
                # count
                count += 1
                if count % 100 == 0:
                    print(".", end=(""), flush=True)

    pixel_array_list = np.asarray(pixel_array_list, dtype=np.float32)
    parent_folder_names = np.array(parent_folder_names)

    return pixel_array_list, parent_folder_names


# 計測時間記録用
start = time.perf_counter()
date_now_start = datetime.datetime.now()
print(date_now_start)

class_names = ['Cine', 'Mag', 'X', 'Y', 'Z']

image_size = 64
batch_size = 1
epochs = 10
input_directory = "E:/saito/AI/classifier/dicom"
images, labels = read_dicom_and_get_parent_folder(input_directory,image_size)

print(labels)


images_len = len(images)
num = int(images_len * 0.8)
train_images, test_images = np.split(images, [num])
train_labels, test_labels = np.split(labels, [num])

# 画素値0-1正規化
train_images /= np.max(train_images)
test_images /= np.max(test_images)


Regularizer = None
# Regularizer = regularizers.l1(0.01)
# Regularizer = regularizers.l2(0.01)
# Regularizer = regularizers.l1_l2(l1=0.01, l2=0.01)

# kernel_regularizer
# bias_regularizer
# activity_regularizer


# Jaccard係数を計算するカスタム評価関数
def jaccard_coefficient(y_true, y_pred):
    # y_trueとy_predはバッチごとの真のラベルと予測ラベルです
    intersection = tf.reduce_sum(tf.minimum(y_true, y_pred))
    union = tf.reduce_sum(tf.maximum(y_true, y_pred))
    jaccard = intersection / union
    return jaccard

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(image_size, image_size)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5)
])
     
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['acc', jaccard_coefficient])


date_now = datetime.datetime.today().strftime("%Y%m%d_%H%M")
path = os.path.join(os.path.dirname(__file__), "reslut-"+str(image_size), date_now)
os.makedirs(path, exist_ok=True)

csv_logger = CSVLogger(filename = os.path.join(path, 'training.log'), separator=',', append=True)
training = model.fit(x=train_images, y=train_labels, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_images, test_labels), shuffle=True)



# %% training結果をディレクトリに保存
# 結果を保存するディレクトリを作成
print("make directory and write log")

# ネットワークをテキスト形式で保存
with open(os.path.join(path, "model_summary.txt"), "w") as fp:
    model.summary(print_fn=lambda x: fp.write(x + "\r\n"))

# ネットワーク定義を保存
open(os.path.join(path, "keras_unet_model.json"), "w").write(model.to_json())

# 学習履歴グラフ表示
def plot_history_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(path, "accuracy.png"))
    plt.close()

def plot_history_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.savefig(os.path.join(path, "loss.png"))
    plt.close()

plot_history_acc(training)
plot_history_loss(training)

# copy file
shutil.copy(__file__, os.path.join(path, os.path.basename(__file__)))

# hdf5ファイルを保存するためのディレクトリの作成
path_hdf5 = os.path.join(os.path.dirname(__file__), "reslut-"+str(64), "hdf5_files", date_now)
os.makedirs(path_hdf5, exist_ok=True)

# hdf5モデルを保存
model.save(os.path.join(path_hdf5, 'keras_unet_model.hdf5'))

# onnxファイルを保存するためのディレクトリの作成
path_onnx = os.path.join(os.path.dirname(__file__), "reslut-"+str(64), "onnx_files", date_now)
os.makedirs(path_onnx, exist_ok=True)

# onnxモデルを保存
onnx_model = onnxmltools.convert_keras(model)
onnxmltools.utils.save_model(onnx_model, os.path.join(path_onnx, "model.onnx"))

def elapsed_time_str(seconds):
    # 秒数を四捨五入
    seconds = int(seconds + 0.5)
    # 時の取得
    h = seconds // 3600
    # 分の取得
    m = (seconds - h * 3600) // 60
    # 秒の取得
    s = seconds - h * 3600 - m * 60
    # hh:mm:ss形式の文字列で返す
    return f"{h:02}:{m:02}:{s:02}"

date_now_end = datetime.datetime.now()

open(os.path.join(path, "time.txt"), "a").write(str(date_now_start)+"\n")
open(os.path.join(path, "time.txt"), "a").write(elapsed_time_str(time.perf_counter() - start)+"\n")
open(os.path.join(path, "time.txt"), "a").write(str(date_now_end)+"\n")

open(os.path.join(path, "learning_info.txt"), "a").write("DATA_SIZE:"+str(image_size)+"\n")
open(os.path.join(path, "learning_info.txt"), "a").write("EPOCHS:"+str(epochs)+"\n")
open(os.path.join(path, "learning_info.txt"), "a").write("BATCH_SIZE:"+str(batch_size)+"\n")
open(os.path.join(path, "learning_info.txt"), "a").write("all_org:"+str(len(images))+"\n")
open(os.path.join(path, "learning_info.txt"), "a").write("num:"+str(num)+"\n")
open(os.path.join(path, "learning_info.txt"), "a").write("learning_org:"+str(len(train_images))+"\n")
open(os.path.join(path, "learning_info.txt"), "a").write("test_org:"+str(len(test_images))+"\n")
if Regularizer is not None:
    open(os.path.join(path, "learning_info.txt"), "a").write("Regularizer:"+str(Regularizer)+"\n")
else:
    open(os.path.join(path, "learning_info.txt"), "a").write("Regularizer:None"+"\n")

print(datetime.datetime.now())
print("training is completed")

import subprocess

def line_notify(msg='GTune process is completed !', token='vncOXztFelsVBTn8gDD1hxLXwEOqLHZI3t4GUwIKjSc'):
    subprocess.run(['curl', '-H', f'Authorization: Bearer {token}', '-F', f'message={msg}', 'https://notify-api.line.me/api/notify'])

line_notify()

