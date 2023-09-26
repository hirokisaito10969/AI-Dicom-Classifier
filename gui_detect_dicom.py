from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import tkinter as tk
import numpy as np
import pydicom
import h5py

def open_dicom_file():
    file_path = filedialog.askopenfilename(filetypes=[("DICOM files", "*.dcm")])
    if file_path:
        display_dicom_image(file_path)
        display_file_name(file_path)
        detect_dicom(file_path)

def display_dicom_image(file_path):
    dicom_data = pydicom.dcmread(file_path)
    pixel_array = dicom_data.pixel_array

    image = Image.fromarray(pixel_array)
    photo = ImageTk.PhotoImage(image)

    label = tk.Label(root, image=photo)
    label.image = photo
    label.pack()

def display_file_name(file_path):
    file_name_label.config(text="選択されたファイル: {}".format(file_path))

def display_ai_detecting(result):
    ai_detecting.config(text="AI:このファイルは: {}です。".format(result))

# Jaccard係数を計算するカスタム評価関数
def jaccard_coefficient(y_true, y_pred):
    # y_trueとy_predはバッチごとの真のラベルと予測ラベルです
    intersection = tf.reduce_sum(tf.minimum(y_true, y_pred))
    union = tf.reduce_sum(tf.maximum(y_true, y_pred))
    jaccard = intersection / union
    return jaccard

def detect_dicom(file_path):
    # HDF5ファイルのパス
    hdf5_path = ''# モデルの読み込み時にカスタムオブジェクトを指定
    model = tf.keras.models.load_model(hdf5_path, custom_objects={'jaccard_coefficient': jaccard_coefficient})
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    # HDF5ファイルの読み込み
    with h5py.File(hdf5_path, 'r') as file:
        # モデルの構築
        # 例: weights = file['weights'][:]
        # ここで、HDF5ファイル内のデータを使用してモデルを構築します。

        # DICOM画像の前処理
        dicom_data = pydicom.dcmread(file_path)
        pixel_array = dicom_data.pixel_array
        processed_image = preprocess_image(pixel_array)

        # 推論
        output = probability_model.predict(processed_image)

    detect_dicom_label.config(text="AI:このファイルは: {}です。".format(output))

def preprocess_image(image):
    # 例: 画像のリサイズと正規化を行う
    image = resize_array_bilinear(image, 64, 64)
    image = (np.expand_dims(image,0))
    processed_image = image / np.max(image)  # 0から1の範囲に正規化
    return processed_image


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

root = tk.Tk()
root.title(u"Detecting Dicom Image Type")
root.geometry("400x300")

open_button = tk.Button(root, text="DICOMファイルを選択", command=open_dicom_file)
open_button.pack()

file_name_label = tk.Label(root, text="選択されたファイル: ")
file_name_label.pack()

ai_detecting = tk.Label(root, text="AI")
ai_detecting.pack()

detect_dicom_label = tk.Label(root, text="AI")
detect_dicom_label.pack()

root.mainloop()
