import tensorflow as tf
import numpy as np
import pydicom
from keras.models import load_model

def jaccard_coefficient(y_true, y_pred):
    # y_trueとy_predはバッチごとの真のラベルと予測ラベルです
    intersection = tf.reduce_sum(tf.minimum(y_true, y_pred))
    union = tf.reduce_sum(tf.maximum(y_true, y_pred))
    jaccard = intersection / union
    return jaccard

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

def preprocess_image(image):
    # 例: 画像のリサイズと正規化を行う
    image = np.array(image)
    image = resize_array_bilinear(image, 64, 64)
    image = (np.expand_dims(image,0))
    image /= np.max(image)  # 0から1の範囲に正規化
    return image

# HDF5ファイルのパス
hdf5_path = ''
model = load_model(hdf5_path, custom_objects={'jaccard_coefficient': jaccard_coefficient})

file_path=""

# DICOM画像の前処理
dicom_data = pydicom.dcmread(file_path)
image = dicom_data.pixel_array
image = preprocess_image(image)

# 推論
output = model.predict(image, verbose=1)

print(output)
