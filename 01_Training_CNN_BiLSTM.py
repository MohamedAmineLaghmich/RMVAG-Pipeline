# ==============================================================================
# MASTER SCRIPT 1: CNN-BiLSTM TRAINING (CSV to .h5 & .pkl)
# ==============================================================================
CROP_TYPE = "Cereales" # Change for "Agrumes", "Maraicheres", "Serres"

import pandas as pd
import numpy as np
import joblib
import gc
from google.colab import files
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical

configs = {
    "Agrumes": {"months": 6, "features": 7, "weight": 5.0, "bands": ['NDVI_01', 'NDWI_01', 'BSI_01', 'B5_01', 'VV_01', 'VH_01', 'VVVH_ratio_01', 'NDVI_03', 'NDWI_03', 'BSI_03', 'B5_03', 'VV_03', 'VH_03', 'VVVH_ratio_03', 'NDVI_05', 'NDWI_05', 'BSI_05', 'B5_05', 'VV_05', 'VH_05', 'VVVH_ratio_05', 'NDVI_07', 'NDWI_07', 'BSI_07', 'B5_07', 'VV_07', 'VH_07', 'VVVH_ratio_07', 'NDVI_09', 'NDWI_09', 'BSI_09', 'B5_09', 'VV_09', 'VH_09', 'VVVH_ratio_09', 'NDVI_11', 'NDWI_11', 'BSI_11', 'B5_11', 'VV_11', 'VH_11', 'VVVH_ratio_11']},
    "Cereales": {"months": 6, "features": 7, "weight": 5.0, "bands": ['NDVI_01', 'NDWI_01', 'BSI_01', 'B5_01', 'B11_01', 'VV_01', 'VH_01', 'NDVI_02', 'NDWI_02', 'BSI_02', 'B5_02', 'B11_02', 'VV_02', 'VH_02', 'NDVI_03', 'NDWI_03', 'BSI_03', 'B5_03', 'B11_03', 'VV_03', 'VH_03', 'NDVI_04', 'NDWI_04', 'BSI_04', 'B5_04', 'B11_04', 'VV_04', 'VH_04', 'NDVI_05', 'NDWI_05', 'BSI_05', 'B5_05', 'B11_05', 'VV_05', 'VH_05', 'NDVI_06', 'NDWI_06', 'BSI_06', 'B5_06', 'B11_06', 'VV_06', 'VH_06']},
    "Maraicheres": {"months": 9, "features": 7, "weight": 15.0, "bands": ['NDVI_01', 'NDWI_01', 'BSI_01', 'B5_01', 'B11_01', 'VV_01', 'VH_01', 'NDVI_02', 'NDWI_02', 'BSI_02', 'B5_02', 'B11_02', 'VV_02', 'VH_02', 'NDVI_03', 'NDWI_03', 'BSI_03', 'B5_03', 'B11_03', 'VV_03', 'VH_03', 'NDVI_04', 'NDWI_04', 'BSI_04', 'B5_04', 'B11_04', 'VV_04', 'VH_04', 'NDVI_05', 'NDWI_05', 'BSI_05', 'B5_05', 'B11_05', 'VV_05', 'VH_05', 'NDVI_06', 'NDWI_06', 'BSI_06', 'B5_06', 'B11_06', 'VV_06', 'VH_06', 'NDVI_07', 'NDWI_07', 'BSI_07', 'B5_07', 'B11_07', 'VV_07', 'VH_07', 'NDVI_08', 'NDWI_08', 'BSI_08', 'B5_08', 'B11_08', 'VV_08', 'VH_08', 'NDVI_09', 'NDWI_09', 'BSI_09', 'B5_09', 'B11_09', 'VV_09', 'VH_09']},
    "Serres": {"months": 3, "features": 6, "weight": 15.0, "bands": ['NDVI_01', 'PMLBI_01', 'BSI_01', 'NDWI_01', 'VV_01', 'VH_01', 'NDVI_02', 'PMLBI_02', 'BSI_02', 'NDWI_02', 'VV_02', 'VH_02', 'NDVI_03', 'PMLBI_03', 'BSI_03', 'NDWI_03', 'VV_03', 'VH_03']}
}

cfg = configs[CROP_TYPE]
n_months, n_features, strict_band_order = cfg["months"], cfg["features"], cfg["bands"]
class_weights = {0: 1.0, 1: cfg["weight"]}

K.clear_session()
print(f"\n📂 Please upload validation CSV for {CROP_TYPE}:")
df = pd.read_csv(next(iter(files.upload().keys())))
X_raw = df[strict_band_order].values 
y_raw = df['Class_ID'].values

scaler = StandardScaler()
X_2d_scaled = scaler.fit_transform(X_raw).astype(np.float32)
joblib.dump(scaler, f'Scaler_{CROP_TYPE}.pkl')

X_3d_scaled = X_2d_scaled.reshape((-1, n_months, n_features)) 
X_train, X_test, y_train, y_test = train_test_split(X_3d_scaled, to_categorical(y_raw, 2), test_size=0.2, random_state=42, stratify=y_raw)

model = Sequential([
    Input(shape=(n_months, n_features)),
    Conv1D(64, 2, activation='relu', padding='same'),
    *( [MaxPooling1D(2)] if n_months > 3 else [] ),
    Bidirectional(LSTM(128, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, class_weight=class_weights, verbose=1)

model.save(f'CNN_BiLSTM_{CROP_TYPE}.h5')
files.download(f'Scaler_{CROP_TYPE}.pkl')
files.download(f'CNN_BiLSTM_{CROP_TYPE}.h5')
