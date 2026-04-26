# ==============================================================================
# INFERENCE SCRIPT: AGRUMES MAP GENERATION
# Architecture: 6 Months x 7 Features = 42 Bands
# ==============================================================================
import numpy as np, rasterio, joblib, os, glob, gc
from rasterio.merge import merge
from tensorflow.keras.models import load_model
from google.colab import files, drive

CROP_TYPE, N_MONTHS, N_FEATURES, TOTAL_BANDS = "Agrumes", 6, 7, 42
DRIVE_FOLDER = "ORMVAG2025"

drive.mount('/content/drive', force_remount=True)
uploaded = files.upload()
scaler = joblib.load([f for f in uploaded.keys() if f.endswith('.pkl')][0])
model = load_model([f for f in uploaded.keys() if f.endswith('.h5')][0])

input_tiles = glob.glob(f'/content/drive/MyDrive/**/{DRIVE_FOLDER}/Agrumes_Stack*.tif', recursive=True)
local_out = '/content/predicted_tiles/'
os.makedirs(local_out, exist_ok=True)

for i, tile_path in enumerate(input_tiles):
    out_path = os.path.join(local_out, f'Pred_{i}.tif')
    with rasterio.open(tile_path) as src:
        meta = src.meta.copy()
        meta.update({'count': 1, 'dtype': 'uint8', 'nodata': 255})
        with rasterio.open(out_path, 'w', **meta) as dst:
            for _, window in src.block_windows(1):
                win_data = src.read(list(range(1, TOTAL_BANDS + 1)), window=window)
                win_flat = win_data.reshape(TOTAL_BANDS, -1).T
                mask = np.any((win_flat != 0) & (win_flat != -32768) & (win_flat != 255), axis=1)
                valid = win_flat[mask]
                out_flat = np.full(win_data.shape[1] * win_data.shape[2], 255, dtype=np.uint8)
                if valid.shape[0] > 0:
                    reshaped = scaler.transform(valid).astype(np.float32).reshape((-1, N_MONTHS, N_FEATURES))
                    preds = (model.predict(reshaped, batch_size=8192, verbose=0)[:, 1] >= 0.90).astype(np.uint8)
                    out_flat[mask] = preds
                dst.write(out_flat.reshape((win_data.shape[1], win_data.shape[2])), 1, window=window)

srcs = [rasterio.open(fp) for fp in glob.glob(os.path.join(local_out, '*.tif'))]
mosaic, out_trans = merge(srcs)
out_meta = srcs[0].meta.copy()
out_meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans, "compress": "lzw"})

final_path = f'/content/drive/MyDrive/{DRIVE_FOLDER}/Final_Map_{CROP_TYPE}.tif'
with rasterio.open(final_path, "w", **out_meta) as dest: dest.write(mosaic)
files.download(final_path)
