# ==============================================================================
# INFERENCE SCRIPT: CEREALES — MASK NDVI
# ==============================================================================
import numpy as np, rasterio, joblib, os, glob, gc
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from tensorflow.keras.models import load_model
from google.colab import files, drive

CROP_TYPE, N_MONTHS, N_FEATURES, TOTAL_BANDS, THRESHOLD = "Cereales", 6, 7, 42, 0.90
DRIVE_FOLDER = "ORMVAG2025"

drive.mount('/content/drive', force_remount=True)
uploaded = files.upload()
scaler = joblib.load([f for f in uploaded if f.endswith('.pkl')][0])
model = load_model([f for f in uploaded if f.endswith('.h5')][0])

print("Upload Summer NDVI Mask (.tif):")
summer_mask_path = list(files.upload().keys())[0]

input_tiles = glob.glob(f'/content/drive/MyDrive/**/{DRIVE_FOLDER}/Cereales_Stack*.tif', recursive=True)
local_dir = '/content/predicted_tiles_90/'
os.makedirs(local_dir, exist_ok=True)

for i, tile_path in enumerate(input_tiles):
    out_path = os.path.join(local_dir, f'Tile_{i}.tif')
    with rasterio.open(tile_path) as src:
        meta = src.meta.copy()
        meta.update({'count': 1, 'dtype': 'uint8', 'nodata': 255})
        with rasterio.open(summer_mask_path) as src_raw:
            with WarpedVRT(src_raw, crs=src.crs, transform=src.transform, width=src.width, height=src.height) as src_summer:
                with rasterio.open(out_path, 'w', **meta) as dst:
                    for _, window in src.block_windows(1):
                        win_data = src.read(list(range(1, TOTAL_BANDS+1)), window=window)
                        win_flat = win_data.reshape(TOTAL_BANDS, -1).T
                        summer_flat = src_summer.read(1, window=window).flatten()
                        
                        mask = np.any((win_flat != 0) & (win_flat != -32768) & (win_flat != 255), axis=1)
                        valid, valid_summer = win_flat[mask], summer_flat[mask]
                        out_flat = np.full(win_data.shape[1] * win_data.shape[2], 255, dtype=np.uint8)

                        if valid.shape[0] > 0:
                            reshaped = scaler.transform(valid).astype(np.float32).reshape((-1, N_MONTHS, N_FEATURES))
                            probs = model.predict(reshaped, batch_size=8192, verbose=0)
                            preds = (probs[:, 1] >= THRESHOLD).astype(np.uint8)

                            preds[np.mean(reshaped[:, :, 1], axis=1) > 0] = 0 # NDWI Water Mask
                            nd_val = src_summer.nodata if src_summer.nodata else 65535
                            preds[(valid_summer > 300) & (valid_summer < 10000) & (valid_summer != nd_val)] = 0 # Summer NDVI
                            
                            out_flat[mask] = preds
                        dst.write(out_flat.reshape((win_data.shape[1], win_data.shape[2])), 1, window=window)

srcs = [rasterio.open(fp) for fp in glob.glob(os.path.join(local_dir, '*.tif'))]
mosaic, out_trans = merge(srcs)
out_meta = srcs[0].meta.copy()
out_meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans, "compress": "lzw"})
final_path = f'/content/drive/MyDrive/{DRIVE_FOLDER}/Final_Map_{CROP_TYPE}_90pct.tif'
with rasterio.open(final_path, "w", **out_meta) as dst: dst.write(mosaic)
files.download(final_path)
