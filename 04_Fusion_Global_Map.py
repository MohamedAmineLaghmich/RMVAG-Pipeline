# ==============================================================================
# SCRIPT: SMART FUSION OF 4 CROP MAPS (Global Master Map)
# ==============================================================================
import rasterio, numpy as np, os
from google.colab import files

print("📂 UPLOAD ALL 4 MAPS (.tif)")
uploaded_files = files.upload()
maps = {"Agrumes": None, "Cereales": None, "Maraicheres": None, "Serres": None}
for f in uploaded_files.keys():
    for k in maps.keys():
        if k.lower()[:4] in f.lower(): maps[k] = f

with rasterio.open(maps["Agrumes"]) as src:
    meta = src.meta.copy()
    base_data = src.read(1)
    global_map = np.full((src.height, src.width), 255, dtype=np.uint8) 
    global_map[base_data != 255] = 0 # 0 = Autre

print("🧩 Merging maps...")
with rasterio.open(maps["Cereales"]) as src: global_map = np.where(src.read(1) == 1, 4, global_map)
with rasterio.open(maps["Maraicheres"]) as src: global_map = np.where(src.read(1) == 1, 3, global_map)
with rasterio.open(maps["Serres"]) as src: global_map = np.where(src.read(1) == 1, 2, global_map)
with rasterio.open(maps["Agrumes"]) as src: global_map = np.where(src.read(1) == 1, 1, global_map)

meta.update({'dtype': 'uint8', 'nodata': 255, 'compress': 'lzw'})
with rasterio.open('Global_Master_Map_ORMVAG.tif', 'w', **meta) as dst: dst.write(global_map, 1)
files.download('Global_Master_Map_ORMVAG.tif')
