# ==============================================================================
# INDEPENDENT MAP VALIDATION SCRIPT
# ==============================================================================
import pandas as pd, json, rasterio, numpy as np
from pyproj import Transformer
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from google.colab import files

crop_types = ["Agrumes", "Cereales", "Maraicheres", "Serres"]

for crop in crop_types:
    print(f"\n📂 UPLOAD Map (.tif) AND Validation (.csv) FOR {crop}")
    uploaded = files.upload()
    csv_file = [f for f in uploaded.keys() if f.endswith('.csv')][0]
    tif_file = [f for f in uploaded.keys() if f.endswith('.tif')][0]

    df = pd.read_csv(csv_file)
    def extract_coords(geo_string):
        try: return json.loads(geo_string)['coordinates'][0], json.loads(geo_string)['coordinates'][1]
        except: return None, None
    df['Lon'], df['Lat'] = zip(*df['.geo'].apply(extract_coords))
    df = df.dropna(subset=['Lon', 'Lat'])

    predicted_classes, true_classes = [], df['Class_ID'].values
    with rasterio.open(tif_file) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        df['X'], df['Y'] = transformer.transform(df['Lon'].values, df['Lat'].values)
        for val in src.sample([(x, y) for x, y in zip(df['X'], df['Y'])]):
            predicted_classes.append(val[0])

    valid_idx = [i for i, p in enumerate(predicted_classes) if p != src.nodata and p in [0, 1]]
    y_true, y_pred = np.array(true_classes)[valid_idx], np.array(predicted_classes)[valid_idx]

    print(f"🌍 KAPPA SCORE: {cohen_kappa_score(y_true, y_pred):.4f}")
    print(f"🧮 CONFUSION MATRIX:\n{confusion_matrix(y_true, y_pred)}")
    print(f"📊 REPORT:\n{classification_report(y_true, y_pred)}")
