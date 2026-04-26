# ==============================================================================
# SCRIPT DATAVIZ PRO : FIGURES POUR ARTICLE SCIENTIFIQUE
# ==============================================================================
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from google.colab import files

df = pd.DataFrame({
    'Secteurs': ['Souk Larbaâ', 'Sidi Allal Tazi', 'Mechraâ Bel Ksiri', 'Sidi Kacem', 'Sidi Slimane'],
    'Agrumes': [2611.11, 9149.04, 8164.14, 8968.84, 9178.20],
    'Céréales': [8059.38, 6826.83, 26630.75, 25444.49, 13214.89],
    'Cultures Maraîchères': [8516.30, 22090.86, 6107.05, 15724.80, 6460.22],
    'Serres': [1138.83, 3190.89, 72.03, 233.61, 582.51]
})
colors, cats = ['#F39C12', '#FCE698', '#27AE60', '#8E44AD'], ['Agrumes', 'Céréales', 'Cultures Maraîchères', 'Serres']
plt.style.use('seaborn-v0_8-whitegrid'); plt.rcParams.update({'font.family': 'serif', 'font.size': 12})

fig, ax = plt.subplots(figsize=(10, 6))
df_pct = df.set_index('Secteurs').div(df.set_index('Secteurs').sum(axis=1), axis=0) * 100
bottom = np.zeros(len(df_pct))

for i, cat in enumerate(cats):
    ax.bar(df_pct.index, df_pct[cat], bottom=bottom, label=cat, color=colors[i], edgecolor='black', width=0.6)
    for j, val in enumerate(df_pct[cat]):
        if val > 5: ax.text(j, bottom[j] + val/2, f'{val:.0f}%', ha='center', va='center', fontweight='bold')
    bottom += df_pct[cat]

ax.set_ylabel('Proportion (%)', fontweight='bold'); ax.set_ylim(0, 100)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
fig.savefig('Barres_100pct.jpg', dpi=300, bbox_inches='tight')
files.download('Barres_100pct.jpg')
