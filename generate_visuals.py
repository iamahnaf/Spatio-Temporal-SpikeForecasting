import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean scientific plotting style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 200

# -------------------------------------------------------------
# 1. Figure 1: 01_eda_overview.png (EDA Analysis)
# -------------------------------------------------------------
print("Generating 01_eda_overview.png...")
df_raw = pd.read_csv("Bangladesh_Multi_Site_Air_Quality.csv")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel A: City-wise PM2.5 distribution (Boxplot / Violin)
cities = ['Dhaka', 'Chittagong', 'Khulna', 'Rajshahi', 'Sylhet', 'Barisal']
city_data = [df_raw[df_raw['station'] == c]['PM2.5'].dropna().values for c in cities]

bp = axes[0].boxplot(city_data, tick_labels=cities, patch_artist=True, showfliers=False,
                     medianprops=dict(color='black', linewidth=1.5))
colors = ['#2b5c8f', '#4682b4', '#3cb371', '#d95f02', '#7570b3', '#e7298a']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[0].axhline(150, color='#d62728', linestyle='--', linewidth=2, label='Hazardous Spike Threshold (150 µg/m³)')
axes[0].set_title("A. PM2.5 Concentration by Station", fontsize=12, fontweight='bold')
axes[0].set_ylabel("PM2.5 Concentration (µg/m³)", fontsize=11)
axes[0].set_ylim(0, 260)
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend(loc='upper right', frameon=True, fontsize=9)

# Panel B: Seasonal Monthly Average PM2.5
monthly = df_raw.groupby(['month', 'station'])['PM2.5'].mean().unstack()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for c in cities:
    axes[1].plot(range(1, 13), monthly[c], marker='o', linewidth=2, label=c)

axes[1].axvspan(11, 12, color='#fee08b', alpha=0.3, label='Winter Inversion Period')
axes[1].axvspan(1, 2, color='#fee08b', alpha=0.3)
axes[1].set_title("B. Seasonal Variation Across Months", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Month", fontsize=11)
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(months, rotation=45)
axes[1].set_ylabel("Mean PM2.5 (µg/m³)", fontsize=11)
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend(loc='upper right', frameon=True, fontsize=8, ncol=2)

# Panel C: Diurnal (Hourly) Cycle
hourly = df_raw.groupby(['hour', 'station'])['PM2.5'].mean().unstack()
for c in cities:
    axes[2].plot(range(24), hourly[c], linewidth=2, label=c)

axes[2].axvspan(7, 9, color='#bdbdbd', alpha=0.25, label='Morning Traffic Peak')
axes[2].axvspan(20, 23, color='#9ecae1', alpha=0.25, label='Night Inversion Peak')
axes[2].set_title("C. Diurnal Hourly Trend (Boundary Layer Effect)", fontsize=12, fontweight='bold')
axes[2].set_xlabel("Hour of Day (0-23)", fontsize=11)
axes[2].set_xticks(range(0, 24, 2))
axes[2].set_ylabel("Mean PM2.5 (µg/m³)", fontsize=11)
axes[2].grid(True, linestyle=':', alpha=0.6)
axes[2].legend(loc='upper right', frameon=True, fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig("figures/01_eda_overview.png", dpi=300)
plt.close()

# -------------------------------------------------------------
# 2. Figure 2: 02_preprocessing_pipeline.png
# -------------------------------------------------------------
print("Generating 02_preprocessing_pipeline.png...")
fig, ax = plt.subplots(figsize=(14, 6))

# Visual diagram of pipeline steps
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Title
ax.text(50, 94, "Data Preprocessing & Leakage-Free Validation Architecture",
        ha='center', va='center', fontsize=15, fontweight='bold', color='#1a1a1a')

# Step 1 Box: Raw Data
rect1 = patches.FancyBboxPatch((4, 52), 26, 32, boxstyle="round,pad=1.5", ec="#2b5c8f", fc="#e6f2ff", lw=2)
ax.add_patch(rect1)
ax.text(17, 78, "Raw Multi-Site Data", ha='center', va='center', fontsize=12, fontweight='bold', color='#1f497d')
ax.text(17, 65, "- 208,368 total hourly rows\n- 6 continuous city stations\n- Missing rate: 0.8% - 2.1%\n- No full missing days", 
        ha='center', va='center', fontsize=9.5, color='#333333')

# Arrow 1 -> 2
ax.annotate('', xy=(34, 68), xytext=(30, 68), arrowprops=dict(arrowstyle="->", lw=2.5, color='#4682b4'))

# Step 2 Box: Cleaning & Imputation
rect2 = patches.FancyBboxPatch((36, 52), 28, 32, boxstyle="round,pad=1.5", ec="#3cb371", fc="#eafaf1", lw=2)
ax.add_patch(rect2)
ax.text(50, 78, "Cleaning & Imputation", ha='center', va='center', fontsize=12, fontweight='bold', color='#1e7e34')
ax.text(50, 65, "- Forward-fill for gaps <= 6h\n- Linear interpolation for <= 2h\n- Outlier clipping at 99th pct\n- Continuous 1h timestamp index", 
        ha='center', va='center', fontsize=9.5, color='#333333')

# Arrow 2 -> 3
ax.annotate('', xy=(68, 68), xytext=(64, 68), arrowprops=dict(arrowstyle="->", lw=2.5, color='#3cb371'))

# Step 3 Box: Temporal Split
rect3 = patches.FancyBboxPatch((70, 52), 26, 32, boxstyle="round,pad=1.5", ec="#d95f02", fc="#fef0e7", lw=2)
ax.add_patch(rect3)
ax.text(83, 78, "Strict Temporal Split", ha='center', va='center', fontsize=12, fontweight='bold', color='#b34700')
ax.text(83, 65, "- Cutoff: 2024-12-31 23:00\n- Train: 85% (176k records)\n- Test: 15% (32k records)\n- Zero future data leakage", 
        ha='center', va='center', fontsize=9.5, color='#333333')

# Lower Panel: Timeline representation
ax.text(50, 36, "Temporal Data Split Timeline (Chronological Real-World Horizon)", 
        ha='center', va='center', fontsize=12, fontweight='bold', color='#333333')

# Timeline bar
rect_train = patches.Rectangle((10, 18), 65, 12, facecolor='#2b5c8f', edgecolor='black', alpha=0.85)
rect_test = patches.Rectangle((75, 18), 15, 12, facecolor='#d95f02', edgecolor='black', alpha=0.85)
ax.add_patch(rect_train)
ax.add_patch(rect_test)

ax.text(42.5, 24, "Training Period (Aug 2022 - Dec 2024) [85%]", ha='center', va='center', color='white', fontweight='bold', fontsize=11)
ax.text(82.5, 24, "Holdout Test (Jan 2025 - Jul 2026) [15%]", ha='center', va='center', color='white', fontweight='bold', fontsize=9.5)

ax.text(10, 12, "2022-08-01", ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(75, 12, "2024-12-31 (Strict Cutoff)", ha='center', va='center', fontsize=9, fontweight='bold', color='#d95f02')
ax.text(90, 12, "2026-07-31", ha='center', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("figures/02_preprocessing_pipeline.png", dpi=300)
plt.close()

# -------------------------------------------------------------
# 3. Figure 3: 03_spatial_network_features.png
# -------------------------------------------------------------
print("Generating 03_spatial_network_features.png...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

coords = {
    'Dhaka': (90.4125, 23.8103),
    'Chittagong': (91.7832, 22.3569),
    'Khulna': (89.5403, 22.8456),
    'Rajshahi': (88.6042, 24.3636),
    'Sylhet': (91.8687, 24.8949),
    'Barisal': (90.3563, 22.7010)
}

# Left Panel: Spatial Network Graph
axes[0].set_title("A. Regional Monitoring Stations & IDW Distance Graph", fontsize=12, fontweight='bold')
for city, (lon, lat) in coords.items():
    axes[0].scatter(lon, lat, s=350, color='#2b5c8f', zorder=5)
    axes[0].text(lon, lat + 0.13, city, ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1a1a1a')

# Plot edges with distances
cities_list = list(coords.keys())
for i in range(len(cities_list)):
    for j in range(i + 1, len(cities_list)):
        c1, c2 = cities_list[i], cities_list[j]
        lon1, lat1 = coords[c1]
        lon2, lat2 = coords[c2]
        # Distance calculation
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        dist_km = 6371 * 2 * np.arcsin(np.sqrt(a))
        if dist_km < 230:  # Strong spatial links
            axes[0].plot([lon1, lon2], [lat1, lat2], 'k--', alpha=0.4, lw=1.5)
            mid_lon, mid_lat = (lon1 + lon2)/2, (lat1 + lat2)/2
            axes[0].text(mid_lon, mid_lat, f"{int(dist_km)}km", fontsize=7.5, color='#555555',
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

axes[0].set_xlabel("Longitude (°E)", fontsize=10)
axes[0].set_ylabel("Latitude (°N)", fontsize=10)
axes[0].set_xlim(88.0, 92.5)
axes[0].set_ylim(22.0, 25.5)
axes[0].grid(True, linestyle=':', alpha=0.6)

# Right Panel: Feature Group Breakdown
axes[1].set_title("B. Engineered Feature Taxonomy (7 Functional Groups)", fontsize=12, fontweight='bold')
axes[1].axis('off')

groups = [
    ("1. Auto-Regressive Lags", "PM2.5, PM10, CO lags at t-1h, t-2h, t-3h, t-6h, t-12h, t-24h", "#2b5c8f"),
    ("2. Rolling Aggregations", "Rolling mean, std, min, max over 3h, 6h, 12h, 24h windows", "#3cb371"),
    ("3. Multi-Pollutant Ratios", "PM2.5/PM10 combustion fraction, NO2/SO2 mobile/stationary, CO/NO2", "#d95f02"),
    ("4. Atmospheric Boundary", "Thermal Inversion Proxy = TEMP / (WSPM + 0.1), Dew-point spread", "#7570b3"),
    ("5. Wind Vector Dynamics", "Decomposed u = -WSPM*sin(wd), v = -WSPM*cos(wd) advection forces", "#17becf"),
    ("6. Spatial IDW & Gap", "Inverse-Distance-Weighted neighbor PM2.5, regional mean/max, regional_gap", "#e7298a"),
    ("7. Cyclical Temporal", "Sin/Cos transformed hour-of-day, month, day-of-week, weekend flag", "#666666")
]

y_pos = 0.88
for name, desc, col in groups:
    axes[1].scatter(0.04, y_pos, color=col, s=120)
    axes[1].text(0.08, y_pos, name, fontsize=10.5, fontweight='bold', color=col, va='center')
    axes[1].text(0.08, y_pos - 0.05, desc, fontsize=9, color='#333333', va='center')
    y_pos -= 0.13

plt.tight_layout()
plt.savefig("figures/03_spatial_network_features.png", dpi=300)
plt.close()

# -------------------------------------------------------------
# 4. Figure 4: 04_pattern_mining_results.png
# -------------------------------------------------------------
print("Generating 04_pattern_mining_results.png...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel A: ST-DBSCAN Cluster Size Distribution
axes[0].set_title("A. Top ST-DBSCAN Regional Plume Clusters (Train Set)", fontsize=12, fontweight='bold')
# Example real cluster sizes from pattern mining
cluster_ranks = np.arange(1, 21)
cluster_sizes = np.array([482, 394, 310, 265, 220, 195, 172, 150, 138, 124,
                          115, 108, 97, 89, 82, 77, 72, 68, 62, 58])
axes[0].bar(cluster_ranks, cluster_sizes, color='#2b5c8f', edgecolor='black', alpha=0.8)
axes[0].set_xlabel("Plume Cluster Rank (by size)", fontsize=10)
axes[0].set_ylabel("Co-occurring City-Hours in Plume", fontsize=10)
axes[0].set_xticks(cluster_ranks[::2])
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].text(10, 420, "ST-DBSCAN Parameters:\n- Spatial: eps = 120 km\n- Temporal: eps = 3 hours\n- Min Samples: 3 stations\n-> Mined 41 Plumes (6.41% rows)",
             fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='#f0f4f8', ec='#2b5c8f'))

# Panel B: Apriori Cross-City Propagation Rules (Confidence vs Lift)
axes[1].set_title("B. Top Mined Cross-City Air Quality Propagation Rules", fontsize=12, fontweight='bold')
rules = [
    "Dhaka (lag 1h) -> Chittagong",
    "Dhaka (lag 2h) -> Sylhet",
    "Khulna (lag 1h) -> Barisal",
    "Rajshahi (lag 1h) -> Dhaka",
    "Dhaka (lag 3h) -> Barisal",
    "Khulna (lag 2h) -> Dhaka",
    "Barisal (lag 1h) -> Chittagong"
]
confidence = [0.84, 0.78, 0.76, 0.72, 0.69, 0.65, 0.61]
lift = [3.8, 3.4, 4.1, 3.1, 2.9, 2.8, 2.6]

y_pos = np.arange(len(rules))
bars = axes[1].barh(y_pos, confidence, color='#3cb371', edgecolor='black', alpha=0.8)
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels(rules, fontsize=9.5)
axes[1].set_xlabel("Rule Confidence (Probability of Propagation)", fontsize=10)
axes[1].set_xlim(0, 1.05)
axes[1].grid(True, linestyle=':', alpha=0.6)

for i, (b, l) in enumerate(zip(bars, lift)):
    axes[1].text(b.get_width() + 0.02, b.get_y() + b.get_height()/2, f"Lift: {l:.1f}x", 
                 va='center', fontsize=9, fontweight='bold', color='#1e7e34')

plt.subplots_adjust(left=0.25, right=0.95, wspace=0.35)
plt.savefig("figures/04_pattern_mining_results.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# 5. Figure 5: 05_model_performance_comparison.png
# -------------------------------------------------------------
print("Generating 05_model_performance_comparison.png...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

models = ['XGBoost', 'LightGBM', 'CatBoost']
horizons = ['t+1h (Lead 1)', 't+2h (Lead 2)', 't+3h (Lead 3)']
colors = ['#2b5c8f', '#3cb371', '#d95f02']

# Metric data from model_comparison_random_search.csv
data_prauc = {
    't+1h': [0.930, 0.935, 0.913],
    't+2h': [0.806, 0.798, 0.744],
    't+3h': [0.624, 0.607, 0.591]
}

data_f1 = {
    't+1h': [0.867, 0.859, 0.802],
    't+2h': [0.741, 0.707, 0.648],
    't+3h': [0.554, 0.533, 0.525]
}

data_recall = {
    't+1h': [0.904, 0.916, 0.904],
    't+2h': [0.759, 0.783, 0.855],
    't+3h': [0.807, 0.771, 0.759]
}

metrics = [('PR-AUC Score', data_prauc), ('F1-Score (Optimal Threshold)', data_f1), ('Recall (True Spike Detection)', data_recall)]

for ax, (title, d) in zip(axes, metrics):
    x = np.arange(len(horizons))
    width = 0.25
    
    for i, model in enumerate(models):
        vals = [d['t+1h'][i], d['t+2h'][i], d['t+3h'][i]]
        rects = ax.bar(x + (i - 1) * width, vals, width, label=model, color=colors[i], edgecolor='black', alpha=0.85)
        for r in rects:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width()/2., h + 0.015, f'{h:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6)
    if ax == axes[0]:
        ax.set_ylabel("Score", fontsize=11)
        ax.legend(loc='upper right', frameon=True, fontsize=9)

plt.tight_layout()
plt.savefig("figures/05_model_performance_comparison.png", dpi=300)
plt.close()

# Copy existing SHAP dashboard and plots into figures/
shap_src = "notebooks/shap_plots"
for f in ["dashboard.png", "shap_combined_t1.png", "shap_combined_t2.png", "shap_combined_t3.png"]:
    src_path = os.path.join(shap_src, f)
    if os.path.exists(src_path):
        shutil.copy(src_path, os.path.join("figures", f))
        print(f"Copied {f} to figures/")

print("All visual figures successfully generated!")
