import os
from PIL import Image, ImageOps, ImageDraw, ImageFont

os.makedirs('figures/collages', exist_ok=True)

def load_rgb(path):
    return Image.open(path).convert('RGB')

# -------------------------------------------------------------------------
# Collage 01: 01_eda.ipynb (8 plots)
# -------------------------------------------------------------------------
print("Creating Collage 01 (EDA)...")
im_dist_10 = load_rgb('figures/notebook_images/01_cell8.png')
im_city_box = load_rgb('figures/notebook_images/01_cell10.png')
im_diurnal  = load_rgb('figures/notebook_images/01_cell12.png')
im_seasonal = load_rgb('figures/notebook_images/01_cell14.png')
im_corr     = load_rgb('figures/notebook_images/01_cell16.png')
im_wind_dir = load_rgb('figures/notebook_images/01_cell18.png')
im_spikes   = load_rgb('figures/notebook_images/01_cell20.png')
im_autocorr = load_rgb('figures/notebook_images/01_cell22.png')

# Layout of Collage 01:
# Target width = 2600 px
W = 2600

# Row 1: Correlation Matrix (left, ~1300) + City Boxplot (right, ~1300)
h_r1 = 900
w1_l = int(im_corr.width * (h_r1 / im_corr.height))
w1_r = int(im_city_box.width * (h_r1 / im_city_box.height))
r1_scale = (W - 30) / (w1_l + w1_r)
w1_l, w1_r = int(w1_l * r1_scale), int(w1_r * r1_scale)
h_r1 = int(h_r1 * r1_scale)

p1_l = im_corr.resize((w1_l, h_r1), Image.Resampling.LANCZOS)
p1_r = im_city_box.resize((w1_r, h_r1), Image.Resampling.LANCZOS)

row1 = Image.new('RGB', (W, h_r1), (255, 255, 255))
row1.paste(p1_l, (0, 0))
row1.paste(p1_r, (w1_l + 30, 0))

# Row 2: Diurnal Cycle (left, ~1300) + Seasonal Cycle (right, ~1300)
h_r2 = 650
w2_l = int(im_diurnal.width * (h_r2 / im_diurnal.height))
w2_r = int(im_seasonal.width * (h_r2 / im_seasonal.height))
r2_scale = (W - 30) / (w2_l + w2_r)
w2_l, w2_r = int(w2_l * r2_scale), int(w2_r * r2_scale)
h_r2 = int(h_r2 * r2_scale)

p2_l = im_diurnal.resize((w2_l, h_r2), Image.Resampling.LANCZOS)
p2_r = im_seasonal.resize((w2_r, h_r2), Image.Resampling.LANCZOS)

row2 = Image.new('RGB', (W, h_r2), (255, 255, 255))
row2.paste(p2_l, (0, 0))
row2.paste(p2_r, (w2_l + 30, 0))

# Row 3: Daily Spikes Timeline (left) + Autocorrelation (right)
h_r3 = 650
w3_l = int(im_spikes.width * (h_r3 / im_spikes.height))
w3_r = int(im_autocorr.width * (h_r3 / im_autocorr.height))
r3_scale = (W - 30) / (w3_l + w3_r)
w3_l, w3_r = int(w3_l * r3_scale), int(w3_r * r3_scale)
h_r3 = int(h_r3 * r3_scale)

p3_l = im_spikes.resize((w3_l, h_r3), Image.Resampling.LANCZOS)
p3_r = im_autocorr.resize((w3_r, h_r3), Image.Resampling.LANCZOS)

row3 = Image.new('RGB', (W, h_r3), (255, 255, 255))
row3.paste(p3_l, (0, 0))
row3.paste(p3_r, (w3_l + 30, 0))

# Row 4: Univariate 10-Feature Distributions (full width W)
h_r4 = int(im_dist_10.height * (W / im_dist_10.width))
row4 = im_dist_10.resize((W, h_r4), Image.Resampling.LANCZOS)

# Assemble Collage 01
total_h1 = row1.height + row2.height + row3.height + row4.height + 90
collage1 = Image.new('RGB', (W, total_h1), (255, 255, 255))
y = 0
for r in [row1, row2, row3, row4]:
    collage1.paste(r, (0, y))
    y += r.height + 30

collage1.save('figures/collages/collage_01_eda.png', quality=95)
print(f"Collage 01 created: {collage1.size}")

# -------------------------------------------------------------------------
# Collage 02: 02_preprocessing.ipynb
# -------------------------------------------------------------------------
print("Creating Collage 02 (Preprocessing)...")
im_prep = load_rgb('figures/02_preprocessing_pipeline.png')
# High resolution copy
im_prep.save('figures/collages/collage_02_preprocessing.png', quality=95)
print(f"Collage 02 created: {im_prep.size}")

# -------------------------------------------------------------------------
# Collage 03: 03_feature_engineering.ipynb (3 plots, equal prominence)
# -------------------------------------------------------------------------
print("Creating Collage 03 (Feature Engineering)...")
im_dist = load_rgb('figures/notebook_images/03_cell2.png')
im_wind = load_rgb('figures/notebook_images/03_cell9.png')
im_idw  = load_rgb('figures/notebook_images/03_cell14.png')

# We have 3 plots:
# 1. Distance between cities (690 x 590)
# 2. Wind vector decomposition (575 x 590)
# 3. Spatial correlation check (590 x 490)
# Let's arrange them side-by-side with uniform height H = 850 px
H3 = 850
w_dist = int(im_dist.width * (H3 / im_dist.height))
w_wind = int(im_wind.width * (H3 / im_wind.height))
w_idw  = int(im_idw.width * (H3 / im_idw.height))

p3_dist = im_dist.resize((w_dist, H3), Image.Resampling.LANCZOS)
p3_wind = im_wind.resize((w_wind, H3), Image.Resampling.LANCZOS)
p3_idw  = im_idw.resize((w_idw, H3), Image.Resampling.LANCZOS)

total_w3 = w_dist + w_wind + w_idw + 60
collage3 = Image.new('RGB', (total_w3, H3), (255, 255, 255))
collage3.paste(p3_dist, (0, 0))
collage3.paste(p3_wind, (w_dist + 30, 0))
collage3.paste(p3_idw, (w_dist + w_wind + 60, 0))

collage3.save('figures/collages/collage_03_feature_engineering.png', quality=95)
print(f"Collage 03 created: {collage3.size}")

# -------------------------------------------------------------------------
# Collage 04: 04_pattern_mining.ipynb (2 plots)
# -------------------------------------------------------------------------
print("Creating Collage 04 (Pattern Mining)...")
im_st = load_rgb('figures/notebook_images/04_cell7.png')
im_ap = load_rgb('figures/notebook_images/04_cell11.png')

# Side-by-side with uniform height H = 850 px
H4 = 850
w_st = int(im_st.width * (H4 / im_st.height))
w_ap = int(im_ap.width * (H4 / im_ap.height))

p4_st = im_st.resize((w_st, H4), Image.Resampling.LANCZOS)
p4_ap = im_ap.resize((w_ap, H4), Image.Resampling.LANCZOS)

total_w4 = w_st + w_ap + 40
collage4 = Image.new('RGB', (total_w4, H4), (255, 255, 255))
collage4.paste(p4_st, (0, 0))
collage4.paste(p4_ap, (w_st + 40, 0))

collage4.save('figures/collages/collage_04_pattern_mining.png', quality=95)
print(f"Collage 04 created: {collage4.size}")

# -------------------------------------------------------------------------
# Collage 05: 05_modeling.ipynb (Multi-panel modeling & evaluation)
# -------------------------------------------------------------------------
print("Creating Collage 05 (Modeling & SHAP)...")
im_cm   = load_rgb('figures/notebook_images/05_cell21.png')
im_roc  = load_rgb('figures/notebook_images/05_cell23.png')
im_pr   = load_rgb('figures/notebook_images/05_cell25.png')
im_dash = load_rgb('figures/notebook_images/05_cell43.png')
im_shap = load_rgb('figures/notebook_images/05_cell31.png')

# Target width = 2400 px
W5 = 2400

# Row 1: ROC Curve (left) + PR Curve (right)
h5_r1 = 800
w5_roc = int(im_roc.width * (h5_r1 / im_roc.height))
w5_pr  = int(im_pr.width * (h5_r1 / im_pr.height))
r5_scale = (W5 - 30) / (w5_roc + w5_pr)
w5_roc, w5_pr = int(w5_roc * r5_scale), int(w5_pr * r5_scale)
h5_r1 = int(h5_r1 * r5_scale)

p5_roc = im_roc.resize((w5_roc, h5_r1), Image.Resampling.LANCZOS)
p5_pr  = im_pr.resize((w5_pr, h5_r1), Image.Resampling.LANCZOS)

row5_1 = Image.new('RGB', (W5, h5_r1), (255, 255, 255))
row5_1.paste(p5_roc, (0, 0))
row5_1.paste(p5_pr, (w5_roc + 30, 0))

# Row 2: Confusion Matrices (t+1, t+2, t+3) scaled to W5
h5_r2 = int(im_cm.height * (W5 / im_cm.width))
row5_2 = im_cm.resize((W5, h5_r2), Image.Resampling.LANCZOS)

# Row 3: Model Performance Dashboard scaled to W5
h5_r3 = int(im_dash.height * (W5 / im_dash.width))
row5_3 = im_dash.resize((W5, h5_r3), Image.Resampling.LANCZOS)

total_h5 = row5_1.height + row5_2.height + row5_3.height + 60
collage5 = Image.new('RGB', (W5, total_h5), (255, 255, 255))
y = 0
for r in [row5_1, row5_2, row5_3]:
    collage5.paste(r, (0, y))
    y += r.height + 30

collage5.save('figures/collages/collage_05_modeling.png', quality=95)
print(f"Collage 05 created: {collage5.size}")
print("All 5 collages successfully generated!")
