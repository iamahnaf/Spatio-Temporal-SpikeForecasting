import os
from PIL import Image, ImageOps, ImageDraw, ImageFont

os.makedirs('figures/collages', exist_ok=True)

def add_header(img, title, font_size=28):
    """Add a clean, high-contrast banner header above an image."""
    banner_height = 48
    new_img = Image.new('RGB', (img.width, img.height + banner_height), color=(248, 249, 250))
    draw = ImageDraw.Draw(new_img)
    # Draw title text (use default font if custom font not available)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text((20, 10), title, fill=(30, 41, 59), font=font)
    draw.line([(0, banner_height - 1), (img.width, banner_height - 1)], fill=(203, 213, 225), width=2)
    # Paste original image
    new_img.paste(img, (0, banner_height))
    return new_img

def pad_to_height(img, target_height, bg_color=(255, 255, 255)):
    if img.height >= target_height:
        return img
    new_img = Image.new('RGB', (img.width, target_height), color=bg_color)
    y_offset = (target_height - img.height) // 2
    new_img.paste(img, (0, y_offset))
    return new_img

# -------------------------------------------------------------
# Collage 01: Exploratory Data Analysis (01_eda.ipynb)
# -------------------------------------------------------------
print("Generating Collage 01 (EDA)...")
im_dist_10 = Image.open('figures/notebook_images/01_cell8.png').convert('RGB')
im_city_box = Image.open('figures/notebook_images/01_cell10.png').convert('RGB')
im_diurnal = Image.open('figures/notebook_images/01_cell12.png').convert('RGB')
im_seasonal = Image.open('figures/notebook_images/01_cell14.png').convert('RGB')
im_corr = Image.open('figures/notebook_images/01_cell16.png').convert('RGB')
im_wind_freq = Image.open('figures/notebook_images/01_cell18.png').convert('RGB')
im_spikes = Image.open('figures/notebook_images/01_cell20.png').convert('RGB')
im_autocorr = Image.open('figures/notebook_images/01_cell22.png').convert('RGB')

# Row 1: Correlation Matrix (left) and PM2.5 by City (right)
# Target width per row = 2200 px
h1 = 700
c1_left = im_corr.resize((int(im_corr.width * (h1 / im_corr.height)), h1), Image.Resampling.LANCZOS)
c1_right = im_city_box.resize((int(im_city_box.width * (h1 / im_city_box.height)), h1), Image.Resampling.LANCZOS)
row1_w = c1_left.width + c1_right.width + 30
row1 = Image.new('RGB', (row1_w, h1), (255, 255, 255))
row1.paste(c1_left, (0, 0))
row1.paste(c1_right, (c1_left.width + 30, 0))

# Row 2: Diurnal (left) and Seasonal (right)
h2 = 500
c2_left = im_diurnal.resize((int(im_diurnal.width * (h2 / im_diurnal.height)), h2), Image.Resampling.LANCZOS)
c2_right = im_seasonal.resize((int(im_seasonal.width * (h2 / im_seasonal.height)), h2), Image.Resampling.LANCZOS)
row2_w = c2_left.width + c2_right.width + 30
row2 = Image.new('RGB', (row2_w, h2), (255, 255, 255))
row2.paste(c2_left, (0, 0))
row2.paste(c2_right, (c2_left.width + 30, 0))

# Row 3: Daily Spikes Timeline (left) and Autocorrelation (right)
h3 = 520
c3_left = im_spikes.resize((int(im_spikes.width * (h3 / im_spikes.height)), h3), Image.Resampling.LANCZOS)
c3_right = im_autocorr.resize((int(im_autocorr.width * (h3 / im_autocorr.height)), h3), Image.Resampling.LANCZOS)
row3_w = c3_left.width + c3_right.width + 30
row3 = Image.new('RGB', (row3_w, h3), (255, 255, 255))
row3.paste(c3_left, (0, 0))
row3.paste(c3_right, (c3_left.width + 30, 0))

# Row 4: Univariate Distributions (full width)
w4 = max(row1_w, row2_w, row3_w)
h4 = int(im_dist_10.height * (w4 / im_dist_10.width))
row4 = im_dist_10.resize((w4, h4), Image.Resampling.LANCZOS)

# Resize all rows to match max width w4
total_w = w4
rows = []
for r in [row1, row2, row3, row4]:
    if r.width != total_w:
        scale = total_w / r.width
        r_resized = r.resize((total_w, int(r.height * scale)), Image.Resampling.LANCZOS)
        rows.append(r_resized)
    else:
        rows.append(r)

total_h = sum(r.height for r in rows) + (len(rows) - 1) * 30 + 40
collage1 = Image.new('RGB', (total_w, total_h), (255, 255, 255))
y_cursor = 20
for r in rows:
    collage1.paste(r, (0, y_cursor))
    y_cursor += r.height + 30

collage1.save('figures/collages/collage_01_eda.png', quality=95)
print(f"Saved Collage 01: size {collage1.size}")

# -------------------------------------------------------------
# Collage 02: Preprocessing (02_preprocessing.ipynb)
# -------------------------------------------------------------
print("Generating Collage 02 (Preprocessing)...")
im_prep = Image.open('figures/02_preprocessing_pipeline.png').convert('RGB')
im_prep.save('figures/collages/collage_02_preprocessing.png', quality=95)
print(f"Saved Collage 02: size {im_prep.size}")

# -------------------------------------------------------------
# Collage 03: Feature Engineering (03_feature_engineering.ipynb)
# -------------------------------------------------------------
print("Generating Collage 03 (Feature Engineering)...")
im_dist = Image.open('figures/notebook_images/03_cell2.png').convert('RGB')
im_wind = Image.open('figures/notebook_images/03_cell9.png').convert('RGB')
im_idw = Image.open('figures/notebook_images/03_cell14.png').convert('RGB')

# Make each plot large, clear, and visible!
# Let's arrange as:
# Left: Distance Matrix (h = 800)
# Right: Wind Vector (top) and IDW Correlation (bottom)
target_h = 900
im_dist_large = im_dist.resize((int(im_dist.width * (target_h / im_dist.height)), target_h), Image.Resampling.LANCZOS)

h_half = (target_h - 20) // 2
im_wind_scaled = im_wind.resize((int(im_wind.width * (h_half / im_wind.height)), h_half), Image.Resampling.LANCZOS)
im_idw_scaled = im_idw.resize((int(im_idw.width * (h_half / im_idw.height)), h_half), Image.Resampling.LANCZOS)

right_w = max(im_wind_scaled.width, im_idw_scaled.width)
im_wind_padded = pad_to_height(im_wind_scaled, h_half)
im_idw_padded = pad_to_height(im_idw_scaled, h_half)

right_col = Image.new('RGB', (right_w, target_h), (255, 255, 255))
right_col.paste(im_wind_padded, ((right_w - im_wind_padded.width) // 2, 0))
right_col.paste(im_idw_padded, ((right_w - im_idw_padded.width) // 2, h_half + 20))

collage3_w = im_dist_large.width + right_w + 40
collage3 = Image.new('RGB', (collage3_w, target_h), (255, 255, 255))
collage3.paste(im_dist_large, (0, 0))
collage3.paste(right_col, (im_dist_large.width + 40, 0))

collage3.save('figures/collages/collage_03_feature_engineering.png', quality=95)
print(f"Saved Collage 03: size {collage3.size}")

# -------------------------------------------------------------
# Collage 04: Pattern Mining (04_pattern_mining.ipynb)
# -------------------------------------------------------------
print("Generating Collage 04 (Pattern Mining)...")
im_st_dbscan = Image.open('figures/notebook_images/04_cell7.png').convert('RGB')
im_apriori = Image.open('figures/notebook_images/04_cell11.png').convert('RGB')

# Generous size, side-by-side with equal height
target_h4 = 750
im_st_scaled = im_st_dbscan.resize((int(im_st_dbscan.width * (target_h4 / im_st_dbscan.height)), target_h4), Image.Resampling.LANCZOS)
im_ap_scaled = im_apriori.resize((int(im_apriori.width * (target_h4 / im_apriori.height)), target_h4), Image.Resampling.LANCZOS)

collage4_w = im_st_scaled.width + im_ap_scaled.width + 40
collage4 = Image.new('RGB', (collage4_w, target_h4), (255, 255, 255))
collage4.paste(im_st_scaled, (0, 0))
collage4.paste(im_ap_scaled, (im_st_scaled.width + 40, 0))

collage4.save('figures/collages/collage_04_pattern_mining.png', quality=95)
print(f"Saved Collage 04: size {collage4.size}")

# -------------------------------------------------------------
# Collage 05: Predictive Modeling & SHAP (05_modeling.ipynb)
# -------------------------------------------------------------
print("Generating Collage 05 (Modeling & SHAP)...")
# Cell 21 (Confusion Matrices), Cell 23 (ROC), Cell 25 (PR), Cell 43 (Performance Dashboard), Cell 31 (SHAP t+1)
im_cm = Image.open('figures/notebook_images/05_cell21.png').convert('RGB')
im_roc = Image.open('figures/notebook_images/05_cell23.png').convert('RGB')
im_pr = Image.open('figures/notebook_images/05_cell25.png').convert('RGB')
im_curves = Image.open('figures/notebook_images/05_cell27.png').convert('RGB')
im_dashboard = Image.open('figures/notebook_images/05_cell43.png').convert('RGB')
im_shap = Image.open('figures/notebook_images/05_cell31.png').convert('RGB')

# Top Row: ROC Curve and Precision-Recall Curve side-by-side
h5_top = 650
im_roc_s = im_roc.resize((int(im_roc.width * (h5_top / im_roc.height)), h5_top), Image.Resampling.LANCZOS)
im_pr_s = im_pr.resize((int(im_pr.width * (h5_top / im_pr.height)), h5_top), Image.Resampling.LANCZOS)
top_w = im_roc_s.width + im_pr_s.width + 30
top_row = Image.new('RGB', (top_w, h5_top), (255, 255, 255))
top_row.paste(im_roc_s, (0, 0))
top_row.paste(im_pr_s, (im_roc_s.width + 30, 0))

# Middle Row: Confusion Matrices (t+1, t+2, t+3)
scale_mid = top_w / im_cm.width
mid_row = im_cm.resize((top_w, int(im_cm.height * scale_mid)), Image.Resampling.LANCZOS)

# Bottom Row: Model Performance Dashboard
scale_dash = top_w / im_dashboard.width
bot_row = im_dashboard.resize((top_w, int(im_dashboard.height * scale_dash)), Image.Resampling.LANCZOS)

collage5_h = top_row.height + mid_row.height + bot_row.height + 60
collage5 = Image.new('RGB', (top_w, collage5_h), (255, 255, 255))
collage5.paste(top_row, (0, 0))
collage5.paste(mid_row, (0, top_row.height + 30))
collage5.paste(bot_row, (0, top_row.height + mid_row.height + 60))

collage5.save('figures/collages/collage_05_modeling.png', quality=95)
print(f"Saved Collage 05: size {collage5.size}")
