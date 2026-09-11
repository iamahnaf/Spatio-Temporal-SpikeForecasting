"""
04_pattern_mining.py
Implements the two "pattern discovery" pieces from the proposal:
  1. ST-DBSCAN  -> finds spatio-temporal clusters ("plumes") of elevated
                   pollution across the 6 cities.
  2. Apriori    -> mines association rules like
                   "Dhaka elevated at t -> Chittagong elevated at t+2"
                   for cross-city pollutant propagation.

Both outputs are turned into extra engineered features merged onto
train_features.csv / test_features.csv (replacing the placeholder
'regional_elevated_frac' proxy from 03_feature_engineering.py - that
proxy feature is left in place too, these are additive).

Run: python 04_pattern_mining.py
Output: train_features_v2.csv, test_features_v2.csv
"""

import pandas as pd
import numpy as np
from collections import deque
from mlxtend.frequent_patterns import apriori, association_rules

TRAIN_IN = "train_features.csv"
TEST_IN = "test_features.csv"
TRAIN_OUT = "train_features_v2.csv"
TEST_OUT = "test_features_v2.csv"

ELEVATED_THRESHOLD = 100     # "elevated" = worth tracking as part of a plume
EPS_SPATIAL_KM = 200         # ST-DBSCAN spatial radius
EPS_TEMPORAL_HR = 2          # ST-DBSCAN temporal window
MIN_SAMPLES = 3              # ST-DBSCAN density threshold

CITY_COORDS = {
    "Dhaka":      (23.8103, 90.4125),
    "Chittagong": (22.3569, 91.7832),
    "Sylhet":     (24.8949, 91.8687),
    "Rajshahi":   (24.3745, 88.6042),
    "Khulna":     (22.8456, 89.5403),
    "Barisal":    (22.7010, 90.3535),
}


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_distance_matrix(coords):
    cities = list(coords.keys())
    dist = pd.DataFrame(index=cities, columns=cities, dtype=float)
    for a in cities:
        for b in cities:
            dist.loc[a, b] = 0.0 if a == b else haversine(*coords[a], *coords[b])
    return dist


# ==========================================================================
# 1. ST-DBSCAN
# ==========================================================================
def st_dbscan(points, dist_matrix, eps_spatial, eps_temporal_hr, min_samples):
    """
    points: DataFrame with columns [station, datetime, PM2.5], one row per
            elevated observation. Index must be reset (0..n-1).
    Returns: numpy array of cluster labels (-1 = noise), aligned to `points`.

    A point q is a spatio-temporal neighbor of point p if:
      haversine_distance(station_p, station_q) <= eps_spatial   AND
      abs(datetime_p - datetime_q) <= eps_temporal_hr

    Standard density-based expansion (like DBSCAN) is then run using this
    joint neighborhood definition.
    """
    # Sort by time so we can use binary search to restrict candidates to a
    # small temporal window instead of scanning all n points (O(n^2) -> ~O(n log n)).
    points = points.sort_values("datetime").reset_index(drop=True)
    n = len(points)
    labels = np.full(n, -2, dtype=int)  # -2 = unvisited, -1 = noise, >=0 = cluster id
    stations = points["station"].values
    times = points["datetime"].values.astype("datetime64[h]").astype(np.int64)  # hours since epoch

    def region_query(idx):
        s_p, t_p = stations[idx], times[idx]
        lo = np.searchsorted(times, t_p - eps_temporal_hr, side="left")
        hi = np.searchsorted(times, t_p + eps_temporal_hr, side="right")
        neighbor_idx = []
        for j in range(lo, hi):
            if j == idx:
                continue
            d = dist_matrix.loc[s_p, stations[j]]
            if d <= eps_spatial:
                neighbor_idx.append(j)
        return neighbor_idx

    cluster_id = 0
    for i in range(n):
        if labels[i] != -2:
            continue
        neighbors = region_query(i)
        if len(neighbors) + 1 < min_samples:
            labels[i] = -1  # noise (may be re-labeled as border point later)
            continue

        labels[i] = cluster_id
        queue = deque(neighbors)
        while queue:
            j = queue.popleft()
            if labels[j] == -1:
                labels[j] = cluster_id  # border point
            if labels[j] != -2:
                continue
            labels[j] = cluster_id
            j_neighbors = region_query(j)
            if len(j_neighbors) + 1 >= min_samples:
                queue.extend(j_neighbors)
        cluster_id += 1

    return labels


def run_st_dbscan(df, dist_matrix):
    elevated = df[df["PM2.5"] > ELEVATED_THRESHOLD][
        ["station", "datetime", "PM2.5"]
    ].copy()
    elevated["datetime"] = pd.to_datetime(elevated["datetime"])
    elevated = elevated.reset_index(drop=True)

    print(f"Running ST-DBSCAN on {len(elevated)} elevated points "
          f"(eps_spatial={EPS_SPATIAL_KM}km, eps_temporal={EPS_TEMPORAL_HR}h, "
          f"min_samples={MIN_SAMPLES})...")

    labels = st_dbscan(elevated, dist_matrix, EPS_SPATIAL_KM,
                        EPS_TEMPORAL_HR, MIN_SAMPLES)
    elevated["plume_cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  -> found {n_clusters} plume clusters, {n_noise} noise points")

    # plume size = number of (station, hour) points in that plume (0 for noise/non-elevated)
    plume_sizes = elevated[elevated["plume_cluster"] != -1].groupby(
        "plume_cluster").size().rename("plume_size")
    elevated = elevated.merge(plume_sizes, left_on="plume_cluster",
                               right_index=True, how="left")
    elevated["plume_size"] = elevated["plume_size"].fillna(0)
    elevated["in_plume"] = (elevated["plume_cluster"] != -1).astype(int)

    return elevated[["station", "datetime", "in_plume", "plume_size", "plume_cluster"]]


# ==========================================================================
# 2. Apriori association rules (cross-city propagation)
# ==========================================================================
def build_transaction_table(df, lags=(1, 2, 3)):
    """
    Builds a wide boolean table: for each timestamp, which cities are
    currently elevated, AND which cities WERE elevated `lag` hours ago
    (as separate lagged columns), so Apriori can find rules like
    'Dhaka_elevated_lag2 -> Chittagong_elevated_now'.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["elevated"] = df["PM2.5"] > ELEVATED_THRESHOLD

    pivot = df.pivot(index="datetime", columns="station", values="elevated").fillna(False)
    pivot = pivot.sort_index()

    table = pd.DataFrame(index=pivot.index)
    for city in pivot.columns:
        table[f"{city}_now"] = pivot[city]
        for lag in lags:
            table[f"{city}_lag{lag}"] = pivot[city].shift(lag).fillna(False).astype(bool)

    table = table.astype(bool)
    return table


def mine_association_rules(transaction_table, min_support=0.01, min_confidence=0.3):
    # max_len=2: we only care about pairwise "A (past) -> B (now)" propagation
    # rules here, not larger itemsets - this also avoids a combinatorial
    # explosion since the *_now/_lag1/_lag2/_lag3 columns for the same city
    # are highly correlated with each other.
    frequent_itemsets = apriori(transaction_table, min_support=min_support,
                                 use_colnames=True, max_len=2)
    if frequent_itemsets.empty:
        print("  No frequent itemsets found at this support threshold.")
        return pd.DataFrame(), frequent_itemsets

    rules = association_rules(frequent_itemsets, metric="confidence",
                               min_threshold=min_confidence)

    # Keep only cross-city propagation rules: antecedent is a *_lagN column
    # (a PAST elevation at another city) predicting a *_now column (current
    # elevation elsewhere) - this is the "propagation" pattern the proposal
    # is after, not trivial same-city/same-time rules.
    def is_propagation_rule(row):
        ante = list(row["antecedents"])
        cons = list(row["consequents"])
        if len(ante) != 1 or len(cons) != 1:
            return False
        a, c = ante[0], cons[0]
        a_city = a.split("_lag")[0] if "_lag" in a else a.split("_now")[0]
        c_city = c.split("_lag")[0] if "_lag" in c else c.split("_now")[0]
        return "_lag" in a and "_now" in c and a_city != c_city

    rules["is_propagation"] = rules.apply(is_propagation_rule, axis=1)
    propagation_rules = rules[rules["is_propagation"]].sort_values(
        "confidence", ascending=False)

    return propagation_rules, frequent_itemsets


def add_apriori_signal_feature(df, propagation_rules):
    """
    For each row (station, datetime), compute the max confidence among
    propagation rules whose antecedent (another city, N hours ago) is
    currently TRUE - i.e. "how strongly does current cross-city history
    suggest THIS city is about to become elevated".
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["elevated"] = df["PM2.5"] > ELEVATED_THRESHOLD
    pivot = df.pivot(index="datetime", columns="station", values="elevated").fillna(False)

    signal = pd.DataFrame(0.0, index=pivot.index, columns=pivot.columns)

    if propagation_rules.empty:
        print("  No propagation rules available - signal feature will be all zeros.")
    else:
        for _, rule in propagation_rules.iterrows():
            ante = list(rule["antecedents"])[0]   # e.g. "Dhaka_lag2"
            cons = list(rule["consequents"])[0]   # e.g. "Chittagong_now"
            ante_city, lag_str = ante.split("_lag")
            cons_city = cons.split("_now")[0]
            lag = int(lag_str)

            ante_true = pivot[ante_city].shift(lag).fillna(False)
            conf = rule["confidence"]
            signal[cons_city] = np.where(
                ante_true & (conf > signal[cons_city]), conf, signal[cons_city]
            )

    signal = signal.stack().rename("apriori_propagation_signal").reset_index()
    signal.columns = ["datetime", "station", "apriori_propagation_signal"]
    return signal


# ==========================================================================
# Main
# ==========================================================================
if __name__ == "__main__":
    dist_matrix = build_distance_matrix(CITY_COORDS)

    train = pd.read_csv(TRAIN_IN)
    test = pd.read_csv(TEST_IN)
    train["datetime"] = pd.to_datetime(train["datetime"])
    test["datetime"] = pd.to_datetime(test["datetime"])

    # ---------------- ST-DBSCAN (fit on train only) ----------------
    print("=" * 60)
    print("ST-DBSCAN")
    print("=" * 60)
    plume_train = run_st_dbscan(train, dist_matrix)
    plume_test = run_st_dbscan(test, dist_matrix)  # cluster ids are separate per split, that's fine - we only use in_plume/plume_size as features

    train = train.merge(plume_train, on=["station", "datetime"], how="left")
    test = test.merge(plume_test, on=["station", "datetime"], how="left")
    for d in (train, test):
        d["in_plume"] = d["in_plume"].fillna(0).astype(int)
        d["plume_size"] = d["plume_size"].fillna(0)
        d.drop(columns=["plume_cluster"], inplace=True)

    # ---------------- Apriori (mine rules on train only, apply to both) ----------------
    print("\n" + "=" * 60)
    print("APRIORI")
    print("=" * 60)
    transactions = build_transaction_table(train, lags=(1, 2, 3))
    propagation_rules, frequent_itemsets = mine_association_rules(
        transactions, min_support=0.01, min_confidence=0.3)

    print(f"\nTop cross-city propagation rules (by confidence):")
    if not propagation_rules.empty:
        display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        print(propagation_rules[display_cols].head(15).to_string(index=False))
    else:
        print("  none found at current thresholds - try lowering min_support/min_confidence")

    signal_train = add_apriori_signal_feature(train, propagation_rules)
    signal_test = add_apriori_signal_feature(test, propagation_rules)

    train = train.merge(signal_train, on=["station", "datetime"], how="left")
    test = test.merge(signal_test, on=["station", "datetime"], how="left")

    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)

    print(f"\nSaved {TRAIN_OUT} {train.shape} and {TEST_OUT} {test.shape}")
    print("\nNew columns added: in_plume, plume_size, apriori_propagation_signal")
    print(f"\n% rows in a plume (train): {train['in_plume'].mean()*100:.2f}%")
    print(f"apriori_propagation_signal > 0 (train): "
          f"{(train['apriori_propagation_signal']>0).mean()*100:.2f}%")