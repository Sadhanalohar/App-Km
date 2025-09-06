# -*- coding: utf-8 -*-
"""
Streamlit mini recommender app (KMeans-based)
Copy this file to app.py in your repo and deploy on Streamlit Cloud.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

# ---------- Configuration ----------
FEATURES = ["likes_count", "comments_count", "shares_count", "impressions", "engagement_rate"]
MODEL_DIR = "models"
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.joblib")
KMEANS_FILE = os.path.join(MODEL_DIR, "kmeans.joblib")
MAP_FILE = os.path.join(MODEL_DIR, "cluster_to_proxy.joblib")
DEFAULT_K = 2

st.set_page_config(page_title="Engagement → Mini Recommender", layout="wide")
st.title("Mini Recommender — Engagement-based (KMeans)")

# ---------- Utility functions ----------
def models_exist():
    return os.path.exists(SCALER_FILE) and os.path.exists(KMEANS_FILE) and os.path.exists(MAP_FILE)

def train_and_save(df_train, k=DEFAULT_K):
    os.makedirs(MODEL_DIR, exist_ok=True)
    X = df_train[FEATURES].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(Xs)
    # interpret clusters: cluster with highest sum of means -> "high engagement"
    centers_scaled = kmeans.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)
    cluster_means_df = pd.DataFrame(centers, columns=FEATURES)
    cluster_score = cluster_means_df.sum(axis=1)
    high_cluster = int(cluster_score.idxmax())
    low_cluster = int(cluster_score.idxmin())
    mapping = {high_cluster: 1, low_cluster: 0}
    # save
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(kmeans, KMEANS_FILE)
    joblib.dump(mapping, MAP_FILE)
    return scaler, kmeans, mapping, cluster_means_df

def load_models():
    scaler = joblib.load(SCALER_FILE)
    kmeans = joblib.load(KMEANS_FILE)
    mapping = joblib.load(MAP_FILE)
    return scaler, kmeans, mapping

def compute_distance_score(x_scaled, kmeans, high_cluster_id):
    centroids = kmeans.cluster_centers_
    high_centroid = centroids[high_cluster_id].reshape(1, -1)
    d = pairwise_distances(x_scaled, high_centroid).reshape(-1)
    max_possible = np.max(pairwise_distances(centroids, high_centroid))
    if max_possible == 0:
        score = np.ones_like(d)
    else:
        score = 1 - (d / max_possible)
        score = np.clip(score, 0.0, 1.0)
    return score

def predict_dataframe(df_in, scaler, kmeans, mapping, threshold=0.5):
    df = df_in.copy()
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0
    X = df[FEATURES].fillna(0).values
    Xs = scaler.transform(X)
    clusters = kmeans.predict(Xs)
    # find high cluster id
    high_cluster_id = next((cid for cid, lab in mapping.items() if lab == 1), 0)
    score = compute_distance_score(Xs, kmeans, high_cluster_id)
    proxy_labels = [int(mapping.get(int(c), 0)) for c in clusters]
    recommend_flags = [(p == 1 and s >= threshold) for p, s in zip(proxy_labels, score)]
    out = df.copy()
    out["cluster"] = clusters
    out["sales_proxy_label"] = proxy_labels
    out["distance_score"] = np.round(score, 3)
    out["recommend_for_promotion"] = recommend_flags
    return out

# ---------- Sidebar: training / models ----------
st.sidebar.header("Model / Training")
st.sidebar.write("You can upload a CSV to train the model, or let the app train on the included CSV in repo.")

uploaded_train = st.sidebar.file_uploader("Upload training CSV (optional)", type=["csv"])
k_choice = st.sidebar.slider("k (number of clusters)", min_value=2, max_value=6, value=DEFAULT_K)

if models_exist():
    st.sidebar.success("Models found in ./models/")
    scaler, kmeans, mapping = load_models()
    st.sidebar.write("Cluster -> proxy mapping:", mapping)
else:
    scaler = kmeans = mapping = None
    st.sidebar.warning("No models found. Upload CSV in sidebar or use default CSV to train.")

if uploaded_train is not None:
    if st.sidebar.button("Train & Save models from uploaded CSV"):
        df_train = pd.read_csv(uploaded_train)
        missing = [f for f in FEATURES if f not in df_train.columns]
        if missing:
            st.sidebar.error(f"Missing columns in uploaded CSV: {missing}")
        else:
            with st.spinner("Training..."):
                scaler, kmeans, mapping, cluster_means_df = train_and_save(df_train, k=k_choice)
            st.sidebar.success("Models trained & saved.")
            st.sidebar.write("Cluster means (approx):")
            st.sidebar.dataframe(cluster_means_df.round(1))

# Option: train from repo CSV if present and models missing
elif not models_exist() and os.path.exists("social_media_clusters_with_labels.csv"):
    if st.sidebar.button("Train from bundled CSV (social_media_clusters_with_labels.csv)"):
        df_train = pd.read_csv("social_media_clusters_with_labels.csv")
        missing = [f for f in FEATURES if f not in df_train.columns]
        if missing:
            st.sidebar.error(f"Bundled CSV missing columns: {missing}")
        else:
            with st.spinner("Training from bundled CSV..."):
                scaler, kmeans, mapping, cluster_means_df = train_and_save(df_train, k=k_choice)
            st.sidebar.success("Models trained & saved from bundled CSV.")
            st.sidebar.dataframe(cluster_means_df.round(1))

# ---------- Main: Single post prediction ----------
st.header("Single post quick check")
st.write("Enter engagement metrics for one post and get a recommendation.")

c1, c2 = st.columns(2)
with c1:
    likes = st.number_input("likes_count", min_value=0, value=100)
    comments = st.number_input("comments_count", min_value=0, value=10)
    shares = st.number_input("shares_count", min_value=0, value=5)
with c2:
    impressions = st.number_input("impressions", min_value=0, value=5000)
    engagement_rate = st.number_input("engagement_rate (decimal)", min_value=0.0, value=0.02, format="%.5f")
single_btn = st.button("Predict single post")

if single_btn:
    if not models_exist() and scaler is None:
        st.error("No trained models available. Please upload a training CSV in the sidebar and train first.")
    else:
        if scaler is None:
            scaler, kmeans, mapping = load_models()
        df_single = pd.DataFrame([{
            "likes_count": likes,
            "comments_count": comments,
            "shares_count": shares,
            "impressions": impressions,
            "engagement_rate": engagement_rate
        }])
        df_out = predict_dataframe(df_single, scaler, kmeans, mapping, threshold=0.5)
        st.table(df_out[["likes_count","comments_count","shares_count","impressions","engagement_rate","cluster","sales_proxy_label","distance_score","recommend_for_promotion"]])

# ---------- Main: Batch CSV predictions ----------
st.markdown("---")
st.header("Batch predictions (CSV upload)")
st.write("Upload a CSV with the same engagement columns and get recommendations for each row.")
uploaded_batch = st.file_uploader("Upload CSV for batch", type=["csv"], key="batch")

score_thresh = st.slider("Minimum distance score to recommend (0–1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if uploaded_batch is not None:
    if not models_exist() and scaler is None:
        st.error("No trained models available. Train models first using the sidebar.")
    else:
        if scaler is None:
            scaler, kmeans, mapping = load_models()
        df_batch = pd.read_csv(uploaded_batch)
        missing = [f for f in FEATURES if f not in df_batch.columns]
        if missing:
            st.error(f"Uploaded CSV missing columns: {missing}")
        else:
            with st.spinner("Predicting..."):
                df_preds = predict_dataframe(df_batch, scaler, kmeans, mapping, threshold=score_thresh)
            st.success("Predictions complete")
            st.dataframe(df_preds.head(100))
            csv_out = df_preds.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv_out, file_name="batch_recommendations.csv", mime="text/csv")

# ---------- Optional: Visualize clusters with PCA ----------
if models_exist() or (scaler is not None and kmeans is not None):
    if st.checkbox("Show PCA cluster preview"):
        if scaler is None:
            scaler, kmeans, mapping = load_models()
        # create a small sample to plot (if training data exists)
        sample_df = None
        if os.path.exists("social_media_clusters_with_labels.csv"):
            try:
                sample_df = pd.read_csv("social_media_clusters_with_labels.csv").sample(n=min(1000, sum(1 for _ in open("social_media_clusters_with_labels.csv"))), random_state=42)
            except Exception:
                sample_df = pd.read_csv("social_media_clusters_with_labels.csv").head(500)
        if sample_df is not None:
            Xs = scaler.transform(sample_df[FEATURES].fillna(0).values)
            pca = PCA(n_components=2, random_state=42)
            proj = pca.fit_transform(Xs)
            sample_df["pca1"] = proj[:,0]
            sample_df["pca2"] = proj[:,1]
            st.write("PCA projection (sample) — colored by predicted cluster")
            st.altair_chart(
                __import__("altair").Chart(sample_df).mark_circle(size=40).encode(
                    x="pca1", y="pca2", color="cluster:N", tooltip=FEATURES
                ).interactive(), use_container_width=True
            )
        else:
            st.info("No sample CSV found to visualize. Upload a CSV or add 'social_media_clusters_with_labels.csv' to repo.")

st.caption("Note: This is a heuristic recommender using clustering on engagement data. For real conversion predictions use supervised training with actual sales labels.")
