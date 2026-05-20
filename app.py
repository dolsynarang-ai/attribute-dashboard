"""
Attribute Score → Performance Dashboard
========================================
Upload two CSVs via the sidebar:
  1. Attribute scores  : id_partner, sku_config, family, total_score, image_score,
                         facet_score, consumer_mandatory_attributes_score
  2. Transactions      : id_partner, sku_config, gmv, units, orders,
                         Glance_Views, Search_Impressions, non_search_impressions

Deploy on Streamlit Cloud → share the URL with anyone.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings, io
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Attribute → Performance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCORE_COLS  = ["total_score","image_score","facet_score","consumer_mandatory_attributes_score"]
METRIC_COLS = ["gmv","units","orders","Glance_Views","Search_Impressions","non_search_impressions"]
METRIC_LABELS = {
    "gmv":"GMV","units":"Units","orders":"Orders",
    "Glance_Views":"Glance Views","Search_Impressions":"Search Impressions",
    "non_search_impressions":"Non-Search Impressions"
}
TIER_COLORS  = {"poor":"#D85A30","fair":"#BA7517","good":"#378ADD","excellent":"#1D9E75"}
TIER_ORDER   = ["poor","fair","good","excellent"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_cols(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+","_",regex=True)
    return df

def fmt(n):
    if pd.isna(n): return "—"
    n = float(n)
    if n >= 1e7:  return f"₹{n/1e7:.1f}Cr"
    if n >= 1e5:  return f"₹{n/1e5:.1f}L"
    if n >= 1e3:  return f"{n/1e3:.1f}K"
    return f"{n:.1f}"

@st.cache_data(show_spinner=False)
def load_and_merge(attr_bytes, txn_bytes):
    attrs = clean_cols(pd.read_csv(io.BytesIO(attr_bytes)))
    txns  = clean_cols(pd.read_csv(io.BytesIO(txn_bytes)))

    # Normalize known column name variants
    rename_map = {"f0_":"gmv","f1_":"units","f2_":"orders",
                  "glance_views":"Glance_Views",
                  "search_impressions":"Search_Impressions",
                  "non_search_impressions":"non_search_impressions"}
    txns.rename(columns={k:v for k,v in rename_map.items() if k in txns.columns}, inplace=True)
    attrs.rename(columns={k:v for k,v in rename_map.items() if k in attrs.columns}, inplace=True)

    merged = pd.merge(attrs, txns, on=["id_partner","sku_config"], how="left")

    # Quality tier
    try:
        merged["quality_tier"] = pd.qcut(
            merged["total_score"].fillna(0), q=4,
            labels=["poor","fair","good","excellent"], duplicates="drop"
        )
    except Exception:
        merged["quality_tier"] = "unknown"

    # Fill rate features
    if "consumer_attributes_filled" in merged and "pft_consumer_mandatory_attribute_count" in merged:
        merged["consumer_fill_rate"] = (
            merged["consumer_attributes_filled"] /
            merged["pft_consumer_mandatory_attribute_count"].replace(0,np.nan)
        ).clip(0,1)

    return merged

@st.cache_data(show_spinner=False)
def run_rf(df_json, target):
    df = pd.read_json(io.StringIO(df_json), orient="split")
    feats = [c for c in SCORE_COLS if c in df.columns]
    mask  = df[target].notna() & (df[target] > 0)
    X, y  = df.loc[mask, feats].fillna(0), np.log1p(df.loc[mask, target])
    if len(X) < 50:
        return None
    rf = RandomForestRegressor(n_estimators=200, max_depth=8,
                                min_samples_leaf=5, random_state=42, n_jobs=-1)
    cv = cross_val_score(rf, X, y, cv=5, scoring="r2")
    rf.fit(X, y)
    imp = dict(sorted(zip(feats, rf.feature_importances_), key=lambda x: -x[1]))
    return {"cv_r2": round(float(cv.mean()),3), "cv_std": round(float(cv.std()),3),
            "importances": imp, "n": int(len(X))}

def pearson_corr(df):
    sc = [c for c in SCORE_COLS  if c in df.columns]
    mc = [c for c in METRIC_COLS if c in df.columns]
    if not sc or not mc: return pd.DataFrame()
    return df[sc+mc].corr()[mc].loc[sc].round(3)

def underperformers(df, score_pct=0.65, vis_quantile=0.33):
    if "total_score" not in df.columns: return pd.DataFrame()
    vis_col = "Glance_Views" if "Glance_Views" in df.columns else None
    if not vis_col: return pd.DataFrame()
    mx = df["total_score"].max()
    hi = df["total_score"] >= score_pct * mx
    lo = df[vis_col] <= df[vis_col].quantile(vis_quantile)
    out = df[hi & lo].copy()
    out["gap"] = (df["total_score"]/mx - df[vis_col].rank(pct=True)).loc[out.index].round(3)
    cols = [c for c in ["id_partner","sku_config","family","total_score","image_score",
                         "facet_score","consumer_mandatory_attributes_score",
                         "Glance_Views","gmv","units","orders","gap"] if c in out.columns]
    return out[cols].sort_values("gap", ascending=False).reset_index(drop=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/combo-chart.png", width=40)
    st.title("Attribute → Performance")
    st.caption("Upload your CSVs to run the analysis")
    st.divider()

    attr_file = st.file_uploader("📋 Attribute scores CSV", type="csv")
    txn_file  = st.file_uploader("💰 Transactions CSV",     type="csv")

    st.divider()

    if attr_file and txn_file:
        st.success("Both files uploaded ✓")
        st.caption(f"Attr: {attr_file.name}")
        st.caption(f"Txn : {txn_file.name}")

    st.divider()
    st.markdown("**Expected columns**")
    with st.expander("Attribute scores"):
        st.caption("id_partner, sku_config, family, total_score, image_score, facet_score, consumer_mandatory_attributes_score")
    with st.expander("Transactions"):
        st.caption("id_partner, sku_config, gmv, units, orders, Glance_Views, Search_Impressions, non_search_impressions")

# ── Guard ─────────────────────────────────────────────────────────────────────
if not attr_file or not txn_file:
    st.markdown("## 👈 Upload both CSVs in the sidebar to begin")
    st.info("This dashboard analyses how attribute/content quality scores relate to sales, visibility, and revenue across your SKU catalogue.")
    col1, col2, col3 = st.columns(3)
    col1.metric("What you'll get", "Correlations")
    col2.metric("", "Quality tier analysis")
    col3.metric("", "Underperformer flags")
    st.stop()

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Merging datasets…"):
    df = load_and_merge(attr_file.read(), txn_file.read())

if df.empty:
    st.error("No rows after join — check that id_partner and sku_config overlap.")
    st.stop()

avail_metrics = [c for c in METRIC_COLS if c in df.columns]
avail_scores  = [c for c in SCORE_COLS  if c in df.columns]
families      = sorted(df["family"].dropna().unique().tolist()) if "family" in df.columns else []

# ── Filters (top bar) ─────────────────────────────────────────────────────────
with st.expander("🔍 Filters", expanded=False):
    fc1, fc2, fc3 = st.columns(3)
    sel_families = fc1.multiselect("Family", families, default=families, key="fam_filter")
    sel_tiers    = fc2.multiselect("Quality tier", TIER_ORDER, default=TIER_ORDER, key="tier_filter")
    live_only    = fc3.checkbox("Live SKUs only (is_live=1)", value=False)

    fdf = df.copy()
    if sel_families: fdf = fdf[fdf["family"].isin(sel_families)]
    if sel_tiers:    fdf = fdf[fdf["quality_tier"].isin(sel_tiers)]
    if live_only and "is_live" in fdf.columns:
        fdf = fdf[fdf["is_live"]==1]

# ── Summary KPIs ──────────────────────────────────────────────────────────────
st.markdown("### Summary")
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Total SKUs",       f"{len(fdf):,}")
k2.metric("With GMV",         f"{fdf['gmv'].notna().sum():,}" if "gmv" in fdf.columns else "—")
k3.metric("Families",         fdf["family"].nunique() if "family" in fdf.columns else "—")
k4.metric("Avg total score",  f"{fdf['total_score'].mean():.1f}" if "total_score" in fdf.columns else "—")
k5.metric("Avg GMV / SKU",    fmt(fdf["gmv"].mean()) if "gmv" in fdf.columns else "—")
k6.metric("Underperformers",  len(underperformers(fdf)))

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📋 Overview","🔗 Correlations","🏷️ Families","📊 Tier analysis","🤖 ML models","⚠️ Underperformers","💡 Key insights","📥 Export"])

# ═══════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════
with tabs[0]:
    c1, c2 = st.columns(2)

    # Tier distribution
    tier_cnt = fdf["quality_tier"].value_counts().reindex(TIER_ORDER).fillna(0).reset_index()
    tier_cnt.columns = ["tier","count"]
    fig = px.bar(tier_cnt, x="tier", y="count", color="tier",
                 color_discrete_map=TIER_COLORS, text="count",
                 title="SKU count by quality tier")
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", height=300)
    c1.plotly_chart(fig, use_container_width=True)

    # Avg GMV by tier
    if "gmv" in fdf.columns:
        gmv_tier = fdf.groupby("quality_tier", observed=True)["gmv"].mean().reindex(TIER_ORDER).reset_index()
        gmv_tier.columns = ["tier","avg_gmv"]
        fig2 = px.bar(gmv_tier, x="tier", y="avg_gmv", color="tier",
                      color_discrete_map=TIER_COLORS, text_auto=".0f",
                      title="Avg GMV by quality tier")
        fig2.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)", height=300,
                           yaxis_tickprefix="₹")
        c2.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    if "Glance_Views" in fdf.columns:
        gv_tier = fdf.groupby("quality_tier", observed=True)["Glance_Views"].mean().reindex(TIER_ORDER).reset_index()
        gv_tier.columns = ["tier","avg_gv"]
        fig3 = px.bar(gv_tier, x="tier", y="avg_gv", color="tier",
                      color_discrete_map=TIER_COLORS, text_auto=".0f",
                      title="Avg Glance Views by tier", log_y=True)
        fig3.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)", height=300)
        c3.plotly_chart(fig3, use_container_width=True)

    if "Search_Impressions" in fdf.columns:
        si_tier = fdf.groupby("quality_tier", observed=True)["Search_Impressions"].mean().reindex(TIER_ORDER).reset_index()
        si_tier.columns = ["tier","avg_si"]
        fig4 = px.bar(si_tier, x="tier", y="avg_si", color="tier",
                      color_discrete_map=TIER_COLORS, text_auto=".0f",
                      title="Avg Search Impressions by tier", log_y=True)
        fig4.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)", height=300)
        c4.plotly_chart(fig4, use_container_width=True)

    # Score distributions
    st.subheader("Score distributions")
    sc_melt = fdf[avail_scores].melt(var_name="score_type", value_name="value")
    fig5 = px.box(sc_melt, x="score_type", y="value", color="score_type",
                  title="Spread of attribute scores across all SKUs")
    fig5.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                       paper_bgcolor="rgba(0,0,0,0)", height=320)
    st.plotly_chart(fig5, use_container_width=True)

# ═══════════════════════════════════════════
# TAB 2 — CORRELATIONS
# ═══════════════════════════════════════════
with tabs[1]:
    corr = pearson_corr(fdf)
    if corr.empty:
        st.warning("Not enough columns to compute correlations.")
    else:
        col_sel = st.selectbox("Select metric", avail_metrics,
                               format_func=lambda x: METRIC_LABELS.get(x,x))
        c_vals = corr[col_sel].sort_values(ascending=False)
        fig = px.bar(c_vals.reset_index(), x=col_sel, y="index", orientation="h",
                     color=col_sel, color_continuous_scale="RdYlGn",
                     range_color=[-0.3, 0.3],
                     title=f"Correlation of attribute scores → {METRIC_LABELS.get(col_sel,col_sel)}",
                     labels={col_sel:"Pearson r","index":"Attribute score"})
        fig.update_layout(height=300, plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Full correlation matrix")
        fig2 = px.imshow(corr, text_auto=True, aspect="auto",
                         color_continuous_scale="RdYlGn", zmin=-0.3, zmax=0.3,
                         title="All attribute scores × all metrics")
        fig2.update_layout(height=350, plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"GMV/units/orders based on {fdf['gmv'].notna().sum():,} SKUs with sales data. Impression metrics cover all {len(fdf):,} SKUs.")

# ═══════════════════════════════════════════
# TAB 3 — FAMILIES
# ═══════════════════════════════════════════
with tabs[2]:
    if "family" not in fdf.columns:
        st.warning("No family column found.")
    else:
        sort_by = st.selectbox("Sort by", ["Avg score","Avg GMV","SKU count"])
        fam_score = fdf.groupby("family")["total_score"].agg(["mean","count"]).round(2)
        fam_score.columns = ["avg_score","sku_count"]

        if "gmv" in fdf.columns:
            fam_gmv = fdf.dropna(subset=["gmv"]).groupby("family")["gmv"].mean().round(2)
            fam_score["avg_gmv"] = fam_gmv

        if sort_by == "Avg score":    fam_score = fam_score.sort_values("avg_score", ascending=False)
        elif sort_by == "Avg GMV":    fam_score = fam_score.sort_values("avg_gmv",   ascending=False)
        else:                          fam_score = fam_score.sort_values("sku_count", ascending=False)

        fig = px.bar(fam_score.reset_index(), x="avg_score", y="family",
                     orientation="h", color="avg_score",
                     color_continuous_scale="RdYlGn", range_color=[20,90],
                     title="Avg total score by family",
                     text=fam_score["avg_score"].values,
                     labels={"avg_score":"Avg score","family":""})
        fig.update_layout(height=max(400, len(fam_score)*22),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          showlegend=False, yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        if "avg_gmv" in fam_score.columns:
            fig2 = px.scatter(fam_score.reset_index(),
                              x="avg_score", y="avg_gmv", size="sku_count",
                              color="avg_score", color_continuous_scale="RdYlGn",
                              range_color=[20,90], text="family",
                              title="Score vs GMV by family (bubble = SKU count)",
                              labels={"avg_score":"Avg score","avg_gmv":"Avg GMV","family":""})
            fig2.update_traces(textposition="top center", textfont_size=10)
            fig2.update_layout(height=500, plot_bgcolor="rgba(0,0,0,0)",
                               paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                               yaxis_tickprefix="₹")
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(fam_score.reset_index().rename(columns={
            "family":"Family","avg_score":"Avg score",
            "sku_count":"SKU count","avg_gmv":"Avg GMV"}),
            use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════
# TAB 4 — TIER ANALYSIS
# ═══════════════════════════════════════════
with tabs[3]:
    st.subheader("Average metrics by quality tier")
    tier_table = fdf.groupby("quality_tier", observed=True)[
        [c for c in METRIC_COLS if c in fdf.columns]].mean().round(2).reindex(TIER_ORDER)
    st.dataframe(tier_table.style.background_gradient(cmap="YlGn", axis=0),
                 use_container_width=True)

    sel_metric_tier = st.selectbox("Pick metric to visualise",
                                   [c for c in METRIC_COLS if c in fdf.columns],
                                   format_func=lambda x: METRIC_LABELS.get(x,x),
                                   key="tier_metric")
    fig = px.box(fdf.dropna(subset=[sel_metric_tier]),
                 x="quality_tier", y=sel_metric_tier,
                 color="quality_tier", color_discrete_map=TIER_COLORS,
                 log_y=True,
                 category_orders={"quality_tier": TIER_ORDER},
                 title=f"{METRIC_LABELS.get(sel_metric_tier,sel_metric_tier)} distribution by tier",
                 labels={"quality_tier":"Tier", sel_metric_tier: METRIC_LABELS.get(sel_metric_tier,sel_metric_tier)})
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", height=380)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════
# TAB 5 — ML MODELS
# ═══════════════════════════════════════════
with tabs[4]:
    st.subheader("Random Forest — feature importance per metric")
    st.caption("Trains a model to predict each metric from attribute scores. Feature importance shows which score type matters most.")

    target_sel = st.selectbox("Target metric to model",
                              [c for c in ["gmv","units","orders","Glance_Views","Search_Impressions"] if c in fdf.columns],
                              format_func=lambda x: METRIC_LABELS.get(x,x))

    if st.button("▶ Train model", type="primary"):
        with st.spinner(f"Training Random Forest for {target_sel}…"):
            res = run_rf(fdf.to_json(orient="split"), target_sel)
            st.session_state["rf_result"] = res
            st.session_state["rf_target"] = target_sel

    if "rf_result" in st.session_state and st.session_state["rf_result"]:
        res = st.session_state["rf_result"]
        tgt = st.session_state["rf_target"]
        c1,c2,c3 = st.columns(3)
        c1.metric("CV R²",    f"{res['cv_r2']:.3f}")
        c2.metric("Std",      f"± {res['cv_std']:.3f}")
        c3.metric("n samples",f"{res['n']:,}")

        imp_df = pd.DataFrame(list(res["importances"].items()),
                              columns=["feature","importance"])
        imp_df["pct"] = (imp_df["importance"]*100).round(2)
        fig = px.bar(imp_df.sort_values("pct"),
                     x="pct", y="feature", orientation="h",
                     color="pct", color_continuous_scale="Blues",
                     title=f"Feature importance → {METRIC_LABELS.get(tgt,tgt)}",
                     labels={"pct":"Importance (%)","feature":""})
        fig.update_layout(height=320, showlegend=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Target is log-transformed {tgt}. R² = {res['cv_r2']} means the model explains {res['cv_r2']*100:.1f}% of variance.")
    elif "rf_result" in st.session_state and not st.session_state["rf_result"]:
        st.warning("Not enough non-zero rows to train a model for this metric.")

# ═══════════════════════════════════════════
# TAB 6 — UNDERPERFORMERS
# ═══════════════════════════════════════════
with tabs[5]:
    st.subheader("High-score, low-visibility SKUs")

    c1, c2 = st.columns(2)
    score_thresh = c1.slider("Min score threshold (%)", 50, 90, 65, 5,
                              help="SKUs in top X% of total_score") / 100
    vis_quant    = c2.slider("Max glance view percentile (%)", 10, 50, 33, 5,
                              help="SKUs in bottom X% of Glance_Views") / 100

    up = underperformers(fdf, score_thresh, vis_quant)

    if up.empty:
        st.success("No underperformers with these thresholds.")
    else:
        st.warning(f"**{len(up):,} SKUs** have high attribute scores but low glance views.")

        if "gap" in up.columns:
            fig = px.histogram(up, x="gap", nbins=20,
                               title="Gap score distribution (higher = bigger mismatch)",
                               color_discrete_sequence=["#D85A30"])
            fig.update_layout(height=250, plot_bgcolor="rgba(0,0,0,0)",
                               paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            up.head(100)
              .style.background_gradient(subset=["total_score"] if "total_score" in up.columns else [], cmap="Greens")
                    .background_gradient(subset=["Glance_Views"] if "Glance_Views" in up.columns else [], cmap="Reds_r"),
            use_container_width=True, hide_index=True
        )

# ═══════════════════════════════════════════
# TAB 7 — KEY INSIGHTS
# ═══════════════════════════════════════════
with tabs[6]:
    st.subheader("Auto-generated insights")

    # Compute live insights from actual data
    if "quality_tier" in fdf.columns and "gmv" in fdf.columns:
        tier_gmv = fdf.groupby("quality_tier", observed=True)["gmv"].mean().reindex(TIER_ORDER)
        poor_gmv = tier_gmv.get("poor", 0)
        exc_gmv  = tier_gmv.get("excellent", 0)
        mult     = exc_gmv / poor_gmv if poor_gmv > 0 else 0

        st.success(f"**Excellent-tier SKUs earn {mult:.1f}× more GMV** than poor-tier SKUs "
                   f"(₹{exc_gmv:,.0f} vs ₹{poor_gmv:,.0f} avg).")

    if "quality_tier" in fdf.columns and "Glance_Views" in fdf.columns:
        tier_gv = fdf.groupby("quality_tier", observed=True)["Glance_Views"].mean().reindex(TIER_ORDER)
        gv_mult = tier_gv.get("excellent",0) / tier_gv.get("poor",1)
        st.success(f"**Excellent SKUs get {gv_mult:.0f}× more Glance Views** than poor-tier SKUs — "
                   f"attribute quality directly drives discoverability.")

    if "family" in fdf.columns and "gmv" in fdf.columns:
        fam_gmv   = fdf.dropna(subset=["gmv"]).groupby("family")["gmv"].mean().sort_values(ascending=False)
        fam_score = fdf.groupby("family")["total_score"].mean().sort_values(ascending=True)
        top_gmv   = fam_gmv.index[0] if len(fam_gmv) else "—"
        low_score = fam_score.index[0] if len(fam_score) else "—"
        st.warning(f"**{top_gmv.replace('_',' ').title()}** has the highest avg GMV "
                   f"(₹{fam_gmv.iloc[0]:,.0f}). Focus attribute enrichment here for max revenue impact.")
        if low_score != top_gmv:
            st.warning(f"**{low_score.replace('_',' ').title()}** has the lowest avg score "
                       f"({fam_score.iloc[0]:.1f}) — biggest content quality gap to close.")

    gmv_pct = fdf["gmv"].notna().mean() * 100 if "gmv" in fdf.columns else 0
    if gmv_pct < 20:
        st.error(f"**Only {gmv_pct:.1f}% of SKUs have GMV data.** "
                 f"{int(len(fdf)*(1-gmv_pct/100)):,} SKUs show no sales. "
                 f"Cross-check with is_live to separate inactive from underperforming.")

    up_count = len(underperformers(fdf))
    if up_count > 0:
        st.error(f"**{up_count} underperforming SKUs flagged** — strong content scores but low visibility. "
                 f"Likely causes: wrong category, suppressed listings, or poor search indexing.")

    corr = pearson_corr(fdf)
    if not corr.empty and "gmv" in corr.columns:
        top_driver = corr["gmv"].abs().idxmax()
        top_r      = corr["gmv"][top_driver]
        st.info(f"**{top_driver.replace('_',' ').title()}** is the strongest GMV driver "
                f"(r = {top_r:+.3f}). Prioritise improving this score across your catalogue.")

# ═══════════════════════════════════════════
# TAB 8 — EXPORT
# ═══════════════════════════════════════════
with tabs[7]:
    st.subheader("Download results")
    c1, c2, c3, c4 = st.columns(4)

    c1.download_button("📥 Merged dataset",
        data=fdf.to_csv(index=False),
        file_name="merged_data.csv", mime="text/csv")

    up_exp = underperformers(fdf)
    if not up_exp.empty:
        c2.download_button("⚠️ Underperformers",
            data=up_exp.to_csv(index=False),
            file_name="underperforming_skus.csv", mime="text/csv")

    corr = pearson_corr(fdf)
    if not corr.empty:
        c3.download_button("🔗 Correlations",
            data=corr.to_csv(),
            file_name="correlations.csv", mime="text/csv")

    if "family" in fdf.columns:
        fam_exp = fdf.groupby("family").agg(
            avg_score=("total_score","mean"),
            sku_count=("sku_config","count"),
            avg_gmv=("gmv","mean"),
            avg_gv=("Glance_Views","mean"),
            avg_search_impr=("Search_Impressions","mean")
        ).round(2).reset_index()
        c4.download_button("🏷️ Family summary",
            data=fam_exp.to_csv(index=False),
            file_name="family_summary.csv", mime="text/csv")

    st.divider()
    st.caption("Tip: Re-upload new CSVs anytime — the dashboard will refresh automatically.")
