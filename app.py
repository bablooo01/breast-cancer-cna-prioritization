import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="CNA Prioritization", page_icon="🧬", layout="wide")

# Load data - DIRECTLY from CSV
@st.cache_data(ttl=0)
def load_data():
    df = pd.read_csv('results/rankings/functional_importance_scores_with_symbols.csv')
    return df

fis_df = load_data()

# Sidebar
st.sidebar.title("🧬 CNA Prioritization")
st.sidebar.markdown("---")

gene_search = st.sidebar.text_input("🔍 Search Gene Symbol:", "")
fis_threshold = st.sidebar.slider("Minimum FIS Score:", 0.0, 1.0, 0.0, 0.01)
n_top = st.sidebar.selectbox("Number of Top Genes:", [10, 20, 50, 100, 200], index=1)
show_labels = st.sidebar.multiselect(
    "Show Genes With Label:",
    options=[0, 1],
    default=[0, 1],
    format_func=lambda x: "✅ Cancer Gene" if x == 1 else "❓ Unknown"
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Genes Ranked:** {len(fis_df)}")
st.sidebar.markdown(f"**FIS Range:** {fis_df['FIS'].min():.3f} - {fis_df['FIS'].max():.3f}")

# Title
st.title("🧬 Explainable Multi-Omics CNA Prioritization")
st.markdown("*An integrated framework for prioritizing copy number alterations using CNA, RNA-seq, and methylation data*")

# Filter
filtered_df = fis_df[(fis_df['FIS'] >= fis_threshold) & (fis_df['true_label'].isin(show_labels))]

if gene_search:
    filtered_df = filtered_df[
        filtered_df['gene_symbol'].str.contains(gene_search, case=False, na=False)
    ]

top_genes = filtered_df.head(n_top)

# Metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Genes Ranked", len(filtered_df))
with col2:
    st.metric("Top Genes Displayed", len(top_genes))
with col3:
    st.metric("Positive (Cancer) Genes", sum(filtered_df['true_label'] == 1))
with col4:
    st.metric("Max FIS", f"{filtered_df['FIS'].max():.3f}" if len(filtered_df) > 0 else "0.000")
with col5:
    st.metric("Avg FIS", f"{filtered_df['FIS'].mean():.3f}" if len(filtered_df) > 0 else "0.000")

# Display rankings
st.subheader(f"Top {len(top_genes)} Genes by FIS")

display_df = pd.DataFrame({
    'Rank': top_genes['rank'].values,
    'Gene ID': top_genes['gene'].values,
    'Gene Symbol': top_genes['gene_symbol'].values,
    'FIS': top_genes['FIS'].values,
    'Cancer Label': ['✅ Cancer Gene' if x == 1 else '❓ Unknown' for x in top_genes['true_label'].values]
})

st.dataframe(display_df, use_container_width=True, height=400)

# Download
csv = display_df.to_csv(index=False)
st.download_button("📥 Download Rankings as CSV", data=csv, file_name="gene_rankings.csv", mime="text/csv")

# Debug
with st.expander("🔍 Debug: Raw Data from CSV"):
    st.dataframe(fis_df[['gene_symbol', 'FIS', 'true_label']].head(20))

# FIS Distribution
st.subheader("FIS Distribution")
fig = px.histogram(
    filtered_df,
    x='FIS',
    color='true_label',
    nbins=50,
    title='Distribution of FIS Scores by Label',
    labels={'true_label': 'Label', 'FIS': 'FIS Score'},
    color_discrete_map={0: '#FF6B6B', 1: '#51CF66'}
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("🧬 **Explainable Multi-Omics CNA Prioritization Framework** | Built with TCGA-BRCA, GENCODE, and ClinVar")