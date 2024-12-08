import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud
import plotly.express as px


# Title and Introduction
st.title("Workforce Distribution Analysis")
st.markdown("""
This app explores workforce distribution using EDA, clustering, and visualizations.
""")

# Specify the file path
file_path = "C:/Users/Sathishwaran007/Downloads/stl/COMF.csv"

# Load the dataset
df = pd.read_csv(file_path)

# Display the title and dataset in the browser
st.title("Clustered Workforce Data")
st.write("Below is the data loaded from your file:")
st.dataframe(df)

# Select numerical columns
numerical_cols = [
    'Main_Workers_Total_Persons', 'Main_Workers_Total_Males', 'Main_Workers_Total_Females',
    'Main_Workers_Rural_Persons', 'Main_Workers_Rural_Males', 'Main_Workers_Rural_Females',
    'Main_Workers_Urban_Persons', 'Main_Workers_Urban_Males', 'Main_Workers_Urban_Females',
    'Marginal_Workers_Total_Persons', 'Marginal_Workers_Total_Males', 'Marginal_Workers_Total_Females',
    'Marginal_Workers_Rural_Persons', 'Marginal_Workers_Rural_Males', 'Marginal_Workers_Rural_Females',
    'Marginal_Workers_Urban_Persons', 'Marginal_Workers_Urban_Males', 'Marginal_Workers_Urban_Females'
]

# Handle missing values
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Normalize the data
scaler = StandardScaler()
try:
    scaled_data = scaler.fit_transform(df[numerical_cols])
except KeyError:
    st.error("Numerical columns not found in dataset.")
    st.stop()

# Clustering
st.sidebar.header("Clustering Settings")
num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=10, value=5)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
df['Cluster'] = clusters

st.write("### Clustered Data")
st.dataframe(df[['State_Code', 'District_Code', 'Cluster']].head())

# PCA for Visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)
df['PCA1'] = reduced_data[:, 0]
df['PCA2'] = reduced_data[:, 1]

st.write("### Cluster Visualization (PCA)")
fig = px.scatter(df, x='PCA1', y='PCA2', color='Cluster',
                 title="Clusters of Workforce Distribution",
                 template="plotly_white")
st.plotly_chart(fig)

# WordCloud for NIC_Name
if 'NIC_Name' in df.columns:
    text = " ".join(df['NIC_Name'].fillna('Unknown'))
else:
    text = "No NIC_Name column available"
wordcloud = WordCloud(background_color='white', width=800, height=400).generate(text)

st.write("### WordCloud of NIC Names")
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

# Distribution of Clusters
st.write("### Cluster Distribution")
fig = px.histogram(df, x='Cluster', title="Cluster Distribution", template="plotly_white")
st.plotly_chart(fig)

# Save Clustered Data
st.download_button("Download Clustered Data", data=df.to_csv(index=False), file_name="clustered_workforce_data.csv", mime="text/csv")
