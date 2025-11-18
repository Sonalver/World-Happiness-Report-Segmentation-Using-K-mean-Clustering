# IMPORT DATASETS AND LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# LOAD DATA
happy_df = pd.read_csv('C:/Users/sonal/PycharmProjects/PythonProject/resources/happiness_report.csv')
print("Full DataFrame:")
print(happy_df, "\n")

print("First 5 rows:")
print(happy_df.head(), "\n")

print("Shape of dataset (rows, columns):")
print(happy_df.shape, "\n")

# Two methods to find number of samples
print("Number of samples (len):", len(happy_df))
print("Last 5 rows:")
print(happy_df.tail(), "\n")

# Country-based filtering
print("Canada stats:")
print(happy_df[happy_df['Country or region'] == 'Canada'], "\n")

print("India stats:")
print(happy_df[happy_df['Country or region'] == 'India'], "\n")

# EXPLORATORY DATA ANALYSIS (EDA)
print("Info:")
print(happy_df.info(), "\n")

print("Null values:")
print(happy_df.isnull().sum(), "\n")

print("Statistical Summary:")
print(happy_df.describe(), "\n")

print("Number of duplicated rows:")
print(happy_df.duplicated().sum(), "\n")

# Country with maximum happiness score
max_score = happy_df['Score'].max()
result = happy_df[happy_df['Score'] == max_score]
print("Country with Maximum Happiness Score:")
print(result, "\n")

# DATA VISUALIZATION – PART 1
# Pairplot for relationships among key variables
plt.figure(figsize=(20, 20))
sns.pairplot(
    happy_df[
        [
            'Score', 'GDP per capita', 'Social support',
            'Healthy life expectancy', 'Freedom to make life choices',
            'Generosity', 'Perceptions of corruption'
        ]
    ]
)
plt.show()

# DISTRIBUTION PLOTS FOR ALL NUMERIC FEATURES
columns = [
    'Score', 'GDP per capita', 'Social support', 'Healthy life expectancy',
    'Freedom to make life choices', 'Generosity', 'Perceptions of corruption'
]

plt.figure(figsize=(20, 50))

for i, col in enumerate(columns):
    plt.subplot(8, 2, i + 1)
    sns.histplot(happy_df[col], kde=True, color='red')
    plt.title(col, fontsize=14)

plt.tight_layout()
plt.show()

# CORRELATION MATRIX
# Select only numeric columns
numeric_df = happy_df.select_dtypes(include=['number'])

# Compute correlation matrix
corr_matrix = numeric_df.corr()
print("\nCorrelation Matrix:\n", corr_matrix)

# Heatmap visualization
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt=".2f")
plt.title("Correlation Matrix Heatmap", fontsize=16)
plt.show()

# SCATTER PLOTS WITH PLOTLY (INTERACTIVE)
# Scatter: Score vs GDP Per Capita
fig = px.scatter(
    happy_df,
    x='GDP per capita',
    y='Score',
    text='Country or region',
    title='Happiness Score vs GDP per Capita'
)
fig.show()

# Scatter with color & size
fig = px.scatter(
    happy_df,
    x='GDP per capita',
    y='Score',
    text='Country or region',
    size='Overall rank',
    color='Country or region',
    title='Happiness Score vs GDP per Capita (Colored & Sized)'
)
fig.show()

# Scatter: Score vs Freedom to Make Life Choices
fig = px.scatter(
    happy_df,
    x='Freedom to make life choices',
    y='Score',
    size='Overall rank',
    color='Country or region',
    hover_name='Country or region',
    trendline='ols',
    title='Happiness Score vs Freedom to Make Life Choices'
)
fig.show()

# SCATTER PLOTS: HEALTHY LIFE EXPECTANCY VS HAPPINESS SCORE
# Simple scatter
fig = px.scatter(
    happy_df,
    x='Healthy life expectancy',
    y='Score',
    text='Country or region'
)
fig.update_traces(textposition='top center')
fig.update_layout(height=800, title='Happiness Score vs Healthy Life Expectancy')
fig.show()

# Scatter with color, size and regression line
fig = px.scatter(
    happy_df,
    x='Healthy life expectancy',
    y='Score',
    size='Overall rank',
    color='Country or region',
    hover_name='Country or region',
    trendline='ols',
    title='Happiness Score vs Healthy Life Expectancy (Regression Included)'
)
fig.show()

# PREPARE DATA FOR CLUSTERING

# Remove non-numeric and outcome columns
df_seg = happy_df.drop(columns=['Overall rank', 'Country or region', 'Score'])
print("\nData used for clustering:\n")
print(df_seg.head())

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_seg)

print("\nScaled data shape:", scaled_data.shape)

# FIND OPTIMAL NUMBER OF CLUSTERS – ELBOW METHOD
inertia_scores = []
range_values = range(1, 20)

for k in range_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_data)
    inertia_scores.append(km.inertia_)

# Proper elbow graph (single clean plot)
plt.figure(figsize=(10, 6))
plt.plot(range_values, inertia_scores, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia / Distortion Score')
plt.title('Elbow Method to Determine Optimal k')
plt.grid(True)
plt.show()

#RUN K-MEANS WITH OPTIMAL k = 3
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)

labels = kmeans.labels_
print("\nCluster labels for each country:\n", labels)

#  EXTRACT & INTERPRET CLUSTER CENTERS

# Cluster centers in scaled form
print("\nCluster centers (scaled values):")
print(kmeans.cluster_centers_)

# Convert scaled cluster centers back to original values
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

cluster_centers_df = pd.DataFrame(
    cluster_centers,
    columns=df_seg.columns
)

print("\nCluster centers (original units):\n")
print(cluster_centers_df)

# CHECK CLUSTER LABEL INFORMATION

print("Labels shape:", labels.shape)        # One label per country
print("Max label:", labels.max())           # Highest cluster ID
print("Min label:", labels.min())           # Lowest cluster ID

# Predict labels again (optional check)
y_kmeans = kmeans.fit_predict(scaled_data)
print("\nPredicted labels:\n", y_kmeans)

# ADD CLUSTER LABELS TO ORIGINAL DATAFRAME

happy_df_cluster = pd.concat([happy_df, pd.DataFrame({'cluster': labels})], axis=1)
print("\nDataFrame with cluster labels added:\n")
print(happy_df_cluster.head())

# HISTOGRAMS FOR EACH FEATURE ACROSS CLUSTERS

for col in df_seg.columns:
    plt.figure(figsize=(35, 10))
    for j in range(3):    # 3 clusters
        plt.subplot(1, 3, j+1)
        cluster = happy_df_cluster[happy_df_cluster['cluster'] == j]
        cluster[col].hist(bins=20)
        plt.title(f"{col}\nCluster {j}")
    plt.show()


# MODEL WITH 4 CLUSTERS
kmeans4 = KMeans(n_clusters=4, random_state=42)
kmeans4.fit(scaled_data)

labels4 = kmeans4.labels_

print("\nLabels for 4 clusters:\n", labels4)
print("\nShape of cluster centers:", kmeans4.cluster_centers_.shape)

print("\nCluster centers (scaled values):\n")
print(kmeans4.cluster_centers_)

# CREATE CLUSTER CENTERS DATAFRAME
cluster_centers4 = pd.DataFrame(
    data=kmeans4.cluster_centers_,
    columns=df_seg.columns
)

print("\nCluster centers in scaled form:\n")
print(cluster_centers4)

# INVERSE TRANSFORM TO ORIGINAL UNITS

cluster_centers4_original = scaler.inverse_transform(cluster_centers4)
cluster_centers4_original = pd.DataFrame(
    data=cluster_centers4_original,
    columns=df_seg.columns
)

print("\nCluster centers in original units (interpretable):\n")
print(cluster_centers4_original)

# LABEL INFORMATION FOR 4 CLUSTERS

print("\nLabels shape:", labels4.shape)
print("Max label:", labels4.max())
print("Min label:", labels4.min())

# Predict cluster labels (already fitted earlier)

y_kmeans = kmeans.fit_predict(scaled_data)
print("\nPredicted Labels:\n", y_kmeans)

# 'labels' created earlier = kmeans.labels_ (same as y_kmeans)
happy_df_cluster = pd.concat(
    [happy_df, pd.DataFrame({'cluster': labels})],
    axis=1
)

print("\nData with Cluster Labels Added:\n")
print(happy_df_cluster.head())

# HISTOGRAMS FOR EACH FEATURE ACROSS CLUSTERS

for feature in df_seg.columns:
    plt.figure(figsize=(35, 10))
    for c in range(3):  # For 3 clusters
        plt.subplot(1, 3, c + 1)
        cluster_group = happy_df_cluster[happy_df_cluster['cluster'] == c]
        cluster_group[feature].hist(bins=20)
        plt.title(f"{feature}\nCluster {c}", fontsize=16)
    plt.show()

# VISUALIZE CLUSTERS — SCORE VS CLUSTER
fig = px.scatter(
    happy_df_cluster,
    x='cluster',
    y='Score',
    size='Overall rank',
    color='Country or region',
    hover_name='Country or region',
    trendline='ols',
    title='Happiness Score vs Cluster'
)
fig.show()

#  VISUALIZE CLUSTERS — GDP per Capita VS CLUSTER

fig = px.scatter(
    happy_df_cluster,
    x='cluster',
    y='GDP per capita',
    size='Overall rank',
    color='Country or region',
    hover_name='Country or region',
    trendline='ols',
    title='GDP per Capita vs Cluster'
)
fig.show()

# ADVANCED VISUALIZATION — 3D CLUSTER PLOT: Economy,Corruption and Life Expectancy

fig1 = px.scatter_3d(
    happy_df_cluster,
    x='GDP per capita',
    y='Perceptions of corruption',
    z='Healthy life expectancy',
    color='cluster',
    size='Score',
    hover_name='Country or region',
    title='3D Cluster Plot: Economy, Corruption & Life Expectancy',
    color_continuous_scale='Portland'
)
fig1.update_traces(marker=dict(opacity=0.8))
fig1.show()

# HAPPINESS SCORE VS OVERALL RANK (Colored by Cluster)

fig2 = px.scatter(
    happy_df_cluster,
    x='Overall rank',
    y='Score',
    color='cluster',
    hover_name='Country or region',
    title='Happiness Score vs Overall Rank (Cluster Colored)',
    color_continuous_scale='Portland'
)
fig2.update_traces(marker=dict(size=10, opacity=0.7))
fig2.show()

#   IMPORTS
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

#   CHECK COLUMNS
print("Columns available:", happy_df_cluster.columns.tolist())

#  3D Scatter Plot: GDP vs Corruption vs Generosity
fig1 = px.scatter_3d(
    happy_df_cluster,
    x='GDP per capita',
    y='Perceptions of corruption',
    z='Generosity',
    color='cluster',
    size='Score',
    hover_name='Country or region',
    title='3D Cluster Plot: Economy, Corruption & Generosity',
    color_continuous_scale='Portland'
)

fig1.update_traces(marker=dict(opacity=0.8))
fig1.show()

# 2D Scatter Plot: Overall Rank vs Happiness Score
fig2 = px.scatter(
    happy_df_cluster,
    x='Overall rank',
    y='Score',
    color='cluster',
    hover_name='Country or region',
    title='Happiness Score vs Overall Rank',
    color_continuous_scale='Portland'
)

fig2.update_traces(marker=dict(size=10, opacity=0.7))
fig2.show()

# Pairplot with Generosity instead of Life Expectancy
sns.pairplot(
    happy_df_cluster,
    vars=['GDP per capita', 'Perceptions of corruption', 'Generosity', 'Score'],
    hue='cluster',
    palette='Set2'
)
plt.suptitle('Pairplot of Features (including Generosity) Colored by Cluster', y=1.02)
plt.show()

#  Geographical Visualization of Clusters
fig = px.choropleth(
    happy_df_cluster,
    locations='Country or region',
    locationmode='country names',
    color='cluster',
    hover_name='Country or region',
    hover_data={
        'Score': True,
        'GDP per capita': True,
        'Perceptions of corruption': True,
        'Generosity': True,
        'cluster': True
    },
    color_continuous_scale='Portland',
    title='Geographical Visualization of Happiness Clusters'
)

fig.update_geos(
    showcountries=True,
    showcoastlines=True,
    showland=True,
    fitbounds="locations"
)

fig.update_layout(height=600)
fig.show()
