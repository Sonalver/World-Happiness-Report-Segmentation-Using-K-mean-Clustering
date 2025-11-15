# IMPORT DATASETS AND LIBRARIES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from chart_studio.plotly import iplot

# Import csv file into pandas dataframe
happy_df = pd.read_csv('C:/Users/sonal/PycharmProjects/PythonProject/resources/happiness_report.csv')
print(happy_df)

# print the first 5 rows of the dataframe
print(happy_df.head)
print(happy_df.shape)

#Find out how many samples exist in the DataFrame using two different methods.
print(len(happy_df))
print(happy_df.tail)

print(happy_df[happy_df['Country or region'] == 'Canada'])
print(happy_df[happy_df['Country or region'] == 'India'])

#PERFORM EXPLORATORY DATA ANALYSIS
# Check the number of non-null values in the dataframe
print(happy_df.info())

# Check Null values
print(happy_df.isnull().sum())

# Obtain the Statistical summary of the dataframe
print(happy_df.describe())

# check the number of duplicated entries in the dataframe
print(happy_df.duplicated().sum())

#country that has the maximum happiness score? What is the perception of corruption in this country
#print(happy_df[happy_df['Score'] == '7.769'])

max_score = happy_df['Score'].max()
result = happy_df[happy_df['Score'] == max_score]
print(result)

#PERFORM DATA VISUALIZATION - PART #1
# Plot the pairplot
fig = plt.figure(figsize = (20,20))
sns.pairplot(happy_df[['Score','GDP per capita', 'Social support', 'Healthy life expectancy',
    'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']])
plt.show()

# Positive correlation between GDP and score
# Positive correlation between Social Support and score

# distplot combines the matplotlib.hist function with seaborn kdeplot()
columns = ['Score','GDP per capita', 'Social support', 'Healthy life expectancy',
    'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption']
plt.figure(figsize = (20, 50))
for i in range(len(columns)):
  plt.subplot(8, 2, i+1)
  sns.histplot(happy_df[columns[i]], color = 'r');
  #sns.displot(data=happy_df, x=happy_df[columns[i]], color='b')
  #sns.kdeplot(happy_df[columns[i]], color='g')
  plt.title(columns[i])
plt.tight_layout()
plt.show()

#Plot the correlation matrix and comment on the results.
# Get the correlation matrix
# Keep only numeric columns
numeric_df = happy_df.select_dtypes(include=['number'])

# Compute correlation matrix
corr_matrix = numeric_df.corr()
print("Correlation Matrix:\n", corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', fmt='.2f')
plt.title("Correlation Matrix Heatmap")
plt.show()

# PERFORM DATA VISUALIZATION - PART #2
# Plot the relationship between score, GDP and region
fig = px.scatter(happy_df, x = 'GDP per capita', y = 'Score', text = 'Country or region')
fig.show()

# Plot the relationship between score and GDP (while adding color and size)
fig = px.scatter(happy_df, x = 'GDP per capita', y = 'Score', text = 'Country or region', size ='Overall rank', color = 'Country or region')
fig.update_layout(title_text = 'Happiness Score vs GDP per Capita')
fig.show()

# Plot the relationship between score and freedom to make life choices
fig = px.scatter(happy_df, x = 'Freedom to make life choices', y = "Score", size = 'Overall rank', color = "Country or region", hover_name = "Country or region",
          trendline = "ols")
fig.update_layout(title_text = 'Happiness Score vs Freedom to make life choices')
fig.show()

#plots for 'Healthy life expectancy' and 'Score'
# Plot the relationship between score and healthy life expectancy
fig = px.scatter(happy_df, x = 'Healthy life expectancy', y = "Score", text = 'Country or region')
fig.update_traces(textposition = 'top center')
fig.update_layout(height = 1000)
fig.show()

# Plot the relationship between score and healthy life expectancy
fig = px.scatter(happy_df, x = 'Healthy life expectancy', y = "Score",
           size = 'Overall rank', color = "Country or region", hover_name = "Country or region",
          trendline = "ols")
fig.update_layout(
    title_text = 'Happiness Score vs Healthy life expectancy'
)
fig.show()

#PREPARE THE DATA TO FEED THE CLUSTERING MODEL
# to create clusters without the use of happiness score and rank to see which countries fall under similar clusters
df_seg = happy_df.drop(columns = ['Overall rank','Country or region','Score'])
print(df_seg)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_seg)
scaled_data.shape

#FIND THE OPTIMAL NUMBER OF CLUSTERS USING ELBOW METHOD
score = []

range_values= range(1,20)

for i in range_values:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(scaled_data)
    score.append(kmeans.inertia_)

    plt.plot(score, 'bx-')
    plt.title('Finding right number of clusters')
    plt.xlabel('Clusters')
    plt.ylabel('scores')
    plt.show()

# From this we can observe that 3rd cluster seems to be forming the elbow of the curve.
# Let's choose the number of clusters to be 3.
kmeans = KMeans(3)
kmeans.fit(scaled_data)

print(kmeans.labels_)
labels = kmeans.labels_
print(kmeans.cluster_centers_.shape)

print(kmeans.cluster_centers_)

cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [df_seg.columns])
print('cluster_centers')

# In order to understand what these numbers mean, let's perform inverse transformation
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [df_seg.columns])
print('cluster_centers')

print(labels.shape) # Labels associated to each data point
print(labels.max())
print(labels.min())

y_kmeans = kmeans.fit_predict(scaled_data)
print(y_kmeans)

# concatenate the clusters labels to our original dataframe
happy_df_cluster = pd.concat([happy_df, pd.DataFrame({'cluster':labels})], axis = 1)
print(happy_df_cluster)

# Plot the histogram of various clusters
for i in df_seg.columns:
    plt.figure(figsize=(35, 10))
    for j in range(3):
        plt.subplot(1, 3, j + 1)
        cluster = happy_df_cluster[happy_df_cluster['cluster'] == j]
        cluster[i].hist(bins=20)
        plt.title('{}    \nCluster {} '.format(i, j))

    plt.show()

#Try the same model with 4 clusters
kmeans = KMeans(4)
kmeans.fit(scaled_data)

print(kmeans.labels_)
labels = kmeans.labels_
print(kmeans.cluster_centers_.shape)

print(kmeans.cluster_centers_)

cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [df_seg.columns])
print('cluster_centers')

# In order to understand what these numbers mean, let's perform inverse transformation
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [df_seg.columns])
print('cluster_centers')

print(labels.shape) # Labels associated to each data point
print(labels.max())
print(labels.min())

y_kmeans = kmeans.fit_predict(scaled_data)
print(y_kmeans)

# concatenate the clusters labels to our original dataframe
happy_df_cluster = pd.concat([happy_df, pd.DataFrame({'cluster':labels})], axis = 1)
print(happy_df_cluster)

# Plot the histogram of various clusters
for i in df_seg.columns:
    plt.figure(figsize=(35, 10))
    for j in range(3):
        plt.subplot(1, 3, j + 1)
        cluster = happy_df_cluster[happy_df_cluster['cluster'] == j]
        cluster[i].hist(bins=20)
        plt.title('{}    \nCluster {} '.format(i, j))

    plt.show()

#VISUALIZE THE CLUSTERS
print(happy_df_cluster)

# Plot the relationship between cluster and score

fig = px.scatter(happy_df_cluster, x = 'cluster', y = "Score",
           size = 'Overall rank', color = "Country or region", hover_name = "Country or region",
          trendline = "ols")

fig.update_layout(
    title_text = 'Happiness Score vs Cluster'
)
fig.show()

# Plot the relationship between cluster and GDP

fig = px.scatter(happy_df_cluster, x='cluster', y='GDP per capita',
           size='Overall rank', color="Country or region", hover_name="Country or region",
          trendline= "ols")

fig.update_layout(
    title_text='GDP vs Clusters'
)
fig.show()

# Visaulizing the clusters with respect to economy, corruption, gdp, rank and their scores
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# 3D Scatter Plot: GDP vs Corruption vs Life Expectancy
fig1 = px.scatter_3d(happy_df_cluster,
                     x='GDP per capita',
                     y='Perceptions of corruption',
                     z='Healthy life expectancy',
                     color='cluster',
                     size='Score',
                     hover_name='Country or region',
                     title='3D Cluster Plot: Economy, Corruption & Life Expectancy',
                     color_continuous_scale='Portland')
fig1.update_traces(marker=dict(opacity=0.8))
fig1.show()

#2D Plot: Overall Rank vs Happiness Score
fig2 = px.scatter(happy_df_cluster,
                  x='Overall rank',
                  y='Score',
                  color='cluster',
                  hover_name='Country or region',
                  title='Happiness Score vs Overall Rank',
                  color_continuous_scale='Portland')
fig2.update_traces(marker=dict(size=10, opacity=0.7))
fig2.show()

#Pairplot: All key numeric variables colored by cluster
sns.pairplot(happy_df_cluster,
             vars=['GDP per capita', 'Perceptions of corruption', 'Healthy life expectancy', 'Score'],
             hue='cluster', palette='Set2')
plt.suptitle('Pairplot of Key Features Colored by Cluster', y=1.02)
plt.show()
#Plot the similar type of visualization having 'Generosity' instead of 'Healthy life expectancy'
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Check that the column 'Generosity' exists
print(happy_df_cluster.columns)

# 1️⃣ 3D Scatter Plot: GDP vs Corruption vs Generosity
fig1 = px.scatter_3d(happy_df_cluster,
                     x='GDP per capita',
                     y='Perceptions of corruption',
                     z='Generosity',
                     color='cluster',
                     size='Score',
                     hover_name='Country or region',
                     title='3D Cluster Plot: Economy, Corruption & Generosity',
                     color_continuous_scale='Portland')
fig1.update_traces(marker=dict(opacity=0.8))
fig1.show()

# 2️⃣ 2D Scatter Plot: Overall Rank vs Happiness Score
fig2 = px.scatter(happy_df_cluster,
                  x='Overall rank',
                  y='Score',
                  color='cluster',
                  hover_name='Country or region',
                  title='Happiness Score vs Overall Rank',
                  color_continuous_scale='Portland')
fig2.update_traces(marker=dict(size=10, opacity=0.7))
fig2.show()

# 3️⃣ Pairplot with Generosity instead of Life Expectancy
sns.pairplot(happy_df_cluster,
             vars=['GDP per capita', 'Perceptions of corruption', 'Generosity', 'Score'],
             hue='cluster', palette='Set2')
plt.suptitle('Pairplot of Key Features (including Generosity) Colored by Cluster', y=1.02)
plt.show()

## Visualizing the clusters geographically
import plotly.express as px

fig = px.choropleth(happy_df_cluster,
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
                    title='Geographical Visualization of Happiness Clusters')

fig.update_geos(showcountries=True, showcoastlines=True, showland=True, fitbounds="locations")
fig.update_layout(height=600)
fig.show()
