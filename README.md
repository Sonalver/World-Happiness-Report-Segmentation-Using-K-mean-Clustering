# World-Happiness-Report-Segmentation-Using-K-mean-Clustering
Using data of the World Happiness Report, the K-Means clustering method is used to classify the happiness indexes of different countries to reveal the significant characteristic differences in the happiness of countries.
Happiness Score Clustering & Visualization
A complete exploratory analysis of the World Happiness dataset using clustering and advanced visualizations.
Project Overview
This project performs clustering on the World Happiness Report dataset and visualizes global patterns using:
3D scatter plots (Plotly)
2D interactive scatter
Pairplots (Seaborn)
Geographical choropleth map
Cluster-based insights
The analysis helps uncover relationships between GDP, corruption perception, generosity, and overall happiness across countries.

Dataset Description
The dataset contains key indicators affecting the happiness score of countries, including:
Feature	Description
Country or region	Name of the country
Score	Happiness score
GDP per capita	Economic indicator
Healthy life expectancy	Average life expectancy
Generosity	Philanthropic behavior
Perceptions of corruption	Trust in institutions
Overall rank	Global happiness ranking
cluster	Cluster assigned using K-means

The cluster column is generated during preprocessing to segment countries into meaningful groups.
Key Tasks Performed
1. Data Loading & Exploration
Preview dataset
Check column availability
Prepare selected features for clustering
2. Clustering (K-Means or similar)
Cluster countries into groups based on economic, social, and behavioral factors.
3. Visualizations
 1. 3D Scatter Plot
Shows interaction between:
GDP per capita
Perceptions of corruption
Generosity
Cluster labels
Helps visualize how generosity and corruption differentiate clusters.

 2. 2D Scatter Plot
Plots:
Overall rank vs Happiness score
Reveals how clusters differ in ranking.
 3. Pairplot (Seaborn)
Compares:GDP per capita,Generosity,Corruption,Score-Great for understanding cross-variable relationships.
 4. World Map (Choropleth)
 Visualizes country clusters geographically, showing regional patterns.

Insights & Findings
 High-happiness clusters
High GDP
Low corruption
High generosity
 Mid-level clusters
Moderate GDP and Score
Mixed corruption perception
 Low-happiness clusters
Low GDP
High corruption
Lower generosity
 Geographical patterns
Scandinavian countries cluster together at the top
Sub-Saharan countries cluster near the bottom
South Asia falls into mid-to-low clusters

Key Correlations That Shape Happiness-
GDP per Capita and Social Support show the strongest positive correlation with Happiness Score.
Freedom to make life choices and Healthy Life Expectancy also contribute significantly.
Perceptions of Corruption negatively correlates with happiness—countries with low corruption tend to be happier.
Generosity, while weaker, still shows a positive effect.
These correlations confirm that economic stability, social well-being, and trust in public systems play major roles in determining happiness.
