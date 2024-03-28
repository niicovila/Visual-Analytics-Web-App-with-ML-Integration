import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import numpy as np
import altair as alt

# Set Streamlit page configuration
st.set_page_config(
    page_title="Data Exploration",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(to top, #00008b, #4b0082);
        opacity: 0.7;
        color: white;
    }
    [data-testid="stSidebar"] .sidebar-content {
        
    }
    </style>
    """, unsafe_allow_html=True)

# The title and text
st.title("Data Exploration 📊 ")
st.write("In this tab, we can see the most relevant information extracted through visual analytics.")

# Function to shorten categories
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("car_ad_display.csv", encoding="ISO-8859-1", sep=";").drop(columns='Unnamed: 0')

    car_map = shorten_categories(df.car.value_counts(), 10)
    df['car'] = df['car'].map(car_map)

    model_map = shorten_categories(df.model.value_counts(), 10)
    df['model'] = df['model'].map(model_map)

    df = df[(df["price"] <= 100000) & (df["price"] >= 1000) & (df["mileage"] <= 600) & 
            (df["engV"] <= 7.5) & (df["year"] >= 1975)]

    return df

df = load_data()

# Basic info
st.subheader("How does the data look like?")
rows, columns = df.shape
st.write(f"We have {rows} rows and {columns} columns.")
st.dataframe(df.head(3), use_container_width=True)

# Target distribution
st.subheader("Distribution of the target:")
fig = px.histogram(df, x='price', nbins=50, title='Prices Distribution', labels={'price': 'Price'})
st.plotly_chart(fig, use_container_width=True)

# Top 10 Most expensive cars
st.subheader("Top 10 Most Expensive Cars:")
df_priceByCar = df[['car', 'price']].groupby('car').mean().reset_index().sort_values('price', ascending=False).head(10)
fig = px.bar(df_priceByCar, x='price', y='car', orientation='h', title='Average Price by Car', labels={'price': 'Average Price', 'car': 'Car'}, color='car', color_discrete_sequence=px.colors.qualitative.Plotly)
st.plotly_chart(fig, use_container_width=True)

# Explore data distributions
st.subheader("Explore Data Distributions:")
columns_to_explore = [x for x in df.columns if x not in ["car", "model"]]

# Create a list to hold the charts
charts_plotly = []

# Generate Plotly charts based on plot_types
for col in columns_to_explore:
    if df[col].dtype == 'object' or df[col].nunique() < 10:
        # For categorical and discrete data, use a count plot (bar chart)
        chart_plotly = px.histogram(df, x=col, title=f'Count Plot of {col}', labels={col: col}, color=col, width=400, height=400)
    else:
        # For continuous data, use a KDE plot
        chart_plotly = px.histogram(df, x=col, marginal='box', title=f'Density Plot of {col}', labels={col: col}, color_discrete_sequence=['lightblue'], width=400, height=400)
    
    charts_plotly.append(chart_plotly)

# Display three charts per row
for i in range(0, len(charts_plotly), 3):
    cols_plotly = st.columns(3)
    for j in range(3):
        if i + j < len(charts_plotly):
            cols_plotly[j].plotly_chart(charts_plotly[i + j])

# Features distribution vs target with larger size
st.subheader("Features Distribution vs Target:")
target = 'price'
features = [x for x in df.columns if x not in ["car", "model", target]]
# Create a list to hold the charts
charts = []
for feature in features:
    if df[feature].dtype == 'object' or df[feature].nunique() < 10:
        # For categorical data, use a boxplot
        chart = alt.Chart(df).mark_boxplot(color='orange').encode(
            x=feature,
            y=target
        ).properties(
            title=f'{feature} vs {target}',
            width=400,
            height=400
        ).interactive()
    else:
        # For numerical data, use a scatter plot
        chart = alt.Chart(df).mark_circle().encode(
            x=feature,
            y=target
        ).properties(
            title=f'{feature} vs {target}',
            width=400,
            height=400
        ).interactive()

    charts.append(chart)
# Display three charts per row
for i in range(0, len(charts), 3):
    cols = st.columns(3)
    for j in range(3):
        if i + j < len(charts):
            cols[j].altair_chart(charts[i + j])

# Yearly price trends
st.subheader("Yearly Price Trends:")
st.subheader("Line Plot for Average Price Over the Years:")
df_priceByYear_altair = df[['year', 'price']].groupby('year').mean().reset_index()
chart_line_price_over_years_altair = alt.Chart(df_priceByYear_altair).mark_line().encode(
    x='year:O',
    y='price:Q'
).properties(
    title='Line Plot for Average Price Over the Years',
    width=500
)
st.altair_chart(chart_line_price_over_years_altair, use_container_width=True)

st.subheader("Yearly Price Evolution Animation:")
df['year'] = pd.to_numeric(df['year'])
# Sort dataframe by year
df = df.sort_values('year')
fig_animated_bubble = px.scatter(df, x='year', y='price', animation_frame='year', size='price', color='car',
                                 title='Animated Bubble Chart for Price Trends Over Time', labels={'price': 'Price'},
                                 size_max=60)  # Increase the size of the points
fig_animated_bubble.update_layout(xaxis=dict(range=[df['year'].min(), df['year'].max()]),
                                  yaxis=dict(range=[df['price'].min()*(-1.2), df['price'].max()]),
                                  sliders=dict(range=[df['year'].min(), df['year'].max()]),
                                  showlegend=False)
st.plotly_chart(fig_animated_bubble, use_container_width=True)

# Relationships between numerical features
st.subheader("Analyze Relations between Numerical Features:")
st.subheader("Parallel Coordinates plot:")
fig_parallel = px.parallel_coordinates(df, dimensions=['mileage', 'engV', 'year', 'price'], color='price', title='Parallel Coordinates Plot for Numerical Features')
st.plotly_chart(fig_parallel, use_container_width=True)

col1, col2 = st.columns(2)

# First column
with col1:
    st.subheader("3D Scatter Plot for Mileage, Engine Volume, and Price:")
    fig_3d_scatter_updated = px.scatter_3d(df, x='mileage', y='engV', z='price', title='3D Scatter Plot for Mileage, Engine Volume, and Price', labels={'mileage': 'Mileage', 'engV': 'Engine Volume', 'price': 'Price'}, color='engType', size_max=5, width=800, height=600)
    st.plotly_chart(fig_3d_scatter_updated, use_container_width=True)

# Second column
with col2:
    st.subheader("3D Scatter Plot for Mileage, Engine Volume, and Year:")
    fig_3d_scatter_updated = px.scatter_3d(df, x='mileage', y='engV', z='year', title='3D Scatter Plot for Mileage, Engine Volume, and Year', labels={'mileage': 'Mileage', 'engV': 'Engine Volume', 'price': 'Price'}, color='engType', size_max=5, width=800, height=600)
    st.plotly_chart(fig_3d_scatter_updated, use_container_width=True)

# Label encoding
st.subheader("Correlation Analysis:")
df_corr = df.copy()

le_car = LabelEncoder()
df_corr['car'] = le_car.fit_transform(df_corr['car'])

le_body = LabelEncoder()
df_corr['body'] = le_body.fit_transform(df_corr['body'])

le_engType = LabelEncoder()
df_corr['engType'] = le_engType.fit_transform(df_corr['engType'])

le_drive = LabelEncoder()
df_corr['drive'] = le_drive.fit_transform(df_corr['drive'])

yes_l = ['yes', 'YES', 'Yes', 'y', 'Y']
df_corr['registration'] = np.where(df_corr['registration'].isin(yes_l), 1, 0)
df_corr = df_corr.drop(columns='model')

# Correlation chart using Plotly Express
corr = df_corr.corr()
fig_corr_heatmap = px.imshow(corr, x=corr.index, y=corr.columns, color_continuous_scale='viridis', title='Correlation Heatmap')
st.plotly_chart(fig_corr_heatmap, use_container_width=True)







