import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import pickle
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
import numpy as np
import plotly.express as px


st.set_page_config(
page_title="Explaniability",
page_icon="💡",
layout="wide",
initial_sidebar_state="expanded")
st.set_option('deprecation.showPyplotGlobalUse', False)
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
# Function to shorten categories
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

@st.cache_data
def load_data():
    df = pd.read_csv("car_ad_display.csv", encoding="ISO-8859-1", sep=";").drop(columns='Unnamed: 0')
    df = df.dropna()
    car_map = shorten_categories(df.car.value_counts(), 10)
    df['car'] = df['car'].map(car_map)

    model_map = shorten_categories(df.model.value_counts(), 10)
    df['model'] = df['model'].map(model_map)

    df = df[(df["price"] <= 100000) & (df["price"] >= 1000) & (df["mileage"] <= 600) & 
            (df["engV"] <= 7.5) & (df["year"] >= 1975)]

    return df
def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

data = load_model()
model = data["model"]
le_car = data["le_car"]
le_body = data["le_body"]
le_engType = data["le_engType"]
le_drive = data["le_drive"]

df = load_data()

df['car'] = le_car.transform(df['car'])
df['body'] = le_body.transform(df['body'])
df['engType'] = le_engType.transform(df['engType'])
df['drive'] = le_drive.transform(df['drive'])

df = df.drop(columns='model')
yes_l = ['yes', 'YES', 'Yes', 'y', 'Y']
df['registration'] = np.where(df['registration'].isin(yes_l), 1, 0)

# Assume 'model' is your trained predictive model

# Split data for Shap explanations
X = df.drop("price", axis=1)
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# Create Shap explainer
explainer = shap.TreeExplainer(model)
shap_res = explainer(X_test)
shap_values = explainer.shap_values(X_test)

# Streamlit app
st.title('Car Price Prediction Model Explainability')

st.subheader('Shap Summary Plot:')
# Display insights
st.markdown("""
The SHAP summary plot provides a comprehensive overview of feature importance and impact on the model prediction. Here are some insights from your model’s explainability:

1. **Year:** This feature has a significant impact on the model output. Higher values (more recent years) tend to increase the prediction, while lower values (older years) decrease it.

2. **EngV:** This feature also influences the model output considerably. Both low and high values of EngV can increase or decrease the model prediction.

3. **Car:** The type of car has a substantial effect on the model output. Different car types can lead to higher or lower predictions.

4. **Driver:** This feature has a noticeable impact on the model output. However, the direction of the impact (positive or negative) depends on the specific driver.

5. **Mileage:** Mileage also affects the model output. Higher mileage tends to decrease the prediction, while lower mileage increases it.

6. **EngType:** The engine type has a significant effect on the model output. Different engine types can lead to higher or lower predictions.

7. **Registration:** Although it impacts the model output, its effect is relatively lower compared to other features.

8. **Body:** Similar to registration, the body type of the car also influences the model output but has a lesser impact compared to other features.

Remember, the color represents the feature value (blue for low and pink for high), and the horizontal location shows whether the effect of that value causes a higher or lower prediction.
""")
shap.summary_plot(shap_res, X_test)
st.pyplot(plt.gcf(), use_container_width=True)

# Display individual feature plots
st.subheader('Individual Feature Plots:')
st.markdown(
"""
This scatter plot shows the SHAP values for the different features. Here's how to get insights on one feature "car"':

1. **Distribution of SHAP values:** The plot shows a wide range of SHAP values, with some values above 0 and some below 0. This indicates that the feature “car” has both positive and negative impacts on the model prediction.

2. **Trend:** The plot appears to have a slight negative trend, with higher SHAP values at lower values of “car”. This suggests that lower values of “car” tend to increase the model prediction, while higher values decrease it.

If you select other features for this plot, you can expect to see a similar distribution of SHAP values. However, the trend might be different depending on the relationship between the selected feature and the model prediction. For example, some features might show a positive trend, indicating that higher feature values increase the model prediction.""")

feature_to_plot = st.selectbox('Select Feature:', X_test.columns)
shap.plots.scatter(shap_res[:, feature_to_plot])
st.pyplot()


st.subheader('Shap Force Plot:')
st_shap(shap.force_plot(explainer.expected_value, shap_values, X_test), 400)
st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:]))

# Display dependence plots
st.subheader('Dependence Plots:')
st.markdown(""" 
This plot reads similar to the prvious one. Here, color represents the shap values of the variable you are comparing against.
""")
st.markdown('Select features to analyze their Shap values:')
feature_x = st.selectbox('Select X Feature:', X_test.columns)
feature_interaction = st.selectbox('Select Interaction Feature:', X.columns)
shap.dependence_plot(feature_x, shap_res.values, X_test, interaction_index=feature_interaction)
st.pyplot()

# Display Shap waterfall plot for a specific instance
st.subheader('Shap Waterfall Plot:')
instance_index = st.slider('Select Instance:', 0, len(X_test) - 1, 0)
shap.plots.waterfall(shap_res[instance_index])
st.pyplot()

st.subheader('Shap Decision Plot:')
# Display Shap decision plot for a specific instance
st.markdown("""This plot shows the contribution of each feature to the model’s prediction for a specific instance1. Here’s how to interpret it:

The x-axis represents the model output value. This is the prediction of the model for the specific instance.
The y-axis lists the features used by the model.
            
Each line in the plot represents a feature. The position of the line along the x-axis shows the effect of that feature on the model’s prediction.
The order of the features on the y-axis is determined by the magnitude of their effects.
The color of the line indicates the value of the feature.
Here, the plot is showing how each attribute  year, mileage, drive, etc.) is contributing to the final prediction of the model. The blue line represents the cumulative SHAP value as you move from the base value (the average prediction over the test dataset) to the model output1. The points on the line represent the SHAP values of the features. The order of the points shows the order of impact of the features.
For example, if the ‘year’ feature is at the top of the y-axis and its line extends far to the right on the x-axis, this means that the ‘year’ feature has a strong positive effect on the model’s prediction for this instance. If the ‘year’ line is colored red, this means that the car’s year has a high value.""")


shap.decision_plot(shap_res[instance_index].base_values, shap_res[instance_index].values, X_test.iloc[instance_index])
st.pyplot()