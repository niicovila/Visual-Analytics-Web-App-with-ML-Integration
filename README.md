# Visual Analytics WebApp with ML Integration and Explainability
This project is a web app that provides exploratory data analysis and car price predictions. The project consists of two main parts:

- A Jupyter Notebook that performs exploratory data analysis (EDA), data preparation, machine learning modeling and model explainability on the car data.

- A Streamlit app that allows users to interactively explore the data and get car price predictions based on several features.
  
## Features
  1. **Data Exploration**: This page includes several interactive plots to visualize and analyze the data.
     
https://github.com/niicovila/Visual-Analytics-Web-App-with-ML-Integration/assets/76247144/4a6dc30e-639b-4b06-9942-3c34cd2c36fa


  2. **Predictive Model**: This feature simplifies the exploration of data by allowing users to select different features to obtain price predictions. Users can interactively choose the features they're interested in, such as mileage, year, or model, and receive instant predictions. These predictions are then conveniently displayed and stored in a table format, making it easy for users to compare and analyze the impact of different features on the predicted prices.

https://github.com/niicovila/Visual-Analytics-Web-App-with-ML-Integration/assets/76247144/fa42ceb8-c366-4d44-8391-b89f16028986


  3. **Model Explainability**: This feature helps users understand how our predictive model works by using SHAP (SHapley Additive exPlanations) values. These values show the contribution of each feature to a prediction. By visualizing SHAP values, users can easily see which factors influence the model's decisions the most. This makes it simpler to trust and interpret the model's predictions, leading to more informed decision-making.
  

https://github.com/niicovila/Visual-Analytics-Web-App-with-ML-Integration/assets/76247144/476dfff8-071a-4f7f-8f7d-bbc433288e8c


## Installation
To run this project, you need to have Python 3.7 or higher and the following packages installed:
^
```
streamlit
pandas
numpy
altair
scikit-learn
plotly
pickle
matplotlib
shap
```
## Usage
**Streamlit App**:
```
cd streamlit_app/
streamlit run Welcome.py
```

**Jupyter Notebook**:
```
jupyter notebook .ipynb
```
