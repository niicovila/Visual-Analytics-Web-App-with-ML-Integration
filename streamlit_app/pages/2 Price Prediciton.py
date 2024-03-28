import pickle
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
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
# Load the pickle file with the model and the label encoders
def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data["model"]
le_car = data["le_car"]
le_body = data["le_body"]
le_engType = data["le_engType"]
le_drive = data["le_drive"]

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

    car_map = shorten_categories(df.car.value_counts(), 10)
    df['car'] = df['car'].map(car_map)

    model_map = shorten_categories(df.model.value_counts(), 10)
    df['model'] = df['model'].map(model_map)

    df = df[df["price"] <= 100000]
    df = df[df["price"] >= 1000]
    df = df[df["mileage"] <= 600]
    df = df[df["engV"] <= 7.5]
    df = df[df["year"] >= 1975]

    return df

df_original = load_data()

if 'predictions_df' not in st.session_state:
        st.session_state.predictions_df = pd.DataFrame(columns=['Car', 'Body', 'Mileage', 'EngV', 'EngType', 'Registered', 'Year', 'Drive', 'Predicted Price'])




st.title("🔮 Car Price Predictor 🔮")
st.write("""### Enter your car information to predict its price!""")

# User Input
all_cars = df_original['car'].unique()
car = st.selectbox('Car brand', all_cars)

body_types = ("crossover", "sedan", "van", "vagon", "hatch", "other")
body = st.selectbox("Body", body_types)

mileage = st.slider("Mileage", 0, 600, 80)
engV = st.slider("EngV", 0.0, 7.0, 3.5)

engType_types = ("Gas", "Petrol", "Diesel", "Other")
engType = st.selectbox("EngType", engType_types)

registered = st.radio("Is it registered?", ('Yes', 'No'))

year = st.slider("Year", 1975, 2015, 2010)

drive_types = ("full", "rear", "front")
drive = st.selectbox("Drive", drive_types)

ok = st.button("Calculate Price")

if ok:
    if ((st.session_state.predictions_df['Car'] == car) &
        (st.session_state.predictions_df['Body'] == body) &
        (st.session_state.predictions_df['Mileage'] == mileage) &
        (st.session_state.predictions_df['EngV'] == engV) &
        (st.session_state.predictions_df['EngType'] == engType) &
        (st.session_state.predictions_df['Registered'] == registered) &
        (st.session_state.predictions_df['Year'] == year) &
        (st.session_state.predictions_df['Drive'] == drive)).any():
        st.warning("Prediction with these parameters already exists.")
    else:
        X_sample = np.array([[car, body, mileage, engV, engType, registered, year, drive]])
        # Apply the encoder and data type corrections:
        X_sample[:, 0] = str(X_sample[:, 0][0] if X_sample[:, 0][0] in list(df_original['car'].unique()) else 'Other')
        X_sample[:, 0] = le_car.transform(X_sample[:, 0])
        X_sample[:, 1] = le_body.transform(X_sample[:, 1])
        X_sample[:, 4] = le_engType.transform(X_sample[:, 4])
        X_sample[:, 5] = int(1 if X_sample[:, 5][0].lower() == 'yes' else 0)
        X_sample[:, 7] = le_drive.transform(X_sample[:, 7])

        X_sample = np.array([[int(X_sample[0, 0]), int(X_sample[0, 1]), int(X_sample[0, 2]),
                            float(X_sample[0, 3]), int(X_sample[0, 4]), int(X_sample[0, 5]),
                            int(X_sample[0, 6]), int(X_sample[0, 7])]])

        # Predict the price
        price = model.predict(X_sample)[0]

        # Create or load dataframe to store predictions

        # Update predictions dataframe
        new_row = pd.DataFrame({
            'Car': [car],
            'Body': [body],
            'Mileage': [mileage],
            'EngV': [engV],
            'EngType': [engType],
            'Registered': [registered],
            'Year': [year],
            'Drive': [drive],
            'Predicted Price': [price]
        })

        st.session_state.predictions_df = pd.concat([st.session_state.predictions_df, new_row], ignore_index=True)

        st.success(f"The estimated price is ${price:.2f}")

# Display the predicted prices dataframe
st.subheader("Predicted Prices:")
st.write(st.session_state.predictions_df)
