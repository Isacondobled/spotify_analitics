import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
import streamlit as st

df_raw = pd.read_csv('spotify_churn_dataset.csv')

df_baking = df_raw.copy()
df_baking = df_baking.drop(columns=['user_id','ads_listened_per_week'])
cat_cols = ['gender', 'country', 'subscription_type', 'device_type']
df_baking[cat_cols] = df_baking[cat_cols].astype('category')
df = df_baking.copy()

df_train, df_test = train_test_split(df, test_size=0.2, random_state=2025, stratify=df['is_churned'])
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

X_train = df_train.drop(columns='is_churned')
y_train = df_train['is_churned']

X_test = df_test.drop(columns='is_churned')
y_test = df_test['is_churned']

num_cols = X_train.select_dtypes('number').columns
cat_cols = X_train.select_dtypes('category').columns

cat_proc = Pipeline(steps=[
    ('cat_proc', OneHotEncoder())
])

num_proc = Pipeline(steps=[
    ('num_proc', StandardScaler())
])

processor = ColumnTransformer(transformers=[
    ('num', num_proc, num_cols),
    ('cat', cat_proc, cat_cols)
])

lr = Pipeline(steps=[
    ('proc', processor),
    ('lr', LogisticRegression())
])

dt = Pipeline(steps=[
    ('proc', processor),
    ('dt', DecisionTreeClassifier(random_state=2025))
])

rf = Pipeline([
    ('proc', processor),
    ('rf', RandomForestClassifier(random_state=2025))
])

bc = Pipeline([
    ('proc', processor),
    ('knn', BernoulliNB())
])

models = [
    (lr, 'Logistic Regression', 'lr'),
    (dt, 'Decission tree', 'dt'),
    (rf, 'Random forest', 'rf'),
    (bc, 'Bernoulli NB', 'bc'),
]

performance = {}

for est, name, sname in models:
  # Training
  est.fit(X_train, y_train)

  # Prediction
  y_hat = est.predict(X_test)

    

st.title('Predicción si un cliente abandonará su plan de spotify')

with st.form(key='my_form'):
  gender = st.selectbox('Gender', ['Female', 'Male', 'Other'])
  age = st.number_input('Age', min_value=0, max_value=100, value=25)
  country = st.selectbox('Country', df['country'].unique())
  subscription_type = st.selectbox('Subscription type', df['subscription_type'].unique())
  listening_time = st.number_input('Listening time', min_value=0, value=100)
  songs_played_per_day = st.number_input('Songs played per day', min_value=0, value=50)
  skip_rate = st.number_input('Skip rate', min_value=0.0, max_value=1.0, value=0.5)
  device_type = st.selectbox('Device type', df['device_type'].unique())
  offline_listening = st.number_input('Offline listening (0 for No, 1 for Yes)', min_value=0, max_value=1, value=0)
  
  submit_button = st.form_submit_button('Predecir')

if submit_button:
            # Crear un dataframe con los datos ingresados
            input_data = pd.DataFrame({
                "gender": [gender],
                "age": [age],
                "country": [country],
                "subscription_type": [subscription_type],
                "listening_time": [listening_time],
                "songs_played_per_day": [songs_played_per_day],
                "skip_rate": [skip_rate],
                "device_type": [device_type],
                "offline_listening": [offline_listening]
            })

            prediction = rf.predict(input_data)[0]
            
            if prediction == 1:
                st.write("The user is likely to churn.")
            else:
                st.write("The user is not likely to churn.")