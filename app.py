from flask import Flask, render_template, request
from app import app
import pickle
import pandas as pd
import io

app = Flask(__name__)

# Load the machine learning model
with open('carEmission.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def get_label_encoder(fuel_type):
    if fuel_type == 'Regular Gasoline': 
        return 2
    elif fuel_type == 'Premium Gasoline':
        return 3
    elif fuel_type == 'Petrol':
        return 0
    elif fuel_type == 'Diesel':
        return 0
    else:
        return 1

def get_proper_table(data):
    label_mapping = {0: 'Diesel/Petrol', 1: 'E85(Ethanol)', 2: 'Regular Gasoline', 3: 'Premium Gasoline'}
    data['FUELTYPE'] = data['FUELTYPE'].map(label_mapping)

    bins = [0, 100, 120, 150, 180, 200, 230, 260, 300, float('inf')]
    labels = ['Excellent', 'Very Good', 'Good', 'Above Average', 'Average', 
              'Below Average', 'Poor', 'Very Poor', 'Extremely Poor']

    data['ENGINESIZE'] = data['ENGINESIZE']*1000
    data['FUEL'] = round(100.0/data['FUEL'],2)

    data['Verdict'] = pd.cut(data['CO2 Emissions (g/km)'], bins=bins, labels=labels)

    return data

@app.route('/')
def index():
    return render_template('index.html', co2_emissions=None, emissions_table=None)

@app.route('/calculate_co2', methods=['POST'])
def calculate_co2():
    fuel_type = request.form['fuel-type']
    engine_size = float(request.form['engine-size']) / 1000
    cylinders = float(request.form['cylinders'])
    fuel_consumption = 100.0/ float(request.form['fuel-consumption'])

    fuel_type_label = get_label_encoder(fuel_type)

    # Use the model to predict CO2 emissions
    co2_emissions = model.predict([[fuel_type_label, engine_size, cylinders, fuel_consumption]])[0]
    co2_emissions = round(co2_emissions, 3)

    return render_template('index.html', co2_emissions=co2_emissions)

@app.route('/calculate_co2_file', methods=['POST'])
def calculate_co2_file():
    file = request.files['file']

    if not file:
        return render_template('index.html', emissions_table=None)

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file.read()))
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(io.BytesIO(file.read()))
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        return render_template('index.html', error_message=str(e))

    prediction_columns = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]

    if len(prediction_columns) < 4:
        return render_template('index.html', error_message="Insufficient columns for prediction")

    # Make predictions
    emissions = []
    for _, row in df.iterrows():
        prediction_values = [row[col] for col in prediction_columns]
        co2_emissions = model.predict([prediction_values])[0]
        emissions.append(round(co2_emissions, 3))

    # Add CO2 emissions to DataFrame
    df['CO2 Emissions (g/km)'] = emissions

    # Generate HTML table
    df = get_proper_table(df)

    emissions_table = df.to_html(index=False)

    return render_template('index.html', emissions_table=emissions_table)

if __name__ == '__main__':
    app.run()
