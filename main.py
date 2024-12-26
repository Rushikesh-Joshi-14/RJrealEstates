import pandas as pd
from flask import Flask,render_template, request
import pickle
import numpy as np
from babel.numbers import format_currency

app = Flask(__name__)
data= pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", "rb"))

@app.route('/')
def index():
    locations = sorted(data["site_location"].unique())
    return render_template('index.html' , locations= locations)

@app.route('/predict', methods=['POST'])
def predict():
    locations= request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath =float(request.form.get('bath'))
    sqft =request.form.get('sqft')

    print(locations , bhk, bath, sqft)
    input = pd.DataFrame([[locations,sqft, bath, bhk]], columns=['site_location','total_sqft','bath', 'bhk'])
    prediction = pipe.predict(input)[0] *1e5 +500000

# Format as Indian Rupees
    formatted_currency = format_currency(prediction, 'INR', locale='en_IN')
    return str(formatted_currency)


if __name__ == "__main__":
    app.run(debug=True, port=5001)