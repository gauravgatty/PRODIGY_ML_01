from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load('model.pkl')


# Home Page
@app.route('/')
def home():

    return render_template(
        'index.html',
        prediction_text='',
        square_feet='',
        bedrooms='',
        bathrooms=''
    )


# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():

    # Get form values
    square_feet = request.form['square_feet']
    bedrooms = request.form['bedrooms']
    bathrooms = request.form['bathrooms']

    # Convert values for model
    features = np.array([[
        float(square_feet),
        int(bedrooms),
        int(bathrooms)
    ]])

    # Predict price
    prediction = model.predict(features)

    predicted_price = round(prediction[0], 2)

    # Return result with old values retained
    return render_template(

        'index.html',

        prediction_text=f'Predicted House Price: ₹ {predicted_price:,.2f}',

        square_feet=square_feet,
        bedrooms=bedrooms,
        bathrooms=bathrooms
    )


# Run App
if __name__ == "__main__":

    app.run(debug=True)