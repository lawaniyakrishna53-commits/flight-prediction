# Flight Cost Prediction

A machine-learning demo that estimates one-way flight fares in Indian rupees from journey details. The project includes a Flask prediction API, a browser-based form, a sample dataset, and a Jupyter notebook for exploring the model.

> **Disclaimer:** The included CSV is synthetic teaching data. Predictions are for learning and demonstration only; they are not live airline prices or booking quotes.

## Features

- Estimate fares based on airline, route, travel date, departure time, stops, cabin class, and booking lead time.
- Display the estimated route distance and flight duration.
- Train a Random Forest regression model from the included sample data if the saved model is unavailable.
- Explore the data and model workflow in `Flight_Cost_Prediction.ipynb`.

The demo supports 10 Indian cities, six airlines, Economy and Business classes, and up to two stops. Estimates are not suitable for real travel or pricing decisions.

## Project files

| File | Description |
| --- | --- |
| `app.py` | Flask API and model-loading/training logic |
| `index.html` | Browser-based flight fare estimator |
| `flight_price_sample.csv` | Synthetic sample fares used for training |
| `flight_price_model.joblib` | Saved Random Forest model used by the API |
| `Flight_Cost_Prediction.ipynb` | Notebook for exploring and training the model |
| `requirements.txt` | Python dependencies |

## Run locally

Requires Python 3.10 or later.

1. Open a terminal in the project folder.
2. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Start the API:

   ```powershell
   python app.py
   ```

   The API runs at `http://127.0.0.1:5000`. Keep this terminal open while using the website.

5. Open `index.html` in a browser and submit the journey form. The page checks the API connection automatically.

To stop the API, press **Ctrl+C** in the terminal.

## API

- `GET /health` — reports whether the API is running.
- `POST /predict` — returns the estimated fare, route distance, estimated duration, model name, and a disclaimer.

Example request body:

```json
{
  "airline": "IndiGo",
  "source": "Delhi",
  "destination": "Mumbai",
  "journey_date": "2027-01-15",
  "departure_time": "09:00",
  "total_stops": 0,
  "travel_class": "Economy",
  "booking_days_before": 21
}
```

The API validates that the source and destination are different supported cities, the airline and cabin class are supported, the stop count is 0–2, the booking lead time is 0–90 days, and the travel date is not in the past.

## Model and data notes

The API uses a scikit-learn pipeline with one-hot encoding for categorical fields and a Random Forest regressor. Route distance is calculated from the city coordinates in `app.py`, and estimated duration is derived from distance and the number of stops.

The sample data is synthetic and should not be interpreted as representative market pricing. Replace it with a suitable, legally obtained historical dataset and retrain the model before using this project for any real-world analysis.
