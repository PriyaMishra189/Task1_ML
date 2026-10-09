# California Housing Predictor

## Setup
Open a terminal in this folder and install dependencies:

```bash
py -m pip install -r requirements.txt
```

## 1. Train and save the model
```bash
py train_model.py
```

This downloads/loads the scikit-learn California Housing dataset, trains Linear Regression, prints MAE/RMSE/R², and creates `california_housing_model.pkl`.

The first run may need an internet connection to download the dataset.

## 2. Start the prediction UI
```bash
py -m streamlit run app.py
```

The browser UI accepts the eight input features and predicts median district house value.

## Notes
- `MedHouseVal` is in units of $100,000; the UI converts the prediction to dollars.
- This is an educational demo, not an individual property appraisal.
- Only load pickle/joblib model files that you trust.
