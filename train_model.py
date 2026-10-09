"""Train a California Housing Linear Regression model and save it locally."""
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_PATH = "california_housing_model.pkl"

def main():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    joblib.dump(model, MODEL_PATH)

    print("Model trained and saved to:", MODEL_PATH)
    print(f"MAE:  {mae:.4f} (in units of $100,000)")
    print(f"RMSE: {rmse:.4f} (in units of $100,000)")
    print(f"R²:   {r2:.4f}")
    print("Run the UI with: streamlit run app.py")

if __name__ == "__main__":
    main()
