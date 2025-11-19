
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class RestaurantLocationRecommender:
    def __init__(self, data_path="data/restaurant_data.csv"):
        self.data = pd.read_csv(data_path)
        self.model = LinearRegression()

    def train(self):
        features = ["Population", "Competitors", "AvgRating", "Footfall"]
        target = "SuccessScore"
        X = self.data[features]
        y = self.data[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        pickle.dump(self.model, open("restaurant_model.pkl", "wb"))
        return self.model.score(X_test, y_test)

    def recommend(self):
        features = ["Population", "Competitors", "AvgRating", "Footfall"]
        self.data["PredictedScore"] = self.model.predict(self.data[features])
        return self.data.loc[self.data["PredictedScore"].idxmax()]

if __name__ == "__main__":
    rec = RestaurantLocationRecommender()
    acc = rec.train()
    print("Accuracy:", acc)
    print("Best area:
", rec.recommend())
