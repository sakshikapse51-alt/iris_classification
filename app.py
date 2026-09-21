from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

# Load the Iris dataset
iris = load_iris()

# Input features:
# 1. Sepal Length
# 2. Sepal Width
# 3. Petal Length
# 4. Petal Width
X = iris.data

# Target labels
y = iris.target

# Create and train the model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save the trained model
joblib.dump(model, "iris_model.pkl")

print("Model trained successfully!")
print("iris_model.pkl created successfully!")
