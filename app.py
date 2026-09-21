import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)

# ============================================================
# LOAD IRIS DATASET
# ============================================================

iris = load_iris()

X = iris.data
y = iris.target

# ============================================================
# TRAIN MACHINE LEARNING MODEL
# ============================================================

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X, y)

# ============================================================
# IRIS SPECIES
# ============================================================

species_names = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}

# ============================================================
# TITLE
# ============================================================

st.title("🌸 Iris Flower Prediction App")

st.write(
    "Enter the measurements of an Iris flower "
    "and the machine learning model will predict its species."
)

st.divider()

# ============================================================
# INPUT SECTION
# ============================================================

st.header("🌿 Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:

    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=4.0,
        step=0.1
    )

with col2:

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

st.divider()

# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🌸 Predict Iris Species",
    use_container_width=True
):

    # Create input with exactly 4 features
    input_data = np.array([
        [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    ])

    # Make prediction
    prediction = model.predict(input_data)

    # Get predicted class
    predicted_class = int(prediction[0])

    # Get flower name
    predicted_species = species_names[predicted_class]

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.success(
        f"🌸 Predicted Iris Species: **{predicted_species}**"
    )

    st.divider()

    st.header("📊 Entered Measurements")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Sepal Length",
            f"{sepal_length:.1f} cm"
        )

        st.metric(
            "Sepal Width",
            f"{sepal_width:.1f} cm"
        )

    with col2:

        st.metric(
            "Petal Length",
            f"{petal_length:.1f} cm"
        )

        st.metric(
            "Petal Width",
            f"{petal_width:.1f} cm"
        )

    # ========================================================
    # PREDICTION PROBABILITY
    # ========================================================

    probabilities = model.predict_proba(input_data)[0]

    st.divider()

    st.header("📈 Prediction Probability")

    st.write(
        f"**Iris Setosa:** {probabilities[0] * 100:.2f}%"
    )

    st.progress(float(probabilities[0]))

    st.write(
        f"**Iris Versicolor:** {probabilities[1] * 100:.2f}%"
    )

    st.progress(float(probabilities[1]))

    st.write(
        f"**Iris Virginica:** {probabilities[2] * 100:.2f}%"
    )

    st.progress(float(probabilities[2]))

# ============================================================
# INFORMATION SECTION
# ============================================================

st.divider()

st.subheader("ℹ️ About this Application")

st.write(
    "This application uses the Iris dataset from scikit-learn "
    "and a K-Nearest Neighbors (KNN) machine learning algorithm "
    "to classify the flower into Setosa, Versicolor, or Virginica."
)
