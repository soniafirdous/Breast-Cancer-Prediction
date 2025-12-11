import streamlit as st
import requests
import pandas as pd




# ---------------- Sidebar ----------------
st.sidebar.title("Breast Cancer Info & App Usage")

# Add an image (replace URL with your own if desired)
st.sidebar.image(
    "download.jpeg",
    caption="Breast Cancer Cells",
    width=250
)


st.sidebar.subheader("What is Breast Cancer?")
st.sidebar.info("""
Breast cancer is a type of cancer that develops from breast cells.  
It is one of the most common cancers in women worldwide.  
Early detection is crucial for effective treatment and recovery.
""")

st.sidebar.subheader("How to Use This App")
st.sidebar.success("""
1. Enter the values for the 30 medical features in the input fields.  
2. Click the **Predict** button.  
3. The app will display:
   - Predicted class (**Benign** or **Malignant**)  
   - Probability of each class in a bar chart.
""")

st.sidebar.subheader("Disclaimer")
st.sidebar.warning("""
This tool is for educational purposes only and **cannot replace medical advice**.  
Always consult a healthcare professional for diagnosis and treatment.
""")



# ---------------- Main App ----------------
st.title("Breast Cancer Prediction")
st.write("Enter features to predict breast cancer:")

# Input fields for 30 features
feature_names = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"
]

# Collect user input
features = [st.number_input(name, value=0.0) for name in feature_names]

class_names = ["Benign", "Malignant"]

if st.button("Predict"):
    url = "http://127.0.0.1:8000/predict"
    data = {"features": features}
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()

        # Display predicted class
        predicted_class = class_names[result['class_label']]
        st.write(f"Predicted class: **{predicted_class}**")

        # Display probabilities nicely
        probabilities = result['probabilities'][0] if isinstance(result['probabilities'][0], list) else result['probabilities']
        prob_df = pd.DataFrame([probabilities], columns=class_names)
        st.write("Prediction Probabilities:")
        st.bar_chart(prob_df.T.rename(columns={0: "Probability"}))  # Transpose for better view

    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
