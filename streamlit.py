import streamlit as st
import requests
from PIL import Image

# Load and set images in the first place
header_images = Image.open('assets/header_images.jpg')
st.image(header_images)

# Add some information about the service
st.title("Ecological Footprint Per Capita Prediction")
st.subheader("Just enter variabel below then click Predict button :sunglasses:")

# Create inbpx form of input
with st.form(key = "continent_form"):
    # Create select box input
    continent= st.selectbox(
        label = "1.\tFrom which continent is this data collected?",
        options = (
            "Asia",
            "Europe",
            "Africa",
            "South America",
            "North America",
            "Oceania"
        )
    )

    # Create form for number input
    hdi = st.number_input(
        label = "2.\tEnter HDI Value:",
        min_value = 0.0,
        max_value = 1.0,
        help = "Value range from 0 to 1"
    )
    
    continent = st.selectbox(
         label="Enter the continent of the country:",
         options=["Asia", "Europe", "Africa", "South America", "Oceania", "North America"],
         help="Select the continent of the country"
    )




    # Create button to submit the form
    submitted = st.form_submit_button("Predict")

    # Condition when form submitted
    if submitted:
        # Create dict of all data in the form
        raw_data = {
            "continent": continent,
            "hdi": hdi

        }

        # Create loading animation while predicting
        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://api:8080/predict", json = raw_data).json()

        # Parse the prediction result
        if res["error_msg"] != "":
            st.error("Error Occurs While Predicting: {}".format(res["error_msg"]))
        else:

            st.write("Predicted Ecological Footprint Per Capita: : {}".format(res["res"]))