from fastapi import FastAPI
from numpy import float64
from pydantic import BaseModel
import uvicorn
import pandas as pd
import util as util
import data_pipeline as data_pipeline
import preprocessing as preprocessing

config_data = util.load_config()
ohe_continent = util.pickle_load(config_data["ohe_continent_path"])
model_data = util.pickle_load(config_data["production_model_path"])

print("config_data:", config_data)
class api_data(BaseModel):
    hdi : float
    continent : object
    
app = FastAPI()

# general endpoint (testing endpoint)
@app.get("/")
def home():
    return "Hello, FastAPI up!"

# health check
@app.get("/health/")
def home():
    return "200"

@app.post("/predict/")
def predict(data: api_data):    
    # Convert data api to dataframe

    data = pd.DataFrame([data])
    data = data.reset_index(drop = True)
    
    # Convert dtype
    
    data = pd.concat(
        [
            data[config_data["predictors"][0]].astype(float),
            data[config_data["predictors"][1]]
        ],
        axis = 1
    )


    #Check range data
    try:
        data_pipeline.check_data(data, config_data, True)
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}
    
    #preprocessing data in serving
    # Encoding continent
    data = preprocessing.ohe_transform(data, "continent", config_data["ohe_continent_path"])
    data.drop(columns=["continent"], inplace=True)
    
    #Standardization
    data = preprocessing.standardizerData(data)
    
    # Predict data
    y_pred = model_data["model_data"]["model_object"].predict(data)

    return {"res" : y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)
