import preprocessing
import util as utils
import pandas as pd
import numpy as np



def test_nan_detector():
    # Arrange
    mock_data = {"hdi" : [0.4, -1, 0.1, 0.7, -1, 0.9]}
    mock_data = pd.DataFrame(mock_data)
    expected_data = {"hdi" : [0.4, np.nan, 0.1, 0.7, np.nan, 0.9]}
    expected_data = pd.DataFrame(expected_data)

    # Act
    processed_data = preprocessing.nan_detector(mock_data)

    # Assert
    assert processed_data.equals(expected_data)

def test_ohe_transform():
    # Arrange
    config = utils.load_config()
    ohe_object = utils.pickle_load(config["ohe_continent_path"])
    mock_data = {
        "continent": [
            "Asia", "Europe",
            "Africa", "South America",
            "Oceania", "North America"
        ]
    }
    mock_data = pd.DataFrame(mock_data)
    expected_data = {
        
            "Asia": [1, 0, 0, 0, 0, 0],
            "Europe": [0, 1, 0, 0, 0, 0],
            "Africa": [0, 0, 1, 0, 0, 0],
            "South America": [0, 0, 0, 1, 0, 0],
            "Oceania": [0, 0, 0, 0, 1, 0],
            "North America": [0, 0, 0, 0, 0, 1]
        
    }
    expected_data = pd.DataFrame(expected_data)
    expected_data = expected_data.astype(str).replace({'\[':'', '\]':''}, regex=True).astype(int)

    # Act
    processed_data = preprocessing.ohe_transform(mock_data, "continent", ohe_object)
    processed_data = processed_data.astype(float)

    # Assert
    assert processed_data.equals(expected_data)


