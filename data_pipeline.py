from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import os
import copy
import util as util



def read_raw_data(config: dict) -> pd.DataFrame:
    # Create variable to store raw dataset
    raw_dataset = pd.DataFrame()

    # Raw dataset dir
    raw_dataset_dir = config["raw_dataset_dir"]

    # Columns to keep
    columns_to_keep = config["columns_to_keep"]

    # Look and load add CSV files
    for file in tqdm(os.listdir(raw_dataset_dir)):
        # Only read CSV files
        if file.endswith('.csv'):  
            csv_data = pd.read_csv(os.path.join(raw_dataset_dir, file))
            raw_dataset = pd.concat([csv_data, raw_dataset])

    # Keep only specified columns
    raw_dataset = raw_dataset[columns_to_keep]

    # Return raw dataset
    return raw_dataset

def check_data(input_data, params, api=False):
    input_data = copy.deepcopy(input_data)
    params = copy.deepcopy(params)

    if not api:
        # Check data types
        assert input_data.select_dtypes("object").columns.to_list() == \
            params["object_columns"], "an error occurs in object column(s)."
        assert input_data.select_dtypes("float").columns.to_list() == \
            params["float64_columns"], "an error occurs in float64 column(s)."
        
    else:
        # In case checking data from api
        # Check data types
        # Predictor that has object dtype only continent
        object_columns = params["object_columns"]
        del object_columns[1:]
        
        float_columns = params["float64_columns"]
        del float_columns[-1]

        # Check data types
        assert input_data.select_dtypes("object").columns.to_list() == \
            object_columns, "an error occurs in object column(s)."
        assert input_data.select_dtypes("float").columns.to_list() == \
            float_columns, "an error occurs in float64 column(s)."
            
    assert set(input_data.continent).issubset(set(params["range_continent"])), \
        "an error occurs in continent range."
    assert input_data.hdi.between(params["range_hdi"][0], params["range_hdi"][1]).sum() == \
        len(input_data), "an error occurs in hdi range."


if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Read all raw dataset
    raw_dataset = read_raw_data(config_data)

    # 3. Reset index
    raw_dataset.reset_index(
        inplace = True,
        drop = True
    )

    # 4. Save raw dataset
    util.pickle_dump(
        raw_dataset,
        config_data["raw_dataset_path"]
    )

    # 5. Handling variable hdi
    raw_dataset.hdi.fillna(
        -1,
        inplace = True
    )
    raw_dataset.hdi = raw_dataset.hdi.astype(float)

    # 6. Handling variable continent
    raw_dataset.continent.fillna(
        -1,
        inplace = True
    )
    raw_dataset.continent = raw_dataset.continent.astype(object)


    # 7. Handling variable EFConsPerCap
    raw_dataset.EFConsPerCap.fillna(
        -1,
        inplace = True
    )
    raw_dataset.EFConsPerCap = raw_dataset.EFConsPerCap.astype(float)
    
    util.pickle_dump(
        raw_dataset,
        config_data["cleaned_raw_dataset_path"]
    )

    # 8. Check data definition
    check_data(raw_dataset, config_data)

    # 9. Splitting input output
    x = raw_dataset[config_data["predictors"]].copy()
    y = raw_dataset.EFConsPerCap.copy()

    # 10. Splitting train test
    x_train, x_test, \
    y_train, y_test = train_test_split(
        x, y,
        test_size = 0.2,
        random_state = 42,
        stratify = raw_dataset["continent"]
    
    )

    # 11. Splitting test valid
    x_valid, x_test, \
    y_valid, y_test = train_test_split(
        x_test, y_test,
        test_size = 0.2,
        random_state = 42,
        stratify = x_test["continent"]

    )

    # 12. Convert hdi and EFConsPerCap columns back to their original data type
    x_test["hdi"] = x_test["hdi"].astype(float)
    x_valid["hdi"] = x_valid["hdi"].astype(float)
    y_test = y_test.astype(float)
    y_valid = y_valid.astype(float)

    # 13. Save train, valid and test set
    util.pickle_dump(x_train, config_data["train_set_path"][0])
    util.pickle_dump(y_train, config_data["train_set_path"][1])

    util.pickle_dump(x_valid, config_data["valid_set_path"][0])
    util.pickle_dump(y_valid, config_data["valid_set_path"][1])

    util.pickle_dump(x_test, config_data["test_set_path"][0])
    util.pickle_dump(y_test, config_data["test_set_path"][1])