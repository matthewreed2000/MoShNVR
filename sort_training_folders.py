import numpy as np
import pandas as pd
import cv2

import sys
import os.path

DATASET_DIR = 'training'
DATA_KEY_PATH = 'training/data_key.csv'
LABELS_PATH = 'mouth_shape_categories.txt'

def main() -> None:
    if len(sys.argv) < 3:
        print("Not enough args")
        return

    # We don't filter the input, because this is a one-off script
    path = sys.argv[1]
    expected = (''.join(sys.argv[2:]))[1:-1].split(',')
    expected = [*map(float, expected)]
    print(expected)

    # Open Existing or Create New Dataframe
    if os.path.exists(DATA_KEY_PATH): # [1]
        df = pd.read_csv(DATA_KEY_PATH, index_col=0) # [2]
    else:
        df = pd.DataFrame()

    # Get Column Names
    labels = []
    with open(LABELS_PATH) as f:
        labels = [x.strip() for x in f]

    # Get a List of Dataset Files
    ds_files = get_files(path)

    # Fill df with input values in the order they appear in the labels
    df = fill_data(df, labels, expected, ds_files)

    # Save progress to DATA_KEY_PATH
    df.to_csv(DATA_KEY_PATH) # [11]
    print(df)

def get_files(path) -> list[str]:
    # Walk through file path and store files in a list
    ds_files = []
    for root, dirs, files in os.walk(DATASET_DIR + '/' + path, topdown=True):
        # Add files to list
        ds_files += [os.path.join(root, f) for f in files if os.path.join(root, f)]

    # Return the files list
    return ds_files

def fill_data(df: pd.DataFrame, labels: list[str], expected: list[float], files: list[str]) -> pd.DataFrame:
    # Create columns if they haven't been already
    for label in ["filename"] + labels:
        if label not in df:
            df[label] = np.nan # [5]

    expected_dict = {k:v for k,v in zip(labels, expected)}

    # Get information for each unvisited file
    for file in files:
        if any(df["filename"] == file):
            continue

        processed = expected_dict
        processed["filename"] = file
        df_processed = pd.DataFrame([processed])
        df_processed.set_index(pd.Index([file]))
        df = pd.concat([df, df_processed]) # [9]
    
    return df

if __name__ == '__main__':
    main()