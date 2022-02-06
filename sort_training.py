import numpy as np
import pandas as pd
import cv2

import os.path

DATASET_DIR = 'training'
DATA_KEY_PATH = 'training/data_key.csv'
LABELS_PATH = 'mouth_shape_categories.txt'

DATASET_IGNORE = ['training/ignore', DATA_KEY_PATH, LABELS_PATH]

WINDOW_DIMS = (1280, 720)

# MAIN
#
# Summary:
#   Controls main program flow
#
def main() -> None:
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
    ds_files = get_files()
    print(ds_files[:10])

    # Allow user to go through and label each image
    fill_data(df, labels, ds_files)
    print(df)

# GET FILES
#
# Summary:
#   Gets all files in the DATASET_DIR and it's subfolders,
#   ignoring DATASET_IGNORE
#
# Returns:
#   List of files (relative paths stored as strings)
#
def get_files() -> list[str]:
    # Get the normalized relative paths for everything in DATASET_IGNORE
    ds_ignore_full = [os.path.normpath(os.path.join('.', x)) for x in DATASET_IGNORE] # [3]

    # Walk through file path and store files in a list
    ds_files = []
    for root, dirs, files in os.walk(DATASET_DIR, topdown=True):
        # Don't enter ignored sub-directories
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in ds_ignore_full] # [4]

        # Add files to list, ignoring DATASET_IGNORE files
        ds_files += [os.path.join(root, f) for f in files if os.path.join(root, f) not in ds_ignore_full]

    # Return the files list
    return ds_files


# FILL DATA
#
# Summary:
#   Prompts user to fill in data
#
# Params:
#   df - The dataframe that stores the information
#   labels - The label names of every column in the dataset
#   files - List of files that the user is filling out info for
#
# Returns:
#   None - The dataframe is modified in place to save memory
#
def fill_data(df: pd.DataFrame, labels: list[str], files: list[str]) -> None:
    # Create columns if they haven't been already
    for label in labels:
        if label not in df:
            df[label] = np.nan # [5]

    # Get information for each unvisited file
    for file in files[:1]:
        if file in df.index:
            continue

        print(process_file(file, labels))


# PROCESS FILE
#
# Summary:
#   Display image to the user and have the user select what
#   category/categories that the image belongs to
#
# Params:
#   file - Location of image file
#   labels - The categories that the user will be deciding between
#
# Returns:
#   The relative amount (percentage) that a label matches the image
#
def process_file(file: str, labels: list[str]) -> dict[str, float]:

    mouse_x, mouse_y = -1, -1
    click_pos_x, click_pos_y = -1, -1
    l_click = False
    r_click = False

    def mouse_callback(event, x, y, flag, param):
        nonlocal mouse_x, mouse_y
        nonlocal l_click, r_click
        nonlocal click_pos_x, click_pos_y

        if event == cv2.EVENT_LBUTTONDOWN:
            l_click = True
            click_pos_x, click_pos_y = x, y, 
        if event == cv2.EVENT_LBUTTONUP:
            l_click = False
            click_pos_x, click_pos_y = -1, -1,
        if event == cv2.EVENT_RBUTTONDOWN:
            r_click = True
            click_pos_x, click_pos_y = x, y, 
        if event == cv2.EVENT_RBUTTONUP:
            r_click = False
            click_pos_x, click_pos_y = -1, -1,

        if r_click and l_click:
            r_click = False
            l_click = False
            click_pos_x, click_pos_y = -1, -1,

        mouse_x, mouse_y = x, y


    window_name = 'img'
    frame = np.zeros((*(WINDOW_DIMS[::-1]), 3), dtype=np.uint8)

    img = cv2.imread(file, cv2.IMREAD_COLOR)

    img_width = img.shape[1]
    img_height = img.shape[0]

    frame[:img_height,:img_width] = img

    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL|cv2.WINDOW_AUTOSIZE) # [7]
    cv2.setMouseCallback(window_name, mouse_callback)

    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1: # [6]
        cv2.imshow(window_name, frame)
        key_code = cv2.waitKey(16)

        if r_click or l_click:
            print(mouse_x, mouse_y)

        if (key_code & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break;


# FILL COLUMN
#
# Summary:
#   Fills in a group of data column-by-column
#
# Params:
#   df - The dataframe that stores the information
#   label - The label of the current column being filled
#   files - List of files that the user is filling out info for
#
# Returns:
#   None - The dataframe is modified in place to save memory
#
def fill_column(df: pd.DataFrame, label: str, files: list[str]) -> None:
    if label not in df:
        df[label] = np.nan

    for file in files:
       df.at[file, label] = 1


#                             #
# Entry point for the program #
#                             #
if __name__ == "__main__":
    main()


# Sources:
# [1] https://www.pythontutorial.net/python-basics/python-check-if-file-exists/
# [2] https://pythonbasics.org/read-csv-with-pandas/
# [3] https://stackoverflow.com/questions/17295086/python-joining-current-directory-and-parent-directory-with-os-path-join
# [4] https://stackoverflow.com/questions/19859840/excluding-directories-in-os-walk
# [5] https://stackoverflow.com/questions/16327055/how-to-add-an-empty-column-to-a-dataframe
# [6] https://medium.com/@mh_yip/opencv-detect-whether-a-window-is-closed-or-close-by-press-x-button-ee51616f7088
# [7] https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/