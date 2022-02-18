import numpy as np
import pandas as pd
import cv2

import os.path

DATASET_DIR = 'training'
DATA_KEY_PATH = 'training/data_key.csv'
LABELS_PATH = 'mouth_shape_categories.txt'

DATASET_IGNORE = ['training/ignore', DATA_KEY_PATH, LABELS_PATH]

WINDOW_DIMS = (1280, 720)
MENU_WIDTH = 300

class FinishEarlyException(Exception): # [10]
    pass

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

    # Allow user to go through and label each image
    df = fill_data(df, labels, ds_files)
    
    # Save progress to DATA_KEY_PATH
    df.to_csv(DATA_KEY_PATH) # [11]
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
#   Updated DataFrame
#
def fill_data(df: pd.DataFrame, labels: list[str], files: list[str]) -> pd.DataFrame:
    # Create columns if they haven't been already
    for label in ["filename"] + labels:
        if label not in df:
            df[label] = np.nan # [5]

    # Get information for each unvisited file
    for file in files:
        if any(df["filename"] == file):
            continue

        try:
            processed = process_file(file, labels)
            processed["filename"] = file
            df_processed = pd.DataFrame([processed])
            df_processed.set_index(pd.Index([file]))
            df = pd.concat([df, df_processed]) # [9]
        except FinishEarlyException:
            break;
    
    return df


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

    # Mouse Input Variables
    mouse_x, mouse_y = -1, -1
    click_pos_x, click_pos_y = -1, -1
    l_click = False
    r_click = False

    # Handle Mouse Input
    def mouse_callback(event, x, y, flag, param):

        # We need access to variables
        nonlocal mouse_x, mouse_y
        nonlocal l_click, r_click
        nonlocal click_pos_x, click_pos_y

        # Left Mouse Down
        if event == cv2.EVENT_LBUTTONDOWN:
            l_click = True
            click_pos_x, click_pos_y = x, y

        # Left Mouse Up
        if event == cv2.EVENT_LBUTTONUP:
            l_click = False
            click_pos_x, click_pos_y = -1, -1

        # Right Mouse Down
        if event == cv2.EVENT_RBUTTONDOWN:
            r_click = True
            click_pos_x, click_pos_y = x, y

        # Right Mouse Up
        if event == cv2.EVENT_RBUTTONUP:
            r_click = False
            click_pos_x, click_pos_y = -1, -1

        # Get current mouse pos regardless of clicks
        mouse_x, mouse_y = x, y

    # Return Value
    values = {k:0.0 for k in labels}

    # I don't like that this is hard coded, but we're running behind on time
    if "Bottom Lip Horizontal Offset" in values:
        values["Bottom Lip Horizontal Offset"] = 0.5

    # Other Variables
    window_name = 'img'
    frame = np.zeros((*(WINDOW_DIMS[::-1]), 3), dtype=np.uint8)
    clicking_id = None
    start_val = -1

    img_shape = (int(WINDOW_DIMS[0] * 0.75), WINDOW_DIMS[1])
    menu_shape = (WINDOW_DIMS[0] - img_shape[0], WINDOW_DIMS[1])

    # Create the initial frame to display to user
    # (It has an image and a menu)
    frame[:img_shape[1],:img_shape[0]] = fit_image(file, img_shape)
    frame[:menu_shape[1],img_shape[0]:], boxes = create_menu(values, menu_shape, (img_shape[0], 0))

    # Define window behavior (Make it handle mouse input correctly)
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL|cv2.WINDOW_AUTOSIZE) # [7]
    cv2.setMouseCallback(window_name, mouse_callback)

    # Display Loop
    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1: # [6]

        # Show Current Frame
        cv2.imshow(window_name, frame)

        # Exit if needed
        key_code = cv2.waitKey(16)
        if (key_code & 0xFF) == ord('q') or (key_code & 0xFF) == 27:
            cv2.destroyAllWindows()
            raise FinishEarlyException

        # Return information if needed
        if (key_code & 0xFF) == 13: # Enter
            return values

        # Handle Input
        if (r_click or l_click) and (clicking_id is None):
            for k in values:
                if has_collided((click_pos_x - img_shape[0], click_pos_y), boxes[k]):
                    clicking_id = k
                    start_val = values[k]
        if not (r_click or l_click):
            clicking_id = None
            start_val = -1

        if l_click and r_click:
            if (clicking_id is not None):
                values[clicking_id] = start_val
            clicking_id = None
            start_val = -1
            l_click = False
            r_click = False
            click_pos_x, click_pos_x = -1, -1

        elif (clicking_id is not None) and l_click:
            values[clicking_id] = 1 if values[clicking_id] == 0 else 0
            l_click = False

        elif (clicking_id is not None) and r_click:
            values[clicking_id] = get_normalized_x(mouse_x - img_shape[0], boxes[k])

        # Update Frame
        frame[:menu_shape[1],img_shape[0]:] = update_menu(values, menu_shape, boxes)

    raise FinishEarlyException


def has_collided(point, box):
    return (point[0] >= box[0]) and (point[0] <= box[2]) and (point[1] >= box[1]) and (point[1] <= box[3])


def get_normalized_x(point_x, box):
    x = (point_x - box[0]) / (box[2] - box[0])
    if x > 1:
        x = 1
    if x < 0:
        x = 0
    return x

# FIT IMAGE
#
# Summary:
#   Resizes image to fit within a bounding box, centering and maintaining
#   image aspect ratio.
#
# Params:
#   file - Location of image file
#   shape - A tuple containing the width and height (in that order) of
#           the bounding box.
#
# Returns:
#   The resized image padded with black to fill the bounding box
#
def fit_image(img_file: str, shape: tuple[int, int]) -> np.ndarray:

    # Read in the image file
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)

    # Get image width and height
    img_h, img_w, _ = img.shape

    # Get bounding box width and height
    bound_w, bound_h = shape

    # Compute aspect ratios
    img_aspect = img_w / img_h
    bound_aspect = bound_w / bound_h

    # Scale the output to maintain aspect ratios
    if img_aspect < bound_aspect:
        resize_shape = (img_w * bound_h // img_h, bound_h)
    else:
        resize_shape = (bound_w, img_h * bound_w // img_w)

    # Center image by calculating top left corner position
    x = (shape[0] - resize_shape[0]) // 2
    y = (shape[1] - resize_shape[1]) // 2

    # Create a black image of the correct size
    new_img = np.zeros((*(shape[::-1]), 3), dtype=np.uint8)

    # Scale and translate the original image and place it over the black image
    new_img[y:y+resize_shape[1], x:x+resize_shape[0]] = cv2.resize(img, resize_shape)

    return new_img


# CREATE MENU
#
# Summary:
#   Generates the initial frame for the menu GUI
#
# Params:
#   values - A dictionary whose keys are mouth shapes and whose values
#           are based on user input
#   shape - A tuple containing the width and height (in that order) of
#           the bounding box of the GUI
#   pos - A tuple containing the x and y coordinates (in that order) of
#           the top left corner of the GUI bounding box
#
# Returns:
#   - The current frame for the menu GUI
#   - Collision boxes for each menu option
#
def create_menu(values: dict[str, float],
                shape: tuple[int, int],
                pos: tuple[int, int]) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    
    # Generate collision boxes in the following format:
    # (Top Left X, Top Left Y, Bottom Right X, Bottom Right Y)
    boxes = {}
    box_height = shape[1] // len(values)
    for i,k in enumerate(values):
        boxes[k] = (0, i * box_height, shape[0], (i + 1) * box_height - 1)

    disp = update_menu(values, shape, boxes)

    return (disp, boxes)


# UPDATE MENU
#
# Summary:
#   Generates the current frame for the menu GUI
#
# Params:
#   values - A dictionary whose keys are mouth shapes and whose values
#           are based on user input
#   shape - A tuple containing the width and height (in that order) of
#           the bounding box of the GUI
#   boxes - Collision boxes for each menu option
#
# Returns:
#   The current frame for the menu GUI
#
def update_menu(values: dict[str, float],
                shape: tuple[int, int],
                boxes: dict[str, tuple[int, int, int, int]]) -> np.ndarray:
    
    disp = np.zeros((*(shape[::-1]), 3), dtype=np.uint8)

    for k,v in boxes.items():
        bar_end = int((v[2] - v[0]) * values[k] + v[0])
        cv2.rectangle(disp, v[:2], (bar_end, v[3]), (255, 128, 0), -1)
        org = (v[0] + 10, ((v[1] + v[3]) // 2) + 10)
        disp = cv2.putText(disp, f"{values[k]:0.2f} {k}", org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # [8]
        cv2.rectangle(disp, v[:2], v[2:], (255, 255, 255), 3)

    return disp


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
# [8] https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
# [9] https://stackoverflow.com/questions/51774826/append-dictionary-to-data-frame
# [10] https://www.programiz.com/python-programming/user-defined-exception
# [11] https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html