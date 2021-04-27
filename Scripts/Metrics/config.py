from enum import Enum
'''
file used to centralize the names and other parameters used in CSV Results
'''
### NAME OF THE CSV FILE WHICH STORES ALL THE METRICS
DIR_RESULTS="Results_AE"
CSV_IMAGE="image_list.csv"
CSV_EXPERIMENTS="experiments.csv"

### DIRECTORY NAMES
DIR_STYLIZED_FISHES=["All_layers",
                    "Layers_1and2",
                    "Layers_3to5"]
DIR_ORIGINAL_FISHES=["FISH_images"]
DIR_ORIGINAL_HABITATS=["HABITAT_images"]
DIR_SAMUEL=["crops", "Crops"]
DIR_IGNORED=[]


### COLUMN NAMES FOR THE IMAGE CSV FILE
COL_IMG_PATH="Image_path"
COL_FILENAME="filename"
COL_TYPE="type"
COL_DIRECTORY="folder"
COL_HABITAT="habitat"
COL_COLOR_CONTROL="color_control"
COL_TV_LOSS="tv_loss"
COL_LAYERS="layers"
COL_FISH_SEX="sex"
COL_FISH_NUMBER="fish_n"
COL_SPECIES="species"
COL_IMG_WIDTH="img_width"
COL_IMG_HEIGHT="img_height"
COL_IMG_EXT="image_extension"

LIST_COLUMNS_IMG=[COL_IMG_PATH, COL_FILENAME, COL_TYPE, COL_DIRECTORY, COL_HABITAT,
    COL_COLOR_CONTROL, COL_TV_LOSS, COL_LAYERS, COL_FISH_SEX, COL_FISH_NUMBER,
    COL_SPECIES,COL_IMG_WIDTH,COL_IMG_HEIGHT,COL_IMG_EXT]

class FILE_TYPE(Enum):
    ORIG_FISH="original fish"
    STYLIZED_FISH="fish stylized"
    HABITAT="habitat"
    ELSE="else"
    def __str__(self):
		return self.name

### COLUMN NAMES FOR CSV EXPERIMENTS
COL_CONTENT_EXP_PATH="exp_fish_path"
COL_STYLE_EXP_PATH="exp_habitat_path"
COL_EXP_ID="exp_id"

LIST_COLUMNS_EXP=[COL_CONTENT_EXP_PATH, COL_STYLE_EXP_PATH, COL_EXP_ID]

### matplotlib parameters
FLAT_UI = ["#8c8c8c", "#5f9e6e", "#cc8963", "#5975a4", "#857aab", "#b55d60", "#c1b37f", "#8d7866", "#d095bf", "#71aec0"]
