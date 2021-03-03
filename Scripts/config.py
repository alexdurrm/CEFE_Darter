'''
file used to centralize the names and other parameters
'''
### NAME OF THE CSV FILE WHICH STORES ALL THE METRICS
CSV_NAME="metrics_depth"


### DIRECTORY NAMES
DIR_STYLIZED_FISHES=["All_layers",
                    "Layers_1and2",
                    "Layers_3to5"]
DIR_ORIGINAL_FISHES=["FISH_images"]
DIR_ORIGINAL_HABITATS=["HABITAT_images"]
DIR_SAMUEL=["crops"]
DIR_IGNORED=[]


### COLUMN NAMES FOR THE CSV FILE
COL_INDEX="filepath"
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
#fourrier slope
COL_F_SLOPE="mean_slope_F"
COL_F_N_SAMPLE="n_sample_F"
COL_F_WIN_SIZE="window_size_F"
COL_F_RESIZE="resize_F"
COL_FFT_RANGE="fft_range"
#local binary Pattern
COL_PATH_LBP="path_LBP"
COL_RADIUS_LBP="radius_LBP"
COL_POINTS_LBP="points_LBP"
COL_RESIZE_LBP="resize_LBP"
#statistical values
COL_STAT_MEAN="mean_stat"
COL_STAT_STD="std_stat"
COL_STAT_SKEW="skewness_stat"
COL_STAT_KURT="kurtosis_stat"
COL_STAT_ENTROPY="entropy_stat"
#deep features
COL_MODEL_NAME="model_name_deep_features"
COL_PATH_DEEP_FEATURES="path_deep_features"
COL_SPARSENESS_DF="sparseness_deep_features"
#GLCM
COL_GLCM_MEAN="GLCM_mean"
COL_GLCM_VAR="GLCM_variance"
COL_GLCM_CORR="GLCM_correlation"
COL_GLCM_CONTRAST="GLCM_contrast"
COL_GLCM_DISSIMIL="GLCM_dissimilarity"
COL_GLCM_HOMO="GLCM_homogeneity"
COL_GLCM_ASM="GLCM_ASM"
COL_GLCM_ENERGY="GLCM_energy"
COL_GLCM_MAXP="GLCM_max_proba"
COL_GLCM_ENTROPY="GLCM_entropy"
COL_GLCM_ANGLES="GLCM_angles"
COL_GLCM_DIST="GLCM_dist"
#GINI
COL_GINI_VALUE="gini_coefficient"
#GABOR
COL_GABOR_ANGLES="gabor_angles"
COL_GABOR_FREQ="gabor_frequencies"
COL_GABOR_VALUES="gabor_values"
#COLOR
COL_COLOR_RATIO="color_ratio"

### keywords given by visual_metrics that are interpreted by run metrics
DATA="data"
FORMAT="format"
SAVING_DIR="saving_directory"
NAME_COL_PATH="name_path_column"


### matplotlib parameters
FLAT_UI = ["#8c8c8c", "#5f9e6e", "#cc8963", "#5975a4", "#857aab", "#b55d60", "#c1b37f", "#8d7866", "#d095bf", "#71aec0"]
