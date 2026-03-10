## which et data to include in the final dataset when loading Multipleyedata
from pathlib import Path

ET_DATA_TO_INCLUDE = [
    "pixel_x",
    "pixel_y",
]  # specify which eyetracking data to include in the final dataset, for all data use :["time", "pixel_x", "pixel_y", "pupil"]
META_DATA_TO_INCLUDE = [
    "file_name",
    "participant_id",
    "text",
    "data",
]  # specifies which metadata is included in the dataset , recommended is ["file_name", "file_path", "participant_id", "text", "data"]
REGEX = "data/**/raw_data/*.csv"  # regex pattern to match eyetracking data files


ROOT_DIR = Path(__file__).absolute().parent.parent  # kaamba dataset folder
STIMULUS_FOLDER = ROOT_DIR / "kaamba_dataset/stimuli"

SCREEN_RESOLUTION = {
    "Goettingen": (1920, 1080),
    "Gazebase": (1680, 1050),
}  # monitor resolution in pixels,
