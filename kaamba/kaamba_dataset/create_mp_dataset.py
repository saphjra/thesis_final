import numpy as np
import polars as pl
from constants import ET_DATA_TO_INCLUDE, META_DATA_TO_INCLUDE

def _create_stimuli_dataframe(img_with_paths: str, n_pages = 23, ) -> pl.DataFrame:
    df = pl.DataFrame(pl.read_csv(img_with_paths))

    text_cols = [f"page_{i + 1}" for i in range(n_pages)]
    img_cols = [f"page_{i + 1}_img_file" for i in range(n_pages)]
    # stimulus_cols = np.array([[1]*n_pages, [2]*n_pages, [3]*n_pages, [4]*n_pages, [6]*n_pages, [7]*n_pages, [8]*n_pages, [9]*n_pages, [10]*n_pages, [11]*n_pages, [12]*n_pages, [13]*n_pages]).flatten()
    stimulus_cols = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13] * n_pages).flatten()
    # reshape pages
    pages_long = (
        df.select(text_cols)
        .unpivot()
        .rename({"value": "text"})
    )

    # reshape image names
    imgs_long = (
        df.select(img_cols)
        .unpivot()
        .rename({"value": "img_name"})
    )

    # combine
    stimuli = pl.DataFrame({
        "file_name": imgs_long["img_name"],
        "text": pages_long["text"],
        "page": np.array([[f"page_{i + 1}"] * df.height for i in range(n_pages)]).flatten(),
        "stimulus_id": stimulus_cols,
    })
    stimuli = stimuli.drop_nulls().cast({"stimulus_id": pl.String})
    return stimuli

def _create_eyetracking_dataframe(data_folder_regex, et_data_to_include=ET_DATA_TO_INCLUDE) -> pl.lazyframe:
    """Reads all CSV files matching the given regex pattern, extracts participant_id and stimulus_id from the file paths, and combines them into a single DataFrame.
    with the et_data_to_include columns as a struct column named "data". and one can decide if pupil and for example timestamps should be included or not.
    The resulting DataFrame will have columns: page, file_path, participant_id, stimulus_id, and data (which is a struct containing the specified et_data_to_include columns)."""

    eyetracking = (
        pl.scan_csv(data_folder_regex, include_file_paths="file_path")
        .with_columns(
            pl.col("file_path")
            .str.extract(r"([^/]+)\.csv$", 1)  # get filename
            .str.extract(r"^([^_]+)", 1)  # get participant id
            .alias("participant_id"),
            pl.col("file_path").str.extract(r"([^/]+)\.csv$", 1).str.extract(r"(\d+)_raw_data$", 1).alias("stimulus_id")
            # extract stimulus id from filename
        )
    )
    eyetracking = eyetracking.with_columns(
        pl.struct(et_data_to_include).alias("data")
    )
    eyetracking = (
        eyetracking
        .group_by([
            "page",
            "file_path",
            "participant_id",
            "stimulus_id"
        ])
        .agg(
            pl.col("data")
        )
    )
    return eyetracking

def create_mp_metadata(img_with_paths: str, out_dir: dir, data_regex = "data/**/raw_data/*.csv", data_to_include = META_DATA_TO_INCLUDE) -> None:
    stimuli = _create_stimuli_dataframe(img_with_paths)
    eyetracking = _create_eyetracking_dataframe(data_regex)
    # Join the two DataFrames on the stimulus_id and page columns
    combined = (
        eyetracking
        .join(stimuli.lazy(), on=["page", "stimulus_id"], how="left")
    )
    combined = combined.drop_nulls().select(data_to_include)
    combined.sink_ndjson(out_dir / "metadata.jsonl")

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from constants import ROOT_DIR


    import os
    POLARS_VERBOSE = 1
    parser = argparse.ArgumentParser(description="Create metadata for MultiplEYE dataset")
    parser.add_argument("--img_with_paths", type=str, default="C:/Users/saphi/PycharmProjects/thesis/kaamba/data/MultiplEYE_DE_DE_Goettingen_1_2026/stimuli_MultiplEYE_DE_DE_Goettingen_1_2026/multipleye_stimuli_experiment_de_de_1_with_img_paths.csv", help="Path to the CSV file containing image paths and text")
    parser.add_argument("--out_dir", type=str, default="C:/Users/saphi/PycharmProjects/thesis/kaamba/data/MultiplEYE_DE_DE_Goettingen_1_2026/stimuli_MultiplEYE_DE_DE_Goettingen_1_2026/stimuli_images_de_de_1", help="Directory to save the output metadata")
    parser.add_argument("--data_regex", type=str, default="C:/Users/saphi/PycharmProjects/thesis/kaamba/data/MultiplEYE_DE_DE_Goettingen_1_2026/**/raw_data/*.csv", help="Regex pattern to match eyetracking data files")
    parser.add_argument("--data_to_include", nargs="+", default=META_DATA_TO_INCLUDE, help="List of metadata fields to include in the output")

    args = parser.parse_args()

    create_mp_metadata(args.img_with_paths, Path(args.out_dir), args.data_regex, args.data_to_include)