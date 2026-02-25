import numpy as np
import polars as pl


def _create_stimuli_dataframe(img_with_paths: str, n_page = 23, ) -> pl.DataFrame:
    df = pl.DataFrame(pl.read_csv(img_with_paths))

    n_pages = 23

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

def _create_eyetracking_dataframe(data_regex = "data/**/raw_data/*.csv") -> pl.LazyDataFrame:
    eyetracking = (
        pl.scan_csv(data_regex, include_file_paths="file_path")
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
        pl.struct(["time", "pixel_x", "pixel_y", "pupil"]).alias("data")
    )
    eyetracking = (
        eyetracking
        .group_by([
            "page",
            "file_path",
            "participant_id",
            "stimulus_id",
            # "file_name",
            # "text",   # if you have it
        ])
        .agg(
            pl.col("data")
        )
    )
    return eyetracking

def create_mp_metadata(img_with_paths: str, out_dir: dir, data_regex = "data/**/raw_data/*.csv") -> None:
    stimuli = _create_stimuli_dataframe(img_with_paths)
    eyetracking = _create_eyetracking_dataframe(data_regex)
    # Join the two DataFrames on the stimulus_id and page columns
    combined = (
        eyetracking
        .join(stimuli.lazy(), on=["page", "stimulus_id"], how="left")
    )
    combined = combined.drop_nulls().select(["file_name", "file_path", "participant_id", "text", "data"])
    combined.sink_ndjson(out_dir / "metadata.jsonl")
