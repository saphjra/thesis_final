"""
gaze_preprocessing.py

Shared preprocessing pipeline for pymovements datasets / single Gaze objects.
Used by dataset_stats.py and evaluate_model.py to ensure consistent event
detection across analysis and evaluation.
"""

from __future__ import annotations


import polars as pl
import pymovements as pm


class GazePreprocessor:
    """
    Encapsulates the standard preprocessing pipeline:
        pix2deg → pos2vel → IDT → microsaccades → compute_event_properties

    Dataset-specific adaptations (currently mcfw-gaze) are applied automatically
    before the shared pipeline when ``apply_dataset`` is called.
    """

    def __init__(
        self,
        threshold_factor: float = 6.0,
        dispersion_threshold: float = 1.0,
        min_fix_duration: int = 100,
        min_sac_duration: int = 30,
        vel_method: str = "fivepoint",
        vel_threshold: float = 30.0,
    ):
        self.threshold_factor = threshold_factor
        self.dispersion_threshold = dispersion_threshold
        self.min_fix_duration = min_fix_duration
        self.min_sac_duration = min_sac_duration
        self.vel_method = vel_method
        self.vel_threshold = vel_threshold

    # ------------------------------------------------------------------
    # Dataset-level (all recordings at once)
    # ------------------------------------------------------------------

    def apply_dataset(self, dataset: pm.Dataset, dataset_name: str) -> None:
        min_fix = self.min_fix_duration
        min_sac = self.min_sac_duration

        if dataset_name == "mcfw-gaze":
            min_fix, min_sac = self._fix_mcfw(dataset, min_fix, min_sac)

        dataset.pix2deg()
        dataset.pos2vel(method=self.vel_method)
        dataset.detect_events(
            "idt",
            minimum_duration=min_fix,
            dispersion_threshold=self.dispersion_threshold,
        )

        # Per-recording try/except — a single zero-variance recording
        # (e.g. a frozen/corrupted segment) should not abort the whole dataset.
        failed = []
        for gaze in dataset.gaze:
            try:
                gaze.detect(
                    "microsaccades",
                    clear=False,
                    threshold_factor=self.threshold_factor,
                    minimum_duration=min_sac,
                )
            except ValueError as e:
                failed.append(
                    (
                        gaze.metadata.get("subject_id"),
                        gaze.metadata.get("stimulus"),
                        str(e),
                    )
                )

        if failed:
            print(
                f"[preprocess] microsaccade detection failed for {len(failed)} recordings "
                f"(likely zero-variance/corrupted segments): {failed[:3]}{'...' if len(failed) > 3 else ''}"
            )

        dataset.compute_event_properties(["amplitude", "dispersion", "peak_velocity"])

    # ------------------------------------------------------------------
    # Single Gaze object (fake sequences, per-recording re-detection)
    # ------------------------------------------------------------------

    def apply_gaze(self, gaze: pm.Gaze, clear: bool = True) -> None:
        """
        Run the shared pipeline on a single Gaze object in-place.

        No dataset-specific adaptations are applied here — call this for
        individually constructed Gaze objects (e.g. generated fake sequences).

        Microsaccade detection is attempted first; if it yields no saccades
        (common for synthetic sequences that lack realistic velocity profiles),
        inter-fixation gaps are labelled as saccades via pm.events.fill.
        """
        gaze.pix2deg()
        gaze.pos2vel(method=self.vel_method)
        gaze.detect(
            "idt",
            clear=clear,
            minimum_duration=self.min_fix_duration,
            dispersion_threshold=self.dispersion_threshold,
        )
        gaze.detect(
            "microsaccades",
            minimum_duration=self.min_sac_duration,
            threshold_factor=self.threshold_factor,
        )

        n_saccades = (
            gaze.events.frame.filter(pl.col("name") == "saccade").height
            if gaze.events is not None and gaze.events.frame is not None
            else 0
        )
        if n_saccades == 0:
            self._fill_saccades(gaze)

        gaze.compute_event_properties(["amplitude", "dispersion", "peak_velocity"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_saccades(gaze: pm.Gaze) -> None:
        """
        Fallback for when microsaccade detection finds no saccades (e.g. synthetic
        sequences with flat velocity profiles).  Uses pm.events.fill to label
        all timesteps not covered by a fixation as saccades, then merges them
        back into the gaze event frame.
        """
        timesteps = gaze.samples["time"]
        filled = pm.events.fill(
            gaze.events,
            timesteps=timesteps,
            minimum_duration=1,
            name="saccade",
        )
        if filled is not None and filled.frame is not None and len(filled.frame) > 0:
            gaze.events.frame = pl.concat(
                [gaze.events.frame, filled.frame], how="diagonal_relaxed"
            ).sort("onset")

    @staticmethod
    def _fix_mcfw(dataset: pm.Dataset, min_fix: int, min_sac: int) -> tuple[int, int]:
        """
        mcfw-gaze stores time in seconds with non-constant intervals, and
        pixel coordinates are normalised to [0, 1].  Fix both in-place.

        Returns adjusted (min_fix, min_sac) rounded to the nearest sample
        interval so IDT duration parameters are valid.
        """
        sr = dataset.gaze[0].experiment.sampling_rate
        interval_ms = int(round(1000 / sr))
        min_fix = max(interval_ms, (min_fix // interval_ms) * interval_ms)
        min_sac = max(interval_ms, (min_sac // interval_ms) * interval_ms)

        for gaze in dataset.gaze:
            screen = gaze.experiment.screen
            if "time" in gaze.samples.columns:
                t0 = int(round(float(gaze.samples["time"][0]) * 1000))
                n = len(gaze.samples)
                gaze.samples = gaze.samples.with_columns(
                    pl.Series(
                        "time",
                        [t0 + i * interval_ms for i in range(n)],
                        dtype=pl.Int64,
                    )
                )
            gaze.samples = gaze.samples.with_columns(
                pl.concat_list(
                    [
                        pl.col("pixel").list.get(0) * screen.width_px,
                        pl.col("pixel").list.get(1) * screen.height_px,
                    ]
                ).alias("pixel"),
            )

        return min_fix, min_sac
