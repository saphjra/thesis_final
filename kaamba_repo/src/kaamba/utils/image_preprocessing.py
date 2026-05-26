import torch
from pathlib import Path


from torchvision.transforms import v2
from torchvision.io import decode_image
from typing import Optional, List
import pymovements as pm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import polars as pl


class MyCustomTransform(v2.Pad):
    def __init__(self, *args, **kwargs):
        super().__init__(padding=0, *args, **kwargs)

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be padded.

        Returns:
            PIL Image or Tensor: Padded image.

        """
        # print(f"I'm transforming an image of shape {img.shape} ")
        pad_vals = [0, 0, img.shape[2] - img.shape[2], img.shape[2] - img.shape[1]]
        return v2.functional.pad(img, pad_vals, self.fill, self.padding_mode)


def _image_transform_coordiantes_preserving(
    self, image_path: Path, screen_width_px: int, screen_height_px: int
) -> torch.Tensor:
    """this function could be used in an architecture where the model needs to preserve
    the original coordinates of the gaze data, for example if the model uses a spatial attention mechanism or a
    SSM with a mechanism resembling cross attention implements that
    directly attends to pixel locations in the image. In this case, we need to ensure that the image is padded to
    the original screen resolution, so that the gaze coordinates still correspond to the correct locations
    in the image. The padding is done using edge values, which means that the original image content is preserved
    and not distorted by resizing. This way, the model can learn to attend to the correct regions of the image based
    on the gaze data, without any misalignment caused by resizing."""
    image = decode_image(str(image_path), mode="RGB")
    assert image.shape == (3, screen_height_px, screen_width_px)

    padding_val = [
        0,
        0,
        screen_width_px - image.shape[2],
        screen_height_px - image.shape[1],
    ]
    transform = v2.Compose(
        [
            v2.Pad(padding=padding_val, padding_mode="edge"),
            v2.Resize(size=None, max_size=self.max_image_size),
            MyCustomTransform(padding_mode="edge"),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(image)


def _image_transform(
    image_path: Path,
    max_image_size: int,
) -> torch.Tensor:
    """this function is used in the version where a global image embedding is extracted and fed into the model,
    for example in a ViT-based architecture. In this case, we can simply resize the image to the desired max size,
    without worrying about preserving the original coordinates of the gaze data.
    The resizing is done while maintaining the aspect ratio, so that the image content is not distorted.
    This way, the model can learn to extract relevant features from the image based on the gaze data,
    without any misalignment caused by resizing."""

    image = decode_image(str(image_path), mode="RGB")

    transform = v2.Compose(
        [
            v2.Resize(size=None, max_size=max_image_size),
            v2.ToDtype(torch.float32, scale=True),
            MyCustomTransform(padding_mode="edge"),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(image)


# File: kaamba_repo/src/kaamba/utils/plotting.py

"""
Plotting utilities for visualizing gaze data with stimulus backgrounds.
"""


def get_stimulus_image(
    gaze: pm.Gaze,
    dataset: pm.Dataset,
) -> Optional[np.ndarray]:
    """
    Load and return the stimulus image for a given gaze object.

    Args:
        gaze: A pymovements Gaze object
        dataset: The pymovements Dataset object containing fileinfo

    Returns:
        Image as numpy array (H, W, C) or None if not found
    """
    try:
        # Get stimulus identifier from metadata
        stimulus_id = gaze.metadata.get("stimulus")
        if stimulus_id is None:
            return None

        # Get stimulus file info from dataset
        stimuli = dataset.fileinfo["ImageStimulus"]
        stimulus_row = stimuli.filter(pl.col("stimulus") == stimulus_id)

        if stimulus_row.is_empty():
            return None

        # Construct full path to image
        image_path = Path(dataset.paths.stimuli) / stimulus_row["filepath"][0]

        if not image_path.exists():
            print(f"Warning: Stimulus image not found at {image_path}")
            return None

        # Load image
        image = Image.open(image_path).convert("RGB")
        return np.array(image)

    except Exception as e:
        print(f"Warning: Could not load stimulus image: {e}")
        return None


def plot_gaze_traces(
    gaze: pm.Gaze,
    dataset: Optional[pm.Dataset] = None,
    figsize: tuple = (14, 10),
    title: Optional[str] = None,
    show_stimulus: bool = True,
    stimulus_alpha: float = 0.7,
    trace_color: str = "red",
    trace_linewidth: float = 1.5,
    **kwargs,
) -> plt.Figure:
    """
    Visualize raw gaze samples as a continuous trajectory (traceplot) with stimulus background.

    This function uses pymovements' traceplot to show how gaze moves over time
    across the stimulus image, with the actual stimulus displayed in the background.

    Args:
        gaze: A pymovements Gaze object containing gaze samples
        dataset: The pymovements Dataset object. Required if show_stimulus=True
        figsize: Figure size as (width, height) in inches
        title: Title for the plot. If None, will use subject_id and stimulus metadata
        show_stimulus: Whether to display the stimulus image in the background
        stimulus_alpha: Transparency of stimulus image (0-1)
        trace_color: Color for the gaze trace line
        trace_linewidth: Width of the gaze trace line
        **kwargs: Additional arguments passed to pm.plotting.traceplot()

    Returns:
        matplotlib Figure object

    Example:
        >>> dataset = pm.Dataset('GazeBase', path=dataset_paths)
        >>> dataset.load()
        >>> fig = plot_gaze_traces(
        ...     gaze=dataset.gaze[0],
        ...     dataset=dataset,
        ...     title='Subject 001 - Stimulus 1'
        ... )
        >>> plt.show()
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # Load and display stimulus image if requested
    if show_stimulus and dataset is not None:
        stimulus_image = get_stimulus_image(gaze, dataset)
        if stimulus_image is not None:
            ax.imshow(
                stimulus_image,
                alpha=stimulus_alpha,
                extent=[
                    0,
                    gaze.experiment.screen.width_px,
                    gaze.experiment.screen.height_px,
                    0,
                ],
            )
            # Flip y-axis to match screen coordinates (origin at top-left)
            ax.invert_yaxis()

    # Create traceplot
    pm.plotting.traceplot(gaze, ax=ax, **kwargs)

    # Set title if provided
    if title is None:
        subject_id = gaze.metadata.get("subject_id", "Unknown")
        stimulus = gaze.metadata.get(
            "stimulus", gaze.metadata.get("stimulus_id", "Unknown")
        )
        title = f"Gaze Trace - Subject: {subject_id}, Stimulus: {stimulus}"

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X Position (pixels)", fontsize=11)
    ax.set_ylabel("Y Position (pixels)", fontsize=11)

    return fig


def plot_gaze_trace_comparison(
    gaze_list: List[pm.Gaze],
    dataset: Optional[pm.Dataset] = None,
    labels: Optional[List[str]] = None,
    figsize: tuple = (18, 12),
    ncols: int = 2,
    show_stimulus: bool = True,
    stimulus_alpha: float = 0.6,
    **kwargs,
) -> plt.Figure:
    """
    Plot multiple gaze traces side-by-side for comparison with stimulus backgrounds.

    Args:
        gaze_list: List of pymovements Gaze objects
        dataset: The pymovements Dataset object. Required if show_stimulus=True
        labels: Labels for each subplot. If None, will use metadata
        figsize: Figure size as (width, height) in inches
        ncols: Number of columns in subplot grid
        show_stimulus: Whether to display stimulus images in the background
        stimulus_alpha: Transparency of stimulus images
        **kwargs: Additional arguments passed to pm.plotting.traceplot()

    Returns:
        matplotlib Figure object

    Example:
        >>> dataset = pm.Dataset('GazeBase', path=dataset_paths)
        >>> dataset.load(subset={'subject_id': ['P01', 'P02']})
        >>> fig = plot_gaze_trace_comparison(
        ...     gaze_list=dataset.gaze,
        ...     dataset=dataset,
        ...     labels=[f"Subject {g.metadata['subject_id']}" for g in dataset.gaze]
        ... )
        >>> plt.show()
    """
    n_traces = len(gaze_list)
    nrows = int(np.ceil(n_traces / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for idx, gaze in enumerate(gaze_list):
        ax = axes[idx]

        # Load and display stimulus image if requested
        if show_stimulus and dataset is not None:
            stimulus_image = get_stimulus_image(gaze, dataset)
            if stimulus_image is not None:
                ax.imshow(
                    stimulus_image,
                    alpha=stimulus_alpha,
                    extent=[
                        0,
                        gaze.experiment.screen.width_px,
                        gaze.experiment.screen.height_px,
                        0,
                    ],
                )
                ax.invert_yaxis()

        # Create traceplot on this axis
        pm.plotting.traceplot(gaze, ax=ax, **kwargs)

        # Set label
        if labels is not None:
            label = labels[idx]
        else:
            subject_id = gaze.metadata.get("subject_id", "Unknown")
            stimulus = gaze.metadata.get("stimulus", "Unknown")
            label = f"Subject {subject_id}\nStimulus {stimulus}"

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("X (px)", fontsize=9)
        ax.set_ylabel("Y (px)", fontsize=9)

    # Hide unused subplots
    for idx in range(n_traces, len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    return fig


def plot_gaze_overlay(
    gaze: pm.Gaze,
    dataset: Optional[pm.Dataset] = None,
    figsize: tuple = (12, 9),
    title: Optional[str] = None,
    stimulus_alpha: float = 0.8,
    show_samples: bool = True,
    sample_color: str = "red",
    sample_size: int = 30,
    sample_alpha: float = 0.6,
) -> plt.Figure:
    """
    Plot stimulus image with gaze sample scatter overlay (alternative to traceplot).

    Useful for seeing where gaze landed without the temporal trajectory.

    Args:
        gaze: A pymovements Gaze object containing gaze samples
        dataset: The pymovements Dataset object. Required if not None
        figsize: Figure size as (width, height) in inches
        title: Title for the plot
        stimulus_alpha: Transparency of stimulus image
        show_samples: Whether to show gaze sample points
        sample_color: Color for gaze sample points
        sample_size: Size of gaze sample markers
        sample_alpha: Transparency of sample points

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Load and display stimulus image
    if dataset is not None:
        stimulus_image = get_stimulus_image(gaze, dataset)
        if stimulus_image is not None:
            ax.imshow(
                stimulus_image,
                alpha=stimulus_alpha,
                extent=[
                    0,
                    gaze.experiment.screen.width_px,
                    gaze.experiment.screen.height_px,
                    0,
                ],
            )
            ax.invert_yaxis()

    # Extract gaze positions
    if show_samples:
        samples_df = gaze.samples.to_pandas()

        if "pixel" in samples_df.columns:
            pixel_list = samples_df["pixel"].tolist()
            x_coords = [
                p[0] if isinstance(p, (list, tuple)) and len(p) > 0 else np.nan
                for p in pixel_list
            ]
            y_coords = [
                p[1] if isinstance(p, (list, tuple)) and len(p) > 1 else np.nan
                for p in pixel_list
            ]
        elif "x" in samples_df.columns and "y" in samples_df.columns:
            x_coords = samples_df["x"].values
            y_coords = samples_df["y"].values
        else:
            x_coords = y_coords = []

        # Plot scatter of gaze points
        ax.scatter(
            x_coords,
            y_coords,
            s=sample_size,
            c=sample_color,
            alpha=sample_alpha,
            edgecolors="none",
            label="Gaze samples",
        )
        ax.legend(loc="upper right", fontsize=10)

    # Set title
    if title is None:
        subject_id = gaze.metadata.get("subject_id", "Unknown")
        stimulus = gaze.metadata.get("stimulus", "Unknown")
        title = f"Gaze Distribution - Subject: {subject_id}, Stimulus: {stimulus}"

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X Position (pixels)", fontsize=11)
    ax.set_ylabel("Y Position (pixels)", fontsize=11)

    return fig


def plot_gaze_statistics(
    gaze: pm.Gaze,
    figsize: tuple = (14, 6),
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot gaze velocity and fixation statistics over time.

    Creates a detailed analysis of gaze behavior including:
    - Gaze position over time (x and y coordinates)
    - Gaze velocity
    - Acceleration

    Args:
        gaze: A pymovements Gaze object containing gaze samples
        figsize: Figure size as (width, height) in inches
        title: Title for the plot

    Returns:
        matplotlib Figure object
    """
    # Get gaze samples as DataFrame
    samples_df = gaze.samples.to_pandas()

    if "pixel" in samples_df.columns:
        # Extract x and y from pixel column
        pixel_list = samples_df["pixel"].tolist()
        x_coords = np.array(
            [
                p[0] if isinstance(p, (list, tuple)) and len(p) > 0 else np.nan
                for p in pixel_list
            ]
        )
        y_coords = np.array(
            [
                p[1] if isinstance(p, (list, tuple)) and len(p) > 1 else np.nan
                for p in pixel_list
            ]
        )
    elif "x" in samples_df.columns and "y" in samples_df.columns:
        x_coords = samples_df["x"].values
        y_coords = samples_df["y"].values
    else:
        raise ValueError("Could not find gaze coordinate columns in samples")

    # Calculate velocity using finite differences
    velocity = np.sqrt(
        np.diff(x_coords, prepend=x_coords[0]) ** 2
        + np.diff(y_coords, prepend=y_coords[0]) ** 2
    )

    time_axis = np.arange(len(x_coords))

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True)

    # Plot X coordinates
    axes[0].plot(time_axis, x_coords, "b-", alpha=0.7, linewidth=1)
    axes[0].set_ylabel("X Position (pixels)", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Gaze X Coordinate Over Time", fontsize=12, fontweight="bold")

    # Plot Y coordinates
    axes[1].plot(time_axis, y_coords, "r-", alpha=0.7, linewidth=1)
    axes[1].set_ylabel("Y Position (pixels)", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Gaze Y Coordinate Over Time", fontsize=12, fontweight="bold")

    # Plot velocity
    axes[2].plot(time_axis, velocity, "g-", alpha=0.7, linewidth=1)
    axes[2].fill_between(time_axis, velocity, alpha=0.3, color="g")
    axes[2].set_ylabel("Velocity (pixels/sample)", fontsize=11)
    axes[2].set_xlabel("Time (samples)", fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("Gaze Velocity Over Time", fontsize=12, fontweight="bold")

    if title is None:
        subject_id = gaze.metadata.get("subject_id", "Unknown")
        stimulus = gaze.metadata.get("stimulus", "Unknown")
        title = f"Gaze Statistics - Subject: {subject_id}, Stimulus: {stimulus}"

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()

    return fig


def scale_image_to_screen(
    image: Image.Image, screen_size: tuple[int, int]
) -> tuple[Image.Image, float, float, float, float]:
    """
    Scale a PIL Image to fit the screen while preserving aspect ratio.

    Returns:
        scaled_image          - the resized PIL Image
        offset_x, offset_y   - top-left position of image on screen (for letterboxing)
        scaled_w, scaled_h   - final image dimensions in pixels
    """
    img_w, img_h = image.size
    scr_w, scr_h = screen_size

    scale = min(scr_w / img_w, scr_h / img_h)

    scaled_w = int(img_w * scale)
    scaled_h = int(img_h * scale)

    scaled_image = image.resize((scaled_w, scaled_h), Image.LANCZOS)

    offset_x = (scr_w - scaled_w) / 2
    offset_y = (scr_h - scaled_h) / 2

    return scaled_image, offset_x, offset_y, scaled_w, scaled_h


def place_on_screen(
    scaled_image: Image.Image,
    screen_size: tuple[int, int],
    offset_x: float,
    offset_y: float,
    background: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """
    Paste the scaled image onto a full-screen canvas with letterbox bars.

    Returns a new PIL Image at screen resolution.
    """
    canvas = Image.new("RGB", screen_size, background)
    canvas.paste(scaled_image, (int(offset_x), int(offset_y)))
    return canvas


def main():
    screen = (1920, 1080)
    for image in range(20, 100):
        img = Image.open(
            f"C:\\Users\saphi\PycharmProjects\\thesis\data\mcfw_gaze\\raw\dataset\stimuli\\{image}.jpg"
        )

        scaled_img, offset_x, offset_y, scaled_w, scaled_h = scale_image_to_screen(
            img, screen
        )

        # Get the full screen image with letterbox bars
        screen_img = place_on_screen(scaled_img, screen, offset_x, offset_y)
        screen_img.save(
            f"C:\\Users\saphi\PycharmProjects\\thesis\data\mcfw_gaze\\stimuli\\{image}.jpg"
        )
