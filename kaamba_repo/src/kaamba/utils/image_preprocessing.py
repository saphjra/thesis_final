import torch
from pathlib import Path


from torchvision.transforms import v2
from torchvision.io import decode_image


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
