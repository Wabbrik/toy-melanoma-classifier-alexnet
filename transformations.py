from typing import Callable, Dict

import hvplot.pandas
import pandas as pd
import panel as pn
from PIL import Image


def pil_image_view(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def plot_view(label: Dict[str, float]) -> pn.pane.HoloViews:
    data = pd.Series(label)
    return data.hvplot.bar(title="Prediction", ylim=(0, 1), color=["red", "blue"]).opts(default_tools=[])


def classification_view(classifier: Callable[[Image.Image], Dict[str, float]], image: Image.Image) -> pn.pane.HoloViews:
    return plot_view(classifier(image))
