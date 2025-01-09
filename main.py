from io import BytesIO
from typing import Dict, Optional

import panel as pn
from panel.widgets import Tqdm
from PIL import Image

from classifier.classifier import CustomModel, classify_image, try_load_cached_model
from classifier.dataset import ClassifierClasses, fetch_dataset
from transformations import classification_view, pil_image_view

MODEL: Optional[CustomModel] = None
IMAGE_DIM = 350
DATASET_PATH: str = fetch_dataset()

pn.extension(design="material", sizing_mode="stretch_width")


def predict(image: Optional[Image.Image]) -> Dict[str, float]:
    if not image:
        return {classification.name: 0 for classification in ClassifierClasses}

    global MODEL
    if not MODEL:
        label_view.visible = False
        tqdm.visible = True
        MODEL = try_load_cached_model(tqdm)
        tqdm.visible = False
        label_view.visible = True

    return classify_image(MODEL, image)


tqdm = Tqdm(visible=False)
image_view = pn.pane.Image(None, height=IMAGE_DIM, width=IMAGE_DIM, fixed_aspect=False, margin=0)
file_input = pn.widgets.FileInput(accept="image/*")
file_input_component = pn.Column("### Upload Image", file_input)
logo_component = pn.pane.Image("resources/logo.png", width=IMAGE_DIM, margin=10)
label_view = pn.Row(
    pn.panel(
        pn.bind(classification_view, predict, image=image_view.param.object),
        defer_load=True,
        loading_indicator=True,
        height=IMAGE_DIM,
        width=IMAGE_DIM,
    ),
)

title_component = pn.Row(
    pn.Column(),
    pn.pane.Markdown(
        "# Melanoma Classification",
        width_policy="max",
        sizing_mode="stretch_both",
        styles={"font-size": "25px"},
    ),
    pn.Column(),
)
input_component = pn.Column("# Input", image_view, file_input_component, width=IMAGE_DIM)
output_component = pn.Row(pn.Column("# Output", tqdm, label_view, width=IMAGE_DIM), logo_component)


def handle_file_upload(file: bytes, image_pane: pn.pane.Image) -> None:
    image_pane.object = pil_image_view(BytesIO(file))


def image_classification_interface() -> pn.layout.FlexBox:
    pn.bind(handle_file_upload, file_input, image_view, watch=True)
    return pn.Row(pn.Column(), pn.Column(title_component, pn.Row(input_component, output_component)), pn.Column())


print(DATASET_PATH)
image_classification_interface().servable()
