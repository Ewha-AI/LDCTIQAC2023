import SimpleITK
from pathlib import Path

from pandas import DataFrame
import torch

from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import evalutils
import os
import json
import numpy as np

from model.model import TwoStage_EDCNN, TwoStage_EDCNN_Ensemble

# TODO: We have this parameter to adapt the paths between local execution and execution in docker. You can use this flag to switch between these two modes.
execute_in_docker = False

class IQARegression(DetectionAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path = Path("/input/images/synthetic-ct/") if execute_in_docker else Path("./test/"),
            output_file = Path("/output/image-quality-scores.json") if execute_in_docker else Path("./output/image-quality-scores.json")
        )
        # TODO: This path should lead to your model weights
        if execute_in_docker:
            path_model = "/opt/algorithm/checkpoints/model.pth"
        else:
            path_model = "./model_weights/model.pth"

        # TODO: Load your model
        self.model = TwoStage_EDCNN_Ensemble()

        self.model.load_state_dict(torch.load(path_model, map_location="cpu"), strict=True)
        print("Successfully loaded model.")

    def save(self):
        with open(str(self._output_file), "w") as f:
            f.write(json.dumps(self._case_results[0], indent=4))

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        # score candidates
        scored_candidates = self.predict(input_image=input_image)

        # Write resulting candidates to result.json for this case
        return scored_candidates

    def predict(self, *, input_image: SimpleITK.Image) -> DataFrame:
        class_dict = dict()
        numbers = [round(i * 0.2, 1) for i in range(21)]
        for i in range(21):
            class_dict[i] = numbers[i]

        # Extract a numpy array with image data from the SimpleITK Image
        image_data = SimpleITK.GetArrayFromImage(input_image)

        model = self.model.cuda()
        
        if len(image_data.shape) == 2:
            image_data = np.expand_dims(image_data, axis=0)


        with torch.no_grad():
            predictions = []
            for i in range(image_data.shape[0]):
                image = torch.tensor(np.expand_dims(np.expand_dims(image_data[i,:,:], axis=0), axis=0))
                image = image.cuda()
                prediction_reg, prediction_cls = model(image)
                prediction_reg = float(prediction_reg.cpu().numpy()[0][0])
                prediction_cls = np.argmax(prediction_cls.cpu().numpy(), axis=1)
                pred = 0.5*(prediction_reg+class_dict[int(prediction_cls)])
                predictions.append(pred)

        return predictions


if __name__ == "__main__":
    # loads the image(s), applies DL detection model & saves the result
    IQARegression().process()
