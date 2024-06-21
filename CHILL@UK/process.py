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

import json
import numpy as np
import time

#CUSTOM PACKAGES###
from model.get_model import get_model
from model.config import config #IMPORTANT: BEFORE SUBMISSION MAKE-SURE THE CONFIG.PY FILE IS CORRECT


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
        # Extract a numpy array with image data from the SimpleITK Image
        image_data = SimpleITK.GetArrayFromImage(input_image)

        if execute_in_docker:
            path_model = "/opt/algorithm/checkpoints/"
        else:
            path_model = "model_weights/"
        
        ##THIS LIST CONTAINS MODEL NAMES WITH THEIR WEIGHTS MUST LOOK-INTO THIS BEFORE SUBMISSION
        models_and_weights=[
            {
            'model_name':'effnetv2l',
            'model_weights':['effnetv2l_42_1','effnetv2l_42_2','effnetv2l_42_3','effnetv2l_42_4']
            }
        ]
        all_model_predictions=[]
        total_images=0
        for cur_model_dict in models_and_weights:
            config['model']=cur_model_dict['model_name']
            model=get_model(config=config)
            model=model.to(config['device'])
            model.eval()
            for cur_model_weight in cur_model_dict['model_weights']:
                model.load_state_dict(torch.load(path_model+cur_model_weight))
                # image_data will be a stack of 100 test images (i.e. shape equal to (100, 512, 512))
                image_data = SimpleITK.GetArrayFromImage(input_image)     
                # when image shape is (512, 512). Do not change this code.
                if len(image_data.shape) == 2:
                    image_data = np.expand_dims(image_data, axis=0)
                with torch.no_grad():

                    # predictions must be a list containing float values (100 float scores for each test batch)
                    predictions = []
                    total_images=image_data.shape[0]
                    for i in range(image_data.shape[0]):
                        image = torch.tensor(np.expand_dims(np.expand_dims(image_data[i,:,:], axis=0), axis=0)) # process one image
                        image=image.to(config['device'])
                        image=(image-torch.min(image))/(torch.max(image)-torch.min(image)) #Incase, if the image pixel values are not in 0-1 range therefore min-max normalization is applied
                        prediction = model(image)
                        prediction = float(prediction.cpu().numpy()[0][0])
                        predictions.append(prediction)
                    all_model_predictions.append(predictions)
        ensembled_predictions=[]

        #Doing average ensemble
        for i in range(total_images):
            cur_prediction_value=0
            for j in range(len(all_model_predictions)):
                cur_prediction_value=cur_prediction_value+all_model_predictions[j][i]
                # print(cur_prediction_value)
                # print(all_model_predictions[j][i])
            # print("-----------------")
            cur_prediction_value=cur_prediction_value/len(all_model_predictions)
            ensembled_predictions.append(cur_prediction_value)

        return ensembled_predictions


if __name__ == "__main__":
    # loads the image(s), applies DL detection model & saves the result
    # start_time=time.time()
    IQARegression().process()
    # end_time=time.time()
    # print("Time taken(s):",end_time-start_time)
