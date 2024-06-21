#!/usr/bin/env python
# coding: utf-8

# In[3]:


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

from model.ViT import DefineModel
from torchvision.models import vit_b_16


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
            #output_file = Path("/output/image-quality-scores.json") if execute_in_docker else Path("./output/image-quality-scores.json")
            output_file = Path("/output/") if execute_in_docker else Path("./output/")
        )
        # TODO: This path should lead to your model weights
        if execute_in_docker:
            path_model = "/opt/algorithm/checkpoints/ViT_395.pth"
        else:
            path_model = "./model_weights/ViT_395.pth"

        # TODO: Load your model
        vit=vit_b_16()
        self.model = DefineModel(model=vit, vit=True,classes=5, in_channels=1).set_model()
        device="cuda"
        device = torch.device("cpu" if not torch.cuda.is_available() else device) 
        self.model.load_state_dict(torch.load(path_model, map_location=device), strict=True)
        print("Successfully loaded model.")

    def save(self):
        file_name=str(self._output_file)+"/"+"image-quality-scores.json"
        print("saving ", file_name)
        
        with open(file_name, "w") as f:
        #with open(str(self._output_file), "w") as f:
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

        model = self.model

        # image_data will be a stack of 100 test images (i.e. shape equal to (100, 512, 512))
        image_data = SimpleITK.GetArrayFromImage(input_image)
        
        # when image shape is (512, 512). Do not change this code.
        if len(image_data.shape) == 2:
            image_data = np.expand_dims(image_data, axis=0)

        with torch.no_grad():

            # predictions must be a list containing float values (100 float scores for each slice image)
            predictions = []

            for i in range(image_data.shape[0]):
                image = torch.tensor(np.expand_dims(np.expand_dims(image_data[i,:,:], axis=0), axis=0)).float() # process one image
                pred_class_vector = model(image)
                pred_class=torch.mean(pred_class_vector, dim=1)
                print(pred_class)
                prediction = float(pred_class.cpu().numpy()[0])*4
                predictions.append(prediction)
                print(predictions)

        return predictions



if __name__ == "__main__":
    # loads the image(s), applies DL detection model & saves the result
    IQARegression().process()
    

