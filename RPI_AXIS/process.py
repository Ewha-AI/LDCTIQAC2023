import SimpleITK
from pathlib import Path
import random
from pandas import DataFrame
import torch
import cv2

from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
#import evalutils

import os
from models.maniqa import MANIQA

import json
import numpy as np

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
class DIQA(DetectionAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path = Path(os.path.join('test')),
            output_file = Path(os.path.join('output','image-quality-scores.json'))
        )
         
        path_model = os.path.join('model_weights','last.pt')
       
        self.model = MANIQA(embed_dim=768, num_outputs=1, dim_mlp=768,
                         patch_size=8, img_size=224, window_size=4,
                         depths=[2, 2], num_heads=[4,4], num_tab=2, scale=0.8)

        self.model.load_state_dict(torch.load(path_model, map_location=torch.device('cpu')))
        
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
                image = image_data[i,32:480,32:480]              
                image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      
                image = np.array(image).astype('float32')
                image = np.transpose(image, (2, 0, 1))
                image = (image - 0.5)/0.5
                image = np.expand_dims(image,axis=0)
                image = torch.tensor(image) # process one image
                prediction = model(image)
                prediction = float(prediction.cpu().numpy())
                predictions.append(prediction)

        return predictions


if __name__ == "__main__":
    cpu_num = 8
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)
    # loads the image(s), applies DL detection model & saves the result
    DIQA().process()
