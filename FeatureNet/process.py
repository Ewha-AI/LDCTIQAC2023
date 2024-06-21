import SimpleITK
from pathlib import Path
import pywt
import skimage

from pandas import DataFrame
import torch

from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
import evalutils
from PIL import Image
import json
import numpy as np

from model.model import ResNet18
from model.featureNet import featureNet


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
           path_model = "/opt/algorithm/checkpoints/resnet18.pth"
        else:
           path_model = "./model_weights/resnet18.pth"
            

        # TODO: Load your model
        self.vit_model = ResNet18()
        self.model = featureNet(self.vit_model)

        self.model.load_state_dict(torch.load(path_model), strict=True)
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

    def make_wave_data(self, image):
        image = np.array(image)
        
        #image = np.pad(image, (126,126), 'constant', constant_values=(0.0))
        
        coeffs2 = pywt.dwt2(np.squeeze(image), 'bior1.3')
        
        return coeffs2

    def make_features(self, image):
        image = np.array(image)*255.0

        dist = [1,3]
        angle = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        features = []
        num_feature = len(dist)*len(angle)
        num_metric = 5

        glcm = skimage.feature.graycomatrix(image.astype(np.uint8),dist, angle)

        tmp = []
        tmp.append(np.array(skimage.feature.graycoprops(glcm,'contrast')).reshape((num_feature,-1)).squeeze())
        tmp.append(np.array(skimage.feature.graycoprops(glcm,'dissimilarity')).reshape((num_feature,-1)).squeeze())
        tmp.append(np.array(skimage.feature.graycoprops(glcm,'homogeneity')).reshape((num_feature,-1)).squeeze())
        tmp.append(np.array(skimage.feature.graycoprops(glcm,'energy')).reshape((num_feature,-1)).squeeze())
        tmp.append(np.array(skimage.feature.graycoprops(glcm,'correlation')).reshape((num_feature,-1)).squeeze())
        tmp = np.array(tmp).reshape((num_feature*num_metric,-1)).squeeze()
        
        return tmp

    def predict(self, *, input_image: SimpleITK.Image) -> DataFrame:
        # Extract a numpy array with image data from the SimpleITK Image
        image_data = SimpleITK.GetArrayFromImage(input_image)

        model = self.model

        # image_data will be a stack of 100 test images (i.e. shape equal to (100, 512, 512))
        image_data = SimpleITK.GetArrayFromImage(input_image)
        
        # when image shape is (512, 512). Do not change this code.
        if len(image_data.shape) == 2:
            image_data = np.expand_dims(image_data, axis=0)

        model.eval()
        with torch.no_grad():

            # predictions must be a list containing float values (100 float scores for each slice image)
            predictions = []

            for i in range(image_data.shape[0]):
                temp = Image.fromarray(image_data[i,:,:])
                
                features = self.make_features(temp)
                
                LL, (LH, HL, HH) = self.make_wave_data(temp)
        
                LL = np.expand_dims(LL, axis=0)
                LH = np.expand_dims(LH, axis=0)
                HL = np.expand_dims(HL, axis=0)
                HH = np.expand_dims(HH, axis=0)
        
                wave_set = np.concatenate((LL,LH,HL,HH), axis=0)
                
                temp = temp.resize((384,384))
                temp = np.array(temp)
                
                image = torch.tensor(np.expand_dims(np.expand_dims(temp, axis=0), axis=0)) # process one image
                image = torch.tensor(image, dtype=torch.float32)
                
                features = torch.tensor(np.expand_dims(features, axis=0)) # process one image
                features = torch.tensor(features, dtype=torch.float32)
                
                wave_set = torch.tensor(np.expand_dims(wave_set, axis=0)) # process one image
                wave_set = torch.tensor(wave_set, dtype=torch.float32)
                
                prediction = model(image, wave_set, features)
                prediction = torch.nn.functional.hardtanh(prediction, 0, 4)
                prediction = float(prediction.cpu().numpy()[0][0])
                predictions.append(prediction)

        return predictions


if __name__ == "__main__":
    # loads the image(s), applies DL detection model & saves the result
    IQARegression().process()


