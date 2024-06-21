import torch
import torchvision.models as models

def get_model(config):
    if config["model"]=="resnet18":
        model=models.resnet18()
        num_ftrs = model.fc.in_features
        model.conv1=torch.nn.Conv2d(in_channels=1,out_channels=64, kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        model.fc=torch.nn.Sequential(
            torch.nn.Linear(num_ftrs,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,1)
        )  
    elif config['model']=='convnextl':
        model=models.convnext_large()
        model.features[0][0]=torch.nn.Conv2d(1, 192, kernel_size=(4, 4), stride=(4, 4))
        num_ftrs=model.classifier[2].in_features
        if config['normalized_output']==True:
            model.classifier[2]=torch.nn.Sequential(
                torch.nn.Linear(num_ftrs,1),
                torch.nn.Sigmoid()
            )
        if config['normalized_output']==False:
            model.classifier[2]=torch.nn.Linear(num_ftrs,1)

        model=model.to(config['device'])
    elif config["model"]=="resnet50":
        ##print("loading resnet-18")
        model = models.resnet50(pretrained=config['imgnet_pretrained'])
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs,1)
        model.conv1=torch.nn.Conv2d(in_channels=1,out_channels=64, kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        model=model.to(config['device'])
    elif config["model"]=="effnetv2s":
        model=models.efficientnet_v2_s("EfficientNet_V2_S_Weights.IMAGENET1K_V1")
        model.features[0][0]=torch.nn.Conv2d(in_channels=1,out_channels=24, kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        num_ftrs=model.classifier[1].in_features 
        model.classifier[1] = torch.nn.Linear(num_ftrs,1)
        model=model.to(config['device'])
    elif config["model"]=="effnetv2m":
        model=models.efficientnet_v2_m("EfficientNet_V2_M_Weights.IMAGENET1K_V1")
        model.features[0][0]=torch.nn.Conv2d(in_channels=1,out_channels=24, kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        num_ftrs=model.classifier[1].in_features 
        model.classifier[1] = torch.nn.Linear(num_ftrs,1)
        model=model.to(config['device'])   
        
    elif config["model"]=="effnetv2l":
        model=models.efficientnet_v2_l()
        if config['multi_channel_input']==True:
            model.features[0][0]=torch.nn.Conv2d(in_channels=2,out_channels=32, kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        else:
            model.features[0][0]=torch.nn.Conv2d(in_channels=1,out_channels=32, kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        
        if config['SimCLR_pretraining']==False and config['tcl_pretraining']==False:
            # In case of config['SimCLR_pretraining']=True model will return 1000 class as the embedding      
            num_ftrs=model.classifier[1].in_features 
            model.classifier[1] = torch.nn.Linear(num_ftrs,1)
            
        if config['tcl_pretraining']==True:
            num_ftrs=model.classifier[1].in_features
            model.classifier[1]=torch.nn.Sequential(
                torch.nn.Linear(num_ftrs,256),
                torch.nn.ReLU(),
                torch.nn.Linear(256,1)
            )   
        model=model.to(config['device'])   
    elif config["model"]=="vit_b32":
        model=models.vit_b_32(pretrained=config['imgnet_pretrained'])
        if config['multi_channel_input']==True:
            model.conv_proj=torch.nn.Conv2d(in_channels=2,out_channels=768, kernel_size=(32,32),stride=(32,32))
        else:
            model.conv_proj=torch.nn.Conv2d(in_channels=1,out_channels=768, kernel_size=(32,32),stride=(32,32))
        num_ftrs=model.heads.head.in_features
        model.heads.head = torch.nn.Linear(num_ftrs,1)
        model=model.to(config['device'])
    elif config["model"]=="swin_v2_b":
        model=models.swin_v2_b(pretrained=config['imgnet_pretrained'])
        if config['multi_channel_input']==True:
            model.features[0][0]=torch.nn.Conv2d(2, 128, kernel_size=(4, 4), stride=(4, 4))
        else:
            model.features[0][0]=torch.nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
        num_ftrs=model.head.in_features
        model.head = torch.nn.Linear(num_ftrs,1)
        model=model.to(config['device'])

    return model