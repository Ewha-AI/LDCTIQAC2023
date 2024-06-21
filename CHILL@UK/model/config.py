# This file describes the configureations for the whole experiment
import torch
import os
config={
    'random_seed':48, # Fix seed value for reproducibility
    'IM_W':512, # Input Image Width
    'IM_H':512, #Input Image Height
    'Batch':4,  # Batch size for the experiment
    'device':torch.device('cuda' if torch.cuda.is_available() else 'cpu'), # Device allocation for model/input
    'LR':0.0001, # Initial learning rate
    'epoch':100, # Total # of epochs to train
    'model':'resnet18', # Model name. Based on this parameter model will be generated from get_model.py
    'imgnet_pretrained':False,   #True: imagenet-pretrained model, False: Random initialization
    'loss_fn':'MSE',   #Loss function to train the model Possible options: MSE, MAE, huber
    'scheduler_warmup':1, # Cosine scheduler warmup
    'scheduler':'cos',  #LR scheduler function, Possible options: cos, step
    'scheduler_step':15, # Step for step scheduler
    'kfold':5,  # Number of folds for cross-validation
    'weight_decay':1e-5,  # Weight decay for adam optimizer
    'huber_delta':1.2,  # delta parameter for Humber loss function
    'wandb':True,   #False: No saving in Wandb, True: graphs, configs, etc. will be saved in wandb
    'multi_channel_input':False, #True: original input+ Normalized input, False: single channel input
    'multi_channel':2,  # Number of channels in case of multi-channel input=True, Otherwise it will be ignored
    'gauss_kernel':5,   # Kernel size of the gaussian blur. should be an odd number
    'only_normalized':False, # True: input is only normalized input
    'self-supervised':False, # True: self-supervised pretraining will be conducted
    'current_label':'gt', # This will alter in each fold first std and then label
    'multi-task':False, #False: single task, True: multi-task: classification+regression
    'krocc_loss':False, # True: MSE will be multiplied by 1-KROCC, False: MSE will not be multiplied
    'rank_mse':False, #True: rank information will be added to MSE, False: no rank info for MSE
    'discordant_penalty':False,# True: penalize loss with number of discordant pairs
    'normalized_output':False, # True: normalized output (0-1), False: general output
    'add_KL':False, #True: Main loss(MSE/MAE)+ KLDIVLOSS*KL_weight, False: KLDIVLOSS will not be added
    'KL_weight':0.01, # weight for KLDIVLOSS
    'SimCLR_temperature':0.5, # Temperature parameter for SimCLR
    'SimCLR_pretraining':False, # True: Enables the experiment for contrastive pretraining
    'Freeze_before_FC':False, # True: Freeze all layers before FC layers for downstream task
    'projection_head':128,
    'tcl_pretraining':False, #True: Triplet contrastive pretraining
    'all_data_training':False,
    'experiment_name': "EfficientNet-V2(L) Full data training" # String value as an experiment name.
}


