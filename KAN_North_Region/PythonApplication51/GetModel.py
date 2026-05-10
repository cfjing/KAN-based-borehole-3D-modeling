#-*- coding : utf-32 -*-
import os

import torch
from torch import nn
import torch.nn.functional as F

import DrawLog

import CONFIG
import PATH

import CustomLayers.KAN_Linear_Module

class CustumModule(torch.nn.Module):
    def __init__(self, in_pos_channels=CONFIG.NUM_FEATURE_POS, in_fac_channels=CONFIG.NUM_FEATURE_FAC, output_channels=CONFIG.NUM_CLASSES):
        super(CustumModule, self).__init__()
        
        
        
    def forward(self, x_pos, x_fac):
        
        x_pos = F.relu(self.pos_lin_1(x_pos))

        x_fac = F.relu(self.fac_lin_1(x_fac))

        x_merge = torch.concat((x_pos, x_fac), dim=-1)
        
        return x_merge

def get_model():
    model = CustumModule()

    if(os.path.exists(PATH.MODEL_WEIGHT_PATH)):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(PATH.MODEL_WEIGHT_PATH,map_location=device))
        model.eval()
        print("Model Weight Loaded!")
    else:
        print("Model Weight Doesn't Exist!")
        DrawLog.init_log()
    return model

def save_model(model):
    torch.save(model.state_dict(), PATH.MODEL_WEIGHT_PATH)

    
if(__name__ == "__main__"):
    model = get_model()
    print(model)
    # Total Number of Parameters in the Statistical Model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
