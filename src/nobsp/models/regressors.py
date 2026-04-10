import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# Class to create a regression model using feed-forward neural networks
#
# @Copyrigth:  Alexander Caicedo, April 2023

class Regression_NN(nn.Module):
    def __init__(self, in_number, out_number):
        super(Regression_NN, self).__init__()
        self.Linear_1 = nn.Linear(in_features = in_number, out_features = 100)
        self.Linear_2 = nn.Linear(in_features = 100, out_features = 500)
        self.Linear_3 = nn.Linear(in_features = 500, out_features = 1000)
        self.Linear_4 = nn.Linear(in_features = 1000, out_features = 1000)
        self.Linear_5 = nn.Linear(in_features = 1000, out_features = out_number)
        
    def forward(self,x):
        x = F.relu(self.Linear_1(x))
        x = F.relu(self.Linear_2(x))
        x = F.relu(self.Linear_3(x))
        x_t = F.relu(self.Linear_4(x)) # Computing the transformation done to the vector in the layer previous to the output
        x = self.Linear_5(x_t)
        
        return x, x_t
    
    def pdp(self, x, in_feat):
        
        # Detect the device where the model is currently located
        model_device = next(self.parameters()).device
        
        # Set to evaluation mode for inference (doesn't change device)
        self.eval()
        
        # Convert input to tensor and move to model's device if needed
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float().to(model_device)
        else:
            x = x.to(model_device)
        
        unique_vals = torch.unique(x[:, in_feat]).cpu().detach().numpy()
        y = []
        x_pdp = x.clone().detach()
        
        for val in unique_vals:
            x_pdp[:, in_feat] = val * torch.ones(x_pdp.size(dim=0), device=model_device)
            X = x_pdp
            
            with torch.inference_mode():  # Use inference mode for better performance
                output = self.forward(X)[0]
            
            a = torch.mean(output, axis=0).cpu().detach().numpy()
            y.append(a)
            
        unique_vals = np.expand_dims(unique_vals, axis=1)
        data = np.concatenate((unique_vals, y), axis=1)
        
        return data