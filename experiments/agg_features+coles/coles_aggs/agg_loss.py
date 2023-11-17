import torch 
import torch.nn.functional as F
import torch.nn as nn

class AggLoss(nn.Module):
    def __init__(self, std_coeff=0.5, cov_coeff=0.5):
        super().__init__()
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, coles_out, agg_out):
        split_count = len(coles_out)/len(agg_out)
        coles_indx = [i*split_count for i in range(len(agg_out))]

        x = coles_out[coles_indx] - coles_out[coles_indx].mean(dim=0)

        y = agg_out - agg_out.mean(dim=0)
        y = F.normalize(agg_out, dim=0)
        
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)

        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov = (x.T @ y) / (len(x) - 1)

        cov_loss = cov.pow_(2).sum().div(x.size()[1]) 

        loss = (
            self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss