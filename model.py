import torch
from torch import nn 
import torch.nn.functional as F


from transformers import DistilBertModel

class IELTSScorer(nn.Module):
    def __init__(self, num_classes):
        super(IELTSScorer, self).__init__()

        self.name = "IELTSScorer"
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        for param in self.distilbert.parameters():
            param.requires_grad = False

        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.LazyLinear(64)
        self.fc2 = nn.LazyLinear(64)
        self.fc3 = nn.LazyLinear(64)
        self.fc4 = nn.LazyLinear(64)
        self.fc5 = nn.LazyLinear(num_classes)


    def forward(self, x):
        x = self.distilbert(x).last_hidden_state
        x = F.gelu(self.drop(self.fc1(x[:, -1, :])))
        x = F.gelu(self.drop(self.fc2(x)))
        x = F.gelu(self.drop(self.fc3(x)))
        x = F.gelu(self.drop(self.fc4(x)))
        return self.fc5(x)

