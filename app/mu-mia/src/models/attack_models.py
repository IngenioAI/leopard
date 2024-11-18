import torch.nn.functional as F
from torch import nn
import torch



class MIAFC(nn.Module):
    def __init__(self, input_dim=10, output_dim=2, dropout=0.2):
        super(MIAFC, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class SAMIA(nn.Module):
    def __init__(self, input_dim=10, output_dim=2, hidden_dim=64, num_layers=3, nhead=4, dropout=0.2):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.length = input_dim // 3
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(3, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=128,
                                                   dropout=dropout, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_dim * self.length, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.bn(x)
        x1, x2, x3 = x[:, :self.length].unsqueeze(2), x[:, self.length:self.length*2].unsqueeze(2), \
                     x[:, self.length*2:].unsqueeze(2)
        x = torch.cat([x1, x2, x3], dim=-1)
        x = F.gelu(self.fc1(x).permute(1, 0, 2))
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2).contiguous()
        x = x.view(-1, self.hidden_dim * self.length)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim=10, output_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim)
        )

    def forward_once(self, x):
        output = self.fc(x)
        return output

    def forward(self, vec1, vec2):
        out1 = self.forward_once(vec1)
        out2 = self.forward_once(vec2)
        return out1, out2

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        euclid_dist = F.pairwise_distance(
            out1, out2, keepdim=True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclid_dist, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclid_dist, min=0.0), 2))
        return loss_contrastive
 

class SiameseNetwork2(nn.Module):
    def __init__(self, input_dim=10, output_dim=1, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(256, 128)
        self.l2_dist = nn.MSELoss(reduction="none")
        self.linear4 = nn.Linear(128, output_dim)

    def forward(self, vec1, vec2):
        out1 = F.relu(self.linear1(vec1))
        out1 = self.dropout1(out1)
        out1 = F.relu(self.linear2(out1))
        out1 = self.dropout2(out1)
        out1 = F.relu(self.linear3(out1))

        out2 = F.relu(self.linear1(vec2))
        out2 = self.dropout1(out2)
        out2 = F.relu(self.linear2(out2))
        out2 = self.dropout2(out2)
        out2 = F.relu(self.linear3(out2))

        x = self.l2_dist(out1, out2)
        x = self.linear4(x)
        return torch.squeeze(x, 1)
    
