import torch.nn as nn
import torch.nn.functional as F

class _Core_DNNRegression(nn.Module):
    def __init__(self, D_in, H, D_out, dropout_rate):
        '''
        D_in  : Number of input features in the context
        H     : List [hidden_size_1, hidden_size_2, ... , hidden_size_n]
        D_out : Number of output actions
        '''
        super(_Core_DNNRegression, self).__init__()
        self.num_hidden_layers = len(H)
        self.input_layer = nn.Linear(D_in, H[0])
        hidden_layers = [None]*(4*(self.num_hidden_layers-1)) # num_hidden_layers - 1 because last is output layer
        for i in range(self.num_hidden_layers-1):
            hidden_layers[4*i]  = nn.Linear(H[i], H[i+1])
            hidden_layers[4*i+1] = nn.BatchNorm1d(H[i+1])
            hidden_layers[4*i+2] = nn.Dropout(p=dropout_rate)
            hidden_layers[4*i+3] = nn.ReLU()
        self.net = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(H[self.num_hidden_layers-1], D_out)
        
    def forward(self, x):
        '''
        x : context 1D Tensor
        '''
        output = F.relu(self.input_layer(x))
        output = self.net(output)
        output = self.output_layer(output)
        
        return output