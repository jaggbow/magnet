from torch import nn


activations = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'gelu': nn.GELU()
}
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_list, out_dim, activation='relu'):
        
        super().__init__()
        assert activation in ['relu', 'tanh', 'gelu']
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_list[0]))
        self.layers.append(activations[activation])
        
        for i in range(len(hidden_list)-1):
            self.layers.append(nn.Linear(hidden_list[i], hidden_list[i+1]))
            self.layers.append(activations[activation])
        self.layers.append(nn.Linear(hidden_list[-1],out_dim))
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out