from torch.nn import Module

class Identity(Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x
