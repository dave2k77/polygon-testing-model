from data import *

class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(30, 2) # model takes input of size 30, and output one of two possible values

    def forward(self, x):
        return self.linear(x)

# Move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LogisticRegressionModel().to(device)
