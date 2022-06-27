import torch

class Classifier(torch.nn.Module):

    def __init__(self):

        super(Classifier, self).__init__()
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False
            ), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm2d(num_features= 16),
            torch.nn.MaxPool2d(2), 
            torch.nn.Dropout(0.2), 
            
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=6,
                stride=1,
                padding=0,
                bias=False
            ), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm2d(num_features= 32),
            torch.nn.MaxPool2d(2), 
            torch.nn.Dropout(0.2), 
            
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=6,
                stride=1,
                padding=0,
                bias=False
            ), 
            torch.nn.ReLU(), 
            torch.nn.BatchNorm2d(num_features= 32), 
            torch.nn.MaxPool2d(2), 
            torch.nn.Dropout(0.2), 
            
            )
        # Categorical Classifier
        self.clf = torch.nn.Sequential(
            torch.nn.Linear(
                in_features= 32*4*4,
                out_features= 2,
                bias=True
            ),

            torch.nn.Softmax(dim=1)
        )
        
       
    
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.features(x)
    
    
    def forward(self, input):
        features = self.features(input)
        if input.requires_grad:
            h = features.register_hook(self.activations_hook)
        clf = self.clf(features.view(features.shape[0], -1))
        return clf, features