import torch

class Discriminator(torch.nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        self.main = torch.nn.Sequential(
            # input is (nc) x 128 x 128
            torch.nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ),
            torch.nn.Dropout(0.5, inplace=False),

           
            
            
            
            torch.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=32
            ),
            torch.nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ),
            torch.nn.Dropout(0.5, inplace=False),

            
            
            
            
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=64
            ),
            torch.nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ),
            torch.nn.Dropout(0.5, inplace=False),
            
            
            

            
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=128
            ),
            torch.nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ), 
            torch.nn.Dropout(0.5, inplace=False),

            
            
            
            
            torch.nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=256
            ),
            torch.nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ), 
            torch.nn.Dropout(0.5, inplace=False),
            
            
            
            
            
            torch.nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False
            ),
            torch.nn.BatchNorm2d(
                num_features=512
            ),
            torch.nn.LeakyReLU(
                negative_slope=0.2,
                inplace=True
            ), 
            torch.nn.Dropout(0.5, inplace=False),
            
            
            
            
            
            
        )
        
        
        # Real / Fake Classifier
        self.police = torch.nn.Sequential(
            torch.nn.Linear(5*5*512, 1), 
            torch.nn.Sigmoid()
            
        )

    def forward(self, input):
        
        features = self.main(input)
        #print(features.shape)
        valid = self.police(features.view(features.shape[0], -1)).view(-1, 1)
        return valid