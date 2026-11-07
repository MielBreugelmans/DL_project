import torch.nn as nn

## Based on: https://arxiv.org/abs/1811.01424

class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, (3, 5, 5)), # input channels = 1 (grayscale), output channels = 64, kernel size = (3, 5, 5) (depth, height, width)
            nn.ReLU(), 
            nn.MaxPool3d(3), # pooling size = 3 (depth, height, width) -> applies to each dimension of the input tensor
            nn.Dropout(0.2),
            nn.Conv3d(64, 64, (3, 5, 5)), ## Output shape: (N, 64, 4, 5, 5) where N is the batch size. 
            #The output shape can be calculated as follows:
            ## General: (D_out = (D_in - kernel_size + 2 * padding) / stride + 1)
            ## Depth: (36 - 3) / 1 + 1 = 34
            ## Height: (48 - 5) / 1 + 1 = 44
            ## Width: (48 - 5) / 1 + 1 = 44
            nn.ReLU(),
            nn.MaxPool3d(2), 
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 5 * 5, 600), # Fully connected layer: input features = 64 * 4 * 5 * 5 because the output of the last Conv3D layer has shape (N, 64, 4, 5, 5), output features = 600
            nn.ReLU(),
            nn.Linear(600, 2),
        )

    def forward(self, x):               # x: (N, 1, 36, 48, 48)
        ## Why depth first? Because Conv3D expects input in the form (N, C, D, H, W) where N is batch size, C is number of channels, D is depth (number of slices), H is height, and W is width. 
        # In our case, we have 1 channel (grayscale), 36 slices (depth), and each slice is 48x48 pixels (height and width).
        return self.classifier(self.features(x))