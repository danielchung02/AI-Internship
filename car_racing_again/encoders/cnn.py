import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, observation_shape: tuple, feature_dim:int | None =256):  # add
        #observation shape = (num of frames, height, width, channels)
        super().__init__()
        input_channels = observation_shape[0] * observation_shape[3]
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = input_channels, out_channels = 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
            nn.ReLU(),
            nn.Flatten()
        )

        dummy = torch.zeros([1, input_channels, observation_shape[1], observation_shape[2]], dtype = torch.float32)
        with torch.no_grad():
            flattened_dim = self.conv(dummy).shape[1]

        # self.projection = nn.Linear(flattened_dim, feature_dim)  # delete
        self.projection = nn.Identity() if feature_dim is None else nn.Linear(flattened_dim, feature_dim)  # add
        self.output_dim = flattened_dim if feature_dim is None else feature_dim  # add

    def forward(self, images: torch.Tensor):
        batch_size = images.shape[0]
        num_frames = images.shape[1]
        height = images.shape[2]
        width = images.shape[3]
        channels = images.shape[4]

        images = torch.permute(images,(0,1,4,2,3))
        images = images.reshape(batch_size, num_frames * channels, height, width)
        images = images.float()/255.0

        conv_features = self.conv(images)
        features = self.projection(conv_features)
        return features


        
