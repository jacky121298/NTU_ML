import torch.nn as nn

class conv_AE(nn.Module):
    def __init__(self):
        super(conv_AE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride = 2, padding = 1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride = 2, padding = 1), # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride = 2, padding = 1), # [batch, 48, 4, 4]
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(48, 24, 4, stride = 2, padding = 1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride = 2, padding = 1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride = 2, padding = 1),   # [batch, 3, 32, 32]
            nn.Tanh(),
        )

    def forward(self, x):
        x1 = self.encoder(x)
        x = self.decoder(x1)
        return x, x1