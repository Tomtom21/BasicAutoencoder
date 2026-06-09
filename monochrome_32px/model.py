from torch import nn

class MonochromeAutoEncoder(nn.Module):
    def __init__(self):
        super(MonochromeAutoEncoder, self).__init__()

        # Encoder layers
        # Input of 32x32
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1) # 32x32
        self.pooling1 = nn.MaxPool2d((2, 2)) # 16x16

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 16x16
        self.pooling2 = nn.MaxPool2d((2, 2)) # 8x8

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1) # 8x8
        self.pooling3 = nn.MaxPool2d((2, 2)) # 4x4

        # Decoder layers
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1) # 8x8

        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1) # 16x16

        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1) # 32x32

    def encode(self, x):
        x = self.conv1(x)
        x = self.pooling1(x)

        x = self.conv2(x)
        x = self.pooling2(x)

        x = self.conv3(x)
        x = self.pooling3(x)

        return x


    def decode(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)

        return x

    def forward(self, x):
        return self.decode(self.encode(x))
