from torch import nn

class CNNEncoder(nn.Module):
    '''
    Modified LeNet encoder
    '''

    def __init__(self):
        super().__init__()

        # Input: 1 x 28 x 28
        # Output: 20 x 12 x 12
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
            )
        
        # Input: 20 x 12 x 12
        # Output: 50 x 4 x 4
        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
            )

        self.linearize = nn.Linear(50 * 4 * 4, 500)

    def forward(self, x):
        rv = self.conv2(self.conv1(x))

        # Flatten the output of conv2 to (batch_size, 50 * 4 * 4)
        rv = rv.view(rv.size(0), -1)
        rv = self.linearize(rv)
        return rv
    
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(500, 10)

    def forward(self, x):
        return self.hidden(x)

class Enc_With_Classifier(nn.Module):
    '''
    LeNet encoder with classifier
    '''

    def __init__(self):
        super().__init__()
        self.encoder = CNNEncoder()
        self.classifier = Classifier()

    def forward(self, x):
        return self.classifier(self.encoder(x))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2),
        )

    def forward(self, x):
        return self.hidden(x)
