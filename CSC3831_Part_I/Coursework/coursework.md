# Part 3 report **Computer Vision & AI**-**CSC3831**

## 1.	

def __init__(self, batch_normalization=False):

    super(NeuralNet, self).__init__()

    self.batch_normalization=batch_normalization

    def conv_block(in_channels, out_channels):

    layers= [nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU()]

    ifbatch_normalization:

    layers.append(nn.BatchNorm2d(out_channels))

    returnlayers

    self.block1= nn.Sequential(*conv_block(3, 32), *conv_block(32, 32), nn.MaxPool2d(2, 2))

    self.block2= nn.Sequential(*conv_block(32, 64), *conv_block(64, 64), nn.MaxPool2d(2, 2))

    self.block3= nn.Sequential(*conv_block(64, 128), *conv_block(128, 128), nn.MaxPool2d(2, 2))

    self.classifier= nn.Sequential(

    nn.Dropout(0.2),

    nn.Flatten(),

    nn.Linear(128*4*4, 128),

    nn.ReLU(),

    nn.Dropout(0.2),

    nn.Linear(128, 10)

    )

    def forward(self, x):

    x=self.block1(x)

    x=self.block2(x)

    x=self.block3(x)

    x=self.classifier(x)

    returnx
