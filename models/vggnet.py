import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DeepMatting(nn.Layer):
    def __init__(self, input_chn, output_chn, use_pretrained=True):
        super(DeepMatting, self).__init__()
        self.input_chn = input_chn

        # encoding
        self.conv11 = nn.Conv2D(input_chn, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2D(64)
        self.conv12 = nn.Conv2D(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2D(64)
        self.pool1 = nn.MaxPool2D((2, 2), stride=2, return_mask=True)

        self.conv21 = nn.Conv2D(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2D(128)
        self.conv22 = nn.Conv2D(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2D(128)
        self.pool2 = nn.MaxPool2D((2, 2), stride=2, return_mask=True)

        self.conv31 = nn.Conv2D(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2D(256)
        self.conv32 = nn.Conv2D(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2D(256)
        self.conv33 = nn.Conv2D(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2D(256)
        self.pool3 = nn.MaxPool2D((2, 2), stride=2, return_mask=True)

        self.conv41 = nn.Conv2D(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2D(512)
        self.conv42 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2D(512)
        self.conv43 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2D(512)
        self.pool4 = nn.MaxPool2D((2, 2), stride=2, return_mask=True)

        self.conv51 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2D(512)
        self.conv52 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2D(512)
        self.conv53 = nn.Conv2D(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2D(512)
        self.pool5 = nn.MaxPool2D((2, 2), stride=2, return_mask=True)

        self.conv6 = nn.Conv2D(512, 4096, kernel_size=7, padding=3)

        self.freeze_bn()

        # decoding
        self.dconv6 = nn.Conv2D(4096, 512, kernel_size=1, padding=0)

        self.unpool5 = nn.MaxUnPool2D((2, 2), stride=2)
        self.dconv5 = nn.Conv2D(512, 512, kernel_size=5, padding=2)

        self.unpool4 = nn.MaxUnPool2D((2, 2), stride=2)
        self.dconv4 = nn.Conv2D(512, 256, kernel_size=5, padding=2)

        self.unpool3 = nn.MaxUnPool2D((2, 2), stride=2)
        self.dconv3 = nn.Conv2D(256, 128, kernel_size=5, padding=2)

        self.unpool2 = nn.MaxUnPool2D((2, 2), stride=2)
        self.dconv2 = nn.Conv2D(128, 64, kernel_size=5, padding=2)

        self.unpool1 = nn.MaxUnPool2D((2, 2), stride=2)
        self.dconv1 = nn.Conv2D(64, 64, kernel_size=5, padding=2)

        self.alpha_pred = nn.Conv2D(64, 1, kernel_size=5, padding=2)
        
        # weights initialization
        self.weights_init_random()

    def forward(self, x):
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, idx1p = self.pool1(x12)

        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, idx2p = self.pool2(x22)

        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3p, idx3p = self.pool3(x33)

        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4p, idx4p = self.pool4(x43)

        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, idx5p = self.pool5(x53)

        x6 = F.relu(self.conv6(x5p))

        x6d = F.relu(self.dconv6(x6))

        x5d = self.unpool5(x6d, indices=idx5p)
        x5d = F.relu(self.dconv5(x5d))

        x4d = self.unpool4(x5d, indices=idx4p)
        x4d = F.relu(self.dconv4(x4d))

        x3d = self.unpool3(x4d, indices=idx3p)
        x3d = F.relu(self.dconv3(x3d))

        x2d = self.unpool2(x3d, indices=idx2p)
        x2d = F.relu(self.dconv2(x2d))

        x1d = self.unpool1(x2d, indices=idx1p)
        x1d = F.relu(self.dconv1(x1d))

        xpred = self.alpha_pred(x1d)

        return xpred

    def weights_init_random(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                initlizer = nn.initializer.KaimingUniform(None, 0, 'relu')
                initlizer(m.weight)
            elif isinstance(m, nn.BatchNorm2D):
                nn.initializer.Constant(1)(m.weight)
                nn.initializer.Constant(0)(m.bias)

    def freeze_bn(self):
        for m in self.sublayers():
            if isinstance(m, nn.BatchNorm2D):
                m.eval()


def vgg16(pretrained=False, **kwargs):
    """Constructs a VGG-16 model.

    """
    model = DeepMatting(input_chn=4, output_chn=1, use_pretrained=pretrained)
    return model


if __name__ == "__main__":
    net = DeepMatting(input_chn=4, output_chn=1, use_pretrained=True)
    net.eval()
    #net.cuda()

    dump_x = paddle.randn((1, 4, 224, 224)).cuda()
    print(paddle.summary(net, input=dump_x))

    from time import time
    import numpy as np
    frame_rate = np.zeros((20, 1))
    for i in range(20):
        x = paddle.randn((1, 4, 320, 320)).cuda()
        paddle.device.cuda.synchronize()
        start = time()
        y = net(x)
        paddle.device.cuda.synchronize()
        end = time()
        del y
        running_frame_rate = 1 * float(1 / (end - start))
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    # print(y.shape)