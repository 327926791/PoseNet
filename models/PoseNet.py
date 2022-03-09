import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

import pickle


def init(key, module, weights=None):
    print(key)
    if weights == None:
        return module
    # initialize bias.data: layer_name_1 in weights
    # initialize  weight.data: layer_name_0 in weights
    module.bias.data = torch.from_numpy(weights[(key + "_1").encode()])
    module.weight.data = torch.from_numpy(weights[(key + "_0").encode()])

    return module


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, key=None, weights=None):
        super(InceptionBlock, self).__init__()

        # TODO: Implement InceptionBlock
        # Use 'key' to load the pretrained weights for the correct InceptionBlock

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            init(('inception_' + str(key) + '/1x1'), nn.Conv2d(in_channels, n1x1, kernel_size=1), weights),
            # nn.Conv2d(in_channels, n1x1, kernel_size=1),
            nn.ReLU(True),
        )

        # 1x1 -> 3x3 conv branch
        self.b2 = nn.Sequential(
            init(('inception_' + str(key) + '/3x3_reduce'), nn.Conv2d(in_channels, n3x3red, kernel_size=1), weights),
            # nn.Conv2d(in_channels, n3x3red, kernel_size=1),
            nn.ReLU(True),
            init(('inception_' + str(key) + '/3x3'), nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1), weights),
            # nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

        # 1x1 -> 5x5 conv branch
        self.b3 = nn.Sequential(
            init(('inception_' + str(key) + '/5x5_reduce'), nn.Conv2d(in_channels, n5x5red, kernel_size=1), weights),
            # nn.Conv2d(in_channels, n5x5red, kernel_size=1),
            nn.ReLU(True),
            init(('inception_' + str(key) + '/5x5'), nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2), weights),
            # nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # nn.ReLU(True),
            init(('inception_' + str(key) + '/pool_proj'), nn.Conv2d(in_channels, pool_planes, kernel_size=1), weights),
            # nn.Conv2d(in_channels, pool_planes, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        # TODO: Feed data through branches and concatenate
        x1 = self.b1(x)
        x2 = self.b2(x)
        x3 = self.b3(x)
        x4 = self.b4(x)
        return torch.cat([x1, x2, x3, x4], 1)


class LossHeader(nn.Module):
    def __init__(self, key, weights=None):
        super(LossHeader, self).__init__()

        # TODO: Define loss headers
        self.type = key
        print('lossheader: ' + str(self.type))

        # self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        # self.avg_pool5x5 = nn.AvgPool2d(kernel_size=5, stride=3)
        if key == 1:
            self.lossheader = nn.Sequential(
                nn.AvgPool2d(kernel_size=5, stride=3),
                # nn.ReLU(),
                init(('loss' + str(key) + '/conv'), nn.Conv2d(512, 128, kernel_size=1, stride=1), weights),
                nn.ReLU(True),
                nn.Flatten(),
                init(('loss' + str(key) + '/fc'), nn.Linear(2048, 1024), weights),
                nn.ReLU(True),
                nn.Dropout(p=0.7),
            )
            # self.conv512 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        elif key == 2:
            self.lossheader = nn.Sequential(
                nn.AvgPool2d(kernel_size=5, stride=3),
                # nn.ReLU(),
                init(('loss' + str(key) + '/conv'), nn.Conv2d(528, 128, kernel_size=1, stride=1), weights),
                nn.ReLU(True),
                nn.Flatten(),
                init(('loss' + str(key) + '/fc'), nn.Linear(2048, 1024), weights),
                nn.ReLU(True),
                nn.Dropout(p=0.7),
            )

            # self.conv528 = nn.Conv2d(528, 128, kernel_size=1, stride=1)
        else: #key == 3
            self.lossheader = nn.Sequential(
                nn.AvgPool2d(kernel_size=7, stride=1),
                # nn.ReLU(),
                nn.Flatten(),
                nn.Linear(1024, 2048),
                nn.ReLU(True),
                nn.Dropout(p=0.4)
            )


        # self.dropout4 = nn.Dropout(p=0.4)
        # self.dropout7 = nn.Dropout(p=0.7)
        # self.relu = nn.ReLU()
        self.xyz_1024 = nn.Linear(1024, 3)
        self.wpqr_1024 = nn.Linear(1024, 4)
        self.xyz_2048 = nn.Linear(2048, 3)
        self.wpqr_2048 = nn.Linear(2048, 4)

    def forward(self, x):
        # TODO: Feed data through loss headers
        # xyz = x
        # wpqr = x
        if self.type == 1:
            out = self.lossheader(x)
            xyz = self.xyz_1024(out)
            wpqr = self.wpqr_1024(out)
        elif self.type == 2:
            out = self.lossheader(x)
            xyz = self.xyz_1024(out)
            wpqr = self.wpqr_1024(out)
        else: #type == 3
            out = self.lossheader(x)
            xyz = self.xyz_2048(out)
            wpqr = self.wpqr_2048(out)
        return xyz, wpqr


class PoseNet(nn.Module):
    def __init__(self, load_weights=True):
        super(PoseNet, self).__init__()

        # Load pretrained weights file
        if load_weights:
            print("Loading pretrained InceptionV1 weights...")
            file = open('pretrained_models/places-googlenet.pickle', "rb")
            weights = pickle.load(file, encoding="bytes")
            file.close()
        # Ignore pretrained weights
        else:
            weights = None

        # for key, value in weights.items():
        #     print(key)
        # print(weights.keys())
        # print(weights)
        # TODO: Define PoseNet layers

        self.pre_layers = nn.Sequential(
            # Example for defining a layer and initializing it with pretrained weights
            init('conv1/7x7_s2', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), weights),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.LocalResponseNorm(5, 0.0001, 0.75),
            init('conv2/3x3_reduce', nn.Conv2d(64, 64, kernel_size=1, stride=1), weights),
            nn.ReLU(True),
            init('conv2/3x3', nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), weights),
            nn.ReLU(True),
            nn.LocalResponseNorm(5, 0.0001, 0.75),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
        )

        # Example for InceptionBlock initialization
        self._3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32, "3a", weights)
        self._3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64, "3b", weights)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64, "4a", weights)
        self._4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64, "4b", weights)
        self._4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64, "4c", weights)
        self._4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64, "4d", weights)
        self._4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128, "4e", weights)

        self._5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128, "5a", weights)
        self._5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128, "5b", weights)

        self.LossHeader1 = LossHeader(1, weights)
        self.LossHeader2 = LossHeader(2, weights)
        self.LossHeader3 = LossHeader(3, weights)

        print("PoseNet model created!")

    def forward(self, x):
        # TODO: Implement PoseNet forward
        print(x.shape)
        out = self.pre_layers(x)
        out = self._3a(out)
        out = self._3b(out)
        out = self.max_pool(out)
        out = self._4a(out)
        loss1_xyz, loss1_wpqr = self.LossHeader1(out)
        out = self._4b(out)
        out = self._4c(out)
        out = self._4d(out)
        loss2_xyz, loss2_wpqr = self.LossHeader2(out)
        out = self._4e(out)
        out = self.max_pool(out)
        out = self._5a(out)
        out = self._5b(out)
        loss3_xyz, loss3_wpqr = self.LossHeader3(out)

        if self.training:
            return loss1_xyz, \
                   loss1_wpqr, \
                   loss2_xyz, \
                   loss2_wpqr, \
                   loss3_xyz, \
                   loss3_wpqr
        else:
            return loss3_xyz, \
                   loss3_wpqr


class PoseLoss(nn.Module):

    def __init__(self, w1_xyz, w2_xyz, w3_xyz, w1_wpqr, w2_wpqr, w3_wpqr):
        super(PoseLoss, self).__init__()

        self.w1_xyz = w1_xyz
        self.w2_xyz = w2_xyz
        self.w3_xyz = w3_xyz
        self.w1_wpqr = w1_wpqr
        self.w2_wpqr = w2_wpqr
        self.w3_wpqr = w3_wpqr


    def forward(self, p1_xyz, p1_wpqr, p2_xyz, p2_wpqr, p3_xyz, p3_wpqr, poseGT):
        # TODO: Implement loss
        # First 3 entries of poseGT are ground truth xyz, last 4 values are ground truth wpqr
        pose_xyz = poseGT[:, 0:3]
        pose_wpqr = poseGT[:, 3:]
        # print("pose_xyz " + str(pose_xyz))
        # print("pose_wpqr " + str(pose_wpqr))
        # print("p1_xyz " + str(p1_xyz))
        # print("p1_wpqr " + str(p1_wpqr))
        # print("p2_xyz " + str(p2_xyz))
        # print("p2_wpqr " + str(p2_wpqr))
        # print("p3_xyz " + str(p3_xyz))
        # print("p3_wpqr " + str(p3_wpqr))

        # pose_xyz = pose_xyz / torch.norm(pose_xyz, 1)
        # pose_wpqr = pose_wpqr / torch.norm(pose_wpqr, 1)

        # p1_wpqr = F.normalize(p1_wpqr, p=2, dim=1)
        # p2_wpqr = F.normalize(p2_wpqr, p=2, dim=1)
        # p3_wpqr = F.normalize(p3_wpqr, p=2, dim=1)
        pose_wpqr = F.normalize(pose_wpqr, p=2, dim=1)

        # print("p1_wpqr after norm " + str(p1_wpqr))
        # print("p2_wpqr after norm " + str(p2_wpqr))
        # print("p3_wpqr after norm " + str(p3_wpqr))
        # print("pose_wpqr after norm " + str(pose_wpqr))
        # pose_q = F.normalize(pose_q, p=2, dim=1)
        mse = torch.nn.MSELoss()
        l1_xyz = mse(p1_xyz, pose_xyz)
        l1_wpqr = mse(p1_wpqr, pose_wpqr) * self.w1_wpqr
        l2_xyz = mse(p2_xyz, pose_xyz)
        l2_wpqr = mse(p2_wpqr, pose_wpqr) * self.w2_wpqr
        l3_xyz = mse(p3_xyz, pose_xyz)
        l3_wpqr = mse(p3_wpqr, pose_wpqr) * self.w3_wpqr

        print("l1_xyz " + str(l1_xyz))
        print("l1_wpqr " + str(l1_wpqr))
        print("l2_xyz " + str(l2_xyz))
        print("l2_wpqr " + str(l2_wpqr))
        print("l3_xyz " + str(l3_xyz))
        print("l3_wpqr " + str(l3_wpqr))

        # l1_xyz = torch.norm((p1_xyz-pose_xyz), 2)
        # l1_wpqr = torch.norm((p1_wpqr - pose_wpqr),2) * self.w1_wpqr
        # l2_xyz = torch.norm((p2_xyz-pose_xyz), 2)
        # l2_wpqr = torch.norm((p2_wpqr - pose_wpqr), 2) * self.w2_wpqr
        # l3_xyz = torch.norm((p3_xyz - pose_xyz), 2)
        # l3_wpqr = torch.norm((p3_wpqr - pose_wpqr), 2) * self.w3_wpqr

        # l1_x = torch.sqrt(torch.sum(Variable(torch.Tensor(np.square(F.pairwise_distance(pose_x, p1_xyz).detach().cpu().numpy())), requires_grad=True)))
        # l1_q = torch.sqrt(torch.sum(Variable(torch.Tensor(np.square(F.pairwise_distance(pose_q, p1_wpqr).detach().cpu().numpy())), requires_grad=True))) * self.w1_wpqr
        # l2_x = torch.sqrt(torch.sum(Variable(torch.Tensor(np.square(F.pairwise_distance(pose_x, p2_xyz).detach().cpu().numpy())), requires_grad=True)))
        # l2_q = torch.sqrt(torch.sum(Variable(torch.Tensor(np.square(F.pairwise_distance(pose_q, p2_wpqr).detach().cpu().numpy())), requires_grad=True))) * self.w2_wpqr
        # l3_x = torch.sqrt(torch.sum(Variable(torch.Tensor(np.square(F.pairwise_distance(pose_x, p3_xyz).detach().cpu().numpy())), requires_grad=True)))
        # l3_q = torch.sqrt(torch.sum(Variable(torch.Tensor(np.square(F.pairwise_distance(pose_q, p3_wpqr).detach().cpu().numpy())), requires_grad=True))) * self.w3_wpqr

        # loss = l1_x + l1_q + l2_x + l2_q + l3_x + l3_q
        loss = self.w1_xyz * (l1_xyz + l1_wpqr) + self.w2_xyz * (l2_xyz + l2_wpqr) + self.w3_xyz * (l3_xyz + l3_wpqr)

        return loss
