import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
from models.PoseNet import PoseNet, PoseLoss
from data.DataSource import *
from optparse import OptionParser
import os
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(epochs, batch_size, learning_rate, save_freq, data_dir):
    save_path = 'checkpoints/checkpoint.pt'
    # train dataset and train loader
    datasource = DataSource(data_dir, train=True)
    train_loader = Data.DataLoader(dataset=datasource, batch_size=batch_size, shuffle=True)

    # load model
    posenet = PoseNet(load_weights=True).to(device)

    # loss function
    criterion = PoseLoss(0.3, 0.3, 1., 300, 300, 300)

    # train the network
    optimizer = torch.optim.Adam(nn.ParameterList(posenet.parameters()),
                     lr=learning_rate, eps=1,
                     weight_decay=0.0625,
                     betas=(0.9, 0.999))

    batches_per_epoch = len(train_loader.batch_sampler)
    train_loss = []
    loss_per_epoch = 0
    for epoch in range(epochs):
        print("Starting epoch {}:".format(epoch))
        posenet.train()
        for step, (images, poses) in enumerate(train_loader):
            b_images = Variable(images, requires_grad=True).to(device)
            poses[0] = np.array(poses[0])
            poses[1] = np.array(poses[1])
            torch.save(posenet.state_dict(), save_path)
            print("Network saved!")

            poses[2] = np.array(poses[2])
            poses[3] = np.array(poses[3])
            poses[4] = np.array(poses[4])
            poses[5] = np.array(poses[5])
            poses[6] = np.array(poses[6])
            poses = np.transpose(poses)
            b_poses = Variable(torch.Tensor(poses), requires_grad=True).to(device)

            # print(b_images.shape)
            p1_x, p1_q, p2_x, p2_q, p3_x, p3_q = posenet(b_images)

            # p_xyz_np = p3_x.detach().cpu().numpy()
            # p_wpqr_np = p3_q.detach().cpu().numpy()
            # for i in range(b_poses.shape[0]):
            #     print("{}".format(step * batch_size + i))
            #     print("GT\t| xyz: {}\twpqr: {}".format(poses[i, :3], poses[i, 3:]))
            #     print("PRED\t| xyz: {}\twpqr: {}".format(p_xyz_np[i], p_wpqr_np[i]))


            loss = criterion(p1_x, p1_q, p2_x, p2_q, p3_x, p3_q, b_poses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("{}/{}: loss = {}".format(step+1, batches_per_epoch, loss / batch_size))
            loss_per_epoch += loss.item() / batch_size
        loss_per_epoch /= batches_per_epoch
        print("append loss: " + str(loss_per_epoch))
        train_loss.append(loss_per_epoch)
        # Save state
        if epoch % save_freq == 0:
            print("checkpoint save")
            save_filename = 'epoch_{}.pth'.format(str(epoch+1).zfill(5))
            save_path = os.path.join('checkpoints', save_filename)
            torch.save(posenet.state_dict(), save_path)

    plt.plot(range(1, epochs + 1), train_loss, 'bo', label="training")
    plt.legend()
    plt.show()

def get_args():
    parser = OptionParser()
    parser.add_option('--epochs', default=200, type='int')
    parser.add_option('--learning_rate', default=0.0001)
    parser.add_option('--batch_size', default=75, type='int')
    parser.add_option('--save_freq', default=20, type='int')
    parser.add_option('--data_dir', default='data/datasets/KingsCollege/')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    main(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, save_freq=args.save_freq, data_dir=args.data_dir)
