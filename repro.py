"""
Running this script eventually gives:
23
eval: split train. loss 4.073383e-03. error 0.62%. misses: 45
eval: split test . loss 2.838382e-02. error 4.09%. misses: 82
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter # pip install tensorboardX

# -----------------------------------------------------------------------------

class ApproxNet(nn.Module):
    """ Approximate repro (see README) of a repro of the 1989 LeCun conv net"""

    def __init__(self, W_c1, W_c2, W_l1, W_l2, b1, b2, enable_convolution_biases=False, init_separately=False):
        super().__init__()
        self.b_c1 = torch.zeros(12, 8, 8)
        self.b_c2 = torch.zeros(8, 4, 4)
        # if enabled, then we learn biases along with the convolution
        if enable_convolution_biases:
            self.b_c1 = nn.Parameter(self.b_c1)
            self.b_c2 = nn.Parameter(self.b_c2)
        # if enabled, then we initialize independently of our C repro init
        if init_separately:
            self.W_c1 = nn.Parameter(winit(25, 5, 5, 1, 12))
            self.W_c2 = nn.Parameter(winit(25 * 12, 5, 5, 12, 8))
            self.W_l1 = nn.Parameter(winit(8 * 4 * 4, 12 * 4 * 4, 30))
            self.W_l2 = nn.Parameter(winit(8 * 4 * 4, 12 * 4 * 4, 30))
            self.b1 = nn.Parameter(torch.zeros(30))
            self.b2 = nn.Parameter(torch.zeros(10))
        else:
            self.W_c1 = nn.Parameter(W_c1)
            self.W_c2 = nn.Parameter(W_c2)
            self.W_l1 = nn.Parameter(W_l1)
            self.W_l2 = nn.Parameter(W_l2)
            self.b1 = nn.Parameter(b1)
            self.b2 = nn.Parameter(b2)

    def forward(self, x):
        x = F.pad(x, (2, 1, 2, 1), 'constant', -1.0)
        x = F.conv2d(x, self.W_c1.permute(3, 2, 0, 1), stride=2) + self.b_c1
        x = torch.tanh(x)
        x = F.pad(x, (2, 1, 2, 1), 'constant', -1.0)
        x = F.conv2d(x, self.W_c2.permute(3, 2, 0, 1), stride=2) + self.b_c2
        x = torch.tanh(x)
        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], 4 * 4 * 8) @ self.W_l1 + self.b1
        x = torch.tanh(x)
        x = x @ self.W_l2 + self.b2
        x = torch.tanh(x)
        return x

class Net(nn.Module):
    """ 1989 LeCun ConvNet per description in the paper """

    def __init__(self):
        super().__init__()
        # initialization as described in the paper to my best ability, but it doesn't look right...
        winit = lambda fan_in, *shape: (torch.rand(*shape) - 0.5) * 2 * 2.4 / fan_in**0.5
        macs = 0 # keep track of MACs (multiply accumulates)
        acts = 0 # keep track of number of activations

        # H1 layer parameters and their initialization
        self.H1w = nn.Parameter(winit(5*5*1, 12, 1, 5, 5))
        self.H1b = nn.Parameter(torch.zeros(12, 8, 8)) # presumably init to zero for biases
        assert self.H1w.nelement() + self.H1b.nelement() == 1068
        macs += (5*5*1) * (8*8) * 12
        acts += (8*8) * 12

        # H2 layer parameters and their initialization
        """
        H2 neurons all connect to only 8 of the 12 input planes, with an unspecified pattern
        I am going to assume the most sensible block pattern where 4 planes at a time connect
        to differently overlapping groups of 8/12 input planes. We will implement this with 3
        separate convolutions that we concatenate the results of.
        """
        self.H2w = nn.Parameter(winit(5*5*8, 12, 8, 5, 5))
        self.H2b = nn.Parameter(torch.zeros(12, 4, 4)) # presumably init to zero for biases
        assert self.H2w.nelement() + self.H2b.nelement() == 2592
        macs += (5*5*8) * (4*4) * 12
        acts += (4*4) * 12

        # H3 is a fully connected layer
        self.H3w = nn.Parameter(winit(4*4*12, 4*4*12, 30))
        self.H3b = nn.Parameter(torch.zeros(30))
        assert self.H3w.nelement() + self.H3b.nelement() == 5790
        macs += (4*4*12) * 30
        acts += 30

        # output layer is also fully connected layer
        self.outw = nn.Parameter(winit(30, 30, 10))
        self.outb = nn.Parameter(-torch.ones(10)) # 9/10 targets are -1, so makes sense to init slightly towards it
        assert self.outw.nelement() + self.outb.nelement() == 310
        macs += 30 * 10
        acts += 10

        self.macs = macs
        self.acts = acts

    def forward(self, x):
        # x has shape (1, 1, 16, 16)
        x = F.pad(x, (2, 2, 2, 2), 'constant', -1.0) # pad by two using constant -1 for background
        x = F.conv2d(x, self.H1w, stride=2) + self.H1b
        x = torch.tanh(x)

        # x is now shape (1, 12, 8, 8)
        x = F.pad(x, (2, 2, 2, 2), 'constant', -1.0) # pad by two using constant -1 for background
        slice1 = F.conv2d(x[:, 0:8], self.H2w[0:4], stride=2) # first 4 planes look at first 8 input planes
        slice2 = F.conv2d(x[:, 4:12], self.H2w[4:8], stride=2) # next 4 planes look at last 8 input planes
        slice3 = F.conv2d(torch.cat((x[:, 0:4], x[:, 8:12]), dim=1), self.H2w[8:12], stride=2) # last 4 planes are cross
        x = torch.cat((slice1, slice2, slice3), dim=1) + self.H2b
        x = torch.tanh(x)

        # x is now shape (1, 12, 4, 4)
        x = x.flatten(start_dim=1) # (1, 12*4*4)
        x = x @ self.H3w + self.H3b
        x = torch.tanh(x)

        # x is now shape (1, 30)
        x = x @ self.outw + self.outb
        x = torch.tanh(x)

        # x is finally shape (1, 10)
        return x

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Train a 1989 LeCun ConvNet on digits")
    parser.add_argument('--learning-rate', '-l', type=float, default=0.03, help="SGD learning rate")
    parser.add_argument('--output-dir'   , '-o', type=str,   default='out/base', help="output directory for training logs")
    parser.add_argument('--compare-at-epoch', '-c', type=int, default=-1, help="epoch after which to compare to the approximate C repro")
    parser.add_argument('--init-separately', '-i', type=bool, default=False, help="init separately (don't init to the same weights as our C repro")
    parser.add_argument('--enable-convolution-biases', '-e', type=bool, default=False, help="learn biases during the convolution layers")
    parser.add_argument('--file-to-compare', '-f', type=str, help="file with inputs/outputs, based on which to compare our results")
    args = parser.parse_args()
    print(vars(args))

    # contains test inputs and outputs as variables, defined in Python syntax
    with open(args.file_to_compare, 'r') as f:
        test_output = f.read()
        exec(test_output)
    W_c1 = torch.tensor(W_c1)
    W_c2 = torch.tensor(W_c2)
    W_l1 = torch.tensor(W_l1)
    W_l2 = torch.tensor(W_l2)
    b1 = torch.tensor(b1)
    b2 = torch.tensor(b2)
    newW_c1 = torch.tensor(newW_c1)
    newW_c2 = torch.tensor(newW_c2)
    newW_l1 = torch.tensor(newW_l1)
    newW_l2 = torch.tensor(newW_l2)
    newb1 = torch.tensor(newb1)
    newb2 = torch.tensor(newb2)

    # init rng
    torch.manual_seed(1337)
    np.random.seed(1337)
    torch.use_deterministic_algorithms(True)

    # set up logging
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    writer = SummaryWriter(args.output_dir)

    # init a model
    model = ApproxNet(W_c1, W_c2, W_l1, W_l2, b1, b2, args.enable_convolution_biases, args.init_separately)
    print("model stats:")
    print("# params:      ", sum(p.numel() for p in model.parameters())) # in paper total is 9,760

    # init data
    Xtr, Ytr = torch.load('train1989.pt')
    Xte, Yte = torch.load('test1989.pt')

    # init optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    def eval_split(split):
        # eval the full train/test set, batched implementation for efficiency
        model.eval()
        X, Y = (Xtr, Ytr) if split == 'train' else (Xte, Yte)
        Yhat = model(X)
        loss = torch.mean((Y - Yhat)**2)
        err = torch.mean((Y.argmax(dim=1) != Yhat.argmax(dim=1)).float())
        print(f"eval: split {split:5s}. loss {loss.item():e}. error {err.item()*100:.2f}%. misses: {int(err.item()*Y.size(0))}")
        writer.add_scalar(f'error/{split}', err.item()*100, pass_num)
        writer.add_scalar(f'loss/{split}', loss.item(), pass_num)

    # train
    for pass_num in range(23):

        # perform one epoch of training
        model.train()
        for step_num in range(Xtr.size(0)):

            # fetch a single example into a batch of 1
            x, y = Xtr[[step_num]], Ytr[[step_num]]
            # forward the model and the loss
            yhat = model(x)
            loss = torch.mean((y - yhat)**2)

            # calculate the gradient and update the parameters
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        # after epoch epoch evaluate the train and test error / metrics
        print(pass_num + 1)
        if pass_num + 1 == args.compare_at_epoch:
            assert torch.allclose(model.W_c1, newW_c1, rtol=1e-3, atol=1e-3)
            assert torch.allclose(model.W_c2, newW_c2, rtol=1e-3, atol=1e-3)
            assert torch.allclose(model.W_l1, newW_l1, rtol=1e-3, atol=1e-3)
            assert torch.allclose(model.W_l2, newW_l2, rtol=1e-3, atol=1e-3)
            assert torch.allclose(model.b1, newb1, rtol=1e-3, atol=1e-3)
            assert torch.allclose(model.b2, newb2, rtol=1e-3, atol=1e-3)
            print("Passed comparison test!")
        eval_split('train')
        eval_split('test')

    # save final model to file
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pt'))
