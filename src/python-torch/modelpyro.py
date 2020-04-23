import os
import sys
import argparse

import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd

import torch.utils.data
import torch.optim as optim
import xlsxwriter

import pyro
from pyro import poutine
import pyro.optim as optim
from pyro.distributions import Bernoulli, Normal
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoGuideList, AutoDiscreteParallel
from pyro.infer import SVI, TraceGraph_ELBO

torch.set_default_tensor_type('torch.FloatTensor')
pyro.enable_validation(True)
pyro.clear_param_store()
# pyro.util.set_rng_seed(26011994)



class PyroDEF(object):
    def __init__(self, nSNPs, nGenes):
        # define the sizes of the layers in the deep exponential family
        self.top_width = 1
        self.bottom_width = nGenes
        self.data_size = nSNPs
        # define hyperparameters that control the prior
        self.p_z = torch.tensor(np.exp(np.random.uniform(-np.log(nSNPs,-np.log(1)))))
        self.mu_w = torch.tensor(0.0)
        self.sigma_w = torch.tensor(0.0)
        # define parameters used to initialize variational parameters
        self.z_mean_init = 0.0
        self.z_sigma_init = 0.1
        self.w_mean_init = 0.0
        self.w_sigma_init = 0.1
        self.softplus = torch.nn.Softplus()

    # define the model
    def model(self, x):
        x_size = x.size(0)
        
        # sample the global weights
        with pyro.plate("w_top_plate", self.top_width * self.bottom_width):
            w_top = pyro.sample("w_top", Normal(self.mu_w, self.sigma_w))
        with pyro.plate("w_bottom_plate", self.bottom_width * self.data_size):
            w_bottom = pyro.sample("w_bottom", Normal(self.mu_w, self.sigma_w))

        
        # sample the local latent random variables
        # (the plate encodes the fact that the z's for different data points are conditionally independent)
        with pyro.plate("data", x_size):
            z_top = pyro.sample("z_top", Bernoulli(self.p_z).expand([self.top_width]).to_event(1))
            # note that we need to use matmul (batch matrix multiplication) as well as appropriate reshaping
            # to make sure our code is fully vectorized
            w_top = w_top.reshape(self.top_width, self.bottom_width) if w_top.dim() == 1 else \
                w_top.reshape(-1, self.top_width, self.bottom_width)
            mean_bottom = torch.softplus(torch.matmul(z_top, w_top))
            z_bottom = pyro.sample("z_bottom", Bernoulli(mean_bottom).to_event(1))

            w_bottom = w_bottom.reshape(self.bottom_width, self.data_size) if w_bottom.dim() == 1 else \
                w_bottom.reshape(-1, self.bottom_width, self.data_size)
            mean_obs = torch.softplus(torch.matmul(z_bottom, w_bottom))

            # observe the data using a Bernoulli likelihood
            pyro.sample('obs', Bernoulli(mean_obs).to_event(1), obs=x)

    # define our custom guide a.k.a. variational distribution.
    def guide(self, x):
        x_size = x.size(0)

        # helper for initializing variational parameters
        def rand_tensor(shape, mean, sigma):
            return mean * torch.ones(shape) + sigma * torch.randn(shape)
        
        # define a helper function to sample z's for a single layer
        def sample_zs(name, width):
            # Sample parameters
            p_z_q = pyro.param("p_z_q_%s" % name,
                               lambda: rand_tensor((x_size, width), self.z_mean_init, self.z_sigma_init))
            p_z_q = torch.softplus(p_z_q)
            # Sample Z's
            pyro.sample("z_%s" % name, Bernoulli(p_z_q).to_event(1))

        # define a helper function to sample w's for a single layer
        def sample_ws(name, width):
            # Sample parameters
            mean_w_q = pyro.param("mean_w_q_%s" % name,
                                  lambda: rand_tensor(width, self.w_mean_init, self.w_sigma_init))
            sigma_w_q = pyro.param("sigma_w_q_%s" % name,
                                   lambda: rand_tensor(width, self.w_mean_init, self.w_sigma_init))
            sigma_w_q = self.softplus(sigma_w_q)
            # Sample weights
            pyro.sample("w_%s" % name, Normal(mean_w_q, sigma_w_q))

        # sample the global weights
        with pyro.plate("w_top_plate", self.top_width * self.bottom_width):
            sample_ws("top", self.top_width * self.bottom_width)
        with pyro.plate("w_bottom_plate", self.bottom_width * self.data_size):
            sample_ws("bottom", self.bottom_width * self.data_size)

        # sample the local latent random variables
        with pyro.plate("data", x_size):
            sample_zs("top", self.top_width)
            sample_zs("bottom", self.bottom_width)


def main(args):
    dataset_path = Path(r"C:\Users\posc8001\Documents\DEF\Data\Simulation_1")
    file_to_open = dataset_path / "small_data.csv"
    f = open(file_to_open)
    data = torch.tensor(np.loadtxt(f, delimiter=',')).float()
    pyro_def = PyroDEF()

    # Specify hyperparameters of optimization
    learning_rate = 0.5
    momentum = 0.05
    opt = optim.AdagradRMSProp({"eta": learning_rate, "t": momentum})

    # Specify parameters of sampling process
    n_samp = 100000

    # Specify the guide
    guide = pyro_def.guide

    # Specify Stochastic Variational Inference
    svi = SVI(pyro_def.model, guide, opt, loss=TraceGraph_ELBO())

    # we use svi_eval during evaluation; since we took care to write down our model in
    # a fully vectorized way, this computation can be done efficiently with large tensor ops
    svi_eval = SVI(pyro_def.model, guide, opt,
                   loss=TraceGraph_ELBO(num_particles=args.eval_particles, vectorize_particles=True))

    # the training loop
    losses, final_w_bottom = [], []
    final_p_z_0 = []
    final_w_top = []
    final_sig_w_top = []
    sample = []
    final_sig_w_bottom = []

    for i in range(15):
        final_w_bottom.append([])

    for i in range(15):
        final_sig_w_bottom.append([])

    for i in range(2):
        final_p_z_0.append([])

    for i in range(6):
        final_w_top.append([])

    for i in range(6):
        final_sig_w_top.append([])

    for k in range(args.num_epochs):
        losses.append(svi.step(data))

        for i in range(2):
            final_p_z_0[i].append(torch.softplus(pyro.param("p_z_q_top")[:, i].mean()))

        for i in range(6):
            final_w_top[i].append(pyro.param("mean_w_q_top")[i].item())        
            
        for i in range(15):
            final_w_bottom[i].append(pyro.param("mean_w_q_bottom")[i].item())

        if k % args.eval_frequency == 0 and k > 0 or k == args.num_epochs - 1:
            loss = svi_eval.evaluate_loss(data)
            print("[epoch %04d] training elbo: %.4g" % (k, loss))

        # if k == args.num_epochs - 1:
        #     # Sample fake data set
        #     p_z_top_1 = torch.softplus(pyro.param("p_z_q_top")[:, 0].mean())
        #     p_z_top_2 = torch.softplus(pyro.param("p_z_q_top")[:, 1].mean())
        #
        #     w1_z_bottom_1 = pyro.param("mean_w_q_top")[0].item()
        #     w1_z_bottom_2 = pyro.param("mean_w_q_top")[1].item()
        #     w1_z_bottom_3 = pyro.param("mean_w_q_top")[2].item()
        #     w2_z_bottom_1 = pyro.param("mean_w_q_top")[3].item()
        #     w2_z_bottom_2 = pyro.param("mean_w_q_top")[4].item()
        #     w2_z_bottom_3 = pyro.param("mean_w_q_top")[5].item()
        #
        #     w1_x_1 = pyro.param("mean_w_q_bottom")[0].item()
        #     w1_x_2 = pyro.param("mean_w_q_bottom")[1].item()
        #     w1_x_3 = pyro.param("mean_w_q_bottom")[2].item()
        #     w1_x_4 = pyro.param("mean_w_q_bottom")[3].item()
        #     w1_x_5 = pyro.param("mean_w_q_bottom")[4].item()
        #     w2_x_1 = pyro.param("mean_w_q_bottom")[5].item()
        #     w2_x_2 = pyro.param("mean_w_q_bottom")[6].item()
        #     w2_x_3 = pyro.param("mean_w_q_bottom")[7].item()
        #     w2_x_4 = pyro.param("mean_w_q_bottom")[8].item()
        #     w2_x_5 = pyro.param("mean_w_q_bottom")[9].item()
        #     w3_x_1 = pyro.param("mean_w_q_bottom")[10].item()
        #     w3_x_2 = pyro.param("mean_w_q_bottom")[11].item()
        #     w3_x_3 = pyro.param("mean_w_q_bottom")[12].item()
        #     w3_x_4 = pyro.param("mean_w_q_bottom")[13].item()
        #     w3_x_5 = pyro.param("mean_w_q_bottom")[14].item()

    plt.plot(losses)
    plt.title("ELBO")
    plt.xlabel("step")
    plt.ylabel("loss");
    plt.show()

    for i in range(final_p_z_0.__len__()):
        plt.plot(final_p_z_0[i])
        plt.title("P Z_top_" + (i+1).__str__())
        plt.show()

    for i in range(final_w_top.__len__()):
        plt.plot(final_w_top[i])
        plt.title("Mean W_top_" + (i+1).__str__())
        plt.show()

    for i in range(final_w_bottom.__len__()):
        plt.plot(final_w_bottom[i])
        plt.title("Mean W_bottom_" + (i+1).__str__())
        plt.show()


if __name__ == '__main__':
    assert pyro.__version__.startswith('0.3.0')
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=10000, type=int, help='number of training epochs')
    parser.add_argument('-ef', '--eval-frequency', default=25, type=int,
                        help='how often to evaluate elbo (number of epochs)')
    parser.add_argument('-ep', '--eval-particles', default=200, type=int,
                        help='number of samples/particles to use during evaluation')
    parser.add_argument('--auto-guide', action='store_true', help='whether to use an automatically constructed guide')
    args = parser.parse_args()
    model = main(args)