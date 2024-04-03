import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import NMF

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from multichannel_net import MultipathwayNet


def plot_kakb(ka, kb):
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ka = ka.T.detach().numpy()
    kb = kb.T.detach().numpy()
    # Plot the first heatmap
    heatmap1 = axes[0].imshow(ka, cmap='hot', interpolation='nearest')
    axes[0].set_title('Ka')  # Title for the first heatmap
    fig.colorbar(heatmap1, ax=axes[0])  # Add a colorbar for the first heatmap

    # Plot the second heatmap
    heatmap2 = axes[1].imshow(kb, cmap='hot', interpolation='nearest')
    axes[1].set_title('Kb')  # Title for the second heatmap
    fig.colorbar(heatmap2, ax=axes[1])  # Add a colorbar for the second heatmap

    # Show the plot
    plt.show()


# X is a 8x8 identity matrix
X_default = torch.eye(8)
Y_default = torch.Tensor([[1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 1, 1, 1],

                          [1, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 1],

                          [1, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1],
                          ]).T

Y_alt = torch.Tensor([[0, 0, 0, 0, 0, 1, 1, 1],
                      [0, 1, 1, 1, 0, 1, 0, 1],
                      [0, 0, 1, 0, 1, 0, 1, 0],
                      [0, 1, 1, 1, 0, 1, 1, 0],
                      [0, 1, 0, 0, 1, 0, 1, 1],
                      [1, 1, 1, 0, 1, 0, 0, 1],
                      [0, 1, 1, 1, 0, 0, 0, 1],
                      [0, 0, 1, 1, 1, 1, 1, 0],
                      [1, 1, 1, 0, 0, 0, 1, 0],
                      [1, 1, 0, 0, 0, 1, 1, 1],
                      [1, 1, 1, 0, 0, 1, 0, 0],
                      [0, 1, 1, 1, 1, 1, 1, 0],
                      [0, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 1],
                      [0, 0, 1, 1, 1, 0, 0, 1]]).T


def random_noise_addition(x, mean=0.01, std=0.003):
    noise = torch.randn_like(x) * std + mean
    noisy_x = x + noise
    return noisy_x


class MPNAnalysis(object):
    def __init__(self, mcn, X=X_default, Y=Y_default, device=None):

        self.mcn = mcn
        self.encoder1 = mcn.copy()
        self.encoder2 = mcn.copy()

        self.X = X
        self.Y = Y

        # self.scl_optimizer = optim.Adam(self.encoder.parameters(), lr=1e-3)

        if device is None:
            if torch.backends.mps.is_built():
                self.device = "mps"
            elif torch.has_cuda:
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print("Using device {}".format(self.device))

        # self.mcn.to(self.device)
        # self.X = self.X.to(self.device)
        # self.Y = self.Y.to(self.device)

        # matrix multiplication of the transpose of self.Y with self.X,
        # divided by the number of rows in self.Y.

        # todo uncomment this when doing the old training
        sigma_yx = self.Y.T.mm(self.X) / self.Y.shape[0]

        U, S, V = torch.svd(sigma_yx, some=False)

        self.U = U
        self.S = S
        self.V = V

        self.loss_history = None
        self.omega_history = None
        self.K_history = None

    # return k = U*omega*V
    def omega2K(self, omega):

        #  make certain operations more efficient by skipping gradient computations
        with torch.no_grad():
            k = omega.mm(self.V)
            k = self.U.T.mm(k)

        return k

    def nt_xent_loss(self, z_i, z_j, temperature=0.5):
        # Normalize the embeddings
        z_i_norm = F.normalize(z_i, p=2, dim=1)
        z_j_norm = F.normalize(z_j, p=2, dim=1)

        # Concatenate the normalized embeddings
        z = torch.cat((z_i_norm, z_j_norm), dim=0)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / temperature
        sim_matrix.fill_diagonal_(
            -9e15)  # Fill diagonal with very small numbers to exclude self-similarity

        # Create labels for 2N examples, where each example's label is its positive pair index

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        labels = torch.arange(z.size(0), device=device)
        labels = (labels + z.size(0) // 2) % z.size(0)

        # Compute NT-Xent loss (i.e., normalized temperature-scaled cross entropy loss)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(sim_matrix, labels)

        return loss

    # oringinal train function
    # def train_mcn(self, timesteps=1000, lr=0.01):
    #
    #     # squared error loss function
    #     loss = torch.nn.MSELoss(reduction='sum')
    #
    #     # stochastic gradient descent (SGD) optimizer (torch.optim.SGD)
    #     # to update the parameters of the neural network (self.mcn) during training.
    #     # The learning rate (lr) is set to the provided value.
    #     optimizer = torch.optim.SGD(params=self.mcn.parameters() , lr=lr)
    #
    #     # just set the model to train mode and do nothign else,
    #     # mcn is the model
    #     self.mcn.train()
    #
    #     loss_history = []
    #     omega_history = []
    #
    #     for t in range(10):
    #         #  model make predictions (output) based on input X.
    #         output = self.mcn(self.X)
    #
    #
    #         loss_val = loss(output, self.Y)
    #
    #         loss_history.append(loss_val.to("cpu").detach().numpy())
    #
    #         omega_history.append(self.mcn.omega())
    #
    #         # computes the gradients of the loss with respect to all the model's parameters.
    #         loss_val.backward()
    #
    #         # use sgd to update the model's parameters (weights and biases) based on these gradients.
    #         optimizer.step()

    #         optimizer.zero_grad()
    #
    #
    #     omega_history = zip(*omega_history)
    #
    #
    #
    #     # convert omegas to Ks
    #     K_history = []
    #     for pathway in omega_history:
    #         # om is the layer in pathway
    #         K_history.append([self.omega2K(om) for om in pathway])
    #
    #
    #     # print(len(K_history))
    #
    #     exit()
    #     self.loss_history = loss_history
    #     self.omega_history = omega_history
    #     self.K_history = K_history
    #
    #     return loss_history, K_history

    # simclr train function
    ############################################################
    # def train_mcn(self, timesteps=1000, lr=0.01):
    #
    #     loss_history = []
    #     omega_history = []
    #
    #     self.encoder1.train()
    #     self.encoder2.train()
    #
    #     optimizer1 = torch.optim.SGD(params=self.encoder1.parameters(), lr=lr)
    #     optimizer2 = torch.optim.SGD(params=self.encoder2.parameters(), lr=lr)
    #
    #
    #
    #     for t in range(300):
    #
    #
    #
    #         # Apply augmentations to create a pair
    #         batch_augmented_1 = random_noise_addition(self.X)
    #         batch_augmented_2 = random_noise_addition(self.X)
    #
    #
    #         # Compute representations
    #         z_i = self.encoder1(batch_augmented_1) + self.encoder2(batch_augmented_1)
    #         z_j = self.encoder1(batch_augmented_2) + self.encoder2(batch_augmented_2)
    #
    #
    #
    #         # Compute contrastive loss
    #         loss_val = self.nt_xent_loss(z_i, z_j)
    #
    #         loss_history.append(loss_val.to("cpu").detach().numpy())
    #
    #
    #         # this is to make the omega list same format as the orignial one, so that
    #         # it can work with the plot functions later
    #         omega = self.encoder1.omega()
    #
    #         for matrix in self.encoder2.omega():
    #             omega.append(matrix)
    #
    #
    #         # omega_history.append(omega)
    #         omega_history.append([self.encoder1(self.X), self.encoder2(self.X)])
    #
    #         loss_val.backward()
    #         optimizer1.step()
    #         optimizer2.step()
    #
    #         optimizer1.zero_grad()
    #         optimizer2.zero_grad()
    #
    #
    #
    #
    #
    #     history = zip(*omega_history)
    #
    #
    #     # sig
    #     print(self.encoder2.omega())
    #     exit()
    #
    #     mat1 = self.encoder1(self.X)
    #     mat2 = self.encoder2(self.X)
    #     ouptut_matrix = mat1+mat2
    #     #ouptut_matrix = ouptut_matrix.T
    #
    #     # U, S, V = torch.svd(mat1, some=False)
    #     U, S, V = torch.svd(ouptut_matrix, some=False)
    #
    #     self.U = U
    #     self.S = S
    #     self.V = V
    #
    #     ka = self.omega2K(mat1)
    #     kb = self.omega2K(mat2)
    #
    #
    #
    #
    #
    #
    #     K_history = []
    #
    #     for pathway in history:
    #         # om is the layer in pathway
    #
    #         K_history.append([self.omega2K(om).T for om in pathway])
    #
    #     self.loss_history = loss_history
    #     self.omega_history = omega_history
    #     self.K_history = K_history
    #
    #     return loss_history, K_history

    #######################################################
    # two sigma:

    def train_mcn(self, timesteps=1000, lr=0.01):

        sigma_yx1 = self.Y.T.mm(self.X) / self.Y.shape[0]
        num_features = sigma_yx1.shape[1]  # Number of features in X
        num_labels = sigma_yx1.shape[0]  # Number of labels in Y
        num_samples = 8

        # generate X and y using non-negative
        torch.manual_seed(0)

        sigma_yx_new = torch.rand((15, 8))
        sigma_yx_new *= self.Y.shape[0]
        n_components = 8  # For example, let's factorize sigma_yx_new into 4 components

        model = NMF(n_components=n_components, init='random', random_state=0,
                    max_iter=500)
        W = model.fit_transform(sigma_yx_new)
        H = model.components_

        Y_new = torch.tensor(W.T, dtype=torch.float)
        X_new = torch.tensor(H.T, dtype=torch.float)

        X = torch.cat([self.X, X_new],
                      dim=0)  # Concatenate along the first axis (rows)
        Y = torch.cat([self.Y, Y_new], dim=0)

        dataset = TensorDataset(X, Y)

        # Step 3: Create a DataLoader
        data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

        # squared error loss function
        loss = torch.nn.MSELoss(reduction='sum')

        # stochastic gradient descent (SGD) optimizer (torch.optim.SGD)
        # to update the parameters of the neural network (self.mcn) during training.
        # The learning rate (lr) is set to the provided value.
        optimizer = torch.optim.SGD(params=self.mcn.parameters(), lr=lr)

        # just set the model to train mode and do nothign else,
        # mcn is the model
        self.mcn.train()

        loss_history = []
        omega_history = []

        for t in range(timesteps):
            #  model make predictions (output) based on input X.
            output = self.mcn(X_new)

            loss_val = loss(output, Y_new)

            loss_history.append(loss_val.to("cpu").detach().numpy())

            omega_history.append(self.mcn.omega())

            # computes the gradients of the loss with respect to all the model's parameters.
            loss_val.backward()

            # use sgd to update the model's parameters (weights and biases) based on these gradients.
            optimizer.step()

            optimizer.zero_grad()

        omega_history = zip(*omega_history)

        sigma_yx_new = Y_new.T.mm(X_new) / Y_new.shape[0]
        U, S, V = torch.svd(sigma_yx_new, some=False)
        self.U = U
        self.V = V
        #
        # sigma =  Y.T.mm(X) / Y.shape[0]
        # U, S, V = torch.svd(sigma, some=False)
        # self.U = U
        # self.V = V

        # convert omegas to Ks
        K_history = []
        for pathway in omega_history:
            # om is the layer in pathway
            K_history.append([self.omega2K(om) for om in pathway])

        # print(len(K_history))

        self.loss_history = loss_history
        self.omega_history = omega_history
        self.K_history = K_history

        return loss_history, K_history

    # all the code below are from the old code
    def plot_K(self, ax, savedir='', labels=None, savename=None, savelabel='',
               min_val=0, max_val=2):

        if self.K_history is None:
            raise Exception(
                "MultipathwayNet must be trained before visualization.")

        num_K = len(self.K_history)

        K_list = [pathway[-1].to("cpu") for pathway in self.K_history]

        # min_val = np.min([torch.min(K) for K in K_list])
        # max_val = np.max([torch.max(K) for K in K_list])

        if labels is None:
            labels = [i for i in range(len(K_list))]

        for i, K in enumerate(K_list):
            im = ax[i].imshow(K, vmin=min_val, vmax=max_val,
                              cmap='magma')  # 'inferno'
            ax[i].set_title(r'$\bf K_{}$'.format(labels[i]), fontsize=20)
            ax[i].axis('off')

        plt.colorbar(im, ax=ax, shrink=1)

    def plot_K_history(self, ax, savename=None, D='unknown', savelabel=''):

        if self.K_history is None:
            raise Exception(
                "MultipathwayNet must be trained before visualization.")

        num_pathways = len(self.K_history)
        timesteps = len(self.K_history[0])

        for i in range(min(self.mcn.input_dim, self.mcn.output_dim)):

            z1 = np.array([K[i, i].to("cpu") for K in self.K_history[0]])
            z2 = np.array([K[i, i].to("cpu") for K in self.K_history[1]])

            x = np.ones(timesteps) * i
            y = np.arange(timesteps)
            if i == 0:
                ax.plot3D(x, y, z1, 'C0', linewidth=4, label=r'$K_{a\alpha}$')
                line = \
                ax.plot3D(x, y, z2, 'C1', linewidth=4, label=r'$K_{b\alpha}$')[
                    0]
                line.set_dashes([1, 1, 1, 1])
            ax.plot3D(x, y, z1, 'C0', linewidth=4)
            line = ax.plot3D(x, y, z2, 'C1', linewidth=4)[0]
            line.set_dashes([2, 1, 2, 1])
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='z', labelsize=10)
        ax.set_xlabel(r'dimension $\alpha$', fontsize=15)
        ax.set_ylabel('epoch', fontsize=15)
        ax.set_zlabel(r'$K_{a,b\alpha}$', fontsize=15)
        ax.legend(fontsize=17, loc='upper left')

        ax.set_box_aspect((2.25, 1.75, 1))
        return


if __name__ == '__main__':

    import argparse

    torch.manual_seed(345345)

    plt.rc('font', size=20)
    plt.rcParams['figure.constrained_layout.use'] = True
    import matplotlib

    matplotlib.rcParams["mathtext.fontset"] = 'cm'

    parser = argparse.ArgumentParser()

    # parser.add_argument('--timesteps', type=int, default=10000)
    # parser.add_argument('--nonlinearity', type=str, default='relu')

    args = parser.parse_args()

    nonlin = None
    # if args.nonlinearity=='relu':
    #     nonlin = torch.nn.ReLU()
    # if args.nonlinearity=='tanh':
    #     nonlin = torch.nn.Tanh()

    # timesteps = args.timesteps

    depth_list = [7, 2]

    fig_train, ax_train = plt.subplots(1, len(depth_list), figsize=(25, 8))
    ax_train[0].set_ylabel('training error')

    fig_history = plt.figure(figsize=(24, 10))
    gs = gridspec.GridSpec(2, 6, width_ratios=[2.2, 1, 1, 2.2, 1, 1],
                           figure=fig_history)

    timestep_list = [1000, 1000, 1400, 10000]

    min_val = 0.0
    max_val = 0.00001

    for di, depth in enumerate(depth_list):
        ax3d = fig_history.add_subplot(gs[di * 3], projection='3d')
        ax2 = fig_history.add_subplot(gs[di * 3 + 1])
        ax3 = fig_history.add_subplot(gs[di * 3 + 2])

        mcn = MultipathwayNet(8, 15, depth=depth, num_pathways=2, width=1000,
                              bias=False, nonlinearity=nonlin)
        mpna = MPNAnalysis(mcn, Y=Y_default)
        mpna.train_mcn(timesteps=timestep_list[di], lr=0.01)

        ax_train[di].plot(mpna.loss_history)
        ax_train[di].set_xlabel('epoch')
        ax_train[di].set_title("$D={}$".format(depth))

        ax3d.set_title("$D=2$")
        mpna.plot_K_history(ax3d, D=depth)

        K_list = [pathway[-1].to("cpu") for pathway in mpna.K_history]
        min_val_temp = np.min([torch.min(K) for K in K_list])
        max_val_temp = np.max([torch.max(K) for K in K_list])

        min_val = min(min_val_temp, min_val)
        max_val = max(max_val_temp, max_val)

    for di, depth in enumerate(depth_list):
        mpna.plot_K([ax2, ax3], labels=['a', 'b'], min_val=min_val,
                    max_val=max_val)

    fig_train.suptitle("Training loss")
    fig_train.savefig('train_loss.pdf')
    fig_history.savefig('test.pdf')

    plt.show()

