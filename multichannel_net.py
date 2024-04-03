import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MultipathwayNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, nonlinearity=None, bias=False, num_pathways=2, depth=2, width=1000, eps=0.01, hidden=None):

        super(MultipathwayNet, self).__init__()

        # hidden is assumed to be a list with entries corresponding to each pathway, each entry a list of the widths of that pathway by depth
        # hidden!=None will override num_pathways, depth, width


        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.nonlinearity = nonlinearity

        # used for deep copu function
        self.depth = depth
        self.width = width
        self.eps = eps

        hidden_size = output_dim
        projection_size = output_dim
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim,hidden_size ),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, projection_size)
        )

        if hidden is None:
            hidden = []
            for pi in range(num_pathways):
                pathway = []
                for di in range(depth-1):
                    pathway.append(width)
                hidden.append(pathway)
        # hidden first layer is path way, each element is a list of length depth and each entry in this list is width

        self.hidden = hidden

        self.hidden_layers = []



        # iterate through each pathway
        for pathway in self.hidden:

            # the num indicator ensure that we only create one pathway so that we can
            # use the mcn class as the encoder


            op_list = []

            # the depth here is the number of neurons (width)
            for di, depth in enumerate(pathway):
                if di==0:
                    # depth here is the dimension of output vecotr
                    op = torch.nn.Linear(self.input_dim, depth, bias=self.bias)

                    # if concat modify here


                else:
                    # input dim is the width of last layer
                    op = torch.nn.Linear(pathway[di-1], depth, bias=self.bias)

                    #  initializes its weight with random values scaled by eps and sets it to require gradients for training
                op.weight = torch.nn.Parameter(torch.randn(op.weight.shape)*eps, requires_grad=True)

                # If the linear layer has a bias (i.e., op.bias is not None), it initializes the bias with zeros and sets it to require gradients for training.
                if op.bias is not None:
                    op.bias = torch.nn.Parameter(torch.zeros_like(op.bias), requires_grad=True)
                op_list.append(op)


            #  creates a final linear layer that connects the last hidden layer to the output layer.
            # since the depth in hidden is the nunmber of hidden layers, not including the last layer
            op = torch.nn.Linear(pathway[-1], self.output_dim, bias=self.bias)
            op.weight = torch.nn.Parameter(torch.randn(op.weight.shape)*eps, requires_grad=True)



            if op.bias is not None:
                op.bias = torch.nn.Parameter(torch.zeros_like(op.bias), requires_grad=True)
            op_list.append(op)


            self.hidden_layers.append(op_list)

            # do a deep copy
            # else:
            #     op_list = []
            #     for first_layer in first_pathway_layers:
            #         new_layer = torch.nn.Linear(first_layer.in_features,
            #                                     first_layer.out_features,
            #                                     bias=self.bias)
            #         new_layer.weight = torch.nn.Parameter(
            #             first_layer.weight.clone().detach(),
            #             requires_grad=True)
            #         if self.bias:
            #             new_layer.bias = torch.nn.Parameter(
            #                 first_layer.bias.clone().detach(),
            #                 requires_grad=True)
            #         op_list.append(new_layer)
            #         # Add the new pathway with the deep copied layers
            #     self.hidden_layers.append(op_list)


        # store name for each paramenter. useful in model saving and loading, accessing and updating specific parameters during training
        for pi, op_list in enumerate(self.hidden_layers):
            for oi, op in enumerate(op_list):
                self.register_parameter(name= "Path_{}_Depth_{}_weight".format(pi, oi), param=op.weight)
                self.register_parameter(name= "Path_{}_Depth_{}_bias".format(pi, oi), param=op.bias)

        # inserts the activation function immediately after each linear layer in the pathway.
        if self.nonlinearity is not None:
            temp_layers = self.hidden_layers
            self.hidden_layers = []
            for op_list in temp_layers:
                new_op_list = []
                for op in op_list:
                    new_op_list.append(op)
                    new_op_list.append(self.nonlinearity)
                self.hidden_layers.append(new_op_list)


    # first forward a single pathway, then forward another pathway, and add the result together
    def forward(self, x):

        # adding 0 to a tensor means adding 0 on every element
        output = 0


        for op_list in self.hidden_layers:
            xtemp = x

            # op is a linear layer created by porch
            for op in op_list:
                xtemp = op(xtemp)

            # here output got added in each pathway

            output += xtemp



        # output = self.projection_head(output)
        return output

    # a deep copy function ( without copying the hidden layer)
    def copy(self):


        new_net = MultipathwayNet(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            nonlinearity=self.nonlinearity,
            bias=self.bias,
            num_pathways=len(self.hidden),
            depth=self.depth,
            # Assuming depth is the number of layers
            width=self.width,
            # Assuming width is the size of the first layer
            eps=self.eps,
            hidden=self.hidden.copy()
        )
        for pi, pathway in enumerate(self.hidden_layers):
            for di, layer in enumerate(pathway):
                # Copy the weights
                with torch.no_grad():
                    new_net.hidden_layers[pi][di].weight = torch.nn.Parameter(
                        layer.weight.clone())

                    # If the layer has a bias, copy it
                    if layer.bias is not None:
                        new_net.hidden_layers[pi][
                            di].bias = torch.nn.Parameter(layer.bias.clone())

        with torch.no_grad():
            for layer, new_layer in zip(self.projection_head,
                                        new_net.projection_head):
                if hasattr(layer, 'weight'):
                    new_layer.weight = torch.nn.Parameter(layer.weight.clone())
                if hasattr(layer, 'bias') and layer.bias is not None:
                    new_layer.bias = torch.nn.Parameter(layer.bias.clone())

        return new_net



    #  omega is the sum of all weight matrics? Y = OmegaX

    def omega(self):
        # no gradients are calculated during the execution of the code inside the block
        with torch.no_grad():
            # creates an identity matrix x with dimensions determined by self.input_dim

            x = torch.eye(self.input_dim).to(self.hidden_layers[0][0].weight.device)
            output = []

            # pass identity matrix to each layer for this pathway,
            # assume each layers input and output dimens are fixed?
            # iterate through pathway
            for op_list in self.hidden_layers:
                xtemp = x
                for op in op_list:
                    # this return the weight matrix?
                    xtemp = op(xtemp)

                output.append(xtemp.T.detach())


        return output
