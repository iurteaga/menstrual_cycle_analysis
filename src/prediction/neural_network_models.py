## Imports/
import sys, os, re, time
import timeit
import argparse
import pdb
import pickle
from itertools import *
# Science
import numpy as np
import pandas as pd
import scipy.stats as stats
# Plotting
import matplotlib.pyplot as plt
from matplotlib import colors
# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# Convolution based (deep) neural network
class conv_nnet(nn.Module):
    '''
        Class inheriting from nn.Module
    '''
    # Init module
    def __init__(self, input_size, output_size, n_layers, kernel_size, stride=1, padding=0, dilation=1, nonlinearity='Tanh', dropout=0, config_file=None):
        '''
            Input:
                input_size: size of the input x (only 1 feature/channel)
                output_size: size of the input x (only 1 feature/channel)
                n_layers: number of recurrent layers (i.e., number of stacked layers)
                kernel_size: size of the convolving kernel per layer
                stride: stride of the convolution per layer. Default: 1
                padding: Zero-padding added to both sides of the input per-layer. Default: 0
                dilation: spacing between kernel elements per layer. Default: 1
                nonlinearity: whether to use 'Tanh' or 'ReLU' nonlinearity
                dropout: If non-zero, introduces a Dropout layer on the inputs of each conv layer, with dropout probability equal to dropout
                config_file: (optional) file used for class configuration
        '''
        super(conv_nnet, self).__init__()
        # Keep config file name
        self.config_file=config_file
        
        # Model layers
        # Dropout if necessary
        if dropout>0:
            self.drop_layer = nn.Dropout(dropout)
        
        # Definition of each convolutional layer
        self.conv_layer = nn.Sequential(
                # Convolution kernel: one input/output channel
                nn.Conv1d(1, 1, kernel_size, stride, padding, dilation),
                # Nonlinearity of output
                getattr(nn, nonlinearity)(),
            )
        
        # Figure out convolution layer input/output sizes per layer
        if (input_size+2*padding) > dilation*(kernel_size-1):
            # There will be at least 1 conv-layer
            conv_input_size=np.zeros(n_layers+1)
            conv_output_size=np.zeros(n_layers)
            # Init
            n_layer=0
            conv_input_size[n_layer]=input_size
            conv_output_size[n_layer]=input_size
            while (n_layer < n_layers) and ( (conv_input_size[n_layer]+2*padding) > dilation*(kernel_size-1)):
                conv_output_size[n_layer]=np.floor((conv_input_size[n_layer]+2*padding-dilation*(kernel_size-1)-1)/stride + 1).astype(int)            
                # update
                conv_input_size[n_layer+1]=conv_output_size[n_layer]
                n_layer+=1
            
            # Number of layers might be less than desired
            self.n_layers=n_layer
            assert np.all(conv_output_size[:self.n_layers]>0)
            fully_connected_input_size=conv_output_size[self.n_layers-1].astype(int)
        else:
            # No convolution is possible
            self.n_layers=0
            # Just fully connected
            fully_connected_input_size=input_size
        
        # Fully connected layer: affine operation y = Wx + b
        self.fc_affine = nn.Linear(fully_connected_input_size, output_size)
    
    # Forward pass
    def forward(self, x):
        '''
            Input:
                x: input data
            Output:
                y: output data
        '''
        
        # Iterate over layers
        for n_layer in np.arange(self.n_layers):
            # Dropout if necessary
            if hasattr(self, 'drop_layer'):
                self.drop_layer(x)
            # Convolutional layer
            x=self.conv_layer(x)
        
        # Final fully connected_layer
        y = self.fc_affine(x)
        
        # Return output
        return y
    
    # Train the model
    def train(self, x, y, optimizer, criterion, n_epochs=100, batch_size=None, loss_epsilon=0.0001, grad_norm_max=0):
        '''
            Input:
                x: input data
                y: true output data
                optimizer: to be used
                criterion: loss function
                n_epochs: to train for
                loss_epsilon: minimum relative loss diference to consider as converged training
                grad_norm_max: whether we are clipping gradient norms, very useful for RNN type models
            Output:
                None: the model will be trained after executing this function
        '''
        # Make sure x is tensor
        if not torch.is_tensor(x):
            # pytorch nn modules need an extra dimension (channel)
            x=torch.from_numpy(x[:,None,:]).float()
            y=torch.from_numpy(y[:,None,:]).float()
        
        if batch_size is None:
            # Use full dataset
            batch_size = int(x.shape[0])
        dataset=DataLoader(TensorDataset(x,y), batch_size=int(batch_size), shuffle=True)
        
        # Training Run
        debug_epoch=np.floor(n_epochs/100).astype(int)
        # used for debugging and plotting, otherwise unnecessary
        self.training_loss_values = []    # loss function values
        self.n_epochs_conv = 0 # number of epochs
        self.time_elapsed = 0 # time elapsed 
        
        # Epoch variables
        epoch=0
        prev_loss=0
        this_loss=np.inf

        # Initiate fit-time counter
        start_time = timeit.default_timer()

        # Iterate
        while (epoch < n_epochs) and (abs(this_loss - prev_loss) >= loss_epsilon*abs(prev_loss)):
            # Keep track of losses over batches
            batch_count=0
            batches_loss=0
            # Mini-batching
            for x_batch,y_batch in dataset:
                # Forward pass of model
                y_hat = self.forward(x_batch)
                # With loss
                loss = criterion(y_hat, y_batch)
        
                # Keep track of this batch and its loss
                batch_count+=1
                batches_loss+=loss.item()
                    
                # Backpropagation and calculate gradients
                optimizer.zero_grad() # clear existing gradients from previous epoch
                loss.backward() # perform a backward pass
                if grad_norm_max>0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_norm_max)
                optimizer.step() # update the weights
            
            # Debugging
            if epoch%debug_epoch == 0:
                print('\t\tEpoch {}/{} with per-batch average loss={}'.format(epoch, n_epochs, batches_loss/batch_count))
            # Keep track of iterations
            epoch+=1
            prev_loss=this_loss
            this_loss=batches_loss
            
            # Keep losses
            self.training_loss_values.append(this_loss)
        
        # Number of epochs and fit-time
        self.n_epochs_conv = epoch
        self.time_elapsed = timeit.default_timer() - start_time

        print('\tModel trained after {} epochs with loss={}'.format(epoch, loss.item()))
    
    # Predict using the (learned) model
    def predict(self, x):
        '''
            Input:
                x: data to use for prediction
            Output:
                prediction: the model's predicted output
        '''
        # Make sure x is tensor
        x_is_numpy=False
        if not torch.is_tensor(x):
            x_is_numpy=True
            # pytorch nn modules need an extra dimension (channel)
            x=torch.from_numpy(x[:,None,:]).float()
            
        with torch.no_grad():
            # Forward pass
            # Remove extra dimension (channel)
            prediction=self(x).detach()[:,0,:]
            
        return prediction.numpy() if x_is_numpy else prediction

# RNN based (deep) neural network
class rnn_nnet(nn.Module):
    '''
        Class inheriting from nn.Module
    '''
    # Init module
    def __init__(self, rnn_type, input_size, output_size, hidden_size, n_layers, nonlinearity='tanh', dropout=0, bidirectional=False, config_file=None):
        '''
            Input:
                rnn_type: type of Recurrent neural network: RNN, LSTM, GRU
                input_size: number of features in the input x
                output_size: number of features in the output y
                hidden_size: number of features in the hidden state h
                n_layers: number of recurrent layers (i.e., number of stacked layers)
                nonlinearity: whether to use 'tanh' or 'relu' nonlinearity
                dropout – If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout
                bidirectional: whether to have bidirectional networks
                config_file: (optional) file used for class configuration
        '''
        
        super(rnn_nnet, self).__init__()
        # Keep config file name
        self.config_file=config_file
        
        # RNN type
        self.rnn_type=rnn_type
        # Keep all parameters
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.num_directions = 2 if bidirectional else 1
        
        # Model layers
        # RNN Layer
        if self.rnn_type in ['LSTM', 'GRU']:
            # Do not need nonlinearity
            self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, n_layers, dropout=dropout, bidirectional=bidirectional)
        else:
            # Define nonlinearity of RNN
            self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, n_layers, nonlinearity=nonlinearity.lower(), dropout=dropout, bidirectional=bidirectional)
        
        # The output is provided by a fully connected layer of the RNN's last hidden output
        # Last output size is (num_layers * num_directions, batch, hidden_size),
        # which we will serialize to num_layers * num_directions * hidden_size
        # Fully connected layer: affine operation y = Wx + b        
        self.fc_affine = nn.Linear(self.n_layers*self.num_directions*self.hidden_size, output_size)
    
    # Forward pass
    def forward(self, x):
        '''
            Input:
                x: input data
            Output:
                y: output data
        '''
        # x is expected to be of size (seq_len, batch, input_size)       
        x=x.permute(2,0,1)
        
        # Initializing hidden states for first input
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        rnn_h = torch.zeros(self.n_layers*self.num_directions, x.size(1), self.hidden_size)
        if self.rnn_type in ['LSTM']:
            # In LSTMs, c_0 of shape (num_layers * num_directions, batch, hidden_size)
            rnn_c = torch.zeros(self.n_layers*self.num_directions, x.size(1), self.hidden_size)
        
        # RNN
        if self.rnn_type in ['RNN', 'GRU']:
            # Only hidden state needed
            h, rnn_h = self.rnn(x, rnn_h)
        elif self.rnn_type in ['LSTM']:
            # LSTM make use of hidden and cell states
            h, (rnn_h, rnn_c) = self.rnn(x, (rnn_h, rnn_c))

        # Here, we simply take the last value:
        # It can be either h[-1,:,:] or rnn_h
        # Permute to have N times 1 times  (num_layers * num_directions * hidden_size)
        rnn_last=rnn_h.permute(1,0,2).contiguous().view(-1,1,self.n_layers*self.num_directions*self.hidden_size)
        y = self.fc_affine(rnn_last)
        
        # Return output
        return y

    # Train the model
    def train(self, x, y, optimizer, criterion, n_epochs=100, batch_size=None, loss_epsilon=0.0001, grad_norm_max=0):
        '''
            Input:
                x: input data
                y: true output data
                optimizer: to be used
                criterion: loss function
                n_epochs: to train for
                loss_epsilon: minimum relative loss diference to consider as converged training
                grad_norm_max: whether we are clipping gradient norms, very useful for RNN type models
            Output:
                None: the model will be trained after executing this function
        '''
        
        # Make sure x is tensor
        if not torch.is_tensor(x):
            # pytorch nn modules need an extra dimension (channel)
            x=torch.from_numpy(x[:,None,:]).float()
            y=torch.from_numpy(y[:,None,:]).float()
        
        if batch_size is None:
            # Use full dataset
            batch_size = int(x.shape[0])
        dataset=DataLoader(TensorDataset(x,y), batch_size=int(batch_size), shuffle=True)
            
        # Training Run
        debug_epoch=np.floor(n_epochs/100).astype(int)
        # used for debugging and plotting, otherwise unnecessary
        self.training_loss_values = []    # loss function values
        self.n_epochs_conv = 0 # number of epochs
        self.time_elapsed = 0 # time elapsed 

        # Epoch variables
        epoch=0
        prev_loss=0
        this_loss=np.inf

        # Initiate fit-time counter
        start_time = timeit.default_timer()

        # Iterate
        while (epoch < n_epochs) and (abs(this_loss - prev_loss) >= loss_epsilon*abs(prev_loss)):
            # Keep track of losses over batches
            batch_count=0
            batches_loss=0
            # Mini-batching
            for x_batch,y_batch in dataset:
                # Forward pass of model
                y_hat = self.forward(x_batch)
                # With loss
                loss = criterion(y_hat, y_batch)
        
                # Keep track of this batch and its loss
                batch_count+=1
                batches_loss+=loss.item()
                    
                # Backpropagation and calculate gradients
                optimizer.zero_grad() # clear existing gradients from previous epoch
                loss.backward() # perform a backward pass
                if grad_norm_max>0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_norm_max)
                optimizer.step() # update the weights
        
            # Debugging
            if epoch%debug_epoch == 0:
                print('\t\tEpoch {}/{} with per-batch average loss={}'.format(epoch, n_epochs, batches_loss/batch_count))
            # Keep track of iterations
            epoch+=1
            prev_loss=this_loss
            this_loss=batches_loss
        
            # Keep losses
            self.training_loss_values.append(this_loss)

        # Number of epochs and fit-time
        self.n_epochs_conv = epoch
        self.time_elapsed = timeit.default_timer() - start_time
            
        print('\tModel trained after {} epochs with loss={}'.format(epoch, loss.item()))
        
    # Predict using the (learned) model
    def predict(self, x):
        '''
            Input:
                x: data to use for prediction
            Output:
                prediction: the model's predicted output
        '''
        # Make sure x is tensor
        x_is_numpy=False
        if not torch.is_tensor(x):
            x_is_numpy=True
            # pytorch nn modules need an extra dimension (channel)
            x=torch.from_numpy(x[:,None,:]).float()
            
        with torch.no_grad():
            # Forward pass
            # Remove extra dimension (channel)
            prediction=self(x).detach()[:,0,:]
            
        return prediction.numpy() if x_is_numpy else prediction
    

