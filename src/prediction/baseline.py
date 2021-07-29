## Imports/
import sys, os, re, time
import pdb
import time
# Science
import numpy as np

class baseline:
    '''
        Baseline class
    '''    
    
    def __init__(self, pred_statistic, weights=None):
        '''
            Input:
                baseline predictive statistic: numpy statistics function, e.g.
                    mean()
                    median()
                    average()
                weights: For average-based baselines, weights to compute the average with
            Output:
                None
        '''        
        self.pred_statistic = pred_statistic
        # If statistic is average
        if self.pred_statistic == np.average:
            self.weights=weights
    
    # Predict using the (learned) model
    def predict(self, x):
        '''
            Input:
                x: input data, numpy array of size I times C_input
            Output:
                y_hat: predicted values, numpy array of size x.shape[0] times 1
        '''    
        # If statistic is average
        if self.pred_statistic == np.average:
            # Use weights if needed
            if np.all(self.weights != None):
                assert self.weights.shape[0] == x.shape[1]
                y_hat=np.average(x, axis=1, weights=self.weights)[:,None] # np.average does not handle keepdims, so we add extra dim
            else:
                y_hat=np.average(x, axis=1)[:,None] # np.average does not handle keepdims, so we add extra dim
        else:
            # Just use predictive statistic
            y_hat=self.pred_statistic(x, axis=1, keepdims=True)
            
        # Return prediction  
        return y_hat
    

