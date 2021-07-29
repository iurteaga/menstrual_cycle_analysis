import numpy as np
import scipy.stats as stats
from scipy.special import gammaln, xlogy, logsumexp
import pdb

class generalized_poisson():
    '''
        See Consul, P. C., & Famoye, F. (2006)
        Lagrangian probability distributions, chapter 9.

    '''
    def __init__(self, my_lambda, xi, x_max=1000):
        '''
            Input:
                my_lambda: lambda parameters of the Generalized Poisson
                xi: xi parameters of the Generalized Poisson
                x_max: maximum support value to consider for the Generalized Poisson
                    For underdispersed Generalized Poisson, x_max will be set according to the given parameters
        '''
        # Recast in array form
        my_lambda=np.atleast_1d(my_lambda)
        xi=np.atleast_1d(xi)
        x_max=np.atleast_1d(x_max)
        # Save only if reasonable
        assert my_lambda.shape == xi.shape, ' Lambda and xi shape must be equal'
        assert np.all(my_lambda > 0.0), 'Lambda parameter must be positive'
        self.my_lambda=my_lambda
        assert np.all(xi >= -1.0) and np.all(xi <= 1.0), 'Xi parameter must be within [-1,1]'
        self.xi=xi
        self.x_max=x_max*np.ones(self.my_lambda.shape)
        if np.any(self.xi<0):
            #Having an x_max less than the corresponding lambda/xi ratio is nonsense
            #assert np.all(self.x_max[xi<0] <= np.ceil(-self.my_lambda[xi<0]/self.xi[xi<0] - 1).astype(int)), 'Specified x_max does not match lambda/xi ratio'
            if np.any(self.x_max[xi<0] <= np.ceil(-self.my_lambda[xi<0]/self.xi[xi<0]-1).astype(int)):
                print('Specified x_max {} does not match lambda/xi ratio {}'.format(
                    self.x_max[xi<0],
                    np.ceil(-self.my_lambda[xi<0]/self.xi[xi<0] - 1).astype(int)
                    ))
            # x_max will be set according to the given parameters
            self.x_max[self.xi<0]=np.ceil(-self.my_lambda[self.xi<0]/self.xi[self.xi<0] - 1).astype(int)
        
    def get_support(self):
        return np.zeros(self.my_lambda.shape), self.x_max+1
    def get_max_support(self):
        return 0, (self.x_max.max()+1).astype(int)

    def mean(self):
        return self.my_lambda/(1-self.xi)

    def var(self):
        return self.my_lambda/np.power(1-self.xi,3)
                    
    def logpmf(self):
        x=np.arange(*self.get_max_support())[(None,)*len(self.my_lambda.shape)]
        lambda_xi_x = self.my_lambda[...,None] + self.xi[...,None]*x
        log_pmf=( np.log(self.my_lambda[...,None]) 
                    + xlogy(x-1,lambda_xi_x) 
                    - lambda_xi_x
                    - gammaln(x+1) # Gamma(x+1)=x!
                )
        # Those values beyond x_max will have nans
        # that we replace with -np.inf
        log_pmf[np.isnan(log_pmf)]=-np.inf
        # Normalized
        return log_pmf-logsumexp(log_pmf, axis=-1, keepdims=True)
        
    def loglik(self,x):
        full_x=np.arange(*self.get_max_support())[(None,)*len(self.my_lambda.shape)]
        assert np.all(x>=full_x.min()) and np.all(x<=full_x.max())
        
        lambda_xi_x = self.my_lambda[...,None] + self.xi[...,None]*full_x
        log_pmf=( np.log(self.my_lambda[...,None]) 
                    + xlogy(full_x-1,lambda_xi_x) 
                    - lambda_xi_x 
                    - gammaln(full_x+1) # Gamma(x+1)=x!
                )
        # Normalized
        return log_pmf[x]-logsumexp(log_pmf, axis=-1, keepdims=True)

    def pmf(self):
        return np.exp(self.logpmf())
    
    def pmf_for_val(self,x):
        lambda_xi_x = self.my_lambda[...,None] + self.xi[...,None]*x
        log_pmf=( np.log(self.my_lambda[...,None]) 
                    + xlogy(x-1,lambda_xi_x) 
                    - lambda_xi_x
                    - gammaln(x+1) # Gamma(x+1)=x!
                )
        # Those values beyond x_max will have nans
        # that we replace with -np.inf
        log_pmf[np.isnan(log_pmf)]=-np.inf
        # Normalized
        return np.exp(log_pmf-logsumexp(log_pmf, axis=-1, keepdims=True))

    def cdf(self):
        return np.exp(self.logpmf()).cumsum(axis=-1)

    def rvs(self, sample_size):
        '''
            Draw samples using the inverse cdf method
        '''
        x=np.arange(*self.get_max_support())[(None,)*len(self.my_lambda.shape)]
        this_cdf=self.cdf()
        u=stats.uniform.rvs(size=self.my_lambda.shape+(1,sample_size,))
        return np.sum(this_cdf[...,None]<=u, axis=-2)

