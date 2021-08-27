import scipy.stats as ss
import numpy as np
from math import sqrt, pi,log, e
from enum import Enum
import scipy.stats as stat
from scipy.stats import norm
from pcm_tools.toolbox import fred_1r_ir_today
from scipy.optimize import newton
import time

def bsm_call(S0, K, sigma, t, div):
    T = t/365
    r = fred_1r_ir_today()
    d1 = (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))
    d2 = (np.log(S0/K) + (r - sigma**2 / 2) * T)/(sigma * np.sqrt(T))
    price = S0 * np.exp(-div * T) * ss.norm.cdf(d1) - K * np.exp(-r * T) * ss.norm.cdf(d2)
    return price

def bsm_put(S0, K, sigma, t, div):
    T = t/365
    r = fred_1r_ir_today()
    d1 = (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))
    d2 = (np.log(S0/K) + (r - sigma**2 / 2) * T)/(sigma * np.sqrt(T))
    price = K * np.exp(-r * T) * ss.norm.cdf(-d2) - S0 * np.exp(-div * T) * ss.norm.cdf(-d1)
    return price

# Black and Scholes call pricing and greeks
def bsm_pricing_call(S0, K, sigma, t, div):
    T = t/365
    r = fred_1r_ir_today()
    d1 = (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))
    d2 = (np.log(S0/K) + (r - sigma**2 / 2) * T)/(sigma * np.sqrt(T))
    
    price = S0 * np.exp(-div * T) * ss.norm.cdf(d1) - K * np.exp(-r * T) * ss.norm.cdf(d2)

    # Calculate call delta
    call_delta = np.exp(-div * T) * ss.norm.cdf(d1)
    # Calculate call gamma
    call_gamma = np.exp(-div * T)/(S0*sigma*np.sqrt(T))*(1/np.sqrt(2*np.pi))*np.exp(-(d1**2)/2)
    # Calculate put gamma
    put_gamma = call_gamma
    # Calculate call theta:
    call_theta = (1/365)*(-S0*sigma*np.exp(-div*T)/(2*np.sqrt(T))*(1/np.sqrt(2*np.pi))*np.exp(-(d1**2)/2) - r*K*np.exp(-r*T)*ss.norm.cdf(d2) + div*S0*np.exp(-div*T)*ss.norm.cdf(d1))                    
    # Calculating vega
    vega = (1/100)*(S0*np.exp(-div*T)*np.sqrt(T))*(1/np.sqrt(2*np.pi))*np.exp(-(d1**2)/2)
    # Calculating call Rho
    rho = (1/100)*K*T*np.exp(-r*T)*ss.norm.cdf(d2)

    
    print ("S0\tstock price at time 0:", S0)
    print ("K\tstrike price:", K)
    print ("r\tcontinuously compounded risk-free rate:", r)
    print ("sigma\timplied volatility of the stock price per year:", sigma)
    print ("T\ttime to maturity in trading years: {:0.04F}".format(T))
    print ("Call_price: {:0.04F}".format(price))
    print ("Delta: {:0.04F}".format(call_delta))
    print ("Gamma: {:0.04F}".format(call_gamma))
    print ("Theta: {:0.04F}".format(call_theta))
    print ("Vega: {:0.04F}".format(vega))
    print ("Rho: {:0.04F}".format(rho))
    
    
# Black and Scholes put pricing and greeks
def bsm_pricing_put(S0, K, sigma, t, div):
    T = t/365
    r = fred_1r_ir_today()
    d1 = (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))
    d2 = (np.log(S0/K) + (r - sigma**2 / 2) * T)/(sigma * np.sqrt(T))
    
    price = K * np.exp(-r * T) * ss.norm.cdf(-d2) - S0 * np.exp(-div * T) * ss.norm.cdf(-d1)
    
    # Calculate put delta
    put_delta = np.exp(-div * T) * (ss.norm.cdf(d1)-1)
    # Calculate put gamma
    put_gamma = np.exp(-div * T)/(S0*sigma*np.sqrt(T))*(1/np.sqrt(2*np.pi))*np.exp(-(d1**2)/2)          
    # Calculate put theta:
    put_theta = (1/365)*(-S0*sigma*np.exp(-div*T)/(2*np.sqrt(T))*(1/np.sqrt(2*np.pi))*np.exp(-(d1**2)/2) + r*K*np.exp(-r*T)*ss.norm.cdf(-d2) - div*S0*np.exp(-div*T)*ss.norm.cdf(d1))
    # Calculate vega:
    vega = (1/100)*(S0*np.exp(-div*T)*np.sqrt(T))*(1/np.sqrt(2*np.pi))*np.exp(-(d1**2)/2)
    # Calculate put rho:
    rho = -(1/100)*K*T*np.exp(-r*T)*ss.norm.cdf(-d2)
   
    
    print ("S0\tstock price at time 0:", S0)
    print ("K\tstrike price:", K)
    print ("r\tcontinuously compounded risk-free rate:", r)
    print ("sigma\timplied volatility of the stock price per year:", sigma)
    print ("T\ttime to maturity in trading years: {:0.04F}".format(T))
    print ("Put price: {:0.04F}".format(price))
    print ("Delta: {:0.04F}".format(put_delta))
    print ("Gamma: {:0.04F}".format(put_gamma))
    print ("Theta: {:0.04F}".format(put_theta))
    print ("Vega: {:0.04F}".format(vega))
    print ("Rho: {:0.04F}".format(rho))






def bsm_pricing_call_return(S0, K, sigma, t, div):
    T = t/365
    r = fred_1r_ir_today()
    d1 = (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))
    d2 = (np.log(S0/K) + (r - sigma**2 / 2) * T)/(sigma * np.sqrt(T))
    
    price = S0 * np.exp(-div * T) * ss.norm.cdf(d1) - K * np.exp(-r * T) * ss.norm.cdf(d2)

    # Calculate call delta
    call_delta = np.exp(-div * T) * ss.norm.cdf(d1)
    # Calculate call gamma
    call_gamma = np.exp(-div * T)/(S0*sigma*np.sqrt(T))*(1/np.sqrt(2*np.pi))*np.exp(-(d1**2)/2)
    # Calculate put gamma
    put_gamma = call_gamma
    # Calculate call theta:
    call_theta = (1/365)*(-S0*sigma*np.exp(-div*T)/(2*np.sqrt(T))*(1/np.sqrt(2*np.pi))*np.exp(-(d1**2)/2) - r*K*np.exp(-r*T)*ss.norm.cdf(d2) + div*S0*np.exp(-div*T)*ss.norm.cdf(d1))                    
    # Calculating vega
    vega = (1/100)*(S0*np.exp(-div*T)*np.sqrt(T))*(1/np.sqrt(2*np.pi))*np.exp(-(d1**2)/2)
    # Calculating call Rho
    rho = (1/100)*K*T*np.exp(-r*T)*ss.norm.cdf(d2)

    return price, call_delta, call_gamma, call_theta, vega, rho

    
    
    
# Black and Scholes put pricing and greeks
def bsm_pricing_put_return(S0, K, sigma, t, div):
    T = t/365
    r = fred_1r_ir_today()
    d1 = (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))
    d2 = (np.log(S0/K) + (r - sigma**2 / 2) * T)/(sigma * np.sqrt(T))
    
    price = K * np.exp(-r * T) * ss.norm.cdf(-d2) - S0 * np.exp(-div * T) * ss.norm.cdf(-d1)
    
    # Calculate put delta
    put_delta = np.exp(-div * T) * (ss.norm.cdf(d1)-1)
    # Calculate put gamma
    put_gamma = np.exp(-div * T)/(S0*sigma*np.sqrt(T))*(1/np.sqrt(2*np.pi))*np.exp(-(d1**2)/2)          
    # Calculate put theta:
    put_theta = (1/365)*(-S0*sigma*np.exp(-div*T)/(2*np.sqrt(T))*(1/np.sqrt(2*np.pi))*np.exp(-(d1**2)/2) + r*K*np.exp(-r*T)*ss.norm.cdf(-d2) - div*S0*np.exp(-div*T)*ss.norm.cdf(d1))
    # Calculate vega:
    vega = (1/100)*(S0*np.exp(-div*T)*np.sqrt(T))*(1/np.sqrt(2*np.pi))*np.exp(-(d1**2)/2)
    # Calculate put rho:
    rho = -(1/100)*K*T*np.exp(-r*T)*ss.norm.cdf(-d2)

    return price, put_delta, put_gamma, put_theta, vega, rho
    
    



def break_even(delta, gamma, total_theta):
    def func(x):
        return delta*x + 0.5*gamma*x**2 + total_theta
    return func
    
    
class BSMerton:
    def __init__(self, args):
        self.Type = int(args[0])                # 1 for a Call, - 1 for a put
        self.S = float(args[1])                 # Underlying asset price
        self.K = float(args[2])                 # Option strike K
        self.r = float(args[3])                 # Continuous risk fee rate
        self.q = float(args[4])                 # Dividend continuous rate
        self.T = float(args[5]) / 365.0         # Compute time to expiry
        self.sigma = float(args[6])             # Underlying volatility
        self.sigmaT = self.sigma * self.T ** 0.5# sigma*T for reusability
        self.d1 = (log(self.S / self.K) + 
                   (self.r - self.q + 0.5 * (self.sigma ** 2)) 
                   * self.T) / self.sigmaT
        self.d2 = self.d1 - self.sigmaT
        [self.Delta] = self.delta()
 
    def delta(self):
        dfq = e ** (-self.q * self.T)
        if self.Type == 1:
            return [dfq * norm.cdf(self.d1)]
        else:
            return [dfq * (norm.cdf(self.d1) - 1)]