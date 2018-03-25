# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 10:37:17 2018

@author: Administrator
"""

from __future__ import division #精确除法

from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
import seaborn as sns
from theano import tensor as tt
import pandas as pd

K=10#DPM混合数目
N=1000#生成对象个数
n=11#每个对象重复观测次数
e=np.ones(11)
I=np.eye(11)

mu0=np.linspace(0,0,num=n)

#生成数据
C1=np.zeros((n,n))
for i in range(0,n):
    for j in range(0,n):
        if i==j:
            C1[i,j]=10
        else:
            C1[i,j]=7

dataSet1=np.random.multivariate_normal(mu0,C1,size=N)

#定义观测时间（各对象观测时间相同）
time_obseved=[-5.,-4.,-3.,-2.,-1.,0.,1.,2.,3.,4.,5.]
time=np.array(time_obseved)
#print(time[1])

'''
#注意：此处定义的H_p(rho)，rho是标量，与下文rho采样K个的H_p(rho)不一致
def H_p(rho):
   
    H_p=np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            if j==i:
                H_p[i,j]=1.
            else:
                H_p[i,j]=np.power(rho,abs(time[j]-time[i]))
    return H_p
'''
#定义K个协方差矩阵 H(ρ_k),其中k=1,2,...,K
#rho=np.ones(K)
def H(rho):
    H=np.zeros((K,n,n))
    for k in range(0,K):
        #H[k]=np.zeros((11,11))
        for i in range(0,n):
            for j in range(0,n):
                if j==i:
                    H[k][i,j]=1.0
                else:
                    H[k][i,j]=np.power(rho[k],np.abs(time[j]-time[i]))        
    return H          


#定义stick_breaking过程
def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])

    return beta * portion_remaining



with pm.Model() as model:
    M=pm.Gamma('M',1.,1.)
    
    sigma=pm.Uniform('sigma',0.,1.)
    mu1=pm.Normal('mu',0.,1.)
    xi=pm.InverseGamma('xi',1.,1.)
    b=pm.Normal('b',0.,xi,shape=N)
    
    sigma_w=pm.Uniform('sigma_w',0.,1.,shape=K)
    rho=pm.Uniform('rho',0.,1.,shape=K)
    
    beta=pm.Beta('beta',1.,M,shape=K)
    w=pm.Deterministic('w',stick_breaking(beta))
    
    
    omega=pm.Mixture('omega',w,pm.MvNormal.dist(mu=mu0,cov=sigma_w[:,np.newaxis,np.newaxis]**2*H(rho)))
    
    obs=pm.MvNormal('obs',mu=(mu1+b)*e+omega,cov=sigma**2*I,observed=dataSet1)
    

    
    