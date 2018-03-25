# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 09:17:36 2018

@author: Cabbage
w_i--来自高斯过程（mean=0）的n_i维变量（y_i观测n_i次），此处n_i=11，且
w_i=,...=w_n(有n个个体)

f(w_i|φ_i=(σ_wi,ρi))是w_i在φ_i下的条件密度，其中φ_i=(σ_wi,ρi)|G ~ G,G~DP(M,G_0)
此处，M=2，G_0=U(0,c)×U(0,1),由于w_i=,...=w_n,故记φ_i=(σ_wi,ρi)=φ=(σ_w,ρ)
f(w_i|G)是w_i的DPM密度函数。

简化模型：
y_i=(μ+b_i)*e+w_i+ε_i
等价于分层模型：
y_i|μ,b_i,w_i,σ^2 ～N((μ+b_i)*e+w_i,σ^2I) → ni维
w_i|φ_i=(σ_wi^2,ρ_i) ～ N(0,σ_wi^2H(ρ_i)) → ni维
φ_1,φ_2,...,φ_n ～ G
              G ～ DP(G_0,M)
              b_i|ζ ～ N(0,ζ)
              σ ～ U(0，A)
              μ ～ N(μ_0,B)
              ζ ～ IG(b1,b2)

"""

from __future__ import division 

from  matplotlib  import pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
import seaborn as sns
from theano import tensor as tt
import pandas as pd

e=np.ones((1,11))

N=200
K=30

e2=np.ones((1,1))
#定义f(w|σ_w,ρ),w来自均值为0的高斯过程
def pdf(omega0,sigmaw0,rho0):
    
    omega0=np.empty_like(e)#1,11,放在外面还是定义里面
    C=np.empty_like(e)#1,11
    
    #sigmaw0=np.empty_like(e2)
    #rho0=np.empty_like(e2)
    sigmaw0=sp.stats.uniform.rvs(0,1,size=(1))#里面？
    rho0=sp.stats.uniform.rvs(0,1,size=(1))#外面？
    
    omega0[0,0]=sp.stats.norm.rvs(loc=0,scale=sigmaw0,size=(1))
    for i in range(1,11):
        omega0[0,i]=sp.stats.norm.rvs(loc=omega0[0,i-1]*rho0,scale=sigmaw0*sp.sqrt(1-rho0),size=(1))
    
    C[0,0]=sp.stats.norm.pdf(omega0[0,0],loc=0,scale=sigmaw0)
    for i in range(1,11):
        C[0,i]=sp.stats.norm.pdf(omega0[0,i],loc=omega0[0,i-1]*rho0,scale=sigmaw0*sp.sqrt(1-rho0))
        
    return C.cumprod(axis=1)[:,-1]#按行连乘，取最后一列,输出是科学计数法，如何转为小数

'''
#计算混合密度函数f(w|G)
def pdfm(omega1,sigmaw1,rho1,w1):
    omega1=np.empty((200,11))
    sigmaw1=np.empty((1,K))
    rho1=np.empty((1,K))
    w1=np.empty((1,K))
    
    pdf_j=np.zeros((1,30))#存储f(w_i|σ_wj,ρ_j),在j=1,2,...,30的30个密度函数值
    pdf_i=np.zeros((200,1))#存储200个f(w_i|G),i=1,2,...,200个密度值

    for i in range(0,200):
        for j in range(0,30):
            pdf_j[0,j]=pdf(omega1[i],sigmaw1[0,j],rho1[0,j])
        
        pdf_i[i]=(w1 * pdf_j).sum(axis=1)
    return pdf_i
'''

#另一种计算混合密度函数f(w|G)的方法
'''
pdf_components=np.zeros((K,N))
for j in range(0,K):
    for i in range(0,N):
        pdf_components[j,i]=pdf(omega[i],sigmaw[0,j],rho[0,j])
#print(pdf_components.shape)
pdfm=(w[...,np.newaxis] * pdf_components).sum(axis=1)
#print(pdfm.shape)#1,200
'''
E1=np.ones((N,11))
E2=np.ones((1,K))
def pdfm(omega1,sigmaw1,rho1,w1):
    omega1=np.empty_like(E1)
    sigmaw1=np.empty_like(E2)
    rho1=np.empty_like(E2)
    w1=np.empty_like(E2)
    pdf_components=np.zeros((K,N))#第k行存储omega1_1、...omega1_200分别在σ_wk,ρ_k下的条件密度值
    for j in range(0,K):
        for i in range(0,N):
            pdf_components[j,i]=pdf(omega1[i],sigmaw1[0,j],rho1[0,j])
    pdfm_=(w1[...,np.newaxis] * pdf_components).sum(axis=1)
    return pdfm_

#模拟产生N=200个(w),即omega样本
mu0=np.linspace(0,0,num=11)
var=np.eye(11)
omega=sp.stats.multivariate_normal.rvs(mean=mu0,cov=var,size=N)#里面是11个数的一行列表


M=2
#K=30#混合类数

#定义w,DPM混合的权重
beta = sp.stats.beta.rvs(1, M, size=(1, K))
w = np.empty_like(beta)
w[:, 0] = beta[:, 0]
w[:, 1:] = beta[:, 1:] * (1 - beta[:, :-1]).cumprod(axis=1)


#采样σ_w,ρ
sigmaw=sp.stats.uniform.rvs(0,1,size=(1,K))
rho=sp.stats.uniform.rvs(0,1,size=(1,K))

#调运pdfm
pdfm(omega,sigmaw,rho,w)
#print(pdfm(omega,sigmaw,rho,w))
#print(pdfm(omega,sigmaw,rho,w).shape)#1,200,2维数组

#转换成列表
dataset0=pdfm(omega,sigmaw,rho,w)
dataset1=dataset0.tolist


#如何画图（200个DPM密度函数值的图）
'''
x_plot=np.linspace(-10,10,num=200)
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x_plot, dataset1, c='gray');

ax.set_yticklabels([]);
'''

    


    







