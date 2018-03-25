# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 13:03:56 2018

@author: Administrator
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
#print(e[:,10])#第11个


#omega0=np.empty_like(e)#1,11
#print(omega0.shape)
'''
sigmaw=sp.stats.uniform.rvs(0,2)
rho=sp.stats.uniform.rvs(0,1)
omega[0,0]=sp.stats.norm.rvs(loc=0,scale=sigmaw,size=1)

for i in range(1,11):
    omega[i]=sp.stats.norm.rvs(loc=omega[i-1]*rho,scale=sigmaw*sp.sqrt(1-rho),size=1)
    
''' 

#定义f(w|σ_w,ρ)
def pdf(omega0,sigmaw0,rho0):
    
    omega0=np.empty_like(e)#1,11
    C=np.empty_like(e)#1,11
    
    sigmaw0=sp.stats.uniform.rvs(0,1,size=1)
    rho0=sp.stats.uniform.rvs(0,1,size=1)
    
    omega0[0,0]=sp.stats.norm.rvs(loc=0,scale=sigmaw0,size=1)
    for i in range(1,11):
        omega0[0,i]=sp.stats.norm.rvs(loc=omega0[0,i-1]*rho0,scale=sigmaw0*sp.sqrt(1-rho0),size=1)
    
    C[0,0]=sp.stats.norm.pdf(omega0[0,0],loc=0,scale=sigmaw0)
    for i in range(1,11):
        C[0,i]=sp.stats.norm.pdf(omega0[0,i],loc=omega0[0,i-1]*rho0,scale=sigmaw0*sp.sqrt(1-rho0))
        
    return C.cumprod(axis=1,dtype=np.float64)[:,-1]#按行连乘，取最后一列,输出是科学计数法，如何转为小数
    #print(C[1:].cumprod(axis=0).shape)


#模拟产生N=200个(w),即omega样本
mu0=np.linspace(0,0,num=11)
var=np.eye(11)
omega=sp.stats.multivariate_normal.rvs(mean=mu0,cov=var,size=200)#里面是11个数的一行列表
#omega=np.matrix(omega)#numpy数组
#print(omega)
#print(omega.shape)#200,11


'''
#给定一组σ_w,ρ的情况下，绘制f(w|σ_w,ρ)
sigmaw=sp.stats.uniform.rvs(0,2,size=1)
rho=sp.stats.uniform.rvs(0,1,size=1)

omegaPdf=np.zeros((200,1))#omegaPdf用于储存产生的200个密度函数值

for i in range(0,200):
    omegaPdf[i]=pdf(omega[i],sigmaw,rho)
print(omegaPdf.shape)#200,1的数组

#画omega（w）的密度函数图
fig,ax=plt.subplots(figsize=(8,6))
ax.plot(omegaPdf,c='green')
ax.set_yticklabels([])
'''

#执行DPM
M=2

K=30#混合类数

'''
#折棍子模型，定义w
def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])

    return beta * portion_remaining

beta=sp.stats.beta.rvs(1,M,size=(1,K))

print(beta.shape)#1,30
w=np.empty_like(beta)

w=stick_breaking(beta)
print(w)
'''
beta = sp.stats.beta.rvs(1, M, size=(1, K))
w = np.empty_like(beta)
w[:, 0] = beta[:, 0]
w[:, 1:] = beta[:, 1:] * (1 - beta[:, :-1]).cumprod(axis=1)
#print(beta.shape)#(1,30)
#print(w.shape)#(1,30)
#print(beta)#小数点有9个
#print(w)

#采样σ_w,ρ
sigmaw=sp.stats.uniform.rvs(0,1,size=(1,K))
rho=sp.stats.uniform.rvs(0,1,size=(1,K))
#print(sigmaw)#9位小数点

#计算混合密度函数f(w|G)

#pdf_j=np.zeros((1,30))#存储f(w_i|σ_wj,ρ_j),在j=1,2,...,30的30个密度函数值
#pdf_i=np.zeros((200,1))#存储200个f(w_i|G),i=1,2,...,200个密度值
pdf_components=np.zeros((30,200))
for j in range(0,30):
    for i in range(0,200):
        pdf_components[j,i]=pdf(omega[i],sigmaw[0,j],rho[0,j])
    
   

     
'''        
for i in range(0,200):
    for j in range(0,30):
        pdf_components[i,j]=pdf(omega[i],sigmaw[0,j],rho[0,j])
     
    pdf_i[i]=(w * pdf_j).sum(axis=1)
    #pdf_i[i]= ("%.2f" % pdf_i[i])
        
print(pdf_i.shape)

for j in range(0,30):
    pdf_j[0,j]=pdf(omega[1],sigmaw[0,j],rho[0,j])
print(pdf_j)
'''
    
'''
#绘制f(w_i|G),i=1,2,...,200
x=np.linspace(-5,5,num=200)
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x,pdf_i, c='gray');
ax.set_yticklabels([]);
'''

def pdfm(omega,sigmaw,rho,w):
    pdf_j=np.zeros((1,30))#存储f(w_i|σ_wj,ρ_j),在j=1,2,...,30的30个密度函数值
    pdf_i=np.zeros((200,1))#存储200个f(w_i|G),i=1,2,...,200个密度值

    for i in range(0,200):
        for j in range(0,30):
            pdf_j[0,j]=pdf(omega[i],sigmaw[0,j],rho[0,j])
        
        pdf_i[i]=(w * pdf_j).sum(axis=1)
    return pdf_i

    



    










