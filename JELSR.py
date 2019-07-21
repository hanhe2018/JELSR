
# coding: utf-8

# In[1]:


#  JELSR feature selection
# reference :"Joint Embedding Learning and Sparse Regression: A Framework for Unsupervised Feature Selection"(2014)
# by  Chenping Hou, Feiping Nie, Xuelong Li, Dongyun Yi and Yi Wu

import numpy as np
import pandas as pd
from scipy.optimize import minimize, NonlinearConstraint
from itertools import repeat


# First define a r,p norm function. ||Q||_{r,p} defined in equation (9) in the paper. 

def LRPNorm(w,r=2,p=2):
    """
    define L_{r,p} norm function to compute the norm with general r, p
    -----
    input
    -----
    w:{numpy array}, shape(n_rows,n_col) 
       input matrix 

    r:{int}
       parameter r in l_{r,p} norm

    p:{int}
       parameter p in l_{r,p} norm
   
    ------
    output
    ------
    resNorm:{float}
        l_{r,p} norm

    """         
    Sum = 0
    row,col = w.shape
    for i in range(row):
        temp = 0
        for j in range(col):
            temp += np.power(np.absolute(w[i,j]),r) 
        temp = np.power(temp,p/r)
        Sum += temp
        resNorm = np.power(Sum, 1/p)
    return resNorm

# Notations: alpha , beta are balanced parameter. 
#            k: neighborhood size
#            m: dimensionality of embedding
#            s: selected feature number
#            n: original data size
#            d: dimensionality of original data

def KNNgraphG(X,k):
    
    """
    Define K Nearest Neighbors Graph of matrix X.  
    -----
    input
    -----
    X: {numpy array},shape(n_features,n_samples)
        input data

    k: {int}
        neighborhood size

    ------
    output
    ------
    GraphG:{dictionary}
       return the index list of k nearest neighbors of each sample observation.

    """    
    d, n = X.shape 
    if k > d:
        k = d
    Distance = np.array(list(repeat(0.0000,n**2))).reshape(n,n)
    for i in range(n):
        for j in range(n):
            Distance[i,j]= np.linalg.norm(X[:,i]-X[:,j])
    GraphG = {}
    for i in range(n):
        GraphG[i] =list(np.argsort(Distance[i,])[0:k+1])
        if i in GraphG[i]:
            GraphG[i].remove(i)
        GraphG[i] = sorted(GraphG[i])    
    return GraphG


def SimilarityM(X,k):
    
    """
    Define similarity matrix S.
    -----
    input
    -----
    X: {numpy array},shape(n_features,n_samples)
        input data

    k: {int}
        neighborhood size
    ------
    output
    ------
    S: {numpy array},shape(n_samples,n_samples)
       similarity matrix
    """    
        
    import statsmodels.api as sm
    d,n = X.shape
    GraphG = KNNgraphG(X,k)

    S = np.array(list(repeat(0.0000,n**2))).reshape(n,n)
    
    for i in range(n):
        idx = GraphG[i]
        regX = X[:,idx]
        regy = X[:,i]
        model = sm.OLS(regy,regX)
        result = model.fit()
        t = 0
        for j in idx:
            templist =  list(result.params/np.sum(result.params))
            templist = [ round(a,4) for a in templist ]
            S[i,j] = templist[t]
            t += 1
        
    return S
                

def LaplaceM(X,k):

    """
    Define Laplace Matrix 
    -----
    input
    -----
    X: {numpy array},shape(n_features,n_samples)
        input data

    k: {int}
        neighborhood size
    ------
    output
    ------
    L: {numpy array},shape(n_samples,n_samples)
       Laplace matrix  

    """        
    d,n = X.shape
    temp = np.identity(n)- SimilarityM(X,k)
    L = np.dot(temp.transpose() , temp)
    return L


def JELSR(X,m,k=3,maxiter= 500, r=2,p=1,alpha = 1, beta =1 , Tol = 1e-3):
    
    """
    JELSR function implement unsupervised feature selection using Joint Embedding Learning Sparse Regression analysis.
    the optimization problem is the from equation (16) in the paper. 
    arg min_{W,YYT = I} tr(YLYT) + beta*(||WTX - Y ||_2 ^2 + alpha *||W||_{r,p}^p  ) 

    ------
    Input
    ------
    X: {numpy array},shape(n_features,n_samples)
        input data
    m: {int}
        dimensionality of embedding
    k: {int}
        neighborhood size,default value is 3 
    maxiter:{int}
        maximum number of iteration, default value is 500
    r:{int}
        parameter r in l_{r,p} norm, default value is 2
    p:{int} 
        parameter p in l_{r,p} norm, default value is 1
    alpha:{float}
        parameter alpha in optimization problem, default value is 1
    beta:{float}    
        parameter beta in optimization problem, default value is 1
    Tol:{float}    
        tolerance to stop the optimization, default value is 1e-3
    
    -------
    Output
    -------
  
    W:{numpy array},shape(n_features,n_embeddings) 
      feature weight matrix
    Y:{numpy array},shape(n_embeddings,n_samples)
      policy matrix  

    Reference:
      "Joint Embedding Learning and Sparse Regression: A Framework for Unsupervised Feature Selection"(2014) 
       by  Chenping Hou, Feiping Nie, Xuelong Li, Dongyun Yi and Yi Wu

    """

    d,n = X.shape
    U = np.identity(d)
    L = LaplaceM(X,k)
   
    
    # define objective function in equation (29) in the paper
    def traceobj(y):
        Y = np.array(y).reshape(m,n)
        A = np.dot(X,X.transpose()) + alpha * U
        Ainv = np.linalg.inv(A)
        mid = LaplaceM(X,k) + beta * np.identity(n) - beta * np.dot(np.dot( X.transpose(), Ainv),X )
        res1 = np.trace( np.dot(np.dot(Y,mid),Y.transpose() ))
        return res1
    
    def constr_func(y):
        Y = np.array(y).reshape(m,n)
        res2 = np.linalg.norm( np.dot(Y,Y.transpose()) -np.identity(m) )
        return res2
   
    y0 = np.zeros(m*n)
    targetvalue = np.zeros(maxiter)
    
    for step in range(maxiter):
        A = np.dot(X,X.transpose()) + alpha * U 
        nonlin_con = NonlinearConstraint(constr_func, lb = 0,ub =1e-4)  
        res = minimize(traceobj,y0,constraints = nonlin_con, tol = 1e-4,options={'maxiter':500,'disp':False})
        Y = res.x.reshape(m,n) 
        W = np.dot(np.dot(np.linalg.inv(A),X),Y.transpose())
        ListW = []
        for j in range(d):
            ListW.append( 1/ (np.linalg.norm(W[j,])*2) )
        U = np.diag(ListW)
        y0 = Y.reshape(1,m*n)[0]
        temp = np.linalg.norm( np.dot(W.transpose(),X)-Y )**2 + alpha * np.power(LRPNorm(W,r,p),p)
        targetvalue[step] = np.trace(np.dot(np.dot(Y,L),Y.transpose())) + beta * temp
        if step >= 1 and np.absolute(targetvalue[step]-targetvalue[step-1]) <= Tol:
            break

    return [W,Y]

# next define the feature selection function, s is the desired number of features.

def feature_selection(X,m,s,k=3):
    """
    Define feature selection function
    ------
    Input
    ------
    X: {numpy array},shape(n_features,n_samples)
        input data
    m: {int}
        dimensionality of embedding
    s: {int}
        number of features selected
    k: {int}
        neighborhood size,default value is 3 
  
    """
    d,n = X.shape
    W = JELSR(X,k,m)[0]
    scores = []
    for i in range(d):
        scores.append( np.linalg.norm(W[i,]))
    selected = np.argsort(scores,0)[::-1][:s]    
    return sorted(selected)


