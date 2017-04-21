# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 16:13:20 2014

@author: ELKAN
"""
from numpy import dot, ones, sign, argmax, argmin, max, sum
from numpy.linalg import norm, svd
import matplotlib.pyplot as plt

def subspacecomp(d,K=10,k=1.0):
    """
    Determine chemical rank from 2D data using subspace comparison[1]. According
    to theories of chemometrics, 2D data is comprised by sample components which
    make up the primary axes and noises that make up the secondary axes[2]. Generally,
    primary axes are important variables, and always locate at the top singular
    values after decompsing the data matrix by singular value decomposition (SVD). 
    Thus, current data system should be represented by these components, i.e.,
    components with top singular values. If we consider this as a sample space,
    it is spanned by orthogonal vectors of those singular values, from which we
    can extract most important variables. In additional to this, there are several
    other algorithms developed to extract important variables directly. Therefore,
    it is possible to find the true important variables if we compare different
    variable extraction methods. Currently, SVD and simplified Borgen method (SBM)[3]
    are adopted for this comparison and return the number of important variables.
    In the comparison, a subspace discrepancy function is defined to measure the
    part of one of the subspaces which is in the orthogonal complement of the
    other. The number of components in current system are then obtained when the
    function arrives its global minimum under predefined number K (which is the
    estimation of the number).
    
    -----------------------------------------------------------------
    Input:
        d:      2D data array with columns are spectra along time domain or sample
                number point and rows are spectra measured along wavelengths; for
                HPLC-DAD data, columns are chromatograms and rows are UV spectra;
        K:      Number of components hypothesized, default is 10;
        k:      Ridge parameter, default is 1.0;
    Output:
        Number of important variables
        
    
    Reference:
    [1] Shen, H. L.; Liang, Y. Z.; Kvalheim, O. M.; Manne, R. Determination of
        chemical rank of two-way data from mixtures using subspace comparisons.
        Chemometr Intell Lab Sys. 2000, 51, 49-59
    [2] Malinowski, E. R. Factor Analysis in Chemistry, 3rd Ed. Wiley, 2002.
    [3] Grande, B. V.; Manne, R. Use of convexity for finding pure variables in
        two-way data from mixtures. Chemometr Intell Lab Sys. 2000, 50, 19-33
    """
    
    u,s,v = svd(d,full_matrices=0)
    v[0] *= sign(sum(u[:,0]))         # The first column of matrix V
    u[:,0] *= sign(sum(u[:,0]))
    
    # ridge paramter
    u = u.T
    v = v.T
    N = s[0]*u[0]
    rp = k*max(N)
    Y = s[1:K][:,None]*u[1:K]/(N+rp)
    # selecting variables
    idx = ones(K,dtype=int)
    
    # first important variable
    idx[0] = argmax(sum(Y**2,axis=0))
    
    # second variable
    Y -= Y[:,idx[0]][:,None]
    idx[1] = argmax(sum(Y**2,axis=0))
    
    for i in xrange(2,K):
        e = Y[:,idx[i-1]][:,None]
        e /= norm(e)
        # orthogonalization
        Y -= dot(e,dot(e.T,Y))
        idx[i] = argmax(sum(Y**2,axis=0))
    
    # Modified Schmidt-Gram Orthogonalization
    P = d[idx]*1.0
    Q = P*1.0
    for i in xrange(K-1):
        Q[i] = P[i]/norm(P[i])
        P[i+1:] -= Q[i]*dot(Q[i],P[i+1:].T)[:,None]
    Q[-1] = P[-1]/norm(P[-1])
        
    # subspace comparison
    t = ones(K)
    for i in xrange(1,K+1):
        G = dot(Q[:i],v[:,:i])
        t[i-1] = i-sum(G**2)
    
    plt.plot(t)
    return argmin(t)+1
    
    
#    x = np.arange(c)
#    m = x*1.0
#    for i in xrange(c):
#        m[i] = np.max(d[i])
#        
#    plt.plot(x,d)
#    plt.xlabel('Scan Time Point')
#    plt.ylabel('Intensity')
#    plt.hold(True)
#    plt.plot(idx,m[idx]+0.1,'ro')
#    for i in xrange(len(idx)):
#        y = np.arange(0,m[idx[i]]+0.1,0.001)
#        xx = np.tile(idx[i],len(y))
#        plt.plot(xx,y,'k--')
#    plt.savefig('fig1.png',dpi=600)
#    plt.figure(2)
#    plt.plot(t)
#    plt.savefig('fig2.png',dpi=600)