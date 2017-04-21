# -*- coding: utf-8 -*-
"""
Curve smoothing via penalized least squares with 3rd derivatives as penalty 
matrix. 

All copyrights to Naiping Dong (np.dong572@gmail.com)
"""

from numpy import dot, ones, diff, array, diag, where, sign, mean, maximum, minimum,\
argmax, max, arange, floor, loadtxt, savetxt, sqrt, sum

from numpy.linalg import norm, cholesky, svd, inv
from numpy.random import rand
from tkinter import Toplevel, Label

import sirqr


def smoother(x,lbd=1200,d=3):
    """
    Smoothing the input curve x using penalized least squares[1]. The penalty
    matrix is 3rd derivative of identity matrix, which can be considered as
    special case of P-splines with 3rd derivatives[2]. It may be possible to
    use 1st or 2nd derivatives. If these are used, the nonzero elements in
    coefficient matrix A, i.e. variable "a" used below, should be changed. For
    1st derivative, matrix A is bidiagonal with
                a = array([[1,-1],[2,-1]]).
    For 2nd derivative, matrix A is tridiagonal with
                a = array([6,-4,1]).
    Since numpy's "dot" operation for large matrix multiplication is very slow,
    we used iteration method to assign the nonzeros elements by "a" directly. To
    speed up the computation, we omit the matrix multiplication but adopt band
    LU factorization or cholesky decomposition to solve banded linear system
    equations[3].
    
    ----------------------------------------------------------------
    Inputs:
        x:      Target curve for smoothing, must be a vector;
        lbd:    Penalty parameter;
        d:      order of derivative
    
    Output:
        Smoothed curve y
    
    Reference:
    [1] Eilers, P. H. C. A perfect smoother. Anal Chem. 2003, 75, 3631-3636.
    [2] Eilers, P. H. C.; Marx, B. D. Flexible smoothing with B-splines and
        penalties. Stat. Sci. 1996, 11, 89-121.
    [3] Golub, G. H.; van Loan, C. F. Matrix Computation. 4th ed, Johns Hopkins
        University Press, Baltimore, 2013, p177-180.
    """
    
    m = len(x)
    k = d+1
    # Construct coefficient matrix
    # Since matrix multiplication using dot command is very expensive, we assign
    # the nonzero elements to the coefficient matrix directly.
    A = diag(ones(m))
    a = [array([[1.,-1.],[2.,-1.]]),
         array([[1.,-2.,1.],[5.,-4.,1.],[6.,-4.,1.]]),
         array([[1.,-3.,3.,-1.],[10.,-12.,6.,-1.],[19.,-15.,6.,-1.],[20.,-15.,6.,-1.]])]

    a = a[d-1]*lbd # nonzeros in matrix A
    a[:,0] += 1
    for i in range(d):
        A[i,i:i+k] = A[i:i+k,i] = a[i]
    for i in range(d,m-d):
        A[i,i:i+k] = A[i:i+k,i] = a[d]
    for i in range(-d,-1):
        A[i-d:i+1,i] = A[i,i-d:i+1] = a[-i-1][::-1]
    A[-k:,-1] = A[-1,-k:] = a[0][::-1]
    
    L = cholesky(A)
    
    # solve the equation system, i.e. Ay = x, recursively
    # solve L*b = x
    diag_L = diag(L)
    y = x*1.
    y /= diag_L
    for i in range(1,d):
        y[i] -= dot(L[i,:i],y[:i])/diag_L[i]
    for i in range(d,m):
        y[i] -= dot(L[i,i-d:i],y[i-d:i])/diag_L[i]
   
    # solve L.T*y = b
    L = L.T
    y /= diag_L
    for i in range(-2,-k,-1):
        y[i] -= dot(L[i,i+1:],y[i+1:])/diag_L[i]
    for i in range(m-k,0,-1):
        y[i] -= dot(L[i,i+1:i+k],y[i+1:i+k])/diag_L[i]
    
    return y


def svr(D):
    """
    SVR plot of 2-D array data[1]. SVR is the ratio of largest and second largest
    singular values obtained by singular value decomposition (SVD). Using SVR,
    each component's peak can be detected from each peak of SVR plot, thus the
    number of components in "D" is determined by counting the peak number via 1st
    and 2nd derivatives.
    
    -------------------------------------------------------------
    Input:
        D:      2-D array data
    
    Output:
        SVR array
        
    Reference:
    [1] Hu, Y. Z.; Shen, W. Y.; Yao, W. F.; Massart, D. L. Using singular value
        ratio for resolving peaks in HPLC-DAD data sets. Chemometr Intell Lab
        Syst. 2005, 77, 97-103.
    """
    
    m = D.shape[0]
    r = ones(m-1)
    for i in range(m-1):
        v = svd(D[i:i+2,:],compute_uv=False)
        r[i] = v[0]/v[1]
    
    return r

    
def peakdetect(x,minpeakh = 18,
               minpeakw = 6):
    """
    Detecting peaks from SVR plot to determine the number of components in the
    input vector x. To accurately detect peaks in x, first derivative of x is
    calculated. The start, center and end points of a peak can be obtained by
    finding the cross points from negative to positive, positive to negative
    and negative to positive[1,2]. If overlapped peaks occur, 2nd derivative
    is calculated and multiple local minimas in a peak range defined in 1st
    derivative curve are probably the peak overlapping another peaks[2]. To
    remove false positives, several criteria like the minimum number of points
    required for a peak and minimum intensity are employed.
    
    ---------------------------------------------------------------
    Inputs:
        x:          SVR array obtained from function "svr";
        D:          Not preprocessed 2-D array raw data; This is used to avoid
                    splitting effect probably caused by sharp signal after baseline
                    correction;
        minpeakh:   Minimum peak intensity required for a peak, default is 18;
        minpeakw:   Minimum peak width required for a peak, represented by the
                    data points between start point and central point of a
                    potential peak, default is 6;
    
    Output:
        peak information with start, center and end points of all peaks, peakinfo
        
    
    References
    [1] Dixon, S. J.; Brereton, R. G.; Soini, H. A.; Novotny, M. V.; Penn, D. J.
        An automated method for peak detection and matching in large gas
        chromatography-mass spectrometry data sets. J Chemometrics. 2006, 20,
        325-340.
    [2] Truyols, G. V., Torres-Lapasio, J. R., van Nederkassel, A. M., Heyden,
        Y. V., Massart, D. L. Automatic program for peak detection and deconvolution
        of multi-overlapped chromatographic signals: Part I: Peak detection. J
        Chromagr A. 2005, 1096, 133-145.
    """
    
    signdx = sign(diff(x))
    posidx = where(diff(signdx)>0)[0]+1
    negidx = where(diff(signdx)<0)[0]+1
    np = len(posidx)
    # preallocation
    peakinfo = []
    
    if np==0 or len(negidx)==0: return peakinfo
    
    # peak searching
    for i in range(np-1):
        nnidx = negidx[(negidx>posidx[i])&(negidx<posidx[i+1])]
        # check whether a plain occurs at the peak top, which will be presented as
        # multiple negative indices between two adjacent positive indices. If existed,
        # middile sites is retained
        centeridx = int(mean(nnidx))+1 if len(nnidx)>=2 else nnidx[0]
        
        # criteria: minimum peak intensity and minimum distance between peak
        # top and starting point
        if centeridx-posidx[i] >= minpeakw and x[centeridx] >= minpeakh:
            # coelute peak check using 2nd derivative
            d2x=diff(x[posidx[i]:posidx[i+1]+1],n=2)
            nx  =len(d2x)
            localmin =[]
            localmax =[]
            for j in range(1,nx-1):
                
                if d2x[j] <= d2x[j-1] and d2x[j] <= d2x[j+1]:
                    idx = arange(maximum(0,j-2),minimum(j+3,nx))
                    idx = idx[where(d2x[idx]==d2x[j])[0]]
                    # to avoid plain minima
                    localmin.append(int(mean(idx))+1 if len(idx)>1 else idx[0])
                    
                if d2x[j] >= d2x[j-1] and d2x[j] >= d2x[j+1]:
                    idx = arange(maximum(0,j-2),minimum(j+3,nx))
                    idx = idx[where(d2x[idx]==d2x[j])[0]]
                    # to avoid plain maxima
                    localmax.append(int(mean(idx))+1 if len(idx)>1 else idx[0])

            if len(localmin)>0:
                localmin = array(localmin)
                localmax = array(localmax)
                localmin += posidx[i]-1
                localmin = localmin[x[localmin]>=minpeakh]
                localmax += posidx[i]-1
                
                if len(localmin)>1:
                    peakinfo.append([posidx[i],localmin[0],localmax[localmax>localmin[0]][0]])
                    for j in range(1,len(localmin)):
                        idx = where(localmax>localmin[j])[0]
                        if len(idx)>0 and j!=len(localmin):
                            peakinfo.append([localmax[idx[0]-1],localmin[j],localmax[idx[0]]])
                        else:
                            peakinfo.append([localmax[-1],localmin[j],posidx[i+1]])
                else:
                    peakinfo.append([posidx[i],centeridx,posidx[i+1]])
    
    # Check the points before the first peak and after the last peak to identify
    # whether there is part peak due to the arbitrary data selection
    ## Check peak AFTER peak
    p = posidx[-1]
    if len(x)-p>=minpeakw:
        if any(negidx>p): # there exist peak maximum
            if x[negidx[-1]]>=minpeakh and negidx[-1]-p>=minpeakw:
                peakinfo.append([p,negidx[-1],len(x)])
        elif x[-1]>=minpeakh:
            peakinfo.append([p,len(x)-int(p/2),len(x)])
    
    ## Check peak BEFORE peak
    p = posidx[0]
    if p>=minpeakw:
        if any(negidx<p): # there exist peak maximum
            if x[negidx[0]]>=minpeakh and p-negidx[0]>=minpeakw:
                peakinfo.insert(0,[0,negidx[0],p])
        elif x[0]>=minpeakh:
            peakinfo.insert(0,[0,int(p/2),p])
    
    return peakinfo


def peakcheck(peakinfo,D,
              minpeakw = 12,
              minpeakh = 18):
    """
    Check the peak to remove false positives. Two criteria are employed here:
    a.  Spectra similarity check: calculate the spectral similarity score between
        adjacent peaks, if the score "s" larger than 0.999, these two peaks are
        combined into single peak;
    b.  The width of a peak must be larger than "minpeakw";
    
    --------------------------------------------------------------------
    Inputs:
        peakinfo:   Peak information obtained from function "peakdetect", which
                    is a matrix with three columns containing start, central,
                    end point of each peak;
        D:          Raw data;
        minpeakw:   Minimum peak width required for a peak;
        minpeakh:   Minimum peak intensity required for a peak;
    
    Output:
        checked peak information, peakinfo
    """
        
    # check criterion a
    n = D.shape[0]
    m = len(peakinfo)
    r = [True]*m
    for i in range(m):
        # check criterion b
        if peakinfo[i][0]!=0 and peakinfo[i][-1]<n-1: # not part peak
            if peakinfo[i][-1]-peakinfo[i][0]<=minpeakw:
                r[i] = False
                continue
        if i<m-1:
            s = dot(D[peakinfo[i][1]],D[peakinfo[i+1][1]])/\
            (norm(D[peakinfo[i][1]])*norm(D[peakinfo[i+1][1]]))
            print(s)
            if s >= 0.99:
                t = D[peakinfo[i][0]:peakinfo[i+1][-1]+1]
                maxc = argmax(max(t,axis=0))
                idx = argmax(t[:,maxc])
                peakinfo[i+1][:2] = [peakinfo[i][0],peakinfo[i][0]+idx-1]
                r[i] = False
            
    peakinfo = [peakinfo[i] for i in range(m) if r[i]]
    return peakinfo

def peaks(raw_data, bdpath=0, tag=True):
    """
    Detect peaks of whole raw data using functions in this module. If tag is true,
    baseline correction should be performed and stored
    """
    
    # baseline correction
    ## show precessing message till all baselines have corrected.
    if tag:
        txt = 'Baseline correcting...\n'+\
              'This will take seconds to minutes, depending on your '+\
              'CPU and RAM, please wait...'
        progresswin = Toplevel()
        progresswin.title('Processing...')
        Label(progresswin,text=txt).pack(side='top',expand=1,fill='both')
        progresswin.resizable(width=False,height=False)
        #
        bd  = sirqr.poolsirqr(raw_data)
        savetxt(bdpath,bd,fmt='%f')
        #
        progresswin.destroy()
    else:
        bd = loadtxt(bdpath)
    # SVR plot
    s = svr(bd)
    
    r = len(s)
    # devide raw_data into intervals with 'npoints' points in each interval to
    # avoid memory error during curve smoothing
    npoints = 1500
    if r <= npoints+npoints/3:
        smoothedSVR = smoother(s)
        peakinfo = peakdetect(smoothedSVR)
        peakinfo = peakcheck(peakinfo)
    else:
        # smoothing and peak detection in each intervals
        nls = int(floor(r/npoints))
        if r % npoints >= npoints/3: nls += 1
        peakinfo = ones((1000,3),dtype=int)
        n = 0
        p1 = 0
        p2 = npoints
        for i in range(nls-1):
            smoothedSVR = smoother(s[p1:p2]*1.0)
            temp = peakdetect(smoothedSVR)
            temp = peakcheck(temp*1,raw_data[p1+1:p2+1])
            tn = temp.shape[0]
            peakinfo[n+tn,:] = temp+p1
            n += tn
            p1 = p2
            # If the distance between last point (i.e. peak end of last peak) of
            # peaks detected and interval end in current interval is less than
            # 30 scan points, the last peak may be arbitrarily cut, thus the start
            # point of next interval should be the last point of second last peak.
            if tn>1 and temp[-1,-1]<p1+p2-30:
                p1 = temp[-2,-1]
                peakinfo[n] = 1
                n -= 1
            p2 = r+1 if i == nls-2 else p2+npoints*(i+1)
        peakinfo = peakinfo[:n]

def sevcomp(D, penlambda = 12500,manner=1):
    """
    Compare the eigenvalues of ordinary PCA and functional PCA of 2-D day to detect
    the number of components. The basic idea behind this is that component signal
    is smoothing, so the eigenvalues obtained after perfroming functional PCA should
    not be varied significantly, whereas eigenvalues for noises do. Thus comparing the
    variation of eigenvalues after functional PCA can obtain the number of components
    in current data. Though smoothing procedure can call the function "smooth" in this
    module directly, it will spend lots of times on constructing and decomposing
    coefficient matrix. Thus in this function, the codes are imbedded.

    For speeding up computation, in place replacement is employed. Further, all the
    computation are performed in vector
    
    Reference:
    Chen, Z. P.; Liang, Y. Z.; Jiang, J. H.; Li, Y.; Qian, J. Y.; Yu, R. Q.
    Determination of the number of components in mixtures using a new approach
    incorporating chemical information. J Chemometr, 1999, 13(1), 15–30.
    """

    evs= svd(D,full_matrices=0,compute_uv=0)
    d = 3
    k = d+1
    r,c  = D.shape
    # smoothing
    # ... Construct coefficient matrix
    # ... Since matrix multiplication using dot command is very expensive, we assign
    # ... the nonzero elements to the coefficient matrix directly.
    A = diag(ones(r))
    a  = array([[1,-3,3,-1],[10,-12,6,-1],[19,-15,6,-1],[20,-15,6,-1]])*penlambda
    a[:,0] += 1
    for i in range(3):
        A[i,i:i+4] = A[i:i+4,i] = a[i]
    for i in range(3,r-3):
        A[i,i:i+4] = A[i:i+4,i] = a[3]
    for i in range(-3,-1):
        A[i-3:i+1,i] = A[i,i-3:i+1] = a[-i-1][::-1]
    A[-4:,-1] = A[-1,-4:] = a[0][::-1]
    
    L = cholesky(A)
    if manner == 1:
        invL = inv(L)
        Y = dot(invL.T,dot(invL,D))
    else:
        # ... Solve the equation system, i.e. Ay = x, recursively. To speed up computation, in place
        # ... replacement is used.
        diag_L = diag(L)
        # ... Smoothing
        Y = D*1.0
        for j in range(c):
            y  = D[:,j]*1.0
            # ... solve L*b = x, here L*y = y for in place replacement
            y /= diag_L
            for i in range(1,d):
                y[i] -= dot(L[i,:i],y[:i])/diag_L[i]
            for i in range(d,r):
                y[i] -= dot(L[i,i-d:i],y[i-d:i])/diag_L[i]
           
            # ... solve L.T*y = b
            y /= diag_L
            for i in range(-2,-k,-1):
                y[i] -= dot(L[i+1:,i],y[i+1:])/diag_L[i]
            for i in range(r-k,0,-1):
                y[i] -= dot(L[i+1:i+k,i],y[i+1:i+k])/diag_L[i]
            Y[:,j] = y
        # smoothing ended
    
    sevs= svd(Y,full_matrices=0,compute_uv=0)
    
    r = sevs/evs
    if not (r<0.8).any():
        n = 0
        return n
    
    n   = where(r<0.8)[0][0]
    return n


def npmpca(D,penlambda=1200,
             noise    =0.001,
             N        =60,
             ncomp    =20):
    """
    Noise perturbation method to determine the number of components
    """
    
    r,c = D.shape
    ind = ones((ncomp,N))
    # Construct coefficient matrix
    # Since matrix multiplication using dot command is very expensive, we assign
    # the nonzero elements to the coefficient matrix directly.
    A = diag(ones(c))
    a = array([6.,-4.,1.])*penlambda
    a[0] += 1
    for i in range(c-2):
        A[i,i:i+3] = A[i:i+3,i] = a
    for i in range(-2,0):
        A[i:,i] = A[i,i:] = a[::-1][:-i]
    
    uc,sc,vc = svd(A,full_matrices=0)
    invcs    = 1./sc
    invA     = dot(uc*invcs,vc)
    invA2    = dot(uc*sqrt(invcs),vc)
    
    # set up parameters
    m = max(D)*noise
    # set up experiments
    for i in range(N):
        R = D+rand(r,c)*m
        Z = dot(R,invA2)
        uz,sz,vz = svd(Z,full_matrices=0)
        vv= dot(dot(invA,R.T),uz)
        d = 1./sum(vv**2,axis=0)
        vv*=d
        
        ur,sr,vr = svd(R,full_matrices=0)
        
        for j in range(ncomp):
            ind[j,i] = abs(dot(vr[j],vv[:,j]))
    
    return ind

