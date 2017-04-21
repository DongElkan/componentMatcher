# -*- coding: utf-8 -*-
"""
SirQR baseline correction

SirQR is a baseline correction algorithm based on iteratively reweighted quantile
regression which is robust, computation efficient, flexible and effective. This
program can be used as a standalone command line for future modifications and
reuse in new project.
Since The coefficient matrix "B" is the concatenation of an identity matrix and
adjusted matrix to transform the smoothing series z into differences of neighboring
elements, the latter of which is a bidiagonal matrix, the multiplication of the
coefficient matrix and other matrices or vectors is equivalent to concatenate
these matrices or vectors and the differences of them (i.e. diff(X)). As matrix
multiplication by numpy (via "dot" function) is very slow, it is much faster to
use numberic computation directly, for example "diff" and "concatenate" functions,
in quantile regression. Further, because the input y for quantile regression has
been extended by a zero vector, we omit the dot product operation involving this.
To retain the original codes in MATLAB, codes for matrix multiplications are
provided in the annotations.

Reference:
Liu, X. B.; Zhang, Z. M.; Sousa, P. F. M.; Chen, C.; Ouyang, M. L.; Wei, Y. C.;
Liang, Y. Z.; Chen, Y.; Zhang, C. P. Selective iteratively reweighted quantile
regression for baseline correction. Anal Bioanal Chem. 2014, 406, 1985-1998.

LICENCE

All Copyrights to Naiping Dong (np.dong572@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from numpy import dot, diff, ones, zeros, concatenate, exp, vstack, amin, sum, abs
 
def _tridiagsolver(m,penlambda,q,r):
    """
    Solving tridiagonal equation system via Thomas algorithm[1].
    
    Reference:
    [1] Wiki: http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    """
    
    # Set up constants
    # ... y
    d = q[:m]*r[:m]
    d[0] = d[0]-q[m]*r[m]*penlambda
    d[1:-1] = d[1:-1]-penlambda*(r[m+1:]*q[m+1:]-r[m:-1]*q[m:-1])       #diff(r[m:]*q[m:],axis=0)
    d[-1] = d[-1]+r[-1]*q[-1]*penlambda
    
    # ... tridiagonal matrix vectors
    q[m:] = q[m:]*penlambda**2
    a = -1*q[m:]                              # elements at i,i-1 and i+1,i
    b = q[:m]*1                               # elements at digonal
    b[0] = b[0]+q[m]
    b[1:-1] = b[1:-1]+q[m:-1]+q[m+1:]
    b[-1] = b[-1]+q[-1]

    
    # LU Decomposition
    a[0] = a[0]/b[0]
    d[0] = d[0]/b[0]
    for i in xrange(1,m-1):
        t = b[i]+a[i-1]*q[m+i-1]
        a[i] = a[i]/t
        d[i] = (d[i]+q[m+i-1]*d[i-1])/t
    d[-1] = (d[-1]+q[-1]*d[-2])/(b[-1]+q[-1]*a[-1])
    
    # output y
    y = d*1
    for i in xrange(2,m+1):
        y[-i] = d[-i]-a[-i+1]*y[-i+1]
    
    return y.T
    

def lp_fnm(m,c,theta,w,penlambda,
           p = 0.9995,
           ebs = 1e-5,
           maxiter = 50):
    """
    Sovlving linear programming problem for quantile regression using interior
    point method with Mehrotra corrector[1]. The problem can be characterized as
    min(c'*x), s.t. A*x = b and 0<x<1
    All quantile regression codes are translated from MATLAB codes obtained from
    http://www.econ.uiuc.edu/~roger/research/rq/rq.html
    The detail of the algorithm can be found in his text book of quantile
    regression by Koenker, R [2].
    
    NOTE: In solving linear programming problems, majority computation times
          are spent inversing the QAQ.T to solve linear equations AQ*dy = rhs.
          Fortunately, as coefficient matrix AQ or A is special, i.e., a 
          concatenation of an identity matrix and the column difference of the
          identity matrix, AQ'*AQ is a tridiagonal matrix. Thus, for equation
          AQ*dy = rhs, a transformation can be obtained as:
                     AQ'*AQ*dy = AQ'*rhs
          Notice that AQ here is the row concatenation of two matrices since in
          the original MATLAB code the input A is the transformed coefficient matrix.
          The new linear equation system therefore has a special coefficient
          matrix, and efficient algorithm named Thomas algorithm can be applied
          to solve this. This leads current implementation linear time complexity.
    
    ---------------------------------------------------------------------------
    Inputs:
        m:          size of the input raw data;
        c:          Coefficient vector in object function c'*x;
        theta:      Quantile of interest;
        w:          Weights, if no, set them to 1.0;
        penlambda:  Penalty value for baseline correction;
        p:          Adjustment parameter, suggested to be 0.9995;
        ebs:        Minimum gap between primal and dual function, default is 1e-5;
        maxiter:    Max iteration, default is 50;
    
    Reference:
    [1] Mehrotra, S. On the Implementation of a Primal-Dual Interior Point Method. 
        SIAM Journal on Optimization. 1992, 2, 575-601.
    [2] Koenker, R. Quantile Regression, Econometric Society Monograph Series,
        Cambridge University Press, 2005.
    """
    
    tridiagsolver = _tridiagsolver
    # This is column number of coefficient matrix * 2, for line 248, i.e.
    # mu = mu*(g/mu)**3/col
    col = 4*m-2
    c = c*-1
    c1 = vstack((c.T,zeros((m-1,1))))
    
    # ...Generate initial feasible point
    s = theta*w
    a = w-s
    b = a[0:m]+penlambda*vstack((-a[m],-1*diff(a[m:],axis=0),a[-1]))   # dot(A.T,a)
    y = tridiagsolver(m,penlambda,ones((col/2,1)),c1)                   # dot(pinvA.T,c.T).T <=> dot(c,pinvA)
    r = concatenate((c-y,-penlambda*diff(y)),axis=1)                   # c-dot(y,A)
    r[r==0] = 0.001
    u = r*(r>0)
    v = u-r
    gap = dot(c,a[:m]) - dot(y,b) + dot(v,w)
    
    # ...Start iterations
    it = 0
    while gap > ebs and it < maxiter:
        it = it+1
        
        # compute affine step
        q = 1/(u/a.T + v/s.T)
        r = u-v
        dy = tridiagsolver(m,penlambda,q.T*1,r.T)
        da = (q*(concatenate((dy,penlambda*diff(dy)),axis=1)-r))    # dot(dy,A)
        ds = -da
        du = -u*(1+da/a.T)
        dv = -v*(1+ds/s.T)
        
        # ...Compute maximum allowable step lengths
        # Here we omit to construct additional function "round" used in the original
        # code to set boundaries for Newton step. Instead, find the minimum of
        # the ratio and 1. Since the 1e20*p is significantly larger than 1, this
        # can be deleted during minimum operation. However, to avoid the error
        # raised due to the empty vector, try-except structure is used.
        try:
            fp = min(p*amin(-a.T[da<0]/da[da<0]),
                     p*amin(-s.T[ds<0]/ds[ds<0]),
                     1)
        except ValueError:
            fp = 1
        
        try:
            fd = min(p*amin(-u[du<0]/du[du<0]),
                     p*amin(-v[dv<0]/dv[dv<0]),
                     1)
        except ValueError:
            fd = 1

        # If full step is feasible, take it. Otherwise modify it using Mehrotra
        # corrector.
        if min(fp,fd) < 1:
            
            # Update mu
            mu = dot(u,a) + dot(v,s)
            g = dot((u+fd*du),(a+fp*da.T)) + dot((v+fd*dv),(s+fp*ds.T))
            mu = mu*(g/mu)**3/col
            
            # Compute modified step
            dadu = da*du
            dsdv = ds*dv
            xi = (mu * (1/a - 1/s)).T
            r = r + dadu - dsdv - xi
            dy = tridiagsolver(m,penlambda,q.T*1,r.T)
            da = q*(concatenate((dy,penlambda*diff(dy)),axis=1)-r)     # dot(A,dy)
            ds = -da
            du = mu/a.T - u - u/a.T*da - dadu
            dv = mu/s.T - v - v/s.T*ds - dsdv
            
            # Compute maximum allowable step lengths
            # print('a=',a,'\n','dy=',dy,'\n','s=',s,'\n','ds=',ds,'\n')
            try:
                fp = min(p*amin(-a.T[da<0]/da[da<0]),
                         p*amin(-s.T[ds<0]/ds[ds<0]),
                         1)
            except ValueError:
                fp = 1
            try:
                fd = min(p*amin(-u[du<0]/du[du<0]),
                         p*amin(-v[dv<0]/dv[dv<0]),
                         1)
            except ValueError:
                fd = 1

        # Take the step
        a = a + fp*da.T
        s = s + fp*ds.T
        y = y + fd*dy
        v = v + fd*dv
        u = u + fd*du
        gap = dot(c,a[:m]) - dot(y,b) + dot(v,w)
        
    return -y.T


def _sirqr(X,
          penlambda = 1.25,
          u = 0.03,
          wlow = 1e-10,
          d = 5e-5,
          maxiter = 5):
    """
    Main function of SirQR
    Inputs:
        X:          Raw data with rows are chromatograms and columns are wavelengths;
        penlambda:  Penalty parameter;
        u:          Quantile used for quantile regression;
        wlow:       Penalty value for possible peak signals to ignore the processing
                    of them in next iteration;
        d:          Parameter for a better fitting to the original signal dataset;
        maxiter:    Maximum iteration;
    """
    
    r,c = X.shape
    wep = 1e-4
    B = zeros((r,c))
    for i in xrange(c):
        w = ones((2*r-1,1))*wep
        x0 = X[:,i][:,None]                     # A column vector
        for j in xrange(maxiter):
            z = lp_fnm(r,x0.T,u,w,penlambda)
            dx = X[:,i][:,None]-z
            ds = abs(sum(dx[dx<d]))
            # Here we avoid determinating the end of the iteration by comparing
            # ds to 5e-5*sum(abs(X[i,:])) as iterating 4 times is enough.
            w[dx>d] = wlow
            w[dx<d] = exp((j+1)*abs(dx[dx<d])/ds).T
            x0 = z
            
        B[:,i][:,None] = dx
        
    return B
