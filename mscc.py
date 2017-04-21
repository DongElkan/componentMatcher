# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 21:57:52 2014

@author: ELKAN
"""

import numpy as np
import sirqr
import numcomp


def idxcheck(X,r):
    """ Check the index to correct it to the top of the peak as the index obtained by MSCC is often
        departed from the top if overlapped peaks appear. To achieve this, simple peak detection
        algorithm, i.e. curve derivative, is employed.
        Input:
            X       the raw data;
            r       the spectrum correlative chromatogram;
        Output:
            corrected index
    """
    idx  =np.argmin(r)
    s    =numcomp.svr(X)
    ms   =numcomp.smoother(s*1.)
    peaks=numcomp.peakdetect(ms)
    mx   =np.max(X,axis=1)
    
    # get the peak range in which idx locates
    def get_idx():
        mpeak       =[]
        curridx     =idx
        for peak in peaks:
            if curridx>=peak[0] and curridx<=peak[-1]:
                mpeak += peak
                break
        
        if len(mpeak)>0:
                c =np.argmax(mx[mpeak[0]:mpeak[-1]+1])
                c+=mpeak[0]
                return c
        return

    # correct index
    c=get_idx()
    if c and r[c]<=.3: return c

    # if not returned, i.e., the peak in which idx belongs is not found, check the peak
    # using original svr curve
    peaks=numcomp.peakdetect(s)
    c    =get_idx()
    if c and r[c]<=0.3: return c
    
    return idx


def projection(basematrix,targetmatrix,n):
    """ Project 'targetmatrix' onto the space spanded by loadings of 'basematrix' to get the correlation
        curve to find whether there has common component between the two matrices.
        n is the number of components in basematrix
    """
    nd= targetmatrix.shape[0]
    r = np.zeros(nd)
    v = np.linalg.svd(basematrix,full_matrices=0)[2]
    if n == 1:
        Y = targetmatrix-np.dot(v[0][:,None],np.dot(v[0],targetmatrix.T)[None,:]).T
    else:
        Y = targetmatrix-np.dot(v[:n].T,np.dot(v[:n],targetmatrix.T)).T
    
    for i in range(nd):
        r[i] = np.dot(targetmatrix[i],Y[i])/(np.linalg.norm(targetmatrix[i])*np.linalg.norm(Y[i]))
    return r


def mscc(Xtest,Xtarget,validrange):
    """ Multicomponent spectrum correlated chromatographic analysis (MSCC). If minimum correlative
        score obtained by Xtest is less than .3, the two curves
        Input:
            Xtest       2-way data for testing
            Xtarget     2-way data as target
            validrange  range in the correlative curve of Xtarget
        Outpu:
            index of minimum correlative score, if no, return None
    """
    rbd=sirqr.poolsirqr(Xtest)
    n  =numcomp.sevcomp(Xtest)
    
    # if no component is found, return
    if n == 0: return

    # calculate correlative curves and return index of curve minimum if satisfied setted condition
    # else return None
    r = projection(Xtest,Xtarget,n)
    
    ## check whether the curve is also a peak at peak area of marker
    ## if so, use the baseline corrected data as base matrix.
    sdr = np.diff(np.sign(np.diff(r[validrange[0]:validrange[1]])))
    pidx= np.where(sdr<0)[0]
    wid = False
    prng= int((validrange[1]-validrange[0])/2)
    if len(pidx) is 1 and ((pidx>prng*0.8)&(pidx<prng*1.2)):
        wid = True

    yeah = False
    if wid or not (r[validrange[0]:validrange[1]]<=0.2).all():
        ## comparing to baseline corrected peaks
        r = projection(rbd,Xtarget,n)
        if (r[validrange[0]:validrange[1]]<=0.2).all():
            yeah = True
    else:
        yeah = True

    if yeah: # find the marker, yeah!
        nm =numcomp.sevcomp(Xtarget)
        rr = projection(Xtarget,rbd,nm)
        rrs= np.where(rr<=0.3)[0]
        if len(rrs)>=3:
            c = idxcheck(rbd,rr)
            return c

    return
    
##if __name__=='__main__':
##    refdata = np.loadtxt(r'F:\data\LC\ITF\XYS-CH\ch_rawherbs\dCH2.txt')
##    refrt = np.loadtxt(r'F:\data\LC\ITF\XYS-CH\ch_rawherbs\tCH2.txt')
##    refwv = np.loadtxt(r'F:\data\LC\ITF\XYS-CH\ch_rawherbs\wCH2.txt')
##    rawdata = np.loadtxt(r'F:\data\LC\ITF\XYS-CH\ch_eit_converted\dXYS-6C-.txt')
##    rawrt = np.loadtxt(r'F:\data\LC\ITF\XYS-CH\ch_eit_converted\tXYS-6C-.txt')
##    rawwv = np.loadtxt(r'F:\data\LC\ITF\XYS-CH\ch_eit_converted\wXYS-6C-.txt')
##    refmrt = [[16.4,16.8],[23.5,24.1],[58.05,58.6]]
