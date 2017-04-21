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
            idx     the index obtained by MSCC;
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


def commonpeakcheck(r, mkrt,startidx, ps, fps):
    """ Check whether multiple markers are assigned to same peak in formula data. This
        is probably caused by very close retention times of these markers whose spectra
        are very similar. If this situation exists, minimas of correlative curves are
        compared to each other and assigned to markers according to their retention
        orders. If there are not enough minimas for the assignments, markers with
        highest minimas are neglected.
        Note: If multiple markers are assigned to same peak, reassign them to peaks
              according to the retention orders is an optimization problem. Therefore
              currently no more than 3 markers are considered. For 3 markers, the
              criteria to decide the assignments are as follows:
                  a. full assignments, i.e. all 3 markers are assigned to formula peaks,
                     are preferred;
                  b. if several full assignments exist, lowest sum correlative score
                     is output;

        Inputs:
            r           correlative curves of markers which assign to same peak in formula
                        data
            mkrt        retention times of markers
            startidx    start indices of markers in retention time array
            ps          peak clusters of the markers
            fps         ids of peak clusters in which True indicates that the peak has
                        been assigned to other markers

        Outputs:
            reordered indices
            """
    nmk   =len(mkrt)
    if nmk<2: return np.argmin(r)
    rt    =[sum(t)/2. for t in mkrt]
    mksidx=sorted(range(len(rt)), key=lambda k: rt[k]) # reserve the original index
    mkrt.sort()

    # get minima indices of each correlative curve
    minsidx=[]
    minsort=[]
    minrs  =[]
    for i in xrange(nmk):
        dx =np.where(np.diff(np.sign(np.diff(r[i])))>0)[0]+1
        dzx=np.where(np.diff(r[i])==0)[0] # get the indices where flats exist
        dx =np.sort(np.concatenate((dx,dzx)))
        dx =dx[r[i][dx]<=0.3]
        ndx=len(dx)
        f  =np.ones(ndx,dtype=bool)
        for j in xrange(ndx):
            if f[j]:
                ddx=dx[j:]-dx[j]
                idx=np.where(ddx==1)[0]
                if len(idx)>0:
                    f[j+idx]=False
            # if the peak has been assigned to other markers, ignore it
            idxt=[k for k in xrange(len(ps[i])) if ps[i][k][0]<=dx[j]<=ps[i][k][2]]
            if len(idxt)>0 and fps[i][idxt[0]]:
                f[j]=False
        dx =dx[f]
        # reserve the original order
        sx =sorted(range(len(dx)), key=lambda k: r[i][dx[k]])
        sdx=[sx.index(j) for j in xrange(len(sx))] # the order number of original list
        minsidx.append(list(dx+startidx[i]))
        # get the indices and scores
        minrs.append(list(r[i][dx]))
        minsort.append(sdx)

    def getidx(minsort1, minsort2, minsidx1, minsidx2, minrs1, minrs2):
        """ Get reordered indices of two markers """
        if len(minsort1)>0 and len(minsort2)>0:
            idx1      =minsort1.index(min(minsort1)) # index of first correlative curve min
            idxt      =minsort2.index(min(minsort2)) # index of second correlative curve min
            # index of first value in second correlative curve with global index greater
            # than that in first curve
            if minsidx1[idx1]>=max(minsidx2)-2:
                idx3  =len(minsidx2)
            else:
                idx3  =next(i for i,v in enumerate(minsidx2) if v>=minsidx1[idx1])
            idxt      =idxt if idxt>=idx3 else idx3 # keep higher index
            if idx3<len(minsidx2) and minsidx1[idx1]-2<=minsidx2[idxt]<=minsidx1[idx1]+2:
                idxt+=1

            try:
                idx2  =minsort2[idxt:].index(min(minsort2[idxt:]))
                s1    =minrs1[idx1]+minrs2[idxt+idx2]
                rgidx1=[minsidx1[idx1],minsidx2[idxt+idx2]]
            except ValueError:
                s1    =minrs1[idx1]
                rgidx1=[minsidx1[idx1], None]

            idxt      =minsort2.index(min(minsort2)) # index of second correlative curve min
            # if no peak is found before the minimum in 2nd curve
            if minsidx2[idxt]<=min(minsidx1):
                idx1  =-1
            else:
                pres  =[t for t in minsidx1 if t<minsidx2[idxt]]
                idx1  =pres.index(max(pres))
            if idx1>=0 and minsidx2[idxt]-2<=minsidx1[idx1]<=minsidx2[idxt]+2:
                idx1-=1
            
            try:
                idx2  =minsort1[:idx1+1].index(min(minsort1[:idx1+1]))
                s2    =minrs1[idx2]+minrs2[idxt]
                rgidx2=[minsidx1[idx2],minsidx2[idxt]]
            except ValueError:
                s2    =minrs2[idxt]
                rgidx2=[None, minsidx2[idxt]]
        else:
            if len(minsort2)>0:
                idxt  =minsort2.index(min(minsort2))
                return [None, minsidx2[idxt]], minrs2[idxt]
            elif len(minsort1)>0:
                idxt  =minsort1.index(min(minsort1))
                return [None, minsidx1[idxt]], minrs1[idxt]
            else:
                return [None, None], float('inf')

        if None in rgidx1 and None in rgidx2:
            rgidx, s =(rgidx1, s1) if s1<=s2 else (rgidx2, s2)
        elif None in rgidx1:
            rgidx, s =rgidx2, s2
        elif None in rgidx2:
            rgidx, s =rgidx1, s1
        else:
            rgidx, s =(rgidx1, s1) if s1<=s2 else (rgidx2, s2)
        
        return rgidx, s

    # .................................................................................
    # get the indices
    # .................................................................................
    if nmk==2:
        idx,s=getidx(minsort[0],minsort[1],minsidx[0],minsidx[1],minrs[0],minrs[1])
    elif nmk==3:
        # assign minimum to first index
        idx1  =minsort[0].index(min(minsort[0]))
        s1    =minrs[0][idx1]
        if minsidx[0][idx1]>max(minsidx[2])-2:
            rgidx1=[minsidx[0][idx1],None,None]
        elif minsidx[0][idx1]>max(minsidx[1])-2:
            idx2  =next(i for i,v in enumerate(minsidx[2]) if v>minsidx[0][idx1])
            idx3  =minsort[2][idx2:].index(min(minsort[2][idx2:]))
            rgidx1=[minsidx[0][idx1],None,minsidx[2][idx2+idx3]]
            s1   +=minrs[2][idx2+idx3]
        else:
            idx2  =next(i for i,v in enumerate(minsidx[1]) if v>minsidx[0][idx1])
            idx3  =next(i for i,v in enumerate(minsidx[2]) if v>minsidx[0][idx1])
            idx,s =getidx(minsort[1][idx2:], minsort[2][idx3:],
                          minsidx[1][idx2:], minsidx[2][idx3:],
                          minrs[1][idx2:], minrs[2][idx3:])
            rgidx1=[minsidx[0][idx1]]+idx
            s1   +=s

        # assign minimum to second index
        idx2  =minsort[1].index(min(minsort[1]))
        s2    =minrs[1][idx2]
        if minsidx[1][idx2]<=min(minsidx[0])+2:
            rgidx2=[None,minsidx[1][idx2]]
        else:
            n     =len([t for t in minsidx[0] if t<minsidx[1][idx2]-2])
            idx1  =minsort[:n].index(min(minsort[:n]))
            rgidx2=[minsidx[0][idx1],minsidx[1][idx2]]
            s2   +=minrs[0][idx1]
            
        if minsidx[1][idx2]>max(minsidx[2])-2:
            rgidx2+=[None]
        else:
            idxt  =next(i for i,v in enumerate(minsidx[2]) if v>minsidx[1][idx2]+2)
            idx3  =minsort[2][idxt:].index(min(minsort[2][idxt:]))
            rgidx2+=[minsidx[2][idxt+idx3]]
            s2   +=minrs[2][idxt+idx3]

        # assign minimum to third index
        idx3      =minsort[2].index(min(minsort[2]))
        s3        =minrs[2][idx3]
        if minsidx[2][idx3]<=min(minsidx[0])+2:
            rgidx3=[None,None,minsidx[0][idx1]]
        elif minsidx[2][idx3]<=min(minsidx[1])+2:
            n     =len([t for t in minsidx[1] if t<minsidx[2][idx3]-2])
            idx2  =minsort[1][:n].index(min(minsort[1][:n]))
            rgidx3=[None,minsidx[1][idx2],minsidx[2][idx3]]
            s3   +=minrs[1][idx2]
        else:
            n1    =len([t for t in minsidx[0] if t<minsidx[2][idx3]-2])
            n2    =len([t for t in minsidx[1] if t<minsidx[2][idx3]-2])
            idx, s=getidx(minsort[0][:n1], minsort[1][:n2],
                          minsidx[0][:n1], minsidx[1][:n2],
                          minrs[0][:n1], minrs[1][:n2])
            rgidx3=idx+[minsidx[2][idx3]]
            s3   +=s

        s=[s1,s2,s3]
        rgidx=[rgidx1,rgidx2,rgidx3]
        l=[]
        for i in xrange(nmk):
            l.append(len([t for t in rgidx[i] if t is not None]))
        idx=[i for i in xrange(nmk) if l[i]==max(l)]
        if len(idx)==1:
            idx =rgidx[idx[0]]
        else:
            ss  =[s[i] for i in idx]
            idx2=ss.index(min(ss))
            idx =rgidx[idx[idx2]]
            
    return idx


def projection(basematrix,targetmatrix,n):
    """ Project 'targetmatrix' onto the space spanded by loadings of 'basematrix' to get
         the correlation curve to find whether there has common component between the
         two matrices. n is the number of components in basematrix.
    """
    nd= targetmatrix.shape[0]
    r = np.zeros(nd)
    v = np.linalg.svd(basematrix,full_matrices=0)[2]
    if n == 1:
        Y = targetmatrix-np.dot(v[0][:,None],np.dot(v[0],targetmatrix.T)[None,:]).T
    else:
        Y = targetmatrix-np.dot(v[:n].T,np.dot(v[:n],targetmatrix.T)).T
    
    for i in xrange(nd):
        r[i] = np.dot(targetmatrix[i],Y[i])/(np.linalg.norm(targetmatrix[i])*np.linalg.norm(Y[i]))
    return r


def mscc(Xtest,Xtarget,validrange):
    """ Multicomponent spectrum correlated chromatographic analysis (MSCC). If minimum
        correlative score obtained by Xtest is less than .3, the two curves should not be
        correlative. However, the information which indicates that the two matrices are
        not correlative should also be provided. Thus, an indicator of False is output
        for this.
        Input:
            Xtest       2-way data for testing
            Xtarget     2-way data as target
            validrange  range in the correlative curve of Xtarget
        Outpu:
            index of minimum correlative score if they are correlative and True, if not
            correlative, return the index as well but an indicator of False is also
            returned.
    """
    rbd=sirqr.poolsirqr(Xtest)
    n  =numcomp.sevcomp(Xtest)
    # score threshold
    thr=0.4
    
    # if no component is found, return
    if n == 0: return

    # calculate correlative curves and return index of curve minimum if satisfied setted
    # condition else return None
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
    if wid or not (r[validrange[0]:validrange[1]]<=thr).all():
        ## comparing to baseline corrected peaks
        r = projection(rbd,Xtarget,n)
        if (r[validrange[0]:validrange[1]]<=thr).all():
            yeah = True
    else:
        yeah = True

    nm =numcomp.sevcomp(Xtarget)
    rr =projection(Xtarget,rbd,min(nm,2))
    if yeah: # find the marker, yeah!
        rrs=np.where(rr<=thr)[0]
        if len(rrs)>=3:
            c = idxcheck(rbd,rr)
            return c, rr, True

    return np.argmin(rr), rr, False
