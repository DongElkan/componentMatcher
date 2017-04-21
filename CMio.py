import numpy as np

def loadtxt(rdfile, tfile, wvfile):
    """ Load the raw text data. As the data converted from new version of software
        have additional encodings, the error will be raised after applying np.loadtxt
        directly. Thus, additional procedure is applied to decode the data.

        >>> d,t,w=loadtxt(rdfile, tfile, wvfile)

        Inputs:
            rdfile      file name of raw data
            tfile       file name of retention times
            wvfile      file name of wavelength

        Output:
            raw data, retention time and wavelength"""
    try:
        t =np.loadtxt(tfile)
        d = np.loadtxt(rdfile)
        wv=np.loadtxt(wvfile)
        return d, t, wv
    except ValueError:
        pass

    # get time data
    t =[]
    with open(tfile,'r') as f:
        for line in f:
            try:
                t.append(float(line.replace('\x00','').replace('\xff\xfe','').strip()))
            except:
                pass
    
    # get wavelength data
    wv=[]
    with open(wvfile,'r') as f:
        for line in f:
            try:
                wv.append(float(line.replace('\x00','').replace('\xff\xfe','').strip()))
            except:
                pass
    
    # get raw data
    d=np.zeros((len(t),len(wv)))
    r=len(t)-1
    n=-1
    with open(rdfile,'r') as f:
        for line in f:
            n+=1
            s =line.replace('\x00','').split()
            if n<=r:
                if n==0:
                    s[0]=s[0].replace('\xff\xfe','')
                x=np.array([float(i) for i in s])
                d[n]=x

    return d, np.array(t), np.array(wv)
