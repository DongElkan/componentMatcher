"""
Batch mode of componentMatcher (CM) to automatically process multiple or lots of HPLC-DAD
data stored in single directory and generate a report to show the results. Since it is not easy
to integrate this module into main GUI of CM, this module is run independently. You just click
the 'CMbatcher.py' file simply, specify the method and assign the directory to the module.
Note: Lib folder must be put in same directory with this module.
"""
import tkMessageBox as tkm
import tkFileDialog as tkd
import Tkinter as tk
import numpy as np
import os
import numcomp
import sirqr
import mscc
import CMio


CURRENTLIBPATH =os.path.join(os.getcwd(),'lib')         # Lib path
# Relative retention time tolerance to find maker peaks when reference
# marker is found
RRTERR         =0.09
# Allowed retention time tolerance: 5% in first 30 min and 10% in the latter time
RTERR          =[.05, .1]


def showerr(title,text):
    """ Show error but eliminate the root window """
    root = tk.Tk()
    root.withdraw()
    tkm.showerror(title,text)
    

def libcheck():
    """ Check whether any reference information occurs in the lib """
    methods=[]
    if not os.path.isdir(CURRENTLIBPATH):
        title = 'NO Lib Folder Found'
        text = 'No library folder is found! This is required!\nYou can construct a method using tool in CMmain.'
        showerr(title,text)
        return methods
    
    files=os.listdir(CURRENTLIBPATH)
    for name in files:
        if name.endswith('.info'):
            methods.append(name.split('.')[0])

    if len(methods) == 0:
        title = 'Method NOT Found'
        text = 'No reference method file is found! This is required!\nYou can construct a method using tool in CMmain.'
        showerr(title,text)
    return methods


def get_datadir():
    """ Get the directory in which data stores and check whether all data files are matched """
    root = tk.Tk()
    root.withdraw()
    datadir = tkd.askdirectory(initialdir=r'F:\data\LC\ITF')
    if len(datadir) == 0: return datadir

    files = os.listdir(datadir)
    if np.mod(len(files),3) != 0:
        tkm.showinfo('Data File NOT Match',
                                 'Raw Data, Time and Wavelength Files Must Be Matched!',
                                 icon='warning')
        datadir = ''
        return datadir

    n = int(len(files)/3)
    for i in xrange(n):
        wvfile = 'w'+files[i][1:]
        rtfile = 't'+files[i][1:]
        if not wvfile in files or not rtfile in files:
            tkm.showinfo('Data File NOT Match',
                                     'Raw Data, Time and Wavelength Files Must Be Matched!',
                                     icon='warning')
            datadir = ''
            break
    return datadir

    
def get_refinfo(method):
    """ Load reference data and corresponding information wavelength, retention
           time, marker retention time.
           Input:
               method: method string returned from 'get_method'
           Output:
               refdata:   reference HPLC-DAD data;
               markerrt: ranges of retention times of markers;
               refrt:        retention time of refdata;
    """
    refdatafile=os.path.join(CURRENTLIBPATH,method+'.txt')
    refinfofile=os.path.join(CURRENTLIBPATH,method+'.info')
    refdata = np.loadtxt(refdatafile)
    with open(refinfofile,'r') as f:
        for line in f:
            eqstr=line.find('=')
            if eqstr != -1:
                line=line.strip()
                if line.startswith('MARKERRT'):
                    t  = line[eqstr+1:].split(';')
                    rts= []
                    for s in t:
                        rts.append([float(v) for v in s.split(' ')])
                    markerrt  =rts
                elif line.startswith('REFMRT'):
                    string    =line[eqstr+1:]
                    try:
                        refmrt=float(string)
                    except ValueError:
                        refmrt=0.
                elif line.startswith('WVTHR'):
                    wvthr=int(line[eqstr+1:])
                elif line.startswith('TIME'):
                    rt    =np.array([float(t) for t in line[eqstr+1:].split(' ')])
                
    return refdata, markerrt, rt, refmrt, wvthr


def get_markeridx(targetidx, idx, rs, mkrt, stidx, ps, fps, l):
    """ Get new indices. Currently only 3 markers are considered. But if more markers exit,
        3 markers with lowest correlative scores are extracted and other markers are set
        to None. """
    if len(idx)>3:
        sx  =sorted(range(len(idx)), key=lambda k: min(rs[idx[k]]))
        sdx =[sx.index(j) for j in xrange(len(sx))] # sorted index
        tidx=[idx[sdx.index(i)] for i in xrange(3)] # get the lowest number
        for i in xrange(3,len(idx)):
            idxt                = idx(sdx.index(i))
            targetidx[idxt][1:] = [np.argmin(rs[idxt]), None]
        idx=tidx
    # get information for reassignment
    r     =[rs[idx[i]][:l[i]] for i in xrange(len(idx))]
    pmkrt =[mkrt[i] for i in idx]
    pstidx=[stidx[i] for i in idx]
    pps   =[ps[i] for i in idx]
    pfps  =[fps[i] for i in idx]
    nidx  =mscc.commonpeakcheck(r, pmkrt, pstidx, pps, pfps)

    for i in xrange(len(nidx)):
        if nidx[i] is None:
            targetidx[idx[i]][1:]=[np.argmin(r[i]), None]
        else:
            targetidx[idx[i]][1]=nidx[i]-pstidx[i]
    return targetidx


def markercheck(mkrt, targetidx, mrtidx, stidx, amt, fps, ps, rs, rt):
    """ Check whether markers are assigned to same peak in formula data. If so, the
        markers are checked and reordered to assign these markers to other peaks.

        Inputs:
            mkrt        marker retention times
            targetidx   marker information in formula data
            mrtidx      original indices of markers to reserve the order of marders
                        after putting reference marker to the first place
            stidx       start indices of markers in formula data
            amt         list of indicators to identify whether the assignment is
                        ambiguous
            fps         list of indicators to identify whether the peak has been assigned
                        to other markers, if so, the assignment is reserved
            ps          marker peak positions in formula data
            rs          correlative curves of markers
            rt          retention times of formula data

        Outputs:
            reordered peaks for markers """
    nmk=len(mkrt)
    # reorder the resulted indices to the original order
    if mrtidx[0] is not 1:
        targetidx.insert(mrtidx[0]-1,targetidx.pop(0))
        mkrt.insert(mrtidx[0]-1,mkrt.pop(0))
        stidx.insert(mrtidx[0]-1,stidx.pop(0))
        amt.insert(mrtidx[0]-1,amt.pop(0))
        fps.insert(mrtidx[0]-1,fps.pop(0))
        ps.insert(mrtidx[0]-1,ps.pop(0))
        rs.insert(mrtidx[0]-1,rs.pop(0))

    # assign peaks to indicators to reserve the peaks that have been assigned to
    # other markers having unique assignments
    for i in xrange(nmk):
        for j in xrange(nmk):
            if targetidx[i][-1] is not None and targetidx[j][-1] is not None and i!=j:
                idxt=targetidx[j][1]+stidx[j]-stidx[i]
                idx =[k for k in xrange(len(ps[i])) if ps[i][k][0]<=idxt<=ps[i][k][2]]
                for k in xrange(len(idx)):  fps[i][idx[k]]=True

    f =[True]*nmk
    for i in xrange(nmk-1):
        if stidx[i] is None or not f[i]: continue
        
        psite=[]
        if targetidx[i][2] is not None:
            psite+=[p for p in ps[i] if p[0]<=targetidx[i][1]<=p[2]]
            
        if len(psite)>0:
            psite=[psite[0][0],psite[0][2],psite[1][2]] if len(psite)>1 else psite[0]
            psite=[p+stidx[i] for p in psite]
            curridx=[i]
            for j in xrange(i+1,nmk):
                if targetidx[j][2] is not None:
                    if psite[0]<=targetidx[j][1]+stidx[j]<=psite[2]:
                        curridx.append(j)
            
            if len(curridx)>1:
                didx   =[curridx[i]-curridx[i-1] for i in xrange(1,len(curridx))]
                if didx.count(1)==len(didx): # if all are adjacent
                    ll =[len(rs[i]) for i in curridx]
                    targetidx=get_markeridx(targetidx,curridx,rs,mkrt,stidx,ps,fps,ll)
                else:
                    # if multiple markers are assigned to same peak, but another
                    # assignments exist between these markers
                    idxt =[j+1 for j in xrange(len(didx)) if didx[j]>1]
                    idxt+=[len(didx)]
                    for j in xrange(len(idxt)-1):
                        # get the index of inserted marker
                        idxi=curridx[idxt[j]-1]+1
                        rti =rt[targetidx[idxi][1]+stidx[idxi]]
                        if j is 0:
                            l=[]
                            for k in xrange(idxt[j]+1):
                                dr =np.where(np.sign(np.diff(rs[curridx[k]]))==1)[0]
                                l.append(len([t for t in dr if rt[t+stidx[curridx[k]]]<rti]))
                            targetidx=get_markeridx(targetidx,curridx[:idxt[j]],
                                                    rs,mkrt,stidx,ps,fps,l)
                        else:
                            for k in xrange(idxt[j]+1,idxt[j+1]+1):
                                dr =np.where(np.sign(np.diff(rs[curridx[k]]))==1)[0]
                                if rti>=rt[dr[-1]+stidx[curridx[k]]]:
                                    targetidx[curridx[k]][-1]=[None]
                                else:
                                    l=next(m for m,v in enumerate(dr)
                                           if rt[v+stidx[curridx[k]]]>rti)
                                    minidx=np.argmin(rs[curridx[k]][l:])
                                    if np.min(rs[curridx[k]][l:])<=.3:
                                        targetidx[curridx[k]][1]=l+minidx
                                    else:
                                        targetidx[curridx[k]][-1]=None

                for j in curridx:
                    amt[j]=True
                    f[j]  =False
        else:
            if targetidx[i][-1] is not None: targetidx[i][-1]=None

    return targetidx, stidx, amt


def batchanal(method,filename,datadir):
    """ Perform batch analysis. This is triggered after selecting
            reference method and take the method as the input"""
    # set up parameters
    datafiles = os.listdir(datadir)
    refdata, markerrt, refrt, refmrt, wvthr=get_refinfo(method)
    print 'Directory %s and method %s are selected for current analysis...' %\
          (datadir, method)
    nummk = len(markerrt)
    tn          = int(len(datafiles)/3)
    centrrt  = [] # marker retention times presented in report
    mkpeaks  = [] # marker peaks
    validrgs = [] # range in markers peaks to identify whether it is matched
    for time in markerrt:
        stidx  = np.where(refrt>=time[0])[0][0]
        mkpeak = refdata[(refrt>=time[0])&(refrt<=time[1])]
        midx   = np.argmax(np.max(mkpeak,axis=1))
        rg     = min(int(midx/4),int((len(mkpeak)-midx)/4))
        validrgs.append([midx-rg, midx+rg])
        mkpeaks.append(mkpeak)
        centrrt.append(refrt[stidx+midx])
    tcentrrt = centrrt[:]
    
    # resort the marker RT list to let the reference marker RT list at the first
    mrtidx     = [i+1 for i in xrange(nummk)] # indices for reserving initial order
    if refmrt>0. and nummk>1:
        i = [j for j in xrange(nummk) if markerrt[j][0]<=refmrt<=markerrt[j][1]][0]
        markerrt.insert(0, markerrt.pop(i))
        mkpeaks.insert(0, mkpeaks.pop(i))
        validrgs.insert(0, validrgs.pop(i))
        mrtidx.insert(0, mrtidx.pop(i))
        centrrt.insert(0, centrrt.pop(i))
        refmrt=centrrt[0]

    # analyzing each data file in selected directory using MSCC and writing
    # results to result.txt
    with open(filename,'w') as f:
        f.write('Method: %s\n' % method)
        f.write('Marker Retention Times: ')
        for i in xrange(nummk-1):
            f.write('%f\t' % tcentrrt[i])
        f.write('%f\n' % tcentrrt[-1])
        f.write('Reference Marker Retention Time: %f\n' % refmrt)
        f.write('Total Number of Raw Files:  %d\n' % tn)
        
        for i in xrange(tn):
            f.write('\n'+'-'*100+'\n')
            f.write('Raw Data: %s\n' % datafiles[i][1:])
            f.write('Marker ID'+'\t'*3+'Matched'+'\t'*3+'Raw RT/min'+'\t'*3+'Ref RT/min\n')
            # MSCC analysis
            rdata, rawrt, rwv=CMio.loadtxt(os.path.join(datadir,datafiles[i]),
                                           os.path.join(datadir,datafiles[i+tn]),
                                           os.path.join(datadir,datafiles[i+2*tn]))
            rdata=rdata[:, rwv>=wvthr]

            targetidx=[]
            ps       =[]
            fps      =[]
            rs       =[]
            amt      =[]
            stidx    =[]
            userrt   =False # indicator for using relative retention time
            for j in xrange(nummk):
                if markerrt[j][1]<=rawrt[-1]:
                    if j is 0: # the first marker
                        # current retention time
                        currrt  =rawrt*1.
                    if userrt:
                        cmrt =[t/refmrt for t in markerrt[j]]
                    else:
                        currerr =RTERR[0] if centrrt[j]<=30. else RTERR[1]
                        currerr*=centrrt[j]
                        cmrt    =markerrt[j]

                    startidx =np.where(currrt>=cmrt[0]-currerr)[0][0]
                    sel_data =rdata[(currrt>=cmrt[0]-currerr)&(currrt<=cmrt[1]+currerr)]
                    idx, r, tag =mscc.mscc(sel_data, mkpeaks[j], validrgs[j])

                    # svr curve and peak detection
                    s =numcomp.svr(sel_data)
                    ms=numcomp.smoother(s)
                    p =numcomp.peakdetect(ms)
                    ps.append(p)
                    fps.append([False]*len(p))
                    rs.append(r)
                    amt.append(False)
                    stidx.append(startidx)

                    targetidx.append([mrtidx[j], idx, None])
                    if tag:
                        targetidx[-1][-1] =np.max(sel_data[idx])
                        if j is 0 and refmrt>0:
                            currrt/=currrt[targetidx[0][1]+stidx[0]]
                            currerr=RRTERR
                            userrt =True
                else:
                    rs.append(0)
                    ps.append(0)
                    fps.append(0)
                    amt.append(False)
                    stidx.append(None)
                    targetidx.append([mrtidx[j], None, None])

            if nummk>1:
                targetidx, stidx, amt = markercheck(markerrt[:], targetidx, mrtidx,
                                                    stidx, amt, fps, ps, rs, rawrt)
            for j in xrange(nummk):
                t = '%d*' if amt[j] else '%d'
                if targetidx[j][2] is not None:
                    f.write((t+'\t'*5+'Y'+'\t'*4+'%.3f'+'\t'*4+'%.3f\n') %\
                            (j+1,rawrt[stidx[j]+targetidx[j][1]],tcentrrt[j]))
                else:
                    f.write((t+'\t'*5+'N'+'\t'*5+'-'+'\t'*4+'%.3f\n') % (j+1,tcentrrt[j]))
            print '%d of total %d files have been processed' % (i+1, tn)
    print 'All results have been saved to '+filename


class get_method():
    """ Select the reference method and perform batch analysis
           If the method input is not in the methods, return and reinput.
           Input:
                methods: method list in current library
            Output:
                return a method string, if any exception occurs, return None
    """
    def __init__(self,params):
        self.root = tk.Tk()
        self.root.withdraw()
        win = tk.Toplevel()
        win.title('Method Selection')
        win.bind('<Return>', self.get_entry)
        win.protocol("WM_DELETE_WINDOW", self.root.quit)
        
        self.refmethods = params['methods']
        self.datadir = params['datadir']
        # select method
        font = ('Times',10)
        txt = 'Methods allowed for selection:'
        n   = 0
        for method in params['methods']:
            n += 1
            txt += '\n'+str(n)+'. '+method
        tk.Label(win, text=txt,
                      justify='left', anchor='w',
                      font=font).pack(fill='x', expand=1, side='top')
        tk.Label(win,
                 text='Please input the method No. (an integer from %d to %d)' % (1,n),
                 font=font).pack(fill='x',expand=1,side='top')
        entrys = []
        ent = tk.Entry(win)
        ent.pack(fill='x', expand=1, side='top')
        entrys.append(ent)

        # input name of the result file, if is empty, the default name is "results"
        txt = 'Please input the name of the results file.'
        tk.Label(win, text=txt, anchor='w',
                      font=font).pack(fill='x', expand=1, side='top')
        txt = 'Note: If you leave it empty, default name is "results".\n' +\
              'But you should be cautious that the results file will be\n'+\
              'overwritten if you run the function multiple times and\n'+\
              'still leave it empty.'
        tk.Label(win, text=txt,
                      justify='left', anchor='w',
                      font=('Times',9), fg='red').pack(fill='x', expand=1, side='top')
        ent = tk.Entry(win)
        ent.pack(fill='x', expand=1, side='top')
        entrys.append(ent)
        
        # button for ok
        tk.Button(win, text='OK',
                  command=lambda:self.get_entry(None),
                  cursor='hand2').pack(side='top')
        # attributes
        self.win    = win
        self.entrys = entrys
        
        win.mainloop()
        
    def get_entry(self,event):
        """ Get the No. of input to select method """
        nn = False
        for ent in self.entrys:
            if len(ent.get().strip())>0:
                nn = True
        if not nn: return
        
        n = len(self.refmethods)
        val = self.entrys[0].get().strip()
        try:
            idx = int(val)
            if idx<1 or idx>n:
                title = 'Out of Range'
                text = 'Integer must be between 1 and '+str(n)+'!'
                showerr(title,text)
                return
        except ValueError:
            tkm.showerror('Invalid Input','Integer must be input for selecting method!')
            return

        name = self.entrys[1].get().strip()
        if len(name)==0:
            name = 'results'
        try:
            open(name,'w').close()
            os.remove(name)
        except IOError:
            tkm.showerror('Invalid Input','The file name is invalid, please check it!')
            return False

        self.win.destroy()
        self.root.quit()
        name += '.txt'
        self.method = self.refmethods[idx-1]
        self.name = name

    def params(self):
        """ Return the name and method obtained from the class """
        return self.name, self.method


def main():
    """ Main for CMbatcher """
    methods = libcheck()
    if len(methods) == 0: return
    datadir    = get_datadir()
    if len(datadir) == 0: return

    batchparams = {'methods': methods,
                                 'datadir': datadir}
    m = get_method(batchparams)
    
    try:
        name, method = m.params()
    except AttributeError:
        return
    
    # batch analyzing
    batchanal(method,name,datadir)
    

if __name__ == '__main__':
    main()
