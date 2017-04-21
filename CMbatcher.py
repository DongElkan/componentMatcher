"""
Batch mode of componentMatcher (CM) to automatically process multiple or lots of HPLC-DAD
data stored in single directory and generate a report to show the results. Since it is not easy
to integrate this module into main GUI of CM, this module is run independently. You just click
the 'CMbatcher.py' file simply, specify the method and assign the directory to the module.
Note: Lib folder must be put in same directory with this module.
"""
import tkinter.messagebox as tkm
import tkinter.filedialog as tkd
import tkinter as tk
import numpy as np
import os
import numcomp
import sirqr
import mscc


CURRENTLIBPATH =os.path.join(os.getcwd(),'lib')         # Lib path
RTERR   = 0.5                                 # Retention error to find maker peaks
MINWV = 240                               # Minimum wavelength to eliminate the influence of solvent absorption


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
    for i in range(n):
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
                    markerrt= rts
                elif line.startswith('TIME'):
                    refrt = np.array([float(t) for t in line[eqstr+1:].split(' ')])
                elif line.startswith('WAVELENGTH'):
                    refwv = np.array([float(wv) for wv in line[eqstr+1:].split(' ')])
    refdata = refdata[:,refwv>=MINWV]
    return refdata, markerrt, refrt


def batchanal(method,filename,datadir):
    """ Perform batch analysis. This is triggered after selecting
            reference method and take the method as the input"""
    # set up parameters
    datafiles = os.listdir(datadir)
    refdata, markerrt, refrt = get_refinfo(method)
    nummk = len(markerrt)
    tn          = int(len(datafiles)/3)
    centrrt  = []
    for time in markerrt:
        centrrt.append((time[0]+time[1])/2.)

    # analyzing each data file in selected directory using MSCC and writing results to result.txt
    with open(filename,'w') as f:
        f.write('Method: %s\n' % method)
        f.write('Marker Retention Times: ')
        for i in range(nummk-1):
            f.write('%f\t' % centrrt[i])
        f.write('%f\n' % centrrt[-1])
        f.write('Total Number of Raw Files:  %d\n' % tn)
        
        for i in range(tn):
            f.write('\n'+'-'*100+'\n')
            f.write('Raw Data: %s\n' % datafiles[i][1:])
            f.write('Marker ID'+'\t'*3+'Matched'+'\t'*3+'Raw RT/min'+'\t'*3+'Ref RT/min\n')
            # MSCC analysis
            rawdata =np.loadtxt(os.path.join(datadir,datafiles[i]))
            rawrt      =np.loadtxt(os.path.join(datadir,datafiles[i+tn]))
            rawwv   =np.loadtxt(os.path.join(datadir,datafiles[i+2*tn]))
            rawdata=rawdata[:,rawwv>=MINWV]

            for j in range(nummk):
                startidx =np.where(rawrt>=markerrt[j][0]-RTERR)[0][0]
                sel_data =rawdata[(rawrt>=markerrt[j][0]-RTERR)&(rawrt<=markerrt[j][1]+RTERR)]
                mkpeak   =refdata[(refrt>=markerrt[j][0])&(refrt<=markerrt[j][1])]
                midx      =np.argmax(np.max(mkpeak,axis=1))
                rg        =min(int(midx/4),int((len(mkpeak)-midx)/4))
                validrange=[midx-rg, midx+rg]
                targetidx =mscc.mscc(sel_data,mkpeak,validrange)

                # Write to file
                if targetidx:
                    f.write(('%d'+'\t'*5+'Y'+'\t'*4+'%.3f'+'\t'*4+'%.3f\n') %\
                            (j+1,rawrt[startidx+targetidx],centrrt[j]))
                else:
                    f.write(('%d'+'\t'*5+'N'+'\t'*5+'-'+'\t'*4+'%.3f\n') % (j+1,centrrt[j]))


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
