"""
Set parameters for configuring figures in CMmain.
"""
import matplotlib.ticker as ticker 
import numpy as np


# get multiplier for the input data to scale data into -10-10 when plotting
def get_multiplier(ylims):
    """ Multiplier correction to set the range of plot
            Output:
                    mp      the 10 base exponential of y intensity
                    expstr  text locates at the top of y axis to show the 10 base exponential
                            to indicate the real extent of y intensity
    """
    # get exponential number
    ymin = ylims[0]
    ymax = ylims[1]
    m 	 = max(abs(ymax),abs(ymin))
    if m<1 and m is not 0:
        decimalnum = 0
        for i in str(m).split('.')[1]:
            decimalnum += 1
            if i is not '0': break
        expstr = r'$\times10^{-'+str(decimalnum)+'}$'
        # multiplier for Y axis value scaling
        mp 	   = 0.1**decimalnum
    elif m>1:
        decimalnum=len(str(m).split('.')[0])-1
        expstr 	  = r'$\times10$' if decimalnum==1 else r'$\times10^{'+str(decimalnum)+'}$'
        # multiplier for Y axis value scaling
        mp 		  = 10**decimalnum
    
    if decimalnum is 0: expstr = r''
    
    return mp,expstr


# set axis tick formats
def set_axticks(axis,xylims,expstr):
    """ Set tick formats of x and y axis to avoid Y tick label overflows when very large y
        values existed.
        Inputs:
            axis        axis of figure
            xylims      x and y axis limits
            expstr      text locates at the top of y axis to show the 10 base exponential
                        to indicate the real extent of y intensity
    """
    axis.set_xlim(xylims[0],xylims[1])
    axis.set_ylim(xylims[2],xylims[3])
    axis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    axis.text(-0.003,1,expstr,
                       transform=axis.transAxes,
                       size     =10)
    axis.tick_params(labelsize=10)


# set the parameters of figure for comfortable visualization
def set_figureparams(curves,rt,axes,texts):
    """ Set parameters of figure to define the figure in component matcher GUI.
		Input:
			curves	data curve for showing
			rt		retention time of the curve
			axes	figure axes where curves is shown
			texts	string list which will be displayed in the figure. Note that the
					last element in the list is the label used to indicate the source
					of the data. Valid values are 'ref' and 'raw' that indicate the
					data are from reference and raw data file.
    """
    # set up parameters for different figure
    figname = texts[2]
    if figname is 'raw':
        position=[0.05,0.56,0.93,0.4]
        fontsize=10
        color   =[u'#FFFACD',u'#6495ED']
        # text parameters for selecting wavelength
        x1,y1 	=0.8,0.93
        bbox 	={'fc': u'#3CB371', 'ec': u'#FF6347', 'boxstyle': 'round,pad=.3'}
        # text position for showing data information
        x2,y2 	=1.-0.007*len(texts[1]),1.02
        
    elif figname is 'ref': # for reference method plot
        position=[0.05,0.08,0.93,0.42]
        fontsize=12
        color 	=[u'#B22222',u'red']
        # text parameters for selecting wavelength
        x1,y1 	=0.84,0.93
        bbox 	={'fc': u'#FFF8DC', 'ec': u'#FF6347', 'boxstyle': 'round,pad=.2'}
        # text position for showing data information
        x2,y2 	=1.-0.00875*len(texts[1]),-0.13
        
    elif figname is 'rswv': # for single wavelength plot
        position=[0.082,0.15,0.88,0.77]
        # text parameters
        x1,y1   =0.64,0.93
        bbox    =dict(facecolor='none', edgecolor='none')
        color   =['r','w']
        fontsize=12
        x2,y2   =.9, -.15
    
    # get parameters
    ## for y axis, cut too high peaks at firt 5 min to enhance peaks in subsequent retention times
    yuplim = np.max(curves[rt>=10])
    if yuplim<np.max(curves[rt<=5]):
        yuplim*=1.3
    xylims     =[rt[0],rt[-1],np.min(curves),yuplim]
    mp,expstr =get_multiplier(xylims[2:])
    xylims[2:]=[i/mp for i in xylims[2:]]
    
    # set parameters
    axes.set_position(position)
    axes.format_coord = lambda x, y: ''
    set_axticks(axes,xylims,expstr)
    # set up text box for selecting wavelength for viewing single wavelength chromatogram
    text = axes.text(x1,y1,texts[0], fontname ='Times New Roman',
                           color    =color[0],
                           size     =fontsize,
                           transform=axes.transAxes,
                           bbox     =bbox,
                           label    =figname,
                           picker   =True)
    # set data information
    axes.text(x2,y2,texts[1], fontname ='Times New Roman',
                    transform=axes.transAxes,
                    size     =fontsize,
                    color    =color[1])
    return mp, yuplim, text


# set text coordinates
def set_textcoords(xaxis,ymaxaxis,yminaxis,axislim,textxys,xycoords,
                   textlen=0.05,
                   ylen   =0.05,
                   rxy    =[0.8,0.9]):
    """ Set x y coordinates of texts when labeling peaks to markers using greedy search procedure.
    
    Note:   Current parameter sets for defining a textbox, for example box
            width 'textlen' and box height 'ylen', is only suitable for text font
            size of 12. If figure size or font size is changed, the parameter
            should be changed correspondingly.
    
        Inputs:
                xaxis      values define x axis when plotting;
                yaxis      y curve or maximum value current if multiple curves are defined;
                axislim    limits of x and y axis;
                textxys    a list contains all text box coordinates if exist, if not, input emplty list [];
                xycoords   coordinates of data point to which text will be annotated;
                textlen    half length of marker text, default is 0.05;
                ylen       height of marker text, current is 0.04, 0.05 is used because if two
                           marker texts locate very closely in figure, 0.04 is not enough to
                           separate the two texts;
                rxy        reserved area for text for displaying single wavelength chromatogram
                           when user clicks the text;
        
        Output:
            coordinates of target text, x, y, and rad of fancy tail, r
        """
    xliml,xlimu =axislim[0]
    yliml,ylimu =axislim[1]
    xc,yc       =xycoords[0],xycoords[1]
    lx,ly   	=xlimu-xliml,ylimu-yliml          # length of x and y axis
    
    # find optimal site for locating text: the nearest location where current
    # text does not intersect with other texts or curves
    ## set x range
    xcp      =round((xc-xliml)/lx+0.01,5)
    rightxpercent=np.arange(xcp,1.-textlen*2,0.01)
    leftxpercent =np.arange(textlen*2,xcp-textlen,0.01)
    ## set y range, if is satisfied...
    ypp = (ylimu-yc)/ly
    if ypp >= 0.1:
        ypercent=[0.9] if ypp<=.15 else np.arange(round(1.-ypp,5)+.08,.9,.01)
        
        ## find optimal location
        def get_xy(site):
            """ Get coordinates of texts from the site side blank space """
            x, y    = [], []
            xpercent= leftxpercent[::-1] if site is 'l' else rightxpercent
            yaxis = ymaxaxis
            
            for xp in xpercent:
                xlow=(xp-textlen)*lx+xliml # left x coordinate of text box
                xup =(xp+textlen)*lx+xliml # right x coordinate of text box
                for yp in ypercent:
                    xs = xaxis[yaxis>=yliml+ly*yp]
                    if not any((xs>=xlow)&(xs<=xup)) and not ((xp+textlen>rxy[0])&(yp+ylen>rxy[1])):
                        x.append(xliml+xp*lx)
                        y.append(yliml+yp*ly)
                        # identify whether current text determined by xr and yr intersects with
                        # other texts
                        for t in textxys:
                            if abs(x[0]-t[0])<=textlen*lx and abs(y[0]-t[1])<=ylen*ly:
                                   x[:] = []
                                   y[:] = []
                                   break
                        if len(x)>0:
                            break
                if len(x)>0:
                    break
            return x, y

        # identify whether higher peak locates very clear to the marked site
        leftk =(ymaxaxis[(xaxis<xc)&(xaxis>xc-(.001+textlen/5.)*lx)]>yc).any()
        rightk=(ymaxaxis[(xaxis>xc)&(xaxis<xc+(.001+textlen/5.)*lx)]>yc).any()
        if leftk and not rightk: # higher peak at left side, not right side
            if len(rightxpercent) is 0: # if right side has no space, put to left side
                xl, yl = get_xy('l')
                if len(xl)>0: return xl[0], yl[0], -0.2, None, None
                xr, yr = [], []
            else:
                xr, yr = get_xy('r')
                if len(xr)>0: return xr[0], yr[0], 0.2, None, None
                xl, yl = [], []
        elif not leftk and rightk: # at right side, not left side
            if len(leftxpercent) is 0: # if right side has no space, put to left side
                xr, yr = get_xy('r')
                if len(xr)>0: return xr[0], yr[0], 0.2, None, None
                xl, yl = [], []
            else:
                xl, yl = get_xy('l')
                if len(xl)>0: return xl[0], yl[0], -0.2, None, None
                xr, yr = [], []
        else: # higher peak at both side or no higher peak adjacent
            xl, yl = get_xy('l')
            xr, yr = get_xy('r')        
        
        # set optimal outputs
        if len(xr) is not 0:
            x,y,r=xr[0],yr[0],0.2
            if len(xl) is not 0:
                dist_right=np.sqrt((x-xc)**2+(y-yc)**2)
                dist_left =np.sqrt((xl[0]-xc)**2+(yc-yl[0])**2)
                             
                if dist_left<dist_right:
                    x,y,r=xl[0],yl[0],-0.2
            return x,y,r, None, None

    # if the above procedure does not return the coordinate, the following procedure is
    # then performed to find blank space under the curves
    yc      =yminaxis[np.where(xaxis>=xc)[0][0]]
    ypp     =(yc-yliml)/ly
    ypercent=np.arange(ylen,round(ypp,5)-.08,.01)[::-1]
    
    def get_xy(site):
        """ Get coordinates of texts from the site side blank space """
        x, y =[], []
        yaxis,lcypp,lcyc = yminaxis,ypp,yc
        if site is 'l':
            xpercent = leftxpercent[::-1]
            # a is x, b is xc, c is current yc, to identify whether arrow intersects
            # with any curve
            f =lambda a, b, c: (yaxis[(yaxis>=a)&(yaxis<=b)]>c).all()
        else:
            xpercent= rightxpercent
            f = lambda a, b, c: (yaxis[(yaxis>=b)&(yaxis<=a)]>c).all()
        
        for xp in xpercent:
            xlow=(xp-textlen)*lx+xliml # left x coordinate of text box
            xup =(xp+textlen)*lx+xliml # right x coordinate of text box
            for yp in ypercent:
                yx = yaxis[(xaxis>=xlow)&(xaxis<=xup)]
                yt = yliml+yp*ly
                if (yx>yt+ylen).all():
                    x.append(xliml+xp*lx)
                    y.append(yt)
                    # identify whether current text determined by xr and yr intersects with
                    # other texts
                    for t in textxys:
                        if abs(x[0]-t[0])<=textlen*lx and abs(y[0]-t[1])<=ylen*ly:
                            x[:] = []
                            y[:] = []
                            break
                    
                    if len(x)>0:
                        # identify whether arrow intersects with any curve
                        if not f(x[0],xc,lcyc):
                            while not f(x[0],xc,lcyc):
                                lcyc -= .01*ly
                            # calculate the length of arrow
                            dist = np.sqrt((x[0]-xc)**2+(y[0]-lcyc)**2)
                            if dist < 0.3:
                                x[:] = []
                                y[:] = []
                        if len(x)>0: break
                    
            if len(x)>0:
                break
        return x, y, xc, lcyc
    
    xr, yr, xc, yc = get_xy('r')
    xl, yl, xc, yc = get_xy('l')
    # set optimal outputs
    if len(xl) is not 0 and len(xr) is 0:
        return xl[0],yl[0],-0.2, xc, yc
    elif len(xr) is not 0:
        x,y,r=xr[0],yr[0],0.2
        if len(xl) is not 0:
            dist_right=np.sqrt((x-xc)**2+(y-yc)**2)
            dist_left =np.sqrt((xl[0]-xc)**2+(yc-yl[0])**2)
                         
            if dist_left<dist_right:
                x,y,r=xl[0],yl[0],-0.2
        return x, y, r, xc, yc

    # if all the above procedures do not find any optimized coordinates, an arbitrary one
    # is adopted
    if len(rightxpercent)>0:
        x,y,r=rightxpercent[0]*lx+xliml,.9*ly+yliml,0.2
    else:
        x,y,r=leftxpercent[-1]*lx+xliml,.9*ly+yliml,0.2
    return x, y, r, None, None
        
