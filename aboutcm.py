from tkinter import Label, Toplevel, Button

"""
About componentMatcher.
"""

def aboutCM():
    font = ('Times', 10)
    text = '        componentMatch is a simple GUI package used for finding common\n'+\
           '        components from HPLC-DAD fingerprints of TCM compound and\n'    +\
           '        single medicine.\n\n'                                           +\
           '        Copyright (c) 2014 Nai-ping Dong and Mok Kam-wah Daniel\n\n'    +\
           '        Licensed under the terms of Apache License, version 2.0.\n'     +\
           '        All copyrights reserved.\n\n\n'                                 +\
           '        Contact:    np.dong572@gmail.com\n\n'                           +\
           '        Python 2.7.8, NumPy 1.9.0, matplotlib 1.4.0.'

    win = Toplevel()
    win.title('About componentMatcher')
    Label(win,
          text      =text,
          font      =font,
          justify   ='left',
          bd        =16,
          background='white',
          padx      =0,
          pady      =0).pack(expand=True,fill='both')
    win.resizable(width=False,height=False)
    Button(win,
           text             ='OK',
           font             =('Times',8,'bold'),
           fg               ='dark green',
           command          =win.destroy,
           width            =6,
           bd               =0.5,
           activebackground ='CadetBlue1',
           bg               ='cornsilk2').pack(side='right')
