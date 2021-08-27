# This is the VBA code for using xlwings to call python.

Sub hi()
 
    RunPython ("from xw import say_hi; say_hi()")
 
End Sub