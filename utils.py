import numpy as np

def time_encoder(time):
    coder = np.zeros(7,dtype=np.int32)
    for i in range(7):
        coder[i] = divmod(time,2**(6-i))[0]
        time -= coder[i]*2**(6-i)
    return coder

def processsa(input):
    #change state(int) to state(5*1)
    rtn = np.zeros(5, dtype=np.int32)
    rtn[input] = 1
    return rtn