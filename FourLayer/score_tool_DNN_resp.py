import numpy as np
from enframe import enframe
import pdb
from polyarea import simpoly
import scipy.io as sio
avfilllength = 61
collar = 30
resp = 145*4

threshold_pass = 0
threshold_reject = 1

def calc_roc(D_val, epoch_map):
    D_val = np.asarray(D_val)
    ctr = 0
    sens = []
    spec = []
    prec = []
    prec2 = []
    
    float_values = [float(x)/10000 for x in range(0,10025,25)]    
 
    for value in float_values:
        globdecision_max = np.zeros((1,len(D_val[0,:])))
        for channel in range(8):
            decision = np.zeros((1,len(D_val[channel])))
            for idx in np.where(D_val[channel]>=1-value)[0]:
                decision[0,idx] = int(1)

            shiftmean = (11-1)/2
            my_dec = np.zeros((len(decision[0]) + shiftmean*2))
            pt1 = decision[:,:shiftmean]
            my_dec[:len(pt1[0])] = np.fliplr([pt1])[0]
            my_dec[len(pt1[0]):len(pt1[0])+ len(decision[0])] = decision[0]
            pt2 = decision[:,-shiftmean:]
            my_dec[len(pt1[0])+ len(decision[0]):] = np.fliplr([pt2])[0]

            decision = np.zeros((1,len(D_val[channel])))
            for zxc in range(1):
                aaa = enframe(np.array(my_dec), 11, 1)
                for idx in np.where(np.sum(aaa,1)>5)[0]:
                    decision[0,idx] = 1
            idxtmp = np.where(decision[0] ==1)[0]
            mymean = 0 
            if value<0.9:
                for kk  in range(len(idxtmp)):
                    if kk>0 and idxtmp[kk]-idxtmp[kk-1]==1:
                        mymean = mymean
                    else:
                        startx = idxtmp[kk]-resp
                        if startx<=0: 
                            startx = 0
                        rng = np.setdiff1d(range(startx,idxtmp[kk]),idxtmp)
                        if len(rng)<1:
                            mymean=0
                        else:
                            mymean = np.nanmean(D_val[channel,rng])
                    if mymean>0:
                        if not D_val[channel,idxtmp[kk]]+value-1*mymean>1:
                            decision[0,idxtmp[kk]] = 0 

            shiftmean = (11-1)/2 
            my_dec = np.zeros((len(decision[0]) + shiftmean*2))
            pt1 = decision[:,:shiftmean] 
            my_dec[:len(pt1[0])] = np.fliplr([pt1])[0]
            my_dec[len(pt1[0]):len(pt1[0])+ len(decision[0])] = decision[0]
            pt2 = decision[:,-shiftmean:]
            my_dec[len(pt1[0])+ len(decision[0]):] = np.fliplr([pt2])[0]

            decision = np.zeros((1,len(D_val[channel])))
            for zxc in range(1):
                aaa = enframe(np.array(my_dec), 11, 1)
                for idx in np.where(np.sum(aaa,1) > 5)[0]:
                    decision[0,idx] = 1

            globdecision_max = np.logical_or(globdecision_max,decision)

        tmp = np.where(np.array(globdecision_max[0]) == 1)
        finidx = []
        
        for kkk in range(collar+1):
            a =tmp[0]-kkk# [val-kkk for val in tmp[0]]
            b =tmp[0]+kkk# [val +kkk for val in tmp[0]]
            finidx.extend(a)          
            finidx.extend(b)
            finidx = list(set(finidx))
        negative_values = np.where(np.array(finidx) <0)
        for neg in negative_values[0]:
            #print(neg)
            finidx[neg] = 0
        positive_values = np.where(np.array(finidx) >= len(decision[0]))
        for pos in positive_values[0]:
            finidx[pos] = len(decision[0])-1
        for pred_idx in finidx:
            globdecision_max[0][pred_idx] = 1

        TN = 0
        FN = 0
        FP = 0
        TP = 0
        calc_vec1 = epoch_map[:len(epoch_map)/2] + globdecision_max[0,:len(epoch_map)/2]*2
        TN = TN + float(len(np.where(np.array(calc_vec1) == 0)[0])) 
        FN = FN + float(len(np.where(np.array(calc_vec1) == 1)[0])) 
        FP = FP + float(len(np.where(np.array(calc_vec1) == 2)[0])) 
        TP = TP + float(len(np.where(np.array(calc_vec1) == 3)[0]))        
        del calc_vec1
        calc_vec2 = epoch_map[len(epoch_map)/2:] + globdecision_max[0,len(epoch_map)/2:]*2
        
        TN = TN + float(len(np.where(np.array(calc_vec2) == 0)[0]))
        FN = FN + float(len(np.where(np.array(calc_vec2) == 1)[0]))
        FP = FP + float(len(np.where(np.array(calc_vec2) == 2)[0]))
        TP = TP + float(len(np.where(np.array(calc_vec2) == 3)[0]))
        del calc_vec2
        
        try:
            sens.append(float(TP/(TP + FN)))
        except ZeroDivisionError:
            sens.append(np.nan)
        try:            
            spec.append(float(TN/(TN + FP)))
        except ZeroDivisionError:
            spec.append(np.nan)
        try:
            prec.append(float(TP/(TP + FP)))
        except ZeroDivisionError:
            prec.append(np.nan)
        try:
            prec2.append(float(TN/(TN + FN)))
        except ZeroDivisionError:
            prec2.append(np.nan)
#        del TN, TP, FN, FP
        ctr = ctr +1
        
    x = [0, 1]
    x.extend(spec)
    x.extend([0])
    y = [0,0]
    y.extend(sens)
    y.extend([1])
    roc_area = simpoly(x, y)
    print("Test ROC area = %f \n" % (100*roc_area))

    Nfix=0.90
    N90 = np.where(np.asarray(spec)>Nfix)[0][-1]
    if N90 == 0:
        M90 = 0
        K90 = 0
    else:
        M90 = np.max(sens[:N90])
        K90 = np.max(spec[:N90])

    x = [Nfix, K90]
    x.extend(spec[:N90])
    x.extend([Nfix])

    y = [0,0]
    y.extend(sens[:N90])
    y.extend([M90])

    roc_area90 = simpoly(x,y)
 
    print("Test ROC90 area = %f \n" % (1000*roc_area90))

    return(roc_area, roc_area90)    
