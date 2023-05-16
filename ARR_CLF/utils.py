from pickle import TRUE
import numpy as np
from matplotlib import pyplot as plt
import random
import os
from PIL import Image
import cv2

L = 256

def mask2time(mask):
    time_change = [0]
    for i in range(1,mask.shape[0]):
        if (mask[i] != mask[i-1]):
            time_change.append(i)
    time_change.append(mask.shape[0])
    return time_change

def find_wave_onset(wave_category: list) -> np.ndarray:
    onsets = []
    prev = 0
    for i, val in enumerate(wave_category):
        if val != 0 and prev == 0:
            onsets.append(i)
        prev = val
    return np.array(onsets)


def find_wave_offset(wave_category: list) -> np.ndarray:
    offsets = []
    prev = 0
    for i, val in enumerate(wave_category):
        if val == 0 and prev != 0:
            offsets.append(i)
        prev = val
    return np.array(offsets)

def find_wave(label):
    onset = find_wave_onset(label)
    offset = find_wave_offset(label)
    if len(onset) > len(offset):
        assert len(onset) == len(offset) + 1
        if onset[-1] == L-1:
            onset = np.delete(onset, -1)
        else:
            offset = np.append(offset, len(label)-1)
    assert len(onset) == len(offset)
    wave_info = np.vstack((onset, offset)).T
    wave_info = np.int32(wave_info)
    return wave_info

def trans_one_hot_label (label):
    num_classes = 4
    one_hot_codes = np.eye(num_classes)
    one_hot_label = np.zeros((label.shape[0],num_classes, label.shape[1]))
    
    for i in range(label.shape[0]):
        for j  in range(label.shape[1]):

            if (label[i, j] == 0):
                one_hot_label[i,:,j] = one_hot_codes[0]
            if (label[i, j] == 1):
                one_hot_label[i,:,j] = one_hot_codes[1]
            if (label[i, j] == 2):
                one_hot_label[i,:,j] = one_hot_codes[2]
            if (label[i, j] == 3):
                one_hot_label[i,:,j] = one_hot_codes[3]
    return one_hot_label

def determin_tpeak(start, finish, ecg, TH_max, TH_min):
    peak = np.max (ecg[start: finish])
    peak_loc = np.argmax(ecg[start: finish]) + start
    valley = np.min (ecg[start: finish])
    valley_loc = np.argmin(ecg[start: finish]) + start
    peak_rel = abs(peak-TH_max)
    valley_rel = abs(TH_min-valley)
    # print('peak_rel', peak_rel)
    # print('valley_rel', valley_rel)
    peak_in = True
    if peak_rel < 40 and valley_rel <40:

        if finish<len(ecg)-1:
            
            for j in range(finish, len(ecg)-1):
                
                if (ecg[j+1]-ecg[j])*(ecg[j]-ecg[j-1])<=0:
                    peak_in = False
                    break
            
            return j, peak_in
        else:
            return peak_loc, peak_in        
    else:
        if abs(peak-TH_max) >= abs(TH_min-valley):
            return peak_loc, peak_in
        else:
            return valley_loc, peak_in
            
def determin_ppeak(start, finish, ecg, TH_max, TH_min):
    peak = np.max (ecg[start: finish])
    peak_loc = np.argmax(ecg[start: finish]) + start
    valley = np.min (ecg[start: finish])
    valley_loc = np.argmin(ecg[start: finish]) + start
    peak_rel = abs(peak-TH_max)
    valley_rel = abs(TH_min-valley)
    # print('peak_rel', peak_rel)
    # print('valley_rel', valley_rel)
    peak_in = True
    if peak_rel < 10 and valley_rel <10:

        peak_in = False
        return peak_loc, peak_in

    else:
        if abs(peak-TH_max) >= abs(TH_min-valley):
            return peak_loc, peak_in
        else:
            return valley_loc, peak_in

def determin_peak(start, finish, ecg, TH_max, TH_min):
    peak = np.max (ecg[start: finish])
    peak_loc = np.argmax(ecg[start: finish]) + start
    valley = np.min (ecg[start: finish])
    valley_loc = np.argmin(ecg[start: finish]) + start
    peak_rel = abs(peak-TH_max)
    valley_rel = abs(TH_min-valley)

    if abs(peak-TH_max) >= abs(TH_min-valley):
        return peak_loc
    else:
        return valley_loc        


def cal_slope(x,i):
    slope = abs(((x[i+1]-x[i])- (x[i]-x[i-1])))
    return slope

def slope_max(ecg, loc, min_len):
    slope_max = 0
    slope_max_loc = loc
    if min_len>0:
        if loc -2 > 0 & loc+min_len+2<len(ecg)-1:
            for i in range(loc,loc+min_len+1):
                slope = cal_slope(ecg,i)
                print(slope)
                if slope > slope_max:
                    slope_max_loc = i
                    slope_max = slope
                else:
                    slope_max_loc = slope_max_loc
        else:
            # print(loc)
            slope_max_loc = loc

    return slope_max_loc


def cal_isoline(points, ecg):
    if (points[-1] != -3000) and (points[5] != -3000):

        iso_line = np.median(ecg[int(points[6])] -  ecg[int(points[5])])
    elif points[5] != -3000:
        iso_line = 0
    else:
        iso_line = 0
    return iso_line



def plot_with_points(ecg, label, points, index,id, label_duiqi, predict):
    time_change = mask2time(label)
    
    # print(time_change)
    fig = plt.figure(figsize=(28, 20))
    ax = plt.subplot(211)
    ax.plot(ecg)
    for i in range(1,len(time_change)):
        if label[time_change[i]-1] == 0:
            # print('black',i)
            seg = np.arange(time_change[i-1],time_change[i])
            ax.plot(seg,ecg[seg], color= 'black',linewidth=5)

        elif label[time_change[i]-1] == 1:
            seg = np.arange(time_change[i-1],time_change[i])
            ax.plot(seg,ecg[seg], color= 'red',linewidth = 5)
            # print('red',i)
        elif label[time_change[i]-1] == 2:
            seg = np.arange(time_change[i-1],time_change[i])
            ax.plot(seg,ecg[seg], color= 'green',linewidth = 5)
            # print('green',i)
        elif label[time_change[i]-1] == 3:
            seg = np.arange(time_change[i-1],time_change[i])
            ax.plot(seg,ecg[seg], color = 'blue',linewidth = 5)            
            # print('blue',i)
    ax2 = plt.subplot(212)
    mark = ['o', '*','o', 'o','o', '*','o','o','o', '*','o']
    col = ['g','g','g','r','r','r','r','r','b','b','b']

    ax2.plot(ecg,linewidth=5 )
    for i in range(len(points)):
        # print(points)
        if points[i] != -3000:
            point = int(points[i])
            # print(point)
            ax2.scatter(points[i], ecg[point], marker=mark[i], c = col[i], s= 600)
    folder_path =  '/Volumes/xxz/mitbih_with_points/'       
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(folder_path+'/'+str(index)+'_'+str(id)+'label'+str(label_duiqi)+'predict'+str(predict)+'.png')
    plt.close()
    # plt.show()

def write_image(image, filename):
    folder_path =  './mitbih_binary_images/'       
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    img = image* 255
    img = np.uint8(img)
    im = Image.fromarray(img, mode = 'L')
    print('save to',folder_path+filename )
    im.save(folder_path+filename, "png")


def candidate(ecg, cos_thres):
    candidates = []
    for i in range(2, len(ecg)-1):
        a = ecg[i] - ecg[i-1]
        b = ecg[i] - ecg[i+1]
        if (a*b - 1 < 0):
            a_mo = np.sqrt(1+a^2)
            b_mo = np.sqrt(1+b^2)
            if ((a * b - 1)/(a_mo * b_mo) > cos_thres):
                candidates.append(i)
            else:
                pass
        elif (a*b - 1 > 0):
            candidates.append(i)
        else:
            if (a == 0) or (b == 0): # 90
                candidates.append(i)
            else: #180
                pass
    return candidates

def turning_point(ecg):
    turning_points = []
    turning_points_dir = []
    for i in range(1, len(ecg)-1):
        if (ecg[i] > ecg[i-1]) and (ecg[i] > ecg[i+1]):
            turning_points.append(i)
            turning_points_dir.append(1)
        elif (ecg[i] < ecg[i-1]) and (ecg[i] < ecg[i+1]):
            turning_points.append(i)
            turning_points_dir.append(-1)
    return turning_points_dir, turning_points

def up_down_flat(a, b, th):
    if (a - b) > th:
        return 'down'
    elif (a - b) < -th:
        return 'up'
    else:
        return 'flat'

def trend(ecg, up_down_th):
    trend_dict ={
        'flat_flat': 0,
        'flat_up': 1,
        'flat_down': 2,
        'up_flat': 3,
        'up_up': 4,
        'up_down': 5,
        'down_flat': 6,
        'down_up': 7,
        'down_down': 8,
    }
    trends = []
    for i in range(1, len(ecg)-1):
        mark_1 = up_down_flat(ecg[i-1], ecg[i], up_down_th)
        mark_2 = up_down_flat(ecg[i], ecg[i+1], up_down_th)
        mark = mark_1 + '_' + mark_2
        trends.append(trend_dict.get(mark))
    return trends
    


def determine_r(q_on, s_off, ecg):
    trend_dir = trend(ecg[q_on-1: s_off + 2],up_down_th = 4 )
    

    r_temp = -3000
    r_loc_temp = -3000
    for i in range(len(trend_dir)):
        if trend_dir[i] == 5:
            if ecg[q_on + i]> r_temp:
                r_temp = ecg[q_on + i]
                r_loc_temp = q_on + i
            else:
                r_temp = r_temp

    for i in range(len(trend_dir)-1):
        if ((trend_dir[i] == 3) and (trend_dir[i+1] == 2)):
            if ecg[q_on + i]> r_temp:
                r_temp = ecg[q_on + i]
                r_loc_temp = q_on + i
            else:
                r_temp = r_temp

    for i in range(len(trend_dir)-2):
        if ((trend_dir[i] == 3) and (trend_dir[i+1] == 0) and (trend_dir[i+2] == 2)):
            if ecg[q_on + i]> r_temp:
                r_temp = ecg[q_on + i]
                r_loc_temp = q_on + i
            else:
                r_temp = r_temp
    # if r_temp == -3000:
    #     print(trend_dir)

    r = r_loc_temp

    return r

def pos_on_off(pos_loc_temp,on,off,r,trend_dir, ecg,is_t ):
    on_m = -3000
    off_m = -3000    
    for i in range(pos_loc_temp, off-on):
        if (trend_dir[i] == 6) and (i > pos_loc_temp -on -5):
            off_m = i + on
            break

    for i in range(pos_loc_temp-on,0, -1):
        if (trend_dir[i] == 1 or trend_dir[i] == 7 )  and (i < pos_loc_temp-on-3):
            on_m = on + i
            break
    if (off_m == -3000):
        if (is_t):

            trend_dir_temp = trend(ecg[off:off+16],1)
            if (len(trend_dir_temp)):
                for i in range(len(trend_dir_temp)):
                    if trend_dir_temp[i] == 6:
                        off_m = i + off
                        break   
        else:
            trend_dir_temp = trend(ecg[off:r],2)
            if(len(trend_dir_temp)):
                for i in range(len(trend_dir_temp)):
                    if trend_dir_temp[i] == 6:
                        off_m = i + off
                        break

    if (on_m == -3000):

        trend_dir_temp = trend(ecg[on-10:on],1)

        if (len(trend_dir_temp)):
            for i in range(len(trend_dir_temp)-1,0,-1):

                if (trend_dir_temp[i] == 1 or trend_dir_temp[i] == 7 ) :
                    on_m = on -10 + i
                    break

    if (on_m == -3000):
        on_m = on
        # print('pos_t: t_on not found',trend_dir)
    if (off_m == -3000):
        off_m = off
        # print('pos_t: t_off not found',trend_dir) 
    return on_m, off_m

def neg_on_off(neg_loc_temp,on,off,r,trend_dir, ecg,is_t):
    on_m = -3000
    off_m = -3000
    for i in range(neg_loc_temp-on, off-on):
        if (trend_dir[i] == 3) and (i > neg_loc_temp -on -5):
            
            off_m = i + on
            break

    for i in range(neg_loc_temp-on,0, -1):
        if (trend_dir[i] == 2 or trend_dir[i] == 5)and (i < neg_loc_temp-on-3):

            on_m = on +i
            break
    if (off_m == -3000):
        if (is_t):
            trend_dir_temp = trend(ecg[off:off+16],1)
            # print(len(trend_dir_temp))
            if (len(trend_dir_temp)):
                for i in range(len(trend_dir_temp)):
                    # print(i)
                    if trend_dir_temp[i] == 3:
                        off_m = i + off
                        break
        else:
            trend_dir_temp = trend(ecg[off:r],2)
            # print(trend_dir_temp)
            if (len(trend_dir_temp)):
                for i in range(len(trend_dir_temp)):
                    if trend_dir_temp[i] == 3:
                        off_m = i + off
                        break                            
    if (on_m == -3000):
        trend_dir_temp = trend(ecg[on-10:on],1)
        if (len(trend_dir_temp)):
            for i in range(len(trend_dir_temp)-1,0,-1):
                if trend_dir_temp[i] == 2 or trend_dir_temp[i] == 5:
                    on_m = on + i -10
                    break

    if (on_m == -3000):
        on_m = on
        # print('pos_t: t_on not found',trend_dir)
    if (off_m == -3000):
        off_m = off
        # print('pos_t: t_off not found',trend_dir) 

    return on_m, off_m



def determine_t(t_on, t_off,r, ecg):
    t_dir = 0
    trend_dir = trend(ecg[t_on-1: t_off + 2],up_down_th = 1 )
    # print(trend_dir)
    t_pos_temp = -3000
    t_pos_loc_temp = -3000
    t_dir = 0
    if len(trend_dir) > 5:
        for i in range(len(trend_dir)-5):
            if (trend_dir[i] == 5) or ((trend_dir[i] == 3) and (trend_dir[i+1] == 2)) or ((trend_dir[i] == 3) and (trend_dir[i+1] == 0) and (trend_dir[i+2] == 2)) or ((trend_dir[i] == 3) and (trend_dir[i+1] == 0) and (trend_dir[i+2] == 0)  and (trend_dir[i+3] == 2)) or ((trend_dir[i] == 3) and (trend_dir[i+1] == 0) and (trend_dir[i+2] == 0)and (trend_dir[i+3] == 0)  and (trend_dir[i+4] == 2)) or ((trend_dir[i] == 3) and (trend_dir[i+1] == 0)and (trend_dir[i+2] == 0) and (trend_dir[i+3] == 0) and (trend_dir[i+4] == 0)  and (trend_dir[i+5] == 2)):
                t_pos_temp = ecg[t_on + i]
                t_pos_loc_temp = t_on + i
            else:
                t_pos_temp = t_pos_temp 
    else:
        for i in range(len(trend_dir)-1):
            if (trend_dir[i] == 5) or ((trend_dir[i] == 3) and (trend_dir[i+1] == 2)):
                t_pos_temp = ecg[t_on + i]
                t_pos_loc_temp = t_on + i
            else:
                t_pos_temp = t_pos_temp 
    t_neg_temp = 3000
    t_neg_loc_temp = 3000
    if len(trend_dir) > 5:
        for i in range(len(trend_dir)-5):
            if (trend_dir[i] == 7) or ((trend_dir[i] == 6) and (trend_dir[i+1] == 1))  or ((trend_dir[i] == 6) and (trend_dir[i+1] == 0) and (trend_dir[i+2] == 1))or ((trend_dir[i] == 6) and (trend_dir[i+1] == 0) and (trend_dir[i+2] == 0) and (trend_dir[i+3] == 1))or ((trend_dir[i] == 6) and (trend_dir[i+1] == 0)and (trend_dir[i+2] == 0) and (trend_dir[i+3] == 0)  and (trend_dir[i+4] == 1))or ((trend_dir[i] == 6) and (trend_dir[i+1] == 0) and (trend_dir[i+2] == 0) and (trend_dir[i+3] == 0) and (trend_dir[i+4] == 0) and (trend_dir[i+5] == 1)):
                if ecg[t_on + i]< t_neg_temp:
                    t_neg_temp = ecg[t_on + i]
                    t_neg_loc_temp = t_on + i
                else:
                    t_neg_temp = t_neg_temp
    else:
        for i in range(len(trend_dir)-1):
            if (trend_dir[i] == 7) or ((trend_dir[i] == 6) and (trend_dir[i+1] == 1)):
                t_pos_temp = ecg[t_on + i]
                t_pos_loc_temp = t_on + i
            else:
                t_pos_temp = t_pos_temp         
    if t_neg_temp == 3000:
        pass
        # print(trend_dir)    



    trend_dir = trend(ecg[t_on-1: t_off + 2],up_down_th = 0 )
    if ((t_pos_temp == -3000) and (t_neg_temp == 3000)):
        print('pos_t and neg_t is not found',trend_dir)
        t_on_m = t_on
        t = np.argmax(ecg[t_on: t_off]) + t_on
        t_off_m = t_off 
    elif ((t_pos_temp != -3000) and (t_neg_temp == 3000)):
        t = t_pos_loc_temp
        t_dir = 1
        t_on_m,t_off_m = pos_on_off(t_pos_loc_temp,t_on,t_off,r,trend_dir, ecg,  is_t = True)
        

     

    elif ((t_pos_temp == -3000) and (t_neg_temp != 3000)):
        t = t_neg_loc_temp
        t_dir = -1
        t_on_m,t_off_m = neg_on_off(t_neg_loc_temp,t_on,t_off,r,trend_dir, ecg, is_t = True)
  
    else:
        
        if (ecg[t_pos_loc_temp] - ecg[t_off])   >= (ecg[t_off] - ecg[t_neg_loc_temp])+10:
            t = t_pos_loc_temp
            t_dir = 1
            t_on_m,t_off_m = pos_on_off(t_pos_loc_temp,t_on,t_off,r,trend_dir, ecg, is_t = True)

 
        else:
            t = t_neg_loc_temp
            t_dir = -1
            t_on_m,t_off_m = neg_on_off(t_neg_loc_temp,t_on,t_off,r,trend_dir, ecg, is_t = True)   
    # print(t_on_m, t, t_off_m)
    return t_on_m, t, t_off_m, t_dir


def determine_p(p_on, p_off,r, ecg):
    p_dir = 0
    trend_dir = trend(ecg[p_on-1: p_off + 2],up_down_th = 0 )
    # print(trend_dir)
    p_pos_temp = -3000
    p_pos_loc_temp = -3000
    if len(trend_dir) > 5:
        for i in range(len(trend_dir)-5):
            if (trend_dir[i] == 5) or ((trend_dir[i] == 3) and (trend_dir[i+1] == 2)) or ((trend_dir[i] == 3) and (trend_dir[i+1] == 0) and (trend_dir[i+2] == 2)) or ((trend_dir[i] == 3) and (trend_dir[i+1] == 0) and (trend_dir[i+2] == 0)  and (trend_dir[i+3] == 2)) or ((trend_dir[i] == 3) and (trend_dir[i+1] == 0) and (trend_dir[i+2] == 0)and (trend_dir[i+3] == 0)  and (trend_dir[i+4] == 2)) or ((trend_dir[i] == 3) and (trend_dir[i+1] == 0)and (trend_dir[i+2] == 0) and (trend_dir[i+3] == 0) and (trend_dir[i+4] == 0)  and (trend_dir[i+5] == 2)):
                p_pos_temp = ecg[p_on + i]
                p_pos_loc_temp = p_on + i
            else:
                p_pos_temp = p_pos_temp 
    else:
        for i in range(len(trend_dir)-1):
            if (trend_dir[i] == 5) or ((trend_dir[i] == 3) and (trend_dir[i+1] == 2)):
                p_pos_temp = ecg[p_on + i]
                p_pos_loc_temp = p_on + i
            else:
                p_pos_temp = p_pos_temp 
    p_neg_temp = 3000
    p_neg_loc_temp = 3000
    if len(trend_dir) > 5:
        for i in range(len(trend_dir)-5):
            if (trend_dir[i] == 7) or ((trend_dir[i] == 6) and (trend_dir[i+1] == 1))  or ((trend_dir[i] == 6) and (trend_dir[i+1] == 0) and (trend_dir[i+2] == 1))or ((trend_dir[i] == 6) and (trend_dir[i+1] == 0) and (trend_dir[i+2] == 0) and (trend_dir[i+3] == 1))or ((trend_dir[i] == 6) and (trend_dir[i+1] == 0)and (trend_dir[i+2] == 0) and (trend_dir[i+3] == 0)  and (trend_dir[i+4] == 1))or ((trend_dir[i] == 6) and (trend_dir[i+1] == 0) and (trend_dir[i+2] == 0) and (trend_dir[i+3] == 0) and (trend_dir[i+4] == 0) and (trend_dir[i+5] == 1)):
                if ecg[p_on + i]< p_neg_temp:
                    p_neg_temp = ecg[p_on + i]
                    p_neg_loc_temp = p_on + i
                else:
                    p_neg_temp = p_neg_temp
    else:
        for i in range(len(trend_dir)-1):
            if (trend_dir[i] == 7) or ((trend_dir[i] == 6) and (trend_dir[i+1] == 1)):
                p_pos_temp = ecg[p_on + i]
                p_pos_loc_temp = p_on + i
            else:
                p_pos_temp = p_pos_temp         
    if p_neg_temp == 3000:
        pass
        # print(trend_dir) 
   
    # print(p_pos_temp, p_neg_temp)
    # print(p_pos_loc_temp, p_neg_loc_temp)

    trend_dir = trend(ecg[p_on-1: p_off + 2],up_down_th = 0 )
    if ((p_pos_temp == -3000) and (p_neg_temp == 3000)):
        print('pos_t and neg_t is not found',trend_dir)
        p_on_m = p_on
        p = np.argmax(ecg[p_on: p_off]) + p_on
        p_off_m = p_off 
    elif ((p_pos_temp != -3000) and (p_neg_temp == 3000)):
        p = p_pos_loc_temp
        p_dir = 1
        p_on_m,p_off_m = pos_on_off(p_pos_loc_temp,p_on,p_off,r,trend_dir, ecg, is_t=False)
        

     

    elif ((p_pos_temp == -3000) and (p_neg_temp != 3000)):
        p = p_neg_loc_temp
        p_dir = -1
        p_on_m,p_off_m = neg_on_off(p_neg_loc_temp,p_on,p_off,r,trend_dir, ecg, is_t=False)
  
    else:
        
        if (ecg[p_pos_loc_temp] - ecg[p_off]) >= (ecg[p_off] - ecg[p_neg_loc_temp])+10:
            p = p_pos_loc_temp
            p_dir = 1
            p_on_m,p_off_m = pos_on_off(p_pos_loc_temp,p_on,p_off,r,trend_dir, ecg, is_t=False)

 
        else:
            p = p_neg_loc_temp
            p_dir = -1
            p_on_m,p_off_m = neg_on_off(p_neg_loc_temp,p_on,p_off,r,trend_dir, ecg, is_t=False)   
    # print(p_on_m, p, p_off_m)
    return p_on_m, p, p_off_m, p_dir   
    
          
def determine_q(p_off, q_on, r, ecg):
    q_loc = q_on
    pq_loc = q_on 

    if ((p_off != -3000) and (r != -3000)):
        # print(ecg[p_off-1: r + 2])
        trend_dir = trend(ecg[p_off-1: r + 2],up_down_th = 1 )
        # print(trend_dir)
        for i in range(len(trend_dir)-1): 
            if (trend_dir[i] == 2) or (trend_dir[i] == 5): # flat_down/up_down
                pq_loc = i + p_off
                break


        for i in range(len(trend_dir)-1,0,-1):
            if (trend_dir[i] == 7) or (trend_dir[i] == 6): #down_up/
                q_loc = i + p_off
                break

    elif ((p_off == -3000) and (r != -3000)):
        trend_dir = trend(ecg[r - 15: r + 2],up_down_th = 1 )
        for i in range(len(trend_dir)-2,0,-1):
            if (trend_dir[i] == 7) or (trend_dir[i] == 6): #down_up/
                q_loc = i + r - 15
                continue

            if (trend_dir[i] == 2) or (trend_dir[i] == 5): # flat_down/up_down
                pq_loc = i + r - 15  
                break
           
    else:
        pq_loc = -3000
        q_loc = -3000
    # print(pq_loc, q_loc)


    return pq_loc, q_loc

def determine_s(s_off,t_on, r, ecg):
    s_loc = -3000
    st_loc = -3000

    if ((t_on != -3000) and (r != -3000)):
        # print(ecg[p_off-1: r + 2])
        trend_dir = trend(ecg[r: t_on],up_down_th = 10 )
        # print(trend_dir)
        for i in range(len(trend_dir)-1): 
            if (trend_dir[i] == 7) or (trend_dir[i] == 6): # down_flat/down_up
                s_loc = i + r
                break              


        for i in range(len(trend_dir)-1,0,-1):
            if (trend_dir[i] == 3) or (trend_dir[i] == 5): #up_flat/up_down
                st_loc = i + r
                break
        if s_loc != -3000 and st_loc == -3000:
            st_loc = s_loc
        elif s_loc == -3000 and st_loc != -3000:
            s_loc = s_off      
        else:
            s_loc = s_off
            st_loc = s_off


    elif ((t_on == -3000) and (r != -3000)):
        trend_dir = trend(ecg[r: r + 30],up_down_th = 8 )
        for i in range(0,len(trend_dir)-2):
            if (trend_dir[i] == 7) or (trend_dir[i] == 6): #down_up/down_flat 
                s_loc = i + r
                continue

            if (trend_dir[i] == 3) or (trend_dir[i] == 5): # up_flat/up_down
                st_loc = i + r  
                break

        if s_loc != -3000 and st_loc == -3000:
            st_loc = s_loc
        elif s_loc == -3000 and st_loc != -3000:
            s_loc = s_off      
        else:
            s_loc = s_off
            st_loc = s_off       
    else:
        s_loc = -3000
        st_loc = -3000
    




    # print(pq_loc, q_loc)
    return s_loc, st_loc

def fiducial_points_vary(label,ecg, id):
    
    if label.shape[1] != 4:
        label_onehot = trans_one_hot_label(label)
    # print(ecg.shape)
    B, L = label_onehot.shape[0], label_onehot.shape[2]
    if len(ecg.shape) == 3:
        ecg = ecg[:,:,0]
    ecg = np.squeeze(ecg)
    
    points = np.zeros((B, 11))
    dirs = np.zeros((B, 2))
    iso_line = np.zeros((B, 1))
    for b in range(B):
    # for b in [25]:
        # candidates = candidate(ecg[b,:], -0.5)
        
        qrs_info = find_wave(label_onehot[b,2,:])
        # print('qrs_info',qrs_info)
        p_info = find_wave(label_onehot[b,1,:])
        # print('p_info',p_info)
        t_info = find_wave(label_onehot[b,3,:])
        # print('t_info',t_info)

        if len(qrs_info) != 0:
            for qrs_i in range(qrs_info.shape[0]):

                if (((106)<= qrs_info[qrs_i, 1]) and ((106)>= qrs_info[qrs_i, 0])):
                    # print('o',qrs_info[qrs_i, 0])
                    # print('1',qrs_info[qrs_i, 1])                    
                    q_on = qrs_info[qrs_i, 0]
                    
                    # q_on = slope_max(ecg[b,:], q_on, 5)
                    s_off = qrs_info[qrs_i, 1]
                    # s_off = slope_max(ecg[b,:], s_off, 20)
                    # print('q_on', q_on)
                    # print('s_off',s_off)
                    ############################# trt -> r #################################
                    if (len(t_info)!= 0):
                        delete = []
                        for k in range(len(t_info)):

                            if (qrs_info[qrs_i, 0]== t_info[k,1]) and (t_info[k,1]-t_info[k,0]<15):
                                q_on = t_info[k,0]
                                delete.append(k)
                                # t_info = np.delete(t_info, k, axis = 0)
                                continue
                            if (qrs_info[qrs_i, 1] == t_info[k,0]) and (t_info[k,1]-t_info[k,0]<15):
                                s_off = t_info[k, 1]
                                delete.append(k)
                                # t_info = np.delete(t_info, k, axis = 0)
                                continue
                        t_info = np.delete(t_info, delete, axis = 0)
                    ########################################################################
                    break
                            
                
                else:
                    dis = np.min(np.abs(qrs_info - L//2), axis=1 )
                    qrs_i = np.argmin(dis)
                    q_on = qrs_info[qrs_i, 0]
                    s_off = qrs_info[qrs_i, 1] 
                    
                    ############################# trt -> r #################################
                    if (len(t_info)!= 0):
                        delete = []
                        for k in range(len(t_info)):

                            if (qrs_info[qrs_i, 0]== t_info[k,1]) and (t_info[k,1]-t_info[k,0]<15):
                                
                                q_on = t_info[k,0]
                                delete.append(k)
                                # t_info = np.delete(t_info, k, axis = 0)
                                continue
                            if (qrs_info[qrs_i, 1] == t_info[k,0]) and (t_info[k,1]-t_info[k,0]<15):
                                print(t_info[k,1])
                                s_off = t_info[k, 1]
                                delete.append(k)
                                # t_info = np.delete(t_info, k, axis = 0)
                                continue
                        t_info = np.delete(t_info, delete, axis = 0)
                    ########################################################################
                    break
            # print(q_on, s_off)
                    # s_off = slope_max(ecg[b,:], s_off, 20)
            # r = determin_peak(q_on, s_off, ecg[b,:], TH_max = max(ecg[b,q_on], ecg[b,s_off]) , TH_min= min(ecg[b,q_on], ecg[b,s_off]))
            r = determine_r(q_on, s_off, ecg[b,:])
            
            if len(t_info)!= 0:
                t_i = np.argmin(np.abs(np.min((t_info - s_off), axis = 1)))
                t_on = t_info[t_i, 0]
                # t_on = slope_max(ecg[b,:], t_on, 5)
                t_off = t_info[t_i, 1]
                # t_off = slope_max(ecg[b,:], t_off, 5)


                if (t_on - s_off) > 150 or (t_on < s_off):
                    t_on = -3000
                    t_off = -3000
                    t = -3000
                    t_dir = 0
                    print(b, 't is detected but not in the right place')
                else:
                    
                    # t, peak_in = determin_tpeak(t_on, t_off, ecg[b,:], TH_max = max(ecg[b,t_on],ecg[b,t_off]), TH_min= min(ecg[b,t_on],ecg[b,t_off]))
                    
                    # if not peak_in:
                    #     t_off= min(t + 15, L-1)
                    t_on, t, t_off,t_dir = determine_t(t_on, t_off,r, ecg[b,:])
            else:
                t_on = -3000
                t_off = -3000
                t = -3000
                t_dir = 0
                print(b, 'no t is detected')

            if len(p_info) != 0:
                
                p_i = np.argmin(np.abs(np.min((p_info - q_on), axis = 1)))
                p_on = p_info[p_i, 0]
                # p_on = slope_max(ecg[b,:], p_on, 5)
                p_off = p_info[p_i, 1]
                # p_off = slope_max(ecg[b,:], p_off, 5)
            
                if (q_on - p_off > 150)  or (q_on < p_off) :
                    p_on = -3000
                    p_off = -3000
                    p = -3000
                    p_dir = 0
                    print(b, 'p is detected but not in the right place')
                else:
                    
                    # p, peak_in = determin_ppeak(p_on, p_off, ecg[b,:], TH_max = max(ecg[b,p_on],ecg[b,p_off]), TH_min=min(ecg[b,p_on],ecg[b,p_off]))
                    # # print(peak_in)
                    # if (not peak_in):
                    #     p_on = -3000
                    #     p_off = -3000
                    #     p = -3000
                    #     print(b, 'no p is detected')
                    p_on, p, p_off,p_dir = determine_p(p_on, p_off,r, ecg[b,:])

                
            else:
                p_on = -3000
                p_off = -3000
                p = -3000
                p_dir = 0
                print(b, 'no p is detected')
            
            pq, q = determine_q(p_off, q_on, r, ecg[b,:])
            s, st = determine_s(s_off, t_on, r, ecg[b,:])

        else:
            q_on = -3000
            s_off = -3000
            r = -3000
            p_on = -3000
            p_off = -3000
            p = -3000   
            t_on = -3000
            t_off = -3000
            t = -3000 
            pq = -3000
            q = -3000 
            s = -3000
            st = -3000 
            p_dir = 0
            t_dir = 0                  
            print(b, 'no qrs is detected')
            # points[b,:] = [p_on, p, p_off, pq, q, r, s, st, t_on, t, t_off]
            # plot_with_points(ecg[b,:], label[b,:],points[b,:], b)

        points[b,:] = [p_on, p, p_off, pq, q, r, s, st, t_on, t, t_off]
        dirs[b, :] = [p_dir, t_dir]
        # iso_line [b] = cal_isoline(points[b,:], ecg[b,:])
        # print(points[b,:])
        # plot_with_points(ecg[b,:], label[b,:],points[b,:], b,id)
    return points, dirs

def fiducial_points(label,ecg,id):

    if label.shape[1] != 4:
        label_onehot = trans_one_hot_label(label)
    B, L = label_onehot.shape[0], label_onehot.shape[2]
    ecg = ecg[:,:,0]
    ecg = np.squeeze(ecg)

    points = np.zeros((B, 9))
    iso_line = np.zeros((B, 1))
    for b in range(B):

        qrs_info = find_wave(label_onehot[b,2,:])
        # print('qrs_info',qrs_info)
        p_info = find_wave(label_onehot[b,1,:])
        # print('p_info',p_info)
        t_info = find_wave(label_onehot[b,3,:])
        # 
        # print('t_info',t_info)

        if len(qrs_info) != 0:
            for qrs_i in range(qrs_info.shape[0]):

                if (((106)<= qrs_info[qrs_i, 1]) and ((106)>= qrs_info[qrs_i, 0])):
                    # print('o',qrs_info[qrs_i, 0])
                    # print('1',qrs_info[qrs_i, 1])                    
                    q_on = qrs_info[qrs_i, 0]
                    # q_on = slope_max(ecg[b,:], q_on, 5)
                    s_off = qrs_info[qrs_i, 1]
                    # s_off = slope_max(ecg[b,:], s_off, 20)
                    # print('q_on', q_on)
                    # print('s_off',s_off)
                    ############################# trt -> r #################################
                    if (len(t_info)!= 0):
                        delete = []
                        for k in range(len(t_info)):

                            if (qrs_info[qrs_i, 0]== t_info[k,1]) and (t_info[k,1]-t_info[k,0]<15):
                                q_on = t_info[k,0]
                                delete.append(k)
                                # t_info = np.delete(t_info, k, axis = 0)
                                continue
                            if (qrs_info[qrs_i, 1] == t_info[k,0]) and (t_info[k,1]-t_info[k,0]<15):
                                s_off = t_info[k, 1]
                                delete.append(k)
                                # t_info = np.delete(t_info, k, axis = 0)
                                continue
                        t_info = np.delete(t_info, delete, axis = 0)
                    ########################################################################
                    break
                            
                
                else:
                    dis = np.min(np.abs(qrs_info - L//2), axis=1 )
                    qrs_i = np.argmin(dis)
                    q_on = qrs_info[qrs_i, 0]
                    s_off = qrs_info[qrs_i, 1] 
                    
                    ############################# trt -> r #################################
                    if (len(t_info)!= 0):
                        delete = []
                        for k in range(len(t_info)):

                            if (qrs_info[qrs_i, 0]== t_info[k,1]) and (t_info[k,1]-t_info[k,0]<15):
                                
                                q_on = t_info[k,0]
                                delete.append(k)
                                # t_info = np.delete(t_info, k, axis = 0)
                                continue
                            if (qrs_info[qrs_i, 1] == t_info[k,0]) and (t_info[k,1]-t_info[k,0]<15):
                                print(t_info[k,1])
                                s_off = t_info[k, 1]
                                delete.append(k)
                                # t_info = np.delete(t_info, k, axis = 0)
                                continue
                        t_info = np.delete(t_info, delete, axis = 0)
                    ########################################################################
                    break
            # print(q_on, s_off)
                    # s_off = slope_max(ecg[b,:], s_off, 20)
            r = determin_peak(q_on, s_off, ecg[b,:], TH_max = max(ecg[b,q_on], ecg[b,s_off]) , TH_min= min(ecg[b,q_on], ecg[b,s_off]))
            if len(t_info)!= 0:
                t_i = np.argmin(np.abs(np.min((t_info - s_off), axis = 1)))
                t_on = t_info[t_i, 0]
                # t_on = slope_max(ecg[b,:], t_on, 5)
                t_off = t_info[t_i, 1]
                # t_off = slope_max(ecg[b,:], t_off, 5)


                if (t_on - s_off) > 150 or (t_on < s_off):
                    t_on = -3000
                    t_off = -3000
                    t = -3000
                    print(b, 't is detected but not in the right place')
                else:
                    
                    t, peak_in = determin_tpeak(t_on, t_off, ecg[b,:], TH_max = max(ecg[b,t_on],ecg[b,t_off]), TH_min= min(ecg[b,t_on],ecg[b,t_off]))
                    
                    if not peak_in:
                        t_off= min(t + 15, L-1)

            else:
                t_on = -3000
                t_off = -3000
                t = -3000
                print(b, 'no t is detected')

            if len(p_info) != 0:
                
                p_i = np.argmin(np.abs(np.min((p_info - q_on), axis = 1)))
                p_on = p_info[p_i, 0]
                # p_on = slope_max(ecg[b,:], p_on, 5)
                p_off = p_info[p_i, 1]
                # p_off = slope_max(ecg[b,:], p_off, 5)
            
                if (q_on - p_off > 150)  or (q_on < p_off) :
                    p_on = -3000
                    p_off = -3000
                    p = -3000
                    print(b, 'p is detected but not in the right place')
                else:
                    
                    p, peak_in = determin_ppeak(p_on, p_off, ecg[b,:], TH_max = max(ecg[b,p_on],ecg[b,p_off]), TH_min=min(ecg[b,p_on],ecg[b,p_off]))
                    
                    if (not peak_in):
                        p_on = -3000
                        p_off = -3000
                        p = -3000
                        print(b, 'no p is detected')
            else:
                p_on = -3000
                p_off = -3000
                p = -3000
                print(b, 'no p is detected')
        else:
            q_on = -3000
            s_off = -3000
            r = -3000
            p_on = -3000
            p_off = -3000
            p = -3000   
            t_on = -3000
            t_off = -3000
            t = -3000                     
            print(b, 'no qrs is detected')
            points[b,:] = [p_on, p, p_off, q_on, r, s_off, t_on, t, t_off]
            # plot_with_points(ecg[b,:], label[b,:],points[b,:], b)

        points[b,:] = [p_on, p, p_off, q_on, r, s_off, t_on, t, t_off]
        # iso_line [b] = cal_isoline(points[b,:], ecg[b,:])
        # print(points[b,:])
        # plot_with_points(ecg[b,:], label[b,:],points[b,:], b,id)
    return points

def embedding(ecg, em_len= 30,em_thres = 8): # num, 256
    # print(ecg.shape)
    ###################### old #######################
    # l = ecg.shape[0]
    # thres = 8
    # ecg_em = np.zeros((em_len,))
    # # print(ecg_em.shape)

    # for j in range(1,min(l,em_len)):
    #     if (ecg[j] - ecg[j-1])>thres:
    #         ecg_em[ j] = 1
    #     elif (ecg[j] - ecg[j-1])<-thres:
    #         ecg_em[j] =-1
    #     else:
    #         ecg_em[j] = 0

###################### new #######################
    l = ecg.shape[0]
    thres = em_thres
    ecg_em = []
    # print(ecg_em.shape)

    for j in range(1,min(l,em_len)):
        if (ecg[j] - ecg[j-1])>thres:
            ecg_em.append(1)
        elif (ecg[j] - ecg[j-1])<-thres:
            ecg_em.append(-1)
    if (len(ecg_em)):
        ecg_em = np.array(ecg_em)
        if len(ecg_em) > em_len:
            ecg_em = ecg_em[0:em_len]
        else:
            ecg_em = np.concatenate((ecg_em,np.zeros((em_len-len(ecg_em),))))
    else:
        ecg_em = np.zeros((em_len,))

    return ecg_em

def select_finetune( data, label):
    data_n = []
    data_s = []
    data_v = []
    data_f = []
    data_q = []
    train_data = []
    train_label = []
    for num in range(data.shape[0]):
        if label[num] == 0:
            data_n.append(data[num].reshape(-1,1).T)
        elif label[num] == 1:
            data_s.append(data[num].reshape(-1,1).T)
        elif label[num] == 2:
            data_v.append(data[num].reshape(-1,1).T)
        elif label[num] == 3:
            data_f.append(data[num].reshape(-1,1).T)
        elif label[num] == 4:
            data_q.append(data[num].reshape(-1,1).T)
    if len(data_n):
        data_n = np.concatenate(data_n, axis = 0)
        n_n = data_n.shape[0]
        train_data.append(data_n)
        label_n = np.zeros(data_n.shape[0])
        print('n_n', data_n.shape[0])
        train_label.append(label_n)
    else:
        n_n = 0

    if len(data_s):
        data_s = np.concatenate(data_s, axis = 0)
        s_n = data_s.shape[0]
        data_s = np.repeat(data_s, int(n_n/s_n)+50, axis = 0)
        train_data.append(data_s)
        label_s = np.ones(data_s.shape[0])
        print('s_n', data_s.shape[0])
        train_label.append(label_s)

    else:
        s_n = 0

    if len(data_v):
        data_v = np.concatenate(data_v, axis = 0)
        v_n = data_v.shape[0]
        data_v = np.repeat(data_v, int(n_n/v_n)+10, axis = 0)
        train_data.append(data_v)
        label_v = np.ones(data_v.shape[0]) * 2
        train_label.append(label_v)
        
    else:
        v_n = 0


    if len(data_f):
        data_f = np.concatenate(data_f, axis = 0)
        f_n = data_f.shape[0]
        data_f = np.repeat(data_f, int(n_n/f_n)+10, axis = 0)
        train_data.append(data_f)
        label_f = np.ones(data_f.shape[0]) * 3
        train_label.append(label_f)
    else:
        f_n = 0

    if len(data_q):
        data_q = np.concatenate(data_q, axis = 0)
        q_n = data_q.shape[0]
        data_q = np.repeat(data_q, int(n_n/q_n)+10, axis = 0)
        train_data.append(data_q)
        label_q = np.ones(data_q.shape[0]) * 4
        train_label.append(label_q)
    else:
        q_n = 0

    train_data = np.concatenate(train_data)
    train_label = np.concatenate(train_label, axis = 0)

    ######### 2 classes #################
    # label_n = np.zeros(data_n.shape[0])
    # label_s = np.ones(data_s.shape[0])    
    # train_label = np.concatenate((label_n,label_s), axis = 0)  
    if (s_n!= 0) or (v_n!= 0) or (f_n!= 0) or (q_n!= 0):  
        return train_data, train_label, False
    else:
        return train_data, train_label, True

def select_train( data, label,mode = 'up'):
    data_n = []
    data_s = []
    data_v = []
    data_f = []
    data_q = []

    for num in range(data.shape[0]):
        if label[num] == 0:
            data_n.append(data[num].reshape(-1,1).T)
        elif label[num] == 1:
            data_s.append(data[num].reshape(-1,1).T)
        elif label[num] == 2:
            data_v.append(data[num].reshape(-1,1).T)
        elif label[num] == 3:
            data_f.append(data[num].reshape(-1,1).T)
        elif label[num] == 4:
            data_q.append(data[num].reshape(-1,1).T)
    
    if len(data_n):
        data_n = np.concatenate(data_n, axis = 0)
    if len(data_s):
        data_s = np.concatenate(data_s, axis = 0)
    if len(data_v):
        data_v = np.concatenate(data_v, axis = 0)
    if len(data_f):
        data_f = np.concatenate(data_f, axis = 0)
    if len(data_q):
        data_q = np.concatenate(data_q, axis = 0)
    print('n:',data_n.shape)
    print('s:',data_s.shape)
    print('v:',data_v.shape)
    print('f:',data_f.shape)
    print('q:',data_q.shape)

    if mode == 'up':
        ######### 5 classes #################
        data_s = np.repeat(data_s, 30, axis = 0)
        data_v = np.repeat(data_v, 20, axis = 0)
        data_f = np.repeat(data_f, 60, axis = 0)
        data_q = np.repeat(data_q, 60, axis = 0)
        train_data = np.concatenate((data_n, data_s,data_v,data_f,data_q))
        # ######### 2 classes #################
        # data_s = np.repeat(data_s, 9, axis = 0)
        # train_data = np.concatenate((data_n, data_s))
    else:
        n_n = data_n.shape[0]
        index = random.sample(range(n_n), 1000)
        data_n = data_n[index] 
        train_data = np.concatenate((data_n, data_s,data_v,data_f,data_q))
    
     ######### 5 classes #################
    label_n = np.zeros(data_n.shape[0])
    label_s = np.ones(data_s.shape[0])
    label_v = np.ones(data_v.shape[0]) * 2
    label_f = np.ones(data_f.shape[0]) * 3
    label_q = np.ones(data_q.shape[0]) * 4
    train_label = np.concatenate((label_n,label_s, label_v, label_f, label_q), axis = 0)   

    ######### 2 classes #################
    # label_n = np.zeros(data_n.shape[0])
    # label_s = np.ones(data_s.shape[0])    
    # train_label = np.concatenate((label_n,label_s), axis = 0)   
    return train_data, train_label

def select_train_for_4classes( data, label,mode = 'up'):
    data_n = []
    data_s = []
    data_v = []
    data_f = []
    data_q = []
    
    for num in range(data.shape[0]):
        if label[num] == 0:
            data_n.append(np.expand_dims(data[num],axis = 0))
        elif label[num] == 1:
            data_s.append(np.expand_dims(data[num], axis = 0))
        elif label[num] == 2:
            data_v.append(np.expand_dims(data[num],axis = 0))
        elif label[num] == 3:
            data_f.append(np.expand_dims(data[num], axis = 0))
        elif label[num] == 4:
            data_q.append(np.expand_dims(data[num], axis = 0))
    if len(data_n):
        data_n = np.concatenate(data_n, axis = 0)
    if len(data_s):
        data_s = np.concatenate(data_s, axis = 0)
    if len(data_v):
        data_v = np.concatenate(data_v, axis = 0)
    if len(data_f):
        data_f = np.concatenate(data_f, axis = 0)
    if len(data_q):
        data_q = np.concatenate(data_q, axis = 0)
    if mode == 'up':
        ######### 5 classes #################
        
        data_s = np.repeat(data_s, 3, axis = 0)
        data_v = np.repeat(data_v, 1, axis = 0)
        data_f = np.repeat(data_f, 5, axis = 0)
        data_q = np.repeat(data_q, 5, axis = 0)
        train_data = np.concatenate((data_s,data_v,data_f,data_q))
        
    
     ######### 4 classes #################

    label_s = np.zeros(data_s.shape[0])
    label_v = np.ones(data_v.shape[0]) * 1
    label_f = np.ones(data_f.shape[0]) * 2
    label_q = np.ones(data_q.shape[0]) * 3
    train_label = np.concatenate((label_s, label_v, label_f, label_q), axis = 0)   
  
    return train_data, train_label

def cal_statistic(cm):
    class_num = cm.shape[0]
    total_pred = cm.sum(0)
    total_true = cm.sum(1)
    # total_true = np.array([17703,   491,  1357,   159])
    # special acc, abnormal inlcuded only
    acc_SP = sum([cm[i, i] for i in range(1, class_num)]) / total_pred[1:class_num].sum()
    pre_i = [cm[i, i] / total_pred[i] for i in range(class_num)]
    rec_i = [cm[i, i] / total_true[i] for i in range(class_num)]
    F1_i = [2 * pre_i[i] * rec_i[i] / (pre_i[i] + rec_i[i]) for i in range(class_num)]

    pre_i = np.array(pre_i)
    rec_i = np.array(rec_i)
    F1_i = np.array(F1_i)
    pre_i[np.isnan(pre_i)] = 0
    rec_i[np.isnan(rec_i)] = 0
    F1_i[np.isnan(F1_i)] = 0

    return list(pre_i), list(rec_i), list(F1_i)

# def label_combines(label):
#     for i in range(len(label)):
#         if label[i] == 3 or label[i] == 4:

def desample(data, rate):
    img_size = int(data.shape[0]/rate)
    rate = int(rate)
    
    data_desample = np.zeros((img_size,))
    for i in range(img_size):
        data_desample[i] = int(np.sum(data[i*rate:(i+1)*rate])/rate)
    return data_desample

def connection(img_ecg, levels):
    for i in range(img_ecg.shape[1]-1):
        
        if levels[i] - levels[i+1] > 1:
            mid_level = int((levels[i] - levels[i+1])/2)
            img_ecg[levels[i+1] + mid_level:levels[i],i] = 1
            img_ecg[levels[i+1]:levels[i+1] + mid_level,i+1] =1
        elif levels[i] - levels[i+1] < -1:
            mid_level = int((levels[i+1] - levels[i])/2)
            img_ecg[levels[i]: mid_level+levels[i],i] = 1
            img_ecg[mid_level+levels[i]:levels[i+1],i+1] =1
        else:
            pass

    return img_ecg    

def expansion(img_ecg, expansion_dim):
    kernel = np.ones((expansion_dim, expansion_dim), np.uint8)
    img_exp = cv2.dilate(img_ecg, kernel, iterations=1)
    return img_exp


def convert_bin(data):
    img_size = 64
    img_ecg = np.zeros((data.shape[0],img_size,img_size))
    for i in range(data.shape[0]):
        if data.ndim == 3:
            data_single = data[i,:, 0]
        else:
            data_single = data[i,:]

        
        data_desample = desample (data_single, rate =data_single.shape[0]/img_size)

        data_max = np.max(data_desample)
        data_min = np.min(data_desample)
        unit = int((data_max-data_min)/img_size) + 1        
        levels = []
        for j in range(img_size):
            if (unit):
                # print(data_max - data_single[j])
                # print('data_max', data_max)
                # print('data_min', data_min)
                # print('data_sample',data_single[j])
                # print('unit', unit)
                # print('level',int((data_max - data_single[j])/unit))
                level = min(int((data_max - data_desample[j])/unit),img_size-1)
                img_ecg[i,level,j] = 1
            else:
                level = 0
            levels.append(level)
        img_ecg[i] = connection(img_ecg[i], levels)
        img_ecg[i] = expansion(img_ecg[i], 3)
            
            
    return img_ecg





class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
import torch
def accuracy_clf(output, target):
    """Computes the precision@k for the specified values of k"""

    batch_size = output.size(0)

    if output.dim() == 2:
        pred =  torch.argmax(output,1)
    else:
        pred = output

    # print(pred.shape)
    # print(target.shape)
    correct = (pred == target.squeeze()).sum().item()
    # print(correct)
    res = correct*100/(batch_size)
    
    return res

