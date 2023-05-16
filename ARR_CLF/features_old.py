from matplotlib.pyplot import axes, axis
import numpy as np

LEADS = ['avf', 'avl', 'avr', 'i', 'ii','iii', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

class Features():
# ecg: (num_samples, length, 2)
# fiducial_points: (num_samples, 9) , p_on, p, p_off, q_on, r, s_off, t_on, t, t_off
# label: (num_samples, 1)
    def __init__(self, ecg, fiducial_points, label): 

        self.ecg = ecg[:,:,0]
        self.ecg = np.squeeze(self.ecg)
        self.label = label
        self.fiducial_points = fiducial_points

    
#features used out of table
    def T_EXIST(self, id):
        return self.fiducial_points[id, -1] != -3000
    
    def RR(self, id):
        if (id > 7):
            rr = self.fiducial_points[id, 4] - self.fiducial_points[id - 1, 4]
        else:
            rr = 0
        return rr
    
    def RR_AVE(self, id):
        if (id > 7):
            sum = 0
            for i in range(1, 9):
                sum = sum + self.RR(id - i)
            rr_ave = sum / 8
        else:
            rr_ave = 0
        return rr_ave
            
    def PP(self, id, lead_id):
        if (self.P_EXIST(id, lead_id).all() and self.P_EXIST(id - 1, lead_id ).all() and (id > 7)):
            pp = self.fiducial_points[id, 1] - self.fiducial_points[id - 1, 1]
        else:
            pp = 0
        return pp
    
    def PP_AVE(self, id, lead_id):
        if (id > 7):
            sum = 0
            for i in range(1, 9):
                sum = sum + self.PP(id - i, lead_id)
            pp_ave = sum / 8
        else:
            pp_ave = 0
        return pp_ave
#features in table
#1    
    def P_DIR(self, id):
        if (self.P_EXIST(id).all()):
            judge = (self.ecg[id, self.fiducial_points[id, 0]] - self.ecg[id, self.fiducial_points[id, 1]] < 0) 
            if (judge):
                p_dir = 1
            else:
                p_dir = -1                    
        else:
            p_dir = 0
        return p_dir
#2    
    def P_AMP(self, id, lead_id):
        p_amp = np.zeros(len(lead_id))
        if (self.P_EXIST(id, lead_id).all()):
            p_amp = self.ecg[id, lead_id, self.fiducial_points[id, 1]]


        return p_amp
#3    
    def P_DURA(self, id, lead_id):
        p_dura = np.zeros((len(lead_id),))
        if (self.P_EXIST(id, lead_id).all):
            p_dura[0:len(lead_id)] = self.fiducial_points[id, 2] - self.fiducial_points[id, 0] 

        return p_dura
#4
    def P_EXIST(self, id, lead_id):
        p_exist = np.zeros((len(lead_id),))
        if (self.fiducial_points[id, 0] != -3000):
            p_exist[0: len(lead_id)]  = 1
        return p_exist
 
#5    
    def PR(self, id, lead_id):
        pr = np.zeros((len(lead_id),))
        if (self.P_EXIST(id, lead_id).all()):
            pr[0:len(lead_id)] = self.fiducial_points[id, 4] - self.fiducial_points[id, 1]

        return pr
#6    
    def PR_RR(self, id, lead_id):
        pr_rr  = np.zeros((len(lead_id),))
        if (id > 7) and (self.RR(id)):
            pr_rr[0:len(lead_id)] = self.PR(id, lead_id)[0] / self.RR(id)

        return pr_rr           
#7      
    def QRS(self, id, lead_id):
        qrs = np.zeros((len(lead_id),))
        qrs[0:len(lead_id)] = (self.fiducial_points[id, 5] - self.fiducial_points[id, 3])

        return qrs
#8    
    def QRS_AMP(self, id, lead_id):
        qrs_amp = np.array(self.ecg[id, lead_id, self.fiducial_points[id, 4]])
        return qrs_amp
#9    
    def QRS_DIR(self, id, lead_id):
        qrs_dir = np.zeros(len(lead_id))
        judge = (self.ecg[id, lead_id, self.fiducial_points[id, 3]] - self.ecg[id, lead_id, self.fiducial_points[id, 4]] < 0) 
        for i in range(0, len(lead_id)):
            if (judge[i]):
                qrs_dir[i] = 1
            else:
                qrs_dir[i] = -1                    
        return qrs_dir
##10  Q_DURA   

#11    
    def Q_AMP_1_4_R_AMP(self, id, lead_id):
        
        return self.ecg[id, lead_id, self.fiducial_points[id, 3]] - self.ecg[id, lead_id, self.fiducial_points[id, 4]] / 4
#12    
    def ISO_S_AMP(self, id, lead_id):
        return self.iso_line[id] - self.ecg[id, lead_id, self.fiducial_points[id, 5]]
#13    
    def ISO_T_AMP(self, id, lead_id):
        return self.iso_line[id] - self.T_AMP(id, lead_id)
#14
    def T_AMP_S_AMP(self, id, lead_id):
        return self.T_AMP(id, lead_id) - self.ecg[id, lead_id, self.fiducial_points[id, 5]]   
#15
    def T_AMP(self, id, lead_id):
        if (self.T_EXIST(id)):
            t_amp = self.ecg[id, lead_id, self.fiducial_points[id, 7]]
        else:
            t_amp = np.zeros(len(lead_id))
        return t_amp
#16    
    def T_DIR(self, id, lead_id):
        t_dir = np.zeros(len(lead_id))
        if (self.T_EXIST(id)):
            judge = (self.ecg[id, lead_id, self.fiducial_points[id, 6]] - self.ecg[id, lead_id, self.fiducial_points[id, 7]] < 0) 
            for i in range(0, len(lead_id)):
                if (judge[i]):
                    t_dir[i] = 1
                else:
                    t_dir[i] = -1                    

        return t_dir
#17   
    def T_AMP_1_10_QRS_AMP(self, id, lead_id):
        return self.T_AMP(id, lead_id) - self.QRS_AMP(id, lead_id) / 10
#18    
    def QT(self, id, lead_id):
        qt = np.zeros((len(lead_id),))
        if (self.T_EXIST(id)):
            qt[0:len(lead_id)] = self.fiducial_points[id, 7] - self.fiducial_points[id, 3]
        return qt
#19    
    def QT_RR(self, id,lead_id):
        qt_rr = np.zeros((len(lead_id),))
        if (id > 7)  and (self.RR(id)):
            qt_rr[0: len(lead_id)] = self.QT(id, lead_id)[0] / self.RR(id)
        return qt_rr            
#20
    def T_AMP_I_T_AMP_III(self, id, lead_id):
        t_amp_i_t_amp_iii = np.zeros((len(lead_id),))
        if (self.T_EXIST(id)):
            t_amp_i_t_amp_iii[0:len(lead_id)] = self.T_AMP(id, [3]) - self.T_AMP(id, [5])
        return t_amp_i_t_amp_iii
#21
    def T_AMP_V5_T_AMP_V1(self, id, lead_id):
        t_amp_v5_t_amp_v1 = np.zeros((len(lead_id),))
        if (self.T_EXIST(id)):
            t_amp_v5_t_amp_v1[0:len(lead_id)] = self.T_AMP(id, [10]) - self.T_AMP(id, [6])
        return t_amp_v5_t_amp_v1
#22
    def T_AMP_V6_T_AMP_V1(self, id, lead_id):
        t_amp_v6_t_amp_v1 = np.zeros((len(lead_id),))
        if (self.T_EXIST(id)):
            t_amp_v6_t_amp_v1[0:len(lead_id)] = self.T_AMP(id, [11]) - self.T_AMP(id, [6])
        return t_amp_v6_t_amp_v1
#23            
    def RR_PRE_RR_AVE(self, id, lead_id): 
        rr_pre_rr_ave = np.zeros((len(lead_id,)))
        rr_pre_rr_ave [0:len(lead_id)]= self.RR(id) - self.RR_AVE(id)
        
        return rr_pre_rr_ave        
#24       
    def RR_DIFF(self, id, lead_id):
        rr_diff = np.zeros((len(lead_id),))
        if (id > 0) and (id < self.ecg.shape[0]-1):
            rr_diff[0:len(lead_id)] = (self.RR(id + 1) - self.RR(id)) - (self.RR(id) - self.RR(id - 1))

        return rr_diff
#25        
    def RR_STD(self, id, lead_id):
        rr_std = np.zeros((len(lead_id),))
        if (id > 7):
            rr_list = []  
            for i in range(1, 9):
                rr_list.append(self.RR(id - i))
            rr_std[0:len(lead_id)] = np.std(rr_list, ddof = 1) 
        return rr_std
#26        
    def PP_AVE_RR_AVE(self, id, lead_id):
        pp_ave_rr_ave = np.zeros((len(lead_id),))
        if (self.RR_AVE(id)):
            pp_ave_rr_ave[0:len(lead_id)] = self.PP_AVE(id, lead_id) - self.RR_AVE(id)
        return pp_ave_rr_ave

#27        
    def PP_PP_AVE(self, id, lead_id):
        pp_ave = np.zeros((len(lead_id),))
        if (self.PP_AVE(id,lead_id)):
            pp_ave[0: len(lead_id)] = self.PP(id, lead_id) - self.PP_AVE(id,lead_id)
        return pp_ave

#28        
    def PP_DIFF(self, id, lead_id):
        pp_diff = np.zeros((len(lead_id),))
        if (id > 0) and (id<self.ecg.shape[0]-1):
            pp_diff[0:len(lead_id)] = (self.PP(id + 1, lead_id) - self.PP(id,  lead_id)) - (self.PP(id, lead_id) - self.PP(id - 1, lead_id))

        return pp_diff
#29        
    def RS(self, id, lead_id):
        rs = np.zeros((len(lead_id),))
        rs[0:len(lead_id)] = self.fiducial_points[id, 5] - self.fiducial_points[id, 4]
        return rs
#30        
    def R_AMP_S_AMP(self, id, lead_id):
        return self.ecg[id, lead_id, self.fiducial_points[id, 4]] - self.ecg[id, lead_id, self.fiducial_points[id, 5]]


##31  F_WAVE 

    def feature_map (self):
        B, L =self. ecg.shape[0], self.ecg.shape[1]
        features_all = []
        lead_id = list(range(12))
        for id in range(B):
            p_dir =  np.expand_dims(self.P_DIR(id, lead_id), axis = 1)
            p_amp = np.expand_dims(self.P_AMP(id, lead_id), axis = 1)
            p_dura = np.expand_dims(self.P_DURA(id, lead_id), axis = 1)
            p_exist= np.expand_dims(self.P_EXIST(id, lead_id), axis = 1)
            pr = np.expand_dims(self.PR(id, lead_id), axis = 1)
            pr_rr = np.expand_dims(self.PR_RR(id, lead_id), axis = 1)
            qrs = np.expand_dims(self.QRS(id, lead_id), axis = 1)
            qrs_amp = np.expand_dims(self.QRS_AMP(id, lead_id), axis = 1)
            qrs_dir = np.expand_dims(self.QRS_DIR(id, lead_id), axis = 1)
            q_amp_1_4_r_amp = np.expand_dims(self.Q_AMP_1_4_R_AMP(id, lead_id), axis = 1)
            iso_s_amp = np.expand_dims(self.ISO_S_AMP(id, lead_id), axis = 1)
            iso_t_amp = np.expand_dims(self.ISO_T_AMP(id, lead_id), axis = 1)
            t_amp_s_amp =np.expand_dims( self.T_AMP_S_AMP(id, lead_id), axis = 1)
            t_amp = np.expand_dims(self.T_AMP(id, lead_id), axis = 1)
            t_dir = np.expand_dims(self.T_DIR(id, lead_id), axis = 1)
            t_amp_1_10_qrs_amp = np.expand_dims(self.T_AMP_1_10_QRS_AMP(id, lead_id), axis = 1)
            qt = np.expand_dims(self.QT(id, lead_id), axis = 1)
            qt_rr = np.expand_dims(self.QT_RR(id, lead_id), axis = 1)
            t_amp_i_t_amp_iii = np.expand_dims(self.T_AMP_I_T_AMP_III( id, lead_id), axis = 1)
            t_amp_v5_t_amp_v1 = np.expand_dims(self.T_AMP_V5_T_AMP_V1( id, lead_id), axis = 1)
            t_amp_v6_t_amp_v1 = np.expand_dims(self.T_AMP_V6_T_AMP_V1( id, lead_id), axis = 1)
            rr_pre_rr_ave = np.expand_dims(self.RR_PRE_RR_AVE( id, lead_id), axis = 1)
            rr_diff = np.expand_dims(self.RR_DIFF( id, lead_id), axis = 1)
            rr_std = np.expand_dims(self.RR_STD( id, lead_id), axis = 1)
            pp_ave_rr_ave = np.expand_dims(self.PP_AVE_RR_AVE( id, lead_id), axis = 1)
            pp_pp_ave = np.expand_dims(self.PP_PP_AVE( id, lead_id), axis = 1)
            pp_diff = np.expand_dims(self.PP_DIFF( id, lead_id), axis = 1)
            rs = np.expand_dims(self.RS(id, lead_id), axis = 1)
            r_amp_s_amp = np.expand_dims(self.R_AMP_S_AMP( id, lead_id), axis = 1)

            features = np.concatenate((p_dir, p_amp,p_dura, p_exist, pr, pr_rr,
            qrs,qrs_amp,qrs_dir, q_amp_1_4_r_amp,iso_s_amp,iso_t_amp,t_amp_s_amp,t_amp,
            t_dir, t_amp_1_10_qrs_amp, qt,qt_rr,t_amp_i_t_amp_iii,t_amp_v5_t_amp_v1, t_amp_v6_t_amp_v1,
            rr_pre_rr_ave,rr_diff, rr_std, pp_ave_rr_ave, pp_pp_ave,pp_diff,rs, r_amp_s_amp ), axis = 1)
            # print(features.shape)
            features = np.expand_dims(features, axis=0)
            features_all.append(features)

        return np.concatenate(features_all, axis=0)











        


        
        
        
        
            

            
          
        
        
        
            
        
                

