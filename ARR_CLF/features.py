from ast import Return
from distutils.log import error
from operator import le
import re
from matplotlib.pyplot import axes, axis
import numpy as np
from utils import embedding
LEADS = ['avf', 'avl', 'avr', 'i', 'ii','iii', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']

class Features():
# ecg: (num_samples, length, 2)
# fiducial_points: (num_samples, 13) , p_on, p, p_off, pq, q, r, s， st, t_on, t, t_off, p_dir, t_dir
# label: (num_samples, 1)
    def __init__(self, ecg, fiducial_points, rrs, label): 
        if (len(ecg.shape) == 3):
            self.ecg = ecg[:,:,0]
        self.ecg = np.squeeze(self.ecg)
        # if label != None:
        self.label = label
        self.rrs = rrs
        
        self.begin_id = 0
        self.fiducial_points = fiducial_points


        # features
        self.latest_8norm_rr = []
        self.latest_8norm_pr = []
        self.latest_8norm_qrs = []

        self.latest_8norm_r_amp = []
        self.latest_8norm_q_amp = []
        self.latest_8norm_s_amp = []
        self.latest_8norm_p_amp = []
        self.latest_8norm_t_amp = []

#  utils
    def NORM(self, id):
        
        if self.label[id] == 0:
            return True
        else:
            return False


    def RR_PRE (self, id):

        rr_pre = self.rrs[id, 1]

        return rr_pre
    

    def RR_POST(self, id):

        rr_post = self.rrs[id, 0]

        return rr_post
    
    
    def RR_AVE (self):
        # print(self.latest_8norm_rr)
        # print(sum(self.latest_8norm_rr) / len(self.latest_8norm_rr))
        return sum(self.latest_8norm_rr) / len(self.latest_8norm_rr)

    def QRS_EXIST(self, id):
        qrs_exist = 0
        if (self.fiducial_points[id, 5] != -3000):
            qrs_exist = 1
        return qrs_exist


    def P_EXIST(self, id):
        
        p_exist = 0
        if (self.fiducial_points[id, 1] != -3000):
            p_exist = 1
        return p_exist
    def T_EXIST(self, id):
        t_exist = 0
        if (self.fiducial_points[id, 9] != -3000):
            t_exist = 1
        return t_exist

    def PR(self, id):
        if (self.P_EXIST(id)):
            pr = self.fiducial_points[id, 5] - self.fiducial_points[id, 1]
        else:
            pr = 0
        return pr



    def PR_AVE(self):
        return sum(self.latest_8norm_pr) / len(self.latest_8norm_pr)
        
    def QRS(self, id,):
        qrs = (self.fiducial_points[id, 6] - self.fiducial_points[id, 4])
        return qrs

    def QRS_AVE(self):
        return sum(self.latest_8norm_qrs) / len(self.latest_8norm_qrs)

    def R_AMP(self, id):
        if self.QRS_EXIST(id):
            # print(int(self.fiducial_points[id, 3]))
            # print(self.fiducial_points[id, 5])
            r_amp = self.fiducial_points[id, 5]
        else:
            r_amp = 0
        return r_amp
    
    def R_AMP_AVE(self):
        return sum(self.latest_8norm_r_amp) / len(self.latest_8norm_r_amp)
    
    def S_AMP(self, id):
        if self.QRS_EXIST(id):
            s_amp = self.ecg[id, int(self.fiducial_points[id, 6])]
        else:
            s_amp = 0
        return s_amp

    def S_AMP_AVE(self):
        return sum(self.latest_8norm_s_amp) / len(self.latest_8norm_s_amp)

    def Q_AMP(self, id):
        if self.QRS_EXIST(id):
            q_amp = self.ecg[id, int(self.fiducial_points[id, 4])]
        else:
            q_amp = 0
        return q_amp

    def Q_AMP_AVE(self):
        return sum(self.latest_8norm_q_amp) / len(self.latest_8norm_q_amp)


    def P_AMP(self, id):
        if(self.P_EXIST(id)):
            p_amp = self.ecg[id, int(self.fiducial_points[id, 1])]
        else:
            p_amp = 0
        return p_amp
    
    def P_AMP_AVE(self):
        return sum(self.latest_8norm_p_amp)/len(self.latest_8norm_p_amp)
    
    def T_AMP(self, id):
        if (self.T_EXIST(id)):
            t_amp = self.ecg[id, int(self.fiducial_points[id, 9])]
        else:
            t_amp = 0
        return t_amp
    
    def T_DIR(self, id):
        return self.fiducial_points[id, 12]
    def T_AMP_AVE(self):
        return sum(self.latest_8norm_t_amp)/len(self.latest_8norm_t_amp)
    

    
    def INIT_AVE(self):
        num_norm = 0
        # print(self.label.shape[0])
        for i in range(0, self.label.shape[0]):
            
            if self.NORM(i):
                num_norm = num_norm + 1

                self.latest_8norm_rr.append(self.RR_PRE(i))
                self.latest_8norm_pr.append(self.PR(i))
                self.latest_8norm_qrs.append(self.QRS(i))
                self.latest_8norm_r_amp.append(self.R_AMP(i))
                self.latest_8norm_s_amp.append(self.S_AMP(i))
                self.latest_8norm_q_amp.append(self.Q_AMP(i))
                self.latest_8norm_p_amp.append(self.P_AMP(i))
                self.latest_8norm_t_amp.append(self.T_AMP(i))
                
                if num_norm == 8:
                    self.begin_id = i + 1
                    break

            
    def UPDATA_AVE(self, id):
        if self.NORM(id):
            self.latest_8norm_rr.append(self.RR_PRE(id))
            del self.latest_8norm_rr[0]
            self.latest_8norm_pr.append(self.PR(id))
            del self.latest_8norm_pr[0]
            self.latest_8norm_qrs.append(self.QRS(id))
            del self.latest_8norm_qrs[0]
            self.latest_8norm_r_amp.append(self.R_AMP(id))
            del self.latest_8norm_r_amp[0]
            self.latest_8norm_q_amp.append(self.Q_AMP(id))
            del self.latest_8norm_q_amp[0]
            self.latest_8norm_s_amp.append(self.S_AMP(id))
            del self.latest_8norm_s_amp[0]

            self.latest_8norm_p_amp.append(self.P_AMP(id))
            del self.latest_8norm_p_amp[0]
            self.latest_8norm_t_amp.append(self.T_AMP(id))
            del self.latest_8norm_t_amp[0]
    
    # features
    def RR_DIFF(self, id):
        if (self.begin_id <= id):
            # print(self.RR_POST(id))
            # print(self.RR_PRE(id))
            rr_diff = self.RR_POST(id) - self.RR_PRE(id)
        else:
            print('pay attention to index')
        return rr_diff
    
    def RR_PRE_RR_AVE(self, id):
        if (self.begin_id <= id):
            rr_pre_rr_ave = self.RR_PRE(id) - self.RR_AVE()
        else:
            print('pay attention to index')
        return rr_pre_rr_ave
    
    def RR_POST_RR_AVE(self, id):
        if (self.begin_id <= id):
            rr_post_rr_ave = self.RR_POST(id) - self.RR_AVE()
        else:
            print('pay attention to index')
        return rr_post_rr_ave

    def PR_CUR_PR_AVE(self, id):
        pr_cur_pr_ave = self.PR(id) - self.PR_AVE()
        return pr_cur_pr_ave
    
    def QRS_CUR_QRS_AVE(self, id):
        qrs_cur_qrs_ave = self.QRS(id) - self.QRS_AVE()
        return qrs_cur_qrs_ave

    def R_AMP_R_AMP_AVE(self, id):
        r_amp_r_amp_ave = self.R_AMP(id) - self.R_AMP_AVE()
        return r_amp_r_amp_ave
    
    def S_AMP_S_AMP_AVE(self,id):
        s_amp_s_amp_ave = self.S_AMP(id) - self.S_AMP_AVE()
        return s_amp_s_amp_ave

    def Q_AMP_Q_AMP_AVE(self,id):
        q_amp_q_amp_ave = self.Q_AMP(id) - self.Q_AMP_AVE()
        return q_amp_q_amp_ave

    def P_AMP_P_AVE(self, id):
        p_amp_p_ave = self.P_AMP(id) - self.P_AMP_AVE()
        return p_amp_p_ave
    
    def T_AMP_T_AVE(self, id):
        t_amp_t_ave = self.T_AMP(id) - self.T_AMP_AVE()
        return t_amp_t_ave

    def QRS_EMB(self, id):
        em_len = 20
        if self.QRS_EXIST(id):
            qrs_emb = embedding(self.ecg[id,int(self.fiducial_points[id,4]):int(self.fiducial_points[id,6])],em_len=em_len)
        else:
            qrs_emb = np.zeros((em_len))
        return qrs_emb
    
    def T_EMB(self, id):
        em_len = 25
        if self.T_EXIST(id):
            t_emb = embedding(self.ecg[id,int(self.fiducial_points[id, 8]):int(self.fiducial_points[id, 10])],em_len = em_len)
        else:
            t_emb = np.zeros((em_len))
        return t_emb



    def feature_map(self):
        B, L = self. ecg.shape[0], self.ecg.shape[1]
        self.INIT_AVE()
        # print('begin_id',self.begin_id)
        features_all = []
        for id in range(self.begin_id + 1, B-1):
           
            rr_diff = self.RR_DIFF( id)
            rr_diff_pre = self.RR_DIFF(id-1)
            rr_diff_post = self.RR_DIFF(id + 1)

            rr_pre_rr_ave = self.RR_PRE_RR_AVE(id)
            rr_pre_rr_ave_pre = self.RR_PRE_RR_AVE(id + 1)
            rr_post_rr_ave = self.RR_POST_RR_AVE( id )
            rr_post_rr_ave_post = self.RR_POST_RR_AVE( id - 1 )
            
            pr_cur_pr_ave = self.PR_CUR_PR_AVE(id)
            qrs_cur_qrs_ave = self.QRS_CUR_QRS_AVE(id)
            r_amp_r_amp_ave = self.R_AMP_R_AMP_AVE(id)
            s_amp_s_amp_ave = self.S_AMP_S_AMP_AVE(id)
            q_amp_q_amp_ave = self.Q_AMP_Q_AMP_AVE(id)

            p_amp_p_ave = self.P_AMP_P_AVE(id)
            t_amp_t_ave = self.T_AMP_T_AVE(id)
            qrs_emb = self.QRS_EMB(id)
            t_emb = self.T_EMB(id)
            t_dir = self.T_DIR(id)

            self.UPDATA_AVE(id)
            # features = np.hstack((rr_diff,rr_diff_pre, rr_diff_post,rr_pre_rr_ave, rr_post_rr_ave,rr_pre_rr_ave_pre,rr_post_rr_ave_post,rr_diff,rr_diff_pre, rr_diff_post,rr_pre_rr_ave, rr_post_rr_ave,rr_pre_rr_ave_pre,rr_post_rr_ave_post,rr_diff,rr_diff_pre, rr_diff_post,rr_pre_rr_ave, rr_post_rr_ave,rr_pre_rr_ave_pre,rr_post_rr_ave_post, qrs_cur_qrs_ave,r_amp_r_amp_ave,q_amp_q_amp_ave,s_amp_s_amp_ave, p_amp_p_ave,t_amp_t_ave,qrs_emb,t_emb))
            features = np.hstack((rr_diff, rr_pre_rr_ave, rr_post_rr_ave,rr_diff, rr_pre_rr_ave, rr_post_rr_ave,rr_diff, rr_pre_rr_ave, rr_post_rr_ave,rr_diff, rr_pre_rr_ave, rr_post_rr_ave, qrs_cur_qrs_ave, qrs_cur_qrs_ave, r_amp_r_amp_ave, r_amp_r_amp_ave, q_amp_q_amp_ave, s_amp_s_amp_ave, p_amp_p_ave, t_amp_t_ave, t_dir, t_dir))
            # print(features.shape)
            features = np.expand_dims(features, axis=0)
            features_all.append(features)

        return np.concatenate(features_all, axis=0)  

    def duiqi_label(self):
        return self.label[self.begin_id + 1:-1,:]

    def duiqi_data(self):
        return self.ecg[self.begin_id + 1:-1,:]









    

