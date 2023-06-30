function [ecg_seg_list,annot_list,feature_matrix,annot_pre_list] = save_features_v2(ecg_data,ANNOT,ATRTIME,pole_buf,q_on,s_off,t_pre,p_cur,t_cur,p_post,filenum)

ecg_seg_list = [];
annot_list =[];
ATRTIME_seg_list = [];
rr_time_list = [];
rr_diff_list = [];
rp_list =[];
ATRTIME_9_list =[];
annot_pre_list =[];
qs_list =[];
r_pole_list =[];
t_pole_list = [];
tp_list =[];
tr_ratio_list =[];
pr_ratio_list = [];
sq_diff_list =[];
sp_list = [];
qp_list = [];
tp_post_list = [];
s_slope_list= [];
pr_inter_list = [];
rr_time_mean_long_list = [];
t_peak_list = [];
number_rpeak_list = [];
feature_matrix = [];
j = 1;
ecg_data = ecg_data(2:end-2);
ANNOT = ANNOT(2:end-2);
ATRTIME = ATRTIME(2:end-2);
pole_buf = pole_buf(2:end-2);
q_on = q_on (2:end-2);
s_off = s_off (2:end-2);
for i = 10:(length(ATRTIME)-10)
    rr_pre = ATRTIME(i)-ATRTIME(i-1);
    rr_post = ATRTIME(i+1)-ATRTIME(i);
    hrv = int32(1/4*rr_pre)+int32(1/2*rr_post);
%     rr = ATRTIME(i-8:i)-ATRTIME(i-9:i-1);
%     hrv = int32(mean(rr));
    [sample_L, sample_R] = hrv_range_select(hrv);
%     [sample_L, sample_R] = hrv_range_select_split(int32(1/4*rr_pre),int32(1/2*rr_post));
    if (ATRTIME(i) -sample_L > 0) && (ATRTIME(i) + sample_R < length(ecg_data))
        ecg_seg = ecg_data(ATRTIME(i) -100:ATRTIME(i) +149);

        ATRTIME_seg = ATRTIME(i-1: i+1);
        rr_time_init = (ATRTIME_seg(2:3)-ATRTIME_seg(1:2));
        rr_diff_init = rr_time_init(2)-rr_time_init(1);
        ATRTIME_9=ATRTIME(i-8: i);
        rr_time_9 = (ATRTIME_9(2:9)-ATRTIME_9(1:8));
        rr_mean_9 = mean(rr_time_9);
        
        rr_time = rr_time_init;
        rr_diff = rr_diff_init;
        rr_time_mean_long = mean(ATRTIME(2:i)-ATRTIME(1:i-1));
        
        
        annot = ANNOT(i);
        annot_pre = ANNOT(i-1);
        

        rp = ecg_data(ATRTIME(i));
        r_pole = pole_buf(i);
        t_pole = t_cur(3,i);
        qs = s_off(i) - q_on(i);
        if p_cur(2,i) == -2000 || (p_cur(1,i)-t_pre(1,i))<20
            tp = 0;
        else
            tp =1;
        end
        if p_post(2,i) == -2000 || (p_post(1,i)-t_cur(1,i))<20
            tp_post = 0;
        else
            tp_post =1;  
        end
            


        tr_ratio = t_cur(2,i)/(ecg_data(ATRTIME(i))+0.01);
        pr_ratio = p_cur(2,i)/(ecg_data(ATRTIME(i))+0.01);
        sq_ratio = ecg_data(q_on(i))-ecg_data(s_off(i));     
%         sp = ecg_data(s_off(i))/(ecg_data(s_off(i-1))+0.01);
%         qp = ecg_data(q_on(i))/(ecg_data(q_on(i-1))+0.01);
%         sp = ecg_data(s_off(i)) - ecg_data(s_off(i-1));
%         qp = ecg_data(q_on(i)) - ecg_data(q_on(i-1));
        sp = ecg_data(s_off(i));
        qp = ecg_data(q_on(i));
        

        s_slope = (ecg_data(s_off(i)-8)-ecg_data(s_off(i)))/8;
        pr_inter = ATRTIME(i)- p_cur(1,i);
        t_peak = t_cur(2,i);

        rr_peak = find(diff(sign(diff(ecg_data(ATRTIME(i)-4:ATRTIME(i)+6))))>0|diff(sign(diff(ecg_data(ATRTIME(i)-4:ATRTIME(i)+6))))<0);
        number_rpeak = length(rr_peak);
        j =j+1;
        
        annot_list = [annot_list annot];
        ATRTIME_seg_list = [ATRTIME_seg_list ATRTIME_seg];
        rp_list =[rp_list rp];
        ATRTIME_9_list =[ATRTIME_9_list ATRTIME_9];
        annot_pre_list =[annot_pre_list annot_pre];
        rr_time_list = [rr_time_list rr_time];
        rr_diff_list = [rr_diff_list rr_diff];
        qs_list =[qs_list qs];
        r_pole_list =[r_pole_list r_pole];
        t_pole_list = [t_pole_list t_pole];
        tp_list =[tp_list tp];
        tr_ratio_list =[tr_ratio_list tr_ratio];
        pr_ratio_list = [pr_ratio_list pr_ratio];
        sq_diff_list =[sq_diff_list sq_ratio];
        sp_list = [sp_list sp];
        qp_list = [qp_list qp];
        tp_post_list = [tp_post_list tp_post];
        s_slope_list = [s_slope_list s_slope];
        pr_inter_list = [pr_inter_list pr_inter];
        rr_time_mean_long_list = [rr_time_mean_long_list rr_time_mean_long];
        t_peak_list = [t_peak_list t_peak];
        number_rpeak_list = [number_rpeak_list number_rpeak];
        ecg_seg_list = [ecg_seg_list ecg_seg];
        end
    end

feature_matrix = [rp_list;rr_time_list;rr_diff_list;qs_list;r_pole_list;t_pole_list;tp_list;tr_ratio_list;pr_ratio_list;sq_diff_list;sp_list;qp_list;tp_post_list;s_slope_list;pr_inter_list;rr_time_mean_long_list;t_peak_list;number_rpeak_list];

end