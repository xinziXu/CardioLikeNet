function [ecg_seg_list,annot_list,feature_matrix,annot_pre_list] = save_features(ecg_data,ANNOT,ATRTIME,pole_buf,q_on,s_off,t_pre,p_cur,t_cur,p_post,filenum)

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
        ATRTIME_seg = ATRTIME(i-1: i+1);
        rr_time_init = (ATRTIME_seg(2:3)-ATRTIME_seg(1:2));
        rr_diff_init = rr_time_init(2)-rr_time_init(1);
        ATRTIME_9=ATRTIME(i-8: i);
        ATRTIME_8=ATRTIME(i-7: i);
        rr_time_9 = (ATRTIME_9(2:9)-ATRTIME_9(1:8));
        rr_mean_9 = mean(rr_time_9);
        rr_time = rr_time_init;
        rr_diff = rr_diff_init;
        
        
        annot = set_label(ANNOT(i));
        annot_pre = set_label(ANNOT(i-1));
        
%         rp = ecg_data(ATRTIME(i))/(mean(ecg_data(ATRTIME_8)));
        rp = ecg_data(ATRTIME(i));
        r_pole = pole_buf(i);
        t_pole = t_cur(3,i);
        qs = s_off(i) - q_on(i);
        tp = p_cur(1,i)-t_pre(1,i);
        if tp < 20
            tp = 0;
        else
            tp =1;
        end
        tp_post = p_post(1,i)-t_cur(1,i);
        if tp_post < 20
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
        
        m = 1;
        ecg_seg = [];
        for k = ATRTIME(i)-sample_L:ATRTIME(i)+sample_R-1
            ecg_seg(m) = ecg_data(k);
            m = m + 1;
        end
        s_slope = (ecg_data(s_off(i)-8)-ecg_data(s_off(i)))/8;
        pr_inter = ATRTIME(i)- p_cur(1,i);
        
        ecg_filename=['./features_doctor_like/ecg_seg/',num2str(filenum),'/'];
        annot_filename=['./features_doctor_like/annot/',num2str(filenum),'/'];
        rp_filename=['./features_doctor_like/rp/',num2str(filenum),'/'];
        rr_time_filename=['./features_doctor_like/rr_time/',num2str(filenum),'/'];
        rr_diff_filename=['./features_doctor_like/rr_diff/',num2str(filenum),'/'];
        annot_pre_filename=['./features_doctor_like/annot_pre/',num2str(filenum),'/'];
        qs_filename=['./features_doctor_like/qs/',num2str(filenum),'/'];
        r_pole_filename=['./features_doctor_like/r_pole/',num2str(filenum),'/'];
        t_pole_filename=['./features_doctor_like/t_pole/',num2str(filenum),'/'];
        tp_filename=['./features_doctor_like/tp/',num2str(filenum),'/'];
        tr_ratio_filename=['./features_doctor_like/tr_ratio/',num2str(filenum),'/'];
        pr_ratio_filename=['./features_doctor_like/pr_ratio/',num2str(filenum),'/'];
        sp_filename=['./features_doctor_like/sq_ratio/',num2str(filenum),'/'];
        
        if ~exist(ecg_filename,'dir')
            mkdir(ecg_filename)
        end
        if ~exist(annot_filename,'dir')
            mkdir(annot_filename)
        end
        if ~exist(rp_filename,'dir')
            mkdir(rp_filename)
        end
        if ~exist(rr_time_filename,'dir')
            mkdir(rr_time_filename)
        end
        if ~exist(rr_diff_filename,'dir')
            mkdir(rr_diff_filename)
        end
        if ~exist(annot_pre_filename,'dir')
            mkdir(annot_pre_filename)
        end
        if ~exist(qs_filename,'dir')
            mkdir(qs_filename)
        end
        if ~exist(r_pole_filename,'dir')
            mkdir(r_pole_filename)
        end
        if ~exist(t_pole_filename,'dir')
            mkdir(t_pole_filename)
        end        
        if ~exist(tp_filename,'dir')
            mkdir(tp_filename)
        end
        if ~exist(tr_ratio_filename,'dir')
            mkdir(tr_ratio_filename)
        end
        if ~exist(pr_ratio_filename,'dir')
            mkdir(pr_ratio_filename)
        end
        if ~exist(sp_filename,'dir')
            mkdir(sp_filename)
        end        
% 
%         save([ecg_filename,num2str(j),'.mat'],'ecg_seg')
%         save([annot_filename,num2str(j),'.mat'],'annot')
%         save([rp_filename,num2str(j),'.mat'],'rp')
%         save([rr_time_filename,num2str(j),'.mat'],'rr_time')
%         save([rr_diff_filename,num2str(j),'.mat'],'rr_diff')
%         save([annot_pre_filename,num2str(j),'.mat'],'annot_pre')
%         save([qs_filename,num2str(j),'.mat'],'qs')
%         save([r_pole_filename,num2str(j),'.mat'],'r_pole')
%         save([t_pole_filename,num2str(j),'.mat'],'t_pole')
%         save([tp_filename,num2str(j),'.mat'],'tp')
%         save([tr_ratio_filename,num2str(j),'.mat'],'tr_ratio') 
%         save([pr_ratio_filename,num2str(j),'.mat'],'pr_ratio')
%         save([sp_filename,num2str(j),'.mat'],'sq_ratio')   
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
        end
end
feature_matrix = [rp_list;rr_time_list;rr_diff_list;qs_list;r_pole_list;t_pole_list;tp_list;tr_ratio_list;pr_ratio_list;sq_diff_list;sp_list;qp_list;tp_post_list;s_slope_list;pr_inter_list];
end