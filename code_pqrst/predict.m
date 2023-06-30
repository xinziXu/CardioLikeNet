function [annot_predict]=predict(feature_normal,feature_matrix,annot_pre)
annot_predict(1) = 1;
rp_n = feature_normal(1);
rr_time1_n = feature_normal(2);
rr_time2_n = feature_normal(3);
rr_diff_n = feature_normal(4);
sq_n = feature_normal(5);
r_pole_n = feature_normal(6);
t_pole_n = feature_normal(7);
tp_n  = feature_normal(8);
tr_n = feature_normal(9);
pr_n = feature_normal(10);
sq_p_n= feature_normal(11);
sp_n = feature_normal(12);
qp_n = feature_normal(13);
tp_post_n  = feature_normal(14);
s_slope_n = feature_normal(15);
pr_inter_n= feature_normal(16);
t_peak_n = feature_normal(17);
for i = 2:length(annot_pre)
    ann_pre = annot_predict(i-1);
    rp = feature_matrix(1,i);
    rr_time1 = feature_matrix(2,i);
    rr_time2 = feature_matrix(3,i);
    rr_diff = feature_matrix(4,i);
    sq = feature_matrix(5,i);
    r_pole = feature_matrix(6,i);
    t_pole = feature_matrix(7,i);
    tp  = feature_matrix(8,i);
    tr = feature_matrix(9,i);
    pr = feature_matrix(10,i);
    sq_p = feature_matrix(11,i);
    sp = feature_matrix(12,i);
    qp = feature_matrix(13,i);
    tp_post  = feature_matrix(14,i);
    s_slope  = feature_matrix(15,i);
    pr_inter  = feature_matrix(16,i);
    rr_time_mean_long = feature_matrix(17,i);
    t_peak = feature_matrix(18,i);
    number_rpeak = feature_matrix(19,i);
    case1 = (((rr_time1-rr_time1_n)<-20) && ((rr_time2-rr_time2_n)>20)) || (((rr_diff-rr_diff_n) > 20));
    case2 = (r_pole - r_pole_n) == -2;
    case3 = ((abs(rp-rp_n)>200) ||  (abs(sp-sp_n)>200) || (abs(qp-qp_n)>200));
    case4 = ((rr_time1-rr_time1_n)<-10) && ((rr_time2-rr_time2_n)>10) && ((rr_diff-rr_diff_n) > 20) &&((tp-tp_n)==-1);
    case5 = (( ((rr_time1-rr_time1_n)<-10) && ((rr_time2-rr_time2_n)>10) && ((rr_diff-rr_diff_n) > 20) )||((tp-tp_n)==-1)) && ( (sq-sq_n) > 80 || abs(sp -sp_n)>80);
    case6 = (t_pole - t_pole_n) == -2 ;
%     case7 = ((tp-tp_n)==-1) && ((tp_post-tp_post_n)==-1;
    case7 = ((tp-tp_n)==-1); 
    case8 = (ann_pre == 2) && (abs(rr_diff- rr_diff_n)<20) && ((rr_time1-rr_time1_n)<-20) && ((rr_time2-rr_time2_n)<-20);
    case9 = ((rr_time1-rr_time1_n)<-30);
    case10 = abs(rp-rp_n)>300;
    case11 = tr>1.5;
    case12 = abs(sq_p-sq_p_n )>80;
    case13 = (sq-sq_n>20) && (rp-rp_n>30);
    case14 = abs(s_slope-s_slope_n) >30;
    case15 = abs(s_slope-s_slope_n) >5 && abs(sq-sq_n )>10 && abs(rp-rp_n)>20;
%     case15 = abs(s_slope-s_slope_n) >5 && abs(sq-sq_n )>10;
    case16 = pr_inter>0.6*rr_time1;
    case17 = (abs(s_slope<5)) && (sq>140);
    case18 = (abs(rp-rp_n)>50) && (abs(t_peak-t_peak_n)>30); 
    case19 = (rr_time1-rr_time_mean_long<-15) && (rr_time2-rr_time_mean_long<-15);
    case20 = (abs(s_slope-s_slope_n) >3) && (abs(sq-sq_n )>3) && (t_pole==-1);
    case21 = number_rpeak>1 && (t_pole==-1);
%     case20 = sq-sq_n>30;
%     case12 = (rp/rp_n)<0.25;
    if case1 || case3|| case4|| case5 || case8  || case9 || case7 
        annot_predict = [annot_predict 2];
    elseif case2 || case6 || case10 ||case11 || case12 ||case13 ||case14 || case15 ||case16 || case17 || case20 || case18||case21
        annot_predict = [annot_predict 3];
    else
        annot_predict = [annot_predict 1];
    end

end
end
