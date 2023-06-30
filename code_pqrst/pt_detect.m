function [t_pre,p_cur,t_cur,p_post]=pt_detect(ecg_h,q_on,s_off,qrs_i_raw,pole_buf)
t_pre_locs = [];
t_pre_amps = [];
t_pre_pole = [];
p_cur_locs = [];
p_cur_amps = [];
p_cur_pole = [];
t_cur_locs = [];
t_cur_amps = [];
t_cur_pole = [];
p_post_locs = [];
p_post_amps = [];
p_post_pole = [];

for i= 2:length(qrs_i_raw)-2
    if s_off(i+1)<length(ecg_h)
    sq_pre = ecg_h(s_off(i-1)+10:q_on(i)-5);
    sq_post = ecg_h (s_off(i)+10:q_on(i+1)-5);

    [pks_pre,locs_pre,poles_pre] = find_pt_v1(sq_pre,s_off(i-1)+10,ecg_h(s_off(i-1)),ecg_h(q_on(i)),pole_buf(i));
    
    [pks_post,locs_post,poles_post] = find_pt_v1(sq_post,s_off(i)+10,ecg_h(s_off(i)),ecg_h(q_on(i+1)),pole_buf(i));
    t_pre_locs = [t_pre_locs locs_pre(1)];
    t_pre_amps = [t_pre_amps pks_pre(1)];
    t_pre_pole = [t_pre_pole poles_pre(1)];
    p_cur_locs = [p_cur_locs locs_pre(end)];
    p_cur_amps = [p_cur_amps pks_pre(end)];
    p_cur_pole = [p_cur_pole poles_pre(end)];
    t_cur_locs = [t_cur_locs locs_post(1)];
    t_cur_amps = [t_cur_amps pks_post(1)];
    t_cur_pole = [t_cur_pole poles_post(1)];
    p_post_locs = [p_post_locs locs_post(end)];
    p_post_amps = [p_post_amps pks_post(end)];
    p_post_pole = [p_post_pole poles_post(end)];
    end
end
p_post =[p_post_locs;p_post_amps;p_post_pole];
t_cur=[t_cur_locs;t_cur_amps;t_cur_pole];
p_cur=[p_cur_locs;p_cur_amps;p_cur_pole];
t_pre =[t_pre_locs;t_pre_amps;t_pre_pole];


% for i = 1: length(qrs_i_raw)
%     if i > 8
%     diffRR = qrs_i_raw(i)-qrs_i_raw(i-9);
%     else
%     diffRR = qrs_i_raw(i+9)-qrs_i_raw(1);
%     end
%     meanRR = mean(diffRR);
%     if q_on(i)-p_win*meanRR>0
%        [y_i,x_i] = max(ecg_h(q_on(i)-p_win*meanRR:q_on(i)));
%        p_locs = [p_locs q_on(i)-p_win*meanRR+x_i-1];
%        p_amps = [p_amps y_i];
%     else
%        [y_i,x_i] = max(1:q_on(i));
%        p_locs = [p_locs x_i];
%        p_amps = [p_amps y_i];     
%     end
%     if s_off(i)+t_win*meanRR<length(ecg_h)
%        [y_max,x_max] = max(ecg_h(s_off(i):(s_off(i)+t_win*meanRR)));
%        [y_min,x_min] = min(ecg_h(s_off(i):(s_off(i)+t_win*meanRR)));
%        if (y_max > t_pos) & (y_min < t_neg)
%            p_locs = [p_locs q_on(i)-p_win*meanRR+x_i-1];
%            p_amps = [p_amps y_i];
%            t_pole = 
%        elseif y_min < t_neg
%            p_locs = [p_locs q_on(i)-p_win*meanRR+x_i-1];
%            p_amps = [p_amps y_i];
%            
%     else
%        [y_i,x_i] = max(1:q_on(i));
%        p_locs = [p_locs x_i];
%        p_amps = [p_amps y_i];     
%     end
%         
% end
end