function [q_on,s_off]=qrs_detect_min(ecg_h,qrs_i_raw,pole_buf,s_width,q_width)
q_on = [];
s_off = [];
for i=1:length(qrs_i_raw)
   
    if pole_buf(i)==1
    if qrs_i_raw(i)+round(s_width*256)>length(ecg_h)
        [y_s,x_s] = find_s(ecg_h(qrs_i_raw(i)+1:end),pole_buf(i));
%         [y_s, x_s] = min(ecg_h(qrs_i_raw(i)+1:end));
        [y_q, x_q] = find_q(ecg_h(qrs_i_raw(i)-round(q_width*256):qrs_i_raw(i)-1),pole_buf(i));
%         [y_q, x_q] = min(ecg_h(qrs_i_raw(i)-round(q_width*256):qrs_i_raw(i)-1));
        s_index = qrs_i_raw(i)+x_s-1;
        q_index = qrs_i_raw(i)-round(q_width*256)+x_q-1;
    elseif qrs_i_raw(i)-round(q_width*256)<0
        [y_s, x_s] = find_s(ecg_h(qrs_i_raw(i)+1:qrs_i_raw(i)+round(s_width*256)),pole_buf(i));
%         [y_s, x_s] = min(ecg_h(qrs_i_raw(i)+1:qrs_i_raw(i)+round(s_width*256)));
        [y_q, x_q] = find_q(ecg_h(1:qrs_i_raw(i)-1),pole_buf(i));
%         [y_q, x_q] = min(ecg_h(1:qrs_i_raw(i)-1));
        s_index = qrs_i_raw(i)+x_s-1;
        q_index = x_q;   
    else
        [y_s, x_s] = find_s(ecg_h(qrs_i_raw(i)+1:qrs_i_raw(i)+round(s_width*256)),pole_buf(i));
%         [y_s, x_s] = min(ecg_h(qrs_i_raw(i)+1:qrs_i_raw(i)+round(s_width*256)));
        [y_q, x_q] = find_q(ecg_h(qrs_i_raw(i)-round(q_width*256):qrs_i_raw(i)-1),pole_buf(i));
%         [y_q, x_q] = min(ecg_h(qrs_i_raw(i)-round(q_width*256):qrs_i_raw(i)-1));
        s_index = qrs_i_raw(i)+x_s-1;
        q_index = qrs_i_raw(i)+x_q-1-round(q_width*256);
    end 

    else
    if qrs_i_raw(i)+round(s_width*256)>length(ecg_h)
        [y_s, x_s] = find_s(ecg_h(qrs_i_raw(i)+1:end),pole_buf(i));
%         [y_s, x_s] = max(ecg_h(qrs_i_raw(i)+1:end));
        [y_q, x_q] = find_q(ecg_h(qrs_i_raw(i)-round(q_width*256):qrs_i_raw(i)-1),pole_buf(i));
%         [y_q, x_q] = max(ecg_h(qrs_i_raw(i)-round(q_width*256):qrs_i_raw(i)-1));
        s_index = qrs_i_raw(i)+x_s-1;
        q_index = qrs_i_raw(i)+x_q-1-round(q_width*256);
    elseif qrs_i_raw(i)-round(q_width*256)<0
        [y_s, x_s] = find_s(ecg_h(qrs_i_raw(i)+1:qrs_i_raw(i)+round(s_width*256)),pole_buf(i));
%         [y_s, x_s] = max(ecg_h(qrs_i_raw(i)+1:qrs_i_raw(i)+round(s_width*256)));
        [y_q, x_q] = find_q(ecg_h(1:qrs_i_raw(i)-1),pole_buf(i));
%         [y_q, x_q] = max(ecg_h(1:qrs_i_raw(i)-1));
        s_index = qrs_i_raw(i)+x_s-1;
        q_index = x_q;   
    else
        [y_s, x_s] = find_s(ecg_h(qrs_i_raw(i)+1:qrs_i_raw(i)+round(s_width*256)),pole_buf(i));
%         [y_s, x_s] = max(ecg_h(qrs_i_raw(i)+1:qrs_i_raw(i)+round(s_width*256)));
        [y_q, x_q] = find_q(ecg_h(qrs_i_raw(i)-round(q_width*256):qrs_i_raw(i)-1),pole_buf(i));
%         [y_q, x_q] = max(ecg_h(qrs_i_raw(i)-round(q_width*256):qrs_i_raw(i)-1));
        s_index = qrs_i_raw(i)+x_s-1;
        q_index = qrs_i_raw(i)+x_q-1-round(q_width*256);        

    end
    end
    s_off=[s_off;s_index];
    q_on=[q_on;q_index];
end
end