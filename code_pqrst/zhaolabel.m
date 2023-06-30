function [annot_new] = zhaolabel(qrs_i_raw,rpeak_time,ANNOT)
    length_predict = length(qrs_i_raw);
    length_true = length(rpeak_time);
    now = 1;
    
    annot_new = [];
    
    for i = 1:length_predict
        diff = inf;
        annot = 0;
        for j = now:length_true
            temp = rpeak_time(j) - qrs_i_raw(i);
            if abs(temp) < abs(diff)
                diff = temp;
                annot = ANNOT(j);
            elseif temp > 0
                now = max(j-1,1);
                break;
            end
        end
        annot_new = [annot_new; annot];
    end

end