function [tn,fp,fn,tp,acc,tnr] = cal_metrics(label, predict)
    tn = 0;
    fp = 0;
    fn = 0;
    tp = 0;
    len = length(label);
    
    for i=1:len
        if label(i) ~= 1
            label(i) =-1;
            
        end
        if predict(i) ~= 1
            predict(i) =-1;
            
        end
        if (label(i) == predict(i)) && (label(i) == 1)
            tp = tp + 1;
        elseif (label(i) == predict(i)) && (label(i) ~= 1)
            tn = tn + 1;
        elseif (label(i) ~= predict(i)) && (label(i) == 1)
            fn = fn + 1;
            
        elseif (label(i) ~= predict(i)) && (label(i) ~= 1)
            fp = fp + 1;
            i+10

        end
    end
    acc = (tp+tn)/len;

    tnr = 1-fp/len;

end