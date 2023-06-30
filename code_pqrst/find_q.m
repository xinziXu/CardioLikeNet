function [y_s,x_s] = find_q(signal,pole)
signal = smooth(signal,4);
if pole == 1
    Indmin = find((diff(sign(diff(signal)))>0))+1;
    PksMin = signal(Indmin);
    if length(Indmin)>0
        IndMin2 = find(signal(Indmin)<20);
        if length(IndMin2)>0
            y_s = signal(Indmin(IndMin2(end)));
            x_s = Indmin(IndMin2(end));
        else
            [y_s,x_s] = min(signal);  
        end
    else
        [y_s,x_s] = min(signal);  
    end
else
    Indmax = find((diff(sign(diff(signal)))<0))+1;
    PksMax = signal(Indmax);
    if length(Indmax)>0
        IndMax2 = find(signal(Indmax)>-30);
        if length(IndMax2)>0
            y_s = signal(Indmax(IndMax2(end)));
            x_s = Indmax(IndMax2(end));
        else
            [y_s,x_s] = max(signal);  
        end
    else
        [y_s,x_s] = max(signal);  
    end        
end
end
