function [y_s,x_s] = find_s(signal,pole)
signal = smooth(signal,4);
if pole == 1
%     Indmin = find((diff(diff(signal))>0.5))+1;
    Indmin = find((diff(sign(diff(signal)))>0))+1;
    PksMin = signal(Indmin);
    if length(Indmin)>0
        IndMin2 = find(signal(Indmin)<20);
        if length(IndMin2)>0
            y_s = signal(Indmin(IndMin2(1)));
            x_s = Indmin(IndMin2(1));
        else
            [y_s,x_s] = min(signal);  
        end
    else
        [y_s,x_s] = min(signal);  
    end
else
    Indmax = find((diff(diff(signal))<-0.2))+1;
%     Indmax = find((diff(sign(diff(signal)))<0))+1;
    PksMax = signal(Indmax);
    if length(Indmax)>0
        IndMax2 = find(signal(Indmax)>-20);
        if length(IndMax2)>0
            y_s = signal(Indmax(IndMax2(1)));
            x_s = Indmax(IndMax2(1));
        else
            [y_s,x_s] = max(signal);  
        end
    else
        [y_s,x_s] = max(signal);  
    end        
end
end
