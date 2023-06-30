function [y_i x_i_h pole]=find_rpeak(ecg_h,locs,fs,THR_SIG, THR_SIG_neg)
THR_SIG = max(THR_SIG,80);

THR_SIG_neg = min_2(THR_SIG_neg, -20);
   if locs <= length(ecg_h) & locs-round(0.150*fs)> 0 
       ecg_max = max(ecg_h(locs-round(0.150*fs):locs));
       ecg_min = min(ecg_h(locs-round(0.150*fs):locs));
       if ecg_max<THR_SIG & ecg_min<THR_SIG_neg
           pole = -1;
           [y_i x_i] = min(ecg_h(locs-round(0.150*fs):locs));
           x_i_h = locs-round(0.150*fs)+x_i -1;
       else
           pole = 1;
           [y_i x_i] = max(ecg_h(locs-round(0.150*fs):locs));
           x_i_h = locs-round(0.150*fs)+x_i -1;
       end
   elseif locs-round(0.150*fs)< 1
       ecg_max = max(ecg_h(1:locs));
       ecg_min = min(ecg_h(1:locs));
       if ecg_max<THR_SIG & ecg_min<THR_SIG_neg
           pole = -1;
           [y_i x_i] = min(ecg_h(1:locs));
           x_i_h = x_i-1;
       else
           pole = 1;
           [y_i x_i] = max(ecg_h(1:locs));
           x_i_h = x_i-1;
       end       
   elseif locs > length(ecg_h)
       ecg_max = max(ecg_h(locs-round(0.150*fs):end));
       ecg_min = min(ecg_h(locs-round(0.150*fs):end));
       if ecg_max<THR_SIG & ecg_min<THR_SIG_neg
           pole = -1;
           [y_i x_i] = min(ecg_h(locs-round(0.150*fs):end));
           x_i_h = locs-round(0.150*fs)+x_i -1;
       else
           pole = 1;
           [y_i x_i] = max(ecg_h(locs-round(0.150*fs):end));
           x_i_h = locs-round(0.150*fs)+x_i -1;
       end

   end


%    if locs <= length(ecg_h)
% %        ecg_diff = diff(ecg_h(locs-round(0.050*fs):locs));
%        ecg_mean = mean(ecg_h(locs-round(0.150*fs):locs));       
%        if ecg_mean>baseline
%            pole = 1;
% %            plot(ecg_h(locs-round(0.150*fs):locs));hold on;
%             [y_i x_i] = max(ecg_h(locs-round(0.150*fs):locs));
%        else
%            pole = -1;
% %            plot(ecg_h(locs-round(0.150*fs):locs));hold on;
%            [y_i x_i] = min(ecg_h(locs-round(0.150*fs):locs));
%        end
%    else
%        ecg_mean = mean(ecg_h(locs-round(0.150*fs):end));
%        if ecg_mean>baseline
%            pole = 1;
%             [y_i x_i] = max(ecg_h(locs-round(0.150*fs):end));
%        else
%            pole = -1;
%            [y_i x_i] = min(ecg_h(locs-round(0.150*fs):end));
%        end       
% 
%    end
end