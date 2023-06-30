function [pks,locs,poles,ecg_m]= find_pt_v1(signal,start_index,s_off,q_on,pole)
pks = [];
locs = [];
poles = [];

ecg_m = smooth(signal, 16);
% ecg_m = signal;
% figure()
% plot(signal)
% hold on;
% plot(ecg_m)
% pause(1)
if pole == 1
    if q_on-s_off>40
        s_off = min_2(s_off + 60,10);
        thres_neg = max(s_off-10,-40);
        thres_pos = min_2(q_on+5,10); 
    else
        s_off = min_2(s_off+5,10);
        thres_neg = max(s_off -5,-40);

        thres_pos =min_2(q_on+5,10); 
    end
else
    s_off = min_2(s_off,0);
    thres_neg = max(s_off-60,-40);
    thres_pos = min_2(q_on-10,0); 
end
%%%%%%%%%%%%%%%%%%%原来的方式%%%%%%%%%%%%%%%%%%5
% if q_on-s_off>40
%     s_off = min_2(s_off + 60,10);
%     thres_neg = max(s_off-60,-40);
%     thres_pos = min_2(q_on+5,10); 
% else
%     s_off = min_2(s_off+5,10);
%     thres_neg = max(s_off -40,-40);
% 
%     thres_pos =min_2(q_on+5,10); 
% end

IndMax = find(diff(sign(diff(ecg_m))) < 0)+1; %波峰峰值点
IndMax2 = find(signal(IndMax) > s_off);%筛选出大于s波高度的峰
IndMin = find(diff(sign(diff(ecg_m))) > 0)+1;%波谷谷值点



if length(IndMax2)>0
        p_index = find(IndMax(IndMax2)>0.6*length(signal));
        t_index = find(IndMax(IndMax2)<0.6*length(signal));
        if length(p_index) == 0
            PksMax = signal(IndMax(IndMax2));
            [max1, pos1] = max(PksMax);
%             PksMax(pos1) = -inf;
%             [max2, pos2] = max(PksMax);
            if length(IndMin)~=0
                PksMin = signal(IndMin);
                [min, pos] = max(-PksMin);
                min = -min; 
                if (min < thres_neg)
                    if IndMax(IndMax2(pos1)) > IndMin(pos)
                        pks = [min max1];
                        locs = [start_index+IndMin(pos) start_index+IndMax(IndMax2(pos1))];
                        poles = [-1 1];
                    else
                        pks = [min min];
                        locs = [start_index+IndMin(pos) start_index+IndMin(pos)];
                        poles = [-1 -1];
                    end
                else
                    pks = [max1 max1];
                    locs = [start_index+IndMax(IndMax2(pos1)) start_index+IndMax(IndMax2(pos1))];
                    poles = [1 1];                   
%                     if IndMax(IndMax2(pos1)) > IndMax(IndMax2(pos2))
%                         pks = [max2 max1];
%                         locs = [start_index+IndMax(IndMax2(pos2)) start_index+IndMax(IndMax2(pos1))];
%                         poles = [1 1];
%                     else
%                         pks = [max1 max2];
%                         locs = [start_index+IndMax(IndMax2(pos1)) start_index+IndMax(IndMax2(pos2))];
%                         poles = [1 1];    
%                     end 
                end
            else
                pks = [max1 max1];
                locs = [start_index+IndMax(IndMax2(pos1)) start_index+IndMax(IndMax2(pos1))];
                poles = [1 1];                   
                
%                 if IndMax(IndMax2(pos1)) > IndMax(IndMax2(pos2))
%                     pks = [max2 max1];
%                     locs = [start_index+IndMax(IndMax2(pos2)) start_index+IndMax(IndMax2(pos1))];
%                     poles = [1 1];
%                 else
%                     pks = [max1 max2];
%                     locs = [start_index+IndMax(IndMax2(pos1)) start_index+IndMax(IndMax2(pos2))];
%                     poles = [1 1];    
%                 end
            end
        elseif length(p_index) > 0 && max(signal(IndMax(IndMax2(p_index))))>thres_pos
            
            if length(t_index) > 0
                 PksMaxp = signal(IndMax(IndMax2(p_index)));
                [max1,pos1]=max(PksMaxp);
                PksMaxt = signal(IndMax(IndMax2(t_index)));
                [max2,pos2]=max(PksMaxt);
                if length(IndMin)~=0
                    PksMin = signal(IndMin);
                    [min, pos] = max(-PksMin);
                    min = -min;
                    if (min < thres_neg) && (max2<thres_pos)
                        pks = [min max1];
                        locs = [start_index+IndMin(pos) start_index+IndMax(IndMax2(p_index(pos1)))];
                        poles = [-1 1];

                    else
                        pks = [max2 max1];
                        locs = [start_index+IndMax(IndMax2(t_index(pos2))) start_index+IndMax(IndMax2(p_index(pos1)))];
                        poles = [1 1]; 
                    end
                else
                    pks = [max2 max1];
                    locs = [start_index+IndMax(IndMax2(t_index(pos2))) start_index+IndMax(IndMax2(p_index(pos1)))];
                    poles = [1 1]; 
                end
            else
                PksMax = signal(IndMax(IndMax2));
                [max1, pos1] = max(PksMax);
                PksMax(pos1) = -inf;
                [max2, pos2] = max(PksMax);
            if length(IndMin)~=0
                    PksMin = signal(IndMin);
                    [min, pos] = max(-PksMin);
                    min = -min;
                    if min<thres_neg
                        if IndMax(IndMax2(pos1)) > IndMin(pos)
                            pks = [min max1];
                            locs = [start_index+IndMin(pos) start_index+IndMax(IndMax2(pos1))];
                            poles = [-1 1];
                        else
                            pks = [min min];
                            locs = [start_index+IndMin(pos) start_index+IndMin(pos)];
                            poles = [-1 -1];
                        end                    
                    else
                        pks = [max1 max1];
                        locs = [start_index+IndMax(IndMax2(pos1)) start_index+IndMax(IndMax2(pos1))];
                        poles = [1 1];                           
                        
%                         if IndMax(IndMax2(pos1)) > IndMax(IndMax2(pos2))
%                             pks = [max2 max1];
%                             locs = [start_index+IndMax(IndMax2(pos2)) start_index+IndMax(IndMax2(pos1))];
%                             poles = [1 1];
%                         else
%                             pks = [max1 max2];
%                             locs = [start_index+IndMax(IndMax2(pos1)) start_index+IndMax(IndMax2(pos2))];
%                             poles = [1 1];    
%                         end                    
                    end
            else
                pks = [max1 max1];
                locs = [start_index+IndMax(IndMax2(pos1)) start_index+IndMax(IndMax2(pos1))];
                poles = [1 1];                 
%                 if IndMax(IndMax2(pos1)) > IndMax(IndMax2(pos2))
%                     pks = [max2 max1];
%                     locs = [start_index+IndMax(IndMax2(pos2)) start_index+IndMax(IndMax2(pos1))];
%                     poles = [1 1];
%                 else
%                     pks = [max1 max2];
%                     locs = [start_index+IndMax(IndMax2(pos1)) start_index+IndMax(IndMax2(pos2))];
%                     poles = [1 1];    
%                 end
            end 
            end


        elseif length(p_index) > 0 && max(signal(IndMax(IndMax2(p_index))))<=thres_pos
            
            PksMax = signal(IndMax(IndMax2));
            [max1, pos1] = max(PksMax);
            PksMax(pos1) = -inf;
            [max2, pos2] = max(PksMax);
            if length(IndMin)~=0
                    PksMin = signal(IndMin);
                    [min, pos] = max(-PksMin);
                    min = -min;
                    if (min < thres_neg)
                         if IndMax(IndMax2(pos1)) > IndMin(pos)
                            pks = [min max1];
                            locs = [start_index+IndMin(pos) start_index+IndMax(IndMax2(pos1))];
                            poles = [-1 1];
                        else
                            pks = [min min];
                            locs = [start_index+IndMin(pos) start_index+IndMin(pos)];
                            poles = [-1 -1];
                         end 
                    else
                        pks = [max1 max1];
                        locs = [start_index+IndMax(IndMax2(pos1)) start_index+IndMax(IndMax2(pos1))];
                        poles = [1 1];   
%                         if IndMax(IndMax2(pos1)) > IndMax(IndMax2(pos2))
%                             pks = [max2 max1];
%                             locs = [start_index+IndMax(IndMax2(pos2)) start_index+IndMax(IndMax2(pos1))];
%                             poles = [1 1];
%                         else
%                             pks = [max1 max2];
%                             locs = [start_index+IndMax(IndMax2(pos1)) start_index+IndMax(IndMax2(pos2))];
%                             poles = [1 1];    
%                         end
                    end
            else
                pks = [max1 max1];
                locs = [start_index+IndMax(IndMax2(pos1)) start_index+IndMax(IndMax2(pos1))];
                poles = [1 1];   
%                 if IndMax(IndMax2(pos1)) > IndMax(IndMax2(pos2))
%                     pks = [max2 max1];
%                     locs = [start_index+IndMax(IndMax2(pos2)) start_index+IndMax(IndMax2(pos1))];
%                     poles = [1 1];
%                 else
%                     pks = [max1 max2];
%                     locs = [start_index+IndMax(IndMax2(pos1)) start_index+IndMax(IndMax2(pos2))];
%                     poles = [1 1];    
%                 end
            end
        end
    


%     elseif length(IndMax2) == 2
%         max1 = signal(IndMax(IndMax2(1))); % two peak
%         max2= signal(IndMax(IndMax2(end)));
%         pos1 = IndMax2(1);
%         pos2 = IndMax2(end);
%         if length(IndMin)~=0
%                 PksMin = signal(IndMin);
%                 [min, pos] = max(-PksMin);
%                 min = -min;
%                 if (min < thres_neg)
%                     if max1>max2
%                      if IndMax(pos1) > IndMin(pos)
%                         pks = [min max1];
%                         locs = [start_index+IndMin(pos) start_index+IndMax(pos1)];
%                         poles = [-1 1];
%                      else 
%                         pks = [max1 min];
%                         locs = [start_index+IndMax(pos1) start_index+IndMin(pos)];
%                         poles = [1 -1];
%                      end
%                     else
%                          if IndMax(pos2) > IndMin(pos)
%                             pks = [min max2];
%                             locs = [start_index+IndMin(pos) start_index+IndMax(pos2)];
%                             poles = [-1 1];
%                          else 
%                             pks = [max2 min];
%                             locs = [start_index+IndMax(pos2) start_index+IndMin(pos)];
%                             poles = [1 -1];
%                          end
%                     end
% 
%                 elseif max2<thres_pos
%                     pks = [max1 max1];
%                     locs = [start_index+IndMax(pos1) start_index+IndMax(pos1)];
%                     poles = [1 1];                       
%                 else
%                     pks = [max1 max2];
%                     locs = [start_index+IndMax(pos1) start_index+IndMax(pos2)];
%                     poles = [1 1] ;   
%                 end
%         else  
%             pks = [max1 max2];
%             locs = [start_index+IndMax(pos1) start_index+IndMax(pos2)];
%             poles = [1 1];
%         end
%    
%     elseif (length(IndMax2) == 1)
%         pos1 = IndMax2(1);
%         max1 = signal(IndMax(IndMax2(1)));
%         if length(IndMin)~=0
%             PksMin = signal(IndMin);
%             [min, pos] = max(-PksMin);
%             min = -min;
% 
%             if min < thres_neg %one_low,one peak
% 
%                 if IndMax(pos1) > IndMin(pos)
%                     pks = [min max1];
%                     locs = [start_index+IndMin(pos) start_index+IndMax(pos1)];
%                     poles = [-1 1];
%                 else
%                     pks = [max1 min];
%                     locs = [start_index+IndMax(pos1) start_index+IndMin(pos)];
%                     poles = [1 -1];
%                 end 
%             else %one_peak
%                 pks = [max1 max1];
%                 locs = [start_index+IndMax(pos1) start_index+IndMax(pos1)];
%                 poles = [1 1];
% 
%             end
%         else %one_peak
%             pks = [max1 max1];
%             locs = [start_index+IndMax(pos1) start_index+IndMax(pos1)];
%             poles = [1 1];
%         end
%     end
%     
    
elseif length(IndMax2) == 0
    
    if length(IndMin)~=0
        PksMin = signal(IndMin);
        [min, pos] = max(-PksMin);
        min = -min; 
        if  min < thres_neg %one_low
            pks = [min min];
            locs = [start_index+IndMin(pos) start_index+IndMin(pos)];
            poles = [-1 -1];   
        else %% 最后要改为很大的值，表示没有pt波
        if length(signal)==0

        pks = [0 0];
        locs = [start_index start_index];
        poles = [1 1];            
       
        else
        [max1,pos1]=max(signal);
        pks = [max1 max1];
        locs = [start_index+pos1 start_index+pos1];
        poles = [1 1];
        end
        end
    else
        if length(signal)==0

        pks = [0 0];
        locs = [start_index start_index];
        poles = [1 1];            
       
        else
        [max1,pos1]=max(signal);
        pks = [max1 max1];
        locs = [start_index+pos1 start_index+pos1];
        poles = [1 1];
        end

    end

end



    
    
    
    
    
    
    
    
    
    
%     PksMax = signal(IndMax);
%     if length(IndMax)>2 
%         IndMax2 = find(signal(IndMax)>thres_pos);
%         if length(IndMax2)>1      
%             max1 = signal(IndMax(IndMax2(end)));
%             max2= signal(IndMax(IndMax2(1)));
%             pos1 = IndMax2(end);
%             pos2 = IndMax2(1);            
%         else
%             [max1, pos1] = max(PksMax);
%             PksMax(pos1) = -inf;
%             [max2, pos2] = max(PksMax);
%          
%         end
%     else
%         [max1, pos1] = max(PksMax);
%         PksMax(pos1) = -inf;
%         [max2, pos2] = max(PksMax);
% 
%     end
%     case1 = ((max2<=thres_pos) && (max1<=thres_pos));
%     case2 = ( (max2<=thres_pos) && (max1>thres_pos));
%     case3 = ( (max2>thres_pos) && (max1>thres_pos)) ;
% 
%     
% 
% %     two_peak = case3 || case4 || case6 || case7;
% %     one_peak = case5;
% %     peak_low = case2;
% %     one_low = case1;
% 
%     if case3
%         if IndMax(pos1) > IndMax(pos2)
%             pks = [max2 max1];
%             locs = [start_index+IndMax(pos2) start_index+IndMax(pos1)];
%             poles = [1 1];
%         else
%             pks = [max1 max2];
%             locs = [start_index+IndMax(pos1) start_index+IndMax(pos2)];
%             poles = [1 1];
%         end   
%     elseif case2
%         pks = [max1 max1];
%         locs = [start_index+IndMax(pos1) start_index+IndMax(pos1)];
%         poles = [1 1];
%     elseif case1
%         [max1, pos1] = max(PksMax);
%         pks = [max1 max1];
%         locs = [start_index+IndMax(pos1) start_index+IndMax(pos1)];
%         poles = [1 1];
%     end
%         
%   
% 
% 
% elseif length(IndMax)==1
%         PksMax = signal(IndMax);
%         [max1, pos1] = max(PksMax);
%         
%         pks = [max1 max1];
%         locs = [start_index+IndMax(pos1) start_index+IndMax(pos1)];
%         poles = [1 1];
%     
% elseif  length(IndMax)==0
%         if length(signal)==0
% 
%         pks = [0 0];
%         locs = [start_index start_index];
%         poles = [1 1];            
%         
%         else
% 
%         [max1,pos1]=max(signal);
%         pks = [max1 max1];
%         locs = [start_index+pos1 start_index+pos1];
%         poles = [1 1];
%         end
%     
% end





% h_d = [-1 -2 0 2 1]*(1/8);%1/8*fs
% ecg_d = conv (signal ,h_d);
% ecg_d = ecg_d/max(ecg_d);
% 
% ecg_s = ecg_d.^2;
% ecg_m = conv(ecg_s ,ones(1 ,16)/16);
% % plot(ecg_m)
% % pause(1)
% thres = max(ecg_m)*1/4;
% thres1 = max(signal)*1/4;
% thres1_neg = 0;
% [pks_temp,locs_temp] = findpeaks(ecg_m,'MINPEAKDISTANCE',round(0.1*256));
% for i = 1:length(pks_temp)
%     if ecg_m(locs_temp(i))>thres
%         [y_i, x_i, pole]=find_ptpeak(signal, locs_temp(i),256, thres1, thres1_neg);
%        if pole == 1
%               locs = [locs start_index+x_i];% save index of bandpass 
%               pcks = [pcks y_i];
%               poles = [poles 1];           
%        else
%               locs = [locs start_index+x_i];% save index of bandpass 
%               pcks = [pcks y_i];
%               poles = [poles -1];       
%     end
%     end       
% end


% if length(locs)>1
%     locs = [locs(1) locs(end)];
%     pks = [pks(1) pks(end)];
%     poles = [poles(1) poles(end)];
% else
%     locs = [locs(1) locs(1)];
%     pks = [pks(1) pks(1)];
%     poles = [poles(1) poles(1)];
%     
    
end