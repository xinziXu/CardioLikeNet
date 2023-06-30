function [t,p]= find_pt_v2(signal,start_index,s_off,q_on,pole)
t = [];
p = [];
% 如果p不存在，那index = q_on,value =-2000 pole = 0; t不存在，那index = s_off, value =
% -2000, pole = 0
%                         p(2)  = -2000;
%                         p(1) = start_index+length(signal)+5;
%                         p(3)  = 0;
%                         t(2)  = -2000;
%                         t(1) = start_index-10;
%                         t(3)  = 0;
ecg_m = smooth(signal, 16);
% ecg_m = signal;
% figure()
% plot(signal)
% hold on;
% plot(ecg_m)
% pause(1)
if pole == 1
    if q_on-s_off>40
        s_off = min_2(s_off + 5,-10);
%         thres_neg = max(s_off-5,-20);
        thres_neg = s_off-10;
        thres_pos = min_2(q_on+5,0) ;
    else
        s_off = min_2(s_off+5,-10);
%         thres_neg = max(s_off-5,-20);
        thres_neg = s_off-10;
        thres_pos =min_2(q_on+5,0) ;
    end
else
    s_off = min_2(s_off,0);
    thres_neg = max(s_off-60,-40);
    thres_pos = min_2(q_on-10,0);
end

% if pole == 1
%     if q_on-s_off>40
%         s_off = min_2(s_off + 5,10);
%         thres_neg = max(s_off-10,-40);
%         thres_pos = min_2(q_on+5,10); 
%     else
%         s_off = min_2(s_off+5,10);
%         thres_neg = max(s_off -5,-40);
% 
%         thres_pos =min_2(q_on+5,10); 
%     end
% else
%     s_off = min_2(s_off,0);
%     thres_neg = max(s_off-60,-40);
%     thres_pos = min_2(q_on-10,0); 
% end
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
                        
                        t(2)  = min;
                        t(1) = start_index+IndMin(pos);
                        t(3)  = -1;
                        p(2)  = -2000;
                        p(1) = start_index+length(signal)+5;
                        p(3)  = 0;
                    else
                        t(2)  = min;
                        t(1) = start_index+IndMin(pos);
                        t(3)  = -1;
                        p(2)  = -2000;
                        p(1) = start_index+length(signal)+5;
                        p(3)  = 0;                        
                    end
                else
                    t(2)  = max1;
                    t(1) = start_index+IndMax(IndMax2(pos1));
                    t(3)  = 1;
                    p(2)  = -2000;
                    p(1) = start_index+length(signal)+5;
                    p(3)  = 0;                 
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
                    t(2)  = max1;
                    t(1) = start_index+IndMax(IndMax2(pos1));
                    t(3)  = 1;                                
                    p(2)  = -2000;
                    p(1) = start_index+length(signal)+5;
                    p(3)  = 0;                
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
      
                    if (min < thres_neg-10)

                        t(2)  = min;
                        t(1) = start_index+IndMin(pos);
                        t(3)  = -1;
                        p(2)  = max1;
                        p(1) = start_index+IndMax(IndMax2(p_index(pos1)));
                        p(3)  = 1;


                    else
                        t(2)  = max2;
                        t(1) = start_index+IndMax(IndMax2(t_index(pos2)));
                        t(3)  = 1;
                        p(2)  = max1;
                        p(1) = start_index+IndMax(IndMax2(p_index(pos1)));
                        p(3)  = 1;
                    end
                else
                    t(2)  = max2;
                    t(1) = start_index+IndMax(IndMax2(t_index(pos2)));
                    t(3)  = 1;
                    p(2)  = max1;
                    p(1) = start_index+IndMax(IndMax2(p_index(pos1)));
                    p(3)  = 1;
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
                            t(2)  = min;
                            t(1) = start_index+IndMin(pos);
                            t(3)  = -1;
                            p(2)  = max1;
                            p(1) = start_index+IndMax(IndMax2(pos1));
                            p(3)  = 1;

                        else
                            t(2)  = min;
                            t(1) = start_index+IndMin(pos);
                            t(3)  = -1;
                            p(2)  = -2000;
                            p(1) = start_index+length(signal)+5;
                            p(3)  = 0;
                        end                    
                    else
                        t(2)  = max1;
                        t(1) = start_index+IndMax(IndMax2(pos1));
                        t(3)  = 1;
                         
                        p(2)  = -2000;
                        p(1) = start_index+length(signal)+5;
                        p(3)  = 0;                        
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
                t(2)  = max1;
                t(1) = start_index+IndMax(IndMax2(pos1));
                t(3)  = 1;
                p(2)  = -2000;
                p(1) = start_index+length(signal)+5;
                p(3)  = 0;             
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
                            t(2)  = min;
                            t(1) = start_index+IndMin(pos);
                            t(3)  = -1;
                            p(2)  = max1;
                            p(1) = start_index+IndMax(IndMax2(pos1));
                            p(3)  = 1;
%                             pks = [min max1];
%                             locs = [start_index+IndMin(pos) start_index+IndMax(IndMax2(pos1))];
%                             poles = [-1 1];
                        else
                            t(2)  = min;
                            t(1) = start_index+IndMin(pos);
                            t(3)  = -1;
                            p(2)  = -2000;
                            p(1) = start_index+length(signal)+5;
                            p(3)  = 0;                           
%                             pks = [min min];
%                             locs = [start_index+IndMin(pos) start_index+IndMin(pos)];
%                             poles = [-1 -1];
                         end 
                    else
                        t(2)  = max1;
                        t(1) = start_index+IndMax(IndMax2(pos1));
                        t(3)  = 1;
                        p(2)  = -2000;
                        p(1) = start_index+length(signal)+5;
                        p(3)  = 0;                       
%                         pks = [max1 max1];
%                         locs = [start_index+IndMax(IndMax2(pos1)) start_index+IndMax(IndMax2(pos1))];
%                         poles = [1 1];   
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
                t(2)  = max1;
                t(1) = start_index+IndMax(IndMax2(pos1));
                t(3)  = 1;
                p(2)  = -2000;
                p(1) = start_index+length(signal)+5;
                p(3)  = 0;
%                 pks = [max1 max1];
%                 locs = [start_index+IndMax(IndMax2(pos1)) start_index+IndMax(IndMax2(pos1))];
%                 poles = [1 1];   
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
            t(2)  = min;
            t(1) = start_index+IndMin(pos);
            t(3)  = -1;
            p(2)  = -2000;
            p(1) = start_index+length(signal)+5;
            p(3)  = 0;           
%             pks = [min min];
%             locs = [start_index+IndMin(pos) start_index+IndMin(pos)];
%             poles = [-1 -1];   
        else %% 最后要改为很大的值，表示没有pt波
        if length(signal)==0
            return
%         pks = [0 0];
%         locs = [start_index start_index];
%         poles = [1 1];            
       
        else
           
        [max1,pos1]=max(signal);
        t(2)  = max1;
        t(1) = start_index+pos1;
        t(3)  = 1; 
        p(2)  = -2000;
        p(1) = start_index+length(signal)+5;
        p(3)  = 0;        
%         pks = [max1 max1];
%         locs = [start_index+pos1 start_index+pos1];
%         poles = [1 1];
        end
        end
    else
        if length(signal)==0
            
        t(2)  = -2000;
        t(1) = start_index-10;
        t(3)  = 0; 
        p(2)  = -2000;
        p(1) = start_index+length(signal)+5;
        p(3)  = 0;      
%         pks = [0 0];
%         locs = [start_index start_index];
%         poles = [1 1];            
       
        else
        [max1,pos1]=max(signal);
        t(2)  = max1;
        t(1) = start_index+pos1;
        t(3)  = 1; 
        p(2)  = -2000;
        p(1) = start_index+length(signal)+5;
        p(3)  = 0;        
%         pks = [max1 max1];
%         locs = [start_index+pos1 start_index+pos1];
%         poles = [1 1];
        end

    end

end

%     
    
end