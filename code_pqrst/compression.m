function ids_final = compression(seg, TH)
% seg: [1,250]
% TH slope_vary
% ****************figure b***********************
   slope_pre = (seg(4:end-3) - seg(1:end-6));
   slope_post = (seg(7:end) - seg(4:end-3));

   ids_pre = find(abs(slope_pre)>abs(slope_post));
   slope = [];
   for i = 1:length(slope_pre)
       if ismember(i,ids_pre)
           slope = [slope slope_pre(i)];
       else
           slope = [slope slope_post(i)];
       end
   end
   slo_changes = abs(slope(2:end) - slope(1:end-1));
   ids = find(abs(slo_changes) > TH);
  
 % ****************figure c-e***********************
       ids_seg = ids + 3; % corresponding seg ids
       % determine peak or valley
       turning_point_type = [];
       for i = 1: length(ids_seg)
           if ((seg(ids_seg(i)+1)-seg(ids_seg(i)))-(seg(ids_seg(i))-seg(ids_seg(i)-1)))<0 %valley
               turning_point_type = [turning_point_type 1];
           else 
               turning_point_type = [turning_point_type -1];
           end
       end
       ids_final = [];
       turning_point_type_final = [];
       ids_final_temp = ids_seg(1);
       turning_point_temp = turning_point_type(1);
       for i = 2:length(ids)
           if ((ids(i)-ids_final_temp)<25) & (turning_point_type(i) ==turning_point_temp)
               if (seg(ids_seg(i)) > seg(ids_final_temp))
                   if (turning_point_type(i)==1) % peak

                       ids_final_temp = ids_seg(i);
                       turning_point_temp = turning_point_type(i);
                   else
     
                           if ~ismember(ids_final_temp,ids_final)
                               ids_final = [ids_final ids_final_temp];
                               turning_point_type_final = [turning_point_type_final ids_final_temp];  
                           end
                   end
               else
                   if (turning_point_type(i)==-1) % valley
                       ids_final_temp = ids_seg(i);
                       turning_point_temp = turning_point_type(i);
                   else
                    if seg(ids_final_temp)>-20
                       if ~ismember(ids_final_temp,ids_final)
                           ids_final = [ids_final ids_final_temp];
                           turning_point_type_final = [turning_point_type_final ids_final_temp]; 
                       end
                    end
                   end 
               end
           else
               if ((seg(ids_final_temp)>-20) & (turning_point_temp ==1)) |(turning_point_temp ==-1)
                   if ~ismember(ids_final_temp,ids_final)
                       ids_final = [ids_final ids_final_temp];
                       turning_point_type_final = [turning_point_type_final turning_point_temp];
                   end
               end
               ids_final_temp = ids_seg(i);
               turning_point_temp = turning_point_type(i);
           end                                                                
       end
   
end