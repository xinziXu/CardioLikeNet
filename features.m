%DS1 = [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230];
%DS2 = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234];
DS1 = [101];
DS2 = [100];
lines = [];
for i = 1:1
    fprintf(strcat('--',num2str(i),'--\n'));
    seg_file = strcat('./seg_file/',num2str(DS1(i)),'_seg.mat'); % Determine the seg_file to load
    load(seg_file,'lines'); % Load into the value of lines from the seg_file
%     for j = 1:size(lines,1) % For each row in data
    for j = 1 
       % Add compression code here
       rr_info = lines(j,1:3); % The first 3 items is the rr info (label, preRR, postRR
       seg = lines(j,4:end); % The remaining items are the segment info
%        figure;
%        plot(seg)
        
       ids_final = compression(seg, 2);
       plot_compressed_signal(seg,ids_final);
    end 
end

% concatenate rr_info with compressed seg and save to file

train_file = 'train_features_all.txt';
csvwrite(train_file,lines); % Write the lines to file

lines = [];
for i = 1:1
    fprintf(strcat('--',num2str(i),'--\n'));
    seg_file = strcat('./seg_file/',num2str(DS2(i)),'_seg.mat');
    load(seg_file,'lines');
    for j = 1:size(lines,1)
       rr_info = lines(j,1:3);
       seg = lines(j,4:end);
    end 
end
test_file = 'test_features_all.txt';
csvwrite(test_file,lines);

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

function plot_compressed_signal(seg, ids)

figure;
plot(seg,'LineWidth',4,'color','b');
for i = 1: length(ids)
    text(ids(i), seg(ids(i)),'o','FontSize',15,'color','r');
end

end