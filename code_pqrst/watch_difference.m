function [difference_normal,difference_abnormal]=watch_difference(annot,feature_normal,feature_matrix)

len = length(annot);
difference_normal = [];
difference_abnormal = [];
    
for i=1:len
    if annot(i)==1
        difference_normal = [difference_normal feature_matrix(:,i)-feature_normal];
    else
        difference_abnormal = [difference_abnormal feature_matrix(:,i)-feature_normal];
    end
end
        

end