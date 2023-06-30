function [feature_normal]=find_normal_features(annot,feature_matrix)
normal_index = find(annot ==1);
m = size(feature_matrix,1);
if length(normal_index)== 0
    feature_normal = zeros(m);
else
    annot_normal = normal_index(1:min_2(8,length(normal_index)));
    rp_normal = mean(feature_matrix(1,annot_normal));
    rr_time_normal = mean(feature_matrix(2:3,annot_normal),2);
    rr_diff_normal = mean(feature_matrix(4,annot_normal));
    qs_normal = mean(feature_matrix(5,annot_normal));
    r_pole_normal = feature_matrix(6,annot_normal(1));
    t_pole_normal = feature_matrix(7,annot_normal(1));
    tp_normal = feature_matrix(8,annot_normal(1));
    % r_pole_normal = 1;
%     t_pole_normal = 1;

    % tp_normal = mean(feature_matrix(8,annot_normal));
    tr_ratio_normal = mean(feature_matrix(9,annot_normal));
    pr_ratio_normal = mean(feature_matrix(10,annot_normal));
    sq_diff_normal = mean(feature_matrix(11,annot_normal));

    sp_normal = mean(feature_matrix(12,annot_normal));
    qp_normal = mean(feature_matrix(13,annot_normal));
    tp_post_normal = feature_matrix(14,annot_normal(1));
    s_slope_normal = mean(feature_matrix(15,annot_normal));
    pr_inter_normal = mean(feature_matrix(16,annot_normal));
    t_peak_normal = mean(feature_matrix(18,annot_normal));
    
    feature_normal = [rp_normal;rr_time_normal; rr_diff_normal;qs_normal;r_pole_normal;t_pole_normal;tp_normal;tr_ratio_normal;pr_ratio_normal;sq_diff_normal;sp_normal;qp_normal;tp_post_normal;s_slope_normal;pr_inter_normal;t_peak_normal];
end
end