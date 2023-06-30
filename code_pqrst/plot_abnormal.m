function plot_abnormal(signal, annot,r_peak, q_on,s_off,p_cur,t_cur,filenum)
for i = 2:length(annot)-2
    if annot(i) ~= 1 

%         signal(p_cur(i-1)-10:t_cur(i+1)+10)
        
        plot_three_beats(signal,p_cur(i-1)-10,p_cur(i-1:i+1),q_on(i-1:i+1),r_peak(i-1:i+1),s_off(i-1:i+1),t_cur(i-1:i+1),filenum,i,annot(i))
    end
end

end