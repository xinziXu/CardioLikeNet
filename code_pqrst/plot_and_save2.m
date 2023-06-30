function plot_and_save2(ecg1, filenum, r_peak_time1,ANNOT,ecg2,r_peak_time2,q_on,s_off,p_on,t_off,folder_name,step)
    length_1 = size(ecg1, 1);
    length_2 = size(ecg2, 1);
%     step = 1500;
    i_1 = 1;
    i_2 = 1;
    while(1)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        start_1 = step*(i_1-1) + 1;
        finish_1 =  min(step*i_1,length_1);
        x = [start_1:finish_1];
        y = ecg1(start_1:finish_1);
        

        subplot(2,1,1)
        plot(x,y,'LineWidth',4,'color','b');
        for j_1 = [1: size(r_peak_time1,1)]
            peak_time_1 = r_peak_time1(j_1);
            number = ANNOT(j_1);
            if start_1 <= peak_time_1 & peak_time_1 <= finish_1
                
                text(x(peak_time_1-start_1+1)-15,y(peak_time_1-start_1+1),'o','FontSize',25,'color','r');
                text(x(peak_time_1-start_1+1),y(peak_time_1-start_1+1)+20,num2str(number),'FontSize',20,'color','k');
                text(x(peak_time_1-start_1+1),1000,num2str(j_1),'FontSize',20,'color','k');
            end
            if peak_time_1 > finish_1
                break;
            end
        end
        title(['第' num2str(filenum) '个人的第' num2str(i_1) '张图, ecg1']);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        start_2 = step*(i_2-1) + 1;
        finish_2 =  min(step*i_2,length_2);
        x = [start_2:finish_2];
        y = ecg2(start_2:finish_2);
        
        subplot(2,1,2)
        plot(x,y,'LineWidth',4,'color','b');
        if size(p_on,2)>size(t_off,2)
            plot_length = size(t_off,2);
        else
            plot_length = size(p_on,2);
        end
            
        for j_2 = 1: plot_length

            peak_time_2 = r_peak_time2(j_2);
            q = q_on(j_2);
            s = s_off(j_2);
            p = p_on(j_2);
            t = t_off(j_2);
%             if start_3 <= peak_time_3 & peak_time_3 <= finish_3 &start_3 <= s & s <= finish_3&start_3 <= q & q <= finish_3
            if start_2 <= peak_time_2 & peak_time_2 <= finish_2
                text(x(peak_time_2-start_2+1)-15,y(peak_time_2-start_2+1),'o','FontSize',25,'color','r');
            end
            if start_2 <= q & q <= finish_2
                text(x(q-start_2+1)-15,y(q-start_2+1),'+','FontSize',25,'color','y');
            end
            if start_2 <= s & s <= finish_2
                text(x(s-start_2+1)-15,y(s-start_2+1),'+','FontSize',25,'color','c');
            end
            if start_2 <= p & p <= finish_2
                text(x(p-start_2+1)-15,y(p-start_2+1),'*','FontSize',25,'color','m');
            end
            if start_2 <= t & t <= finish_2
                text(x(t-start_2+1)-15,y(t-start_2+1),'*','FontSize',25,'color','g');
            end   
        end

        title(['第' num2str(filenum) '个人的第' num2str(i_2) '张图, ecg2']);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        folder = ['./' folder_name '/' num2str(filenum)];
        if ~exist(folder,'dir')
            mkdir(folder)
        end
        saveas(gcf,[folder,'/',[num2str(filenum) '_' num2str(i_1)],'.jpg']);
        i_1 = i_1 + 1;
        i_2 = i_2 + 1;
        if finish_1 >= length_1 || finish_2 >= length_2
            break;
        end
    end
end