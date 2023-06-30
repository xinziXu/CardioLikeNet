function y = MedianFilt_v1(x,R,inteval)
% iCount = 1;
% 
% for iSampleIndex = startSample:endSample
%     
%     startRange = iSampleIndex-MidPoint+1;
%     iRange = startRange:startRange+WinSize-1;
%     
%     Output(iCount,:) = median( Input(iRange,:) );
%     
%     iCount = iCount + 1;   
% end
% end


% function [y] = myMedfilt(x, R)
y = x;
for i = 1:length(x)
    if ((i+R)<= length(x) && (i-R)>= 1)
        BL = median(x((i-R):inteval:(i+R)));
%         length(x((i-R):inteval:(i+R)))
    elseif ((i+R)<= length(x) && (i-R)< 1)
        BL = median(x(1:inteval:(i+R)));
    elseif ((i+R)> length(x) && (i-R)>= 1)
        BL = median(x((i-R):inteval:end));
    end
    y(i) = y(i)- BL;
end
