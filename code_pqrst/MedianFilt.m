function y = MedianFilt(x,R)
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
        BL = median(x((i-R):(i+R)));
    elseif ((i+R)<= length(x) && (i-R)< 1)
        BL = median(x(1:(i+R)));
    elseif ((i+R)> length(x) && (i-R)>= 1)
        BL = median(x((i-R):end));
    end
    y(i) = y(i)- BL;
end
