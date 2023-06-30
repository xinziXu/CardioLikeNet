function [annot]=set_label(ANNOT)
annot = 1;
% if (ANNOT~=1)
%     annot = 0;
if (ANNOT==1)
    annot = 1;
elseif (ANNOT==2)
    annot = 2;
elseif (ANNOT==3)
    annot = 3;
elseif (ANNOT==4)
    annot = 4;
elseif (ANNOT==5)
    annot = 5;
else
    annot = 6;
end
end
