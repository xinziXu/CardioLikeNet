function [ecg_data,ANNOT,ATRTIME,RRtime]=read_data(filenum)
% -------------------------------------------------------------------------
%version1.0 created by Y. Zhao 07/21/2018
%------ SPECIFY DATA ------------------------------------------------------
%------ -------------------------------------------------------
PATH= 'F:\Design\Designs\CAC\ECG_data'; % path
%100 101 103 105 106 108 109 111-119 121-124 are records selected at random
%200-203 205 207-210 212-215 219-223 228 230-234 are records selected to
%include less common but clinically important arrhythmias
%102 104 107 217 are paced records that should be excluded
HEADERFILE=[num2str(filenum),'.hea.txt'];      % .hea header file
ATRFILE= [num2str(filenum),'.atr'];         % .atr anaotation file
DATAFILE=[num2str(filenum),'.dat'];         % .dat bianry file store ECG data
%SAMPLES2READ=2048;         % set the length of the proccesed data
                            % 102 and 104 shouldn't be set as the leads are Modified V
                            % No dataset2 for 101, 103, 105, 106, 
%annotations for peaks should oberserv first to extract the training data
%e.g. 106 is 1,5; 105 is 1,8,5; remeber 28 and 14 are not peaks annotation
%106 strange is stored seperately; 107 has no P wave stored
%seperately;108 use valley to pick the R wave
%
%------ LOAD HEADER DATA --------------------------------------------------
%------ content of header 117 -----------------------------------------------------
%      117 2 360 650000
%      117.dat 212 200 11 1024 839 31170 0 MLII
%      117.dat 212 200 11 1024 930 28083 0 V2
%      # 69 M 950 654 x2
%      # None
%-------------------------------------------------------------------------
signalh= fullfile(PATH, HEADERFILE);    % get the full path name of header file
fid1=fopen(signalh,'r');    % 
z= fgetl(fid1);             % get the first line
A= sscanf(z, '%*s %d %d %d',[1,3]); % read the first line but skip the first string by *
nosig= A(1);    % get the signal number
sfreq=A(2);     % sampling frequency
clear A;        % 
for k=1:nosig           % get sample format of each signal
    z= fgetl(fid1);     %continue the previous fgetl get the next line
    A= sscanf(z, '%*s %d %d %d %d %d',[1,5]);
    dformat(k)= A(1);           % digitalization format
    gain(k)= A(2);              % adc gain
    bitres(k)= A(3);            % adc resolution
    zerovalue(k)= A(4);         % ECG zero value
    firstvalue(k)= A(5);        % start value 
    clear A;
end
fclose(fid1);
%------ LOAD BINARY DATA --------------------------------------------------
if dformat~= [212,212], error('this script does not apply binary formats different to 212.'); end
signald= fullfile(PATH, DATAFILE);            % 
fid2=fopen(signald,'r');
A= fread(fid2, 'uint8')';  % one byte 8bits, 2 samples 3 bytes (212 mean) matrix with 3 rows, each 8 bits long, = 2*12bit
B=reshape(A,3,[]);
fclose(fid2);
%Each sample is represented by a 12-bit two's complement amplitude.
%The first sample is obtained from the 12 least significant bits 
%of the first byte pair (stored least significant byte first). 
%The second sample is formed from the 4 remaining bits of the first
%byte pair (which are the 4 high bits of the 12-bit sample) and the 
%next byte (which contains the remaining 8 bits of the second sample). 
%|--8LSB-1-|-4HSB-2-|-4HSB-1-|--8LSB-2-|
% 11111111    1111     1111    11111111
%|--Byte-1-|--Byte-2---------|--Byte-3-|
M2H= bitshift(B(2,:), -4);        %get the 4 high significant bits for the second
M1H= bitand(B(2,:), 15);          %get for the first
PRL=bitshift(bitand(B(2,:),8),9);     % sign-bit  8=1000-->1 0000 0000 0000 
PRR=bitshift(bitand(B(2,:),128),5);   % sign-bit  128=1000 0000-->1 0000 0000 0000
M( : , 1)= bitshift(M1H,8)+ B(1,:)-PRL; %
M( : , 2)= bitshift(M2H,8)+ B(3,:)-PRR; %
if M(1,:) ~= firstvalue 
    error('inconsistency in the first bit values'); 
end
switch nosig
case 2
    ecg_data=M;
    [SAMPLES2READ,colum_m]=size(M);
    TIME=0:(SAMPLES2READ-1);
case 1
    M=M';
    ecg_data=reshape(M,1,[]);
    SAMPLES2READ=length(ecg_data);
    TIME=0:(SAMPLES2READ-1);
otherwise  % this case did not appear up to now!
    % here M has to be sorted!!!
    disp('Sorting algorithm for more than 2 signals not programmed yet!');
end
clear A M1H M2H PRR PRL M;

%------ LOAD ATTRIBUTES DATA ----------------------------------------------
%MIT annotation format
%if the first byte dont equal to zero or the second byte equal to 0x5B or
%0x5D then it is MIT annotation format otherwise its AHA format. If it is MIT
%format then can read the annotation 2bytes by 2bytes(first byte is leat significant 
%second byte is most significant byte) and the format is
%given as for example 0x7012: 0111 00 00 0001 0010 the practical value is 
% high 6bits stands for the type, in this case 0111 00 equals to 28 means rythm
%change, and the left bits stands for the time interval from last beat. in
%this case the left bits value is 00 0001 0010 which equals 18, that means
%the beats time from last beat is 18/Fs=18/360=50ms. Since this is the
%first annotation, then it is the time that the first beat position.
%There are several special cases: 
%First, if the high 6bits are 63, then the
%following 10bits stand for the how many following bytes(it is always even
%if the number is odd, then one more byte should be counted) are used to store the
%extra annotation information instead of time, e.g. 0xFC03: 1111 11 00 0000
%0011, the high 6bits are 1111 11 which is 63, then the following number is
%3 means 3bytes are used but the number should be even so 4 following bytes 
%are used to store the extra information. And the next 16bits should starts
%after the 4bytes. 
%Second, if the first 6bits is 59, should read next three 16bits the high 
%6bits of the third 16bits is the type, and the first and second 16bits
%give the time.
%Third, if the first 6bits is 60, 61 or 62 skip.
atrd= fullfile(PATH, ATRFILE);      % attribute file with annotation data
fid3=fopen(atrd,'r');
A= fread(fid3, [2, inf], 'uint8')';
fclose(fid3);
ATRTIME=[];
ANNOT=[];
RRtime=[];
sa=size(A);
saa=sa(1);
i=1;
while i<=saa
    annoth=bitshift(A(i,2),-2); %get the high 6bits of the second byte
    if annoth==59
        ANNOT=[ANNOT;bitshift(A(i+3,2),-2)];
        ATRTIME=[ATRTIME;A(i+2,1)+bitshift(A(i+2,2),8)+...
                bitshift(A(i+1,1),16)+bitshift(A(i+1,2),24)];
        i=i+3;
    elseif annoth==60
        % nothing to do! skip this 16bits
    elseif annoth==61
        % nothing to do!
    elseif annoth==62
        % nothing to do!
    elseif annoth==63
        hilfe=bitshift(bitand(A(i,2),3),8)+A(i,1);
        hilfe=hilfe+mod(hilfe,2);
        i=i+hilfe/2;
    else
        ATRTIME=[ATRTIME;bitshift(bitand(A(i,2),3),8)+A(i,1)];
        ANNOT=[ANNOT;bitshift(A(i,2),-2)];
    end
   i=i+1;
end
ANNOT(length(ANNOT))=[];       % remove the last one 
ATRTIME(length(ATRTIME))=[];   % which is end of file flag
clear A;
%type list on Physino web
%N L R A a J S V F ! e j E P f p Q
%followings are heart beat types index/typecode/meaning
%index code meaning  AAMItype
% 1 	N	Normal beat                              N +
% 2 	L	Left bundle branch block beat            N +
% 3 	R	Right bundle branch block beat           N +
% 4 	a	Aberrated atrial premature beat          S +
% 5 	V	Premature ventricular contraction        V +
% 6 	F	Fusion of ventricular and normal beat    F +
% 7 	J	Nodal (junctional) premature beat        S +
% 8 	A	Atrial premature beat                    S +
% 9  	S	Premature or ectopic supraventricular beat  S +     
% 10	E	Ventricular escape beat                  V  +
% 11	j	Nodal (junctional) escape beat           S   +
% 12	/(P)	Paced beat                            Q
% 13	Q	Unclassifiable beat                   Q   +
% 14	~	Signal quality change                 Interrupting
% 16	|	Isolated QRS-like artifact            Interrupting
% 18	s	ST change                    Noninterrupting
% 19	T	T-wave change                Noninterrupting
% 20	*	Systole                      Noninterrupting
% 21	D	Diastole                     Noninterrupting
% 22	"	Comment annotation           Noninterrupting
% 23	=	Measurement annotation       Noninterrupting
% 24	p	P-wave peak                  Noninterrupting   +
% 25	B	Left or right bundle branch block            N
% 26	^	Non-conducted pacer spike    Noninterrupting
% 27	t	T-wave peak                  Noninterrupting
% 28	+	Rythm change                 Noninterrupting
% 29	u	U-wave peak                  Noninterrupting
% 30	?	Learning                     
% 31	!	Ventricular flutter wave                        +
% 32	[	Start of ventricular flutter/fibrillation        Interrupting
% 33	]	End of ventricular flutter/fibrillation          Interrupting
% 34	e	Atrial escape beat                               S  +
% 35	n	Supraventricular espace beat                     S
% 37	x	Non-conducted P-wave (blocked APB)                
% 38	f	Fusion of paced and normal beat                  Q  +
% 39	(	Waveform onset, PQ junction(begin of QRS)        Noninterrupting
% 40	)	Waveform end, JPT(J point, end of QRS)           Noninterrupting
% 41	r	R-on-T premature ventricular contraction         V
%summary N:1 2 3 25;  S: 4 7 8 9 11 34 35;  V: 5 10 41;   F: 6
%           AAMI practice
%   N L R A a J S V F ! e j E P f p Q  %types in the database
% 1 	N	Normal beat                                   N +
% 2 	L	Left bundle branch block beat                 N +
% 3 	R	Right bundle branch block beat                N +
% 31	!	Ventricular flutter wave (only tape207 has)     +
% 24	p	P-wave peak                                     +
% 8 	A	Atrial premature beat                         S +
% 4 	a	Aberrated atrial premature beat               S +
% 7 	J	Nodal (junctional) premature beat             S +
% 9  	S	Premature or ectopic supraventricular beat    S +
% 34	e	Atrial escape beat                            N +
% 11	j	Nodal (junctional) escape beat                N +

% 5 	V	Premature ventricular contraction             V +
% 10	E	Ventricular escape beat                       V +

% 6 	F	Fusion of ventricular and normal beat         F +

% 12	/(P)	Paced beat                                Q +
% 38	f	Fusion of paced and normal beat               Q +
% 13	Q	Unclassifiable beat                           Q +

ATRTIME= (cumsum(ATRTIME));
index_typen=find(ANNOT==1|ANNOT==2|ANNOT==3|ANNOT==11|ANNOT==34); %type N
index_types=find(ANNOT==4|ANNOT==7|ANNOT==8|ANNOT==9); %type S
index_typev=find(ANNOT==5|ANNOT==10); %type V
index_typef=find(ANNOT==6); %type F
index_typeq=find(ANNOT==12|ANNOT==13|ANNOT==38); %type Q
ANNOT(index_typen)=1;
ANNOT(index_types)=1;
ANNOT(index_typev)=1;
ANNOT(index_typef)=1;
ANNOT(index_typeq)=1;
index_nonbeat=find(ANNOT<1|ANNOT>5);
ANNOT(index_nonbeat)=[];
ATRTIME(index_nonbeat)=[];
RRtime=ATRTIME(2:end)-ATRTIME(1:end-1);
end