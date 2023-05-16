database = 0; % 0 mitdb 1 qtdb 2 lued 

if database == 0
    DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230];
    DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234];
    original_samplerate = 360;
    present_samplerate = 250;
    %DS1
    for i = 1:22
        data_file = strcat('mitdb/', num2str(DS1(i)));
        data = rdsamp(data_file, [], [], [], [3], []);
        sampletime = (1:length(data)) / original_samplerate;
        resampletime = 1 / present_samplerate:1 / present_samplerate:(length(data) / original_samplerate);
        data = interp1(sampletime, double(data), resampletime);

        if DS1(i) == 114
            data = data(:, 2); % lead II of 114 is from the second channel
        else
            data = data(:, 1);
        end

        % 12bit
        ECG_12bit = quan_data(12, data);

        % figure;
        % plot(ECG_12bit);
        data_f = preprocessing_0phase(ECG_12bit,1,1);
        figure;
        % plot(data_f)
        save_path = './denoised_mitdbdata/';

        if ~isfolder(save_path)
            mkdir(save_path);
        end

        save(strcat(save_path, 'denoised_', num2str(DS1(i)), '_data.mat'), 'data_f');
    end

    % DS2
    for i = 1:22
        data_file = strcat('mitdb/', num2str(DS2(i)));
        data = rdsamp(data_file, [], [], [], [3], []);
        sampletime = (1:length(data)) / original_samplerate;
        resampletime = 1 / present_samplerate:1 / present_samplerate:(length(data) / original_samplerate);
        data = interp1(sampletime, double(data), resampletime);
        data = data(:, 1);

        % 12bit
        ECG_12bit = quan_data(12, data);

        %     figure;
        %     plot(ECG_12bit);
        data_f = preprocessing(ECG_12bit,1,1);
        %     figure;
        %     plot(data_f)
        save_path = './denoised_mitdbdata/';

        if ~isfolder(save_path)
            mkdir(save_path);
        end

        save(strcat(save_path, 'denoised_', num2str(DS2(i)), '_data.mat'), 'data_f');
    end

elseif database == 1

    file_path = './database/QTDataset/';
    id = 1:210;
%     id = 1;
    for i = 1:length(id)
        filenum = id(i);
        [data, p, qrs, t] = read_qtdb(file_path, filenum);

        ECG_12bit = quan_data(12, data);

        data_f = preprocessing_0phase(ECG_12bit,1,1);
        % figure;
        % plot(ECG_12bit)
        save_path = './denoised_qtdbdata/';

        if ~isfolder(save_path)
            mkdir(save_path);
        end

        save(strcat(save_path, 'denoised_', num2str(id(i)), '_data.mat'), 'data_f');
        % figure;
        % plot(data_f)

    end
    
elseif database == 2
    fileindex = 1:200; % 7, 34, 111, 116,have wrong labels

    fileindex = fileindex(~ismember(fileindex,7));
    fileindex = fileindex(~ismember(fileindex,34));
    fileindex = fileindex(~ismember(fileindex,111));
    fileindex = fileindex(~ismember(fileindex,116));  
    data_path = "./database/lued_mat/";
    leads = {'avf', 'avl', 'avr', 'i', 'ii', 'iii', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'};
    present_samplerate = 250;
    original_samplerate =500;
    
    for i = 1:length(fileindex)
        filenum = fileindex(i)
        file_name = strcat(data_path, num2str(fileindex(i)));
        data = load(file_name);
        signal = data.signal; 
        % plot(signal(:,1))
        sampletime = (1:length(signal)) / original_samplerate;
        resampletime = 1 / present_samplerate:1 / present_samplerate:(length(signal) / original_samplerate);
        signal = interp1(sampletime, double(signal), resampletime);        
        for lead= 1: size(signal,2)
            signal_one_lead = signal(:,lead);
            ECG_12bit = quan_data(12, signal_one_lead);

            data_f = preprocessing_0phase(ECG_12bit, 0,1);
%             figure;
%             plot(ECG_12bit)
            save_path = './denoised_lueddata/';

            if ~isfolder(save_path)
                mkdir(save_path);
            end

            save(strcat(save_path, 'denoised_', num2str(fileindex(i)),'_',leads{lead},'_data.mat'), 'data_f');
%             figure;
%             plot(data_f)
        end

    end
elseif database == 3
    x_train = load('X_train.mat');  x_train = x_train.X_train;
%     y_train = load('y_train.mat');  y_train = y_train.y_train;
    present_samplerate = 250;
    original_samplerate =500;
    leads = {"i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"};
    for i = 1: size(x_train, 1)
        for j = 1: size(x_train, 3)
    % for i = 2
    %     for j = 1: 12
            signal = x_train(i,:,j);
%             signal = signal(:,j);
            
            sampletime = (1:length(signal)) / original_samplerate;
            resampletime = 1 / present_samplerate:1 / present_samplerate:(length(signal) / original_samplerate);
            signal = interp1(sampletime, double(signal), resampletime); 
            ECG_12bit = quan_data(12, signal);
  
            data_f = preprocessing_0phase(ECG_12bit, 1,1);
            data_f = data_f(501:2500);
%             figure;
%             plot(ECG_12bit)
            save_path = './denoised_ptbxldata/';
            if ~isfolder(save_path)
                mkdir(save_path);
            end
% 
            save(strcat(save_path, 'denoised_', num2str(i),'_',leads{j},'_traindata.mat'), 'data_f');
            % figure;
            % plot(data_f)
            
        end   
    end


    x_test = load('X_test.mat');  x_test = x_test.X_test;
%     y_test = load('y_test.mat');  y_test = y_test.y_test;
    present_samplerate = 250;
    original_samplerate =500;
    leads = {"i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"};
    for i = 1: size(x_test, 1)
        for j = 1: size(x_test, 3)
    % for i = 2
    %     for j = 1: 12
            signal = x_test(i,:,j);
%             signal = signal(:,j);
            
            sampletime = (1:length(signal)) / original_samplerate;
            resampletime = 1 / present_samplerate:1 / present_samplerate:(length(signal) / original_samplerate);
            signal = interp1(sampletime, double(signal), resampletime); 
            ECG_12bit = quan_data(12, signal);
  
            data_f = preprocessing_0phase(ECG_12bit, 1,1);
            data_f = data_f(501:2500);
%             figure;
%             plot(ECG_12bit)
            save_path = './denoised_ptbxldata/';
            if ~isfolder(save_path)
                mkdir(save_path);
            end
% 
            save(strcat(save_path, 'denoised_', num2str(i),'_',leads{j},'_testdata.mat'), 'data_f');
            % figure;
            % plot(data_f)
            
        end   
    end
elseif database == 4
    
    data = load('train_ptb_loc.mat')
    x_train = data.trainX;

%     y_train = data.trainY;
    record_list = data.record_list;
    location_list = data.location_list;
    present_samplerate = 250;
    original_samplerate =1000;
    leads = {"i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"};
    for i = 1: size(x_train, 2)
    % for i = 5      
        signal = x_train{1,i};
        for j = 1: length(leads)
        % for j = 12
            signal_one_lead = signal(j, :);         
            sampletime = (1:length(signal)) / original_samplerate;
            resampletime = 1 / present_samplerate:1 / present_samplerate:(length(signal_one_lead) / original_samplerate);
            signal_one_lead = interp1(sampletime, double(signal_one_lead), resampletime); 
            ECG_12bit = quan_data(12, signal_one_lead); 
            data_f = preprocessing_0phase(ECG_12bit, 1,1);
            data_f = data_f(501:end);
            % figure;
            % plot(ECG_12bit)
            save_path = './denoised_ptbdata/';
            if ~isfolder(save_path)
                mkdir(save_path);
            end
% %         
            % strtrim(location_list{i})

            save(strcat(save_path, 'denoised_', record_list{1,i}(end-7:end), '_',strtrim(location_list{i}),'_',leads{j},'_traindata.mat'), 'data_f');
            % figure;
            % plot(data_f)
%             
        end   
    end 

    data = load('test_ptb_loc.mat');
    x_test = data.testX;

%     y_test = data.testY;
    record_list = data.record_list;
    location_list = data.location_list;

    present_samplerate = 250;
    original_samplerate =1000;
    leads = {"i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"};
    for i = 1: size(x_test, 2)
    % for i = 5      
        signal = x_test{1,i};
        for j = 1: length(leads)
        % for j = 12
            signal_one_lead = signal(j, :);         
            sampletime = (1:length(signal)) / original_samplerate;
            resampletime = 1 / present_samplerate:1 / present_samplerate:(length(signal_one_lead) / original_samplerate);
            signal_one_lead = interp1(sampletime, double(signal_one_lead), resampletime); 
            ECG_12bit = quan_data(12, signal_one_lead); 
            data_f = preprocessing_0phase(ECG_12bit, 1,1);
            data_f = data_f(501:end);
            % figure;
            % plot(ECG_12bit)
            save_path = './denoised_ptbdata/';
            if ~isfolder(save_path)
                mkdir(save_path);
            end
% % 
            
            save(strcat(save_path, 'denoised_',record_list(i, end - 7:end), '_',strtrim(location_list(i,:)),'_',leads{j},'_testdata.mat'), 'data_f');
            % figure;
            % plot(data_f)
%             
        end   
    end     
elseif database == 5
    DS1 = [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230];
    DS2 = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234];
    original_samplerate = 360;
    present_samplerate = 250;
    %DS1
    for i = 1:length(DS1)
        data_file = strcat('./database/mitdb/', num2str(DS1(i)));
        data = rdsamp(data_file, [], [], [], [3], []);
        sampletime = (1:length(data)) / original_samplerate;
        resampletime = 1 / present_samplerate:1 / present_samplerate:(length(data) / original_samplerate);
        data = interp1(sampletime, double(data), resampletime);

        for j = 1:size(data, 2)

            % 12bit
            ECG_12bit = quan_data(12, data(:, j));

            % figure;
            % plot(ECG_12bit);
            data_f = preprocessing_0phase(ECG_12bit, 1, 1);
            % figure;
            % plot(data_f)
            save_path = './denoised_mitdbdata/';
            
            if ~isfolder(save_path)
                mkdir(save_path);
            end

            save(strcat(save_path, 'denoised_', num2str(DS1(i)), '_', num2str(j), '_data.mat'), 'data_f');
        end

    end

    % DS2
    for i = 1:length(DS2)
        data_file = strcat('./database/mitdb/', num2str(DS2(i)));
        data = rdsamp(data_file, [], [], [], [3], []);
        sampletime = (1:length(data)) / original_samplerate;
        resampletime = 1 / present_samplerate:1 / present_samplerate:(length(data) / original_samplerate);
        data = interp1(sampletime, double(data), resampletime);
        

        for j = 1:size(data, 2)

            % 12bit
            ECG_12bit = quan_data(12, data(:, j));

            % figure;
            % plot(ECG_12bit);
            data_f = preprocessing_0phase(ECG_12bit, 1, 1);
            % figure;
            % plot(data_f)
            save_path = './denoised_mitdbdata/';

            if ~isfolder(save_path)
                mkdir(save_path);
            end

            save(strcat(save_path, 'denoised_', num2str(DS2(i)), '_', num2str(j), '_data.mat'), 'data_f');
        end
    end 
end

function ecg_f = preprocessing(ecg, lf_enable, hf_enable)

    %low-pass filter
    COFF_LF = load('LF.mat'); %chebyshefII,pass:35,stop:80,a��50db
    SOS_LF = COFF_LF.SOS;
    G_LF = COFF_LF.G;

    %quantizer
    qpath = quantizer('fixed', 'round', 'saturate', [10, 8]);
    SOS_LF_FIX = quantize(qpath, SOS_LF);
    SOS_LF_FIX = SOS_LF_FIX * (2^8); % gain=26db=20times
    if (lf_enable)
        ecg_lf1 = filter_2o(SOS_LF_FIX(1, 1:3), SOS_LF_FIX(1, 4:6), ecg * (2^8));
        ecg_lf = filter_2o(SOS_LF_FIX(2, 1:3), SOS_LF_FIX(2, 4:6), ecg_lf1);
        % ecg_lf = filter_2o(SOS_LF_FIX(3,1:3),SOS_LF_FIX(3,4:6), ecg_lf2);
        % ecg_lf = filter_2o(SOS_LF_FIX(4,1:3),SOS_LF_FIX(4,4:6), ecg_lf3);
        ecg_lf = ecg_lf / (2^8) / 32;
    else
        ecg_lf = ecg;
    end
    %

    % high-pass filter
    COFF_HF = load('HF.mat'); %chebyshefII,pass:0.1,stop:0.2,a��50db
    SOS_HF = COFF_HF.SOS;
    % fvtool(SOS_HF)
    G_HF = COFF_HF.G;

    %quantizer
    qpath = quantizer('fixed', 'round', 'saturate', [22, 20]);
    SOS_HF_FIX = quantize(qpath, SOS_HF);
    SOS_HF_FIX = SOS_HF_FIX * 2^20;

    if (hf_enable)
        ecg_hf1 = filter_2o(SOS_HF_FIX(1, 1:3), SOS_HF_FIX(1, 4:6), ecg_lf * 2^20);
        ecg_hf2 = filter_2o(SOS_HF_FIX(2, 1:3), SOS_HF_FIX(2, 4:6), ecg_hf1);
        ecg_hf = filter_2o(SOS_HF_FIX(3, 1:3), SOS_HF_FIX(3, 4:6), ecg_hf2);
        ecg_hf = ecg_hf / (2^20);
    else
%         ecg_hf = filtfilt(SOS_HF, G_HF,ecg_lf);
%         ecg_hf = ecg_hf;
        ecg_hf = ecg_lf;
    end

    % smothing filter

    ecg_f = conv(ecg_hf, ones(1, 8),'same');
    ecg_f = floor(ecg_f / 8);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

function signal_f = filter_2o(b, a, signal)
    signal_f = signal;

    for i = 3:length (signal)
        signal_f(i) = floor((b(1) * signal(i) + b(2) * signal(i - 1) +b(3) * signal(i - 2) - a(2) * signal_f(i - 1) - a(3) * signal_f(i - 2)) / a(1));
    end

end

function [ecgSignal, p, qrs, t] = read_qtdb(signal_path, id)
    %     id = 1;
    %     signal_path = 'E:\xxz\QTDataset';
    p = [];
    qrs = [];
    t = [];
    file = [signal_path '/ecg' num2str(id) '.mat'];
    ecg_data = load(file);

    Fs = ecg_data.Fs;
    ecgSignal = ecg_data.ecgSignal;
    signalRegionLabels = ecg_data.signalRegionLabels;

    for i = 1:size(signalRegionLabels, 1)
        Value = char(signalRegionLabels(i, 2).Value);
        ROILimits = signalRegionLabels(i, 1).ROILimits;

        if strcmp(Value, 'P')
            p = [p; ROILimits];
        elseif strcmp(Value, 'QRS')
            qrs = [qrs; ROILimits];
        elseif strcmp(Value, 'T')
            t = [t; ROILimits];
        end

    end

end

function data_quan = quan_data (bit, data)
    ECG_max = max(data);
    ECG_min = min(data);
    ECG_minmax = (data - ECG_min) / (ECG_max - ECG_min);
    data_quan = round((ECG_minmax - 0.5) * 2^bit * 0.9) + 2^(bit - 1);

end


function ecg_f = preprocessing_0phase(ecg, lf_enable, hf_enable)
    
    
    %high-pass filter
    
    fc = 0.6;
    si = 4; % sampling interval (ms)
    delta = 0.6;
    g = tan(pi*fc*si*0.001);
    
    g1 = 1 + 2* delta*g +  g^2;
    g2 = 2 * g^2 -2;
    g3 = 1 - 2 * delta * g+ g^2;
    
    b1 = g^2 / g1;
    b3 = b1;
    b2 = 2 * b1;
    a2 = g2 / g1;
    a3 = g3 / g1;
    
    b = [b1 b2 b3];
    a = [1 a2 a3];
    
    dec_bit = 16;
    
    qpath = quantizer('fixed', 'round', 'saturate', [dec_bit+ 2, dec_bit]);
    b_fixed = quantize(qpath, b)* 2^dec_bit;
    a_fixed = quantize(qpath, a)* 2^dec_bit;
    
    ecg_e = ecg* 2^dec_bit;
    
    ecg_bl = ecg_e;
    ecg_bl(1) =  floor((b_fixed(1) + b_fixed(2) + b_fixed(3) - a_fixed(2) - a_fixed(3)) * ecg_e(1)/a_fixed(1));
    ecg_bl(2) = floor((b_fixed(1) * ecg_e(2) + ( b_fixed(2) +  b_fixed(3) - a_fixed(3)) * ecg_e(1) - a_fixed(2)*ecg_bl(1))/a_fixed(1));
    
    
    for i = 3: length(ecg_e)
        ecg_bl(i) = floor((b_fixed(1) * ecg_e(i) +  b_fixed(2) * ecg_e(i-1)...
        +  b_fixed(3) * ecg_e(i-2) - a_fixed(2) * ecg_bl(i-1) -a_fixed(3) * ecg_bl(i-2))/a_fixed(1));
    end
    
%     ecg_bl = filter(b_fixed, a_fixed, ecg);
    ecg_hf = ecg_e - ecg_bl;
    
    ecg_hf = ecg_hf/2^dec_bit;
    
    if hf_enable
        ecg_hf = ecg_hf;
    else
        ecg_hf = ecg;
    end
        
    
    % low-pass filter
%     COFF_LF = load('LF.mat'); %chebyshefII,pass:35,stop:80,a��50db
%     SOS_LF = COFF_LF.SOS;
% %     fvtool(SOS_LF)
%     G_LF = COFF_LF.G;
% 
%     %quantizer
%     qpath = quantizer('fixed', 'round', 'saturate', [10, 8]);
%     SOS_LF_FIX = quantize(qpath, SOS_LF);
%     SOS_LF_FIX = SOS_LF_FIX * (2^8); % gain=26db=20times
%     if (lf_enable)
%         ecg_lf1 = filter_2o(SOS_LF_FIX(1, 1:3), SOS_LF_FIX(1, 4:6), ecg_hf * (2^8));
%         ecg_lf = filter_2o(SOS_LF_FIX(2, 1:3), SOS_LF_FIX(2, 4:6), ecg_lf1);
%         % ecg_lf = filter_2o(SOS_LF_FIX(3,1:3),SOS_LF_FIX(3,4:6), ecg_lf2);
%         % ecg_lf = filter_2o(SOS_LF_FIX(4,1:3),SOS_LF_FIX(4,4:6), ecg_lf3);
%         ecg_lf = ecg_lf / (2^8) / 32;
%     else
%         ecg_lf = ecg_hf;
%     end
    
    fc = 30;
    si = 4;
    delta = 0.6;
    g = tan(pi*fc*si*0.001);
    
    g1 = 1 + 2* delta*g +  g^2;
    g2 = 2 * g^2 -2;
    g3 = 1 - 2 * delta * g+ g^2;
    
    b1 = g^2 / g1;
    b3 = b1;
    b2 = 2 * b1;
    a2 = g2 / g1;
    a3 = g3 / g1;
    
    b = [b1 b2 b3];
    a = [1 a2 a3];
    

    
    dec_bit = 16;
    qpath = quantizer('fixed', 'round', 'saturate', [dec_bit+ 2, dec_bit]);
    b_fixed = quantize(qpath, b)* 2^dec_bit;
    a_fixed = quantize(qpath, a)* 2^dec_bit;
%     fvtool(b_fixed,a_fixed)
    ecg_e = ecg_hf* 2^dec_bit;
    
    ecg_bl = ecg_e;
    ecg_bl(1) =  ((b_fixed(1) + b_fixed(2) + b_fixed(3) - a_fixed(2) - a_fixed(3)) * ecg_e(1)/a_fixed(1));
    ecg_bl(2) = ((b_fixed(1) * ecg_e(2) + ( b_fixed(2) +  b_fixed(3) - a_fixed(3)) * ecg_e(1) - a_fixed(2)*ecg_bl(1))/a_fixed(1));
    
    
    for i = 3: length(ecg_e)
        ecg_bl(i) = ((b_fixed(1) * ecg_e(i) +  b_fixed(2) * ecg_e(i-1)...
        +  b_fixed(3) * ecg_e(i-2) - a_fixed(2) * ecg_bl(i-1) -a_fixed(3) * ecg_bl(i-2))/a_fixed(1));
    end
    
    
    ecg_lf = ecg_bl/2^dec_bit;
    
    if lf_enable
        ecg_lf = ecg_lf;
    else
        ecg_lf = ecg_hf;
    end
%             
    % smothing filter

    ecg_f = conv(ecg_lf, ones(1, 8),'same');
    ecg_f = floor(ecg_f / 8);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
end

