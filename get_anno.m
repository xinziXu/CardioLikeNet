addpath('./mcode')
database = 0; % 0 mitdb 1 qtdb 2 lued
if database == 0
    file_index = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234];
    original_samplerate = 360;
    present_samplerate = 250;
    % classN = ['N', 'L', 'R'];
    classN = ['N','L','R','e','j'];
    classS = ['A', 'a', 'J', 'S'];
    classV = ['V', 'E'];
    classF = ['F'];
    classQ = ['f', 'Q'];

    for i = 1:48
        anno = {};
        file_name = strcat('./database/mitdb/', num2str(file_index(i)));
        [ann, anntype] = rdann(file_name, 'atr', []);
        ann = double(int32(ann * present_samplerate / original_samplerate));
        cnt = 0;

        for j = 1:length(ann)

            if (ismember(anntype(j), classN))
                cnt = cnt + 1;
                anno{cnt, 1} = ann(j);
                anno{cnt, 2} = 'N';
            elseif (ismember(anntype(j), classS))
                cnt = cnt + 1;
                anno{cnt, 1} = ann(j);
                anno{cnt, 2} = 'S';
            elseif (ismember(anntype(j), classV))
                cnt = cnt + 1;
                anno{cnt, 1} = ann(j);
                anno{cnt, 2} = 'V';
            elseif (ismember(anntype(j), classF))
                cnt = cnt + 1;
                anno{cnt, 1} = ann(j);
                anno{cnt, 2} = 'F';
            elseif (ismember(anntype(j), classQ))
                cnt = cnt + 1;
                anno{cnt, 1} = ann(j);
                anno{cnt, 2} = 'Q';
            end

        end

        save_anno_path = './annotmitdb/';

        if ~isfolder(save_anno_path)
            mkdir(save_anno_path);
        end

        anno_file = strcat(save_anno_path, num2str(file_index(i)), '_anno.mat');
        save(anno_file, 'anno');
    end

elseif database == 1
    id = 1:210;

    for i = 1:length(id)
        filenum = id(i);
        file_path = './database/QTDataset/';
        [data, p, qrs, t] = read_qtdb(file_path, filenum);
        anno = zeros(length(data), 1);
        
        for num_p = 1:size(p, 1) % p 1
            anno(p(num_p, 1):p(num_p, 2)) = 1;
        end

        for num_qrs = 1:size(qrs, 1) % qrs 2
            anno(qrs(num_qrs, 1):qrs(num_qrs, 2)) = 2;
        end

        for num_t = 1:size(t, 1) % t 3
            anno(t(num_t, 1):t(num_t, 2)) = 3;
        end
        save_anno_path = './annotqtdb/';

        if ~isfolder(save_anno_path)
            mkdir(save_anno_path);
        end

        anno_file = strcat(save_anno_path, num2str(id(i)), '_anno.mat');
        save(anno_file, 'anno');
        segment_file = strcat(save_anno_path, num2str(id(i)), '_se.mat');      
        save (segment_file, 'p', 'qrs', 't');  

    end

 
    
elseif database == 2
    fileindex = 1:200; % 7, 34, 111, 116,have wrong labels
%     fileindex = 1;
    fileindex = fileindex(~ismember(fileindex,7));
    fileindex = fileindex(~ismember(fileindex,34));
    fileindex = fileindex(~ismember(fileindex,111));
    fileindex = fileindex(~ismember(fileindex,116));
%     fileindex = 1;
    leads = ['avf', 'avl', 'avr', 'i', 'ii', 'iii', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'];
    data_path = "./database/lued_mat/";
    original_samplerate = 500;
    present_samplerate = 250;
    for i = 1:length(fileindex)
        fileindex(i)
        file_name = strcat(data_path, num2str(fileindex(i)));
        data = load(file_name);
        signal = data.signal; symbol = data.symbol; symbol_index = data.symbol_index;
        sampletime = (1:length(signal)) / original_samplerate;
        resampletime = 1 / present_samplerate:1 / present_samplerate:(length(signal) / original_samplerate);
        signal = interp1(sampletime, double(signal), resampletime);
%         symbol_index = double(int32(symbol_index * present_samplerate / original_samplerate));
        if ~iscell (symbol_index)
            symbol_index = mat2cell(symbol_index, ones(1,12), size(symbol_index,2))';
            symbol = mat2cell(symbol, ones(1,12), size(symbol,2))';
        end

        [anno, p, qrs, t] = lued_annot_gen(signal, symbol, symbol_index);

        save_anno_path = './annotlued/';

        if ~isfolder(save_anno_path)
            mkdir(save_anno_path);
        end

        anno_file = strcat(save_anno_path, num2str(fileindex(i)), '_anno.mat');
        segment_file = strcat(save_anno_path, num2str(fileindex(i)), '_se.mat');
        save(anno_file, 'anno')        
        save (segment_file, 'p', 'qrs', 't')
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

function [anno, p, qrs, t] = lued_annot_gen(signal, symbol, symbol_index)
    anno = zeros (length(signal), 12);
    p = cell(12,1); % 12*  num_p* 3, p_on, p_off, p
    qrs = cell(12,1); 
    t = cell(12,1);
    present_samplerate = 250;
    original_samplerate =500;
    for lead = 1:12
        symbol_one_lead = symbol{lead};
        symbol_index_one_lead = symbol_index{lead};
        symbol_index_one_lead = double(int32(symbol_index_one_lead * present_samplerate / original_samplerate));

        for j = 1:length(symbol_one_lead)
            if symbol_one_lead(j) == 'p'

                [start_end, annot_temp] = update_anno(1,symbol_index_one_lead, symbol_one_lead, j, length(signal));
                p{lead} =[p{lead} ;start_end];
                anno(:,lead) = anno(:,lead)+ annot_temp;
            elseif symbol_one_lead(j) == 'N'

                [start_end, annot_temp] = update_anno(2,symbol_index_one_lead, symbol_one_lead, j, length(signal));
                qrs{lead} = [qrs{lead}; start_end];
                anno(:,lead) = anno(:,lead)+ annot_temp;
            elseif symbol_one_lead(j) == 't'

                [start_end, annot_temp] = update_anno(3,symbol_index_one_lead, symbol_one_lead, j, length(signal));
                t{lead} = [t{lead}; start_end];
                anno(:,lead) = anno(:,lead)+ annot_temp;
            end
        end
    end

end

function [start_end, anno] = update_anno(seg_label,symbol_index_one_lead, symbol_one_lead, j, signal_length)
    anno = zeros(signal_length,1);

    start_end = [];
    if j == 1
        if symbol_one_lead(j+1) == ')'
            
            start_end = [symbol_index_one_lead(j) symbol_index_one_lead(j+1) symbol_index_one_lead(j)];
        else
            fprintf('error!')   
        end                     
    elseif j == length(symbol_one_lead)
        if symbol_one_lead(j-1) == '('
            anno(symbol_index_one_lead(j-1):symbol_index_one_lead(j)) = seg_label;
            start_end = [symbol_index_one_lead(j-1) symbol_index_one_lead(j) symbol_index_one_lead(j)];
        else
            fprintf('error!')   
        end 
    else
        if (symbol_one_lead(j-1) == '(') && (symbol_one_lead(j+1) == ')')
            anno(symbol_index_one_lead(j-1):symbol_index_one_lead(j+1)) = seg_label; % p=1
            start_end = [symbol_index_one_lead(j-1) symbol_index_one_lead(j+1) symbol_index_one_lead(j)];
        elseif (symbol_one_lead(j-1) ~= '(') && (symbol_one_lead(j+1) == ')')
            anno(symbol_index_one_lead(j):symbol_index_one_lead(j+1)) = seg_label;
            start_end = [symbol_index_one_lead(j) symbol_index_one_lead(j+1) symbol_index_one_lead(j)];
        elseif (symbol_one_lead(j-1) == '(') && (symbol_one_lead(j+1) ~= ')')
            anno(symbol_index_one_lead(j-1):symbol_index_one_lead(j)) = seg_label;
            start_end = [symbol_index_one_lead(j-1) symbol_index_one_lead(j) symbol_index_one_lead(j)];
        
        else
            fprintf('error!') 
        end
        
    end
end
