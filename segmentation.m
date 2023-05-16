database = 0;
fs = 250;

if database == 0
    DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230];
    DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234];
    class = ['N', 'S', 'V', 'F', 'Q'];
    fs = 250;

    % DS1
    for i = 1:22
        lines = [];
        pre = 100;
        post = 150;
        anno_file = strcat('./annot/', num2str(DS1(i)), '_anno.mat');
        data_file = strcat('denoised_mitdbdata/', 'denoised_', num2str(DS1(i)), '_data.mat');
        anno = load(anno_file); anno = anno.anno;
        data = load(data_file); data = data.data;
        r_peak_loc = rpeak_detection (data, fs);

        for j = 2:size(anno, 1) - 1
            label = find(class == anno{j, 2});
            %anondan Q
            %         if label == 5
            %            continue;
            %         end

            % too long RR
            pre_RR = anno{j, 1} - anno{j - 1, 1};
            post_RR = anno{j + 1, 1} - anno{j, 1};

            if pre_RR > 500 || post_RR > 500
                continue;
            end

            peak = anno{j, 1};
            seg = data(peak - pre:peak + post)';
            line = [label pre_RR post_RR seg];
            lines = [lines; line];
        end

        save_path = './seg_file/';

        if ~isfolder(save_path)
            mkdir(save_path);
        end

        seg_file = strcat(save_path, num2str(DS1(i)), '_seg.mat');

        save(seg_file, 'lines');
    end

    %DS2
    for i = 1:22
        lines = [];
        seg_file = strcat('./seg_file/', num2str(DS2(i)), '_seg.mat');
        anno_file = strcat('./annot/', num2str(DS2(i)), '_anno.mat');
        data_file = strcat('denoised_data/', 'denoised_', num2str(DS2(i)), '_data.mat');
        anno = load(anno_file); anno = anno.anno;
        data = load(data_file); data = data.data;

        for j = 2:size(anno, 1) - 1
            label = find(class == anno{j, 2});
            %anondan Q
            %         if label == 5
            %            continue;
            %         end

            % too long RR
            pre_RR = anno{j, 1} - anno{j - 1, 1};
            post_RR = anno{j + 1, 1} - anno{j, 1};

            if pre_RR > 500 || post_RR > 500
                continue;
            end

            peak = anno{j, 1};
            seg = data(peak - pre:peak + post)';
            line = [label pre_RR post_RR seg];
            lines = [lines; line];
        end

        save(seg_file, 'lines');
    end

elseif database == 1
    id = 1:210;
    % id = 194;
    qrs_metric = [];

    for i = 1:length(id)
        segs = [];
        label_segs = [];

        anno_file = strcat('./annotqtdb/', num2str(id(i)), '_anno.mat');
        anno_se_file = strcat('./annotqtdb/', num2str(id(i)), '_se.mat');
        data_file = strcat('./denoised_qtdbdata/', 'denoised_', num2str(id(i)), '_data.mat');
        anno = load(anno_file); anno = anno.anno;
        data = load(data_file); data = data.data_f;
        se = load(anno_se_file); p = se.p; qrs = se.qrs; t = se.t;

        % remove unstable filtered data
        filter_length = 300;
        data = data(filter_length:end);
        anno = anno(filter_length:end);

        for p_num = 1:size(p, 1)

            if (p(p_num, 1) < filter_length) & (filter_length < p(p_num, 2))
                p = p(p_num:end, :);
                p (p_num, 1) = filter_length;
                p = p - filter_length + 1;
                break;
            elseif filter_length >= p(p_num, 2)
                continue;
            else
                p = p(p_num:end, :);
                p = p - filter_length + 1;
                break;
            end

        end

        for qrs_num = 1:size(qrs, 1)

            if (qrs(qrs_num, 1) < filter_length) & (filter_length < qrs(qrs_num, 2))
                qrs = qrs(qrs_num:end, :);
                qrs (qrs_num, 1) = filter_length;
                qrs = qrs - filter_length + 1;
                break;
            elseif filter_length >= qrs(qrs_num, 2)
                continue;
            else
                qrs = qrs(qrs_num:end, :);
                qrs = qrs - filter_length + 1;
                break;
            end

        end

        for t_num = 1:size(t, 1)

            if (t(t_num, 1) < filter_length) & (filter_length < t(t_num, 2))
                t = t(t_num:end, :);
                t (t_num, 1) = filter_length;
                t = t - filter_length + 1;
                break;
            elseif filter_length >= t(t_num, 2)
                continue;
            else
                t = t(t_num:end, :);
                t = t - filter_length + 1;
                break;
            end

        end

        r_peak_loc = rpeak_detection (data, fs);
        [Se, P, DER] = pt_metrics(qrs, r_peak_loc);
        qrs_metric = [qrs_metric; Se P DER];
        performance = mean(qrs_metric, 1);

        segs = [];
        labels = [];

        pre = 105;
        post = 150;

        for j = 2:length(r_peak_loc) -1
            rr_pre = r_peak_loc(j) - r_peak_loc(j - 1);
            rr_post = r_peak_loc(j + 1) - r_peak_loc(j);

            if rr_pre > 500 || rr_post > 500
                continue;
            end

            if (r_peak_loc(j) + post < length(data)) && (r_peak_loc(j) - pre > 0)
                seg = data(r_peak_loc(j) - pre:r_peak_loc(j) + post);
                label = anno(r_peak_loc(j) - pre:r_peak_loc(j) + post);
                segs = [segs; seg'];
                % size(segs)
                labels = [labels label];
                
                % size(labels)
 

            else
                continue;
            end

        end

        if isempty(segs) || isempty(labels)
            i
        end
        save_path = './segqtdb_250/';

        if ~isfolder(save_path)
            mkdir(save_path);
        end

        seg_file = strcat(save_path, num2str(id(i)), '_seg.mat');
        label_seg_file = strcat(save_path, num2str(id(i)), '_labelseg.mat');
        label_segs  = labels';
        % size(label_segs)
        % size(segs)
        save(seg_file, 'segs');
        save(label_seg_file, 'label_segs');
   
    end

elseif database == 2
    fileindex = 1:200; % 7, 34, 111, 116,have wrong labels
    % fileindex = 1;
    fileindex = fileindex(~ismember(fileindex, 7));
    fileindex = fileindex(~ismember(fileindex, 34));
    fileindex = fileindex(~ismember(fileindex, 111));
    fileindex = fileindex(~ismember(fileindex, 116));
    data_path = "./database/lued_mat/";
    leads = {'avf', 'avl', 'avr', 'i', 'ii', 'iii', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'};

    for i = 1:length(fileindex)
        filenum = fileindex(i)

        anno_file = strcat('./annotlued/', num2str(fileindex(i)), '_anno.mat');
        anno = load(anno_file); anno = anno.anno;

        anno_se_file = strcat('./annotlued/', num2str(fileindex(i)), '_se.mat');
        se = load(anno_se_file); p = se.p{5}; qrs = se.qrs{5}; t = se.t{5}; %lead ii
        r_peak_loc = qrs(:, 3);

        data_12lead = [];

        for lead = 1:12
            data_file = strcat('./denoised_lueddata/', 'denoised_', num2str(fileindex(i)), '_', leads{lead}, '_data.mat');
            data = load(data_file); data = data.data_f;
            data_12lead = [data_12lead data];

        end

        segs = [];
        label_segs = [];
        pre = 105;
        post = 150;

        if length(r_peak_loc) > 2

            for j = 2:(length(r_peak_loc) - 1)
                rr_pre = r_peak_loc(j) - r_peak_loc(j - 1);
                rr_post = r_peak_loc(j + 1) - r_peak_loc(j);

                if rr_pre > 500 || rr_post > 500
                    continue;
                end

                if (r_peak_loc(j) + post < size(data_12lead, 1)) && (r_peak_loc(j) - pre > 0)
                    seg = data_12lead(r_peak_loc(j) - pre:r_peak_loc(j) + post, :);
                    label = anno(r_peak_loc(j) - pre:r_peak_loc(j) + post,:);
                    segs = cat(3, segs, seg');
                    label_segs = cat(3, label_segs, label');
                    
        
    
                else
                    continue;
                end

                save_path = './seglued_250/';

                if ~isfolder(save_path)
                    mkdir(save_path);
                end

                seg_file = strcat(save_path, num2str(fileindex(i)), '_seg.mat');
                label_seg_file = strcat(save_path, num2str(fileindex(i)), '_labelseg.mat');

                size(segs)
                size(label_segs)

                save(seg_file, 'segs');
                save(label_seg_file, 'label_segs');

            end

        else
            continue;
        end

    end

elseif database == 3
    x_train = load('X_train.mat'); x_train = x_train.X_train;
    y_train = load('y_train.mat'); y_train = y_train.y_train;
    leads = {'avf', 'avl', 'avr', 'i', 'ii', 'iii', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'};
    fileindex = 1:size(x_train, 1);
    %
    for i = 1:size(x_train, 1)
        % for i = 197

        anno = y_train{i};

        if size(anno, 1) == 0
            continue
            fileindex = fileindex(~ismember(fileindex, i));
        elseif size(anno, 2) == 1

            if strcmp(anno, 'CD') || strcmp(anno, 'HYP')|| strcmp(anno, 'STTC')
                fileindex = fileindex(~ismember(fileindex, i));
                continue
            end

        end

        data_12lead = [];

        for lead = 1:12
            data_file = strcat('./denoised_ptbxldata/', 'denoised_', num2str(i), '_', leads{lead}, '_traindata.mat');
            data = load(data_file); data = data.data_f;
            data_12lead = [data_12lead data'];

        end

        r_peak_loc = rpeak_detection (data_12lead(:, 5), fs);
        segs = [];
        labels = [];
        pre = 105;
        post = 150;

        if length(r_peak_loc) > 2

            for j = 2:(length(r_peak_loc) - 1)
                rr_pre = r_peak_loc(j) - r_peak_loc(j - 1);
                rr_post = r_peak_loc(j + 1) - r_peak_loc(j);

                if rr_pre > 500 || rr_post > 500
                    continue;
                end


                if (r_peak_loc(j) + post < size(data_12lead, 1)) && (r_peak_loc(j) - pre > 0)
                    seg = data_12lead(r_peak_loc(j) - pre:r_peak_loc(j) + post, :);
                    label = anno2num(anno);
                    segs = cat(3, segs, seg');
                    labels =[labels label];
    
                else
                    continue;
                end
            end

        end

        if isempty(segs) || isempty(labels)
            continue
        end

        save_path = './segptbxldb_250/';

        if ~isfolder(save_path)
            mkdir(save_path);
        end

        if size(segs, 1) == 0 || size(labels, 1) == 0
            continue
        end

        seg_file = strcat(save_path, num2str(i), '_trainseg.mat');
        label_seg_file = strcat(save_path, num2str(i), '_trainlabelseg.mat');

        save(seg_file, 'segs');
        save(label_seg_file, 'labels');
        size(segs)
        size(labels)

    end

    x_test = load('X_test.mat'); x_test = x_test.X_test;
    y_test = load('y_test.mat'); y_test = y_test.y_test;
    leads = {'avf', 'avl', 'avr', 'i', 'ii', 'iii', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6'};
    fileindex = 1:size(x_test, 1);
    %
    for i = 1:size(x_test, 1)
        % for i = 197

        anno = y_test{i};

        if size(anno, 1) == 0
            continue
            fileindex = fileindex(~ismember(fileindex, i));
        elseif size(anno, 2) == 1

            if strcmp(anno, 'CD') || strcmp(anno, 'HYP')|| strcmp(anno, 'STTC')
                fileindex = fileindex(~ismember(fileindex, i));
                continue
            end

        end

        data_12lead = [];

        for lead = 1:12
            data_file = strcat('./denoised_ptbxldata/', 'denoised_', num2str(i), '_', leads{lead}, '_testdata.mat');
            data = load(data_file); data = data.data_f;
            data_12lead = [data_12lead data'];

        end

        r_peak_loc = rpeak_detection (data_12lead(:, 5), fs);
        segs = [];
        labels = [];
        pre = 105;
        post = 150;

        if length(r_peak_loc) > 2

            for j = 2:(length(r_peak_loc) - 1)
                rr_pre = r_peak_loc(j) - r_peak_loc(j - 1);
                rr_post = r_peak_loc(j + 1) - r_peak_loc(j);

                if rr_pre > 500 || rr_post > 500
                    continue;
                end

                if (r_peak_loc(j) + post < size(data_12lead, 1)) && (r_peak_loc(j) - pre > 0)
                    seg = data_12lead(r_peak_loc(j) - pre:r_peak_loc(j) + post, :);
                    label = anno2num(anno);
                    segs = cat(3, segs, seg');
                    labels =[labels label];
    
                else
                    continue;
                end
            end

            % else
            %     fprintf(num2str(i),  'error!')
        end

        if isempty(segs) || isempty(labels)
            continue
        end


        save_path = './segptbxldb_250/';

        if ~isfolder(save_path)
            mkdir(save_path);
        end

        seg_file = strcat(save_path, num2str(i), '_testseg.mat');
        label_seg_file = strcat(save_path, num2str(i), '_testlabelseg.mat');

        save(seg_file, 'segs');
        save(label_seg_file, 'labels');
        size(segs)
        size(labels)

    end

elseif database == 4

    data = load('train_ptb_loc.mat');
    % x_train = data.trainX;
    y_train = data.trainY;
    record_list = data.record_list;
    location_list = data.location_list;
    leads = {"avr", "avl", "avf", "i", "ii", "iii", "v1", "v2", "v3", "v4", "v5", "v6"};

    for i = 1:size(y_train, 2)
        % for i = 345
        if cell2mat(y_train(i)) == 0
            anno = 0;
        else
            anno = 1;
        end

        % anno = y_train(i);
        data_12lead = [];

        for lead = 1:12
            data_file = strcat('./denoised_ptbdata/', 'denoised_', record_list{1, i}(end - 7:end), '_',strtrim(location_list{i}),'_', leads{lead}, '_traindata.mat');
            data = load(data_file); data = data.data_f;
            data_12lead = [data_12lead data'];

        end

        r_peak_loc = rpeak_detection (data_12lead(:, 5), fs);
        segs = [];
        labels = [];
        pre = 105;
        post = 150;
        

        if length(r_peak_loc) > 2

            for j = 2:(length(r_peak_loc) - 1)
                rr_pre = r_peak_loc(j) - r_peak_loc(j - 1);
                rr_post = r_peak_loc(j + 1) - r_peak_loc(j);

                if rr_pre > 500 || rr_post > 500

                    continue;
                end
                if (r_peak_loc(j) + post < size(data_12lead, 1)) && (r_peak_loc(j) - pre > 0)
                    seg = data_12lead(r_peak_loc(j) - pre:r_peak_loc(j) + post, :);
                    label = anno;
                    segs = cat(3, segs, seg');
                    labels =[labels label];
                end

            end

        else
            continue
        end

        if isempty(segs) || isempty(labels)
            continue
        end
        save_path = './segptbdb_250/';

        if ~isfolder(save_path)
            mkdir(save_path);
        end

        seg_file = strcat(save_path, record_list{1, i}(end - 7:end), '_',strtrim(location_list{i}),'_trainseg.mat');
        label_seg_file = strcat(save_path, record_list{1, i}(end - 7:end), '_',strtrim(location_list{i}), '_trainlabelseg.mat');

        save(seg_file, 'segs');
        save(label_seg_file, 'labels');
        size(segs)
        size(labels)

    end

    data = load('test_ptb_loc.mat');
    % x_test = data.testX;
    y_test = data.testY;
    record_list = data.record_list;
    location_list = data.location_list;
    leads = {"avr", "avl", "avf", "i", "ii", "iii", "v1", "v2", "v3", "v4", "v5", "v6"};

    for i = 1:size(y_test, 2)
        % for i = 28
        if y_test(i) == 0
            anno = 0;
        else
            anno = 1;
        end

        data_12lead = [];

        for lead = 1:12
            data_file = strcat('./denoised_ptbdata/', 'denoised_', record_list(i, end - 7:end),'_',strtrim(location_list(i,:)),'_', leads{lead}, '_testdata.mat');
            data = load(data_file); data = data.data_f;
            data_12lead = [data_12lead data'];

        end

        r_peak_loc = rpeak_detection (data_12lead(:, 5), fs);
        segs = [];
        labels = [];

        if length(r_peak_loc) > 2

            for j = 2:(length(r_peak_loc) - 1)
                rr_pre = r_peak_loc(j) - r_peak_loc(j - 1);
                rr_post = r_peak_loc(j + 1) - r_peak_loc(j);

                if rr_pre > 500 || rr_post > 500
                    continue;
                end

                if (r_peak_loc(j) + post < size(data_12lead, 1)) && (r_peak_loc(j) - pre > 0)
                    seg = data_12lead(r_peak_loc(j) - pre:r_peak_loc(j) + post, :);
                    label = anno;
                    segs = cat(3, segs, seg');
                    labels =[labels label];
                end

            end

        else
            continue
        end

        if isempty(segs) || isempty(labels)
            continue
        end

        save_path = './segptbdb_250/';

        if ~isfolder(save_path)
            mkdir(save_path);
        end

        seg_file = strcat(save_path,  record_list(i, end - 7:end),'_',strtrim(location_list(i,:)), '_testseg.mat');
        label_seg_file = strcat(save_path,  record_list(i, end - 7:end),'_',strtrim(location_list(i,:)), '_testlabelseg.mat');

        save(seg_file, 'segs');
        save(label_seg_file, 'labels');
        size(segs)
        size(labels)

    end

elseif database == 5
    DS1 = [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230];
    DS2 = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234];
    class = ['N', 'S', 'V', 'F', 'Q'];
    fs = 250;

    % DS1
    for i = 1:length(DS1)

        anno_file = strcat('./annotmitdb/', num2str(DS1(i)), '_anno.mat');
        anno = load(anno_file); anno = anno.anno;

        data_2lead = [];

        for lead = 1:2
            data_file = strcat('./denoised_mitdbdata/', 'denoised_', num2str(DS1(i)), '_', num2str(lead), '_data.mat');
            data = load(data_file); data = data.data_f;
            data_2lead = [data_2lead data];

        end

        %     r_peak_loc = rpeak_detection (data_2lead(:,1), fs);
        r_peak_loc = [anno{:, 1}];
        segs = [];
        labels = [];
        fea_plus = [];
        pre = 105;
        post = 150;

        for j = 2:(length(r_peak_loc) - 1)
            rr_pre = r_peak_loc(j) - r_peak_loc(j - 1);
            rr_post = r_peak_loc(j + 1) - r_peak_loc(j);

            if rr_pre > 500 || rr_post > 500
                continue;
            end

            if (r_peak_loc(j) + post < size(data_2lead, 1)) && (r_peak_loc(j) - pre > 0)
                seg = data_2lead(r_peak_loc(j) - pre:r_peak_loc(j) + post, :);
                label = find(class == anno{j, 2}) - 1;
                segs = cat(3, segs, seg');
                labels = [labels label];
                fea = [rr_post rr_pre];
                fea_plus = [fea_plus; fea];

            else
                continue;
            end

        end

        save_path = './segmitdb_250/';

        if ~isfolder(save_path)
            mkdir(save_path);
        end

        seg_file = strcat(save_path, num2str(DS1(i)), '_trainseg.mat');
        label_seg_file = strcat(save_path, num2str(DS1(i)), '_trainlabelseg.mat');
        fea_seg_file = strcat(save_path, num2str(DS1(i)), '_trainfeaseg.mat');
        size(fea_plus)
        save(seg_file, 'segs');
        save(label_seg_file, 'labels');
        save(fea_seg_file, 'fea_plus');
    end

    % %DS2
    for i = 1:length(DS2)

        anno_file = strcat('./annotmitdb/', num2str(DS2(i)), '_anno.mat');
        anno = load(anno_file); anno = anno.anno;

        data_2lead = [];

        for lead = 1:2
            data_file = strcat('./denoised_mitdbdata/', 'denoised_', num2str(DS2(i)), '_', num2str(lead), '_data.mat');
            data = load(data_file); data = data.data_f;
            data_2lead = [data_2lead data];

        end

        %     r_peak_loc = rpeak_detection (data_2lead(:,1), fs);
        r_peak_loc = [anno{:, 1}];
        segs = [];
        labels = [];
        fea_plus = [];
        pre = 105;
        post = 150;

        for j = 2:(length(r_peak_loc) - 1)
            rr_pre = r_peak_loc(j) - r_peak_loc(j - 1);
            rr_post = r_peak_loc(j + 1) - r_peak_loc(j);

            if rr_pre > 500 || rr_post > 500
                continue;
            end

            if (r_peak_loc(j) + post < size(data_2lead, 1)) && (r_peak_loc(j) - pre > 0)
                seg = data_2lead(r_peak_loc(j) - pre:r_peak_loc(j) + post, :);
                label = find(class == anno{j, 2}) - 1;
                segs = cat(3, segs, seg');
                labels = [labels label];
                fea = [rr_post rr_pre];
                fea_plus = [fea_plus; fea];

            else
                continue;
            end

        end

        save_path = './segmitdb_250/';

        if ~isfolder(save_path)
            mkdir(save_path);
        end

        seg_file = strcat(save_path, num2str(DS2(i)), '_testseg.mat');
        label_seg_file = strcat(save_path, num2str(DS2(i)), '_testlabelseg.mat');
        fea_seg_file = strcat(save_path, num2str(DS2(i)), '_testfeaseg.mat');
        save(seg_file, 'segs');
        save(label_seg_file, 'labels');
        save(fea_seg_file, 'fea_plus')
    end
elseif database == 6
    DS =[101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230,100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234];
    
    class = ['N', 'S', 'V', 'F', 'Q'];
    fs = 250;

    % DS1
    for i = 1:length(DS)

        anno_file = strcat('./annotmitdb/', num2str(DS(i)), '_anno.mat');
        anno = load(anno_file); anno = anno.anno;

        data_2lead = [];

        for lead = 1:2
            data_file = strcat('./denoised_mitdbdata/', 'denoised_', num2str(DS(i)), '_', num2str(lead), '_data.mat');
            data = load(data_file); data = data.data_f;
            data_2lead = [data_2lead data];

        end

        %     r_peak_loc = rpeak_detection (data_2lead(:,1), fs);
        r_peak_loc = [anno{:, 1}];
        segs = [];
        labels = [];
        fea_plus = [];
        pre = 105;
        post = 150;

        for j = 2:(length(r_peak_loc) - 1)
            rr_pre = r_peak_loc(j) - r_peak_loc(j - 1);
            rr_post = r_peak_loc(j + 1) - r_peak_loc(j);

            if rr_pre > 500 || rr_post > 500
                continue;
            end

            if (r_peak_loc(j) + post < size(data_2lead, 1)) && (r_peak_loc(j) - pre > 0)
                seg = data_2lead(r_peak_loc(j) - pre:r_peak_loc(j) + post, :);
                label = find(class == anno{j, 2}) - 1;
                segs = cat(3, segs, seg');
                labels = [labels label];
                fea = [rr_post rr_pre];
                fea_plus = [fea_plus; fea];

            else
                continue;
            end

        end

        save_path = './segmitdb_250/';

        if ~isfolder(save_path)
            mkdir(save_path);
        end

        seg_file = strcat(save_path, num2str(DS(i)), '_seg.mat');
        label_seg_file = strcat(save_path, num2str(DS(i)), '_labelseg.mat');
        fea_seg_file = strcat(save_path, num2str(DS(i)), '_feaseg.mat');
        save(seg_file, 'segs');
        save(label_seg_file, 'labels');
        save(fea_seg_file, 'fea_plus');
    end
end

function label = anno2num(anno)

    if strcmp(anno, 'NORM')
        label = 0;
    else
        label = 5;
    end

end

function rpeak_i = rpeak_detection (ecg_f, fs)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    h_d = [-1 -2 0 2 1] * (1/8); %1/8*fs
    ecg_d = conv (ecg_f, h_d);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ecg_s = ecg_d.^2;
    ecg_m = conv(ecg_s, ones(1, 8));
    ecg_m = floor(ecg_m / 8);

    [pks, locs] = findpeaks(ecg_m, 'MINPEAKDISTANCE', round(0.2 * fs));
    % initialize the training phase (2 seconds of the signal) to determine the TH_m and THR_NOISE
    TH_m = max(ecg_m(4 * fs:6 * fs)) * 3/4;
    TH_pos = max(ecg_f(4 * fs:6 * fs)) * 1/2; % 0.25 of the max amplitude
    TH_neg = 0;

    refer_i = [];
    refer_amp = [];
    rpeak_amp = [];
    rpeak_i = [];
    rpeak_pole = [];
    rpeak_amp_pos = [];
    rpeak_amp_neg = [];
    TH_m_buf = [];
    TH_pos_buf = [];
    TH_neg_buf = [];
    rpeak_pole = [];

    for i = 1:length(pks)

        if (length(refer_i) > 8)
            diffRR = diff (refer_i (end - 8:end));
            mean_RR = mean(diffRR);

            if (locs(i) - refer_i(end)) >= 1.6 * mean_RR
                locs_temp_index = find((locs(i) - round(0.2 * fs)) > locs > (refer_i(end) + round(0.25 * fs)));

                if ~isempty(locs_temp_index)
                    [pks_temp, i_max] = max(ecg_m(locs(locs_temp_index)));
                    locs_temp = locs(locs_temp_index(i_max));

                    if ((pks_temp > 0.14 * TH_m) && ((locs_temp - refer_i(end)) > 0.25 * mean_RR))
                        refer_amp = [refer_amp pks_temp];
                        refer_i = [refer_i locs_temp];
                        Th_m = mean(refer_amp(end - 7:end)) * 0.75;
                        % TH_m = 0.25 * pks_temp + 0.25 * TH_m;

                        TH_pos = 0.2 * TH_pos;
                        TH_neg = 0.2 * TH_neg;
                        [y_i_t x_i_t pole] = find_rpeak(ecg_f, locs_temp, fs, TH_pos, TH_neg);

                        Slope1 = mean(diff(ecg_f(x_i_t - round(0.02 * fs):x_i_t)));
                        Slope2 = mean(diff(ecg_f(rpeak_i(end) - round(0.02 * fs):rpeak_i(end))));

                        if abs(Slope1) > 3 || abs(Slope1) >= abs(0.8 * (Slope2))
                            rpeak_i = [rpeak_i x_i_t]; % save index of bandpass
                            rpeak_amp = [rpeak_amp y_i_t]; %save amplitude of bandpas
                            % TH_pos = 0.25 * y_i_t + 0.75 * TH_pos; %when found with the second thres
                            TH_pos_buf = [TH_pos_buf TH_pos];
                            rpeak_pole = [rpeak_pole pole];

                            if (pole == 1)
                                rpeak_amp_pos = [rpeak_amp_pos y_i_t];

                                if (length(rpeak_amp_pos) > 7)
                                    Th_pos = mean(rpeak_amp_pos(end - 7:end)) * 0.75;
                                else
                                    Th_pos = mean(rpeak_amp_pos) * 0.75;
                                end

                            else
                                rpeak_amp_neg = [rpeak_amp_neg y_i_t];

                                if (length(rpeak_amp_neg) > 7)
                                    Th_neg = mean(rpeak_amp_neg(end - 7:end)) * 0.75;
                                else
                                    Th_neg = mean(rpeak_amp_neg) * 0.75;
                                end

                            end

                        end

                    end

                end

                if pks(i) > TH_m

                    if (locs(i) - refer_i(end)) > 0.25 * fs

                        refer_i = [refer_i locs(i)];
                        refer_amp = [refer_amp pks(i)];
                        Th_m = mean(refer_amp(end - 7:end)) * 0.75;

                        [y_i x_i pole] = find_rpeak(ecg_f, locs(i), fs, TH_pos, TH_neg);

                        if pole == 1

                            if y_i > TH_pos
                                rpeak_i = [rpeak_i x_i]; % save index of bandpass
                                rpeak_amp = [rpeak_amp y_i]; %save amplitude of bandpass
                                TH_pos_buf = [TH_pos_buf TH_pos];
                                rpeak_pole = [rpeak_pole 1];
                                rpeak_amp_pos = [rpeak_amp_pos y_i];

                                if (length(rpeak_amp_pos) > 7)
                                    Th_pos = mean(rpeak_amp_pos(end - 7:end)) * 0.75;
                                else
                                    Th_pos = mean(rpeak_amp_pos) * 0.75;
                                end

                            end

                        else

                            if y_i < TH_neg
                                rpeak_i = [rpeak_i x_i]; % save index of bandpass
                                rpeak_amp = [rpeak_amp y_i]; %save amplitude of bandpass
                                TH_pos_buf = [TH_pos_buf TH_neg];
                                rpeak_pole = [rpeak_pole -1];
                                rpeak_amp_neg = [rpeak_amp_neg y_i];

                                if (length(rpeak_amp_neg) > 7)
                                    Th_neg = mean(rpeak_amp_neg(end - 7:end)) * 0.75;
                                else
                                    Th_neg = mean(rpeak_amp_neg) * 0.75;
                                end

                            end

                        end

                    end

                end

            else

                if pks(i) > TH_m

                    if (locs(i) - refer_i(end)) > 0.25 * fs

                        refer_i = [refer_i locs(i)];
                        refer_amp = [refer_amp pks(i)];
                        Th_m = mean(refer_amp(end - 7:end)) * 0.75;
                        [y_i x_i pole] = find_rpeak(ecg_f, locs(i), fs, TH_pos, TH_neg);

                        if pole == 1

                            if y_i > TH_pos
                                rpeak_i = [rpeak_i x_i]; % save index of bandpass
                                rpeak_amp = [rpeak_amp y_i]; %save amplitude of bandpass
                                TH_pos_buf = [TH_pos_buf TH_pos];
                                rpeak_pole = [rpeak_pole 1];
                                rpeak_amp_pos = [rpeak_amp_pos y_i];

                                if (length(rpeak_amp_pos) > 7)
                                    Th_pos = mean(rpeak_amp_pos(end - 7:end)) * 0.75;
                                else
                                    Th_pos = mean(rpeak_amp_pos) * 0.75;
                                end

                            end

                        else

                            if y_i < TH_neg
                                rpeak_i = [rpeak_i x_i]; % save index of bandpass
                                rpeak_amp = [rpeak_amp y_i]; %save amplitude of bandpass
                                TH_pos_buf = [TH_pos_buf TH_neg];
                                rpeak_pole = [rpeak_pole -1];
                                rpeak_amp_neg = [rpeak_amp_neg y_i];

                                if (length(rpeak_amp_neg) > 7)
                                    Th_neg = mean(rpeak_amp_neg(end - 7:end)) * 0.75;
                                else
                                    Th_neg = mean(rpeak_amp_neg) * 0.75;
                                end

                            end

                        end

                    end

                end

            end

        else

            if (pks(i) > TH_m)
                refer_i = [refer_i locs(i)];
                refer_amp = [refer_amp pks(i)];
                Th_m = mean(refer_amp) * 0.75;
                [y_i x_i pole] = find_rpeak(ecg_f, locs(i), fs, TH_pos, TH_neg);

                if pole == 1

                    if y_i > TH_pos
                        rpeak_i = [rpeak_i x_i]; % save index of bandpass
                        rpeak_amp = [rpeak_amp y_i]; %save amplitude of bandpass
                        TH_pos_buf = [TH_pos_buf TH_pos];
                        rpeak_pole = [rpeak_pole 1];
                        rpeak_amp_pos = [rpeak_amp_pos y_i];

                        if (length(rpeak_amp_pos) > 7)
                            Th_pos = mean(rpeak_amp_pos(end - 7:end)) * 0.75;
                        else
                            Th_pos = mean(rpeak_amp_pos) * 0.75;
                        end

                    end

                else

                    if y_i < TH_neg
                        rpeak_i = [rpeak_i x_i];
                        rpeak_amp = [rpeak_amp y_i];
                        TH_pos_buf = [TH_pos_buf TH_neg];
                        rpeak_pole = [rpeak_pole -1];
                        rpeak_amp_neg = [rpeak_amp_neg y_i];

                        if (length(rpeak_amp_neg) > 7)
                            Th_neg = mean(rpeak_amp_neg(end - 7:end)) * 0.75;
                        else
                            Th_neg = mean(rpeak_amp_neg) * 0.75;
                        end

                    end

                end

            end

        end

    end

end

function [y_i x_i_h pole] = find_rpeak(ecg_f, locs, fs, THR_SIG, THR_SIG_neg)
    % THR_SIG = max(THR_SIG, 80);
    % THR_SIG_neg = min_2(THR_SIG_neg, -20);

    if locs <= length(ecg_f) & locs - round(0.150 * fs) > 0
        ecg_max = max(ecg_f(locs - round(0.150 * fs):locs));
        ecg_min = min(ecg_f(locs - round(0.150 * fs):locs));

        if ecg_max < THR_SIG & ecg_min < THR_SIG_neg
            pole = -1;
            [y_i x_i] = min(ecg_f(locs - round(0.150 * fs):locs));
            x_i_h = locs - round(0.150 * fs) + x_i -1;
        else
            pole = 1;
            [y_i x_i] = max(ecg_f(locs - round(0.150 * fs):locs));
            x_i_h = locs - round(0.150 * fs) + x_i -1;
        end

    elseif locs - round(0.150 * fs) < 1
        ecg_max = max(ecg_f(1:locs));
        ecg_min = min(ecg_f(1:locs));

        if ecg_max < THR_SIG & ecg_min < THR_SIG_neg
            pole = -1;
            [y_i x_i] = min(ecg_f(1:locs));
            x_i_h = x_i - 1;
        else
            pole = 1;
            [y_i x_i] = max(ecg_f(1:locs));
            x_i_h = x_i - 1;
        end

    elseif locs > length(ecg_f)
        ecg_max = max(ecg_f(locs - round(0.150 * fs):end));
        ecg_min = min(ecg_f(locs - round(0.150 * fs):end));

        if ecg_max < THR_SIG & ecg_min < THR_SIG_neg
            pole = -1;
            [y_i x_i] = min(ecg_f(locs - round(0.150 * fs):end));
            x_i_h = locs - round(0.150 * fs) + x_i -1;
        else
            pole = 1;
            [y_i x_i] = max(ecg_f(locs - round(0.150 * fs):end));
            x_i_h = locs - round(0.150 * fs) + x_i -1;
        end

    end

end

function seg = fix_length(seg_temp, len, mid_loc)

    if (size(seg_temp, 1) ~= 1)

        %         if size(seg_temp, 2) < len
        %             seg = [seg_temp zeros(size(seg_temp, 1), len - length(seg_temp))];
        %         else
        %             seg = seg_temp(:, 1:len);
        %         end
        if (size(seg_temp, 2) - mid_loc < len / 2) && (mid_loc < len / 2)
            seg = [zeros(size(seg_temp, 1), len / 2 - mid_loc) seg_temp zeros(size(seg_temp, 1), len / 2 - size(seg_temp, 2) + mid_loc)];
        elseif (size(seg_temp, 2) - mid_loc < len / 2) && (mid_loc >= len / 2)
            seg = [seg_temp(:, mid_loc - len / 2 + 1:end) zeros(size(seg_temp, 1), len / 2 - size(seg_temp, 2) + mid_loc)];
        elseif (size(seg_temp, 2) - mid_loc >= len / 2) && (mid_loc < len / 2)
            seg = [zeros(size(seg_temp, 1), len / 2 - mid_loc) seg_temp(:, 1:mid_loc + len / 2)];
        else
            seg = seg_temp(:, mid_loc - len / 2 + 1:mid_loc + len / 2);
        end

    else

        %         if length(seg_temp) < len
        %             seg = [zeros(1, len - length(seg_temp)) seg_temp zeros(1, len - length(seg_temp))];
        %         else
        %             seg = seg_temp(1:len);
        %         end
        if (length(seg_temp) - mid_loc < len / 2) && (mid_loc < len / 2)
            seg = [zeros(1, len / 2 - mid_loc) seg_temp zeros(1, len / 2 - length(seg_temp) + mid_loc)];
            length(seg);
        elseif (length(seg_temp) - mid_loc < len / 2) && (mid_loc >= len / 2)
            seg = [seg_temp(mid_loc - len / 2 + 1:end) zeros(1, len / 2 - length(seg_temp) + mid_loc)];
            length(seg);
        elseif (length(seg_temp) - mid_loc >= len / 2) && (mid_loc < len / 2)
            seg = [zeros(1, len / 2 - mid_loc) seg_temp(1:mid_loc + len / 2)];
            length(seg);
        else
            seg = seg_temp(mid_loc - len / 2 + 1:mid_loc + len / 2);
            length(seg);
        end

    end

end

function [Se, P, DER] = pt_metrics(p, predict_answer)

    L_true = size(p, 1);
    L_pre = length(predict_answer);
    TP = 0;
    Now_pre = 1;

    for i = 1:L_true

        if Now_pre <= L_pre

            for j = Now_pre:L_pre
                left = p(i, 1) - 30;
                right = p(i, 2) + 30;

                if left <= predict_answer(j) && right >= predict_answer(j)
                    TP = TP + 1;
                    Now_pre = Now_pre + 1;
                    break;
                elseif predict_answer(j) >= right
                    break;
                end

            end

        end

    end

    FN = L_true - TP;
    FP = L_pre - TP;
    Se = TP / (TP + FN);
    P = TP / (TP + FP);
    DER = (FN + FP) / (TP + FN);

end
