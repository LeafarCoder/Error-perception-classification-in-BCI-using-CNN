%% Paths
dataset_path2 = "D:\Thesis\BCI_datasets\Monitoring error-related potentials 2015\";
store_path2 = "D:\Thesis\Database_Monitoring_ErrP_temp\";

files_list2 = dir(dataset_path2 + '*.mat');
files_list2 = {files_list2.name};

%%
fs = 512;
labels = cell(6,2,10);
for i = 1:length(files_list2)
    file = files_list2{i};
    load(dataset_path2 + file);
    
    s_s = split(file,'_');
    subj = s_s{1}; del = isletter(subj); subj(del) = []; subj = str2num(subj);
    sess = s_s{2}; del = isletter(sess); sess(del) = []; sess = str2num(sess);
    for r = 1:10
        disp(num2str(i) + ", " + num2str(r));
        file_name = file(1:end-4) + "_run" + num2str(r) + ".csv";
        eeg = run{r}.eeg;
        
        % Add time
        time = linspace(0,size(eeg,1)/fs,size(eeg,1));
        eeg(:,2:end+1) = eeg(:,1:end);
        eeg(:,1) = time;

        % Get event location (samples)
        event_types = run{r}.header.EVENT.TYP;
        event_pos = run{r}.header.EVENT.POS;
        % types related to events (wrong or correct). remove all others        
        event_pos(~ismember(event_types, [5,6,9,10])) = [];
        events = zeros(1,size(eeg,1));
        events(event_pos) = 1;
        % Add events
        eeg(:,end+1) = events;
        
        run_labels = zeros(1,length(event_pos));
        % 5,10 (correct, noErrP); 6,9 (incorrect: ErrP)
        run_labels(ismember(event_types, [5,10])) = 1;
        run_labels(~ismember(event_types, [5,6,9,10])) = [];
        for k = 1:length(run_labels)
            labels{subj,sess,r,k} = run_labels(k);
        end
        
        header = {'Time',run{1}.header.Label{1:end-1},'FeedBackEvent'};
        
        % Save as .csv
        eeg_table = array2table(eeg,'VariableNames',header);
        writetable(eeg_table, store_path2 + "\csv_files\" + file_name);
        
    end
end

%% Convert labels to proper format
% IdFeedBack                Prediction
% Subject1_s01_run1_FB001   0
% Subject1_s01_run1_FB002   0
% ...
cell_labels = cell(1,2);
cnt = 1;
for subj = 1:6
    for sess = 1:2
        for run = 1:10
            % how many trials in this run?
            s = length(find([labels{subj,sess,run,:}]));
            for trial = 1:s
                cell_labels{cnt,1} = "S"+num2str(subj)+"_Sess"+num2str(sess)+"_run"+num2str(run)+"_FB"+num2str(trial);
                cell_labels{cnt,2} = labels{subj,sess,run,trial};
                cnt = cnt+1;
            end
        end
    end
end

labels_table = cell2table(cell_labels,'VariableNames',{'IdFeedBack','Prediction'});
writetable(labels_table, store_path2 + "\csv_files\AllLabels.csv");
