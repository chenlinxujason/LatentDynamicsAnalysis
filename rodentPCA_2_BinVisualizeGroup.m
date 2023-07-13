% Binning spiking, bin size = 50ms
% This plots the raster plots for each trial in one DNMS session
% blue represents LEFT lever, red represents RIGHT lever
% Author:  Zhouxiao Lu
% Date: Mar. 27, 2023
% Last modified on: Apr. 24, 2023

clear; clc; close all;
%p1 = fullfile('toolbox');
%addpath(genpath(p1));

% For single session
% Open file dialog box to select a .mat file
[file,path] = uigetfile('*.mat','Select the .mat file with only animal ID and session');% open the folder for user to select file
if isequal(file,0)
    disp('User selected Cancel');
else
    
 load(fullfile(path,file));  % Load the selected .mat file
 end

tic;

AnimalID = input('Enter animal ID: ', 's');%i.e 1029,1036,etc
SessionID = input('Enter session ID: ', 's');% comment this for all data
binSize = input('Enter bin size in sec(i.e.,0.001sec,meaning 1ms): ', 's');% comment this for all data
Arrange = input('Enter data arrangement way(choose between "neuron","time",or "average"):','s');
Data.Arrange = Arrange;

% set the name
DataFolder = '/home/song/PCA'; % User need to specific the folder that containing .nex files
myPath = fullfile(DataFolder, AnimalID);
%save_name = strcat(DataFolder,'/',AnimalID,'/1_TrialsData_peri',num2str(t_peri),'s');

% get all the mat file names from code 1 in the folder
dataname = dir(fullfile(myPath, '*.mat'));% lists all files names in the current folder.
filenames = {dataname.name};% lists all files names in the current folder
names = cell(1,length(filenames)); % Preallocate cell array for efficiency

%% Define bin size in seconds
bin_size = str2double(binSize);% in sec. i.e, 0.001 = 1ms
bin_size_ms = bin_size .* 1000;% ms,only for naming
binString = strcat('bin', num2str(bin_size_ms),'ms');

%for d = 1:length(filenames) % this is for AllData
  
 %load(filenames{d}); % this is for AllData. Load .mat file in the folder
 Data.bin_size = bin_size;

 trial_Realdata = Data.trial_neuron_timestamps;

% Define the time resolution for the raster plot
dt = 1/Data.Frequency; % seconds
t_extra = Data.ExtraMarginTime;

% Set animal ID and session ID
%matches = regexp(file, '(\d+)_(\d+)\.mat', 'tokens');
%AnimalID = str2double(matches{1}{1});
%SessionID = str2double(matches{1}{2});

% For plot
%save_name = strcat(DataFolder,'/',num2str(AnimalID),'/2_RasterPlots/',num2str(AnimalID),'_',num2str(SessionID),'_',binString);

%if ~exist(save_name,'dir')
%    mkdir(save_name);
%end


%% Loop over each trial

for i = 1:size(trial_Realdata, 2)
    figure('Visible','off');
    % Get the timestamps for the current trial
    trial_spikes = trial_Realdata(:, i);
    sample_type = Data.SamplePosition{i,1};
    response_type = Data.ResponsePosition{i,1};

    % Define the time window for plotting
    t_start = Data.trials_timestamps(i,1)-t_extra;
    t_end = Data.trials_timestamps(i,2)+t_extra;

    % Create time bins
    bins = t_start:bin_size:t_end;
    num_bins = length(bins)-1;
    t_window = [bins(1),bins(end)];

    % Allocate memory for the binned spike counts
    binned_spikes = zeros(1,num_bins);
    

    % Loop over each neuron
    for j = 1:size(trial_Realdata, 1)
        % Get the timestamps for the current neuron in the current trial
        neuron_spikes = trial_spikes{j,1}';
        % Loop through each time bin
        for k = 1:num_bins
            % Identify spikes that occurred within the current bin
            spikes_in_bin = neuron_spikes >= bins(k) & neuron_spikes < bins(k+1);
            
            % Count the number of spikes in the bin
            binned_spikes(k) = sum(spikes_in_bin);
        end
        %Data.binned{j,i} = binned_spikes;
        binned{j,i} = binned_spikes;
        
        %neuron_spikes = binned_spikes;
        
        % Plot a vertical line for each spike
        %subplot(2,1,1)
        %spikes = t_start + find(binned_spikes)*bin_size;
        %y_vals = j * ones(size(spikes));
        %line([spikes; spikes], [y_vals-0.5; y_vals+0.5], 'Color', 'k');
        %xx = (bins(1:end-1) + bins(2:end)) / 2;
        %subplot(2,1,2)
        %plot(xx,binned_spikes+j);
        %hold on
    end
    hold off
        
    % generate figures
    %for m = 1:2
    %    subplot(2,1,m)
        
        % Set the axis limits and labels
    %    xlim(t_window);
    %    ylim([0, size(trial_data, 1)+1]);
    %    ylabel('Neuron');
    %    xlabel('Time (s)');
        
        % Draw reactangles  
        % draw the SAMPLE rectangle
    %    t_peri = Data.PeriTime;
    %    rect_x = t_start + t_extra;
    %    rect_y = 0;
    %    rect_width = 2*t_peri;
    %    rect_height = size(trial_data,1);
    %    if strcmp(sample_type,'LEFT')
    %        rectangle('Position', [rect_x, rect_y, rect_width, rect_height], ...
    %                  'FaceColor', [0, 0, 1, 0.5],'EdgeColor', 'none');
    %    else
    %        rectangle('Position', [rect_x, rect_y, rect_width, rect_height], ...
    %                  'FaceColor', [1, 0, 0, 0.5],'EdgeColor', 'none');
    %    end
    
        % draw the RESPONSE rectangle
    %    rect_x = t_end - t_extra - 2*t_peri;
    %    rect_y = 0;
    %    rect_width = 2*t_peri;
    %    rect_height = size(trial_data,1);
    %    if strcmp(response_type,'LEFT')
    %        rectangle('Position', [rect_x, rect_y, rect_width, rect_height], ...
    %                  'FaceColor', [0, 0, 1, 0.5],'EdgeColor', 'none');
    %    else
    %        rectangle('Position', [rect_x, rect_y, rect_width, rect_height], ...
    %                  'FaceColor', [1, 0, 0, 0.5],'EdgeColor', 'none');
    %    end
    %end
    
    
    % Set the title of the subplot to the trial number
    %sgtitle(sprintf('Animal %d, Session %d, Trial %d, Bin size %d', AnimalID, SessionID, i,bin_size));
    %fig_name = sprintf('trial%d.png', i);
    %file_path = fullfile(save_name, fig_name);

    % Save the figure in PNG/EPS format with high resolution
    %saveas(gcf,file_path)
    % saveas(gcf,file_path,'epsc')
%end % for figure end
end % for trial end

%% Grouping
groups = {'LEFT','SAMPLE','CORRECT';'LEFT','SAMPLE','ERROR';
    'LEFT','NONMATCH','CORRECT';'LEFT','NONMATCH','ERROR';
    'RIGHT','SAMPLE','CORRECT';'RIGHT','SAMPLE','ERROR';
    'RIGHT','NONMATCH','CORRECT';'RIGHT','NONMATCH','ERROR'};

%groups = {'LEFT','SAMPLE';'RIGHT','NONMATCH';
%    'LEFT','NONMATCH';'RIGHT','SAMPLE'};

%% Define time window for event
extra_time = Data.ExtraMarginTime;%4s
t_peri = Data.PeriTime;%2s
t_start = round(extra_time/bin_size);% 
t_end = t_start+(2.*round(t_peri/bin_size));% 2.*PeriTime, +/- 2sec
% 2s: [-1s,1s]

for i = 1:size(Data.trial_neuron_timestamps,1)
    for j = 1:size(Data.trial_neuron_timestamps,2)
        trial_Realdata = binned{i,j};%raw real data
        
        [row,col] = size(binned{i,j});
        rng(89);
        trial_Noise = binned{i,j}(randperm(col));%noise
        
        n = length(trial_Realdata);
        error_trial_idx = [];
        
        if t_end > n
         continue; % This will skip to the next iteration of the j loop
        end
        
        sample_data = trial_Realdata(t_start:t_end);% real data sample
        nonmatch_data = trial_Realdata(n-t_end:n-t_start);% real data nonmatch
        sample_neuron{i,j} = sample_data;
        nonmatch_neuron{i,j} = nonmatch_data;
        
        sample_Noise = trial_Noise(t_start:t_end);% noise sample
        nonmatch_Noise= trial_Noise(n-t_end:n-t_start);% noise nonmatch
        sample_Noiseneuron{i,j} = sample_Noise;
        nonmatch_Noiseneuron{i,j} = nonmatch_Noise;
    end
end

%% Grouping step 1 - create 8 events group

for i = 1:length(groups)
    str_position = groups(i,1);
    str_phase = groups(i,2);
    str_type = groups(i,3);% uncomment this for 8 events
    
    if strcmp(str_phase,'SAMPLE');
        phase = Data.SamplePosition;
        dataReal = sample_neuron;
        dataNoise = sample_Noiseneuron;
        
    else strcmp(str_phase,'NONMATCH');
        phase = Data.ResponsePosition;
        dataReal = nonmatch_neuron;
        dataNoise = nonmatch_Noiseneuron;
    end
    type = Data.TrialType;% uncomment this for 8 events
    ind_position = find(strcmp(phase,str_position));
    ind_type = find(strcmp(type,str_type));% uncomment this for 8 events
    ind = intersect(ind_position,ind_type); % uncomment this for 8 events
    %ind = ind_position;% comment this out for 4 events
    
    m = zeros(size(binned,1),t_end-t_start+1);% for average (not use in PCA)
    p = m;% for average (not use in PCA)
    
    D_real = [];% for stack, PCA
    D_noise = [];% for stack, PCA
    for j = 1:length(ind)
        k = ind(j);
        d_real = cell2mat(dataReal(:,k));
        d_noise = cell2mat(dataNoise(:,k));
        
switch Arrange
            case 'neuron'
               D_real = [D_real;d_real];% stack neuron
               D_noise = [D_noise;d_noise];% stack neuron
            case 'time'
               D_real = [D_real,d_real];% stack time
               D_noise = [D_noise,d_noise];% stack time
            case 'average'
               m = m + d_real;
               p = p + d_noise;
        end
    
    end

    switch Arrange
        case 'average'
               Data.group_RawReal{i} = m/j; % average among events for each group
               Data.group_RawNoise{i} = p/j;%
        otherwise
               Data.group_RawReal{i} = D_real;
               Data.group_RawNoise{i} = D_noise;
    end
end
Data.groups = groups;

    
%% for single session Data - Don't use this!
% mat_name = sprintf('%s%s%s%s%s', startString, AnimalID, seperateString, SessionID, binString);
%example name: Animal1029_195bin1ms
%end % this is for AllData

%% for single session Data - new one (use this)
 str = string(file);
 parts = split(str, '.');  % split the string at the period
 newStr = parts(1); 
 switch Arrange
     case 'neuron'
         mat_name = sprintf('%s_%s_stackneuron',newStr,binString);
     case 'time'
         mat_name = sprintf('%s_%s_stacktime',newStr,binString);
     case 'average'
         mat_name = sprintf('%s_%s_average',newStr,binString);
 end
 save(mat_name,'Data','-v7.3');
 disp('job done')
elapsed_time = toc;
fprintf('Elapsed time: %.2f seconds\n', elapsed_time);


%% for all session Data
  %str = string(filenames(d));
  %parts = split(str, '.');  % split the string at the period
  %newStr = parts(1); 
  %mat_name = sprintf('%s_%s',newStr,binString);
  %save(mat_name,'Data','-v7.3');
  %disp(d)
  %disp('job done')
  %end % For all data
  %elapsed_time = toc;
  %fprintf('Elapsed time: %.2f seconds\n', elapsed_time);