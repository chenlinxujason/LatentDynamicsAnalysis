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
% [file,path] = uigetfile('*.mat','Select the .mat file with only animal ID and session');% open the folder for user to select file
% if isequal(file,0)
%     disp('User selected Cancel');
% else
%     
%  load(fullfile(path,file));  % Load the selected .mat file
%  end

tic;

AnimalID = input('Enter animal ID: ', 's');%i.e 1029,1036,etc
%SessionID = input('Enter session ID: ', 's');% comment this for all data
binSize = input('Enter bin size in sec(i.e.,0.001sec,meaning 1ms): ', 's');% comment this for all data

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
binString = strcat('bin', num2str(bin_size_ms),'ms','_test');

for d = 1:length(filenames) % this is for AllData
  
 load(filenames{d}); % Load .mat file in the folder
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
        Data.binned{j,i} = binned_spikes;
        %binned{j,i} = binned_spikes;
        
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

%% Define time window for event
bin_size = Data.bin_size;
bin_size_ms = bin_size .* 1000;% ms,only for naming
extra_time = Data.ExtraMarginTime;
t_peri = Data.PeriTime;
t_start = round(extra_time/bin_size);
t_end = t_start+(2.*round(t_peri/bin_size));

%% for single session Data - new one (use this)
%  str = string(file);% not file name!
%  parts = split(str, '.');  % split the string at the period
%  newStr = parts(1); 
%  mat_name = sprintf('%s_%s',newStr,binString);
%  save(mat_name,'Data','-v7.3');
%  disp('job done')
% elapsed_time = toc;
% fprintf('Elapsed time: %.2f seconds\n', elapsed_time);

%% for all session Data
 str = string(filenames(d));
 parts = split(str, '.');  % split the string at the period
 newStr = parts(1); 
 mat_name = sprintf('%s_%s',newStr,binString);
 save(mat_name,'Data','-v7.3');
 disp(d)
 disp('job done')
 end % For all data
 elapsed_time = toc;
 fprintf('Elapsed time: %.2f seconds\n', elapsed_time);
