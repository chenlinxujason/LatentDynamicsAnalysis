% This file is used to extract neural information from NEX files
% No binning, no smoothing
% Including trialwise data
% Author:  Zhouxiao Lu
% Date: Mar. 27, 2023
% Last modified on: Mar. 27, 2023

clear; clc; close all;
%p1 = fullfile('toolbox');
%addpath(genpath(p1));

% ask the user to enter the animal ID
AnimalID = input('Enter animal ID: ', 's');%i.e 1029,1036,etc
t_peri = input('Enter perievent time (s): ');%i.e, 2s, [-2,2]
t_margin = input('Enter extra margin time (s): ');%i.e,1s

% set the file path, create the folder to store data
DataFolder = '/home/song/PCA';
%DataFolder = 'D:\Rodent WFU DNMS'; % User need to specific the folder that containing .nex files
myPath = fullfile(DataFolder, AnimalID);
save_name = strcat(DataFolder,'/',AnimalID,'/1_TrialsData_peri',num2str(t_peri),'s');

% get all the NEX file names in the folder
dataname = dir(fullfile(myPath, '*.nex'));%lists files and folders in the current folder.
filenames = {dataname.name};

% key event names
str_trial = 'TRIAL';
str_Left_Sample = 'A_SAMPLES'; % Left lever was pressed
str_Left_Match = 'A_MATCH'; % Left
str_Right_Nonmatch = 'B_NONMATCH'; % Left
str_Right_Sample = 'B_SAMPLES'; % Right lever was pressed
str_Right_Match = 'B_MATCH'; % Right
str_Left_Nonmatch = 'A_NONMATCH'; % Right

AllData = struct();% Store all raw data in the assigned folder

% set the name
names = cell(1,length(filenames)); % Preallocate cell array for efficiency
startString = 'Animal';% This string is to assign a start letter to name the field
seperateString = '_';
for d = 1:length(filenames)
    str = sprintf('%s%s%s%d', startString, AnimalID, seperateString, d); 
    names{d} = str;
end

for d = 1:length(filenames)
    % read the NEX file
    file_name = fullfile(myPath, filenames{d});
    nexFileData = readNexFile(file_name);

    % extract the neuron names and event markers
    num_neuron = length(nexFileData.neurons);
    str_neuron_name = strings([1,num_neuron]);
    num_events = length(nexFileData.events);
    str_events_name = strings([1,num_events]);
    freq = nexFileData.freq;
    t_delta = 1/freq;
    if isfield(nexFileData, 'markers')
        markers = nexFileData.markers;
    else
        markers = nexFileData.events;
    end
    num_markers = length(markers);
    str_markers = strings([1,num_markers]);
    for i = 1:num_neuron
        str_neuron_name(i) = nexFileData.neurons{i,1}.name;
    end
    for i = 1:num_markers
        str_markers(i) = markers{i,1}.name;
    end
    for i = 1:num_events
        str_events_name(i) = nexFileData.events{i,1}.name;
    end

    % extract sample-phase timestamps
    left_sample_indices = find(str_events_name == str_Left_Sample);
    left_sample_timestamps = nexFileData.events{left_sample_indices, 1}.timestamps;
    if isempty(left_sample_indices)
        left_sample_timestamps = [];
        num_left_samples = 0;
    else
        left_sample_timestamps = nexFileData.events{left_sample_indices, 1}.timestamps;
        num_left_samples = length(left_sample_timestamps);
    end
    
    right_sample_indices = find(str_events_name == str_Right_Sample);
    if isempty(right_sample_indices)
        right_sample_timestamps = [];
        num_right_samples = 0;
    else
        right_sample_timestamps = nexFileData.events{right_sample_indices, 1}.timestamps;
    end
    samples_timestamps = sort([left_sample_timestamps;right_sample_timestamps]);

    % extract response-phase timestamps
    % match
    left_match_indices = find((str_events_name == str_Left_Match));
    if isempty(left_match_indices)
        left_match_timestamps = [];
        num_left_match = 0;
    else
        left_match_timestamps = nexFileData.events{left_match_indices, 1}.timestamps;
        num_left_match = length(left_match_timestamps);
    end

    right_match_indices = find((str_events_name == str_Right_Match));
    if isempty(right_match_indices)
        right_match_timestamps = [];
        num_right_match = 0;
    else
        right_match_timestamps = nexFileData.events{right_match_indices, 1}.timestamps;
        num_right_match = length(right_match_timestamps);
    end

    % nonmatch
    left_nonmatch_indices = find((str_events_name == str_Right_Nonmatch));
    if isempty(left_nonmatch_indices)
        left_nonmatch_timestamps = [];
        num_left_nonmatch = 0;
    else
        left_nonmatch_timestamps = nexFileData.events{left_nonmatch_indices, 1}.timestamps;
        num_left_nonmatch = length(left_nonmatch_timestamps);
    end

    right_nonmatch_indices = find((str_events_name == str_Left_Nonmatch));
    if isempty(right_nonmatch_indices)
        right_nonmatch_timestamps = [];
        num_right_nonmatch = 0;
    else
        right_nonmatch_timestamps = nexFileData.events{right_nonmatch_indices, 1}.timestamps;
        num_right_nonmatch = length(right_nonmatch_timestamps);
    end

    % combined response-phase timestamps
    responses_timestamps = sort([left_match_timestamps;right_match_timestamps;left_nonmatch_timestamps;right_nonmatch_timestamps]);
    response_timestamps = find(~isnan(responses_timestamps));
    responses_timestamps = responses_timestamps(response_timestamps);

    % extract timestamps for each trial
    num_trial = min(length(samples_timestamps),length(responses_timestamps));
    trials_timestamps = cat(2,samples_timestamps(1:num_trial),responses_timestamps(1:num_trial));
    trials_timestamps(:,3) = trials_timestamps(:,2)-trials_timestamps(:,1);

    % extract trial types
    is_Left_Sample = ismember(samples_timestamps(1:num_trial), left_sample_timestamps);
    type_Samples = cell(size(samples_timestamps(1:num_trial)));
    type_Samples(is_Left_Sample) = {'LEFT'};
    type_Samples(~is_Left_Sample) = {'RIGHT'};

    is_Left_Response_1 = ismember(responses_timestamps(1:num_trial), left_nonmatch_timestamps);
    is_Left_Response_2 = ismember(responses_timestamps(1:num_trial), left_match_timestamps);
    is_Left_Response = is_Left_Response_1 | is_Left_Response_2;
    type_Responses = cell(size(responses_timestamps(1:num_trial)));
    type_Responses(is_Left_Response) = {'LEFT'};
    type_Responses(~is_Left_Response) = {'RIGHT'};

    % Preallocate memory for type_trial
    type_trial = cell(length(trials_timestamps), 1);

    % Compare type_Samples and type_Responses
    is_failure = strcmp(type_Samples, type_Responses);

    % Set type_trial based on is_success
    type_trial(is_failure) = {'ERROR'};
    type_trial(~is_failure) = {'CORRECT'};

    % Preallocate memory for Data
    Data.trial_neuron_timestamps = cell(num_neuron, 1);

    for i = 1:num_neuron
        datas = nexFileData.neurons{i, 1}.timestamps;
        data = zeros(length(datas), num_trial);

        for j = 1:num_trial
            low = trials_timestamps(j, 1) - t_peri - t_margin;
            high = trials_timestamps(j, 2) + t_peri + t_margin;
            ind_data = find((datas >= low) & (datas < high));
            Data.trial_neuron_timestamps{i,j} = datas(ind_data);

        end

    Data.SamplePosition = type_Samples;
    Data.ResponsePosition = type_Responses;
    Data.TrialType = type_trial;
    Data.Animal = AnimalID;
    Data.Session = file_name;
    Data.PeriTime = t_peri;
    Data.ExtraMarginTime = t_margin;
    Data.trials_timestamps = trials_timestamps;
    Data.Frequency = freq;
    end

    % save data
    if ~exist(save_name,'dir')
        mkdir(save_name);
    end
    %mat_name = sprintf('%s_%s',AnimalID,string(d));
    str = string(filenames(d));
    parts = split(str, '-');  % split the string at the period
    newStr = parts(1); 
    mat_name = newStr;
    %mat_name = sprintf('%s',string(filenames(d)));
    %mat_name = sprintf('%s.mat',mat_name);
    %mat_name = fullfile(save_name, mat_name);
     
    % change structure name and save
    %str = string(filenames(d));
    %parts = split(str, '-');  % split the string at the period
    %newStr = parts(1);  % take the first part
    %newStruct = Data;
    %assignin('base', Data, newStr);
    %newfilename = [newStr '.mat'];  % create filename
    %save(newfilename, newStr);  % save the structure
    
    AllData.(names{d}) = Data;
   
    save(mat_name,'Data')
    save(AnimalID,'AllData');
    disp('job done')
    disp(file_name)
    disp(d)
end