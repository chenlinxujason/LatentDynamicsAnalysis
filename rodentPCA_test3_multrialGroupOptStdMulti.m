clear; clc; close all;
Com = input('Enter comparison type(choose between "type","trial",or "all"):','s');
SessionData.Com = Com;

Arrange = input('Enter data arrangement way(choose between "neuron","time",or "average"):','s');
SessionData.Arrange = Arrange;


[file1,path1] = uigetfile('*.mat','Select the first binned session .mat file');
if isequal(file1,0)
    disp('User selected Cancel');
    return
end


[file2,path2] = uigetfile('*.mat','Select the second binned session .mat file');
if isequal(file2,0)
    disp('User selected Cancel');
    return
end

dlgtitle = 'Input(CAPITAL)';
dims = [2 48];% [wide length] 

switch Com
       case 'all'
            disp('Compare 8 events');
            input_info = {'8 events'};
       case 'type'
            disp('Types include: LEFT,RIGHT,SAMPLE,NONMATCH,CORRECT,ERROR');
            prompt = {'Enter Type you would like to compare (i.e., LEFT):'};
            input_info = inputdlg(prompt,dlgtitle,dims);
       case 'trial'
            disp('Trials include: LEFT SAMPLE CORRECT, RIGHT SAMPLE ERROR, etc');
            prompt = {'Enter Trial, position (LEFT or RIGHT):', 'Enter Trial, phase (SAMPLE or NONMATCH):', 'Enter Trial, outcome (CORRECT or ERROR):'};
            input_info = inputdlg(prompt,dlgtitle,dims);
        otherwise
            disp('User input error');
            return;
end

tic
SessionData.input_info = input_info;
%% Load the selected .mat files
load(fullfile(path1,file1)); 
Data1 = Data;
    
load(fullfile(path2,file2)); 
Data2 = Data;

groups = {'LEFT','SAMPLE','CORRECT';'LEFT','SAMPLE','ERROR';
    'LEFT','NONMATCH','CORRECT';'LEFT','NONMATCH','ERROR';
    'RIGHT','SAMPLE','CORRECT';'RIGHT','SAMPLE','ERROR';
    'RIGHT','NONMATCH','CORRECT';'RIGHT','NONMATCH','ERROR'};

%% Define time window for event
bin_size = Data.bin_size;
bin_size_ms = bin_size.*1000;% covert binSize from sec to ms
extra_time = Data.ExtraMarginTime;
t_peri = Data.PeriTime;
t_start = round(extra_time/bin_size);
if t_start == 0
    t_start = 1;
end
t_end = t_start+(2.*round(t_peri/bin_size));

%% Get session(animal)1 ID
str1 = file1;
parts1 = split(str1, '_');  
newStr1 = parts1(1); 
newStr1 = char(newStr1);
SessionData.session1 = newStr1;

%% Get session(animal)2 ID
str2 = file2;
parts2 = split(str2, '_');  
newStr2 = parts2(1); 
newStr2 = char(newStr2);
SessionData.session2 = newStr2;

%% Get bin size information
binStr = parts1{2,1};
binStr = split(binStr,'.'); 
binStr = char(binStr{1,1});
SessionData.binStr = binStr;

%% Save time windows
SessionData.extra_time = extra_time;
SessionData.t_peri = t_peri;
SessionData.t_start = t_start;
SessionData.t_end = t_end;

%% Save groups
SessionData.groups = groups;


%% Define standard deviation vector for smooth
%% bin 1ms
% sigma_vec1 = linspace(1,30,30);% bin1ms
% sigma_vec2 = logspace(log10(30),log10(100),10);% bin1ms
% sigma_vec = [sigma_vec1 sigma_vec2(2:end)];

%% bin 10ms
sigma_vec1 = linspace(1,3,21);% bin1ms
sigma_vec2 = logspace(log10(3),log10(10),10);% bin1ms
sigma_vec = [sigma_vec1 sigma_vec2(2:end)];

% repeat times
numTrial = 5;
disp(['Compare between:' newStr1 ' vs. ' newStr2])
disp(['Shuffle times: ' num2str(numTrial)]);

% Preallocate memory
r_real = zeros(1,length(sigma_vec));
r_noise = zeros(1,length(sigma_vec));
%MseOne = zeros(1,length(sigma_vec));
%MseTwo = zeros(1,length(sigma_vec));

R_real = zeros(numTrial,length(sigma_vec));
R_noise = zeros(numTrial,length(sigma_vec));
%trialMseOne = zeros(numTrial,length(sigma_vec));
%trialMseTwo = zeros(numTrial,length(sigma_vec));

binned1 = Data1.binned;
binned2 = Data2.binned;
numNeuron1 = size(Data1.binned,1);
numNeuron2 = size(Data2.binned,1);

for trial = 1:numTrial % 15 trials 

%% shuffle Data1
 for i = 1:size(Data1.trial_neuron_timestamps,1)
     for j = 1:size(Data1.trial_neuron_timestamps,2)
        trial_Realdata1 = binned1{i,j};%raw real data
        
        [row,col] = size(binned1{i,j});
        %rng(89);
        trial_Noise = binned1{i,j}(randperm(col));%noise
        
        n = length(trial_Realdata1);
                 
        if t_end > n
         continue; % This will skip to the next iteration of the j loop
        end

        sample_data = trial_Realdata1(t_start:t_end);% real data sample
        nonmatch_data = trial_Realdata1(n-t_end:n-t_start);% real data nonmatch
        Data1.sample_neuron{i,j} = sample_data;
        Data1.nonmatch_neuron{i,j} = nonmatch_data;
        
        sample_Noise = trial_Noise(t_start:t_end);% noise sample
        nonmatch_Noise= trial_Noise(n-t_end:n-t_start);% noise nonmatch
        Data1.sample_Noiseneuron{i,j} = sample_Noise;
        Data1.nonmatch_Noiseneuron{i,j} = nonmatch_Noise;
    end
end

%% Grouping step 1 - create 8 events group

for i = 1:length(groups)
    str_position = groups(i,1);
    str_phase = groups(i,2);
    str_type = groups(i,3);
    
    if strcmp(str_phase,'SAMPLE');
        phase = Data1.SamplePosition;
        dataReal1 = Data1.sample_neuron;
        dataNoise1 = Data1.sample_Noiseneuron;
        
    else strcmp(str_phase,'NONMATCH');
        phase = Data1.ResponsePosition;
        dataReal1 = Data1.nonmatch_neuron;
        dataNoise1 = Data1.nonmatch_Noiseneuron;
    end
    type = Data1.TrialType;
    ind_position = find(strcmp(phase,str_position));
    ind_type = find(strcmp(type,str_type));
    ind = intersect(ind_position,ind_type);

    m1 = zeros(size(binned1,1),t_end-t_start+1);% for average (not use in PCA)
    p1 = m1;% for average (not use in PCA)
    
    D1_real = [];% for stack, PCA
    D1_noise = [];% for stack, PCA
    for j = 1:length(ind)
        k = ind(j);
        d1_real = cell2mat(dataReal1(:,k));
        d1_noise = cell2mat(dataNoise1(:,k));
        
        switch Arrange
            case 'neuron'
               D1_real = [D1_real;d1_real];% stack neuron
               D1_noise = [D1_noise;d1_noise];% stack neuron
            case 'time'
               D1_real = [D1_real,d1_real];% stack time
               D1_noise = [D1_noise,d1_noise];% stack time
            case 'average'
               m1 = m1 + d_real;
               p1 = p1 + d_noise;
        end
    
    end

    switch Arrange
        case 'average'
               Data1.group_RawReal{i} = m1/j; % average among events for each group
               Data1.group_RawNoise{i} = p1/j;%
        otherwise
               Data1.group_RawReal{i} = D1_real;
               Data1.group_RawNoise{i} = D1_noise;
    end
end
Data1.groups = groups;
% End shuffle Data1

%% Data2

 for i = 1:size(Data2.trial_neuron_timestamps,1)
     for j = 1:size(Data2.trial_neuron_timestamps,2)
        trial_Realdata2 = binned2{i,j};%raw real data
        
        [row,col] = size(binned2{i,j});
        %rng(89);
        trial_Noise = binned2{i,j}(randperm(col));%noise
        
        n = length(trial_Realdata2);
                 
        if t_end > n
         continue; % This will skip to the next iteration of the j loop
        end

        sample_data = trial_Realdata2(t_start:t_end);% real data sample
        nonmatch_data = trial_Realdata2(n-t_end:n-t_start);% real data nonmatch
        Data2.sample_neuron{i,j} = sample_data;
        Data2.nonmatch_neuron{i,j} = nonmatch_data;
        
        sample_Noise = trial_Noise(t_start:t_end);% noise sample
        nonmatch_Noise= trial_Noise(n-t_end:n-t_start);% noise nonmatch
        Data2.sample_Noiseneuron{i,j} = sample_Noise;
        Data2.nonmatch_Noiseneuron{i,j} = nonmatch_Noise;
    end
end

%% Grouping step 1 - create 8 events group

for i = 1:length(groups)
    str_position = groups(i,1);
    str_phase = groups(i,2);
    str_type = groups(i,3);
    
    if strcmp(str_phase,'SAMPLE');
        phase = Data2.SamplePosition;
        dataReal2 = Data2.sample_neuron;
        dataNoise2 = Data2.sample_Noiseneuron;
        
    else strcmp(str_phase,'NONMATCH');
        phase = Data2.ResponsePosition;
        dataReal2 = Data2.nonmatch_neuron;
        dataNoise2 = Data2.nonmatch_Noiseneuron;
    end
    type = Data2.TrialType;
    ind_position = find(strcmp(phase,str_position));
    ind_type = find(strcmp(type,str_type));
    ind = intersect(ind_position,ind_type);

    m2 = zeros(size(binned2,1),t_end-t_start+1);% for average (not use in PCA)
    p2 = m2;% for average (not use in PCA)
    
    D2_real = [];% for stack, PCA
    D2_noise = [];% for stack, PCA
    for j = 1:length(ind)
        k = ind(j);
        d2_real = cell2mat(dataReal2(:,k));
        d2_noise = cell2mat(dataNoise2(:,k));
        
        switch Arrange
           case 'neuron'
                D2_real = [D2_real;d2_real];% stack neuron
                D2_noise = [D2_noise;d2_noise];% stack neuron
            case 'time'
                D2_real = [D2_real,d2_real];% stack time
                D2_noise = [D2_noise,d2_noise];% stack time
            case 'average'
                m2 = m2 + d_real;
                p2 = p2 + d_noise;
        end
    end

         switch Arrange
             case 'average'
                 Data2.group_RawReal{i} = m/j; % average among events for each group
                 Data2.group_RawNoise{i} = p/j;%
             otherwise    
                 Data2.group_RawReal{i} = D2_real;
                 Data2.group_RawNoise{i} = D2_noise;
         end
end
Data2.groups = groups;

%% Switch case for index
switch Com
    case 'all'
    n = min(length(Data1.group_RawReal),...
            length(Data2.group_RawReal));% the number of event for comparison
    ind = 1:n;
    
    case 'type'% case1: type-wise comparison
    for i = 1:size(groups,2)
    g = find(strcmp(groups(:,i), input_info));
    
    if ~isempty(g)
        ind = g;
        break
    else
      continue
    end
    end

    case 'trial'% case2: trial wise comparison 
    ind = find(strcmp(groups(:,1),input_info{1}) & ...
    strcmp(groups(:,2),input_info{2}) & strcmp(groups(:,3),input_info{3}));
end

%% smooth
for std = 1:length(sigma_vec)
%    window = bin_size_ms.*5*sigma_vec(std); % window size in ms
    window = 5*sigma_vec(std); % window size
    x = -window:window; % domain of the kernel
    kernel = exp(-x.^2/(2*sigma_vec(std)^2)) / (sigma_vec(std)*sqrt(2*pi)); % Gaussian kernel

    % Smooth spike firing counts with Gaussian kernel
    % Prelocate memory
    group_SmoothedReal1 = cell(1,length(ind));
    group_SmoothedNoise1 = cell(1,length(ind));
    group_SmoothedReal2 = cell(1,length(ind));
    group_SmoothedNoise2 = cell(1,length(ind));

    for i = 1:length(ind) % in most cases,8, which corresponding to 8 events
        
         Realspike1_counts = Data1.group_RawReal{i};%raw Real data1
         Noisespike1_counts = Data1.group_RawNoise{i};%raw Noise data1
         
         Realspike2_counts = Data2.group_RawReal{i};%raw Real data2
         Noisespike2_counts = Data2.group_RawNoise{i};%raw Noise data2
         
         % Preallocate an array for the smoothed counts
         Realsmooth1_counts = zeros(size(Realspike1_counts));  
         Noisesmooth1_counts = zeros(size(Noisespike1_counts));
         
         Realsmooth2_counts = zeros(size(Realspike2_counts));  
         Noisesmooth2_counts = zeros(size(Noisespike2_counts));
         
           for j = 1:size(Realspike1_counts,1)  % Loop over each row
               Realsmooth1_counts(j,:) = conv(Realspike1_counts(j,:), kernel, 'same');
               Noisesmooth1_counts(j,:) = conv(Noisespike1_counts(j,:), kernel, 'same');
           end
           
           for j = 1:size(Realspike2_counts,1)  % Loop over each row
               Realsmooth2_counts(j,:) = conv(Realspike2_counts(j,:), kernel, 'same');
               Noisesmooth2_counts(j,:) = conv(Noisespike2_counts(j,:), kernel, 'same');
           end

         group_SmoothedReal1{i} = Realsmooth1_counts;
         group_SmoothedNoise1{i} = Noisesmooth1_counts;
         
         group_SmoothedReal2{i} = Realsmooth2_counts;
         group_SmoothedNoise2{i} = Noisesmooth2_counts;
    end

%% Extract data 
if strcmp (Com, 'all')
       EventOne = group_SmoothedReal1;
       NoiseOne = group_SmoothedNoise1;
       
       EventTwo = group_SmoothedReal2;
       NoiseTwo = group_SmoothedNoise2;
else    
    for j = 1:length(ind)
        k = ind(j);
        EventOne{j} = group_SmoothedReal1{k};
        NoiseOne{j} = group_SmoothedNoise1{k};

        EventTwo{j} = group_SmoothedReal2{k};
        NoiseTwo{j} = group_SmoothedNoise2{k};
    end
end
    
%% Calculate PCA using the same mode
switch Arrange
    case 'neuron'
%% vertcat - stack neuron
numRows1 = cellfun(@(x) size(x, 1), EventOne);
numRows2 = cellfun(@(x) size(x, 1), EventTwo);

numTri1 = numRows1/numNeuron1;% number of trials for animal1
numTri2 = numRows2/numNeuron2;% number of trials for animal2

EventOneMatrix = vertcat(EventOne{:}); 
EventTwoMatrix = vertcat(EventTwo{:}); 
NoiseOneMatrix = vertcat(NoiseOne{:}); 
NoiseTwoMatrix = vertcat(NoiseTwo{:});

[coeff1R,score1R,explained1R,m1R] = PCA(EventOneMatrix',3,0.6);
[coeff2R,score2R,explained2R,m2R] = PCA(EventTwoMatrix',3,0.6);
[coeff1N,score1N,explained1N,m1N] = PCA(NoiseOneMatrix',3,0.6);
[coeff2N,score2N,explained2N,m2N] = PCA(NoiseTwoMatrix',3,0.6);
mR = min(m1R,m2R);
mN = min(m1N,m2N);
m = min(mR,mN);  % average the top 4 PC for comparison
numPC = min(m,3); 

startIdx1 = [1 cumsum(numRows1(1:end-1))+1]; % start indices of each block
endIdx1 = cumsum(numRows1); % end indices of each block

startIdx2 = [1 cumsum(numRows2(1:end-1))+1]; % start indices of each block
endIdx2 = cumsum(numRows2); % end indices of each block

EventOnePC = cell(1, length(ind)); % initialize cell array
EventTwoPC = cell(1, length(ind)); % initialize cell array
EventOneLD = cell(1, length(ind)); % initialize cell array
EventTwoLD = cell(1, length(ind)); % initialize cell array

NoiseOnePC = cell(1, length(ind)); % initialize cell array
NoiseTwoPC = cell(1, length(ind)); % initialize cell array
NoiseOneLD = cell(1, length(ind)); % initialize cell array
NoiseTwoLD = cell(1, length(ind)); % initialize cell array

for i = 1:length(ind)% length(ind) = length(numRows1) = 8
    EventOnePC{i} = coeff1R(startIdx1(i):endIdx1(i), 1:mR);
    EventOneLD{i} = EventOne{i}' * EventOnePC{i};
    
    EventTwoPC{i} = coeff2R(startIdx2(i):endIdx2(i), 1:mR);
    EventTwoLD{i} = EventTwo{i}' * EventTwoPC{i};
end

for i = 1:length(ind)% length(ind) = length(numRows1) = 8
    NoiseOnePC{i} = coeff1N(startIdx1(i):endIdx1(i), 1:mN);
    NoiseOneLD{i} = NoiseOne{i}' * NoiseOnePC{i};
    
    NoiseTwoPC{i} = coeff2N(startIdx2(i):endIdx2(i), 1:mN); 
    NoiseTwoLD{i} = NoiseTwo{i}' * NoiseTwoPC{i};
end

% scale trial
for i = 1:length(ind)
     EventOneLD{i} = EventOneLD{i}/numTri1(i);
     EventTwoLD{i} = EventTwoLD{i}/numTri2(i);
     NoiseOneLD{i} = NoiseOneLD{i}/numTri1(i);
     NoiseTwoLD{i} = NoiseTwoLD{i}/numTri2(i);
end

    case 'time'

%% horzcat - stack time
% scale original data before stack 
time_length = t_end - t_start + 1;
numCols1 = cellfun(@(x) size(x,2), EventOne);%minCols1 = min(numCols1);
numCols2 = cellfun(@(x) size(x,2), EventTwo);%minCols2 = min(numCols2);

numTri1 = numCols1/time_length;% number of trials for animal1
numTri2 = numCols2/time_length;% number of trials for animal2

% scale1 = (sum(numTri1)-numTri1)/sum(numTri1);
% scale2 = (sum(numTri2)-numTri2)/sum(numTri2);
% 
% for i = 1:length(ind)
%     EventOne{i} = scale1(i)*EventOne{i};
%     EventTwo{i} = scale2(i)*EventTwo{i};
%     NoiseOne{i} = scale1(i)*NoiseOne{i};
%     NoiseTwo{i} = scale2(i)*NoiseTwo{i};
% end

EventOneMatrix = horzcat(EventOne{:}); 
EventTwoMatrix = horzcat(EventTwo{:});
NoiseOneMatrix = horzcat(NoiseOne{:}); 
NoiseTwoMatrix = horzcat(NoiseTwo{:});


[coeff1R,score1R,explained1R,m1R] = PCA(EventOneMatrix',3,0.6);
[coeff2R,score2R,explained2R,m2R] = PCA(EventTwoMatrix',3,0.6);
[coeff1N,score1N,explained1N,m1N] = PCA(NoiseOneMatrix',3,0.6);
[coeff2N,score2N,explained2N,m2N] = PCA(NoiseTwoMatrix',3,0.6);
mR = min(m1R,m2R);
mN = min(m1N,m2N);
m = min(mR,mN);  % average the top 4 PC for comparison
% numPC = min(m,3);
numPC = 3;

EventOnePC = coeff1R(:,1:mR);
EventTwoPC = coeff2R(:,1:mR);
NoiseOnePC = coeff1N(:,1:mN);
NoiseTwoPC = coeff2N(:,1:mN);

LDrOne = EventOneMatrix'*(EventOnePC);% LDOne is A
LDrTwo = EventTwoMatrix'*(EventTwoPC);% LDTwo 
LDnOne = NoiseOneMatrix'*(NoiseOnePC);% LDOne is A
LDnTwo = NoiseTwoMatrix'*(NoiseTwoPC);% LDTwo 

EventOneLD = mat2cell(LDrOne, numCols1, mR);
EventTwoLD = mat2cell(LDrTwo, numCols2, mR);
NoiseOneLD = mat2cell(LDnOne, numCols1, mN);
NoiseTwoLD = mat2cell(LDnTwo, numCols2, mN);

% reshape1 = cell(1,length(ind));
% reshape2 = cell(1,length(ind));
EventOneLDAve = cell(1,length(ind));
EventTwoLDAve = cell(1,length(ind));
NoiseOneLDAve = cell(1,length(ind));
NoiseTwoLDAve = cell(1,length(ind));

for i = 1:length(ind)
    % Check if the number of rows in the current cell is a multiple of the current factor
%     if mod(size(EventOneLD{i},1),numTri1(i)) ~= 0 ||...
%             mod(size(EventTwoLD{i},1),numTri2(i)) ~= 0
%         error('Not a multiple of factor');
%     end
    
    for j = 1:mR
        reshape1r = reshape(EventOneLD{i}(:,j),time_length,numTri1(i));
        reshape2r = reshape(EventTwoLD{i}(:,j),time_length,numTri2(i));
        LDrOneAve(:,j) = mean(reshape1r,2);
        LDrTwoAve(:,j) = mean(reshape2r,2);
    end

    for j = 1:mN
        reshape1n = reshape(NoiseOneLD{i}(:,j),time_length,numTri1(i));
        reshape2n = reshape(NoiseTwoLD{i}(:,j),time_length,numTri2(i));
        LDnOneAve(:,j) = mean(reshape1n,2);
        LDnTwoAve(:,j) = mean(reshape2n,2);
    end
    EventOneLDAve{i} = LDrOneAve;
    EventTwoLDAve{i} = LDrTwoAve;
    NoiseOneLDAve{i} = LDnOneAve;
    NoiseTwoLDAve{i} = LDnTwoAve;
end

% %% Average PCA (only for stack time!)
EventOneLD_NotAve = EventOneLD;
EventTwoLD_NotAve = EventTwoLD;
EventOneLD = EventOneLDAve;
EventTwoLD = EventTwoLDAve;

NoiseOneLD_NotAve = NoiseOneLD;
NoiseTwoLD_NotAve = NoiseTwoLD;
NoiseOneLD = NoiseOneLDAve;
NoiseTwoLD = NoiseTwoLDAve;

    otherwise
        return;
end

%% Calculate PCA using different modes
% for i = 1:length(ind)
%     %% EventOne and NoiseOne
%     % PCA (keep the mean of the original dataset as the first PC)
%     [coeff1R,score1R,explained1R,m1R] = PCA(EventOne{i}',1,0.9);% 1:demean+pc0; 2:keep mean
%     [coeff1N,score1N,explained1N,m1N] = PCA(NoiseOne{i}',1,0.9);% 1:demean+pc0; 2:keep mean
%     
%     % MATLAB PCA
%     %[coeff1R,score1R,latent1R,tsquared1R,explained1R,mu1R] = pca(EventOne{i}');% MATLAB
%     %[coeff1N,score1N,latent1N,tsquared1N,explained1N,mu1N] = pca(NoiseOne{i}');% MATLAB     
%     %wcoeff1R = score1R.* sqrt(explained1R)';% weighted real PC
%     %wcoeff1N = coeff1N.* sqrt(explained1N)';% weighted noise PC
%     
%     EventOneLD{i} = score1R;
%     %wEventOnePC{i} = wcoeff1R;
%     %EventOnePC_explain{1,i} = explained1R;% sum is equal to 1
%     
%     NoiseOneLD{std}{i} = score1N;
%     %wNoiseOnePC{i} = wcoeff1N;
%     %NoiseOnePC_explain{1,i} = explained1N;
%     
%     %xx_group1 = length(explained1R(:,1));
%     %xx_group1 = linspace(1,xx_group1,xx_group1);
%     
%     %% EventTwo and NoiseTwo
%     % PCA (keep the mean of the original dataset as the first PC)
%     [coeff2R,score2R,explained2R,m2R] = PCA(EventTwo{i}',1,0.9);% 1:demean+pc0; 2:keep mean
%     [coeff2N,score2N,explained2N,m2N] = PCA(NoiseTwo{i}',1,0.9);% 1:demean+pc0; 2:keep mean
%      
%     % MATLAB PCA
%     %[coeff2R,score2R,latent2R,tsquared2R,explained2R,mu2R] = pca(EventTwo{i}');% MATLAB
%     %[coeff2N,score2N,latent2N,tsquared2N,explained2N,mu2N] = pca(NoiseTwo{i}');% MATLAB
%     %wcoeff2R = coeff2R.* sqrt(explained2R)';% weighted real PC
%     %wcoeff2N = coeff2N.* sqrt(explained2N)';% weighted noise PC
%     
%     EventTwoLD{i} = score2R;
%     %wEventTwoPC{i} = wcoeff2R;
%     %EventTwoPC_explain{1,i} = explained2R;
%     
%     NoiseTwoLD{std}{i} = score2N;
%     %wNoiseTwoPC{i} = wcoeff2N;
%     %NoiseTwoPC_explain{1,i} = explained2N;
%     
%     %xx_group2 = length(explained2R(:,1));
%     %xx_group2 = linspace(1,xx_group2,xx_group2);
% end
% End calculate PCA using MATLAB pca function - type-wise comparison

%% Calculate CCA
%% CCA in same mode

% numRowsLD1 = cellfun(@(x) size(x,1), EventOneLD);
% numRowsLD2 = cellfun(@(x) size(x,1), EventTwoLD);
% 
% EventOneLDMatrix = vertcat(EventOneLD{:});% MUST vertcat!
% EventTwoLDMatrix = vertcat(EventTwoLD{:}); % MUST vertcat!
% NoiseOneLDMatrix = vertcat(NoiseOneLD{std}{:});% MUST vertcat!
% NoiseTwoLDMatrix = vertcat(NoiseTwoLD{std}{:}); % MUST vertcat!
% 
% [rA,rB,rU,rV] = CCA(EventOneLDMatrix,EventTwoLDMatrix,numPC);
% [nA,nB,nU,nV] = CCA(NoiseOneLDMatrix,NoiseTwoLDMatrix,numPC);
% 
% rU_com = mat2cell(rU, numRowsLD1, numPC);
% rV_com = mat2cell(rV, numRowsLD2, numPC);
% nU_com = mat2cell(nU, numRowsLD1, numPC);
% nV_com = mat2cell(nV, numRowsLD2, numPC);
% 
% EventOneCV = rU_com';
% EventTwoCV = rV_com';
% NoiseOneCV{std} = nU_com';
% NoiseTwoCV{std} = nV_com';

%% CCA in 'different' mode
EventOneCV = cell(1,length(ind));
EventTwoCV = cell(1,length(ind));
NoiseOneCV = cell(1,length(ind));
NoiseTwoCV = cell(1,length(ind));

for i = 1:length(ind)
%% MATLAB CCA - m
%[rA{i},rB{i},rR,rU,rV,rStats] = canoncorr(EventOneLD{i}(:,1:mR), EventTwoLD{i}(:,1:numPC));
%[nA{i},nB{i},nR,nU,nV,nStats] = canoncorr(NoiseOneLD{i}(:,1:mN), NoiseTwoLD{i}(:,1:numPC));

%% Take 3 PC
% [rA,rB,rU,rV] = CCA(EventOneLD{i},EventTwoLD{i},numPC);
% [nA,nB,nU,nV] = CCA(NoiseOneLD{i},NoiseTwoLD{i},numPC);

%% Take m PC
% if m >= numPC
 numPC = m;
[rA,rB,rU,rV] = CCA(EventOneLD{i},EventTwoLD{i},m);
[nA,nB,nU,nV] = CCA(NoiseOneLD{i},NoiseTwoLD{i},m);
% else
% [rA,rB,rU,rV] = CCA(EventOneLD{i},EventTwoLD{i},numPC);
% [nA,nB,nU,nV] = CCA(NoiseOneLD{i},NoiseTwoLD{i},numPC);
% end

EventOneCV{i} = rU;
EventTwoCV{i} = rV;

NoiseOneCV{i} = nU;
NoiseTwoCV{i} = nV;
end

% End Calculate CCA using MATLAB pca function


%% Calculate correlation coefficient (Perason r)

% Bin the top 4 LD for position, phase, and trial type
EventOneCVbin = cell(1,numPC);
EventTwoCVbin = cell(1,numPC);
NoiseOneCVbin = cell(1,numPC);
NoiseTwoCVbin = cell(1,numPC);

for i = 1:numPC
 for j = 1:length(ind)
    %EventOneLDbin{i}(:,j) = EventOneLD{j}(:,i);
    %EventTwoLDbin{i}(:,j) = EventTwoLD{j}(:,i);
    EventOneCVbin{i}(:,j) = EventOneCV{j}(:,i);
    EventTwoCVbin{i}(:,j) = EventTwoCV{j}(:,i);
    
    %NoiseOneLDbin{i}(:,j) = NoiseOneLD{j}(:,i);
    %NoiseTwoLDbin{i}(:,j) = NoiseTwoLD{j}(:,i);
    NoiseOneCVbin{i}(:,j) = NoiseOneCV{j}(:,i);
    NoiseTwoCVbin{i}(:,j) = NoiseTwoCV{j}(:,i);
 end
end

% Calculate unweighted correlation coefficients for top 3 CV
r_CVreal = zeros(1,numPC);
r_CVnoise = zeros(1,numPC);

for i = 1:numPC % top 3 neural modes
    
% calculate correlation coefficient for LD
% R_LDreal = corrcoef(EventOneLDbin{i},EventTwoLDbin{i});
% r_LDreal{std}(i) = R_LDreal(1,2);
% r_LDreal(i) = R_LDreal(1,2);

% calculate correlation coefficient for CV
R_CVreal = corrcoef(EventOneCVbin{i},EventTwoCVbin{i});
%r_CVreal{std}(i) = R_CVreal(1,2);
r_CVreal(i) = R_CVreal(1,2);

% calculate correlation coefficient for noise LD
% R_LDnoise = corrcoef(NoiseOneLDbin{i},NoiseTwoLDbin{i});
% r_LDnoise{std}(i) = R_LDnoise(1,2);
% r_LDnoise(i) = R_LDnoise(1,2);

% calculate correlation coefficient for noise CV
R_CVnoise = corrcoef(NoiseOneCVbin{i},NoiseTwoCVbin{i});
%r_CVnoise{std}(i) = R_CVnoise(1,2);
r_CVnoise(i) = R_CVnoise(1,2);
end

% Calculate unweighted correlation coefficient combining top 3 neural modes

% reshape the binned PC matrix and CC matrix
%EventOneLDcom = cat(1,EventOneLDbin{:}); 
%EventTwoLDcom = cat(1,EventTwoLDbin{:});
EventOneCVcom = cat(1,EventOneCVbin{:}); 
EventTwoCVcom = cat(1,EventTwoCVbin{:});

%NoiseOneLDcom = cat(1,NoiseOneLDbin{:}); 
%NoiseTwoLDcom = cat(1,NoiseTwoLDbin{:});
NoiseOneCVcom = cat(1,NoiseOneCVbin{:}); 
NoiseTwoCVcom = cat(1,NoiseTwoCVbin{:});

% Calculate combined correlation coefficient for LD
% R_LDreal = corrcoef(EventOneLDcom,EventTwoLDcom);
% r_LDreal = [r_LDreal,R_LDreal(1,2)];

% Calculate combined correlation coefficient for CV
%R_CVreal = corrcoef(EventOneCVcom,EventTwoCVcom);
%r_CVreal = [r_CVreal{std},R_CVreal(1,2)];

% Calculate combined correlation coefficient for noise LD
%R_LDnoise = corrcoef(NoiseOneLDcom,NoiseTwoLDcom);
%r_LDnoise = [r_LDnoise,R_LDnoise(1,2)];

% Calculate combined correlation coefficient for noise CV
%R_CVnoise = corrcoef(NoiseOneCVcom,NoiseTwoCVcom);
%r_CVnoise = [r_CVnoise{std},R_CVnoise(1,2)];

%r_real(std) = r_CVreal(4);
%r_noise(std) = r_CVnoise(4);

%% Calculate coefficient for weighted CC 

% Sqrt weight
% weightCVreal = sqrt(abs(r_CVreal(1:numPC))/sum(abs(r_CVreal(1:numPC))));% sqrt weight
% weightCVnoise = sqrt(abs(r_CVnoise(1:numPC))/sum(abs(r_CVnoise(1:numPC))));% sqrt weight

% Square weight
weightCVreal = (r_CVreal(1:numPC).^2)/sum(r_CVreal(1:numPC).^2);% sqare weight
weightCVnoise = (r_CVnoise(1:numPC).^2)/sum(r_CVnoise(1:numPC).^2);% sqare weight

%% % Weighted sum CV
r_wCVreal = dot(r_CVreal,weightCVreal);
r_wCVnoise = dot(r_CVnoise,weightCVnoise);

%% Combined weighted CV

% for i = 1:numPC
% wEventOneCVbin{i} = weightCVreal(i).*EventOneCVbin{i};
% wEventTwoCVbin{i} = weightCVreal(i).*EventTwoCVbin{i};
% 
% wNoiseOneCVbin{i} = weightCVnoise(i).*NoiseOneCVbin{i};
% wNoiseTwoCVbin{i} = weightCVnoise(i).*NoiseTwoCVbin{i};
% 
% end
% 
% wEventOneCVcom = cat(1,wEventOneCVbin{:});
% wEventTwoCVcom = cat(1,wEventTwoCVbin{:});
% 
% wNoiseOneCVcom = cat(1,wNoiseOneCVbin{:});
% wNoiseTwoCVcom = cat(1,wNoiseTwoCVbin{:});

% calculated combined weighted CV
% R_wCVreal = corrcoef(wEventOneCVcom,wEventTwoCVcom);
% r_wCVreal = R_wCVreal(1,2);
% R_wCVnoise = corrcoef(wNoiseOneCVcom,wNoiseTwoCVcom);
% r_wCVnoise = R_wCVnoise(1,2);

r_real(std) = r_wCVreal;
r_noise(std) = r_wCVnoise;

%Data.r_LD = r_LD;%[r_1stLD, r_2ndLD, r_3rdLD, r_comLD] 
%Data.r_CV = r_CV;%[r_1stCV, r_2ndCV, r_3rdCV, r_comCV] 

% End calculate correlation coefficient
 end %std end

%across trial
R_real(trial,:) = r_real;
R_noise(trial,:) = r_noise;

end % for trial end

%% Calculate correlation difference
meanRreal = mean(R_real,1);
meanRnoise = mean(R_noise,1);

%diff = r_real - r_noise;
diff = meanRreal - meanRnoise;
[indx,indy] = max(diff);%indy is the index, cos_similarity
% [indx,indy] = max(abs(diff));%indy is the index, cos_similarity
opt_std = sigma_vec(indy);% optimized std

%% Save optimal std
SessionData.OptSigma = opt_std;

%% Figure: std difference
fig1 = figure(1);
fig1_name = sprintf('%s vs. %s correlation difference',newStr1,newStr2);
sgtitle(fig1_name)
subplot(2,1,1)
semilogx(sigma_vec, meanRreal, 'b');%,'linewidth',1.5);  % real r
%semilogx(sigma_vec, r_real, 'b','linewidth',1.5);  % real r
hold on
semilogx(sigma_vec, meanRnoise, 'r');%,'linewidth',1.5);  % noise r
%semilogx(sigma_vec, r_noise, 'r','linewidth',1.5);  % noise r
hold on
semilogx(sigma_vec, diff, 'k');%,'linewidth',1.5);  % difference
hold off
my_title = sprintf('arg max CorrDiff(sigma)=%s bins(binSize=%dms)',num2str(opt_std),bin_size_ms);
title(my_title);
% xlabel('Standard Deviation(ms)');ylabel('Correlation Coefficient');xlim([0,200]);
% legend('Spike Trains','Noise','Correlation Difference');

subplot(2,1,2)
%sz = 5;
plot(sigma_vec, meanRreal, 'b');%,'linewidth',1.5);  % real r
%plot(sigma_vec, r_real, 'b','linewidth',1.5);  % real r
%scatter(sigma_vec, r_real,sz,'b','filled');
hold on
plot(sigma_vec, meanRnoise, 'r');%,'linewidth',1.5);  % real r
%plot(sigma_vec, r_noise, 'r','linewidth',1.5);  % real r
%scatter(sigma_vec, r_noise,sz,'r','filled');  % noise r
hold on
plot(sigma_vec, diff, 'k');%,'linewidth',1.5);  % real r
%scatter(sigma_vec, diff,sz,'k','filled'); % difference
hold off
my_title = sprintf('arg max CorrDiff(sigma)=%s bins(binSize=%dms)',num2str(opt_std),bin_size_ms);
title(my_title);
% xlabel('Standard Deviation(ms)');ylabel('Correlation Coefficient');xlim([0,200]);
% legend('Spike Trains','Noise','Correlation Difference');

fig2 = figure(2);
switch Arrange
    case 'neuron'
         fig2_name = sprintf('%s vs. %s correlation difference(log)stackneuron-bin%dms %dtrials',...
    newStr1,newStr2,bin_size_ms,numTrial);
         mat_name = sprintf('%s vs %s_%sSmoothOptStd_%dtrials_stackneuron',newStr1,newStr2,binStr,numTrial);

    case 'time'
        fig2_name = sprintf('%s vs. %s correlation difference(log)stacktime-bin%dms %dtrials',...
    newStr1,newStr2,bin_size_ms,numTrial);
        mat_name = sprintf('%s vs %s_%sSmoothOptStd_%dtrials_stacktime',newStr1,newStr2,binStr,numTrial);

    otherwise
        return;
end

%sgtitle(fig1_name)
semilogx(sigma_vec, meanRreal, 'b');%,'linewidth',1.5);  % real r
%semilogx(sigma_vec, r_real, 'b','linewidth',1.5);  % real r
hold on
semilogx(sigma_vec, meanRnoise, 'r');%,'linewidth',1.5);  % noise r
%semilogx(sigma_vec, r_noise, 'r','linewidth',1.5);  % noise r
hold on
semilogx(sigma_vec, diff, 'k');%,'linewidth',1.5);  % difference
hold off
%my_title = sprintf('arg max CorrDiff(sigma)=%s ms(log plot)',num2str(opt_std));
%title(my_title);
%xlabel('Number of Bins(std=#*bin size)');ylabel('Correlation Coefficient');xlim([0,110]);
%legend('Spike Trains','Noise','Correlation Coefficient Difference');

%% Save data
SessionData.group1 = Data1.group_RawReal;
SessionData.group2 = Data2.group_RawReal;
SessionData.numNeuron1 = numNeuron1;
SessionData.numNeuron2 = numNeuron2;
% SessionData.NoiseOneLD = NoiseOneLD{opt_std};
% SessionData.NoiseTwoLD = NoiseTwoLD{opt_std};
% SessionData.NoiseOneCV = NoiseOneCV{opt_std};
% SessionData.NoiseTwoCV = NoiseTwoCV{opt_std};

% mat_name = sprintf('%s vs %s_%sSmoothOptStd_%dtrials',newStr1,newStr2,binStr,numTrial);
save(mat_name,'SessionData','-v7.3');
disp('data saved')

elapsed_time = toc;
fprintf('Elapsed time: %.2f seconds\n', elapsed_time);

%% Save figure
FigureFolder = 'Results'; % User need to specific the folder that containing .nex files
FigurePath = fullfile(path1, FigureFolder);

fig_name = sprintf('%s.png',fig2_name);% figure name
fig_path = fullfile(FigurePath, fig_name);% still figure name
saveas(fig2, fig_path);

disp('figures saved')

