clear; clc; close all;
% Open file dialog box to select a .mat file
[file,path] = uigetfile('*.mat','Select the binned .mat file');
if isequal(file,0)
    disp('User selected Cancel');
    return
end
    

%p1 = fullfile('toolbox');
%addpath(genpath(p1));
%DataFolder = '../Datasets';

%sigma = input('Enter standard deviation(ms): '); % *50ms = standard deviation
%threshold = input('Enter PCA threshold(0 to 1): ');
threshold = 0.9;

Com = input('Enter comparison type(choose between "type" or "trial"):','s');
dlgtitle = 'Input(CAPITAL)';
dims = [2 48];% [wide length] 

switch Com
    case 'type'
    disp('Types include: LEFT,RIGHT,SAMPLE,NONMATCH,CORRECT,ERROR');
    prompt = {'Enter Type1(i.e.,LEFT):',...
        'Enter Type2(i.e.,RIGHT):'};
    input_info = inputdlg(prompt,dlgtitle,dims);

    case 'trial'
    disp('Trials include: LEFT SAMPLE CORRECT,RIGHT SAMPLE CORRECT,etc');
    prompt = {'Enter Trial1,position(LEFT or RIGHT):',...
        'Enter Trial1,phase(SAMPLE or NONMATCH):',...
        'Enter Trial1,outcome(CORRECT or ERROR):',...
        'Enter Trial2,position(LEFT or RIGHT):',...
        'Enter Trial2,phase(SAMPLE or NONMATCH):',...
        'Enter Trial2,outcome(CORRECT or ERROR):'};
    input_info = inputdlg(prompt,dlgtitle,dims);
     
    otherwise
         disp('User input error');
         return;
end
   
tic
% Load the selected .mat file
load(fullfile(path,file));  


%AnimalID = input('Enter animal ID: ', 's');%i.e 1029,1036,etc
%SessionID = input('Enter session ID: ', 's');%i.e 195,etc

groups = Data.groups;
Data.input_info = input_info;
Data.Com = Com;

%% Get bin size information
str = file;
parts = split(str, '_'); 
binStr = parts{2,1};
binStr = split(binStr,'.'); 
binStr = char(binStr{1,1});
Data.binStr = binStr;

%% Grouping step 2
switch Com
    case 'type'% case1: type-wise comparison
    for i = 1:size(groups,2)
    i_one = find(strcmp(groups(:,i), input_info{1}));
    i_two = find(strcmp(groups(:,i), input_info{2}));
    
    if ~isempty(i_one)
        ind_one = i_one;
        ind_two = i_two;
        break
    else
      continue
    end
    end

    case 'trial'% case2: trial wise comparison 
    ind_one = find(strcmp(groups(:,1),input_info{1}) & ...
    strcmp(groups(:,2),input_info{2}) & strcmp(groups(:,3),input_info{3}));

    ind_two = find(strcmp(groups(:,1),input_info{4}) & ...
    strcmp(groups(:,2),input_info{5}) & strcmp(groups(:,3),input_info{6}));
end


% Define time window for event
bin_size = Data.bin_size;
bin_size_ms = bin_size .* 1000;% ms,only for naming
extra_time = Data.ExtraMarginTime;
t_peri = Data.PeriTime;
t_start = extra_time/bin_size;
t_end = t_start+(t_peri/bin_size);

%BinSmoothString = strcat('Bin', num2str(bin_size_ms),'ms','Smooth',num2str(sigma),'ms');
%bin_size_ms = 1; %1ms,only for BinSmoothString


%% Smooth
% Define std vector 
sigma_vec1 = linspace(1,50,50);%50 points from 1ms to 50ms
sigma_vec2 = logspace(log10(50),log10(300),30);% 30 points from 50ms to 400ms with log scale

% Combine x1 and x2.
sigma_vec = [sigma_vec1 sigma_vec2(2:end)];


for std = 1:length(sigma_vec)
   
    window = 5*sigma_vec(std); % window size
    x = -window:window; % domain of the kernel
    kernel = exp(-x.^2/(2*sigma_vec(std)^2)) / (sigma_vec(std)*sqrt(2*pi)); % Gaussian kernel

    % Smooth spike firing counts with Gaussian kernel
    for i = 1:length(Data.group_RawReal)
        
         Realspike_counts = Data.group_RawReal{i};%raw Real data
         Noisespike_counts = Data.group_RawNoise{i};%raw Noise data
         
         % Preallocate an array for the smoothed counts
         Realsmooth_counts = zeros(size(Realspike_counts));  
         Noisesmooth_counts = zeros(size(Noisespike_counts));
         
           for j = 1:size(Realspike_counts,1)  % Loop over each row
               Realsmooth_counts(j,:) = conv(Realspike_counts(j,:), kernel, 'same');
               Noisesmooth_counts(j,:) = conv(Noisespike_counts(j,:), kernel, 'same');
           end

         group_SmoothedReal{i} = Realsmooth_counts;
         group_SmoothedNoise{i} = Noisesmooth_counts;
    end

%% Extract data 

for j = 1:length(ind_one)
    k = ind_one(j);
    EventOne{j} = group_SmoothedReal{k};
    NoiseOne{j} = group_SmoothedNoise{k};
end
    
for j = 1:length(ind_two)
    k = ind_two(j);
    EventTwo{j} = group_SmoothedReal{k};
    NoiseTwo{j} = group_SmoothedNoise{k};
end
    
%% Calculate PCA 

for i = 1:length(ind_one)
   %% EventOne and NoiseOne
    % PCA (keep the mean of the original dataset as the first PC)
     [coeff1R,score1R,explained1R,m1R] = PCA(EventOne{i}',2,0.9);% 1:demean+pc0; 2:keep mean
     [coeff1N,score1N,explained1N,m1N] = PCA(NoiseOne{i}',2,0.9);% 1:demean+pc0; 2:keep mean
    
     % MATLAB PCA
%     [coeff1R,score1R,latent1R,tsquared1R,explained1R,mu1R] = pca(EventOne{i}');% MATLAB
%     [coeff1N,score1N,latent1N,tsquared1N,explained1N,mu1N] = pca(NoiseOne{i}');% MATLAB     
%     %wcoeff1R = coeff1R.* sqrt(explained1R)';% weighted real PC
    %wcoeff1N = coeff1N.* sqrt(explained1N)';% weighted noise PC
    
    EventOneLD{i} = score1R;
    %wEventOnePC{i} = wcoeff1R;
    %EventOnePC_explain{1,i} = explained1R;% sum is equal to 1
    
    NoiseOneLD{i} = score1N;
    %wNoiseOnePC{i} = wcoeff1N;
    %NoiseOnePC_explain{1,i} = explained1N;
    
    %xx_group1 = length(explained1R(:,1));
    %xx_group1 = linspace(1,xx_group1,xx_group1);
end

for i = 1:length(ind_two)
    %% EventTwo and NoiseTwo
    % PCA (keep the mean of the original dataset as the first PC)
    [coeff2R,score2R,explained2R,m2R] = PCA(EventTwo{i}',2,0.9);% 1:demean+pc0; 2:keep mean
    [coeff2N,score2N,explained2N,m2N] = PCA(NoiseTwo{i}',2,0.9);% 1:demean+pc0; 2:keep mean
     
    % MATLAB PCA
%     [coeff2R,score2R,latent2R,tsquared2R,explained2R,mu2R] = pca(EventTwo{i}');% MATLAB
%     [coeff2N,score2N,latent2N,tsquared2N,explained2N,mu2N] = pca(NoiseTwo{i}');% MATLAB
%     %wcoeff2R = coeff2R.* sqrt(explained2R)';% weighted real PC
    %wcoeff2N = coeff2N.* sqrt(explained2N)';% weighted noise PC
    
    EventTwoLD{i} = score2R;
    %wEventTwoPC{i} = wcoeff2R;
    %EventTwoPC_explain{1,i} = explained2R;
    
    NoiseTwoLD{i} = score2N;
    %wNoiseTwoPC{i} = wcoeff2N;
    %NoiseTwoPC_explain{1,i} = explained2N;
    
    %xx_group2 = length(explained2R(:,1));
    %xx_group2 = linspace(1,xx_group2,xx_group2);
end
% End calculate PCA


%% Calculate CCA
%numPC = min(n_PC0_group1(sigma),n_PC0_group2(sigma));% Define the number of PC for CCA comparison
%numPC = min(numPC,3);
numPC = 3;
for i = 1:length(ind_one)
% MATLAB CCA
%[rA{i},rB{i},rR,rU,rV,rStats] = canoncorr(EventOneLD{i}(:,1:numPC), EventTwoLD{i}(:,1:numPC));
%[nA{i},nB{i},nR,nU,nV,nStats] = canoncorr(NoiseOneLD{i}(:,1:numPC), NoiseTwoLD{i}(:,1:numPC));

% CCA
[rA{i},rB{i},rU,rV] = CCA(EventOneLD{i},EventTwoLD{i},numPC);
[nA{i},nB{i},nU,nV] = CCA(NoiseOneLD{i},NoiseTwoLD{i},numPC);

EventOneCV{i} = rU;
EventTwoCV{i} = rV;

NoiseOneCV{i} = nU;
NoiseTwoCV{i} = nV;
end

%% Calculate correlation coefficient (Perason r)

% Bin the top 3 PC for position, phase, and trial type
for i = 1:numPC
 for j = 1:length(ind_one)
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
for i = 1:numPC % top 3 neural modes
    
% calculate correlation coefficient for PC
%R_LDreal = corrcoef(EventOneLDbin{i},EventTwoLDbin{i});
%r_LDreal{std}(i) = R_LDreal(1,2);
%r_LDreal(i) = R_LDreal(1,2);

% calculate correlation coefficient for CC
R_CVreal = corrcoef(EventOneCVbin{i},EventTwoCVbin{i});
%r_CVreal{std}(i) = R_CVreal(1,2);
r_CVreal(i) = R_CVreal(1,2);

% calculate correlation coefficient for noise LD
%R_LDnoise = corrcoef(NoiseOneLDbin{i},NoiseTwoLDbin{i});
%r_LDnoise{std}(i) = R_LDnoise(1,2);
%r_LDnoise(i) = R_LDnoise(1,2);

% calculate correlation coefficient for noise CV
R_CVnoise = corrcoef(NoiseOneCVbin{i},NoiseTwoCVbin{i});
%r_CVnoise{std}(i) = R_CVnoise(1,2);
r_CVnoise(i) = R_CVnoise(1,2);
end

% Calculate unweighted correlation coefficient combining top 3 neural modes

% reshape the binned LD matrix and CV matrix
%EventOneLDcom = cat(1,EventOneLDbin{:}); 
%EventTwoLDcom = cat(1,EventTwoLDbin{:});
EventOneCVcom = cat(1,EventOneCVbin{:}); 
EventTwoCVcom = cat(1,EventTwoCVbin{:});

%NoiseOneLDcom = cat(1,NoiseOneLDbin{:}); 
%NoiseTwoLDcom = cat(1,NoiseTwoLDbin{:});
NoiseOneCVcom = cat(1,NoiseOneCVbin{:}); 
NoiseTwoCVcom = cat(1,NoiseTwoCVbin{:});

% Calculate combined correlation coefficient for unweighted LD
%R_LDreal = corrcoef(EventOneLDcom,EventTwoLDcom);
%r_LDreal = [r_LDreal,R_LDreal(1,2)];

% Calculate combined unweighted correlation coefficient for CV
%R_CVreal = corrcoef(EventOneCVcom,EventTwoCVcom);
%r_CVreal = [r_CVreal{std},R_CVreal(1,2)];

% Calculate combined correlation coefficient for unweighted noise LD
%R_LDnoise = corrcoef(NoiseOneLDcom,NoiseTwoLDcom);
%r_LDnoise = [r_LDnoise,R_LDnoise(1,2)];

% Calculate combined unweighted correlation coefficient for noise CV
%R_CVnoise = corrcoef(NoiseOneCVcom,NoiseTwoCVcom);
%r_CVnoise = [r_CVnoise{std},R_CVnoise(1,2)];

%r_real(std) = r_CVreal(4);
%r_noise(std) = r_CVnoise(4);

%% Calculate coefficient for weighted CC 

% Sqrt weight
weightCVreal = sqrt(abs(r_CVreal(1:numPC))/sum(abs(r_CVreal(1:numPC))));% sqrt weight
weightCVnoise = sqrt(abs(r_CVnoise(1:numPC))/sum(abs(r_CVnoise(1:numPC))));% sqrt weight

% Square weight
% weightCVreal = (r_CVreal(1:numPC).^2)/sum(r_CVreal(1:numPC).^2);% sqare weight
% weightCVnoise = (r_CVnoise(1:numPC).^2)/sum(r_CVnoise(1:numPC).^2);% sqare weight

for i = 1:numPC
wEventOneCVbin{i} = weightCVreal(i).*EventOneCVbin{i};
wEventTwoCVbin{i} = weightCVreal(i).*EventTwoCVbin{i};

wNoiseOneCVbin{i} = weightCVnoise(i).*NoiseOneCVbin{i};
wNoiseTwoCVbin{i} = weightCVnoise(i).*NoiseTwoCVbin{i};

end

wEventOneCVcom = cat(1,wEventOneCVbin{:});
wEventTwoCVcom = cat(1,wEventTwoCVbin{:});

wNoiseOneCVcom = cat(1,wNoiseOneCVbin{:});
wNoiseTwoCVcom = cat(1,wNoiseTwoCVbin{:});

% calculated combined weighted CV
R_wCVreal = corrcoef(wEventOneCVcom,wEventTwoCVcom);
r_wCVreal = R_wCVreal(1,2);

R_wCVnoise = corrcoef(wNoiseOneCVcom,wNoiseTwoCVcom);
r_wCVnoise = R_wCVnoise(1,2);

r_real(std) = r_wCVreal;
r_noise(std) = r_wCVnoise;

clear r_wCVreal r_wCVnoise

%Data.r_LD = r_LD;%[r_1stLD, r_2ndLD, r_3rdLD, r_comLD] 
%Data.r_CV = r_CV;%[r_1stCV, r_2ndCV, r_3rdCV, r_comCV] 

% End calculate correlation coefficient
end %std end

%% Calculate correlation difference
diff = r_real - r_noise;
[indx,indy] = max(abs(diff));%indy is the index, cos_similarity
opt_std = sigma_vec(indy);% optimized std

Data.OptSigma = opt_std;
% End calculate correlation difference

%% Extract the same of two events
if strcmp(Com, 'type')
   Data.EventOneName = input_info{1};
   Data.EventTwoName = input_info{2};

else
   Data.EventOneName = strcat(input_info{1},input_info{2},input_info{3});
   Data.EventTwoName = strcat(input_info{4},input_info{5},input_info{6});
end
% End extract the same of two events

%% Figure: std difference
% fig1 = figure(1);
% fig1_name = sprintf('%s vs. %s correlation difference',newStr1,newStr2);
% %sgtitle(fig1_name)
% %subplot(2,1,1)
% semilogx(sigma_vec, r_real, 'b','linewidth',1.5);  % real r
% hold on
% semilogx(sigma_vec, r_noise, 'r','linewidth',1.5);  % noise r
% hold on
% semilogx(sigma_vec, diff, 'k','linewidth',1.5);  % difference
% hold off
% %my_title = sprintf('arg max CorrDiff(sigma)=%s ms(log plot)',num2str(opt_std));
% %title(my_title);
% xlabel('Standard Deviation(ms)');ylabel('Correlation Coefficient');xlim([0,200]);
% legend('Spike Trains','Noise','Correlation Difference');
% 
% %subplot(2,1,2)
% %sz = 5;
% %plot(sigma_vec, r_real, 'b','linewidth',1.5);  % real r
% %scatter(sigma_vec, r_real,sz,'b','filled');
% %hold on
% %plot(sigma_vec, r_noise, 'r','linewidth',1.5);  % real r
% %scatter(sigma_vec, r_noise,sz,'r','filled');  % noise r
% %hold on
% %plot(sigma_vec, diff, 'k','linewidth',1.5);  % real r
% %scatter(sigma_vec, diff,sz,'k','filled'); % difference
% %hold off
% %my_title = sprintf('arg max CorrDiff(sigma)=%s ms',num2str(opt_std));
% %title(my_title);
% %xlabel('Standard deviation(ms)');ylabel('Value');xlim([0,200]);
% %legend('real correlation','noise correlation','difference');
% 
fig2 = figure(2);
fig2_name = sprintf('%s vs. %s correlation difference(logscale)',newStr1,newStr2);
%sgtitle(fig3_name)

semilogx(sigma_vec, r_real, 'b')%;,'linewidth',1.5);  % real r
hold on
semilogx(sigma_vec, r_noise, 'r')%;,'linewidth',1.5);  % noise r
hold on
semilogx(sigma_vec, diff, 'k')%;,'linewidth',1.5);  % difference
hold off
%my_title = sprintf('arg max CorrDiff(sigma)=%s ms(log plot)',num2str(opt_std));
%title(my_title);
xlabel('Standard Deviation(ms)');ylabel('Correlation Coefficient');xlim([0,120]);
legend('Spike Trains','Noise','Correlation Coefficient Difference');


%% Save figure
FigureFolder = 'Results'; % User need to specific the folder that containing .nex files
FigurePath = fullfile(path, FigureFolder);

fig_name = sprintf('%s.png',fig2_name);% figure name
fig_path = fullfile(FigurePath, fig_name);% still figure name
saveas(fig2, fig_path);

disp('figures saved')

%% Save data
str = file;
parts = split(str, '.');  % split the string at the period
newStr = parts(1); 
newStr = char(newStr);
mat_name = sprintf('%sSmoothOptStd - %s vs %s',newStr,Data.EventOneName,Data.EventTwoName);
save(mat_name,'Data','-v7.3');
disp('data saved')

elapsed_time = toc;
fprintf('Elapsed time: %.2f seconds\n', elapsed_time);