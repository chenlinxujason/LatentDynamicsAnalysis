clear; clc; close all;
% Open file dialog box to select a .mat file
[file,path] = uigetfile('*.mat','Select the .mat file with seesion compare Opt std');
if isequal(file,0)
    disp('User selected Cancel');
else
 tic    
    load(fullfile(path,file));% Load the selected .mat file  
end

Arrange = SessionData.Arrange;
disp(['Data arrangement type: stack ' Arrange]);

%% Extract comparsion type (in most cases, 'all', which means 8 events)
Com = SessionData.Com;
input_info = SessionData.input_info;

%% Extract the names of 8 events and time
groups = SessionData.groups;
extra_time = SessionData.extra_time;
t_peri = SessionData.t_peri;
t_start = SessionData.t_start;
t_end = SessionData.t_end;


%% Extract the name of two sessions and/or animals this code comparing
session1Name = SessionData.session1;
session2Name = SessionData.session2;
binStr = SessionData.binStr;
disp_str = sprintf('Compare %s and %s',session1Name,session2Name);
disp(disp_str);

%% Extract raw spike data
group1 = SessionData.group1;
group2 = SessionData.group2;
% NoiseOneLD = SessionData.NoiseOneLD;
% NoiseTwoLD = SessionData.NoiseTwoLD;
% NoiseOneCV = SessionData.NoiseOneCV;
% NoiseTwoCV = SessionData.NoiseTwoCV;
% group1_Noise = SessionData.group1_Noise;
% group2_Noise = SessionData.group2_Noise;

numNeuron1 = SessionData.numNeuron1;
numNeuron2 = SessionData.numNeuron2;

%% Switch case for index
switch Com
    case 'all' % case1 compare 8 events for two sessions and/or animals
    n = min(length(group1),length(group2));% the number of event for comparison
    ind = 1:n;
    % no input_info in this case
    
    case 'type'% case2: type-wise comparison
    for i = 1:size(groups,2)
    g = find(strcmp(groups(:,i), input_info));
    
    if ~isempty(g)
        ind = g;
        break
    else
      continue
    end
    end

    case 'trial'% case3: trial wise comparison 
    ind = find(strcmp(groups(:,1),input_info{1}) & ...
    strcmp(groups(:,2),input_info{2}) & strcmp(groups(:,3),input_info{3}));
end

%% Smooth
% Define Gaussian kernel parameters
sigma = SessionData.OptSigma;% optimized std
% sigma = 20;
window = 5*sigma; % window size
x = -window:window; % domain of the kernel
kernel = exp(-x.^2/(2*sigma^2)) / (sigma*sqrt(2*pi)); % Gaussian kernel


  % Smooth spike firing counts with Gaussian kernel
    for i = 1:n % in most cases, n=8, which corresponding to 8 events
        
         Realspike1_counts = group1{i};%raw Real data1         
         Realspike2_counts = group2{i};%raw Real data2

%          Noisespike1_counts = group1_Noise{i};%raw Noise data1         
%          Noisespike2_counts = group2_Noise{i};%raw Noise data2
%          
         % Preallocate an array for the smoothed counts
         Realsmooth1_counts = zeros(size(Realspike1_counts));           
         Realsmooth2_counts = zeros(size(Realspike2_counts));  
          
%          Noisesmooth1_counts = zeros(size(Noisespike1_counts));           
%          Noisesmooth2_counts = zeros(size(Noisespike2_counts)); 

           for j = 1:size(Realspike1_counts,1)  % Loop over each row
               Realsmooth1_counts(j,:) = conv(Realspike1_counts(j,:), kernel, 'same');
%                Noisesmooth1_counts(j,:) = conv(Noisespike1_counts(j,:), kernel, 'same');
           end
           
           for j = 1:size(Realspike2_counts,1)  % Loop over each row
               Realsmooth2_counts(j,:) = conv(Realspike2_counts(j,:), kernel, 'same');
%                Noisesmooth2_counts(j,:) = conv(Noisespike2_counts(j,:), kernel, 'same');
           end

         group1_Smoothed{i} = Realsmooth1_counts;         
         group2_Smoothed{i} = Realsmooth2_counts;

         
%          noise1_Smoothed{i} = Noisesmooth1_counts;         
%          noise2_Smoothed{i} = Noisesmooth2_counts;        
    end

%% Extract data 
if strcmp (Com, 'all')
       EventOne = group1_Smoothed;       
       EventTwo = group2_Smoothed;

%        NoiseOne = noise1_Smoothed;
%        NoiseTwo = noise2_Smoothed;
else    
    for j = 1:length(ind)
        k = ind(j);
        EventOne{j} = group1_Smoothed{k};
        EventTwo{j} = group2_Smoothed{k};

%        NoiseOne{j} = noise1_Smoothed{k};
%        NoiseTwo{j} = noise2_Smoothed{k};
    end
end    
%% Calculate PCA 

%% PCA - using same neural modes (combined 8 events firing rates) 
switch Arrange
    case 'neuron'
%% vertcat - stack neuron

numRows1 = cellfun(@(x) size(x,1), EventOne);
numRows2 = cellfun(@(x) size(x,1), EventTwo);

numTri1 = numRows1/numNeuron1;% number of trials for animal1
numTri2 = numRows2/numNeuron2;% number of trials for animal2

% repeated trials
% maxRows1 = max(cellfun(@(x) size(x,1), EventOne)); % Get the maximum number of rows
% maxRows2 = max(cellfun(@(x) size(x,1), EventTwo));

% numRows1 = maxRows1*ones(1,length(numRows1ch));
% numRows2 = maxRows2*ones(1,length(numRows2ch));
% 
% for i = 1:length(ind)
%     SizeOne = size(EventOne{i},1);
%     SizeTwo = size(EventTwo{i},1);
% 
%     repeatedOne = floor(maxRows1/SizeOne);
%     remainingOne = maxRows1 - SizeOne*repeatedOne;
%     repeatedTwo = floor(maxRows2/SizeTwo);
%     remainingTwo = maxRows2 - SizeTwo*repeatedTwo;
% 
%     EventOne{i} = [repmat(EventOne{i},repeatedOne,1);...
%         EventOne{i}(1:remainingOne,:)];
%     EventTwo{i} = [repmat(EventTwo{i},repeatedTwo,1);...
%         EventTwo{i}(1:remainingTwo,:)];
% 
% end
% % repeated trials
EventOnePC = cell(1, length(ind)); % initialize cell array
EventTwoPC = cell(1, length(ind)); % initialize cell array
EventOneLD = cell(1, length(ind)); % initialize cell array
EventTwoLD = cell(1, length(ind)); % initialize cell array

EventOneMatrix = vertcat(EventOne{:}); 
EventTwoMatrix = vertcat(EventTwo{:}); 

[coeff1,score1,explained1,m1] = PCA(EventOneMatrix',1,0.6);
[coeff2,score2,explained2,m2] = PCA(EventTwoMatrix',1,0.6);
m = min(m1,m2);
numPC = 4;  % take the first 4 PC
% m = 4;

startIdx1 = [1 cumsum(numRows1(1:end-1))+1]; % start indices of each block
endIdx1 = cumsum(numRows1); % end indices of each block

startIdx2 = [1 cumsum(numRows2(1:end-1))+1]; % start indices of each block
endIdx2 = cumsum(numRows2); % end indices of each block

for i = 1:length(ind)% length(ind) = length(numRows1) = 8
    EventOnePC{i} = coeff1(startIdx1(i):endIdx1(i), 1:m);
    EventOneLD{i} = EventOne{i}' * EventOnePC{i}; 
    EventTwoPC{i} = coeff2(startIdx2(i):endIdx2(i), 1:m);
    EventTwoLD{i} = EventTwo{i}' * EventTwoPC{i};
end

% scale trial
for i = 1:length(ind)
     EventOneLD{i} = EventOneLD{i}/numTri1(i);
     EventTwoLD{i} = EventTwoLD{i}/numTri2(i);
end

    case 'time'
%% horzcat - stack time
EventOneMatrix = horzcat(EventOne{:}); 
EventTwoMatrix = horzcat(EventTwo{:});

[coeff1,score1,explained1,m1] = PCA(EventOneMatrix',3,0.6);
[coeff2,score2,explained2,m2] = PCA(EventTwoMatrix',3,0.6);
m = min(m1,m2);  % take m-fold neural modes for comparison
numPC = min(m,4);  % averaged over top 4 neural modes 


EventOnePC = coeff1(:,1:m);
EventTwoPC = coeff2(:,1:m);
time_length = t_end - t_start + 1;

numCols1 = cellfun(@(x) size(x,2), EventOne);%minCols1 = min(numCols1);
numCols2 = cellfun(@(x) size(x,2), EventTwo);%minCols2 = min(numCols2);

numTri1 = numCols1/time_length;% number of trials for animal1
numTri2 = numCols2/time_length;% number of trials for animal2
minTri = min([numTri1,numTri2]);

LDOne = EventOneMatrix'*(EventOnePC);% LDOne is A
LDTwo = EventTwoMatrix'*(EventTwoPC);% LDTwo 

EventOneLD = mat2cell(LDOne, numCols1, m);
EventTwoLD = mat2cell(LDTwo, numCols2, m);

% Prelocate memory
EventOneLDAve = cell(1,length(ind));
EventTwoLDAve = cell(1,length(ind));
EventOneAve = cell(1,length(ind));
EventTwoAve = cell(1,length(ind));
LDOneAve = zeros(time_length,m);
LDTwoAve = zeros(time_length,m);
FROneAve = zeros(time_length,m);
FRTwoAve = zeros(time_length,m);
EventOneLDMinTri = cell(1,length(ind));
EventTwoLDMinTri = cell(1,length(ind));

for i = 1:length(ind)
    % Check if the number of rows in the current cell is a multiple of the current factor
    if mod(size(EventOneLD{i},1),numTri1(i)) ~= 0 ||...
            mod(size(EventTwoLD{i},1),numTri2(i)) ~= 0
        error('Not a multiple of factor');
    end
    
    for j = 1:m
        reshapeLD1 = reshape(EventOneLD{i}(:,j),time_length,numTri1(i));
        reshapeLD2 = reshape(EventTwoLD{i}(:,j),time_length,numTri2(i));
        reshapeFR1 = reshape(EventOne{i}(j,:),time_length,numTri1(i));
        reshapeFR2 = reshape(EventTwo{i}(j,:),time_length,numTri2(i));

        LDOneAve(:,j) = mean(reshapeLD1,2);
        LDTwoAve(:,j) = mean(reshapeLD2,2);
        FROneAve(:,j) = mean(reshapeFR1,2);
        FRTwoAve(:,j) = mean(reshapeFR2,2);
    end
    
    EventOneLDAve{i} = LDOneAve;
    EventTwoLDAve{i} = LDTwoAve;
    EventOneAve{i} = FROneAve;
    EventTwoAve{i} = FRTwoAve;
    
    EventOneLDMinTri{i} = EventOneLD{i}(1:minTri*time_length,:);
    EventTwoLDMinTri{i} = EventTwoLD{i}(1:minTri*time_length,:); 

end

EventOneLD_NotAve = EventOneLD;
EventTwoLD_NotAve = EventTwoLD;


% Average PCA (for stack time visualization)
EventOneLD = EventOneLDAve;
EventTwoLD = EventTwoLDAve;

% Take minimal trials and NOT average PCA (for calculate CC)
%EventOneLD = EventOneLDMinTri;
%EventTwoLD = EventTwoLDMinTri;


    otherwise
        return;
end
%% PCA - using seperate neural modes

% for i = 1:length(ind)      
%     % PCA (keep the mean of the original dataset as the first LD)
%     [coeff1,score1,explained1,m1] = PCA(EventOne{i}',2,0.9);% 1:demean+pc0; 2:keep mean
% %     [coeff1N,score1N,explained1N,m1N] = PCA(NoiseOne{i}',2,0.9);% 1:demean+pc0; 2:keep mean
% 
%     % MATLAB PCA
%     %[coeff1,score1,latent1,tsquared1,explained1,mu1] = pca(EventOne{i}');% MATLAB
%     %wcoeff1 = score1.* sqrt(explained1)';
%     
%     EventOneLD{i} = score1;
% %     NoiseOneLD{i} = score1N;
%     %wEventOnePC{i} = wcoeff1;
%     EventOnePC_explain{1,i} = explained1;
%     
%     % find variance
%     explained1(:,2) = cumsum(explained1(:,1));
%     numPC1(i) = find(explained1(:,2)>=threshold, 1);% 
%     xx_group1 = length(explained1(:,1));
%     xx_group1 = linspace(1,xx_group1,xx_group1);
%  
% 
%     % PCA (keep the mean of the original dataset as the first LD)
%     [coeff2,score2,explained2,m2] = PCA(EventTwo{i}',2,0.9);% % 1:demean+pc0; 2:keep mean
% %     [coeff2N,score2N,explained2N,m2N] = PCA(NoiseTwo{i}',2,0.9);% % 1:demean+pc0; 2:keep mean
% 
%     %[coeff2,score2,latent2,tsquared2,explained2,mu2] = pca(EventTwo{i}');%MATLAB pca
%     %wcoeff2 = score2.* sqrt(explained2)';% weighted LD
%     
%     EventTwoLD{i} = score2;
% %     NoiseTwoLD{i} = score2N;
%     %wEventTwoPC{i} = wcoeff2;
%     EventTwoPC_explain{1,i} = explained2;
%     
%     % find variance
%     explained2(:,2) = cumsum(explained2(:,1));
%     numPC2(i) = find(explained2(:,2)>=threshold, 1);
%     xx_group2 = length(explained2(:,1));
%     xx_group2 = linspace(1,xx_group2,xx_group2);
%     
%    
% end
%% End of calculating PCA

%% Calculate CCA
time_length = t_end - t_start + 1;
EventOneCV = cell(1, length(ind));
EventTwoCV = cell(1, length(ind));

% for i = 1:length(ind)
%     EventOneLD{i} = Ave(EventOneLD{i},time_length,1);
%     EventTwoLD{i} = Ave(EventTwoLD{i},time_length,1);
% %      NoiseOneLD{i} = Ave(NoiseOneLD{i},time_length,1);
% % %    NoiseTwoLD{i} = Ave(NoiseTwoLD{i},time_length,1);
% end

%% CCA in same mode
% numEvents = length(EventOne);% Or directly 8
% numTimePoints = size(EventOne{1}, 2);  % Or directly 4001
% EventOneLDMatrix = vertcat(EventOneLD{:}); 
% EventTwoLDMatrix = vertcat(EventTwoLD{:}); 
% 
% [A,B,U,V] = CCA(EventOneLDMatrix,EventTwoLDMatrix,numPC);
% 
% % U_com = mat2cell(U, repmat(numTimePoints,[1,numEvents]),size(U,2));
% % V_com = mat2cell(V, repmat(numTimePoints,[1,numEvents]),size(V,2));
% 
% U_com = mat2cell(U, repmat(4001,[1,numEvents]),size(U,2));
% V_com = mat2cell(V, repmat(4001,[1,numEvents]),size(V,2));
% 
% EventOneCV = U_com';
% EventTwoCV = V_com';

%% CCA in seperate modes

for i = 1:length(ind)
    if m >= numPC
        [A,B,U,V] = CCA(EventOneLD{i},EventTwoLD{i},m);
    else
        [A,B,U,V] = CCA(EventOneLD{i},EventTwoLD{i},numPC);
    end
%    [AN,BN,UN,VN] = CCA(NoiseOneLD{i},NoiseTwoLD{i},numPC);

%  [A,B,R,U,V,Stats] = canoncorr(EventOneLD{i}(:,1:numPC), EventTwoLD{i}(:,1:numPC));
%  U_unscaled = EventOneLD{i}(:,1:numPC)*A;
%  V_unscaled = EventTwoLD{i}(:,1:numPC)*B;
%  U = U_unscaled; V = V_unscaled;

EventOneCV{i} = U;
EventTwoCV{i} = V;

% NoiseOneCV{i} = UN;
% NoiseTwoCV{i} = VN;

end

%% End of calculating CCA using unweighted LD


%% Frequency Spectrum Analysis
% fs = 1000; % 1/binsize,binsize = 0.001s
% f_up = 15; % upper limit for FFT plot, Hz
% for i = 1:length(ind)
%     for j = 1:numPC
%         % Signal
%     [f,F1_LD,P1_LD] = SpFFT(EventOneLD{i}(:,j),fs);% LD1 specturm
%     [f,F2_LD,P2_LD] = SpFFT(EventTwoLD{i}(:,j),fs);% LD2 specturm
%     [f,F1_CV,P1_CV] = SpFFT(EventOneCV{i}(:,j),fs);% CV1 specturm
%     [f,F2_CV,P2_CV] = SpFFT(EventTwoCV{i}(:,j),fs);% CV2 specturm
%     
%     EventOneLDsp{i}(:,j) = P1_LD;
%     EventTwoLDsp{i}(:,j) = P2_LD;
%     EventOneCVsp{i}(:,j) = P1_CV;
%     EventTwoCVsp{i}(:,j) = P2_CV;
% 
%     % Noise spectrum 1: shuffle the raw data
% %   [f,P1_nLD] = SpFFT(NoiseOneLD{i}(:,j),fs);% LD1 specturm
% %   [f,P2_nLD] = SpFFT(NoiseTwoLD{i}(:,j),fs);% LD2 specturm
% %   [f,P1_nCV] = SpFFT(NoiseOneCV{i}(:,j),fs);% CV1 specturm
% %   [f,P2_nCV] = SpFFT(NoiseTwoCV{i}(:,j),fs);% CV2 specturm
%     
%     % Noise spectrum 2: randomize the phase
% %     [f,F1_nLD,P1_nLD] = RandPhase(EventOneLD{i}(:,j),fs);% LD1 specturm
% %     [f,F2_nLD,P2_nLD] = RandPhase(EventTwoLD{i}(:,j),fs);% LD2 specturm
% %     [f,F1_nCV,P1_nCV] = RandPhase(EventOneCV{i}(:,j),fs);% CV1 specturm
% %     [f,F2_nCV,P2_nCV] = RandPhase(EventTwoCV{i}(:,j),fs);% CV2 specturm
% %     
% %     NoiseOneLDifft{i}(:,j) = F1_nLD;
% %     NoiseTwoLDifft{i}(:,j) = F2_nLD;
% %     NoiseOneCVifft{i}(:,j) = F1_nCV;
% %     NoiseTwoCVifft{i}(:,j) = F2_nCV;
% %     
% %     NoiseOneLDsp{i}(:,j) = P1_nLD;
% %     NoiseTwoLDsp{i}(:,j) = P2_nLD;
% %     NoiseOneCVsp{i}(:,j) = P1_nCV;
% %     NoiseTwoCVsp{i}(:,j) = P2_nCV;
%     end
% end


%% Calculate correlation coefficient (Perason r) for top 3 neural modes

% Bin the top 4 PC for position, phase, and trial type
EventOneLDbin = cell(1,numPC);
EventTwoLDbin = cell(1,numPC);
EventOneCVbin = cell(1,numPC);
EventTwoCVbin = cell(1,numPC);

for i = 1:numPC
 for j = 1:length(ind)
    EventOneLDbin{i}(:,j) = EventOneLD{j}(:,i);
    EventTwoLDbin{i}(:,j) = EventTwoLD{j}(:,i);
  
    EventOneCVbin{i}(:,j) = EventOneCV{j}(:,i);
    EventTwoCVbin{i}(:,j) = EventTwoCV{j}(:,i);   

 end
end

% Calculate correlation coefficients for top 4 CV
r_LD = zeros(1,numPC);
r_CV = zeros(1,numPC);
for i = 1:numPC % top 4 neural modes
    
% calculate correlation coefficient for unweighted LD
R_LD = corrcoef(EventOneLDbin{i},EventTwoLDbin{i});
r_LD(i) = R_LD(1,2);

% calculate correlation coefficient for CV
R_CV = corrcoef(EventOneCVbin{i},EventTwoCVbin{i});
r_CV(i) = R_CV(1,2);

end
%% End of calculating correlation coefficient (Perason r) for top 3 neural modes


%% Calculate combined correlation coefficient for top 3 neural modes

% Combine the binned LD matrix and CV matrix
EventOneLDcom = cat(1,EventOneLDbin{:}); 
EventTwoLDcom = cat(1,EventTwoLDbin{:});

EventOneCVcom = cat(1,EventOneCVbin{:}); 
EventTwoCVcom = cat(1,EventTwoCVbin{:});

% Calculate combined correlation coefficient for unweighted LD
R_LD = corrcoef(EventOneLDcom,EventTwoLDcom);
% r_LD = [r_LD,R_LD(1,2)];
r_LD = [r_LD,mean(r_LD)];

% Calculate combined correlation coefficient for unweighted CV
R_CV = corrcoef(EventOneCVcom,EventTwoCVcom);
% r_CV = [r_CV,R_CV(1,2)];
r_CV = [r_CV,mean(r_CV)];
%% End of calculate combined correlation coefficient for top 3 neural modes



%% Calculate coefficient coefficient (Perason r) for weighted CV
%% weight1 CV
weight1CV = (r_CV(1:numPC).^2)/sum(r_CV(1:numPC).^2);% sqare weight
% weight1CV = sqrt(abs(r_CV(1:numPC))/sum(abs(r_CV(1:numPC))));% sqare root of weight

%% % Weighted sum CV - weight1
r_w1CV = dot(r_CV(1:numPC),weight1CV);

%% % Combined sum CV - weight1
w1EventOneCVbin = cell(1,numPC);
w1EventTwoCVbin = cell(1,numPC);
for i = 1:numPC
w1EventOneCVbin{i} = weight1CV(i).*EventOneCVbin{i};
w1EventTwoCVbin{i} = weight1CV(i).*EventTwoCVbin{i};
end

w1EventOneCVcom = cat(1,w1EventOneCVbin{:});
w1EventTwoCVcom = cat(1,w1EventTwoCVbin{:});
% 
% % calculated combined weighted CV (weight1)
% R_w1CV = corrcoef(w1EventOneCVcom,w1EventTwoCVcom);
% r_w1CV = R_w1CV(1,2);

%% weight2 - using Date variance explained as weight
%% For stack time - need average raw firing rate

W1 = zeros(length(ind),numPC); 
W2 = zeros(length(ind),numPC); 
w2EventOneCVbin = cell(1,numPC);
w2EventTwoCVbin = cell(1,numPC);

for i = 1:length(ind)
    for j = 1:numPC
        switch Arrange
            case 'neuron'
    W1(i,j)= weightCV(EventOne{i}',EventOneCV{i}(:,numPC));% stack neuron
    W2(i,j)= weightCV(EventTwo{i}',EventTwoCV{i}(:,numPC));% stack neuron
            
            case 'time'
%    % Stack time
    W1(i,j)= weightCV(EventOneAve{i},EventOneCV{i}(:,numPC));% average stack time
    W2(i,j)= weightCV(EventTwoAve{i},EventTwoCV{i}(:,numPC));% average stack time
            
            otherwise
                return;
        end
%     W1(i,j)= weightCV(EventOneAve{i}',EventOneCV{i}(:,numPC));% not average stack time
%     W2(i,j)= weightCV(EventTwoAve{i}',EventTwoCV{i}(:,numPC));% not average stack time
    end
end

for i = 1:numPC
    for j = 1:length(ind)
       w2EventOneCVbin{i}(:,j) = W1(j,i).*EventOneCVbin{i}(:,j);
       w2EventTwoCVbin{i}(:,j) = W2(j,i).*EventTwoCVbin{i}(:,j);
    end
end

w2EventOneCVcom = cat(1,w2EventOneCVbin{:});
w2EventTwoCVcom = cat(1,w2EventTwoCVbin{:});

%% % Weighted sum CV - weight2, Data exp
temW = r_CV(1:numPC)'.* sqrt(W1(1:numPC,1)).* sqrt(W2(1:numPC,1));% can be any column, not just 1
r_w2CV = sum(temW);
%% combined weighted CV - weight2, Data exp
% R_w2CV = corrcoef(w2EventOneCVcom,w2EventTwoCVcom);
% r_w2CV = R_w2CV(1,2);

%% 8x8 event-based comparison - correlatiob coefficient matrix 
R1 = corrcoef(EventOneCVcom); %1029u195 aligned
R1_w1 = corrcoef(w1EventOneCVcom); %1029u195 weighted aligned (CV corr)
R1_w2 = corrcoef(w2EventOneCVcom); %1029u195 weighted aligned (PC exp)

R2 = corrcoef(EventTwoCVcom); %1029u196 aligned
R2_w1 = corrcoef(w1EventTwoCVcom); %1029u196 weighted aligned (CV corr)
R2_w2 = corrcoef(w2EventTwoCVcom); %1029u196 weighted aligned (PC exp)

% Pad R with an extra row and column so that the color map will
% display all 8x8 vectors
R1 = padarray(R1, [1 1], 'post'); R1_w1 = padarray(R1_w1, [1 1], 'post');
R1_w2 = padarray(R1_w2, [1 1], 'post');

R2 = padarray(R2, [1 1], 'post'); R2_w1 = padarray(R2_w1, [1 1], 'post');
R2_w2 = padarray(R2_w2, [1 1], 'post');

%% Save correlation coefficient
Correlation.r_LD = r_LD(4);
Correlation.r_CV = r_CV(4);
Correlation.r_w_corCV = r_w1CV; % correlation weight
Correlation.r_w_pcwCV = r_w2CV; % PC explained weight

%% End of calculating coefficient (Perason r) for weighted CV


%% Generate Figures 
t_peri = SessionData.t_peri;
x = linspace(-t_peri,t_peri,length(EventOneLD{1}(:,1))); % binned time

%% set up event legend
groups_combined = cell(size(groups,1),1);
for i = 1:size(groups,1)
    groups_combined{i} = strjoin(groups(i,:),' ');  % Join with a space
end
labels = {'LSC', 'LSE', 'LNC', 'LNE', 'RSC', 'RSE', 'RNC', 'RNE'};
styles = {'r-', 'r-', 'r--', 'r--', 'b-', 'b-', 'b--', 'b--'}; % line styles
widths = [2, 0.5, 2, 0.5, 2, 0.5, 2, 0.5]; % line widths
%% set up color map
% Create axes
ax = gobjects(1,8);  % array to store axes handles

% Define your data range for the color scale
for i = 1:n
cmin1(i) = min(EventOne{i}(:));
cmin2(i) = min(EventTwo{i}(:));
cmax1(i) = max(EventOne{i}(:));
cmax2(i) = max(EventTwo{i}(:));
end

Cmin1 = min(cmin1);Cmin2 = min(cmin2);
cmin = min(Cmin1,Cmin2);
Cmax1 = max(cmax1);Cmax2 = max(cmax2);
cmax = max(Cmax1,Cmax2);

%% Figure1:firing rate for session1
% fig1 = figure(1);
% fig1_name = sprintf('%s smoothed firing rate_%dms',session1Name,sigma);
% sgtitle(fig1_name)
% colormap(jet);
% for i=1:length(EventOne)
%     ax(i) = subplot(1,length(EventOne),i);
%     h = pcolor(ax(i),EventOne{i});
%     set(h, 'EdgeColor', 'none');
%     caxis(ax(i), [cmin cmax]); % Synchronize color scale across subplots
%     %axis off; % Remove axis
%     %my_title = sprintf('%s',string(groups(ind(i),:)));
%     my_title = sprintf('%s,%s,%s',string(groups(ind(i),1)),...
%         string(groups(ind(i),2)),string(groups(ind(i),3)));
%     titleObj = title(my_title);%colorbar;
%     titleObj.FontSize = 6.5;
%     xlabel('Time(ms)');ylabel('Number of Neurons');
%     hold on
% end
% hold off
% axes('Position', [0.92, 0.11, 0.02, 0.815], 'Visible', 'off'); % adjust position as needed
% colorbar;
% caxis([cmin cmax]);

%% Figure2:firing rate for session2
% fig2 = figure(2);
% fig2_name = sprintf('%s smoothed firing rate_%dms',session2Name,sigma);
% sgtitle(fig2_name)
% colormap(jet);
% for i=1:length(EventTwo)
%     ax(i) = subplot(1,length(EventTwo),i);%(2,length(EventTwo),i+length(EventOne))
%     h = pcolor(EventTwo{i});
%     set(h, 'EdgeColor', 'none');
%     caxis(ax(i), [cmin cmax]); % Synchronize color scale across subplots
%     %axis off;% Remove axis
%     %my_title = sprintf('%s',string(groups(ind(i),:)));
%     my_title = sprintf('%s,%s,%s',string(groups(ind(i),1)),...
%         string(groups(ind(i),2)),string(groups(ind(i),3)));
%     titleObj = title(my_title);
%     titleObj.FontSize = 6.5;
%     xlabel('Time(ms)');ylabel('Number of Neurons');
%     hold on
% end
% hold off
% axes('Position', [0.92, 0.11, 0.02, 0.815], 'Visible', 'off'); % adjust position as needed
% colorbar;
% caxis([cmin cmax]);


%% Figure3: unweighted latent dynamics, EventOne vs EventTwo
fig3 = figure(3);
fig3_name = sprintf('%s vs. %s,unaligned latent dynamics(unweighted PCA)',session1Name,session2Name);
sgtitle(fig3_name)

for plotId = 1:2:6 
  EventOneLDid = 0.5.*plotId + 0.5;
  subplot(3,2,plotId)%1,3,5
%   newcolors = [0 0 1; 1 0.5 0; 1 1 0; 0.5 0 0.5; 0 1 0; 0.5 0.5 1; 1 0 0; 0 0 0];  % Define the color order
%   set(gca, 'ColorOrder', newcolors, 'NextPlot', 'replacechildren');  % Set the color order for the current axes
  for i=1:length(EventOne)
  plot(x,EventOneLD{1,i}(:,EventOneLDid),styles{i},...
      'LineWidth', widths(i));%,'linewidth',1.5);%1,2,3, no legend
  hold on 
  end
  %lgd = legend(pcOne, groups_combined);
  %lgd.Position = [0.15, 0.15, 0.2, 0.2];
  hold off
%xlabel('Event Time(sec)');
% my_ylabel = sprintf('Neural Mode%s',string(EventOneLDid));
% ylabel(my_ylabel);
legend(labels);
end

for plotId = 2:2:7 % 2,4,6
EventTwoLDid = 0.5.*plotId;% 1,2,3(n)
subplot(3,2,plotId)
% newcolors = [0 0 1; 1 0.5 0; 1 1 0; 0.5 0 0.5; 0 1 0; 0.5 0.5 1; 1 0 0; 0 0 0];  % Define the color order
% set(gca, 'ColorOrder', newcolors, 'NextPlot', 'replacechildren');  % Set the color order for the current axes
for i=1:length(EventTwo)
    plot(x,EventTwoLD{1,i}(:,EventTwoLDid), styles{i}, ...
        'LineWidth', widths(i));%,'linewidth',1.5);% 1st PC
    hold on 
end
hold off
legend(labels);
%xlabel('Event Time(sec)');
% my_ylabel = sprintf('Neural Mode%s',string(EventTwoLDid));
% ylabel(my_ylabel);
end

%% Figure4: CCA, EventOne vs EventTwo
fig4 = figure(4);
fig4_name = sprintf('%s vs. %s aligned latent dynamics(CCA)',session1Name,session2Name);
sgtitle(fig4_name)
for plotId = 1:2:6 %1,3,5
  EventOneCVid = 0.5.*plotId + 0.5;%1,2,3 
  subplot(3,2,plotId)
%   newcolors = [0 0 1; 1 0.5 0; 1 1 0; 0.5 0 0.5; 0 1 0; 0.5 0.5 1; 1 0 0; 0 0 0];  % Define the color order
%   set(gca, 'ColorOrder', newcolors, 'NextPlot', 'replacechildren');  % Set the color order for the current axes
  for i=1:length(EventOne)
  plot(x,EventOneCV{1,i}(:,EventOneCVid), styles{i}, ...
      'LineWidth', widths(i));%,'linewidth',1.5);% 1st PC
  hold on 
  end
  hold off
% xlabel('Event Time(sec)');
% my_ylabel = sprintf('Neural Mode%s',string(EventOneCVid));
% ylabel(my_ylabel);
legend(labels);
end

for plotId = 2:2:7 % 2,4,6
EventTwoCVid = 0.5.*plotId;% 1,2,3
subplot(3,2,plotId)
% newcolors = [0 0 1; 1 0.5 0; 1 1 0; 0.5 0 0.5; 0 1 0; 0.5 0.5 1; 1 0 0; 0 0 0];  % Define the color order
% set(gca, 'ColorOrder', newcolors, 'NextPlot', 'replacechildren');  % Set the color order for the current axes
for i=1:length(EventTwo)
    plot(x,EventTwoCV{1,i}(:,EventTwoCVid), styles{i}, ...
        'LineWidth', widths(i));%,'linewidth',1.5);% 1st PC
    hold on 
end
hold off
% xlabel('Time(s)');
% my_ylabel = sprintf('Neural Mode%s',string(EventTwoCVid));
% ylabel(my_ylabel);
legend(labels);
end

%% Figure5: Correlation, EventOne vs. EventTwo, LD and CV
fig5 = figure(5);
fig5_name = sprintf('%s vs. %s Correlation',session1Name,session2Name);
sgtitle(fig5_name)
subplot(1,2,1)
% corr = [r_LD(1) r_wLD(1) r_CV(1);r_LD(2) r_wLD(2) r_CV(2);...
%     r_LD(3) r_wLD(3) r_CV(3)]; 

corr = [r_LD(1) r_CV(1);r_LD(2) r_CV(2);...
    r_LD(3) r_CV(3); r_LD(4) r_CV(4)]; 

bar(abs(corr));
title('Top 4 Neural Modes Correlation');
xlabel('Neural Mode');ylabel('|Corrleation|');
legend('Unaligned','Aligned');
ylim([0,1]);

subplot(1,2,2)
% corr = [r_LD(end);r_wLD(end);r_CV(end);r_wCV]; 
corr = [r_LD(end);r_CV(end);r_w1CV;r_w2CV]; 
%corr = [r_LD(end);r_wCV]; 
x = 1;
bar(x,abs(corr));
title('Combined Correlation');
xlabel('Top4 Neural Mode');ylabel('|Corrleation Coefficient|');
% legend(['unweighted LD,r=',num2str(corr(1))],['weighted LD,r=',num2str(corr(2))],...
% ['unweighted CV,r=',num2str(corr(3))],['weighted CV,r=',num2str(corr(4))]);
legend('Unaligned','Aligned','Weighted Aligned(CV corr)','Weighted Aligned(Data exp)');
%legend('Unaligned','Aligned');
ylim([0,1]);

% %% Figure6: Frequency Specturm Analysis for LD
% fig6 = figure(6);
% fig6_name = sprintf('%s vs. %s Frequency Specturm of unaligned latent dynamics(PCA)',session1Name,session2Name);
% sgtitle(fig6_name)
% for plotId = 1:2:6 %1,3,5
%   EventOneLDspid = 0.5.*plotId + 0.5;%1,2,3 
%   subplot(3,2,plotId)
%   newcolors = [0 0 1; 1 0.5 0; 1 1 0; 0.5 0 0.5; 0 1 0; 0.5 0.5 1; 1 0 0; 0 0 0];  % Define the color order
%   set(gca, 'ColorOrder', newcolors, 'NextPlot', 'replacechildren');  % Set the color order for the current axes
%   for i=1:length(EventOne)
%   plot(f,EventOneLDsp{1,i}(:,EventOneLDspid));%,'linewidth',1.5);% 1st PC
%   xlim([0,f_up]);
%   hold on 
%   end
%   hold off
% xlabel('Frequency(Hz)');
% my_ylabel = sprintf('|P(f)|Mode%s',string(EventOneLDspid));
% ylabel(my_ylabel);
% legend('fs=1000Hz,f(Ny)=500Hz');
% end
% 
% for plotId = 2:2:7 % 2,4,6
% EventTwoLDspid = 0.5.*plotId;% 1,2,3
% subplot(3,2,plotId)
% newcolors = [0 0 1; 1 0.5 0; 1 1 0; 0.5 0 0.5; 0 1 0; 0.5 0.5 1; 1 0 0; 0 0 0];  % Define the color order
% set(gca, 'ColorOrder', newcolors, 'NextPlot', 'replacechildren');  % Set the color order for the current axes
% for i=1:length(EventTwo)
%     plot(f,EventTwoLDsp{1,i}(:,EventTwoLDspid));%,'linewidth',1.5);% 1st PC
%     xlim([0,f_up]);
%     hold on 
% end
% hold off
% xlabel('Frequency(Hz)');
% my_ylabel = sprintf('|P(f)|Mode%s',string(EventTwoLDspid));
% ylabel(my_ylabel);
% legend('fs=1000Hz,f(Ny)=500Hz');
% end

% 
% %% Figure7: Frequency Specturm Analysis for CV
% fig7 = figure(7);
% fig7_name = sprintf('%s vs. %s Frequency Specturm of aligned latent dynamics(CCA)',session1Name,session2Name);
% sgtitle(fig7_name)
% for plotId = 1:2:6 %1,3,5
%   EventOneCVspid = 0.5.*plotId + 0.5;%1,2,3 
%   subplot(3,2,plotId)
%   newcolors = [0 0 1; 1 0.5 0; 1 1 0; 0.5 0 0.5; 0 1 0; 0.5 0.5 1; 1 0 0; 0 0 0];  % Define the color order
%   set(gca, 'ColorOrder', newcolors, 'NextPlot', 'replacechildren');  % Set the color order for the current axes
%   for i=1:length(EventOne)
%   plot(f,EventOneCVsp{1,i}(:,EventOneCVspid));%,'linewidth',1.5);% 1st PC
%   xlim([0,f_up]);
%   hold on 
%   end
%   hold off
% xlabel('Frequency(Hz)');
% my_ylabel = sprintf('|P(f)|Mode%s',string(EventOneCVspid));
% ylabel(my_ylabel);
% legend('fs=1000Hz,f(Ny)=500Hz');
% end
% 
% for plotId = 2:2:7 % 2,4,6
% EventTwoCVspid = 0.5.*plotId;% 1,2,3
% subplot(3,2,plotId)
% newcolors = [0 0 1; 1 0.5 0; 1 1 0; 0.5 0 0.5; 0 1 0; 0.5 0.5 1; 1 0 0; 0 0 0];  % Define the color order
% set(gca, 'ColorOrder', newcolors, 'NextPlot', 'replacechildren');  % Set the color order for the current axes
% for i=1:length(EventTwo)
%     plot(f,EventTwoCVsp{1,i}(:,EventTwoCVspid));%,'linewidth',1.5);% 1st PC
%     xlim([0,f_up]);
%     hold on 
% end
% hold off
% xlabel('Frequency(Hz)');
% my_ylabel = sprintf('|P(f)|Mode%s',string(EventTwoCVspid));
% ylabel(my_ylabel);
% legend('fs=1000Hz,f(Ny)=500Hz');
% end

% 
% %% Figure8: Frequency Specturm Analysis for Noise LD
% fig8 = figure(8);
% fig8_name = sprintf('%s vs. %s noise Frequency Specturm of unaligned latent dynamics(PCA)',session1Name,session2Name);
% sgtitle(fig8_name)
% for plotId = 1:2:6 %1,3,5
%   NoiseOneLDspid = 0.5.*plotId + 0.5;%1,2,3 
%   subplot(3,2,plotId)
%   newcolors = [0 0 1; 1 0.5 0; 1 1 0; 0.5 0 0.5; 0 1 0; 0.5 0.5 1; 1 0 0; 0 0 0];  % Define the color order
%   set(gca, 'ColorOrder', newcolors, 'NextPlot', 'replacechildren');  % Set the color order for the current axes
%   for i=1:length(NoiseOne)
%   plot(f,NoiseOneLDsp{1,i}(:,NoiseOneLDspid));%,'linewidth',1.5);% 1st PC
%   xlim([0,f_up]);
%   hold on 
%   end
%   hold off
% xlabel('Frequency(Hz)');
% my_ylabel = sprintf('|P(f)|Mode%s',string(NoiseOneLDspid));
% ylabel(my_ylabel);
% legend('fs=1000Hz,f(Ny)=500Hz');
% end
% 
% for plotId = 2:2:7 % 2,4,6
% NoiseTwoLDspid = 0.5.*plotId;% 1,2,3
% subplot(3,2,plotId)
% newcolors = [0 0 1; 1 0.5 0; 1 1 0; 0.5 0 0.5; 0 1 0; 0.5 0.5 1; 1 0 0; 0 0 0];  % Define the color order
% set(gca, 'ColorOrder', newcolors, 'NextPlot', 'replacechildren');  % Set the color order for the current axes
% for i=1:length(NoiseTwo)
%     plot(f,NoiseTwoLDsp{1,i}(:,NoiseTwoLDspid));%,'linewidth',1.5);% 1st PC
%     xlim([0,f_up]);
%     hold on 
% end
% hold off
% xlabel('Frequency(Hz)');
% my_ylabel = sprintf('|P(f)|Mode%s',string(NoiseTwoLDspid));
% ylabel(my_ylabel);
% legend('fs=1000Hz,f(Ny)=500Hz');
% end
% 
% %% Figure9: noise Frequency Specturm Analysis for CV
% fig9 = figure(9);
% fig9_name = sprintf('%s vs. %s noise Frequency Specturm of aligned latent dynamics(CCA)',session1Name,session2Name);
% sgtitle(fig9_name)
% for plotId = 1:2:6 %1,3,5
%   NoiseOneCVspid = 0.5.*plotId + 0.5;%1,2,3 
%   subplot(3,2,plotId)
%   newcolors = [0 0 1; 1 0.5 0; 1 1 0; 0.5 0 0.5; 0 1 0; 0.5 0.5 1; 1 0 0; 0 0 0];  % Define the color order
%   set(gca, 'ColorOrder', newcolors, 'NextPlot', 'replacechildren');  % Set the color order for the current axes
%   for i=1:length(NoiseOne)
%   plot(f,NoiseOneCVsp{1,i}(:,NoiseOneCVspid));%,'linewidth',1.5);% 1st PC
%   xlim([0,f_up]);
%   hold on 
%   end
%   hold off
% xlabel('Frequency(Hz)');
% my_ylabel = sprintf('|P(f)|Mode%s',string(NoiseOneCVspid));
% ylabel(my_ylabel);
% legend('fs=1000Hz,f(Ny)=500Hz');
% end
% 
% for plotId = 2:2:7 % 2,4,6
% NoiseTwoCVspid = 0.5.*plotId;% 1,2,3
% subplot(3,2,plotId)
% newcolors = [0 0 1; 1 0.5 0; 1 1 0; 0.5 0 0.5; 0 1 0; 0.5 0.5 1; 1 0 0; 0 0 0];  % Define the color order
% set(gca, 'ColorOrder', newcolors, 'NextPlot', 'replacechildren');  % Set the color order for the current axes
% for i=1:length(NoiseTwo)
%     plot(f,NoiseTwoCVsp{1,i}(:,NoiseTwoCVspid));%,'linewidth',1.5);% 1st PC
%     xlim([0,f_up]);
%     hold on 
% end
% hold off
% xlabel('Frequency(Hz)');
% my_ylabel = sprintf('|P(f)|Mode%s',string(NoiseTwoCVspid));
% ylabel(my_ylabel);
% legend('fs=1000Hz,f(Ny)=500Hz');
% end

%% Figure10: STFT for CV
% fig10 = figure(10);
% sgtitle('u195 aligned 1st LD')
% colormap(jet);
% 
% % for plotId = 1:2:16
% for plotId = 1:8
%     subplot(1,8,plotId)
% %     stftID = 0.5.*plotId + 0.5;%1,2,3 
%       stftID = plotId;%1,2,3 
% %     surf(T,F,10*log10(EventOneLDstft{stftID}),'edgecolor','none'); 
% %     axis tight; 
% %     view(0,90);
%    
%    imagesc(T,F,10*log10(EventOneCVstft{stftID})); 
%    axis tight; 
%    set(gca,'YDir','normal'); % to display lower frequencies at the bottom
%    ylim([0,f_up]);
%    
%    xlabel('Time(Seconds)'); 
%    ylabel('Frequency(Hz)');
%    my_title = sprintf('Event%d',plotId);title(my_title);
% end
% 
% fig11 = figure(11);
% sgtitle('u195 random phase(noise) aligned 1st LD')
% colormap(jet);
% 
% % for plotId = 2:2:17
% for plotId = 1:8
%     subplot(1,8,plotId)
% %     stftID = 0.5.*plotId;%1,2,3 
%     stftID = plotId;%1,2,3 
% %     surf(T,F,10*log10(NoiseOneLDstft{stftID}),'edgecolor','none'); 
% %     axis tight; 
% %     view(0,90);
% 
%    imagesc(T,F,10*log10(NoiseOneCVstft{stftID})); 
%    axis tight; 
%    set(gca,'YDir','normal'); % to display lower frequencies at the bottom
%    ylim([0,f_up]); 
%    
%    xlabel('Time(sec)'); 
%    ylabel('Frequency(Hz)');
%    my_title = sprintf('Event%d',plotId);title(my_title);
% end

%% Figure12:correlation matrix of 8 events (Rat1) - aligned, not weighted CV
fig12 = figure(12);
imagesc(R1);% R1,rat1; R2,rat2
ax12 = gca; % # NEED TO CALL AFTER PLOT THE DATE!! specify the axis
colorbar; % Show the color scale
colormap(jet); % Choose the color map, you can change it according to your preference
title('Correlation Coefficient Matrix (aligned CV)');
labels = {'LSC', 'LSE', 'LNC', 'LNE', 'RSC', 'RSE', 'RNC', 'RNE'};

% Set the tick locations and labels
xticks(ax12, 0.5:1:8.5); yticks(ax12, 0.5:1:8.5);

% Set grid
grid(ax12, 'on'); set(ax12, 'GridColor', 'k', 'GridAlpha', 1);

% Limit the axis to not show the padded row and column
axis(ax12, [0.5 8.5 0.5 8.5]);

% Remove x and y tick labels
xticklabels(ax12, []); yticklabels(ax12, []);

% Manually place labels at desired locations
for i = 1:length(labels) 
    % x label
    text(ax12, i, 8.8, labels{i}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    % y label
    text(ax12, 0.2, i, labels{i}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end

%% Figure13:correlation matrix of 8 events (Rat1) - weighted aligned CV (CV corr)
fig13 = figure(13);
imagesc(R1_w1);% R1,rat1; R2,rat2
ax13 = gca; % # NEED TO CALL AFTER PLOT THE DATE!! specify the axis
colorbar; % Show the color scale
colormap(jet); % Choose the color map, you can change it according to your preference
title('Correlation Coefficient Matrix (weighted aligned CV - CV corr)');

% Set the tick locations and labels
xticks(ax13, 0.5:1:8.5); yticks(ax13, 0.5:1:8.5);

% Set grid
grid(ax13, 'on'); set(ax13, 'GridColor', 'k', 'GridAlpha', 1);

% Limit the axis to not show the padded row and column
axis(ax13, [0.5 8.5 0.5 8.5]);

% Remove x and y tick labels
xticklabels(ax13, []); yticklabels(ax13, []);

% Manually place labels at desired locations
for i = 1:length(labels) 
    % x label
    text(ax13, i, 8.8, labels{i}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    % y label
    text(ax13, 0.2, i, labels{i}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end


%% Figure14:correlation matrix of 8 events (Rat1) - weighted aligned CV (CV corr)
fig14 = figure(14);
imagesc(R1_w2);% R1,rat1; R2,rat2
ax14 = gca; % # NEED TO CALL AFTER PLOT THE DATE!! specify the axis
colorbar; % Show the color scale
colormap(jet); % Choose the color map, you can change it according to your preference
title('Correlation Coefficient Matrix (weighted aligned CV - Data exp)');
%labels = {'LSC', 'LSE', 'LNC', 'LNE', 'RSC', 'RSE', 'RNC', 'RNE'};
% Set the tick locations and labels
xticks(ax14, 0.5:1:8.5); yticks(ax14, 0.5:1:8.5);

% Set grid
grid(ax14, 'on'); set(ax14, 'GridColor', 'k', 'GridAlpha', 1);

% Limit the axis to not show the padded row and column
axis(ax14, [0.5 8.5 0.5 8.5]);

% Remove x and y tick labels
xticklabels(ax14, []); yticklabels(ax14, []);

% Manually place labels at desired locations
for i = 1:length(labels) 
    % x label
    text(ax14, i, 8.8, labels{i}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    % y label
    text(ax14, 0.2, i, labels{i}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end

%% Figure15: 3D projected data - unaligned
fig15 = figure(15);
fig15_name = sprintf('%s vs. %s 3D unaligned latent dynamics(PCA)',session1Name,session2Name);
sgtitle(fig15_name)
subplot(1,2,1)
% newcolors = [0 0 1; 1 0.5 0; 1 1 0; 0.5 0 0.5; 0 1 0; 0.5 0.5 1; 1 0 0; 0 0 0];  % Define the color order
% set(gca, 'ColorOrder', newcolors, 'NextPlot', 'replacechildren');  % Set the color order for the current axes
for i = 1:length(ind)
    plot3(EventOneLD{i}(:,1),EventOneLD{i}(:,2),EventOneLD{i}(:,3),styles{i}, ...
        'LineWidth', widths(i));
    hold on
end
hold off
legend(labels);

subplot(1,2,2)
% newcolors = [0 0 1; 1 0.5 0; 1 1 0; 0.5 0 0.5; 0 1 0; 0.5 0.5 1; 1 0 0; 0 0 0];  % Define the color order
% set(gca, 'ColorOrder', newcolors, 'NextPlot', 'replacechildren');  % Set the color order for the current axes
for i = 1:length(ind)
    plot3(EventTwoLD{i}(:,1),EventTwoLD{i}(:,2),EventTwoLD{i}(:,3),styles{i},...
        'LineWidth', widths(i));
    hold on
end
hold off
legend(labels);

%% Figure16: 3D projected data - aligned
fig16 = figure(16);
fig16_name = sprintf('%s vs. %s 3D aligned latent dynamics(CCA)',session1Name,session2Name);
sgtitle(fig16_name)
subplot(1,2,1)
% newcolors = [0 0 1; 1 0.5 0; 1 1 0; 0.5 0 0.5; 0 1 0; 0.5 0.5 1; 1 0 0; 0 0 0];  % Define the color order
% set(gca, 'ColorOrder', newcolors, 'NextPlot', 'replacechildren');  % Set the color order for the current axes
for i = 1:length(ind)
    plot3(EventOneCV{i}(:,1),EventOneCV{i}(:,2),EventOneCV{i}(:,3), styles{i}, ...
        'LineWidth', widths(i));
    hold on
end
hold off
legend(labels);

subplot(1,2,2)
% newcolors = [0 0 1; 1 0.5 0; 1 1 0; 0.5 0 0.5; 0 1 0; 0.5 0.5 1; 1 0 0; 0 0 0];  % Define the color order
% set(gca, 'ColorOrder', newcolors, 'NextPlot', 'replacechildren');  % Set the color order for the current axes
for i = 1:length(ind)
    plot3(EventTwoCV{i}(:,1),EventTwoCV{i}(:,2),EventTwoCV{i}(:,3), styles{i},...
        'LineWidth', widths(i));
    hold on
end
hold off
legend(labels);
%% Save Figures
FigureFolder = 'Results'; % User need to specific the folder that containing .nex files
FigurePath = fullfile(path, FigureFolder);

% Store figure name in a cell array
%figArray = [fig1, fig2, fig3, fig4, fig5, fig6, fig7];% include ICA 
% figArray = [fig1, fig2, fig3, fig4, fig5];% exclude ICA
% figNameArray = cell(1,length(figArray)); % Preallocate a cell array of size 1x8

% for i=1:length(figArray)
%     figNameArray{i} = eval(['fig', num2str(i),'_name']);% read fig_name
%     %figArray{i} = (['fig', num2str(i)]);
% end

% Save figure
% for i=1:length(figArray)   
%     fig_name = sprintf('%s.png',char(figNameArray{i}));% figure name
%     fig_path = fullfile(FigurePath, fig_name);% still figure name
%     saveas(figArray(i), fig_path);
% end


% disp('figures saved')

%% Save data
% str = file;
% parts = split(str, 'S');  % split the string at the period
% newStr = parts(1);
% newStr = char(newStr);
% mat_name = sprintf('%s vs %s%s_Correlation',session1Name,session2Name,binStr);
% save(mat_name,'Correlation','-v7.3');
% disp('data saved')

elapsed_time = toc;
fprintf('Elapsed time: %.2f seconds\n', elapsed_time);