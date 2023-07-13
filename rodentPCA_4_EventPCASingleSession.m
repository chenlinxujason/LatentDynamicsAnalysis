clear; clc; close all;
% Open file dialog box to select a .mat file
[file,path] = uigetfile('*.mat','Select the .mat file with Opt std');
if isequal(file,0)
    disp('User selected Cancel');
else
 tic    
    load(fullfile(path,file));% Load the selected .mat file  
end

input_info = Data.input_info;
Com = Data.Com;
groups = Data.groups;
disp_str = sprintf('Compare %s and %s',input_info{1},input_info{2});
disp(disp_str);

%AnimalID = input('Enter animal ID: ', 's');%i.e 1029,1036,etc
%SessionID = input('Enter session ID: ', 's');%i.e 195,etc

%% Smooth
% Define Gaussian kernel parameters
sigma = Data.OptSigma;% optimized std
window = 5*sigma; % window size
x = -window:window; % domain of the kernel
kernel = exp(-x.^2/(2*sigma^2)) / (sigma*sqrt(2*pi)); % Gaussian kernel
   
  % Smooth spike firing counts with Gaussian kernel
    for i = 1:length(Data.group_RawReal)
        
         RawSpike_counts = Data.group_RawReal{i};%raw Real data
         
         % Preallocate an array for the smoothed counts
         SmoothSpike_counts = zeros(size(RawSpike_counts));  
         
           for j = 1:size(RawSpike_counts,1)  % Loop over each row
               SmoothSpike_counts(j,:) = conv(RawSpike_counts(j,:), kernel, 'same');
           end
         group_SmoothedReal{i} = SmoothSpike_counts;
    end

%% Extract data 
if strcmp(Com, 'type')
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
end
%% Grouping step 2, case2 - trial wise comparison 
if strcmp(Com, 'trial')
    ind_one = find(strcmp(groups(:,1),input_info{1}) & ...
    strcmp(groups(:,2),input_info{2}) & strcmp(groups(:,3),input_info{3}));

    ind_two = find(strcmp(groups(:,1),input_info{4}) & ...
    strcmp(groups(:,2),input_info{5}) & strcmp(groups(:,3),input_info{6}));
end

for j = 1:length(ind_one)
    k = ind_one(j);
    EventOne{j} = group_SmoothedReal{k};
end
    
for j = 1:length(ind_two)
    k = ind_two(j);
    EventTwo{j} = group_SmoothedReal{k};
end
    
%% Calculate PCA and ICA 
threshold = 0.8;% 80% of total variance
numPC1 = zeros(1,8);
numPC2 = zeros(1,8);

for i = 1:length(ind_one) 
    % PCA (keep the mean of the original dataset as the first LD)
    [coeff1,score1,explained1,m1] = PCA(EventOne{i}',2,0.9);% 1:demean+pc0; 2:keep mean
    
    % MATLAB PCA
    %[coeff1,score1,latent1,tsquared1,explained1,mu1] = pca(EventOne{i});% MATLAB
    %wcoeff1 = coeff1.* sqrt(explained1)';
    
    EventOneLD{i} = score1;
    %wEventOnePC{i} = wcoeff1;
    EventOnePC_explain{1,i} = explained1;
    
    % Calculate unweighted ICA using the first 3 LD
    %Md1 = rica(score1,3);
    %IC1 = transform(Md1,score1);
    
    % Calculate weighted ICA using the first 3 weighted LD
    %wscore1 = score1.*latent1';
    %wMd1 = rica(wscore1,3);
    %wIC1 = transform(wMd1,wscore1);
   
    %EventOneIC{i} = IC1;
    %wEventOneIC{i} = wIC1;
    
    % find variance
    explained1(:,2) = cumsum(explained1(:,1));
    numPC1(i) = find(explained1(:,2)>=threshold, 1);% 
    xx_group1 = length(explained1(:,1));
    xx_group1 = linspace(1,xx_group1,xx_group1);

end

for i = 1:length(ind_two)
    % PCA (keep the mean of the original dataset as the first LD)
    [coeff2,score2,explained2,m2] = PCA(EventTwo{i}',2,0.9);% % 1:demean+pc0; 2:keep mean
    
    %[coeff2,score2,latent2,tsquared2,explained2,mu2] = pca(EventTwo{i});%MATLAB pca
    %wcoeff2 = coeff2.* sqrt(explained2)';% weighted PC
    
    EventTwoLD{i} = score2;
    %wEventTwoPC{i} = wcoeff2;
    EventTwoPC_explain{1,i} = explained2;
    
    % Calculate unweighted ICA using the first 3 projected data
    %Md2 = rica(score2,3);
    %IC2 = transform(Md2,score2);
    
    % Calculate weighted ICA using the first 3 weighted projected data
    %wscore2 = score2.*latent2';
    %wMd2 = rica(wscore2,3);
    %wIC2 = transform(wMd2,wscore2);
    
    %EventTwoIC{i} = IC2;
    %wEventTwoIC{i} = wIC2;
    
    % find variance
    explained2(:,2) = cumsum(explained2(:,1));
    numPC2(i) = find(explained2(:,2)>=threshold, 1);
    xx_group2 = length(explained2(:,1));
    xx_group2 = linspace(1,xx_group2,xx_group2);

end
%% End of calculating PCA and ICA


%% Calculate CCA
%numPC = min(n_PC0_group1(sigma),n_PC0_group2(sigma));% Define the number of PC for CCA comparison
%numPC = min(numPC,3);
numPC = 3;
for i = 1:length(ind_one)
   [A,B,U,V] = CCA(EventOneLD{i},EventTwoLD{i},numPC);

%  [A,B,R,U,V,Stats] = canoncorr(EventOneLD{i}(:,1:numPC), EventTwoLD{i}(:,1:numPC));
%  U_unscaled = EventOneLD{i}(:,1:numPC)*A;
%  V_unscaled = EventTwoLD{i}(:,1:numPC)*B;
%  U = U_unscaled; V = V_unscaled;

EventOneCV{i} = U;
EventTwoCV{i} = V;

end
%% End of calculating CCA using unweighted LD

%% Calculate correlation coefficient (Perason r) for top 3 neural modes

% Bin the top 3 LD for position, phase, and trial type
for i = 1:numPC
 for j = 1:length(ind_one)
    EventOneLDbin{i}(:,j) = EventOneLD{j}(:,i);
    EventTwoLDbin{i}(:,j) = EventTwoLD{j}(:,i);
    
    %wEventOnePCbin{i}(:,j) = wEventOnePC{j}(:,i);
    %wEventTwoPCbin{i}(:,j) = wEventTwoPC{j}(:,i);
    
    EventOneCVbin{i}(:,j) = EventOneCV{j}(:,i);
    EventTwoCVbin{i}(:,j) = EventTwoCV{j}(:,i);
    
    %EventOneICbin{i}(:,j) = EventOneIC{j}(:,i);
    %EventTwoICbin{i}(:,j) = EventTwoIC{j}(:,i);
    
 end
end

% Calculate correlation coefficients for top 3 CV
for i = 1:numPC % top 3 neural modes
    
% calculate correlation coefficient for unweighted PC
R_LD = corrcoef(EventOneLDbin{i},EventTwoLDbin{i});
r_LD(i) = R_LD(1,2);

% calculate correlation coefficient for weighted PC
%R_wPC = corrcoef(wEventOnePCbin{i},wEventTwoPCbin{i});
%r_wPC(i) = R_wPC(1,2);

% calculate correlation coefficient for CV
R_CV = corrcoef(EventOneCVbin{i},EventTwoCVbin{i});
r_CV(i) = R_CV(1,2);

% calculate correlation coefficient for unweighted IC
%R_IC = corrcoef(EventOneICbin{i},EventTwoICbin{i});
%r_IC(i) = R_IC(1,2);

end

%% End of calculating correlation coefficient (Perason r) for top 3 neural modes

%% Calculate combined correlation coefficient for top 3 neural modes

% Combine the binned PC matrix and CV matrix
EventOneLDcom = cat(1,EventOneLDbin{:}); 
EventTwoLDcom = cat(1,EventTwoLDbin{:});

%wEventOnePCcom = cat(1,wEventOnePCbin{:}); 
%wEventTwoPCcom = cat(1,wEventTwoPCbin{:});

EventOneCVcom = cat(1,EventOneCVbin{:}); 
EventTwoCVcom = cat(1,EventTwoCVbin{:});

%EventOneICcom = cat(1,EventOneICbin{:}); 
%EventTwoICcom = cat(1,EventTwoICbin{:});

% Calculate combined correlation coefficient for unweighted LD
R_LD = corrcoef(EventOneLDcom,EventTwoLDcom);
r_LD = [r_LD,R_LD(1,2)];

% Calculate combined correlation coefficient for weighted LD
%R_wPC = corrcoef(wEventOnePCcom,wEventTwoPCcom);
%r_wPC = [r_wPC,R_wPC(1,2)];

% Calculate combined unweighted correlation coefficient for  CV
R_CV = corrcoef(EventOneCVcom,EventTwoCVcom);
r_CV = [r_CV,R_CV(1,2)];

% Calculate combined correlation coefficient for unweighted IC
%R_IC = corrcoef(EventOneICcom,EventTwoICcom);
%r_IC = [r_IC,R_IC(1,2)];

%% End of calculate combined correlation coefficient for top 3 neural modes

%% Calculate coefficient coefficient (Perason r) for weighted CV 

% Sqrt weight
%weightCV = sqrt(r_CV(1:numPC)/sum(r_CV(1:numPC)));% sqrt weight

% Square weight
weightCV = (r_CV(1:numPC).^2)/sum(r_CV(1:numPC).^2);% sqare weight

for i = 1:numPC
wEventOneCVbin{i} = weightCV(i).*EventOneCVbin{i};
wEventTwoCVbin{i} = weightCV(i).*EventTwoCVbin{i};
end

wEventOneCVcom = cat(1,wEventOneCVbin{:});
wEventTwoCVcom = cat(1,wEventTwoCVbin{:});

% calculated combined weighted CV
R_wCV = corrcoef(wEventOneCVcom,wEventTwoCVcom);
r_wCV = R_wCV(1,2);


%Data.r_LD = r_LD;%[r_1stLD, r_2ndLD, r_3rdLD, r_comLD] 
%Data.r_CV = r_CV;%[r_1stCV, r_2ndCV, r_3rdCV, r_comCV] 
%Data.r_IC = r_IC;%[r_1stIC, r_2ndIC, r_3rdIC, r_comIC] 
%Data.rwCV = r_wCV;

%% End of calculating coefficient (Perason r) for weighted CV

%% Generate Figures
x = linspace(-2,2,length(EventOneLD{1}(:,1))); % binned time

%% Figure1:firing rate
fig1 = figure(1);
fig1_name = sprintf('%s vs. %s,smoothed firing rate',Data.EventOneName,Data.EventTwoName);
sgtitle(fig1_name)
for i=1:length(EventOne)
    subplot(2,length(EventOne),i)
    h = pcolor(EventOne{i});
    set(h, 'EdgeColor', 'none');
    my_title = sprintf('%s %s %s',string(groups(ind_one(i),1)),...
        string(groups(ind_one(i),2)),string(groups(ind_one(i),3)));
    title(my_title);xlabel('Time(ms)');ylabel('Number of Neurons');colorbar;
    hold on
end

for i=1:length(EventTwo)
    subplot(2,length(EventTwo),i+length(EventOne))
    h = pcolor(EventTwo{i});
    set(h, 'EdgeColor', 'none');
    my_title = sprintf('%s %s %s',string(groups(ind_two(i),1)),...
        string(groups(ind_two(i),2)),string(groups(ind_two(i),3)));
    title(my_title);xlabel('Time(ms)');ylabel('Number of Neurons');colorbar;
    hold on
end

%% Figure2: unweighted PCA, EventOne vs EventTwo
fig2 = figure(2);
fig2_name = sprintf('%s vs. %s,unaligned latent dynamics(unweighted PCA)',Data.EventOneName,Data.EventTwoName);
sgtitle(fig2_name)
for plotId = 1:2:6 
  EventOneLDid = 0.5.*plotId + 0.5;
  subplot(3,2,plotId)%1,3,5
  for i=1:length(EventOne)
  plot(x,EventOneLD{1,i}(:,EventOneLDid));%,'linewidth',2);% 1,2,3
  hold on 
  end
%xlabel('Event Time(sec)');
my_ylabel = sprintf('Neural Mode%s',string(EventOneLDid));
ylabel(my_ylabel);
end

for plotId = 2:2:7 % 2,4,6
EventTwoLDid = 0.5.*plotId;% 1,2,3(n)
subplot(3,2,plotId)
for i=1:length(EventTwo)
    plot(x,EventTwoLD{1,i}(:,EventTwoLDid));%,'linewidth',2);% 1st PC
    hold on 
end
xlabel('Event Time(sec)');
my_ylabel = sprintf('Neural Mode%s',string(EventTwoLDid));
ylabel(my_ylabel);
end

%% Figure4: CCA, EventOne vs EventTwo
fig4 = figure(4);
fig4_name = sprintf('%s vs. %s aligned latent dynamics(CCA)',Data.EventOneName,Data.EventTwoName);
sgtitle(fig4_name)
for plotId = 1:2:6 %1,3,5
  EventOneCVid = 0.5.*plotId + 0.5;%1,2,3 
  subplot(3,2,plotId)
  for i=1:length(EventOne)
  plot(x,EventOneCV{1,i}(:,EventOneCVid));%,'linewidth',2);% 1st PC
  hold on 
  end
xlabel('Event Time(sec)');
my_ylabel = sprintf('Neural Mode%s',string(EventOneCVid));
ylabel(my_ylabel);
end

for plotId = 2:2:7 % 2,4,6
EventTwoCVid = 0.5.*plotId;% 1,2,3
subplot(3,2,plotId)
for i=1:length(EventTwo)
    plot(x,EventTwoCV{1,i}(:,EventTwoCVid));%,'linewidth',2);% 1st PC
    hold on 
end
xlabel('Event Time(sec)');
my_ylabel = sprintf('Neural Mode%s',string(EventTwoCVid));
ylabel(my_ylabel);
end

%% Figure5: Correlation, EventOne vs. EventTwo, PC and CV
fig5 = figure(5);
fig5_name = sprintf('%s vs. %s Correlation',Data.EventOneName,Data.EventTwoName);
sgtitle(fig5_name)
subplot(1,2,1)
% corr = [r_LD(1) r_wLD(1) r_CV(1);r_LD(2) r_wLD(2) r_CV(2);...
%     r_LD(3) r_wLD(3) r_CV(3)]; 

corr = [r_LD(1) r_CV(1);r_LD(2) r_CV(2);...
    r_LD(3) r_CV(3)]; 

bar(abs(corr));
title('Top 3 Neural Modes Correlation');
xlabel('Neural Mode');ylabel('|Corrleation|');
legend('Unaligned','Aligned');
ylim([0,1]);


subplot(1,2,2)
% corr = [r_LD(4);r_wLD(4);r_CV(4);r_wCV]; 
corr = [r_LD(4);r_CV(4);r_wCV]; 
%corr = [r_LD(4);r_wCV]; 
x = 1;
bar(x,abs(corr));
title('Combined Correlation');
xlabel('Top3 Neural Mode');ylabel('|Corrleation Coefficient|');
% legend(['unweighted LD,r=',num2str(corr(1))],['weighted LD,r=',num2str(corr(2))],...
% ['unweighted CV,r=',num2str(corr(3))],['weighted CV,r=',num2str(corr(4))]);
legend('Unaligned','Aligned','Weighted Aligned');
%legend('Unaligned','Aligned');
ylim([0,1]);



%% Figure7: unweighted ICA, EventOne vs EventTwo
%fig7 = figure(7);
%fig7_name = sprintf('%s vs. %s,unweighted IC',Data.EventOneName,Data.EventTwoName);
%sgtitle(fig7_name)
%for plotId = 1:2:6 
%  EventOneICid = 0.5.*plotId + 0.5;
%  subplot(3,2,plotId)%1,3,5
%  for i=1:length(EventOne)
%  plot(x,EventOneIC{1,i}(:,EventOneICid),'linewidth',2);% 1,2,3
%  hold on 
%  end
%xlabel('Time(s)');
%my_ylabel = sprintf('unweighted IC%s',string(EventOneICid));
%ylabel(my_ylabel);
%end

%for plotId = 2:2:7 % 2,4,6
%EventTwoICid = 0.5.*plotId;% 1,2,3(n)
%subplot(3,2,plotId)
%for i=1:length(EventTwo)
%    plot(x,EventTwoIC{1,i}(:,EventTwoICid),'linewidth',2);% 1st PC
%    hold on 
%end
%xlabel('Time(s)');
%my_ylabel = sprintf('unweighted IC%s',string(EventTwoICid));
%ylabel(my_ylabel);
%end

%% Figure7: weighted ICA, EventOne vs EventTwo
%fig7 = figure(7);
%fig7_name = sprintf('%s vs. %s,weighted IC',Data.EventOneName,Data.EventTwoName);
%sgtitle(fig7_name)
%for plotId = 1:2:6 
%  wEventOneICid = 0.5.*plotId + 0.5;
%  subplot(3,2,plotId)%1,3,5
%  for i=1:length(EventOne)
%  plot(x,wEventOneIC{1,i}(:,wEventOneICid),'linewidth',2);% 1,2,3
%  hold on 
%  end
%xlabel('Time(s)');
%my_ylabel = sprintf('weighted IC%s',string(wEventOneICid));
%ylabel(my_ylabel);
%end

%for plotId = 2:2:7 % 2,4,6
%wEventTwoICid = 0.5.*plotId;% 1,2,3(n)
%subplot(3,2,plotId)
%for i=1:length(EventTwo)
%    plot(x,wEventTwoIC{1,i}(:,wEventTwoICid),'linewidth',2);% 1st PC
%    hold on 
%end
%xlabel('Time(s)');
%my_ylabel = sprintf('weighted IC%s',string(wEventTwoICid));
%ylabel(my_ylabel);
%end

%% Save figure
FigureFolder = 'Results'; % User need to specific the folder that containing .nex files
FigurePath = fullfile(path, FigureFolder);

% Store figure name in a cell array
%figArray = [fig1, fig2, fig3, fig4, fig5, fig6, fig7];% include ICA 
figArray = [fig1, fig2, fig3, fig4, fig5];% exclude ICA
figNameArray = cell(1,length(figArray)); % Preallocate a cell array of size 1x8

for i=1:length(figArray)
    figNameArray{i} = eval(['fig', num2str(i),'_name']);% read fig_name
    %figArray{i} = (['fig', num2str(i)]);
end

% Save figure
for i=1:length(figArray)   
    fig_name = sprintf('%s.png',char(figNameArray{i}));% figure name
    fig_path = fullfile(FigurePath, fig_name);% still figure name
    saveas(figArray(i), fig_path);
end


disp('figures saved')

%% Save data
%str = file;
%parts = split(str, 'S');  % split the string at the period
%newStr = parts(1);
%newStr = char(newStr);
%mat_name = sprintf('%s vs %s%s_Correlation',Data.EventOneName,Data.EventTwoName,binStr);
%save(mat_name,'Data','-v7.3');
%disp('data saved')

elapsed_time = toc;
fprintf('Elapsed time: %.2f seconds\n', elapsed_time);