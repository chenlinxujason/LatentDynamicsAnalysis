clear   
clc
close all

tic

time = 2000; % Gaussian window length (ms) 2000ms
num_points = 2000;% use 2000 sample points, equal to Gaussian window length
t = linspace(0, time, num_points);

% Generate the PDFs for the two distributions with 2000 sample points
% pdf1 = 5.*normpdf(t, 400, 50);
% pdf2 = 5.*normpdf(t, 550, 50);
% pdf3 = 5.*normpdf(t, 1000, 50);
% pdf4 = 5.*normpdf(t, 1600, 50);

pdf1 = 5.*normpdf(t, 400, 100);
pdf2 = 5.*normpdf(t, 800, 100);
pdf3 = 5.*normpdf(t, 1200, 100);
pdf4 = 5.*normpdf(t, 1600, 100);



Com = input('1 or n(1: one PC; n: multiple pc):','s');% 1: one PC; n: multiple pc
switch Com
    case '1'
%% One PC
y1 = pdf1 + (0.5/500);
y2 = pdf3 + (0.5/500);

for i = 1:90
    RawSpikes1(i,:) = rand(1,num_points)< pdf1;%blue
    RawSpikes2(i,:) = rand(1,num_points)< pdf2;   
end

% empty
for i = 91:100
    RawSpikes1(i,:) = 0;
    RawSpikes2(i,:) = 0;   
end

y1_2d = y1' * y1;
y2_2d = y2' * y2;
Distr1 = y1_2d;
Distr2 = y2_2d;

     case 'n'
%% Multiple PC

%% Spike1 kernel - 1D
y1_k1 = pdf1 + (1/500); % spike1, 1st PC
y1_k2 = pdf2 + (50/500);% spike1, 2nd PC

%% Spike1 kernel - 2D(for PCA),using outer product
y1_k1_2d = y1_k1' * y1_k1;
y1_k2_2d = y1_k2' * y1_k2;
Distr1 = [y1_k1_2d; y1_k2_2d];

%% Spike2 kernel
y2_k1 = pdf3 + (1/500);% spike2, 1st PC
y2_k2 = pdf1 + (50/500);% spike2, 2nd PC

%% Spike2 kernel - 2D(for PCA),using outer product
y2_k1_2d = y2_k1' * y2_k1;
y2_k2_2d = y2_k2' * y2_k2;
Distr2 = [y2_k1_2d; y2_k2_2d];

%% Spike1 - spike trains
% 1st PC
for i = 1:50
    RawSpikes1(i,:) = rand(1,num_points)< pdf1;%blue
end

% 2nd PC
for i = 51:100
    RawSpikes1(i,:) = rand(1,num_points)< pdf2;%blue
end

% empty
for i = 100:101
    RawSpikes1(i,:) = 0;
end


%% Spike2 - spike trains
   % 1st PC
for i = 1:50
    RawSpikes2(i,:) = rand(1,num_points)< pdf3;%blue
end

% 2nd PC
for i =51:100
    RawSpikes2(i,:) = rand(1,num_points)< pdf1;%blue
end

% empty
for i = 100:101
    RawSpikes2(i,:) = 0;
end


end

% real PC

% MATLAB PCA
%[coeff_1r,score_1r,latent_1r,tsquared_1r,explained_1r,mu_1r]= pca(Distr1);
%[coeff_2r,score_2r,latent_2r,tsquared_2r,explained_2r,mu_2r]= pca(Distr2);

%% PCA -demean (1) standard
%[coeff1r,score1r,explained1r,m1r]= PCA(Distr1',1,0.9);% method2: s_tot + s_var based
%[coeff2r,score2r,explained2r,m2r]= PCA(Distr2',1,0.9);% method2: s_tot + s_var based

%% PCA -no demean (2)
[coeff1r,score1r,explained1r,m1r]= PCA(Distr1',3,0.6);% method2: s_tot + s_var based
[coeff2r,score2r,explained2r,m2r]= PCA(Distr2',3,0.6);% method2: s_tot + s_var based

SpikeOneRealLD = score1r;
SpikeTwoRealLD = score2r;

% real CV
numPC = 2;
% MATLAB
%[A,B,R,U,V,Stats] = canoncorr(SpikeOneRealPC(:,1:numPC), SpikeTwoRealPC(:,1:numPC));
%U_unscaled = SpikeOneRealPC(:,1:numPC)*A;
%V_unscaled = SpikeTwoRealPC(:,1:numPC)*B;

[A,B,U,V] = CCA(SpikeOneRealLD,SpikeTwoRealLD,numPC);

SpikeOneRealCV = U;
SpikeTwoRealCV = V;

% weighted real PC
%score_1r = score1r.* latent1r';
%score_2r = score2r.* latent2r';

% Concatenate spikes1 and spikes2
RawSpikes1 = double(RawSpikes1); RawSpikes2 = double(RawSpikes2);
num_neurons = size(RawSpikes1,1);


[row_1,col_1] = size(RawSpikes1);
[row_2,col_2] = size(RawSpikes2);

% Define standard deviation vector 
std_vec1 = linspace(1,40,40);%50 points from 1ms to 50ms
std_vec2 = logspace(log10(40),log10(100),10);% 30 points from 50ms to 300ms log scale

% Combine x1 and x2.
std_vec = [std_vec1 std_vec2(2:end)];

% repeat times
numTrial = 5;

% Preallocate memory
r_real = zeros(1,length(std_vec));
r_noise = zeros(1,length(std_vec));
MseOne = zeros(1,length(std_vec));
MseTwo = zeros(1,length(std_vec));

R_real = zeros(numTrial,length(std_vec));
R_noise = zeros(numTrial,length(std_vec));
trialMseOne = zeros(numTrial,length(std_vec));
trialMseTwo = zeros(numTrial,length(std_vec));

for trial = 1:numTrial % 100 trials 
 for e = 1:num_neurons
    RawSpikeRow1 = RawSpikes1(e,:);
    RawSpikeRow2 = RawSpikes2(e,:);
    %rng(89);%89
    k1 = RawSpikeRow1(randperm(col_1));
    %rng(89);%89
    k2 = RawSpikeRow2(randperm(col_2));
    h1 = [];
    h2 = [];

          if e == 1
                RawNoise1 = [k1;h1];
                RawNoise2 = [k2;h2];
          else
                RawNoise1 = [RawNoise1;k1];
                RawNoise2 = [RawNoise2;k2];
               
          end
end

SimData.RawSpikes1 = RawSpikes1;
SimData.RawSpikes2 = RawSpikes2;
SimData.RawNoise1 = RawNoise1;
SimData.RawNoise2 = RawNoise2;
SimData.Distr1 = Distr1;
SimData.Distr2 = Distr2;


%% Smooth

for std = 1:length(std_vec)
    
    window = 5*std_vec(std); % window size
    x = -window:window; % domain of the kernel
    kernel = exp(-x.^2/(2*std_vec(std)^2)) / (std_vec(std)*sqrt(2*pi)); % Gaussian kernel
             
    % Preallocate an array for the smoothed counts
     SmoothSpikes1 = zeros(size(RawSpikes1));  
     SmoothSpikes2 = zeros(size(RawSpikes2));
     SmoothNoise1 = zeros(size(RawNoise1));  
     SmoothNoise2 = zeros(size(RawNoise2));
    
     for j = 1:num_neurons   % Loop over each row(neurons)
        SmoothSpikes1(j,:) = conv(RawSpikes1(j,:), kernel, 'same');
        SmoothSpikes2(j,:) = conv(RawSpikes2(j,:), kernel, 'same');
        SmoothNoise1(j,:) = conv(RawNoise1(j,:), kernel, 'same');
        SmoothNoise2(j,:) = conv(RawNoise2(j,:), kernel, 'same');
     end   

%% Calculate PCA 
%% EventOne and NoiseOne
   %% PCA -demean (1)
   %[coeff1R,score1R,explained1R,m1R]= PCA(SmoothSpikes1',1,0.9);% method2: s_tot + s_var based
   %[coeff1N,score1N,explained1N,m1N]= PCA(SmoothNoise1',1,0.9);% method2: s_tot + s_var based

   %% PCA -no demean (2)
   [coeff1R,score1R,explained1R,m1R]= PCA(SmoothSpikes1',3,0.6);% method2: s_tot + s_var based
   [coeff1N,score1N,explained1N,m1N]= PCA(SmoothNoise1',3,0.6);% method2: s_tot + s_var based

   %% MATLAB PCA
   %[coeff1R,score1R,latent1R,tsquared1R,explained1R,mu1R] = pca(SmoothSpikes1);% MATLAB
   %[coeff1N,score1N,latent1N,tsquared1N,explained1N,mu1N] = pca(SmoothNoise1);% MATLAB
    %wcoeff1R = coeff1R.* sqrt(explained1R)';% weighted real PC
    %wcoeff1N = coeff1N.* sqrt(explained1N)';% weighted noise PC
    
    SpikeOneLD = score1R;
    EventOnePC_explain = explained1R;
    
    NoiseOneLD = score1N;
    NoiseOnePC_explain = explained1N;
    
%% EventTwo and NoiseTwo
    %% PCA -demean (1)
   %[coeff2R,score2R,explained2R,m2R]= PCA(SmoothSpikes2',1,0.9);% method2: s_tot + s_var based
   %[coeff2N,score2N,explained2N,m2N]= PCA(SmoothNoise2',1,0.9);% method2: s_tot + s_var based

   %% PCA -no demean (2)
   [coeff2R,score2R,explained2R,m2R]= PCA(SmoothSpikes2',2,0.9);% method2: s_tot + s_var based
   [coeff2N,score2N,explained2N,m2N]= PCA(SmoothNoise2',2,0.9);% method2: s_tot + s_var based

    % MATLAB PCA
   %[coeff2R,score2R,latent2R,tsquared2R,explained2R,mu2R] = pca(SmoothSpikes2);% MATLAB
   %[coeff2N,score2N,latent2N,tsquared2N,explained2N,mu2N] = pca(SmoothNoise2);% MATLAB
  % wcoeff2R = coeff2R.* sqrt(explained2R)';% weighted real PC
  % wcoeff2N = coeff2N.* sqrt(explained2N)';% weighted noise PC
    
    SpikeTwoLD = score2R;
    EventTwoPC_explain = explained2R;
    
    NoiseTwoLD = score2N;
    NoiseTwoPC_explain = explained2N;
 
% End calculate PCA using MATLAB pca function 


%% Calculate CCA use weighted PC
%numPC = min(n_PC0_group1(sigma),n_PC0_group2(sigma));% Define the number of PC for CCA comparison
%numPC = min(numPC,3);
numPC = 3;

[rA,rB,rU,rV] = CCA(SpikeOneLD,SpikeTwoLD,numPC);
%% MATLAB
%[rA,rB,rR,rU,rV,rStats] = canoncorr(SpikeOnePC(:,1:numPC), SpikeTwoPC(:,1:numPC));
%rU_unscaled = SpikeOnePC(:,1:numPC)*rA;
%rV_unscaled = SpikeTwoPC(:,1:numPC)*rB;
%rU = rU_unscaled; rV = rV_unscaled;

SpikeOneCV{std} = rU;
SpikeTwoCV{std} = rV;

[nA,nB,nU,nV] = CCA(NoiseOneLD, NoiseTwoLD,numPC);%nU_unscaled = NoiseOnePC(:,1:numPC)*nA;
%nV_unscaled = NoiseTwoPC(:,1:numPC)*nB;
%nU = nU_unscaled; nV = nV_unscaled;

NoiseOneCV = nU;
NoiseTwoCV = nV;


% End Calculate CCA

%% Calculate correlation coefficient (Perason r)

% Calculate correlation coefficients for top 3 neural modes
for i = 1:numPC % top 3 neural modes
    
% calculate correlation coefficient for PC
R_PCreal = corrcoef(SpikeOneLD(:,i),SpikeTwoLD(:,i));
r_PCreal{std}(i) = R_PCreal(1,2);

% calculate correlation coefficient for CC
R_CVreal = corrcoef(SpikeOneCV{std}(:,i),SpikeTwoCV{std}(:,i));
r_CVreal{std}(i) = R_CVreal(1,2);

% calculate correlation coefficient for noise PC
R_PCnoise = corrcoef(NoiseOneLD(:,i),NoiseTwoLD(:,i));
r_PCnoise{std}(i) = R_PCnoise(1,2);

% calculate correlation coefficient for noise CV
R_CVnoise = corrcoef(NoiseOneCV(:,i),NoiseTwoCV(:,i));
r_CVnoise{std}(i) = R_CVnoise(1,2);

end

% Calculate combined correlation coefficient for the first 3 PCs
R_PCreal = corrcoef(SpikeOneLD(:,1:3),SpikeTwoLD(:,1:3));
r_PCreal{std} = [r_PCreal{std},R_PCreal(1,2)];

% Calculate combined correlation coefficient for the first 3 CVs
R_CVreal = corrcoef(SpikeOneCV{std}(:,1:3),SpikeTwoCV{std}(:,1:3));
r_CVreal{std} = [r_CVreal{std},R_CVreal(1,2)];

% Calculate combined correlation coefficient for the first 3 noise PCs
R_PCnoise = corrcoef(NoiseOneLD(:,1:3),NoiseTwoLD(:,1:3));
r_PCnoise{std} = [r_PCnoise{std},R_PCnoise(1,2)];

% Calculate combined correlation coefficient for the first 3 noise CVs
R_CVnoise = corrcoef(NoiseOneCV(:,1:3),NoiseTwoCV(:,1:3));
r_CVnoise{std} = [r_CVnoise{std},R_CVnoise(1,2)];

%% Calculate coefficient for weighted CC 

% Sqrt weight
%weightCVreal = sqrt(r_CVreal(1:numPC)/sum(r_CVreal(1:numPC)));% sqrt weight
%weightCVnoise = sqrt(r_CVnoise(1:numPC)/sum(r_CVnoise(1:numPC)));% sqrt weight

% Square weight
weightCVreal{std} = (r_CVreal{std}(1:numPC).^2)/sum(r_CVreal{std}(1:numPC).^2);% sqare weight
weightCVnoise{std} = (r_CVnoise{std}(1:numPC).^2)/sum(r_CVnoise{std}(1:numPC).^2);% sqare weight

% calculate weighted CV
wSpikeOneCV = weightCVreal{std}.*SpikeOneCV{std};
wSpikeTwoCV = weightCVreal{std}.*SpikeTwoCV{std};

wNoiseOneCV = weightCVnoise{std}.*NoiseOneCV;
wNoiseTwoCV = weightCVnoise{std}.*NoiseTwoCV;

% calculated combined weighted CV correlation coefficient
R_wCVreal = corrcoef(wSpikeOneCV(:,1:3),wSpikeTwoCV(:,1:3));
r_wCVreal = R_wCVreal(1,2);

R_wCVnoise = corrcoef(wNoiseOneCV(:,1:3),wNoiseTwoCV(:,1:3));
r_wCVnoise = R_wCVnoise(1,2);

r_real(std) = r_wCVreal;
r_noise(std) = r_wCVnoise;

  
% MSE
%  MseOne(std) = abs(immse(SpikeOneRealLD(:,1),SpikeOneLD(:,1)));
%  MseTwo(std) = abs(immse(SpikeTwoRealLD(:,1),SpikeTwoLD(:,1)));

% Cosine Sim
 CosSimOneLD(std) = (cosSim(SpikeOneRealLD(:,1),SpikeOneLD(:,1)));% LD
 CosSimTwoLD(std) = (cosSim(SpikeTwoRealLD(:,1),SpikeTwoLD(:,1)));% LD
% 

 CosSimOneCV(std) = (cosSim(SpikeOneRealCV(:,1),rU(:,1)));% correct!
 CosSimTwoCV(std) = (cosSim(SpikeTwoRealCV(:,1),rV(:,1)));% correct!

%  CosSimOne(std) = abs((cosSim(SpikeOneRealCV(:,1),rU(:,1))));% correct!
%  CosSimTwo(std) = abs((cosSim(SpikeTwoRealCV(:,1),rV(:,1))));% correct!

% CosSimOne(std) = (cosSim([SpikeOneRealCV(:,1);SpikeOneRealCV(:,2)],...
%     [rU(:,1);rU(:,2)]));% not correct!
% CosSimTwo(std) = (cosSim([SpikeTwoRealCV(:,1);SpikeTwoRealCV(:,2)],...
%     [rV(:,1);rV(:,2)]));% not correct!

% Varianced
VarOne(std) = weightCV(SmoothSpikes1',rU(:,1));
VarTwo(std) = weightCV(SmoothSpikes2',rV(:,1));

%clear r_PCreal r_CVreal r_PCnoise r_CVnoise r_wCVreal r_wCVnoise

%Data.r_PC = r_PC;%[r_1stPC, r_2ndPC, r_3rdPC, r_comPC] 
%Data.r_CC = r_CC;%[r_1stCC, r_2ndCC, r_3rdCC, r_comCC] 

% End calculate correlation coefficient
end %std end
%across trial
R_real(trial,:) = r_real;
R_noise(trial,:) = r_noise;

% trialMseOne(trial,:) = MseOne;
% trialMseTwo(trial,:) = MseTwo;

 end % for trial end


%% Calculate correlation difference

% average across trials
R_noise_sorted = sort(R_noise, 1); % Sort each column
R_noise_trimmed = R_noise_sorted(2:end-1, :); % Remove the first (minimum) and last (maximum) rows
meanRnoise = mean(R_noise_trimmed); % Compute the mean of each column
meanRreal = R_real(1,:);

diff = meanRreal - meanRnoise;

% meanRreal = r_real;
% meanRnoise = r_noise;
[indx,indy] = max(abs(diff));%indy is the index, cos_similarity
opt_std = std_vec(indy);% optimized std

%SimData.OptSigma = opt_std;
%save('SinglePC','SimData','-v7.3');
%disp('data saved')

%% FIGURE

%% Figure1: std difference
fig1 = figure(1);
sgtitle('Corelation difference')
subplot(2,1,1)
semilogx(std_vec, meanRreal, 'b');  % real r
hold on
semilogx(std_vec, meanRnoise, 'r');  % noise r
hold on
semilogx(std_vec, diff, 'k');  % difference
hold off
xlabel('Standard deviation(ms)');
ylabel('Value');
legend('mean real correlation','mean noise correlation','difference');

subplot(2,1,2)
sz = 5;
plot(std_vec, meanRreal, 'b');  % real r
%scatter(std_vec, meanRreal,sz,'b','filled');
hold on
plot(std_vec, meanRnoise, 'r');  % real r
%scatter(std_vec, meanRnoise,sz,'r','filled');  % noise r
hold on
plot(std_vec, diff, 'k');  % real r
%scatter(std_vec, diff,sz,'k','filled'); % difference
hold off
xlabel('Standard deviation(ms)');
ylabel('Value');
legend('real correlation','noise correlation','difference');
%xlim([0,100]);

%% Figure2: raw spike and kernels
fig2 = figure(2);
subplot(3,1,1)
switch Com
    case '1'
plot(y1, 'b', 'LineWidth', 2);
hold on
plot(y2, 'r', 'LineWidth', 2);
hold off
xlabel('Time (ms)');
ylabel('Probability density');
title('Gaussian distributions of two spike trains');
legend('kernel1','kernel2');

        
    case 'n'
plot(y1_k1, 'b', 'LineWidth', 2);
hold on
plot(y2_k1, 'r', 'LineWidth', 2);
hold on
plot(y1_k2, 'b', 'LineWidth', 2);
hold on
plot(y2_k2, 'r', 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Probability density');
title('Gaussian distributions of two spike trains');
legend('kernel1','kernel2');
end

subplot(3,1,2)
sz = 8;
for i = 1:num_neurons
    xs1 = find(RawSpikes1(i,:));
    scatter(xs1,i*ones(size(xs1)), sz, 'b', 'filled');
    hold on
    
    xs2 = find(RawSpikes2(i,:));
    scatter(xs2,i*ones(size(xs2)), sz, 'r', 'filled');
    hold on
end
title('Raw spike trains');
legend('neuron population1','neuron population2');  
xlim([0,2000]);
xlabel('Time (ms)');
ylabel('Neuron index'); 

subplot(3,1,3)
for i = 1:num_neurons
    xs1 = find(RawNoise1(i,:));
    scatter(xs1,i*ones(size(xs1)), sz, 'b', 'filled');
    hold on
    
    xs2 = find(RawNoise2(i,:));
    scatter(xs2,i*ones(size(xs2)), sz, 'r', 'filled');
    hold on
end
title('Raw spike trains(shuffle)');
legend('neuron population1','neuron population2');  
xlim([0,2000]);
xlabel('Time (ms)');
ylabel('Neuron index'); 


%% Figure3: corr diff, MSE, cos
fig3 = figure(3);
subplot(3,1,1)
semilogx(std_vec, meanRreal, 'b');%,'LineWidth',1.5);  % real r
hold on
semilogx(std_vec, meanRnoise, 'r');%,'LineWidth',1.5);  % noise r
hold on
semilogx(std_vec, diff, 'k');%,'LineWidth',1.5);  % difference
hold off
my_title = sprintf('Correlation difference(optimal std=%dms)',opt_std);
title(my_title);
xlabel('Standard deviation(ms)');ylabel('Value');
legend('mean real data correlation','mean noise correlation','difference');

subplot(3,1,2)
% plot(std_vec, CosSimOne, 'b');%,'linewidth',2);  % real r
semilogx(std_vec, (CosSimOneLD), 'b');%,'linewidth',2);  % real r
% plot(std_vec, MseOne, 'b');%,'linewidth',2);  % real r
hold on
% plot(std_vec, CosSimTwo, 'r');%,'linewidth',2);  % noise r
semilogx(std_vec, (CosSimTwoLD), 'r');%,'linewidth',2);  % noise r
% plot(std_vec, MseTwo, 'r');%,'linewidth',2);  % noise r
hold off
my_title = sprintf('Cosine similarity(optimal std=%dms)',opt_std);
title(my_title);
xlabel('Standard deviation(ms)');ylabel('Cos similiarity');
legend('Spike1 LD vs. real Spike1 LD','Spike2 LD vs. real Spike2 LD');

subplot(3,1,3)
semilogx(std_vec, (CosSimOneCV), 'b');%,'linewidth',2);  % real r
hold on
semilogx(std_vec, (CosSimTwoCV), 'r');%,'linewidth',2);  % noise r
hold off
my_title = sprintf('Cosine similarity(optimal std=%dms)',opt_std);
title(my_title);
xlabel('Standard deviation(ms)');ylabel('Cos similiarity');
legend('Spike1 CV vs. real Spike1 CV','Spike2 CV vs. real Spike2 CV');


fig4 = figure(4);
sgtitle('Demean method');
subplot(1,2,1)
h = pcolor(SmoothSpikes1);
set(h, 'EdgeColor', 'none'); 
title('Smoothed Spike1');xlabel('time (a.u.)');ylabel('Number of Neurons');

subplot(1,2,2)
h = pcolor(SmoothSpikes2);
set(h, 'EdgeColor', 'none');
title('Smoothed Spike2');xlabel('time (a.u.)');ylabel('Number of Neurons');
elapsed_time = toc;
fprintf('Elapsed time: %.2f seconds\n', elapsed_time);

% fig5 = figure(5);
% semilogx(std_vec, VarOne, 'b');%,'linewidth',2);  % real r
% hold on
% % plot(std_vec, CosSimTwo, 'r');%,'linewidth',2);  % noise r
% semilogx(std_vec, VarTwo, 'r');%,'linewidth',2);  % noise r
% hold off
% my_title = sprintf('Variance(optimal std=%dms)',opt_std);
% title(my_title);
% xlabel('Standard deviation(ms)');ylabel('Variance');
% legend('Spike1 Variance','Spike2 Variance');
%%%%%%%%%%%%%%%%%%%%%% signal PCA and CCA
% fig6 = figure(6);
% sig = opt_std;
% subplot(2,2,1)
% plot(SpikeOneRealCV(:,1));title('Spike1 real CV');
% 
% subplot(2,2,2)
% plot(SpikeTwoRealCV(:,1));title('Spike2 real CV');
% 
% subplot(2,2,3)
% plot(SpikeOneCV{int16(opt_std)}(:,1));title('Spike1 CV');
% 
% subplot(2,2,4)
% plot(SpikeTwoCV{int16(opt_std)}(:,1));title('Spike2 CV');
% 
% 
