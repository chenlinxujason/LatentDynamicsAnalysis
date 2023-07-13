function [f,P] = SpFFT(x,fs)

% x: time domain signal
% fs: sampling frequency

Y = fft(x); % calculate frequency domain data (complex vector)
P2 = abs(Y/length(x));% get magnitude by taking absolute value of complex vector
P1 = P2(1:length(x)/2+1);
P1(2:end-1) = 2*P1(2:end-1);% only get one-side since specturm is symmetric
P = P1;

%fs = 5000; % Sampling frequency of raw data(Hz). dt is the time-step in your time vector.
fs = 1000; % dt=bin size = 0.001s, 1/dt=1000;
f_unscale = (0:(length(x)/2))/length(x);% Frequency plot, from 0Hz to Nyquist frequency
f_scale = fs * (0:(length(x)/2))/length(x);% scale by sampling frequency
f = f_scale;

% figure;
% plot(f, P1);
% title('Single-Sided Amplitude Spectrum of U(t)');
% xlabel('f (Hz)');
% ylabel('|P1(f)|');