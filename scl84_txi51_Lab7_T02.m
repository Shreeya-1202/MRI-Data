%% Lab 7: Laplace Transforms, Transfer Functions, and Filtering
%scl84 & txi51
%Shreeya Lingam & Tanishka Isaac
%November 20,2023

%% Part 1: Laplace Transforms & Inverse Laplace Transforms 
clear 
clc


syms t s;

% Laplace Transform 

func1 = 8*heaviside(t-1) - 2*heaviside(t-3);

func2 = sin(3*pi*t)* heaviside(t-2);

func3 = exp(-2*(t - 1))* heaviside(t);
func4 = exp(-4*t)* heaviside(t-3);




%The transforms 

function1s = laplace(func1,t,s)


function2s = laplace(func2,t,s)


function3s = laplace(func3,t,s)

function4s = laplace(func4,t,s)


% Inverse Laplace Transform 


invfunc1 = 1/(4+s);

invfunc2 = 1;

invfunc3 = (27*s - 4)/ (s^2 - s - 2);

invfunc4 = 10 / ((s+4)^4);

% The transforms


invfunc1t = ilaplace(invfunc1,s,t)

invfunc2t = ilaplace(invfunc2,s,t)

invfunc3t = ilaplace(invfunc3,s,t)

invfunc4t = ilaplace(invfunc4,s,t)


%% %% %% Graphs
clc
clear


syms t s;
t = [1:0.1:10]

a = 8* heaviside(t-1) - (2*heaviside(t-3));

 
b = (sin(3*pi*t).*heaviside(t-2));

c = exp(-2*(t - 1)).*heaviside(t);

 
d = exp(-4*t).* heaviside(t-3);





figure (1)
plot(t,a)
xlabel('Time(s)')
ylabel('Amplitude')
title('Graph for function A')

figure (2)
plot(t,b)
xlabel('Time(s)')
ylabel('Amplitude')
title('Graph for function B')

figure (3)
plot(t,c)
xlabel('Time(s)')
ylabel('Amplitude')
title('Graph for function C')

figure (4)
plot(t,d)
xlabel('Time(s)')
ylabel('Amplitude')
title('Graph for function D')
 



%% PART 2: LAPLACE TRANSFORMS & TRANSFER FUNCTIONS 

clear 
clc

syms s
% Output and input of the system 
input = 18*s^4 + 216*s^3 + 150*s^2 + 162*s +14
output = (48*s^3 + 72*s^2 - 640*s -480)

% Extract the coefficents of the numerator and denominator
den_coeff = [18 216 150 162 14 ];
num_coeff = [ 0 48 72 -640 -480];

% Calculate the transfer function
H = tf(num_coeff,den_coeff)

% Find the poles, the zeroes, and the gain of the transfer fucntion 
[z,p,k] = tf2zp(num_coeff,den_coeff)

% Create a 1x2 subplot 
figure(5)
subplot(1,2,1)
% Use pzmap() to plot the poles and zeros of the system 
pzmap(H)
sgrid 
grid on 
% Plot the response of the system using the step() function 
subplot(1,2,2)
step(H)

%% Part 3 
clear 
clc 

% Load the MRI data
load('EBME358_Lab7_MRI_data.mat')

% QUESTION 1
% Get the spatial image of the original MRI data 
spatial_image = ifft2(MRIdata);

% Get the magnitude of the results
mag_spatial_image = abs(spatial_image);

% Plot the magnitude of the results 
figure(6)
plot(mag_spatial_image)
% Display the MRI data 
imagesc(mag_spatial_image);
colormap gray;
axis off;
axis equal;
title('2-D Inverse Discrete Fourier Transform')

% QUESTION 2
% Create a 2x2 subplot 
figure(7)
% Plot the 2D FT, with the original data
subplot(2,2,1)
plot(mag_spatial_image)
% Display the MRI data 
imagesc(mag_spatial_image);
colormap gray;
axis off;
axis equal;
title('2-D Inverse Discrete Fourier Transform')

% Set every other column of the original data set to zero 
new_mag_spatial_image = MRIdata;
new_mag_spatial_image(:,1:2:end)=0;

new_mag_spatial_image_1 = new_mag_spatial_image 

% Find the inverse fourier transform 
new_mag_spatial_image_2 = ifft2(new_mag_spatial_image_1)

% Get the magnitude of the results 
new_mag_spatial_image_3 = abs(new_mag_spatial_image_2)

% Plot the 2D FT, with every other column set to zero 
subplot(2,2,2)
plot(new_mag_spatial_image_3)
% Display the MRI data 
imagesc(new_mag_spatial_image_3);
colormap gray;
axis off;
axis equal;
title('2-D Inverse Discrete Fourier Transform, with every other coumn set to zero')

% Set every other row of the original data set to zero 
new1_mag_spatial_image = MRIdata;
new1_mag_spatial_image(1:2:end,:)=0;
 
new1_mag_spatial_image_1 = new1_mag_spatial_image 

% Find the inverse fourier transform 
new1_mag_spatial_image_2 = ifft2(new1_mag_spatial_image_1)

% Get the magnitude of the results 
new1_mag_spatial_image_3 = abs(new1_mag_spatial_image_2)

subplot(2,2,3)
% Plot the 2D FT, with every other row set to zero 
plot(new1_mag_spatial_image_3)
% Display the MRI data 
imagesc(new1_mag_spatial_image_3);
colormap gray;
axis off;
axis equal;
title('2-D Inverse Discrete Fourier Transform, with every other row set to zero')


% Load the MRI data
load('EBME358_Lab7_MRI_data.mat')
% Set every other row and column of the original data to zero
new2_mag_spatial_image = MRIdata; 
% First set every column to zero
new2_mag_spatial_image(:,1:2:end)=0;
% Then set every other row to zero 
new2_mag_spatial_image_1 = new2_mag_spatial_image;
new2_mag_spatial_image_1(1:2:end,:)=0;

new2_mag_spatial_image_2 = new2_mag_spatial_image_1;

% Find the inverse fourier transform 
new2_mag_spatial_image_3 = ifft2(new2_mag_spatial_image_2);

% Get the magnitude of the results 
new2_mag_spatial_image_4 = abs(new2_mag_spatial_image_3);


subplot(2,2,4)
% Plot the 2D FT, with every other row and column set to zero 
plot(new2_mag_spatial_image_4)
% Display the MRI data 
imagesc(new2_mag_spatial_image_4);
colormap gray;
axis off;
axis equal;
title('2-D Inverse Discrete Fourier Transform, with every other row and column set to zero')
%% QUESTION 3.3

clear 
clc 

% Load the MRI data
load('EBME358_Lab7_MRI_data.mat')

% Create a matrix of zeros, 128 x 128
zero_matrix = zeros(128);

% Finding the center of a 128 x 128 matrix
centerValue = (size(zero_matrix)+1)/2;

% SET THE RADIUS = 50
% Set the rows and column near the center of the matrix to 1
radius = 50;
a = 64.5;
b = 64.5; 
[x1, y1] = meshgrid(1:128);
zero_matrix((x1-a).^2 + (y1-b).^2 < radius^2) = 1;

% Multiply this matrix with the original fourier transform MRI data
new_matrix = zero_matrix .* MRIdata;

% Inverse Transform the matrix to the spatial image 
spatial_image_radius50 = ifft2(new_matrix);

% Get the magintude of the data 
mag_spatial_image_radius50 = abs(spatial_image_radius50);

% Plot the spatial image with radius 50 
figure(8)
plot(mag_spatial_image_radius50)
% Display the MRI data 
imagesc(mag_spatial_image_radius50);
colormap gray;
axis off;
axis equal;
title('Ideal low pass filter with radius = 50')

% SET THE RADIUS = 25
% Set the rows and column near the center of the matrix to 1 

% Create a matrix of zeros, 128x 128
zero_matrix_25 = zeros(128);

radius = 25;
c = 64.5;
d = 64.5;
[x2, y2] = meshgrid(1:128);
zero_matrix_25((x2-c).^2 + (y2-d).^2 < radius^2) = 1;


% Multiply this matrix with the original fourier transform MRI data
new_matrix_25 = zero_matrix_25 .* MRIdata;

% Inverse Transform the matrix to the spatial image 
spatial_image_radius25 = ifft2(new_matrix_25);

% Get the magintude of the data 
mag_spatial_image_radius25 = abs(spatial_image_radius25);

% Plot the spatial image with radius 25
figure(9)
plot(mag_spatial_image_radius25)
% Display the MRI data 
imagesc(mag_spatial_image_radius25);
colormap gray;
axis off;
axis equal;
title('Ideal low pass filter with radius = 25')
%% Question 4


% Load the MRI data
load('EBME358_Lab7_MRI_data.mat')

% Make a matrix of ones the same size as the image 
one_matrix_50 = ones(128);

% Put zeros in a circle in the center with a radius of 50 
radius = 50;
i = 64.5;
j = 64.5;
[x,y] = meshgrid(1:128);
one_matrix_50((x-i).^2 + (y-j).^2 < radius^2) = 0; 

% Multiply the new matrix by the original FT
new_4_matrix_50 = one_matrix_50 .* MRIdata;

% Inverse transform the matrix to the spatial image
spatialimage_r50 = ifft2(new_4_matrix_50);

% Get the magnitude of the data 
mag_spatialimage_r50 = abs(spatialimage_r50);

% RADIUS = 25
% Make a matrix of ones the same size as the image 
one_matrix_25 = ones(128);

% Put zeros in a circle in the center with a radius of 50 
radius = 25;
g = 64.5;
h = 64.5;
[x,y] = meshgrid(1:128);
one_matrix_25((x-g).^2 + (y-h).^2 < radius^2) = 0; 

% Multiply the new matrix by the original FT
new_4_matrix_25 = one_matrix_25 .* MRIdata;

% Inverse transform the matrix to the spatial image
spatialimage_r25 = ifft2(new_4_matrix_25);

% Get the magnitude of the data 
mag_spatialimage_r25 = abs(spatialimage_r25);
% Plot both images in a subplot 
figure(10)
subplot(1,2,1)
plot(mag_spatialimage_r50)
% Display the MRI data 
imagesc(mag_spatialimage_r50);
colormap gray;
axis off;
axis equal;
title('Ideal High pass filter, radius = 50')

subplot(1,2,2)
plot(mag_spatialimage_r25)
% Display the MRI data 
imagesc(mag_spatialimage_r25);
colormap gray;
axis off;
axis equal;
title('Ideal High pass filter, radius = 25')


%% PART 4: LAPLACE TRANSFORMS & TRANSFER FUNCTIONS 
load('EBME358_Lab7_P4.mat')
%Initializing variables 
Fs = 44100;

halffs = Fs/2

dt = 1/Fs;

Tnot= (length(Q2)-1)*dt;

Fnot = 1/Tnot;

timev = 0:dt:Tnot;
fftQ2 = abs(fftshift(fft(Q2)));
figure (11)

plot(timev, Q2)
ylabel("Amplitude")
xlabel("Time(Seconds)")

freqv = -halffs:Fnot:halffs;
figure (12)
plot(freqv,fftQ2)
ylabel("Amplitude")
xlabel("Frequency(Hz)")
%Displaying initial spectrogram 
figure(13)
spectrogram(Q2,[],[],[],Fs,'yaxis')

% Get ride of 5kHz noise
cutoff_freq = 5000; % Frequency to be removed in Hz
bandwidth = 5; % Bandwidth of the filter
[b, a] = butter(2, [(cutoff_freq - bandwidth/2) (cutoff_freq + bandwidth/2)] / (Fs/2), 'stop');

outputAudio = filtfilt(b, a, Q2);

figure(14)
spectrogram(outputAudio,[],[],[],Fs,'yaxis')


% low pass filtering to get ride of high-pitched noise
figure(15)
[num, denom] = butter(6,6000,'low','s');
[dnum, ddenom] = bilinear(num,denom,Fs);
LP_Filter = tf(num,denom);
bodeplot(LP_Filter)
title('BP of Low Pass Filter')


outputAudio = filter(dnum,ddenom,outputAudio);

figure(16)
spectrogram(outputAudio,[],[],[],Fs,'yaxis')

% detrend with linear trend to get ride of high pitch noise at the start
% and the low pitch noise at the end
outputAudio = detrend(outputAudio,1);

figure(17)
spectrogram(outputAudio,[],[],[],Fs,'yaxis')


% high-pass filter to get rid of 0.01 frequency noise
cutoff_freq = 0.01; % Frequency to be removed in Hz
bandwidth = 0.005; % Bandwidth of the filter
[b, a] = butter(1, [(cutoff_freq - bandwidth/2) (cutoff_freq + bandwidth/2)] / (Fs/2), 'stop');
outputAudio = filtfilt(b, a, outputAudio);

figure(18) 
spectrogram(outputAudio,[],[],[],Fs,'yaxis')

% % Play the original and filtered audio
sound(outputAudio, Fs)
