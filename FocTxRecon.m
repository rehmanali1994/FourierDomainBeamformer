clear
clc

% Setup path
addpath(genpath(pwd));

% Load Channel Data
load('FocTxDataset.mat'); % Point Targets and Lesions Phantom

% Get All Dimension Information
[nt, nRx, nTx] = size(rcvdata);

% Points to Focus and Get Image At
dBrange = [-80, 0]; c = 1540;
num_r = 600; rlims = [4e-3, 36e-3];
r_img = linspace(rlims(1), rlims(2), num_r);

% Focused Transmit Image Reconstruction
tx_origin_x = txBeamOrigins(:,1); % Transmit Beam Origin [m]
img_h = zeros(numel(r_img), numel(tx_origin_x));
disp('Beginning Dynamic Receive Beamforming');
for kk = 1:numel(tx_origin_x)
    X = tx_origin_x(kk)*ones(size(r_img));
    Y = zeros(size(X)); Z = r_img;
    foc_pts = [X(:), Y(:), Z(:)];
    focData = focus_data(t, rcvdata(:,:,kk), ...
        foc_pts, rxAptPos, txBeamOrigins(kk,:), 0, c);
    img_h(:, kk) = sum(focData,2);
    disp(['Image line at x = ', num2str(tx_origin_x(kk))]);
end 

% Upsample the Image
upsamp_x = 4; upsamp_z = 2;
x_img = tx_origin_x(1):(diff(tx_origin_x)/upsamp_x):tx_origin_x(end);
z_img = r_img(1):(diff(r_img)/upsamp_z):r_img(end);
[TX_ORIG_X, R_IMG] = meshgrid(tx_origin_x, r_img);
[X_IMG, Z_IMG] = meshgrid(x_img, z_img);
IMG = max(interp2(TX_ORIG_X,R_IMG,abs(img_h),X_IMG,Z_IMG,'spline',0),0);

% Show Scan Converted Image
imagesc(1000*x_img, 1000*z_img, ...
    20*log10(IMG/max(IMG(:))), dBrange); 
axis image; xlabel('Lateral [mm]'); ylabel('Axial [mm]');
title('DAS Beamforming'); colormap(gray); colorbar();