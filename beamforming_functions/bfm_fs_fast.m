function foc_data = bfm_fs_fast(t, signal, foc_pts, ...
    rxAptPos, txAptPos, dc_rx, dc_tx, speed_of_sound)
% BFM_FS - Beamforms (Delay and Sum) the RF data at desired locations
%
% The function interpolates the RF signals collected using the full synthetic sequence
% to focus the data at desired locations
%
% INPUTS:
% t                  - T x 1 time vector for samples of the input signal
% signal             - T x N x M matrix containing input RF data to be interpolated
% foc_pts            - P x 3 matrix with position of focal points [m]
% rxAptPos           - N x 3 matrix with positions of the Rx apertures (elements) [m]
% txAptPos           - M x 3 matrix with positions of the Tx apertures (elements) [m]
% dc_rx, dc_tx       - time offsets [s] for Tx and Rx; scalars, N (M) x 1 vectors, or P x N (M) matrix
% speed_of_sound     - speed of sounds [m/s]; default 1540 m/s
%
% OUTPUT:
% foc_data - P x 1 vector with interpolated (RF) data points
%

% time from the focus to receive  apertures (array elements)
rx_times = calc_times(foc_pts,rxAptPos,dc_rx,speed_of_sound);

% time from the transmit apertures (array elements) to focus
tx_times = calc_times(foc_pts,txAptPos,dc_tx,speed_of_sound);

% focused but not summed rf data
foc_data=zeros(size(foc_pts,1), 1);
for i=1:size(rx_times,2)
    for j=1:size(tx_times,2)
        % Simple 1D Interp From MATLAB
        foc_data = foc_data + ...
            interp1(t,signal(:,i,j),rx_times(:,i)+tx_times(:,j),'spline',0);
    end
    disp(['Rx Elem ', num2str(i)]);
end