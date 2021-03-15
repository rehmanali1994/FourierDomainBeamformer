function foc_data = focus_data(t,signal,foci_rx,elempos,tx_center,dc,speed_of_sound)
%
% FOCUS_DATA - Focuses the RF data at desired locations
%
% The function interpolates the RF signals received on individual array elements
% to focus the data at desired locations.
%
% INPUTS:
% t                  - T x 1 time vector for samples of the input signal
% signal             - T x N matrix containing input RF data to be interpolated
% foci_rx            - M x 3 matrix with position of Rx focal points of interest [m]
% elempos            - N x 3 matrix with element positions [m]
% tx_center          - 1 x 3 vector with the position of the center of the Tx aperture
%                    - (Tx center); [0,0,0] by default  
% dc                 - time offset [s]; scalar, N x 1 vector, M x N matrix
% speed_of_sound     - speed of sounds [m/s]; default 1540 m/s
%
% OUTPUT:
% foc_data - M x N vector with interpolated (RF) data points
%
% NOTE: Intended to focus the data from diverging (transmit) waves or
% to focus the data along the transmit beam (a single A-line per Tx event)
%

tx_center = [tx_center(:)]';

% time from the Rx focus to array elements
rx_times = calc_times(foci_rx,elempos,dc,speed_of_sound);

% time from the array elements to Rx focus
tx_distance = foci_rx-repmat(tx_center,size(foci_rx,1),1);
tx_times = repmat(sqrt(sum(tx_distance.^2,2))/speed_of_sound,1,size(elempos,1));

% two-way delays
foc_times = rx_times + tx_times;

% interpolate focused result on grid
elmts = 1:size(elempos,1);
[E, T] = meshgrid(elmts, t);
foc_data = interp2(E, T, signal, ...
    ones(size(foci_rx,1),1)*elmts, foc_times, 'linear', 0);

