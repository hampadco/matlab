function [CSR, CRR, FS, beta_val, PL] = CalcCSR_CRR(Z, N60, FC, sigma_v, sigma_v_prime, Mw, amax)
%% Compute CSR, CRR, FS, beta, PL using Idriss & Boulanger (2008)
%  Based on formulas from Kumar et al. (2023)
%
%  Inputs (all Nx1 vectors):
%   Z           - depth (m)
%   N60         - SPT blow count
%   FC          - fines content (%)
%   sigma_v     - total vertical stress (kPa)
%   sigma_v_prime - effective vertical stress (kPa)
%   Mw          - earthquake magnitude
%   amax        - peak ground acceleration (g)
%
%  Outputs (all Nx1):
%   CSR, CRR, FS, beta_val, PL

g_const = 1; % amax already in g units

%% Stress reduction factor rd (Idriss & Boulanger 2008)
rd = zeros(size(Z));
for i = 1:numel(Z)
    z = Z(i);
    if z <= 9.15
        rd(i) = 1.0 - 0.00765 * z;
    elseif z <= 23
        rd(i) = 1.174 - 0.0267 * z;
    else
        rd(i) = 0.744 - 0.008 * z;
        rd(i) = max(rd(i), 0.1);
    end
end

%% Magnitude Scaling Factor (MSF) - Idriss (1999)
MSF = 10.^(2.24) ./ (Mw.^2.56);

%% Overburden correction factor Kσ
Pa = 100; % atmospheric pressure in kPa
f_sigma = 1 - 0.005 * min(N60, 37);
f_sigma = max(f_sigma, 0.6);
Ksigma = 1 - f_sigma .* log(max(sigma_v_prime, 1) ./ Pa);
Ksigma = min(Ksigma, 1.1);
Ksigma = max(Ksigma, 0.6);

%% CSR (Eq. 1)
CSR = 0.65 .* (sigma_v ./ max(sigma_v_prime, 1)) .* (amax ./ g_const) .* rd ./ (MSF .* Ksigma);

%% (N1)60 correction: CN factor
CN = sqrt(Pa ./ max(sigma_v_prime, 1));
CN = min(CN, 1.7);
N1_60 = CN .* N60;
N1_60 = min(N1_60, 46);

%% Fines content correction for (N1)60cs (Eq. 3-4)
delta_N = zeros(size(FC));
for i = 1:numel(FC)
    fc = FC(i);
    if fc <= 5
        delta_N(i) = 0;
    elseif fc <= 35
        delta_N(i) = exp(1.63 + 9.7/(fc+0.01) - (15.7/(fc+0.01))^2);
    else
        delta_N(i) = 5.0;
    end
end
N1_60cs = N1_60 + delta_N;
N1_60cs = min(N1_60cs, 46);

%% CRR (Eq. 2) - Idriss & Boulanger 2008
CRR = exp(N1_60cs/14.1 + (N1_60cs/126).^2 - (N1_60cs/23.6).^3 + (N1_60cs/25.4).^4 - 2.8);

%% Factor of Safety (Eq. 5)
FS = CRR ./ max(CSR, eps);

%% Reliability index beta (Eq. 7) and Probability of Liquefaction PL (Eq. 8)
mean_FS = mean(FS);
std_FS = std(FS);

if std_FS > 0
    beta_val = (mean_FS - 1) / std_FS;
else
    beta_val = Inf;
end

PL = 1 - normcdf(beta_val);

end
