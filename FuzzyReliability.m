function FuzzyRelResults = FuzzyReliability(Data, N_samples)
%% Fuzzy Reliability Analysis: Computes fuzzy beta and PL using alpha-cuts
%  Approach: Use fuzzy inputs (COV-based) to compute CSR, CRR, FS at each
%  alpha level, then derive beta and PL as fuzzy triangles.
%
%  Data:      Nx8 matrix [Z, N60, F, Si, Si1, Mw, a, L]
%  N_samples: Monte Carlo samples per alpha level (default: 100)

if nargin < 2
    N_samples = 100;
end

%% COV values from Kumar et al. (2023)
COV = [0, 0.4, 0.35, 0.2, 0.2, 0.1, 0.2];

alpha_levels = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0];
n_alpha = numel(alpha_levels);

%% Storage for fuzzy beta and PL
beta_min = zeros(n_alpha, 1);
beta_max = zeros(n_alpha, 1);
PL_min = zeros(n_alpha, 1);
PL_max = zeros(n_alpha, 1);

%% Also store per-sample FS statistics for each alpha
FS_mean_min = zeros(n_alpha, 1);
FS_mean_max = zeros(n_alpha, 1);

%% Main loop over alpha levels
for a_idx = 1:n_alpha
    alpha = alpha_levels(a_idx);
    fprintf('Reliability: alpha = %.1f (%d/%d)...\n', alpha, a_idx, n_alpha);

    delta_frac = (1 - alpha);

    if delta_frac == 0
        %% Crisp computation
        [~, ~, FS, b, p] = CalcCSR_CRR(Data(:,1), Data(:,2), Data(:,3), ...
            Data(:,4), Data(:,5), Data(:,6), Data(:,7));
        beta_min(a_idx) = b;
        beta_max(a_idx) = b;
        PL_min(a_idx) = p;
        PL_max(a_idx) = p;
        FS_mean_min(a_idx) = mean(FS);
        FS_mean_max(a_idx) = mean(FS);
    else
        temp_beta = zeros(N_samples, 1);
        temp_PL = zeros(N_samples, 1);
        temp_FS_mean = zeros(N_samples, 1);

        for s = 1:N_samples
            Data_p = Data;
            for col = 1:7
                if COV(col) > 0
                    delta = delta_frac * COV(col) * Data(:, col);
                    noise = -1 + 2*rand(size(Data,1), 1);
                    Data_p(:, col) = Data(:, col) + delta .* noise;
                    Data_p(:, col) = max(Data_p(:, col), 0);
                end
            end

            [~, ~, FS, b, p] = CalcCSR_CRR(Data_p(:,1), Data_p(:,2), Data_p(:,3), ...
                Data_p(:,4), Data_p(:,5), Data_p(:,6), Data_p(:,7));
            temp_beta(s) = b;
            temp_PL(s) = p;
            temp_FS_mean(s) = mean(FS);
        end

        beta_min(a_idx) = min(temp_beta);
        beta_max(a_idx) = max(temp_beta);
        PL_min(a_idx) = min(temp_PL);
        PL_max(a_idx) = max(temp_PL);
        FS_mean_min(a_idx) = min(temp_FS_mean);
        FS_mean_max(a_idx) = max(temp_FS_mean);
    end
end

%% Package results
FuzzyRelResults.alpha_levels = alpha_levels;
FuzzyRelResults.beta_min = beta_min;
FuzzyRelResults.beta_max = beta_max;
FuzzyRelResults.PL_min = PL_min;
FuzzyRelResults.PL_max = PL_max;
FuzzyRelResults.FS_mean_min = FS_mean_min;
FuzzyRelResults.FS_mean_max = FS_mean_max;
FuzzyRelResults.COV = COV;

%% Compare with actual liquefaction labels
L_actual = Data(:, end);
actual_liq_ratio = sum(L_actual == 1) / numel(L_actual);
[~, ~, ~, beta_crisp, PL_crisp] = CalcCSR_CRR(Data(:,1), Data(:,2), Data(:,3), ...
    Data(:,4), Data(:,5), Data(:,6), Data(:,7));

fprintf('\n========== RELIABILITY RESULTS ==========\n');
fprintf('Crisp beta = %.4f\n', beta_crisp);
fprintf('Crisp PL   = %.4f\n', PL_crisp);
fprintf('Actual liquefaction ratio = %.4f\n', actual_liq_ratio);
fprintf('Fuzzy beta range (alpha=0): [%.4f, %.4f]\n', beta_min(end), beta_max(end));
fprintf('Fuzzy PL range (alpha=0):   [%.4f, %.4f]\n', PL_min(end), PL_max(end));

%% Plot Fuzzy Triangles for beta and PL
figure('Position', [100 100 1200 500], 'Name', 'Fuzzy Reliability');

subplot(1,3,1);
x_poly = [beta_min; flipud(beta_max)];
y_poly = [alpha_levels(:); flipud(alpha_levels(:))];
fill(x_poly, y_poly, 'b', 'FaceAlpha', 0.15, 'EdgeColor', 'b', 'LineWidth', 1.5);
hold on;
plot(beta_min, alpha_levels, 'b-', 'LineWidth', 1.5);
plot(beta_max, alpha_levels, 'b-', 'LineWidth', 1.5);
xlabel('\beta (Reliability Index)');
ylabel('\alpha');
title('Fuzzy Triangle: \beta');
grid on; ylim([0 1]);

subplot(1,3,2);
x_poly = [PL_min; flipud(PL_max)];
y_poly = [alpha_levels(:); flipud(alpha_levels(:))];
fill(x_poly, y_poly, 'r', 'FaceAlpha', 0.15, 'EdgeColor', 'r', 'LineWidth', 1.5);
hold on;
plot(PL_min, alpha_levels, 'r-', 'LineWidth', 1.5);
plot(PL_max, alpha_levels, 'r-', 'LineWidth', 1.5);
xline(actual_liq_ratio, 'k--', 'LineWidth', 1.5);
xlabel('P_L (Probability of Liquefaction)');
ylabel('\alpha');
title('Fuzzy Triangle: P_L');
legend('Fuzzy PL', '', '', 'Actual Ratio', 'Location', 'best');
grid on; ylim([0 1]);

subplot(1,3,3);
x_poly = [FS_mean_min; flipud(FS_mean_max)];
y_poly = [alpha_levels(:); flipud(alpha_levels(:))];
fill(x_poly, y_poly, [0 0.6 0], 'FaceAlpha', 0.15, 'EdgeColor', [0 0.6 0], 'LineWidth', 1.5);
hold on;
plot(FS_mean_min, alpha_levels, '-', 'Color', [0 0.6 0], 'LineWidth', 1.5);
plot(FS_mean_max, alpha_levels, '-', 'Color', [0 0.6 0], 'LineWidth', 1.5);
xline(1, 'k--', 'LineWidth', 1);
xlabel('Mean FS');
ylabel('\alpha');
title('Fuzzy Triangle: Mean Factor of Safety');
legend('Fuzzy FS', '', '', 'FS=1', 'Location', 'best');
grid on; ylim([0 1]);

sgtitle('Fuzzy Reliability Analysis', 'FontSize', 14, 'FontWeight', 'bold');
print('FuzzyReliability_Plots', '-dpng', '-r300');

save('FuzzyReliability_Results.mat', 'FuzzyRelResults');
fprintf('Fuzzy reliability results saved to FuzzyReliability_Results.mat\n');

end
