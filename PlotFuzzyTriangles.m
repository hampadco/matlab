function PlotFuzzyTriangles(FuzzyResults)
%% Plot fuzzy triangles for each performance metric.
%  Similar to Figs. 5-7 in Ghasemi & Derakhshani (2021)
%  Each subplot: X = metric value, Y = alpha level, 3 models overlaid

alpha_levels = FuzzyResults.alpha_levels;
metric_names = FuzzyResults.metric_names;
model_names  = FuzzyResults.model_names;
results_min  = FuzzyResults.results_min;
results_max  = FuzzyResults.results_max;

n_metrics = numel(metric_names);
n_models  = numel(model_names);

colors = {'b', 'r', [0 0.6 0]};
line_styles = {'-', '--', '-.'};

figure('Position', [50 50 1400 900], 'Name', 'Fuzzy Triangles - Performance Metrics');

for k = 1:n_metrics
    subplot(2, 3, k);
    hold on;

    for m = 1:n_models
        min_vals = results_min(:, k, m);
        max_vals = results_max(:, k, m);

        %% Build closed polygon for the fuzzy triangle
        x_poly = [min_vals; flipud(max_vals)];
        y_poly = [alpha_levels(:); flipud(alpha_levels(:))];

        fill(x_poly, y_poly, colors{m}, 'FaceAlpha', 0.1, 'EdgeColor', colors{m}, ...
             'LineWidth', 1.5, 'LineStyle', line_styles{m});

        plot(min_vals, alpha_levels, 'Color', colors{m}, 'LineWidth', 1.5, 'LineStyle', line_styles{m});
        plot(max_vals, alpha_levels, 'Color', colors{m}, 'LineWidth', 1.5, 'LineStyle', line_styles{m});
    end

    xlabel(metric_names{k}, 'FontSize', 11);
    ylabel('\alpha', 'FontSize', 12);
    title(['Fuzzy Triangle: ', metric_names{k}], 'FontSize', 12);
    legend(model_names{:}, 'Location', 'best');
    grid on;
    ylim([0, 1]);
    hold off;
end

sgtitle('Fuzzy Uncertainty Analysis of Performance Metrics', 'FontSize', 14, 'FontWeight', 'bold');
print('FuzzyTriangles_Metrics', '-dpng', '-r300');

%% Compute and display support widths
fprintf('\n========== FUZZY TRIANGLE SUPPORT WIDTHS ==========\n');
fprintf('(Width at alpha=0: narrower = more robust)\n\n');
fprintf('%-10s', 'Model');
for k = 1:n_metrics
    fprintf('%12s', metric_names{k});
end
fprintf('\n');

alpha0_idx = find(alpha_levels == 0, 1);
if isempty(alpha0_idx)
    alpha0_idx = numel(alpha_levels);
end

for m = 1:n_models
    fprintf('%-10s', model_names{m});
    for k = 1:n_metrics
        width = results_max(alpha0_idx, k, m) - results_min(alpha0_idx, k, m);
        fprintf('%12.4f', width);
    end
    fprintf('\n');
end

end
