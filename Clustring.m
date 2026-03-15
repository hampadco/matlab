%%
clear
clc
%%
Data = table2array(readtable("date (2).csv"));
% Clustring
k = 3;
[idx, C, sumd, D] = kmeans(Data, k);
%%
Data1=Data(idx == 1,:);% Cluster1
Data2=Data(idx == 2,:);% Cluster 2
Data3=Data(idx == 3,:);% Cluster 3
%% Convert array to table 
T1 = array2table(Data1, 'VariableNames', {'Z'	'N60'	'F'	'Si'	'Si1'	'Mw'	'a'	'L'});
T2 = array2table(Data2, 'VariableNames', {'Z'	'N60'	'F'	'Si'	'Si1'	'Mw'	'a'	'L'});
T3 = array2table(Data3, 'VariableNames', {'Z'	'N60'	'F'	'Si'	'Si1'	'Mw'	'a'	'L'});
%% Write data to excel
writetable(T1, 'date_cluster1.csv');
writetable(T2, 'date_cluster2.csv');
writetable(T3, 'date_cluster3.csv');