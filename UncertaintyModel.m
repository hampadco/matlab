%% uncertainty Analysis
%% Run all models
Data = table2array(readtable("data2.csv"));
Total_Answer2 = zeros(8,6);
Total_Answer = zeros(8,6);
Computation_Time = zeros(4,1);
count = 0;
for i=1:100
    try
I = randperm(size(Data,1),size(Data,1));
Data2 = Data(I,:);
Pre_Trained=1;
%% model tree
tic
[Answer_Train_Tree,Answer_Test_Tree,Answer2_Tree] = ModelTree(Data2);
Ctime_tree = toc;
close all
%% model ann
tic
[Answer_Train_ANN,Answer_Test_ANN,Answer2_ANN] = ModelAnn(Data2,Pre_Trained);
Ctime_ANN = toc;
close all
%% model M5P
tic
[Answer_Train_M5,Answer_Test_M5,Answer2_M5] = ModelingByM5(Data2);
Ctime_M5 = toc;
close all
%% model M5p_GP
tic
[Answer_Train_GP,Answer_Test_GP,Answer2_GP] = ModelingByGP(Data2);
Ctime_GP = toc;
close all
%%
Total_Answer = [Answer_Train_M5,Answer_Test_M5,Answer_Train_M5+Answer_Test_M5;Answer_Train_GP,Answer_Test_GP,Answer_Train_GP+Answer_Test_GP;Answer_Train_ANN,Answer_Test_ANN,Answer_Train_ANN+Answer_Test_ANN;Answer_Train_Tree,Answer_Test_Tree,Answer_Train_Tree+Answer_Test_Tree];
Computation_Time = Computation_Time+[Ctime_tree,Ctime_ANN,Ctime_M5,Ctime_GP];
    catch
    end
Total_Answer2 = Total_Answer2 + Total_Answer;
count = count + 1;
end
Total_Answer2 = Total_Answer2/count;
Computation_Time = Computation_Time/100;
save('Uncertainty results')



