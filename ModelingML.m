%% Run all models
Data = table2array(readtable("data2.csv"));
%Data(:,1:end-1) = (Data(:,1:end-1) - mean(Data(:,1:end-1)))./(std(Data(:,1:end-1)));
Pre_Trained=1;
%% model tree
tic
[Answer_Train_Tree,Answer_Test_Tree,Answer2_Tree] = ModelTree(Data);
Ctime_tree = toc;
%% model ann
tic
[Answer_Train_ANN,Answer_Test_ANN,Answer2_ANN] = ModelAnn(Data,Pre_Trained);
Ctime_ANN = toc;
%% model M5P
tic
[Answer_Train_M5,Answer_Test_M5,Answer2_M5] = ModelingByM5(Data);
Ctime_M5 = toc;
%% model M5p_GP
tic
[Answer_Train_GP,Answer_Test_GP,Answer2_GP] = ModelingByGP(Data);
Ctime_GP = toc;
%%
Total_Answer = [Answer_Train_M5,Answer_Test_M5,Answer_Train_M5+Answer_Test_M5;Answer_Train_GP,Answer_Test_GP,Answer_Train_GP+Answer_Test_GP;Answer_Train_ANN,Answer_Test_ANN,Answer_Train_ANN+Answer_Test_ANN;Answer_Train_Tree,Answer_Test_Tree,Answer_Train_Tree+Answer_Test_Tree];
save('All results')



