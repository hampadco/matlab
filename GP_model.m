function [y] = GP_model(Data,lm)
   if lm==1
       y = LM_1(Data(:,1), Data(:,2), Data(:,3), Data(:,4), Data(:,5), Data(:,6), Data(:,7));
   elseif lm==2
       y = LM_2(Data(:,1), Data(:,2), Data(:,3), Data(:,4), Data(:,5), Data(:,6), Data(:,7));
   elseif lm==3
       y = LM_3(Data(:,1), Data(:,2), Data(:,3), Data(:,4), Data(:,5), Data(:,6), Data(:,7));
   elseif lm==4
       y = LM_4(Data(:,1), Data(:,2), Data(:,3), Data(:,4), Data(:,5), Data(:,6), Data(:,7));
   elseif lm==5
       y = LM_5(Data(:,1), Data(:,2), Data(:,3), Data(:,4), Data(:,5), Data(:,6), Data(:,7));
   elseif lm==6
       y = LM_6(Data(:,1), Data(:,2), Data(:,3), Data(:,4), Data(:,5), Data(:,6), Data(:,7));
   elseif lm==7
       y = LM_7(Data(:,1), Data(:,2), Data(:,3), Data(:,4), Data(:,5), Data(:,6), Data(:,7));
   elseif lm==8
       y = LM_8(Data(:,1), Data(:,2), Data(:,3), Data(:,4), Data(:,5), Data(:,6), Data(:,7));
   elseif lm==9
       y = LM_9(Data(:,1), Data(:,2), Data(:,3), Data(:,4), Data(:,5), Data(:,6), Data(:,7));
   elseif lm==10
       y = LM_10(Data(:,1), Data(:,2), Data(:,3), Data(:,4), Data(:,5), Data(:,6), Data(:,7));
   elseif lm==11
       y = LM_11(Data(:,1), Data(:,2), Data(:,3), Data(:,4), Data(:,5), Data(:,6), Data(:,7));
   elseif lm==12
       y = LM_12(Data(:,1), Data(:,2), Data(:,3), Data(:,4), Data(:,5), Data(:,6), Data(:,7));
end