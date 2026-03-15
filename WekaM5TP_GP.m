function LM = WekaM5TP_GP(N60, F, a, Z, Si,Si1, Mw)
%%
if N60 <= 22.2  
   if F <= 93.5  
      if a <= 0.195  
          LM=1;
          elseif a >  0.195  
         if F <= 71.5  
             LM=2;
             elseif F >  71.5  
            if Si1 <= 85.8  
               if Si <= 91.55  
                  if Z <= 1.55  
                      LM=3;
                      elseif Z >  1.55  
                     if Si1 <= 33.85  
                         LM=4;
                         elseif Si1 >  33.85  
                             LM=5;
end
end
               elseif Si >  91.55  
                   LM=6;
end
            elseif Si1 >  85.8  
                LM=7;
end
end
end
   elseif F >  93.5  
      if Si <= 167.05  
          LM=8;
          elseif Si >  167.05  
              LM=9;
end
end
elseif N60 >  22.2  
   if N60 <= 28.5  
      if a <= 0.335  
          LM=10;
          elseif a >  0.335  
              LM=11;
end
   elseif N60 >  28.5  
       LM=12;
end
end
end