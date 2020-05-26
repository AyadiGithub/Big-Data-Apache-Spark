#Pipelines give a predefined/recipe steps for data pre-processing or/and feature engineering. 
#1) string indexing - giving strings an index (from 0, to .... n)

#2) Data Scaling/Normalization (making data range -1 to +1 or 0 to +1). 
#This is done to make sure each feature/column/dimension has the same value range

#3)One Hot Enconding - Transforming a single column with multiple class numbers 
#into multiple columns with only binary class numbers. 

#4)With pipelines, the pre-processing is saved and the different ML algos can be applied to it.

#5) fit() - starts training | evaluate() - validating | score() - gives predicted value

