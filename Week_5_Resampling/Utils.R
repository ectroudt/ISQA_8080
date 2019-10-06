##################
# This is a combination of functions for pre-processing 
# the datasets used in ISQA 8720. 
# 
# Author: Dr. Christian Haas
##################

library(dplyr)

# this function prepares the titanic dataset for our course
# it converts some of the integer columns to factors, adds the NA as a level for the 'embarked' variable,
# and relevels the target variable such that the positive class is used correctly in classification
prepare_titanic <- function(titanic, target_variable = 'survived', positive_class = '1'){
  
  # convert variables into factors
  #titanic <- titanic %>% mutate_at(c('sex','pclass','sibsp','parch'), as.factor)
  titanic <- titanic %>% mutate_at(c('survived', 'sex','pclass'), as.factor)

  # then, we convert the 0/1 target variable into no/yes, as 0/1 can lead to problems when using as the dependent variable
  titanic <- titanic %>% mutate(survived, survived = recode(survived, '0' = 'No', '1' = 'Yes'))
  if(target_variable == 'survived'){
    if(positive_class == '1'){
      titanic[[target_variable]] <- relevel(titanic[[target_variable]], 'Yes')
    }
  }
  
  # add the NA as factor level to the embarked variable
  titanic$embarked <- addNA(titanic$embarked)
  titanic$cabin <- addNA(titanic$cabin)
  return(titanic)
}

# this function prepares the heart dataset for our course
# it converts some of the integer columns to factors, adds the NA as a level for the 'Ca' and 'Thal' variables,
# and relevels the target variable such that the positive class is used correctly in classification
prepare_heart <- function(heart, target_variable = 'AHD', positive_class = 'Yes'){
  
  # note that the first column is only the observation index. we can delete this.
  # this is indicated by the column name X1 in R
  if(colnames(heart)[1] == 'X1'){
    heart <- heart %>% select(-c(1))
  }
  
  # as we can see, some variables are actually factors, not numerical
  heart <- heart %>% mutate_at(c("Sex","Fbs","RestECG","ExAng","Slope","Ca"), as.factor)
  
  # also, make sure character vectors are converted to factors
  heart <- heart %>% mutate_if(is.character, as.factor)
  
  heart$Ca <- addNA(heart$Ca, ifany = TRUE)
  heart$Thal <- addNA(heart$Thal, ifany = TRUE)
  
  # relevel the target variable according to the positive class
  heart[[target_variable]] <- relevel(heart[[target_variable]], positive_class)
  
  return(heart)
}
