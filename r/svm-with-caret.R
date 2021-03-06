library(dplyr)
library(readr)
library(tm)
library(caret)
library(mlr3)
library(tidymodels)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) 

## Load text and convert to DTM ------------------------------------------------

text_df <- readr::read_csv(file = "data/section_1.csv") %>%
  rename(class_label = label)

# Should remove stopwords and apply stemming to the data here

corpus <- tm::Corpus(tm::VectorSource(text_df$summary))

# Nb. By default DocumentTermMatrix() doesnt remove punctuation, has minWordLength=3, and has minTermFreq=1
dtm <- tm::DocumentTermMatrix(corpus,
                              control = list(removePunctuation = TRUE,
                                             tolower = TRUE,
                                             wordLengths = c(1, Inf),
                                             bounds = list(global = c(1, Inf))
                              ))

# The dtm object has various properties eg:
# $nrow : the number of documents
# $ncol : the total number of unique terms/words in the corpus
# $dimnames$Terms : a vector of all the unique terms
# $i,$,j,$v : vectors that map each the frequency(v) of each term(j) in each document(i)

# Basic overview
inspect(dtm)

# Calculate Term frequency using {slam}
term_freq <- slam::col_sums(dtm) %>% as.data.frame()
  arrange(desc(frequency))

# Calculate words per document
doc_lengths <- slam::row_sums(dtm) %>% as.data.frame()


## Prepare for training --------------------------------------------------------
feature_matrix <- as.data.frame(as.matrix(dtm)) # convert dtm to datframe

labels <- select(text_df, class_label)


# Train SVM Model with {caret}
# Caret Controls
tr_cont <- caret::trainControl(method = "repeatedcv", 
                               number = 5,  
                               repeats = 3,
                               verboseIter = TRUE,
                               savePredictions = TRUE)
# Train Model
set.seed(3537185) # Seed must be here for reproducibility
fit <- caret::train(x = feature_matrix,
                    y = as.factor(labels$class_label),
                    method = "svmLinear3",  # "svmLinear" "gbm"  "rpart2"
                    metric = "Accuracy",
                    trControl = tr_cont,
                    tuneLength = 4 # Different costs are used to tune eg. 0.25, 0.5....
                    )
fit
# The training object has various properties eg:
# $results : shows the cost, Loss and accuracy from training  
# $pred : shows all the predictions vs actuals for each fold&repeat
# $finalModel : Details and parameter weights for the best model

test<-fit$pred %>% filter(cost == fit$bestTune[[1]], Loss == fit$bestTune[[2]])

cm <- caret::confusionMatrix(test$pred, test$obs)

prediction <- predict(fit, feature_matrix)
caret::confusionMatrix(prediction, as.factor(labels$class_label))

# cm_info <- data.frame("Cost" = c,
#                       "Loss" = l,
#                       "Accuracy" = cm$overall["Accuracy"],
#                       "Sensitivity" = cm$byClass["Sensitivity"],
#                       "Specificity" = cm$byClass["Specificity"],
#                       "Precision" = cm$byClass["Precision"],
#                       "F1" = cm$byClass["F1"],
#                       "True_Negative" = cm$table[1,1],
#                       "False_Negative" = cm$table[2,1],
#                       "False_Positive" = cm$table[1,2],
#                       "True_Positive" = cm$table[2,2])


