library(tm)
library(tidyverse)
library(caret)
library(purrr)
library(tibble)
library(dplyr)
db <- read.csv("/Users/riccardocervero/Desktop/news_sample.csv")
db.train <- read.csv("/Users/riccardocervero/Desktop/train1.csv")
db.test <- anti_join(db,db.train)
VAR <- c('title','content','publication','section','category')
db <- db[,VAR]
db.train <- db.train[,VAR]
db.test <- db.test[,VAR]
n <- nrow(db)
## shuffle delle righe
db.train <- db.train[sample(nrow(db.train),replace = F),]
db.test <- db.test[sample(nrow(db.test),replace = F),]
dim(db.train)
View(db)
View(db.train)
## filtra il dataset in base alla categoria
categories <- unique(db$section)
get_per_category <- function(category, var)
  db.train[db.train$section == category, var, drop = TRUE]
## restituisce la matrice della frequenza delle parole all'interno dei
## dati, escludendo le parole con frequenza < 2
get_matrix_words <- function(data, min_freq=2) {
  tdm <- as.matrix(TermDocumentMatrix(Corpus(VectorSource(data)),
                                      control = list(
                                        stopwords = TRUE,
                                        removePunctuation = TRUE,
                                        removeNumbers = TRUE)))
  tdm.sums <- c(tdm %*% rep(1, ncol(tdm)))
  names(tdm.sums) <- rownames(tdm)
  return(tdm.sums[which(tdm.sums >= min_freq)])
}
## al modello si passa una matrice, non una lista
list_to_dataframe <- function(l) {
  words <- unique(names(flatten(l)))
  out <- map_dfc(names(l), function(category) {
    out.cat <- map_dbl(words, ~ifelse(is.na(l[[category]][.x]),
                                      0,
                                      l[[category]][.x]))
    out.cat <- out.cat / sum(out.cat)
  })
  out <- as.matrix(out)
  colnames(out) <- names(l)
  rownames(out) <- words
  out[is.nan(out)] <- 0
  return(out)
}
## combina tutte le funzioni precedenti per ottenere i dati
get_matrix <- function(var, data) {
  txts <- map(categories, ~get_per_category(.x, var))
  names(txts) <- categories
  
  matrix_per_var <- map(txts, get_matrix_words)
  names(matrix_per_var) <- names(txts)
  
  return(list_to_dataframe(matrix_per_var))
}
## le matrici headlines_matrix <- get_matrix("title", db.train)
## e content_matrix <- get_matrix("content", db.train)
## son troppo pesanti per poter essere allocate normalmente, per cui sono
## richiamate più avanti in modo tale da evitare allocamenti inutili in RAM

## probabilità a priori della distribuzione uniforme
prior <- 1 / length(categories)
## costruiamo il modello (per una sola categoria)
bayes.model <- function(prob, data.train, category, min_prob=1e-20) {
  ## training: la probabilità è "deformata" in base alla probabilità
  ## data dalle osservazioni, calcolata però con un calcolo matriciale
  ## per risparmiare tempo
  words <- names(data.train[data.train[, category] > 0, category])
  probs <- c(prior * (data.train[words, category] /
                        data.train[words, ] %*% rep(1, ncol(data.train))))
  names(probs) <- words
  ## restituisce la funzione di previsione
  predict <- function(w) {
    ifelse(probs[w] == 0 || is.na(probs[w]), min_prob, probs[w])
  }
}
## costruisce il predittore 
predictor <- function(prior, data) {
  ## costruisce un modello per ogni categoria usando gli stessi dati
  predictor.categories <- map(categories,
                              ~bayes.model(prior, data, .x))
  names(predictor.categories) <- categories
  ## ritorna la funzione di previsione
  predict <- function(txt) {
    ## il testo è esploso
    txt.splitted <- strsplit(as.character(txt), " ")[[1]]
    results <- rep(prior, length(predictor.categories))
    names(results) <- names(predictor.categories)
    for (word in txt.splitted) {
      results.new <- sapply(predictor.categories, function(fn) fn(word))
      results <- results * results.new
      results <- results / sum(results)
      if (any(results == 1)) return(results)
    }
    return(results)
  }
}
## Ogni closure rimane isolata dalle altre: ogni predittore
## per tipo (titolo o contenuto) usa dei predittori per singola
## categoria (i tipi di articolo) isolati tra di loro

## semplicemente prende la classe più probabile
predict.obs <- function(predictor, txt)
  names(which.max(predictor(txt)))

## come sopra, ma effettua in batch per un vettore
predict.vector <- function(predictor, v)
  sapply(v, function(x) predict.obs(predictor, x))

## costruiamo e alleniamo i predittori
headlines.predictor <- predictor(prior, get_matrix("title", db.train))
content.predictor <- predictor(prior, get_matrix("content", db.train))

## parte per l'analisi delle prestazioni del classificatore, si può
## eliminare tranquillamente in futuro
results.headline <- mean(predict.vector(headlines.predictor,
                                        db.test$title)
                         == db.test$section)
results.content <- mean(predict.vector(content.predictor,
                                       db.test$content)
                        == db.test$section)
results.headline
results.content

## parte di Machine Learning puro
## input: the row dataset from the .csv file
## output: the dataset ready for the ML algorithm
## Sarebbe stato necessario inserire la matrice con le probabilità grezze per
## ogni categoria e non solamente la categoria più probabile, ma si è scelto di
## non farlo per non appesantire ulteriormente la complessità computazionale 
##dell'algoritmo, che non potrebbe comunque dare particolari risultati,
##perché le probabilità tendono quasi sempre a 0 o 1, evitando la maledizione della dimensionalità.
## Basta dunque solamente mettere qualcosa per bilanciare titolo e contenuto.
prepare.dataset <- function(data,
                            predictor.headlines,
                            predictor.contents) {
  tibble(section = data$section,
         publication = data$publication,
         category = data$category,
         title = predict.vector(predictor.headlines,
                                as.character(data$title)),
         content = predict.vector(predictor.contents,
                                  as.character(data$content)))
}
## prepara le matrici per l'algoritmo
test.set <- prepare.dataset(db.test,
                            headlines.predictor,
                            content.predictor)
write.csv(test.set,
          file = "/Users/riccardocervero/Desktop/testset.csv")

train.set <- prepare.dataset(db.train,
                             headlines.predictor,
                             content.predictor)
write.csv(train.set,
          file = "/Users/riccardocervero/Desktop/trainset.csv")

## Modello Recurive Partitioning, 
## modello ricorsivo ad albero decisionale 
##che ha un maggior rischio di overfitting rispetto a RF (73.92431%)
model.rpart <- train(section ~ publication + title + content,
                  data = train.set,
                  method = "rpart",
                  na.action = na.omit)

## Modello Random Forest (84.82806%)
model.rf <- train(section ~ publication + title + content,
                  data = train.set,
                  method = "rf",
                  na.action = na.omit)

## Modello C5.0 (84.74166%)
model.c <- train(section ~ publication + title + content,
                  data = train.set,
                  method = "C5.0",
                  na.action = na.omit)

## Modello SVM (84.41334%)
## http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf
## https://www.hvitfeldt.me/blog/binary-text-classification-with-tidytext-and-caret/#svm
# È un SVM lineare per classificazione multiclasse 
model.svm <- train(section ~ publication + title + content,
                   data = train.set,
                   method = "svmLinear3",
                   na.action = na.omit)

## Modello Logistic Boost (86.11929%)
model.lb <- train(section ~ publication + title + content,
                   data = train.set,
                   method = "LogitBoost",
                   na.action = na.omit)

## mostra i risultati
test.set$results.rpart <- predict(model.rpart, newdata = test.set)
test.set$results.rf <- predict(model.rf, newdata = test.set)
test.set$results.c <- predict(model.c, newdata = test.set)
test.set$results.svm <- predict(model.svm, newdata = test.set)
test.set$results.lb <- predict(model.lb, newdata = test.set)
## Calcolo dell'accuratezza
mean(test.set$results.rpart == test.set$section)
mean(test.set$results.rf == test.set$section)
mean(test.set$results.c == test.set$section)
mean(test.set$results.svm == test.set$section)
mean(test.set$results.lb == test.set$section,na.rm=TRUE)

##Cross Validation con resampling stratificato
cv <- function(db,iter=3,method) {
  i=1
  while (i<iter+1) {
    split_data <- function(db,SplitRatio=0.7) {
      db$split = caTools::sample.split(db$section,SplitRatio = SplitRatio)
      train=subset(db, db$split==TRUE)
      test=subset(db, db$split==FALSE)
      #Shuffle
      db.train <- train[sample(nrow(train),replace = F),]
      db.test <- test[sample(nrow(test),replace = F),]
      return(list(db.train=db.train,
                  db.test=db.test))
    }
    spl_data <- split_data(db)
    VAR <- c('title','content','publication','section','category')
    db.train <- spl_data$db.train[,VAR]
    db.test <- spl_data$db.test[,VAR]
    #Addestra il modello
    headlines.predictor <- predictor(prior, get_matrix("title", db.train))
    content.predictor <- predictor(prior, get_matrix("content", db.train))
    ##Prepara le matrici per l'algoritmo
    test.set <- prepare.dataset(db.test,headlines.predictor,content.predictor)
    train.set <- prepare.dataset(db.train,headlines.predictor,content.predictor)
    ##Addestra il modello di ML
    model <- train(section ~ publication + title + content,
                   data = train.set,
                   method = method,
                   na.action = na.omit)
    ##Calcola il risultato
    test.set$results.model <- predict(model, newdata = test.set,drop.unused.levels = TRUE)
    res <- vector()
    res[i] = mean(test.set$results.model == test.set$section,na.rm=TRUE)
    print("Completed iteration")
    i=i+1
  }
  return(list(accuracy=mean(res,na.rm=TRUE)))
}

rpart = cv(db,method = "rpart")
rf = cv(db,method = "rf")
c05 = cv(db=db,method = "C5.0")
svm = cv(db=db,method = "svmLinear3")
lb = cv(db=db,method = "LogitBoost")

save.image(file = "/Users/riccardocervero/Desktop/completed.RData")