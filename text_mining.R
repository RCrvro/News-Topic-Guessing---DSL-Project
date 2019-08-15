library(tm)
library(tidyverse)
library(caret)
library(purrr)
library(tibble)
library(dplyr)

## Filtra il dataset in base alla categoria
categories <- unique(db$section)
get_per_category <- function(data, category, var)
  data[data$section == category, var, drop = TRUE]
## Restituisce la matrice della frequenza delle parole all'interno dei
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
## Trasformazione della lista in matrice, da passare al modello
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
## Combinazione di tutte le funzioni precedenti per ottenere i dati
## su cui allenare i modelli
get_matrix <- function(var, data) {
  txts <- map(categories, ~get_per_category(data, .x, var))
  names(txts) <- categories
  
  matrix_per_var <- map(txts, get_matrix_words)
  names(matrix_per_var) <- names(txts)
  
  return(list_to_dataframe(matrix_per_var))
}
## Le matrici headlines_matrix e content_matrix son troppo pesanti per poter 
## essere allocate normalmente, per cui sono richiamate più avanti in modo tale
## da evitare allocamenti inutili in RAM.

## Probabilità a priori della distribuzione uniforme
prior <- 1 / length(categories)
## Costruiamo il modello (per una sola categoria)
bayes.model <- function(prob, data.train, category, min_prob=1e-20) {
  ## Nella fase di training, la probabilità è "deformata" in base alla probabilità
  ## data dalle osservazioni, ottenuta con un calcolo matriciale
  ## per risparmiare tempo
  words <- names(data.train[data.train[, category] > 0, category])
  probs <- c(prior * (data.train[words, category] /
                        data.train[words, ] %*% rep(1, ncol(data.train))))
  names(probs) <- words
  ## Restituisce la funzione di previsione
  predict <- function(w) {
    ifelse(probs[w] == 0 || is.na(probs[w]), min_prob, probs[w])
  }
}
## Costruisce il predittore 
predictor <- function(prior, data) {
  ## Costruisce un modello per ogni categoria usando gli stessi dati
  predictor.categories <- map(categories,
                              ~bayes.model(prior, data, .x))
  names(predictor.categories) <- categories
  ## Restituisce la funzione di previsione
  predict <- function(txt) {
    ## Il testo viene esploso
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

predictor.tracked <- function(prior, data) {
  ## Costruisce un modello per ogni categoria usando gli stessi dati
  predictor.categories <- map(categories,
                              ~bayes.model(prior, data, .x))
  names(predictor.categories) <- categories
  ## Restituisce la funzione di previsione
  predict <- function(txt) {
    ## Il testo viene esploso
    txt.splitted <- strsplit(as.character(txt), " ")[[1]]
    results <- rep(prior, length(predictor.categories))
    names(results) <- names(predictor.categories)
    results.matrix <- tibble(probs = results,
                             category = names(results),
                             iteration = 0,
                             word = "")
    iteration <- 0
    for (word in txt.splitted) {
      iteration <- iteration + 1
      results.new <- sapply(predictor.categories, function(fn) fn(word))
      results <- results * results.new
      results <- results / sum(results)
      results.matrix <- results.matrix %>%
        add_row(probs = results,
                category = names(results),
                iteration = iteration,
                word = word)
      ## results.matrix <- rbind(results.matrix, results)
    }
    ## colnames(results.matrix) <- names(results)
    ## results.matrix <- results.matrix %>%
    ##   add_column(index = 0:length(txt.splitted),
    ##              word_index = c("", txt.splitted))
    return(results.matrix)
  }
}
predictor.tracked.headlines <- predictor.tracked(prior,
                                                 get_matrix("title", db.train))

## titoli scelti a mano
probs1 <-  predictor.tracked.headlines("disney have trouble find new ceo")
probs2 <- predictor.tracked.headlines("moon lunar eclipse comet evening happen tonight")
probs3 <- predictor.tracked.headlines("win oscar pool blisteringly accurate prediction")


library(gganimate)
library(ggstance)

plot_animation <- function(data, outfile) {
  first_cat <- data[1, "category", drop = T]
  words <- data[which(data$category == first_cat), "word", drop = T]
  words.vec <- c("")
  for (i in 2:length(words))
    words.vec <- c(words.vec, paste(words[2:i],
                                    collapse = " "))
  data$word <- rep(words.vec, each = length(categories))
  title <- words.vec[length(words.vec)]
  ggplot(data, aes(x = factor(category),
                   y = probs,
                   fill = category)) +
    geom_bar(stat = "Identity",
             position = "identity",
             aes(fill = category)) +
    ylim(0, 1) +
    labs(title = "{closest_state}",
         x = "",
         y = "Probability") +
    coord_flip() +
    theme(legend.position = "none") +
    transition_states(word,
                      transition_length = 0.5,
                      state_length = 0) +
    enter_fade() +
    exit_shrink() +
    ease_aes("quadratic-in-out") %>%
  animate(height = 1000, width = 1000) +
    anim_save(outfile)
}

plot_animation(probs1, "/home/fede/grafico1.gif")

plot_animation(probs2, "/home/fede/grafico2.gif")

plot_animation(probs3, "/home/fede/grafico3.gif")



## Ogni closure rimane isolata dalle altre: ogni predittore
## per tipo (titolo o contenuto) usa dei predittori per singola
## categoria (i tipi di articolo) isolati tra di loro

## Infine, prende la classe più probabile
predict.obs <- function(predictor, txt)
  names(which.max(predictor(txt)))

## Come appena effettuato, ma in batch per un vettore
predict.vector <- function(predictor, v)
  sapply(v, function(x) predict.obs(predictor, x))


## Analisi delle prestazioni del classificatore
results.headline <- mean(predict.vector(headlines.predictor,
                                        db.test$title)
                         == db.test$section)
results.content <- mean(predict.vector(content.predictor,
                                       db.test$content)
                        == db.test$section)
results.headline
results.content

##Machine Learning (considerando una doppia stratificazione del training set)
## Sarebbe stato necessario inserire la matrice con le probabilità grezze per
## ogni categoria e non solamente la categoria più probabile, ma si è scelto di
## non farlo per non appesantire ulteriormente la complessità computazionale 
## dell'algoritmo, che non potrebbe comunque dare particolari risultati,
##perché le probabilità tendono quasi sempre a 0 o 1, e per evitare la maledizione della dimensionalità.
## Pertanto, l'algoritmo di ML servirà solo per bilanciare titolo e contenuto.
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


## 3-Cross Validation con resampling stratificato in base alla sola variabile target 'section'
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
    ##Costruisce e addestra il modello di ML
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

## Modello Recurive Partitioning, modello ricorsivo ad albero decisionale 
rpart = cv(db,method = "rpart")
## Il risultato medio è 61.34793%.


## Modello Random Forest
rf = cv(db,method = "rf")
## Il risultato medio è 72.92627%.
         
## Modello C5.0
c05 = cv(db,method = "C5.0")
## Il risultato medio è 72.90707%.
         

## Modello SVM lineare per classificazione multiclasse
svm = cv(db,method = "svmLinear3")
## Il risultato medio è 73.00307%.
         

## Modello Logistic Boost
lb = cv(db,method = "LogitBoost")
## Il risultato medio è 76.94287%, miglior risultato.

                 
## 3-Cross Validation con doppia stratificazione (in base a 'section' e 'publication')
db <- read.csv("news_sample.csv")
# Funzione di depurazione dell'accuratezza
depurate <- function(model) {
  cm_ <- data.frame(confusionMatrix.train(model,norm = "none")[['table']])
  cm <- cm_[cm_$Prediction!=cm$Reference,]
  cm <- cm[order(-cm$Freq),]
  err1 <- cm[(cm$Prediction=='news' & cm$Reference=='law & politics'),'Freq']+
    cm[(cm$Prediction=='law & politics' & cm$Reference=='news'),'Freq']
  err3 <- cm[(cm$Prediction=='crime' & cm$Reference=='news'),'Freq']+
    cm[(cm$Prediction=='news' & cm$Reference=='crime'),'Freq']
  err4 <- cm[(cm$Prediction=='food' & cm$Reference=='health & lifestyle'),'Freq']+
    cm[(cm$Prediction=='health & lifestyle' & cm$Reference=='food'),'Freq']
  err5 <- cm[(cm$Prediction=='economy, business & jobs' & cm$Reference=='law & politics'),'Freq']+
    cm[(cm$Prediction=='law & politics' & cm$Reference=='economy, business & jobs'),'Freq']
  err <- (err1+err3+err4+err5)
  err_perc <- err/sum(cm_$Freq)
  tot <- sum(cm_$Freq)-err
  true_accuracy <- sum(cm_[cm_$Prediction==cm_$Reference,'Freq'])/tot
  return(list(err1=err1,
              err2=err2,
              err3=err3,
              err4=err4,
              err5=err5,
              true_accuracy=true_accuracy))
}

# Funzione che effettua la 3-fold Cross Validation e calcola l'accuratezza depurata
cv_double <- function(db,iter=3,method) {
  res <- vector()
  ta <- vector()
  i=1
  while (i<iter+1) {
    train <- splitstackshape::stratified(db,c("section","publication"),size = 0.7)
    test <- anti_join(db,train)
    db.train <- as_tibble(train[sample(nrow(train),replace = F),])
    db.test <- test[sample(nrow(test),replace = F),]
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
    res[i] = mean(test.set$results.model == test.set$section,na.rm=TRUE)
    ta[i]= depurate(model = model)$true_accuracy
    print("Completed iteration")
    i=i+1
  }
  return(list(accuracy=mean(res,na.rm=TRUE),
              true_accuracy=mean(ta,na.rm=TRUE)))
}

## Risultati:

#Recursive Paritioning:
rpart_double = cv_double(db,method = "rpart")
#Singola: 0.6134793
#Doppia: 0.6532038
#Depurata: 0.8564617
rpart<-c(0.6134793,0.6532038,0.8564617)

#Random Forest:
rf_double = cv_double(db,method = "rf")
#Singola: 0.7292627
#Doppia: 0.7741296
#Depurata: 0.9235851
rf<-c(0.7292627,0.7741296,0.9235851)

#C.05
c05_double = cv_double(db=db,method = "C5.0")
#Singola: 0.7290707
#Doppia: 0.7772938
#Depurata: 0.9211711
c05<-c(0.7290707,0.7772938,0.9211711)

#Support Vector Machine
svm_double = cv_double(db=db,method = "svmLinear3")
#Singola: 0.7300307
#Doppia: 0.7614315
#Depurata: 0.9237299
svm<-c(0.7300307,0.7614315,0.9237299)

#Logistic Boost
lb_double = cv_double(db=db,method = "LogitBoost")
#Singola: 0.7694287
#Doppia: 0.8116422
#Depurata: 0.9287011
lb<-c(0.7694287,0.8116422,0.9287011)

#Rappresentazione dei risultati
r <- read.csv("/Users/riccardocervero/Desktop/result.csv")
TI <- as.factor(TI)
ggplot(r,aes(r$M,r$V,fill=r$TI))+
  geom_bar(position="dodge",stat="identity")+
  labs(title = "Confronto fra risultati per modello",
       x= "Modello",
       y="Accuratezza") +
  scale_fill_discrete(name="Tipo di accuratezza")+
  scale_y_continuous(breaks = seq(0, 1, 0.05))
         
  #Altra funzione di Cross Validation (versione di Fede)
  cross_validation <- function(data, model) {
  third <- nrow(data) / 3
  accuracy <- c()
  num_sample <- c()
  for (t in c(third * 1:3)) {
    sample <- seq(t, t + third - 1)
    num_sample <- c(num_sample, length(sample))
    print("sample done")
    headlines.predictor <- predictor(prior,
                                     get_matrix("title", db[-sample]))
    print("headlines predictor done")
    content.predictor <- predictor(prior,
                                   get_matrix("content", db[-sample]))
    print("contents predictor done")
    train.set <- prepare.dataset(data[-sample, ],
                                headlines.predictor,
                                content.predictor)
    print("train set done")
    test.set <- prepare.dataset(data[sample, ],
                                headlines.predictor,
                                content.predictor)
    print("test set done")
    train(section ~ publication + title + content,
          data = train.set,
          method = model,
          na.action = na.omit) -> model
    print("model done")
    accuracy <- c(accuracy, mean(predict(model,
                                         newdata = test.set,
                                         drop.unused.levels = TRUE)
                                 == test.set$section,
                                 na.rm = TRUE), accuracy)
    print(num_sample)
    print(accuracy)
  }
  return(weighted.mean(accuracy, num_sample))
}
         
save.image(file = "/Users/riccardocervero/Desktop/completed.RData")
