library(tm)
library(tidyverse)
library(caret)


## ho usato tibble al posto del dataframe di R per poterlo integrare
## facilmente con purrr e tidyverse
db <- read_csv("news_preprocessed.csv")
n <- nrow(db)

## TODO: cross validation

## effettua un sample
sample.prop <- 0.3
db <- db[sample.int(n, sample.prop * n), ]
n <- nrow(db)
n

## divide in train-set e test-set
test.sample <- sample.int(n, n/3)
db.train <- db[-test.sample, ]
db.test <- db[test.sample, ]

## i possibili tipi di articolo, giusto per avere una costante
categories <- unique(db$section)

## filtra il dataset in base alla categoria
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

## al modello si passa una matrice, non una lista: questa funzione
## effettua la conversione (molto alla cazzo, ma funziona)
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

## ed ecco le matrici
##   headlines_matrix <- get_matrix("title", db.train)
##   content_matrix <- get_matrix("content", db.train)
## son troppo pesanti per poter essere allocate normalmente: sono
## richiamate più avanti in modo tale da evitare allocamenti inutili
## in RAM (davvero, non credevo ne servisse così tanta)


## la probabilità a priori della distribuzione (uniforme)
prior <- 1 / length(categories)

## costruiamo il nostro bel modello (per una sola categoria)
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
    txt.splitted <- strsplit(txt, " ")[[1]]
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

## lo so, una closure che usa un'altra closure, un po' difficile da
## capire se non sai bene cosa sia una closure, quindi spiego un
## attimo qui: son delle funzioni che restituiscono funzioni,
## conservando i dati della funzione esterna; del tipo (per fare un
## contatore di eventi):
## contatore <- function() {
##   x <- 0
##   aumenta_di_1 <- function() {
##        # <<- vuol dire che non prende una nuova variabile
##        # ma è l'x di prima
##     x <<- x + 1
##     return(x)
##   }
##  return(aumenta_di_1)
## }
## evento.A <- contatore()
## evento.A()  # -> 1
## evento.A()  # -> 2
## evento.B <- contatore()
## evento.B()  # -> 1
## evento.A()  # -> 3
## ...
## ogni closure rimane isolata dalle altre: in pratica ogni predittore
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

## lo so: dovevamo mettere la matrice con le probabilità grezze per
## ogni categoria e non solamente la categoria più probabile; ma il
## fatto è che già impiega una vita così, a gestire anche la matrice
## con tutto il suo peso avrebbe impiegato mooooooolto di più e senza
## dare particolari risultati perché le probabilità tendono quasi
## sempre a 0 o 1 (si evita la maledizione della dimensionalità).
## Insomma, basta solamente mettere qualcosa per bilanciare titolo e
## contenuto e siamo a posto.
prepare.dataset <- function(data,
                            predictor.headlines,
                            predictor.contents) {
  tibble(section = data$section,
         publication = data$publication,
         category = data$category,
         title = predict.vector(predictor.headlines,
           data$title),
         content = predict.vector(predictor.contents,
           data$content))
}

## prepara le matrici per l'algoritmo
test.set <- prepare.dataset(db.test,
                            headlines.predictor,
                            content.predictor)
write.csv(test.set,
          file = "testset.csv")

train.set <- prepare.dataset(db.train,
                             headlines.predictor,
                             content.predictor)
write.csv(train.set,
          file = "trainset.csv")

## allena il modello
model.rf <- train(section ~ publication + title + content,
                  data = train.set,
                  method = "rf",
                  na.action = na.omit)

## non riesco ad installare il pacchetto, ma sarebbe carino provarlo
## model.j48 <- train(section ~ publication + title + content,
##                   data = train.set2,
##                   method = "J48",
##                   na.action = na.omit)

## mostra i risultati
test.set$results.rf <- predict(model.rf, newdata = test.set)
# test.set2$results.j48 <- predict(model.j48, newdata = test.set2)
## questa è l'accuratezza *.*
mean(test.set$results.rf == test.set$section)
# mean(test.set2$results.j48 == test.set2$section)

save.image(file = "completed.RData")
