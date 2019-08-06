library(tm)
library(tidyverse)
library(caret)


## ho usato tibble al posto del dataframe di R per poterlo integrare
## facilmente con purrr e tidyverse
db <- read_csv("news_preprocessed.csv")
n <- nrow(db)

## TODO: cross validation
db.train <- db[sample.int(n - 200, n/30), ]

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

## i titoli di giornale divisi per categoria
headlines <- map(categories, ~get_per_category(.x, "title"))
names(headlines) <- categories

## gli articoli divisi per categoria
articles <- map(categories, ~get_per_category(.x, "content"))
names(articles) <- categories

## le matrici di ricorrenza delle parole, usate per allenare il modello
matrix_per_headlines <- map(headlines, get_matrix_words)
names(matrix_per_headlines) <- names(headlines)

matrix_per_content <- map(articles, get_matrix_words)
names(matrix_per_content) <- names(articles)


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

## ed ecco le matrici
headlines_matrix <- list_to_dataframe(matrix_per_headlines)
content_matrix <- list_to_dataframe(matrix_per_content)

## liberiamo un po' di memoria, che tm mangia RAM a colazione
## matrix_per_headlines <- NULL
## matrix_per_content <- NULL
## articles <- NULL
## headlines <- NULL

## la probabilità a priori della distribuzione (uniforme)
prior <- 1 / length(categories)

## costruiamo il nostro bel modello (per una sola categoria)
bayes.model <- function(prob, data.train, category, min_prob=1e-8) {
  ## training: la probabilità è "deformata" in base alla probabilità
  ## data dalle osservazioni, calcolata però con un calcolo matriciale
  ## per risparmiare tempo
  words <- names(data.train[data.train[, category] > 0, category])
  probs <- c(1e5 * prior * (data.train[words, category] /
                            data.train[words, ] %*% rep(1, ncol(data.train))))
  names(probs) <- words

  ## restituisce la funzione di previsione
  predict <- function(txt) {
    ## il testo è esploso
    txt.splitted <- strsplit(txt, " ")[[1]]
    txt.selected <- txt.splitted[which(txt.splitted %in% words)]
    ## è effettuata la produttoria della probabilità per ogni parola
    prod(probs[txt.selected]) *
      min_prob * (length(txt.splitted) - length(txt.selected))
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
    ## si verificano i risultati di ogni categoria
    results <- sapply(predictor.categories, function(fn) fn(txt))
    ## si normalizzano con somma = 1
    results <- results / sum(results)
    ## si eliminano i valori NaN (si sa mai) in caso di divisione per
    ## 0, che avviene quando le probabilità sono particolarmente
    ## basse... paradossalmente sarebbe il caso in cui l'algoritmo
    ## presta meglio, perché ha più dati a disposizione, però per non
    ## si sa bene quale ragione fa così (= a furia di moltiplicare per
    ## valori < 1 p -> 0)
    results[is.nan(results)] <- 0
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
headlines.predictor <- predictor(prior, headlines_matrix)
content.predictor <- predictor(prior, content_matrix)

## i dati per l'allenamento non servono più: togliamoli per liberare spazio
## headlines_matrix <- NULL
## content_matrix <- NULL

## non è detto che i dati non si ripetano nel train, ma
## tendenzialmente non dovrebbe accadere (troppo)
test.set <- db[sample.int(n, 200), ]

## parte per l'analisi delle prestazioni del classificatore, si può
## eliminare tranquillamente in futuro
results.headline <- mean(predict.vector(headlines.predictor,
                                        test.set$title)
                         == test.set$section)
results.content <- mean(predict.vector(content.predictor,
                                       test.set$content)
                        == test.set$section)
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
test.set2 <- prepare.dataset(test.set,
                             headlines.predictor,
                             content.predictor)
train.set2 <- prepare.dataset(db.train,
                              headlines.predictor,
                              content.predictor)

## allena il modello
model.rf <- train(section ~ publication + title + content,
                  data = train.set2,
                  method = "rf",
                  na.action = na.omit)

## non riesco ad installare il pacchetto, ma sarebbe carino provarlo
## model.j48 <- train(section ~ publication + title + content,
##                   data = train.set2,
##                   method = "J48",
##                   na.action = na.omit)

## mostra i risultati
test.set2$results.rf <- predict(model.rf, newdata = test.set2)
# test.set2$results.j48 <- predict(model.j48, newdata = test.set2)
## questa è l'accuratezza *.*
mean(test.set2$results.rf == test.set2$section)
# mean(test.set2$results.j48 == test.set2$section)
