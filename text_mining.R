library(tm)
library(tidyverse)


db <- read_csv("news_preprocessed.csv")
n <- nrow(db)

db.train <- db[sample.int(n, n/10), ]

categories <- unique(db$section)

get_per_category <- function(category, var)
  db.train[db.train$section == category, var, drop = TRUE]

get_matrix_words <- function(data, min_freq=2) {
  tdm <- TermDocumentMatrix(Corpus(VectorSource(data)),
                            control = list(
                              stopwords = TRUE,
                              removePunctuation = TRUE,
                              removeNumbers = TRUE))
  tdm.sums <- rowSums(as.matrix(tdm))
  return(tdm.sums[which(tdm.sums >= min_freq)])
}

headlines <- map(categories, ~get_per_category(.x, "title"))
names(headlines) <- categories

articles <- map(categories, ~get_per_category(.x, "content"))
names(articles) <- categories

matrix_per_headlines <- map(headlines, get_matrix_words)
names(matrix_per_headlines) <- names(headlines)

matrix_per_content <- map(articles, get_matrix_words)
names(matrix_per_content) <- names(articles)

list_to_dataframe <- function(l) {
  words <- c()
  for (category in l)
    words <- c(words, names(category))
  words <- unique(words)
  out <- list(word = words)
  for (category in names(l)) {
    out[[category]] <- map_dbl(words,
                               ~ifelse(is.na(l[[category]][.x]),
                                       0,
                                       l[[category]][.x]))
    out[[category]] <- out[[category]] / sum(out[[category]])
    out[[category]][is.nan(out[[category]])] <- 0
  }
  return(as.tibble(out))
}

headlines_matrix <- list_to_dataframe(matrix_per_headlines)
content_matrix <- list_to_dataframe(matrix_per_content)

prior <- 1 / length(categories)

bayes.model <- function(prob, data.train, category, min_prob=1e-6) {
  words <- data.train$word
  probs <- rep(prob, length(words))
  names(probs) <- words
  
  words.category <- data.train[, category, drop = TRUE]
  names(words.category) <- words
  
  
  words.total <- rowSums(data.train[, colnames(data.train) != "word"])
  names(words.total) <- words

  train <- function(w)
    probs[w] <<- probs[w] * (words.category[w] / words.total[w])

  probs <- map_dbl(words, train)
  names(probs) <- words

  predict <- function(txt) {
    txt.splitted <- strsplit(txt, " ")[[1]]
    prod(map_dbl(txt.splitted,
                 ~ifelse(probs[.x] == 0 || is.na(probs[.x]),
                         min_prob,
                         probs[.x])))
  }
  return(predict)
}

predictor <- function(prior, data) {
  predictor.categories <- map(categories,
                              ~bayes.model(prior, data, .x))
  names(predictor.categories) <- categories
  predict <- function(txt) {
    results <- map_dbl(predictor.categories, ~.x(txt))
    results <- results / sum(results)
    return(results)
  }
}

predict.obs <- function(predictor, txt)
  names(which.max(predictor(txt)))

predict.vector <- function(predictor, v)
  map_chr(v, ~predict.obs(predictor, .x))


headlines.predictor <- predictor(prior, headlines_matrix)

content.predictor <- predictor(prior, content_matrix)

test.set <- head(db, n=200)
results.headline <- mean(predict.vector(headlines.predictor,
                                        test.set$title)
                         == test.set$section)
results.content <- mean(predict.vector(content.predictor,
                                       test.set$title)
                        == test.set$section)
results.headline
results.content
