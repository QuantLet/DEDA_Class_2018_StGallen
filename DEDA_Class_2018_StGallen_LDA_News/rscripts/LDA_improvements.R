#'@title:        DEDA_Class_2018_StGallen
#'@description:  Project - Text mining and sentiment analysis
#'@authors:      Antoine Gex-Fabry, Marco Hassan, Jonas Clemens

# Clean environment
rm(list = ls())
# Required libraries
library(dplyr)
library(tidytext)
library(tm)
library(readr)
library(ggplot2)
library(tidytext)
library(wordcloud)
library(reshape2)
library(topicmodels)
library(tidyr)
library(circlize)
library(igraph)
library(ggraph)
library(stringr)
library(yarrr)
# Determine your path
path = "your_path"
setwd(path)

## ------ Import /Prepare data --------
data = read_csv("news_en.csv")
documents = data[c(1, 3, 4)]

# Keep only the articles for which there are at least (approx.) 250 words.
documents = documents %>% mutate(words = sapply(strsplit(Body, " "), length)) %>% 
    filter(words >= 250) %>% select(-words)

# From this we select only a random subsamble of size 1000
set.seed(1234)
N = 1000  # number of articles to select
index = seq(1, dim(documents)[1], by = 1)
index_sample = c(sort(sample(index, N, replace = FALSE)))

# Select the random articles
documents = documents[index_sample, ]

# Load the list of stop words
data("stop_words")

articles = data_frame()
for (i in 1:N) {
    # Get the article
    doc = documents$Body[i]
    # Turns it to a string, remove numbers and put each sentence into a line Each
    # sentence is assumed to be separated by '. '
    text = toString(doc) %>% removeNumbers() %>% strsplit(". ", fixed = TRUE) %>% 
        unlist() %>% tolower()
    # Remove Punctuations
    text = gsub("[,]", "", text)
    text = gsub("[.]", "", text)
    text = gsub("[!]", "", text)
    text = gsub("[?]", "", text)
    text = gsub("[:]", "", text)
    text = gsub("’s", "", text, fixed = TRUE)
    text = gsub("reuters", "", text, fixed = TRUE)
    # Turn the dataFrame into a data_frame (tipple), necessary for further parts Also
    # removes the stop words
    text_df = data_frame(line = 1:length(text), text = text) %>% unnest_tokens(word, 
        text) %>% anti_join(stop_words)
    
    # Add information about the id (article number) and category, for descriptive
    # stats
    text_df$id = rep(documents$id[i], dim(text_df)[1])
    text_df$Kat = rep(documents$Kat[i], dim(text_df)[1])
    
    # Put it together to get the articles
    articles = rbind(articles, text_df)
}

# ----- IMPROVEMENTS OVER SINGLE WORDS ------

## Selection based on frequency instead of stop words
colnames(data)[1] = "id"
data$id = as.character(data$id)
articles$id = as.character(articles$id)
dat = articles %>% left_join(data[, c("id", "Date")]) %>% select(-line)

total_words = dat %>% group_by(Kat) %>% count(word, sort = T) %>% mutate(total = sum(n)) %>% 
    mutate(fraction = n/total)

## Plot the words frequency
ggplot(total_words, aes(fraction, fill = Kat)) + geom_histogram(show.legend = FALSE) + 
    xlim(NA, 9e-04) + facet_wrap(~Kat, ncol = 2, scales = "free_y")

# Term Frequency
freq_by_rank = total_words %>% group_by(Kat) %>% distinct() %>% arrange(desc(fraction)) %>% 
    mutate(rank = row_number())

# Exponential decay
freq_by_rank %>% ggplot(aes(rank, fraction, color = Kat)) + geom_line(size = 1, alpha = 0.8, 
    show.legend = FALSE) + scale_x_log10() + scale_y_log10() + labs(title = "Frequency by rank") + 
    theme_bw()

# Calculate and Bind the term frequency
dat = total_words %>% bind_tf_idf(word, Kat, n)

dat3 = dat %>% group_by(Kat) %>% distinct(tf_idf, .keep_all = T) %>% top_n(n = 10, 
    wt = tf_idf)

dat3 %>% ggplot(aes(reorder(word, tf_idf), tf_idf, fill = Kat)) + geom_col(show.legend = FALSE) + 
    labs(x = NULL, y = "tf-idf") + facet_wrap(~Kat, ncol = 2, scales = "free") + 
    coord_flip()

## Bigrams instead of single words
dat2 = data[index_sample, ]

dat2 = dat2 %>% unnest_tokens(bigram, Body, token = "ngrams", n = 2) %>% select(-Headline)

dat2$bigram = gsub("[^A-Za-z ]", "", dat2$bigram)

dat2 = dat2 %>% separate(bigram, c("word1", "word2"), sep = " ")

dat2 = dat2 %>% filter(!word1 == "", !word2 == "")

dat3 = dat2 %>% filter(!word1 %in% stop_words$word) %>% filter(!word2 %in% stop_words$word)

## reunite the bigrams after having removed stop words
bigrams_united = dat3 %>% unite(bigram, word1, word2, sep = " ")

total_words = bigrams_united %>% group_by(Kat) %>% count(bigram, sort = T) %>% mutate(total = sum(n)) %>% 
    mutate(fraction = n/total)

dat3 = total_words %>% bind_tf_idf(bigram, Kat, n)

## Plot bigram words
dat3 %>% arrange(desc(tf_idf)) %>% mutate(word = factor(bigram, levels = rev(unique(bigram)))) %>% 
    group_by(Kat) %>% distinct(tf_idf, .keep_all = T) %>% top_n(n = 10, wt = tf_idf) %>% 
    ungroup() %>% ggplot(aes(reorder(word, tf_idf), tf_idf, fill = Kat)) + geom_col(show.legend = FALSE) + 
    labs(x = NULL, y = "tf-idf") + facet_wrap(~Kat, ncol = 2, scales = "free") + 
    coord_flip()

# Check associations of words in the bigram
dat3 %>% separate(bigram, c("word1", "word2"), sep = " ") %>% filter(word1 == "gun") %>% 
    count(word2, sort = TRUE)

# Issues of sentiments analysis with negations
negation_words = c("not", "no", "never", "without")

AFINN = get_sentiments("afinn")

dat4 = dat2 %>% filter(word1 %in% negation_words) %>% filter(!word2 %in% stop_words$word) %>% 
    inner_join(AFINN, by = c(word2 = "word")) %>% count(word1, word2, score, sort = TRUE)

dat4 %>% mutate(contribution = n * score) %>% arrange(desc(abs(contribution))) %>% 
    head(20) %>% mutate(word2 = reorder(word2, contribution)) %>% ggplot(aes(word2, 
    n * score, fill = n * score > 0)) + geom_col(show.legend = FALSE) + xlab(paste0("Words preceded by negation words")) + 
    ylab("Sentiment score * number of occurrences") + coord_flip()

# Plots sentiment over the first 1'000 articles.
dat = data[1:1000, ] %>% unnest_tokens(word, Body)

data(stop_words)

dat = dat %>% anti_join(stop_words, by = "word") %>% select(-Headline)

dat$word = gsub("[^A-Za-z]", "", dat$word)

dat = dat %>% filter(!word == "")

dat$Kat = as.factor(dat$Kat)

sentim = dat %>% filter(!word %in% negation_words) %>% inner_join(AFINN, by = c(word = "word")) %>% 
    count(word, score, sort = TRUE)

sentim = left_join(dat, sentim, by = "word") %>% filter(!score == "NA")

total_sentim = sentim %>% group_by(Kat) %>% mutate(total = sum(score)) %>% mutate(fraction = n/total)


total_sentim = total_sentim %>% separate(Date, c("year", "month", "day"), sep = "-") %>% 
    select(-c("month", "day"))


pirateplot(formula = score ~ Kat + year, data = total_sentim, xlab = NULL, ylab = "Sentiment score", 
    main = "Sentiment score distribution by newspaper and month", pal = "google", 
    point.o = 0.2, avg.line.o = 1, theme = 0, point.pch = 16, point.cex = 1.5, jitter.val = 0.1, 
    cex.lab = 0.9, cex.names = 0.7)

# Plot bigrams in networks maps.
dat5 = total_words %>% bind_tf_idf(bigram, Kat, n)

bigram_graph = dat5 %>% filter(n > 45) %>% graph_from_data_frame()

bigram_graph

set.seed(2017)

ggraph(bigram_graph, layout = "fr") + geom_edge_link() + geom_node_point() + geom_node_text(aes(label = name), 
    vjust = 1, hjust = 1) + # xlim(-5,7) + ylim(1,10) +
xlab(NULL) + ylab(NULL) + theme(axis.text.x = element_blank(), axis.text.y = element_blank()) + 
    theme_bw()
