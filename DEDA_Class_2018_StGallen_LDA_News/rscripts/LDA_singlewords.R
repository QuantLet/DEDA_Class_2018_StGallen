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
head(articles, 5)

## ------ Descriptive Information -------- Number of distinct words.
nwords = length(unique(articles$word))
nwords
# Number of 'documents'
ndocs = length(unique(articles$id))
ndocs

# Number of articles / proportion by category
articles %>% distinct(id, Kat) %>% group_by(Kat) %>% summarize(n = n(), proportion = n/ndocs) %>% 
    ungroup()

# Average number of words by category Note that it doesn't have the stop words
# anymore and that the first filter to have at least 250 words is an approx.
articles %>% group_by(Kat, id) %>% summarize(n = n()) %>% ungroup() %>% group_by(Kat) %>% 
    summarize(averageWords = round(mean(n, 1)))

# For the two graphs, be careful to adjust the filter(n > NUMBER) to get a
# reasonable graph Most frequent words
articles %>% count(word, sort = TRUE) %>% top_n(10, wt = n) %>% mutate(word = reorder(word, 
    n)) %>% ggplot(aes(word, n)) + geom_bar(stat = "identity") + xlab(NULL) + coord_flip() + 
    theme_bw() + ggtitle("10 Most Frequent Words (overall)")

# Find the top 10 words
top10 = articles %>% count(word, sort = TRUE) %>% top_n(10)

# Get the positions ot be the same as the graph before
positions = rev(top10$word)

# Most frequent words by category
articles %>% group_by(Kat) %>% count(word, sort = TRUE) %>% filter(word %in% top10$word) %>% 
    mutate(word = reorder(word, n)) %>% ungroup() %>% ggplot(aes(reorder(word, n), 
    n, fill = Kat)) + geom_bar(stat = "identity") + xlab(NULL) + scale_x_discrete(limits = positions) + 
    coord_flip() + theme_bw() + ggtitle("Most frequent words by category")

## ------ Short sentiments exploration -------- Compare different lexicon
wordsSentiment = c()
lexicons = c("nrc", "bing")
i = 0
for (lex in lexicons) {
    i = i + 1
    SA = get_sentiments(lex) %>% filter(sentiment == "positive" | sentiment == "negative")
    
    # Count the number of words found by the lexicon
    sent_data = articles %>% inner_join(SA)
    # Proportion of unique words comparison
    wordsSentiment[i] = length(unique(sent_data$word))/length(unique(articles$word))
}
print(wordsSentiment)

# Get the most represented lexicon
pos = which(wordsSentiment == max(wordsSentiment))
SA = get_sentiments(lexicons[pos]) %>% filter(sentiment == "positive" | sentiment == 
    "negative")

# Tables
articles %>% inner_join(SA) %>% count(sentiment, sort = TRUE)

articles %>% inner_join(SA) %>% count(word, sentiment, sort = TRUE)

# Basic Wordcloud
articles %>% anti_join(stop_words) %>% count(word) %>% with(wordcloud(word, n, max.words = 100))

# Adding sentiments
articles %>% inner_join(get_sentiments("nrc") %>% filter(sentiment == "positive" | 
    sentiment == "negative")) %>% count(word, sentiment, sort = TRUE) %>% acast(word ~ 
    sentiment, value.var = "n", fill = 0) %>% comparison.cloud(colors = c("red", 
    "darkgreen"), max.words = 100)

## ------ Latent Dirichlet Allocation -------- Get a DocumentTermMatrix object
new_data = articles[c(3, 2)]
colnames(new_data) = c("document", "term")
# Here is the required format
new_data = new_data %>% group_by(document, term) %>% mutate(count = n()) %>% distinct() %>% 
    ungroup()

head(new_data, 5)

# Here is the transformation to get the DocumentTermMatrix object
dtm = new_data %>% cast_dtm(document, term, count)
dtm

inspect(dtm[1:4, 1:18])

# Tidy the DocumentTermMatrix Object
dtm_tidy = tidy(dtm)
dtm_tidy
# This will create a new tibble with the 'terms' (= words) indicating in which
# document they appear (document = article), and how many times each word appear
# by document.

# Add the sentiments
dtm_sentiment = dtm_tidy %>% inner_join(get_sentiments("nrc") %>% filter(sentiment == 
    "positive" | sentiment == "negative"), by = c(term = "word"))

head(dtm_sentiment, 5)

# Top 5 articles most negative sentiments -> with headlines Increase the id of
# the documents + 1 (because I did so previously in the code)
data$id = as.character(data$id + 1)
colnames(data)[1] = "document"
worst5 = dtm_sentiment %>% count(document, sentiment, wt = count) %>% spread(sentiment, 
    n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(sentiment) %>% 
    top_n(-5, wt = sentiment) %>% inner_join(data[, c(1, 2, 4, 5)], by = "document") %>% 
    arrange(sentiment)

# Top 5 articles most positive sentiments
top5 = dtm_sentiment %>% count(document, sentiment, wt = count) %>% spread(sentiment, 
    n, fill = 0) %>% mutate(sentiment = positive - negative) %>% arrange(sentiment) %>% 
    top_n(5, wt = sentiment) %>% inner_join(data[, c(1, 2, 4, 5)], by = "document") %>% 
    arrange(desc(sentiment))

# Print the headlines
print("Headlines of most negative articles:")
for (i in 1:5) {
    print(paste(as.character(i), ".", worst5$Headline[i]))
}
print("Headlines of most positive articles:")
for (i in 1:5) {
    print(paste(as.character(i), ".", top5$Headline[i]))
}

# Contribution to sentiments
dtm_sentiment %>% count(sentiment, term, wt = count) %>% filter(n >= N/5) %>% mutate(n = ifelse(sentiment == 
    "negative", -n, n)) %>% mutate(term = reorder(term, n)) %>% ggplot(aes(term, 
    n, fill = sentiment)) + geom_bar(stat = "identity") + theme_bw() + theme(axis.text.x = element_text(angle = 90, 
    hjust = 1)) + ylab("Contribution to sentiment")

## FIT FOR 5 TOPICS k is the number of topics
ap_lda = LDA(dtm, k = 5, control = list(seed = 1234))
ap_lda

# Most likely word by topic
get_terms(ap_lda)
# Most liekly topic by document get_topics(ap_lda, 1)
hist(get_topics(ap_lda, 1), breaks = 0:5, main = "Histogram of Topics", col = "lightblue", 
    border = "darkblue")

ap_topics = tidy(ap_lda, matrix = "beta")
ap_topics

ap_top_terms = ap_topics %>% group_by(topic) %>% top_n(10, beta) %>% ungroup() %>% 
    arrange(topic, -beta)

# Most frequent words by topics
ap_top_terms %>% # mutate(term = reorder(term, beta)) %>%
ggplot(aes(x = reorder(term, beta), y = reorder(beta, term), fill = factor(topic))) + 
    geom_col(show.legend = FALSE) + theme_bw() + theme(axis.text.x = element_blank()) + 
    facet_wrap(~topic, scales = "free") + xlab("") + ylab("beta") + coord_flip()

# Information about the beta
beta_spread = ap_topics %>% mutate(topic = paste0("topic", topic)) %>% spread(topic, 
    beta) %>% filter(topic1 > 0.001 | topic2 > 0.001 | topic3 > 0.001 | topic4 > 
    0.001 | topic5 > 0.001) %>% mutate(log_ratio = log2(topic5/topic3)) %>% arrange(log_ratio)

# Show words at the limit between two topics ?
beta_spread %>% select(log_ratio, term) %>% filter(abs(log_ratio) < 0.25) %>% arrange(log_ratio) %>% 
    ggplot(aes(x = reorder(term, log_ratio), y = log_ratio)) + geom_bar(stat = "identity") + 
    theme_bw() + ggtitle("Beta log ratio for Topics 3 and 5") + coord_flip()

# Relationship topic-categories GRAPH = Relationship between Category and Topic
# Source:
# https://www.datacamp.com/community/tutorials/ML-NLP-lyric-analysis#buildingmodels
category = documents[c(1, 3)]
colnames(category) = c("document", "Kat")
category$document = as.character(category$document)

source_topic_relationship = tidy(ap_lda, matrix = "gamma") %>% # join to the tidy form to get the genre field
inner_join(category, by = "document") %>% select(Kat, topic, gamma) %>% group_by(Kat, 
    topic) %>% # avg gamma (document) probability per genre/topic
mutate(mean = mean(gamma)) %>% select(Kat, topic, mean) %>% ungroup() %>% # re-label topics
mutate(topic = paste("Topic", topic, sep = " ")) %>% distinct()

circos.clear()  #very important! Reset the circular layout parameters
# this is the long form of grid.col just to show you what I'm doing you can also
# assign the genre names individual colors as well
grid.col = c(`Topic 1` = "grey", `Topic 2` = "grey", `Topic 3` = "grey", `Topic 4` = "grey", 
    `Topic 5` = "grey")

# set the gap size between top and bottom halves set gap size to 15
circos.par(gap.after = c(rep(5, length(unique(source_topic_relationship[[1]])) - 
    1), 15, rep(5, length(unique(source_topic_relationship[[2]])) - 1), 15))
chordDiagram(source_topic_relationship, grid.col = grid.col, annotationTrack = "grid", 
    preAllocateTracks = list(track.height = max(strwidth(unlist(dimnames(source_topic_relationship))))))
# go back to the first track and customize sector labels use niceFacing to pivot
# the label names to be perpendicular
circos.track(track.index = 1, panel.fun = function(x, y) {
    circos.text(CELL_META$xcenter, CELL_META$ylim[1], CELL_META$sector.index, facing = "clockwise", 
        niceFacing = TRUE, adj = c(0, 0.5))
}, bg.border = NA)  # here set bg.border to NA is important
title("Relationship Between Topic and Categories")

## FOR 12 TOPICS k is the number of topics
ap_lda = LDA(dtm, k = 12, control = list(seed = 1234))
ap_lda

# Most likely word by topic
get_terms(ap_lda)
# Most liekly topic by document get_topics(ap_lda, 1)
hist(get_topics(ap_lda, 1), breaks = 0:12, main = "Histogram of Topics", col = "lightblue", 
    border = "darkblue")

ap_topics = tidy(ap_lda, matrix = "beta")
ap_topics

ap_top_terms = ap_topics %>% group_by(topic) %>% top_n(10, beta) %>% ungroup() %>% 
    arrange(topic, -beta)

# Most frequent words by topics
ap_top_terms %>% mutate(term = reorder(term, beta)) %>% ggplot(aes(x = reorder(term, 
    beta), y = beta, fill = factor(topic))) + geom_col(show.legend = FALSE) + theme_bw() + 
    facet_wrap(~topic, scales = "free") + coord_flip()

beta_spread = ap_topics %>% mutate(topic = paste0("topic", topic)) %>% spread(topic, 
    beta) %>% filter(topic1 > 0.001 | topic2 > 0.001 | topic3 > 0.001 | topic4 > 
    0.001 | topic5 > 0.001 | topic6 > 0.001 | topic7 > 0.001 | topic8 > 0.001 | topic9 > 
    0.001 | topic10 > 0.001 | topic11 > 0.001 | topic12 > 0.01) %>% mutate(log_ratio = log2(topic2/topic1)) %>% 
    arrange(log_ratio)

# beta_spread

# Show words at the limit between two topics ?
beta_spread %>% select(log_ratio, term) %>% filter(abs(log_ratio) < 0.25) %>% arrange(log_ratio) %>% 
    ggplot(aes(x = reorder(term, log_ratio), y = log_ratio)) + geom_bar(stat = "identity") + 
    theme_bw() + coord_flip()

# GRAPH = Relationship between Category and Topic Source:
# https://www.datacamp.com/community/tutorials/ML-NLP-lyric-analysis#buildingmodels
category = documents[c(1, 3)]
colnames(category) = c("document", "Kat")
category$document = as.character(category$document)

source_topic_relationship = tidy(ap_lda, matrix = "gamma") %>% # join to the tidy form to get the genre field
inner_join(category, by = "document") %>% select(Kat, topic, gamma) %>% group_by(Kat, 
    topic) %>% # avg gamma (document) probability per genre/topic
mutate(mean = mean(gamma)) %>% select(Kat, topic, mean) %>% ungroup() %>% # re-label topics
mutate(topic = paste("Topic", topic, sep = " ")) %>% distinct()

circos.clear()  #very important! Reset the circular layout parameters
# this is the long form of grid.col just to show you what I'm doing you can also
# assign the genre names individual colors as well
grid.col = c(`Topic 1` = "grey", `Topic 2` = "grey", `Topic 3` = "grey", `Topic 4` = "grey", 
    `Topic 5` = "grey", `Topic 6` = "grey", `Topic 7` = "grey", `Topic 8` = "grey", 
    `Topic 9` = "grey", `Topic 10` = "grey", `Topic 11` = "grey", `Topic 12` = "grey")

# set the gap size between top and bottom halves set gap size to 15
circos.par(gap.after = c(rep(5, length(unique(source_topic_relationship[[1]])) - 
    1), 15, rep(5, length(unique(source_topic_relationship[[2]])) - 1), 15))
chordDiagram(source_topic_relationship, grid.col = grid.col, annotationTrack = "grid", 
    preAllocateTracks = list(track.height = max(strwidth(unlist(dimnames(source_topic_relationship))))))
# go back to the first track and customize sector labels use niceFacing to pivot
# the label names to be perpendicular
circos.track(track.index = 1, panel.fun = function(x, y) {
    circos.text(CELL_META$xcenter, CELL_META$ylim[1], CELL_META$sector.index, facing = "clockwise", 
        niceFacing = TRUE, adj = c(0, 0.5))
}, bg.border = NA)  # here set bg.border to NA is important
title("Relationship Between Topic and Categories")