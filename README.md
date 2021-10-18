# Identifying Reddit Topics Using K-Means

Reddit is an online discussion site, where people come together to bring up and discuss about various topics. A sample of about 6000 reddit posts was provided. The objective of this notebook was to identify the topics present among them.

The data was preprocessed by cleaning and removal of duplicated entries, and the frequently-appearing terms in the post titles were identified in order to get preliminary insights as to what the topics might be.

Clustering was performed on the the clean post titles. First, the post titles were vectorized using TDF-IDF. After that, the dimensionality of the vectors was reduced using TSVD. The criterion for dimensionality reduction was that the cumulative variance explained needed to be at least 80%. K-Means clustering was then performed and the optimum number(s) of clusters was taken based on internal validation criteria. The clusters formed were interpreted in an attempt to identify the topic of each.

## Project Rationale

Reddit can be an interesting space to observe social phenomena. It functions almost like an interactive bulletin board, but with much greater accessibility due to it being online. By analyzing a sample of reddit posts we may be able to gain insights about online users' interests.

## Writeup and Output Viewing

A Jupyter Notebook contains codes used as well as the project output.

See `notebooks/Identifying Reddit Topics Using K-Means.ipynb`

## Repository Structure

### notebooks

Contains the main notebook `Identifying Reddit Topics Using K-Means.ipynb` detailing analyses done on the data as well as pertinent findings and insights.

### reddit_mini_project

Contains documented, user-defined utility functions used for analysis.

### data

Contains a text file with a small sample of about 6000 reddit posts.