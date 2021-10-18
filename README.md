# Clustering Player Groups from the NBA Roster

## Summary

Basketball is a sport in which numerous statistics can be derived from players. From points and assists, to blocks and steals, each basketball game offers rich data that people can make use of to generate insights. The objective of this study was to cluster players based on their statistics and attempt to identify who the best players are in the NBA, and which other players are most similar to them.

Player data per game was collected from the Basketball Reference Website for players in the 2018-2019 NBA season. The data was cleaned and preprocessed. Some preprocessing done includes: delimiter rows were removed, duplicated entries due to player trades were consolidated, numeric columns that were interpreted as objects were cast as numeric (int or float).

Exploratory data analysis was performed for the purpose of dimensionality reduction. Although manual feature selection based on domain knowledge was used to reduce dimensionality, correlations were also taken into account in removing variables. Furthermore, principal component analysis (PCA) was done in order to identify the features which contribute more to the variance. However, PCA was not used for any purpose other than for deriving insights for the manual feature selection.

The data was scaled using the MinMaxScaler in order to mitigate the effect of variables with large magnitudes. KMeans clustering was performed on the players for each of the five most recent seasons of the NBA, and it was discovered that two clusters stood out each year in terms of efficiency: the star Point Guard cluster and the star Center cluster. This does not mean that players from other positions could not excel - rather, they were clustered into one of these two clusters.

Coaches should look out for the players in the two star clusters each year. These clusters are provided in the Conclusion section for reference.

## Writeup and Output Viewing

A Jupyter Notebook contains codes used as well as the project output.

See `notebooks/Clustering Player Groups from the NBA Roster.ipynb`

## Repository Structure

### notebooks

Contains the main notebook `Clustering Player Groups from the NBA Roster.ipynb` detailing analyses done on the data as well as pertinent findings and insights.

#### notebooks/archive

Contains the notebook `Web Scraping Notebook.ipynb.ipynb` which contains code used to scrape a basketball reference web page for the player data.

### nba_playerclusters

Contains documented, user-defined utility functions used for analysis.

### data

Contains an sqlite database file where I stored the data scraped from the basketball reference website.