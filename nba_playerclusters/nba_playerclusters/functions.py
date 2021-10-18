import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from bs4 import BeautifulSoup
import requests
from time import sleep
import os

import sqlite3

from IPython.display import HTML, display
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from IPython.display import clear_output

from scipy.spatial.distance import cityblock
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
from PIL import Image as PILImage
from wordcloud import WordCloud

import math

def scrape_and_store(player_db):
    """
    Receives string input for year, and performs the preprocessing above.
    Stores the data in a SQL table
    
    Parameters
    ---------
    year : str
        four-digit str of the year (e.g. 2018)
    player_db : str or path
        path to player database
    """
    resp = requests.get('https://www.basketball-reference.com/leagues/NBA_{}_per_game.html'.format(year), headers=browser_headers)
    rsp_soup = BeautifulSoup(resp.text, 'lxml')
    tables = rsp_soup.select('table')
    all_players_data = pd.read_html(str(tables[0]))[0]
    all_players_data.drop(all_players_data[all_players_data['Rk']=='Rk'].index, inplace=True)
    all_players_data.head()
    all_traded = list(all_players_data[all_players_data['Tm']=='TOT']['Player'])

    for traded_player in all_traded:
        player_df = all_players_data[all_players_data['Player']==traded_player]
        player_df.iloc[0][4] = player_df.iloc[-1][4]
        all_players_data[all_players_data['Player']==traded_player] = player_df
    all_players_data = all_players_data.drop_duplicates(subset=['Rk', 'Player'])
    player_data_holder = pd.DataFrame()

    for col in all_players_data.columns:
        player_data_holder[col] = pd.to_numeric(all_players_data[col], errors='ignore')

    all_players_data = player_data_holder.copy()

    conn = sqlite3.connect(player_db)
    table_name = 'all_players_data_{}'.format(year)
    all_players_data.to_sql(table_name, conn, if_exists='replace')
    print("Success! Stored data in {}".format(table_name))
    sleep(15)
    
def read_year(year, player_db):
    """
    Read player data for the specified year.
    
    Parameters
    ---------
    year : str
        four-digit str of the year (e.g. 2018)
    player_db : str or path
        path to player database
    """
    conn = sqlite3.connect(player_db)
    df = pd.read_sql('''
    SELECT * FROM all_players_data_{}
    '''.format(str(year)), conn, index_col='Rk').drop(columns='index')
    return df
    
def illustrate_clusters_particular(df, *k):
    """
    Illustrate the clusters obtained using a word count bar graph, as well as
    a word cloud.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to take data and clusters from
        
    """
    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    j = 0
    for i in k:
        cluster_text = ' '.join(df[df.iloc[:, -3]==i]['Player_conc'])
        ax[j].imshow(create_word_cloud(cluster_text))
        ax[j].set_title('Cluster {}'.format(i), fontsize=20)
        ax[j].axis('off')
        j += 1
    plt.show()

def intra_to_inter(X, y, dist, r):
    """Compute intracluster to intercluster distance ratio
    
    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    y : array
        Class label of each point
    dist : callable
        Distance between two points. It should accept two arrays, each 
        corresponding to the coordinates of each point
    r : integer
        Number of pairs to sample
        
    Returns
    -------
    ratio : float
        Intracluster to intercluster distance ratio
    """
    intra = []
    inter = []
    np.random.seed(11)
    
    for a, b in np.random.randint(low=0, high=len(y), size=[r,2]):
        if a == b:
            continue   
        
        if y[a] == y[b]:
            intra.append(dist(X[a], X[b]))
        else:
            inter.append(dist(X[a], X[b]))
            
    intra_ave = np.mean(intra)
    inter_ave = np.mean(inter)
    
    return intra_ave / inter_ave

def cluster_range(X, clusterer, k_start, k_stop, step=1, actual=None):
    """
    Accepts the design matrix, the clustering object, the initial and final
    values to step through, and, optionally, actual labels. Returns a 
    dictionary of the cluster labels, internal validation values and, 
    if actual labels is given, external validation values, for every k.
    
    Parameters
    ----------
    X : array
        the design matrix
    clusterer : clustering object
    k_start : int
        initial value
    k_stop : int
        final value
    step : int, optional
        step in values, default 1
    actual : array
        actual labels, default None
        
    Returns
    -------
    c_range : dict
    """
    
    amis, ars, ps, chs, iidrs, inertias, scs, ys = [], [], [], [], [], [], \
    [], []

    for k in range(k_start, k_stop+1, step):
        print("Clustering with k = {}...".format(k), end='')
        clusterer.n_clusters = k
        km = clusterer
        X_predict = km.fit_predict(X)
        chs.append(calinski_harabasz_score(X, X_predict))
        iidrs.append(intra_to_inter(X, X_predict, euclidean, 50))
        inertias.append(km.inertia_)
        scs.append(silhouette_score(X, X_predict))
        ys.append(X_predict)
        
        if type(actual) != type(None):
            amis.append(adjusted_mutual_info_score(actual, X_predict, 
                                                   average_method='max'))
            ars.append(adjusted_rand_score(actual, X_predict))
            ps.append(purity(actual, X_predict))
        print(' Done!')
        
    c_range = {}
    c_range['chs'] = chs
    c_range['iidrs'] = iidrs
    c_range['inertias'] = inertias
    c_range['scs'] = scs
    c_range['ys'] = ys
    
    # Optionally,
    if type(actual) != type(None):
        c_range['amis'] = amis
        c_range['ars'] = ars
        c_range['ps'] = ps
    return c_range

def plot_internal(inertias, chs, iidrs, scs):
    """
    Plot internal validation values.
    
    Parameters
    ----------
    inertias : list or array-like
    chs : list or array-like
    iidrs : list or array-like
    inertias : list or array-like
    """
    sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(12, 8))
    ks = np.arange(2, len(inertias)+2)
    ax.plot(ks, inertias, '-o', label='SSE', ms=7)
    ax.plot(ks, chs, '-ro', label='CH', ms=7)
    ax.set_title('Internal Validation Criteria')
    ax.set_xlabel('$k$')
    ax.set_ylabel('SSE/CH')
    ax.set_xticks(ks)
    lines, labels = ax.get_legend_handles_labels()
    ax2 = ax.twinx()
    ax2.plot(ks, iidrs, '-go', label='Inter-intra', ms=7)
    ax2.plot(ks, scs, '-ko', label='Silhouette coefficient', ms=7)
    ax2.set_ylabel('Inter-Intra/Silhouette')
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines+lines2, labels+labels2)
    return ax

def plot_clusters(X, ys, X_TSNE):
    """
    Plot clusters given the design matrix and cluster labels.
    
    Parameters
    ----------
    X : matrix
        the design matrix
    ys : list or array-like
        the cluster labels
    X_TSNE : array
        pass the reduced TSNE array
    """
    k_max = len(ys) + 1
    k_mid = k_max//2 + 2
    fig, ax = plt.subplots(2, k_max//2, dpi=150, sharex=True, sharey=True, 
                           figsize=(7,4), subplot_kw=dict(aspect='equal'),
                           gridspec_kw=dict(wspace=0.01))
    for k, y in zip(range(2, k_max+1), ys):
        if k < k_mid:
#             ax[0][k%k_mid-2].scatter(*zip(*X), c=y, s=1, alpha=0.8)
            ax[0][k%k_mid-2].scatter(X_TSNE[:,0], X_TSNE[:,1], c=y, s=1)
            ax[0][k%k_mid-2].set_title('$k=%d$'%k)
            
        else:
#             ax[1][k%k_mid].scatter(*zip(*X), c=y, s=1, alpha=0.8)
            ax[1][k%k_mid].scatter(X_TSNE[:,0], X_TSNE[:,1], c=y, s=1)
            ax[1][k%k_mid].set_title('$k=%d$'%k)
    return ax

def illustrate_clusters_subplots(df, k):
    """
    Illustrate the clusters obtained using a word count bar graph, as well as
    a word cloud.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to take data and clusters from
        
    """
    rows = math.ceil(k/2)
    fig, ax = plt.subplots(rows, 2, figsize=(20, rows*7))
    df['Player_conc'] = df['Player'].apply(lambda x: x.replace(' ', ''))
    for i in range(k):
        cluster_text = ' '.join(df[df['kmeans_{}'.format(k)]==i]['Player_conc'])
        ax[i//2][i%2].imshow(create_word_cloud(cluster_text))
        ax[i//2][i%2].set_title('Cluster {}'.format(i))
        ax[i//2][i%2].axis('off')
    if not(k % 2):
        pass
    elif k % 2 == 1:
        ax[k//2][1].axis('off')
    plt.show()
        
def create_word_cloud(string):
    """
    Creates a word cloud from an input string using the WordCloud module.
    
    Parameter
    ---------
    string : str
        the string to create the word cloud from
        
    Return
    ------
    cloud : WordCloud object
    """
    cloud = WordCloud(background_color="white", width=1000,
                     height=750, max_font_size=60, max_words=300,
                     min_font_size=50).generate(string)
    return cloud

def plot_cluster_positions(df, algo_k):
    groups = df.groupby(algo_k)
    for k in range(df[algo_k].max()+1):
        current_cluster = groups.get_group(k)
        print("Cluster {}".format(k))
        current_cluster['Pos'].value_counts().plot.barh()
        plt.xlabel('Position')
        plt.xlabel('Number of Players in Cluster')
        plt.show()
        
def replace_pos(pos):
    pos_dict = {'PG-SG': 'PG', 'SG-PG': 'SG', 'SG-SF': 'SG', 'SG-PF': 'SG', 'SF-SG': 'SF', 'SF-PF': 'SF', 'PF-SF': 'PF', 'PF-C': 'PF', 'C-PF': 'C'}
    if pos in pos_dict:
        pos = pos_dict[pos]
    return pos
    
def pos_to_num(pos):
    pos_to_num_dict = {'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5}
    return pos_to_num_dict[pos]