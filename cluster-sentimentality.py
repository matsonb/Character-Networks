from tqdm import trange
import codecs
import numpy as np
import labMTsimple
from labMTsimple.speedy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def distance(x,y):
  sum = 0
  for i in range(len(x)):
    sum+=(x[i]-y[i])**2
  return sum**0.5

if __name__ == "__main__":

  f = codecs.open("sentimentalityLists.txt","r","utf8")
  lines = f.read().split("\n")
  f.close()
  sentimentalityList = []
  for line in lines:
    lineSentiment = []
    for sentiment in line.split(","):
      if sentiment == "":
        continue
      lineSentiment.append(float(sentiment))
    if(len(lineSentiment) > 0):
      sentimentalityList.append(lineSentiment)






  '''
  average_film = [np.mean(sentimentalityList[:][i]) for i in range(len(sentimentalityList[0]))]
  plt.plot(range(len(average_film)),average_film)
  plt.show()
  '''









  '''
  U, s, V = np.linalg.svd(sentimentalityList,full_matrices=False)

  # row -- movie
  # colum -- mode
  W = np.dot(U,np.diag(s))

  for i in range(len(W)):

    total = sum([abs(entry) for entry in W[i]])
    for j in range(len(W[i])):
      W[i][j] = W[i][j]/total

  for i in range(len(W[0])):
    column = []
    for j in range(len(W)):
      column.append(abs(W[j][i]))
    avg_coefficient = np.mean(column)
    if avg_coefficient>.03:
      print(avg_coefficient)
      plt.plot(range(len(V[i])),V[i])
      plt.show()
  '''











  
  '''
  n_clusters = 5
  clusterer = AgglomerativeClustering(n_clusters)
  cluster_labels = clusterer.fit_predict(sentimentalityList)
  for i in range(n_clusters):
    cluster = [sentimentalityList[j] for j in range(len(sentimentalityList)) if cluster_labels[j]==i]
    print(len(cluster))
    weights = []
    for j in range(len(cluster)):
      weights.append(0)
    for j in range(len(cluster)):
      for k in range(len(cluster)):
        weights[j]+=distance(cluster[j],cluster[k])
    minIndex = 0
    for j in range(len(cluster)):
      if weights[j]<weights[minIndex]:
        minIndex = j
    representative = cluster[minIndex]
    windows = range(len(representative))
    plt.plot(windows,representative)
    plt.show()
  '''


  '''
  n_clusters = 5
  clusterer = AgglomerativeClustering(n_clusters)
  cluster_labels = clusterer.fit_predict(sentimentalityList)
  for i in range(n_clusters):
    cluster = [sentimentalityList[j] for j in range(len(sentimentalityList)) if cluster_labels[j]==i]
    average_film = cluster[0]
    for j in range(1,len(cluster)):
      for k in range(len(cluster[j])):
        average_film[k] = average_film[k]/(j+1)+cluster[j][k]/(j+1)
    windows = range(len(average_film))
    plt.plot(windows,average_film)
    plt.show()
  '''

  
  '''
  range_n_clusters = [4,5,6,7]
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
  axes = [ax1,ax2,ax3,ax4]

  for i,n_clusters in enumerate(range_n_clusters):
    ax = axes[i]

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax.set_xlim([-.2, .5])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(sentimentalityList) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = AgglomerativeClustering(n_clusters = n_clusters)
    cluster_labels = clusterer.fit_predict(sentimentalityList)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(sentimentalityList, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(sentimentalityList, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
      # Aggregate the silhouette scores for samples belonging to
      # cluster i, and sort them
      ith_cluster_silhouette_values = \
          sample_silhouette_values[cluster_labels == i]

      ith_cluster_silhouette_values.sort()

      size_cluster_i = ith_cluster_silhouette_values.shape[0]
      y_upper = y_lower + size_cluster_i

      color = cm.spectral(float(i) / n_clusters)
      ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

      # Label the silhouette plots with their cluster numbers at the middle
      ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

      # Compute the new y_lower for next plot
      y_lower = y_upper + 10  # 10 for the 0 samples

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.2, 0, 0.2, 0.4])


  plt.show()
  '''
  













  '''
  clustering = linkage(sentimentalityList,method='ward')
  plt.title('Hierarchical Clustering Dendrogram')
  plt.xlabel('sample index')
  plt.ylabel('distance')
  dendrogram(clustering,leaf_rotation=90.,leaf_font_size=8.)
  plt.show()
  '''

  
  '''
  clustering = linkage(sentimentalityList,method='ward')
  # Using that, I got 6 clusters for 1000 word windows, 5 clusters for 500 word windows
  n_clusters = 6
  clusters = fcluster(clustering,n_clusters,criterion='maxclust')
  for cluster in range(n_clusters):
    shapes = [sentimentalityList[j] for j in range(len(sentimentalityList)) if clusters[j]==cluster+1]
    average_shape = []
    for j in range(len(shapes[0])):
      sum = 0
      for k in range(len(shapes)):
        sum+=shapes[k][j]
      sum = sum/len(shapes)
      average_shape.append(sum)
    windows = range(len(average_shape))
    plt.plot(windows,average_shape)
    plt.show()
  '''