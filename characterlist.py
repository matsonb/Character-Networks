from __future__ import print_function


from tqdm import trange
import codecs # handle utf8
import re
import numpy as np
import scipy.stats as sp
import labMTsimple
from labMTsimple.speedy import *
import networkx as nx
import pydotplus
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score

def netsimile(graphs, clusters = 22): # There are 22 (non-null) genre types in genre-list.txt
  features = getFeatures(graphs)
  signatures = getSignatures(features)
  return cluster(signatures, clusters)

def getFeatures(graphs):
  features = []
  for graph in graphs:
    graphFeatures = [[],[],[],[],[],[],[]]
    degrees = graph.degree()
    clustering = nx.clustering(graph)
    for node in graph:
      graphFeatures[0].append(degrees[node])
      graphFeatures[1].append(clustering[node])
      egonet = nx.generators.ego.ego_graph(graph,node)
      egonetNeighbors = set()
      egonetOutEdges = 0
      neighborDegree = []
      neighborClustering = []
      for neighbor in egonet:
        if neighbor == node:
          continue
        neighborDegree.append(degrees[neighbor])
        neighborClustering.append(clustering[neighbor])
        for secondNeighbor in graph[neighbor]:
          if secondNeighbor in egonet:
            continue
          egonetOutEdges+=1
          egonetNeighbors.add(secondNeighbor)
      graphFeatures[2].append(np.mean(neighborDegree))
      graphFeatures[3].append(np.mean(neighborClustering))
      graphFeatures[4].append(nx.number_of_edges(egonet))
      graphFeatures[5].append(egonetOutEdges)
      graphFeatures[6].append(len(egonetNeighbors))
    features.append(graphFeatures)
  return features

def getSignatures(features):
  signatures = []
  for graphFeatures in features:
    signature = []
    for i in range(7):
      signature.append(np.median(graphFeatures[i]))
      signature.append(np.mean(graphFeatures[i]))
      signature.append(np.std(graphFeatures[i]))
      signature.append(sp.skew(graphFeatures[i]))
      signature.append(sp.kurtosis(graphFeatures[i]))
    signatures.append(signature)
  return signatures

def cluster(signatures, clusters):
  clusterer = AgglomerativeClustering(n_clusters=clusters)
  return clusterer.fit_predict(signatures)

def dictify(wordVec):
    '''Turn a word list into a word,count hash.'''
    thedict = dict()
    for word in wordVec:
        if word in thedict:
            thedict[word] += 1
        else:
            thedict[word] = 1
    return thedict

def listify(raw_text,lang="en"):
    """Make a list of words from a string."""

    punctuation_to_replace = ["---","--","''"]
    for punctuation in punctuation_to_replace:
        raw_text = raw_text.replace(punctuation," ")
    # four groups here: numbers, links, emoticons, words
    # could be storing which of these things matched it...but don't need to
    words = [x.lower() for x in re.findall(r"(?:[0-9][0-9,\.]*[0-9])|(?:http://[\w\./\-\?\&\#]+)|(?:[\w\@\#\'\&\]\[]+)|(?:[b}/3D;p)|'\-@x#^_0\\P(o:O{X$\[=<>\]*B]+)",raw_text,flags=re.UNICODE)]

    return words
  
  
if __name__ == "__main__":

  genreCharacterList = []
  genreCharacterAttributes = []
  graphList = []

  interactionLimit = 2
  a = labMTsimple.speedy.LabMT(stopVal=1.0)
  filenames = open("movie-title-list.txt","r").read().split("\n")
  genrelist = open("genre-list.txt","r").read().split("\n")
  for j in trange(len(filenames)):
  # for j in trange(1):
    # m = movies[i]
    # print(m.title)
    genrelist[j] = genrelist[j].split(",")
    # if not "Horror" in genrelist[j]:
    #   continue
    f = codecs.open("rawer-take2/"+filenames[j]+"-annotated-condensed.txt","r","utf8")
    script = f.read()
    f.close()
    if script == "APPEARS BAD":
      f = codecs.open("rawer-take2/"+filenames[j]+"-character-list.txt","w","utf8")
      f.write("APPEARS BAD")
      f.close()
      f = codecs.open("rawer-take2/"+filenames[j]+"-speaking-order-list.txt","w","utf8")
      f.write("APPEARS BAD")
      f.close()
      if os.path.isfile("Graphs/"+filenames[j]+"-graph.dot"):
        os.remove("Graphs/"+filenames[j]+"-graph.dot")
      continue
    lines = script.split("\n")
    characters = {}
    speakingOrder = []
    for i,line in enumerate(lines):
      if lines[i][0] == "S":
        k = i+1
        speech = ""
        while k < len(lines) and len(lines[k])>0 and lines[k][0] != "S":
          if lines[k][0] == "D":
            speech = speech + lines[k][2:]
          k+=1
        if len(speakingOrder) > 0 and lines[i][2:] == speakingOrder[-1]:
          characters[lines[i][2:]] = [characters[lines[i][2:]][0],characters[lines[i][2:]][1]+" "+speech]
        speakingOrder.append(lines[i][2:])
        if lines[i][2:] in characters:
          characters[lines[i][2:]] = [characters[lines[i][2:]][0]+1,characters[lines[i][2:]][1]+" "+speech]
        else:
          characters[lines[i][2:]] = [1.,speech]
    characterNames = list(characters.keys())

    if len(characterNames) < 2:
      f = codecs.open("rawer-take2/"+filenames[j]+"-character-list.txt","w","utf8")
      f.write("APPEARS BAD")
      f.close()
      f = codecs.open("rawer-take2/"+filenames[j]+"-speaking-order-list.txt","w","utf8")
      f.write("APPEARS BAD")
      f.close()
      if os.path.isfile("Graphs/"+filenames[j]+"-graph.dot"):
        os.remove("Graphs/"+filenames[j]+"-graph.dot")
      continue

    characterNames.sort()
    for name in characterNames:
      characters[name][1] = 0*a.score(dictify(listify(characters[name][1])))
      characters[name].append([])

    scene = 0.
    for line in lines:
      if line[0]=="L":
        scene+=1
      elif line[0]=="S":
        if len(characters[line[2:]][2]) == 0 or characters[line[2:]][2][-1] != scene:
          characters[line[2:]][2].append(scene)

    for character in characterNames:
      # characters[character][0] = characters[character][0]/len(speakingOrder)
      scenes = characters[character][2]
      characters[character][2] = 0*scenes[len(scenes)/2]/scene

    interactions = {}
    characterNetwork = nx.Graph(name = filenames[j], genre = genrelist[j])
    maxInteractions = 0
    for i in range(len(characterNames)):
      interactions[characterNames[i]] = {}
      characterNetwork.add_node(characterNames[i])
      for k in range(len(characterNames)):
        interactions[characterNames[i]][characterNames[k]] = 0
    for i in range(len(speakingOrder)):
      for k in range(-interactionLimit,interactionLimit+1):
        if i+k in range(len(speakingOrder)) and speakingOrder[i]!=speakingOrder[i+k]:
          interactions[speakingOrder[i]][speakingOrder[i+k]]+=1
          if interactions[speakingOrder[i]][speakingOrder[i+k]] > maxInteractions:
            maxInteractions = interactions[speakingOrder[i]][speakingOrder[i+k]]
    for i in range(len(characterNames)):
      for k in range(i+1,len(characterNames)):
        if interactions[characterNames[i]][characterNames[k]] > 0:
          characterNetwork.add_edge(characterNames[i],characterNames[k],
            penwidth = (20+interactions[characterNames[i]][characterNames[k]])/25,
            weight = interactions[characterNames[i]][characterNames[k]])
    centrality = nx.eigenvector_centrality_numpy(characterNetwork)
    for key in centrality:
      characterNetwork.node[key]["centrality"] = centrality[key]
      characters[key].append(centrality[key])
    # if len(characterNetwork.edges(characterNames[i]))==0:
      # characterNetwork.remove_node(characterNames[i])
    
    graphList.append(characterNetwork)

    for name in characterNames:
      if characters[name][0]>5:
        characters[name][0]=0
        genreCharacterList.append(filenames[j]+" "+name)
        genreCharacterAttributes.append(characters[name])

    graph = nx.nx_pydot.to_pydot(characterNetwork)
    graphString = graph.to_string()
    f = codecs.open("rawer-take2/"+filenames[j]+"-character-list.txt","w","utf8")
    f.write("\n".join([characterNames[i] for i in range(len(characterNames))]))
    f.close()
    f = codecs.open("rawer-take2/"+filenames[j]+"-speaking-order-list.txt","w","utf8")
    f.write("\n".join([speakingOrder[i] for i in range(len(speakingOrder))]))
    f.close()  
    f = codecs.open("Graphs/"+filenames[j]+"-graph.dot","w","utf8")
    f.write(graphString)
    f.close()


  signatures = getSignatures(getFeatures(graphList))

  range_n_clusters = [4,5,6,7]
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
  fig.set_size_inches(36, 7)
  axes = [ax1,ax2,ax3,ax4]

  for i,n_clusters in enumerate(range_n_clusters):
    clusterBest = []
    clusterSecondBest = []
    for  j in range(n_clusters):
      clusterBest.append(-1)
      clusterSecondBest.append(-1)
    ax = axes[i]

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax.set_xlim([-.6, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(signatures) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    cluster_labels = cluster(signatures,n_clusters)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(signatures, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(signatures, cluster_labels)

    for j in range(len(signatures)):
      clusterNumber = cluster_labels[j]
      clusterBestCurrent = clusterBest[clusterNumber]
      if clusterBestCurrent == -1 or sample_silhouette_values[j] > sample_silhouette_values[clusterBestCurrent]:
        clusterSecondBest[clusterNumber] = clusterBest[clusterNumber]
        clusterBest[clusterNumber] = j

    for j in range(n_clusters):
      print("Cluster "+str(j)+": "+graphList[clusterBest[j]].graph["name"]+" and "+graphList[clusterSecondBest[j]].graph["name"])

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

    ax.set_title("The silhouette plot for the various clusters.")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.6,-0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

  plt.suptitle(("Silhouette analysis for Ward clustering on sample data "
                "with n_clusters = 5,10,15,20"),
               fontsize=14, fontweight='bold')

  plt.show()








