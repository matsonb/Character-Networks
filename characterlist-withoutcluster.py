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
import community


def netsimile(graphs, clusters = 6):
  features = getFeatures(graphs)
  signatures = getSignatures(features)
  return cluster(signatures, clusters)

def features(graphs):
  features = []
  for graph in graphs:
    graphFeatures = [[],[],[],[],[],[],[]]
    degrees = graph.degrees()
    clustering = graph.clustering()
    for node in graph:
      graphFeatures[0].append(degree[node])
      graphFeatures[1].append(clustering[node])
      egonet = nx.generators.ego.ego_graph(graph,node)
      egonetNeighbors = set()
      egonetOutEdges = 0
      neighborDegree = []
      neighborClustering = []
      for neighbor in egonet:
        if neighbor == node:
          continue
        neighborDegree.append(degree[neighbor])
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
  cluster_labels = clusterer.fit_predict(signatures)
  return signatures




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

  interactionLimit = 4
  a = labMTsimple.speedy.LabMT(stopVal=1.0)
  filenames = open("movie-title-list.txt","r").read().split("\n")
  genrelist = open("genre-list.txt","r").read().split("\n")
  for j in trange(len(filenames)):
  # for j in trange(10,15):
    # m = movies[i]
    # print(m.title)
    genrelist[j] = genrelist[j].split(",")
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
    characterNetwork = nx.Graph()
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
        if interactions[characterNames[i]][characterNames[k]] > 5:
          characterNetwork.add_edge(characterNames[i],characterNames[k],
            penwidth = (20+interactions[characterNames[i]][characterNames[k]])/25,
            weight = interactions[characterNames[i]][characterNames[k]])
      if len(characterNetwork.edges(characterNames[i]))==0:
        characterNetwork.remove_node(characterNames[i])

    if len(characterNetwork)<3:
      if os.path.isfile("Graphs/"+filenames[j]+"-graph.dot"):
        os.remove("Graphs/"+filenames[j]+"-graph.dot")
      continue

    centrality = nx.eigenvector_centrality_numpy(characterNetwork)
    for key in centrality:
      characterNetwork.node[key]["centrality"] = centrality[key]
      characters[key].append(centrality[key])

    partition = community.best_partition(characterNetwork)

    
    colors = ["white","black","blue","green","yellow","orange","brown","azure","pink","gold",
              "purple", "aquamarine", "violet", "yellowgreen", "indigo", "aliceblue","cyan", "tomato"]

    for key in partition:
      characterNetwork.node[key]["fillcolor"] = colors[partition[key]]
      characterNetwork.node[key]["fontcolor"] = "red"
      characterNetwork.node[key]["style"] = "filled"


    characterNetworkCopy = characterNetwork.copy()
    for edge in characterNetworkCopy.edges():
      if not partition[edge[0]]==partition[edge[1]]:
        characterNetworkCopy.remove_edge(edge[0],edge[1])
    '''
    
    # print(partition.keys())
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(characterNetworkCopy)
    count = 0
    for com in set(partition.values()):
      count+=1
      list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
      for node in list_nodes:
        nx.draw_networkx_nodes(characterNetwork, pos, [node], node_size = max([centrality[node]*600,1]), node_color = colors[count])
    for edge in characterNetwork.edges():
      nx.draw_networkx_edges(characterNetwork,pos, [edge], width=(interactions[edge[0]][edge[1]]+50)/50, alpha=0.5)
    plt.title(filenames[j])
    plt.show()

    '''
    
    
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