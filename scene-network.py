from tqdm import trange
import codecs # handle utf8
import re
import numpy as np
import networkx as nx
import pydotplus
import os
import community
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.stats as stats


if __name__ == "__main__":
  filenames = open("movie-title-list.txt","r").read().split("\n")
  densities = []
  plotDensities = [0 for i in range(1,40)]
  clusteringCoefficients = []
  plotClustering = [0 for i in range(101)]
  # for j in trange(5):
  for j in trange(len(filenames)):
    f = codecs.open("rawer-take2/"+filenames[j]+"-annotated-condensed.txt","r","utf8")
    script = f.read()
    f.close()
    if script == "APPEARS BAD":
      if os.path.isfile("Graphs-Scenes"+filenames[j]+"-graph.dot"):
        os.remove("Graphs-Scenes/"+filenames[j]+"-graph.dot")
      if os.path.isfile("Graphs-Scenes-png/"+filenames[j]+".png"):
        os.remove("Graphs-Scenes-png/"+filenames[j]+".png")
      continue
    lines = script.split("\n")
    characters = set()
    scenes = []
    scene = set()
    number_characters_by_scene = [0.]
    number_edges_by_scene = []
    for i,line in enumerate(lines):
      if line[0]=="N":
        scenes.append(scene)
        scene = set()
        number_characters_by_scene.append(number_characters_by_scene[-1])
        continue
      if line[0]=="S":
        scene.add(line[2:])
        if not line[2:] in characters:
          characters.add(line[2:])
          number_characters_by_scene[-1]+=1
    if len(scenes) < 50 or len(characters)<5:
      if os.path.isfile("Graphs-Scenes"+filenames[j]+"-graph.dot"):
        os.remove("Graphs-Scenes/"+filenames[j]+"-graph.dot")
      if os.path.isfile("Graphs-Scenes-png/"+filenames[j]+".png"):
        os.remove("Graphs-Scenes-png/"+filenames[j]+".png")
      continue
    characterNetwork = nx.Graph()
    for character in characters:
      characterNetwork.add_node(character,scenes = 0)
    for scene in scenes:
      if len(number_edges_by_scene) == 0:
        number_edges_by_scene.append(0.)
      else:
        number_edges_by_scene.append(number_edges_by_scene[-1])
      for char1 in scene:
        characterNetwork.node[char1]["scenes"]=characterNetwork.node[char1]["scenes"]+1
        for char2 in scene:
          if not char1 == char2 and char1<char2:
            if char2 in characterNetwork.neighbors(char1):
              characterNetwork[char1][char2]["weight"] = characterNetwork[char1][char2]["weight"]+1
            else:
              characterNetwork.add_edge(char1,char2,weight=1)
              number_edges_by_scene[-1]+=1
    if number_characters_by_scene[-1]==0 or number_edges_by_scene[-1]==0:
      if os.path.isfile("Graphs-Scenes"+filenames[j]+"-graph.dot"):
        os.remove("Graphs-Scenes/"+filenames[j]+"-graph.dot")
      if os.path.isfile("Graphs-Scenes-png/"+filenames[j]+".png"):
        os.remove("Graphs-Scenes-png/"+filenames[j]+".png")
      continue

    total_characters = number_characters_by_scene[-1]
    percentage_characters_by_scene = [current_scene/total_characters for current_scene in number_characters_by_scene]

    total_edges = number_edges_by_scene[-1]
    percentage_edges_by_scene = [current_scene/total_edges for current_scene in number_edges_by_scene]

    plt.plot(range(len(percentage_characters_by_scene)),percentage_characters_by_scene,'b')
    plt.plot(range(len(percentage_edges_by_scene)),percentage_edges_by_scene,'r')
    plt.title(filenames[j])
    plt.savefig("Character-Network-Growth/"+filenames[j]+".png", bbox_inches="tight")
    plt.clf()

    percentage_edges_by_percent = []
    percentage_characters_by_percent = []
    for i in range(101):
      ideal_index = len(scenes)/101. * i
      min_index = int(np.floor(ideal_index))
      number_edges = percentage_edges_by_scene[min_index]
      number_characters = percentage_characters_by_scene[min_index]
      if min_index < len(scenes)-1:
        number_edges+=(ideal_index-min_index)*(percentage_edges_by_scene[min_index+1]-number_edges)
        number_characters+=(ideal_index-min_index)*(percentage_characters_by_scene[min_index+1]-number_characters)
      percentage_characters_by_percent.append(number_characters*100)
      percentage_edges_by_percent.append(number_edges*100)
    plt.plot(range(101),percentage_characters_by_percent,'b',label="Characters/Nodes")
    plt.plot(range(101),percentage_edges_by_percent,'r',label="Interactions/Edges")
    plt.xlabel("Percentage of scenes")
    plt.ylabel("Percentage of elements added")
    plt.title(filenames[j].replace("-"," "))
    plt.legend(loc=4)
    plt.savefig("Character-Network-Growth-Hundred-Scenes/"+filenames[j]+".png", bbox_inches="tight")
    plt.clf()



    # degreeList = characterNetwork.degree()
    # betweennessList = nx.betweenness_centrality(characterNetwork)
    # charsByDegree = sorted(characters, key=lambda character : degreeList[character], reverse = True)
    # charsByBetweenness = sorted(characters, key=lambda character: betweennessList[character], reverse = True)

    '''

    density = int(np.floor(100*nx.density(characterNetwork)))
    if density<40 and density>0:
      plotDensities[density-1]+=1
      densities.append(density)
    clustering = int(np.floor(100*np.mean(nx.clustering(characterNetwork).values())))
    clusteringCoefficients.append(clustering)
    plotClustering[clustering]+=1
    '''


    
    for edge in characterNetwork.edges():
      edge = characterNetwork[edge[0]][edge[1]]
      edge["penwidth"] = (edge["weight"]+4)/4

    partition = community.best_partition(characterNetwork)
    colors = ["red","blue","green","yellow","orange","brown","azure","pink","gold",
              "purple", "aquamarine", "violet", "yellowgreen", "indigo", "aliceblue","cyan", "tomato","white"]
    for key in partition:
      if partition[key]>=len(colors):
        continue
      characterNetwork.node[key]["fillcolor"] = colors[partition[key]]
      characterNetwork.node[key]["fontcolor"] = "black"
      characterNetwork.node[key]["style"] = "filled"

    centrality = nx.eigenvector_centrality_numpy(characterNetwork)

    characterNetworkCopy = characterNetwork.copy()
    for edge in characterNetworkCopy.edges():
      if not partition[edge[0]]==partition[edge[1]]:
        characterNetworkCopy.remove_edge(edge[0],edge[1])
    
    pos = nx.spring_layout(characterNetworkCopy)
    for com in set(partition.values()):
      list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
      if com>len(colors)-1:
        color = -1
      else:
        color = com
      for node in list_nodes:
        nx.draw_networkx_nodes(characterNetwork, pos, [node], node_size = characterNetwork.node[node]["scenes"]*5, node_color = colors[color])
    for edge in characterNetwork.edges():
      nx.draw_networkx_edges(characterNetwork,pos, [edge], width=(characterNetwork[edge[0]][edge[1]]["weight"]+25)/25, alpha=0.5)
    plt.title(filenames[j].replace("-"," "))
    plt.gca().xaxis.set_visible(False)
    plt.gca().yaxis.set_visible(False)
    plt.savefig("Graphs-Scenes-png/"+filenames[j]+".png", bbox_inches="tight")
    plt.clf()



    graph = nx.nx_pydot.to_pydot(characterNetwork)
    # graph.set_name(filenames[j].replace("-"," "))
    graphString = graph.to_string()
    f = codecs.open("Graphs-Scenes/"+filenames[j]+"-graph.dot","w","utf8")
    f.write(graphString)
    f.close()
    
  '''
  print(np.mean(densities))
  print(stats.variation(densities))
  print(stats.skew(densities))
  plt.plot(range(1,40),plotDensities)
  plt.show()

  print(np.mean(clusteringCoefficients))
  print(stats.variation(clusteringCoefficients))
  print(stats.skew(clusteringCoefficients))
  plt.plot(range(101),plotClustering)
  plt.show()
  '''

