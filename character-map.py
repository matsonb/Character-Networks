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
import matplotlib.patches as patches


if __name__ == "__main__":
  filenames = open("movie-title-list.txt","r").read().split("\n")
  densities = []
  plotDensities = [0 for i in range(1,40)]
  clusteringCoefficients = []
  plotClustering = [0 for i in range(101)]
  for j in trange(5):
  # for j in trange(len(filenames)):
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

    for character in characterNetwork.nodes():
      if characterNetwork.node[character]["scenes"]<2:
        characterNetwork.remove_node(character)

    partition = community.best_partition(characterNetwork)
    colors = ['b','g','r','c','m','y','k']
    styles = ['-']#,'--','-.',':']
    style_list = {}
    for com in partition.values():
      character_list = [character for character in characterNetwork.nodes() if partition[character]==com]
      for i,character in enumerate(character_list):
        style_list[character] = colors[com%len(colors)]+styles[i%len(styles)]

    for i in range(len(scenes)):
      new_scene = set()
      for character in scenes[i]:
        if character in characterNetwork.nodes():
          new_scene.add(character)
      scenes[i] = new_scene
    empty_scenes = [scene for scene in scenes if len(scene)==0]
    for scene in empty_scenes:
      scenes.remove(scene)

    character_plots = {}
    for i,character in enumerate(sorted(characterNetwork.nodes(), key = lambda character: partition[character])):
      character_position = [30*i+30 for k in range(5*len(scenes))]
      character_plots[character] = character_position

    i = 0
    while i < len(scenes):
      if len(scenes[i])==0:
        i+=1
        continue
      position = np.mean([character_plots[character][0] for character in scenes[i]])
      offset = 0
      scene = scenes[i]
      scene_index = i
      while i < len(scenes)-1 and (scene.issubset(scenes[i+1]) or scenes[i+1].issubset(scene)):
        if scene.issubset(scenes[i+1]):
          scene = scenes[i+1]
        i+=1
      for character in scene:
        character_plots[character][5*scene_index:] = [position + offset for k in range(len(character_plots[character][5*scene_index:]))]
        offset+=5
      plt.gca().add_patch(patches.Rectangle((np.max([5*scene_index-1,1]),position-3),5*(i-scene_index+1)+1,offset+1,fill=False))
      i+=1
    for character in character_plots:
      plt.plot(range(5*len(scenes)),character_plots[character],style_list[character])
    plt.title(filenames[j].replace("-"," "))
    plt.show()
    plt.clf()







