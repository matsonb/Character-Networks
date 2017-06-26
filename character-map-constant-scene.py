from tqdm import trange
import codecs # handle utf8
import re
import numpy as np
import networkx as nx
import os
import community
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.stats as stats
import matplotlib.patches as patches


if __name__ == "__main__":
  plt.figure(figsize=(16,12))
  filenames = open("movie-title-list.txt","r").read().split("\n")
  densities = []
  plotDensities = [0 for i in range(1,40)]
  clusteringCoefficients = []
  plotClustering = [0 for i in range(101)]
  # for j in trange(600,700):
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
    locations = [""]
    for i,line in enumerate(lines):
      if line[0]=="N":
        line = line[line[2:].find(" ")+2:line.rfind("-")].strip(" -")
        if re.match("EXT",line) or re.match("INT",line):
          line = line[5:]
          if re.match("RIOR",line):
            line = line[5:]
        scenes.append(scene)
        locations.append(line)
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

    if len(characterNetwork.nodes())<2 or partition[characterNetwork.nodes()[0]]==characterNetwork.nodes()[0]:
      if os.path.isfile("Graphs-Scenes"+filenames[j]+"-graph.dot"):
        os.remove("Graphs-Scenes/"+filenames[j]+"-graph.dot")
      if os.path.isfile("Graphs-Scenes-png/"+filenames[j]+".png"):
        os.remove("Graphs-Scenes-png/"+filenames[j]+".png")
      continue


    colors = ['b','g','r','c','m','y','k']
    # styles = ['-']#,'--','-.',':']
    style_list = {}
    for com in partition.values():
      character_list = [character for character in characterNetwork.nodes() if partition[character]==com]
      for i,character in enumerate(character_list):
        style_list[character] = colors[com%len(colors)]#+styles[i%len(styles)]
    
    for i in range(len(scenes)):
      new_scene = set()
      for character in scenes[i]:
        if character in characterNetwork.nodes():
          new_scene.add(character)
      scenes[i] = new_scene
    
    empty_scenes = []

    for i in range(len(scenes)):
      if len(scenes[i]) < 2:
        empty_scenes.append(i)
    for i in range(1,len(empty_scenes)+1):
      scenes.remove(scenes[empty_scenes[-i]])
      locations.remove(locations[empty_scenes[-i]])
    used_characters = set()
    for scene in scenes:
      for character in scene:
        used_characters.add(character)
    for character in characterNetwork.nodes():
      if character not in used_characters:
        characterNetwork.remove_node(character)
    '''
    last_removal = []
    for i,scene in enumerate(scenes):
      if len(used_characters.intersection(scene)) < 2:
        last_removal.append(i)
    for i in range(1,len(last_removal)+1):
      scenes.remove(scenes[last_removal[-i]])
      locations.remove(locations[last_removal[-i]])
    final_used_characters = set()
    for scene in scenes:
      for character in scene:
        final_used_characters.add(character)
    for character in characterNetwork.nodes():
      if character not in final_used_characters:
        characterNetwork.remove_node(character)
    print(filenames[j]+str(len(scenes)))
    '''

    character_plots = {}
    for i,character in enumerate(sorted(characterNetwork.nodes(), key = lambda character: partition[character])):
      character_position = [30*i+30 for k in range(10)]
      character_plots[character] = character_position

    i = 0
    scene_number = 2
    while i < len(scenes):
      if len(scenes[i])==0:
        i+=1
        continue
      position = np.mean([character_plots[character][0] for character in scenes[i]])
      offset = 0
      scene = scenes[i]
      scene_index = i
      location = locations[i]
      while i < len(scenes)-1 and (scene.issubset(scenes[i+1]) or scenes[i+1].issubset(scene) or location == locations[i+1]):
        if scene.issubset(scenes[i+1]):
          scene = scenes[i+1]
          location = locations[i+1]
        i+=1
      for character in scene:
        current_pos = character_plots[character][-1]*1.
        length = 10*scene_number - len(character_plots[character])
        # character_plots[character][len(character_plots[character]):10*scene_number+3] = [current_pos for k in range(length)]
        character_plots[character][len(character_plots[character]):10*scene_number+3] = [current_pos+k*(position+offset-current_pos)/(length) for k in range(length)]
        character_plots[character][10*scene_number+3:10*scene_number+10] = [position + offset for k in range(7)]
        offset+=5
      plt.gca().add_patch(patches.Rectangle((10*scene_number-1,position-3),9,offset+1,fill=False))
      if not location.find("-") == -1 and location.find("-")<20:
        end_index = location.find("-")
      else:
        end_index = 20
      plt.text(10*scene_number+4,position-4,location[0:end_index],horizontalalignment="center",verticalalignment="top",size=7)
      i+=1
      scene_number+=1
    included_character = False
    for character in character_plots:
      if len(character_plots[character]) > 15:
        included_character = True
        plt.plot(range(len(character_plots[character])),character_plots[character],style_list[character])
        plt.text(-5,character_plots[character][0],character,horizontalalignment='right',verticalalignment='center')
    plt.title(filenames[j].replace("-"," "))
    plt.gca().yaxis.set_visible(False)
    plt.gca().xaxis.set_visible(False)
    if included_character:
      plt.savefig("Character-Maps/"+filenames[j]+".png",dpi=200)
    plt.clf()







