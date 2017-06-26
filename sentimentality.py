from tqdm import trange
import codecs
import labMTsimple
from labMTsimple.speedy import *

def dictify(wordVec):
  '''Turn a word list into a word,count hash.'''
  thedict = dict()
  for word in wordVec:
    if word in thedict:
      thedict[word] += 1
    else:
      thedict[word] = 1
  return thedict

def distance(x,y):
  distance = 0.
  for i in range(len(x)):
    distance += abs(x[i]-y[i])
  return distance/len(x)


if __name__ == "__main__":
  windowSize = 500
  windowNumber = 48164/windowSize +1 # 48164 is max number of words in a film
  a = labMTsimple.speedy.LabMT(stopVal=1.0)
  filenames = open("movie-title-list.txt","r").read().split("\n")
  sentimentalityList = []
  # for j in range(30):
  for j in trange(len(filenames)):
    f = codecs.open("rawer-take2/"+filenames[j]+"-annotated-condensed.txt","r","utf8")
    script = f.read()
    f.close()
    if script == "APPEARS BAD":
      continue
    words = script.split(" ")
    for word in words:
      if word == word.upper() and not word == "A" and not word == "I":
        words.remove(word)
    mean = a.score(dictify(words))
    sentimentality = []
    windowStart = (len(words)-windowSize)/windowNumber
    for i in range(windowNumber):
      sentimentality.append(str((a.score(dictify(words[windowStart*i:windowStart*i+1000]))-mean)))
    sentimentalityList.append(sentimentality)

  f = codecs.open("sentimentalityLists.txt","w","utf8")
  for j in range(len(sentimentalityList)):
    f.write(",".join(sentimentalityList[j]))
    f.write("\n")
  f.close()
  