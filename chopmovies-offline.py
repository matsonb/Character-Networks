from tqdm import trange
import codecs # handle utf8
import re
import numpy as np
import shutil
import subprocess
import unirest
import datetime

def appearsGood(types):
  speaker = 0
  action = 0
  dialogue = 0
  slug = 0
  for type in types:
    if type=="s":
      speaker+=1
    elif type=="a":
      action+=1
    elif type=="d":
      dialogue+=1
    elif type=="l":
      slug+=1
  if speaker < 5 or action < 5 or dialogue < 5 or slug < 5:
    return False
  return True


  

def findParens(lines,types):
  line = lines[-1]
  parse = re.findall(r"(.*?)(\(.*?\))(.*?)",line)
  if len(parse)>0 and len(parse[0])>1:
    lines[-1] = parse[0][0].strip()
    lines.append(parse[0][1].strip())
    lines.append(parse[0][2].strip())
    types.append("p")
    types.append(types[-2])
    if lines[-3] == "":
      del lines[-3]
      del types[-3]
    if lines[-2] == "":
      del lines[-2]
      del types[-2]
    if lines[-1] == "":
      del lines[-1]
      del types[-1]
    else:
      findParens(lines,types)

  
if __name__ == "__main__":
  filenames = open("movie-title-list.txt","r").read().split("\n")
  for j in trange(len(filenames)):
  # for j in trange(100):
    # m = movies[i]
    # print(m.title)
    f = codecs.open("rawer-take2/"+filenames[j]+".txt.clean01","r","utf8")
    script = f.read()
    f.close()
    lines = script.split("\n")
    types = ["u" for line in lines]
    line_types = {"u":"unknown",
     "b":"blank",
     "s":"speaker",
     "a":"action",
     "p":"speaking direction",
     "d":"dialogue",
     "l":"slug (scene)",
     "r":"right (cut to, etc)",}

    bold_spacings = []
    general_spacings = []
    for i,line in enumerate(lines):
      blank = re.findall(r"^(<b>)?(([^a-zA-Z]|CONTINUED)*)(</b>)?$",line,re.IGNORECASE)
      if len(blank)>0:
        types[i] = "b"
        continue
      bold = re.findall(r"<b>(\s*)(.*?)</b>",line,re.IGNORECASE)
      if len(bold) > 0:
        space = bold[0][0]
        if not re.search(r"(CUT|FADE|BACK) (TO|IN|OUT)",bold[0][1]):
          bold_spacings.append(len(space))
        text = bold[0][1].rstrip()
        types[i] = "l"
        continue
      line_match = re.findall(r"^(\s*)(.*?)$",line)
      if len(line_match) > 0:
        space = line_match[0][0]
        general_spacings.append(len(space))
        text = line_match[0][1].rstrip()
        types[i] = "a"

    # print(filenames[j])
    # print(bold_spacings[:100])
    # print(np.mean(bold_spacings))
    # print(general_spacings[:100])
    # print(np.mean(general_spacings))

    
    condensedLines = []
    condensedTypes = []

    for i,line in enumerate(lines):
      blank = re.findall(r"^(<b>)?(([^a-zA-Z]|CONTINUED|PAGE)*)(</b>)?$",line,re.IGNORECASE)
      if len(blank)>0:
        lines[i] = blank[0][1]
        types[i] = "b"
        continue
      bold = re.findall(r"<b>(\s*)(.*?)</b>",line,re.IGNORECASE)
      line_match = re.findall(r"^(\s*)(.*?)$",line)
      if len(bold) > 0:
        space = bold[0][0]
        if len(space) > np.mean(bold_spacings):
          types[i] = "s"
        if re.search(r"(CUT|FADE|BACK|DISSOLVE) (TO|IN|OUT)",bold[0][1]):
          types[i] = "r"
        text = bold[0][1].rstrip(" ")
        if re.match(r"\(.*\)\r?$",text):
          types[i] = "p"
        lines[i] = space+text
      elif len(line_match) > 0:
        space = line_match[0][0]
        if re.match(r"\(.*?\)\r?$",line_match[0][1]):
          types[i] = "p"
        elif len(space) > np.mean(general_spacings):
          types[i] = "d"
      if len(condensedTypes) > 0 and condensedTypes[-1] == types[i]:
        condensedLines[-1] = condensedLines[-1] + " " + lines[i].strip()
      else:
        if(len(condensedLines)) > 0 and condensedTypes[-1] in ["d","s"]:
          findParens(condensedLines,condensedTypes)
        condensedTypes.append(types[i])
        condensedLines.append(lines[i].strip())
    f = codecs.open("rawer-take2/"+filenames[j]+"-annotated.txt","w","utf8")
    f.write("\n".join([types[i].upper()+" "+lines[i] for i in range(len(lines))]))
    f.close()
    if appearsGood(condensedTypes):
      f = codecs.open("rawer-take2/"+filenames[j]+"-annotated-condensed.txt","w","utf8")
      f.write("\n".join([condensedTypes[i].upper()+" "+condensedLines[i] for i in range(len(condensedLines))]))
      f.close()
    else:
      f = codecs.open("rawer-take2/"+filenames[j]+"-annotated-condensed.txt","w","utf8")
      f.write("APPEARS BAD")
      f.close()