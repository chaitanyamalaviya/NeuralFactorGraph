import os

langs = ['da', 'sv', 'bg', 'ru', 'es', 'pt']

with open("run_gpu.sh") as f:
  data = f.readlines()

pyscript = "python tagger.py --gpu --langs language --tgt_size 100 --patience 10"

for lang in langs:
  wdata = data + [pyscript.replace("language", lang)]
  script_name = "scripts/run_tagger_" + lang + "_trunc100.sh"
  with open(script_name, 'w') as f:
    f.writelines(wdata)
  os.system("sbatch -J " + lang + "_trunc100 " + script_name)
