import os

langs = ['es/pt', 'fi/hu']

with open("run_gpu.sh") as f:
  data = f.readlines()

pyscript = "python tagger.py --gpu --langs language --tgt_size 1000 --model_type specific"

for lang in langs:
  wdata = data + [pyscript.replace("language", lang)]
  script_name = "scripts/run_tagger_" + lang.replace("/","-") + "_cross_ling_specific-1000.sh"
  with open(script_name, 'w') as f:
    f.writelines(wdata)
  os.system("sbatch -J " + lang + "b1k_cross_ling_spec " + script_name)
