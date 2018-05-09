import os

#langs = ['da/sv', 'sv/da', 'bg/ru', 'ru/bg', 'es/pt', 'pt/es']
langs = ['es/pt']
langs = ['ru/bg']

with open("run_gpu.sh") as f:
  data = f.readlines()

pyscript = "python -u baselineTagger.py --gpu --langs language --tgt_size 100 --model_type specific --test"

for lang in langs:
  wdata = data + [pyscript.replace("language", lang)]
  script_name = "scripts/run_tagger_" + lang.replace("/","-") + "_cross_ling-100-test.sh"
  with open(script_name, 'w') as f:
    f.writelines(wdata)
  os.system("sbatch -J " + lang + "_baseline_cross_ling " + script_name)
