import os

#langs = ['da/sv', 'sv/da', 'bg/ru', 'ru/bg', 'es/pt', 'pt/es']
langs = ['bg','sv','hu','pt']
langs = ['sv']
with open("run_gpu.sh") as f:
  data = f.readlines()

pyscript = "python tagger.py --langs language --tgt_size 100 --model_type mono --gpu"

for lang in langs:
  wdata = data + [pyscript.replace("language", lang)]
  script_name = "scripts/run_tagger_" + lang.replace("/","-") + "_cross_ling_mono-100.sh"
  with open(script_name, 'w') as f:
    f.writelines(wdata)
  os.system("sbatch -J " + lang + "np_mono_cross_ling " + script_name)
