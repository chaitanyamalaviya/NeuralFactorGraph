import os

#langs = ['da/sv', 'sv/da', 'bg/ru', 'ru/bg', 'es/pt', 'pt/es']
langs = ['sv','bg','hu','pt']
#langs = ['ru']

with open("run_cpu.sh") as f:
  data = f.readlines()

pyscript = "python -u traincrf.py --langs language --batch_size 64 --model_type mono --model_name null_model_dcrf --test"

for lang in langs:
  wdata = data + [pyscript.replace("language", lang)]
  script_name = "scripts/run_tagger_" + lang.replace("/","-") + "_cross_ling_dcrf_mono-full-test.sh"
  with open(script_name, 'w') as f:
    f.writelines(wdata)
  os.system("sbatch -J " + lang + "_test_mono_cross_ling " + script_name)
