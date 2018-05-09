import subprocess
import pdb
import os

ps = subprocess.Popen(('squeue'), stdout=subprocess.PIPE)
output = subprocess.check_output(('grep', 'cmalaviy'), stdin=ps.stdout)
pids = []
for i, elem in enumerate(output.strip().split()):
	if i%8==0:
		pids.append(elem)

for pid in pids:
	print(pid)
        if os.path.isfile('slurm-' + pid + '.out'):
		d = subprocess.check_output('grep epoch slurm-' + pid + '.out', shell=True)        
		print(d)
		print("\n")
