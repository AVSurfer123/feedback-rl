import time
import datetime
import itertools
import os
from dotmap import DotMap
from run_learning import run_learning

params = DotMap()
params.eta_model = "default" #path to eta model to plug into 
params.timesteps = 20000
params.eval_freq = 1000
params.save_freq = 10000
params.gamma =  args.gamma
params.learning_rate = args.learning_rate

#TO DO: Setup values to loop over
eta_model = ["model1", "model2"]
gamma = [0.99, 0.98]

combos = list(itertools.product(eta_model, gamma))

num_combos = len(combos)
curr_combo = 1
time_left = 0
beginning_time = datetime.datetime.now().strftime("%m/%d/%Y_%H:%M:%S")

for combo in combos:

	startime = time.time()

	f = open(os.path.join(os.path.dirname(__file__), "log.txt"), "+a")
	f.truncate(0)
	f.write("Started current run at {} \n".format(beginning_time))
	f.write("On Combo: {} out of {} \n".format(curr_combo, num_combos))
	f.write("Estimated Time Left: {} \n".format(time_left))
	f.close()

	#TO DO: Unpack values based on order passed into line 11
	params.eta_model = combo[0]
	params.gamma = combo[1]

	run_learning(params)

	time_left = (time.time() - startime) * (num_combos - curr_combo)
	time_left = str(datetime.timedelta(seconds=time_left))

	curr_combo += 1

f = open(os.path.join(os.path.dirname(__file__), "log.txt"), "+a")
f.truncate(0)
f.write("Done \n")
f.write("Started: {} \n".format(beginning_time))
f.write("Finished: {} \n".format(datetime.datetime.now().strftime("%m/%d/%Y_%H:%M:%S")))
f.close()