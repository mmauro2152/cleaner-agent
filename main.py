import numpy as np
import matplotlib.pyplot as plt
import agentpy as ap
from math import floor
from random import randint, random
from typing import List
import json

res_file_name = "results.json"
results = []

class CleanerEnvironment(ap.Grid):

    def remaining_trash(self):
        return list(filter(lambda ag: ag.type == "TrashAgent", self.agents))

    def get_agent_pos(self, agent:ap.Agent):
        return np.array(self.positions[agent])
    
    def get_trash_at_pos(self, pos):
        trash = list(filter(lambda ag: ag.type == "TrashAgent", list(self.grid[pos[0]][pos[1]][0])))
        return trash

class TrashAgent(ap.Agent):
    def setup(self, env):
      self.env:CleanerEnvironment = env

    def get_pos(self):
        return self.env.get_agent_pos(self)

class CleanerAgent(ap.Agent):
    def setup(self, env):
        self.picked_count:int = 0
        self.step_counter:int = 0
        self.steps_to_trash:List[int] = []
        self.env:CleanerEnvironment = env
        self.total_steps:int = 0

    def get_pos(self):
        return self.env.get_agent_pos(self)
    
    def clean(self, trash):
        self.env.remove_agents(trash)
        self.picked_count += len(trash)
        self.steps_to_trash.append(self.step_counter)
        self.step_counter = 0

    def execute(self):
        trash = self.env.get_trash_at_pos(self.get_pos())
        if len(trash) != 0:
            self.clean(trash)

        else :
            dir = (randint(-1, 1), randint(-1, 1))
            self.env.move_by(self, dir)
            self.step_counter += 1
            self.total_steps += 1

class CleanerModel(ap.Model):
    def setup(self):
        self.first_run = True
        self.trash_pc = self.p["trash_pc"]
        self.cleaner_count = self.p["cleaner_count"]
        self.m = self.p["m"]
        self.n = self.p["n"]
        self.environment = CleanerEnvironment(self, (self.m, self.n))
        self.rand_pos = self.p["rand_pos"]
        
        #add trash
        self.environment.add_agents([TrashAgent(self, self.environment) for _ in range(floor(self.m * self.n * self.trash_pc))], random=True)

        #add cleaners
        if self.rand_pos:
            self.environment.add_agents([CleanerAgent(self, self.environment) for _ in range(self.cleaner_count)], random=True)
        else:    
            self.environment.add_agents([CleanerAgent(self, self.environment) for _ in range(self.cleaner_count)], [(1, 1) for _ in range(self.cleaner_count)])

    def step(self):
        for ag in list(self.environment.agents):
            if ag.type == "CleanerAgent":
                ag.execute()

                if len(self.environment.remaining_trash()) == 0:
                    self.stop()
                    break

    def end(self):
        cleaner_agents:List[CleanerAgent] = list(filter(lambda ag: ag.type == "CleanerAgent", self.environment.agents))
        avg_picked = 0
        max_picked = None
        min_picked = None       
        avg_steps = 0
        avg_steps_counter = 0
        avg_total_steps = 0
        for ag in cleaner_agents:
            avg_picked += ag.picked_count
            max_picked = ag.picked_count if max_picked == None or ag.picked_count > max_picked else max_picked
            min_picked = ag.picked_count if min_picked == None or ag.picked_count < min_picked else min_picked

            avg_steps += sum(ag.steps_to_trash)
            avg_steps_counter += len(ag.steps_to_trash)
            avg_total_steps += ag.total_steps

        trash_count = len(self.environment.remaining_trash())
        remaining_trash_pc = trash_count / (self.m * self.n) if trash_count > 0 else 0

        self.report("cleaner_count", self.cleaner_count)
        self.report("avg trash picked", avg_picked / self.cleaner_count) 
        self.report("max trash picked", max_picked) 
        self.report("min trash picked", min_picked) 
        self.report("avg steps to trash", avg_steps / avg_steps_counter) 
        self.report("avg total steps", avg_total_steps / self.cleaner_count)
        self.report("remaining trash", "{}%".format("%.2f"%(remaining_trash_pc * 100)))
        self.report("current step", self.t)
        results.append(self.reporters)    
    
def my_plot(model, ax):
    grid = np.zeros(model.environment.shape)
    print(model.environment.positions)
    for agent, pos in model.environment.positions.items():
        grid[pos] = agent.id
        
    ax.imshow(grid, cmap='Greys')

def run_experiment(params, file_name):
        
    sample = ap.Sample(params, 5)
    exp = ap.Experiment(CleanerModel, sample, iterations=5, record=True)
    exp.run()

    with open(file_name, "w") as res_file:
        res_file.write("[\n")

        first = True
        for res in results:
            if first == False: 
                res_file.write(",\n")

            res_file.write(json.dumps(res))
            first = False

        res_file.write("\n]")

# fig, ax = plt.subplots()
# parameters = {'print': True, 'steps':max_steps}
# cleaner_model = CleanerModel(parameters)

# anim = ap.animate(cleaner_model, fig, ax, my_plot)
# anim.save("animation.gif")
# cleaner_model.end()
# print(cleaner_model.reporters)

# animation = ap.animate(cleaner_model, fig, ax, my_plot)
# plt.show()

params = {
    "steps": 300,
    "cleaner_count": ap.IntRange(100, 300),
    "trash_pc": .6,
    "m": 50,
    "n": 50,
    "rand_pos": False
}

run_experiment(params, "results.json")
params["rand_pos"] = True
results = []
run_experiment(params, "results_rand.json")
