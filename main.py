import numpy as np
import matplotlib.pyplot as plt
import agentpy as ap
from math import floor
from random import randint

m = 40
n = 20
trash_pc = 0.56
cleaner_count = 50
max_steps =  150

class CleanerEnvironment(ap.Grid):
            
    def get_agent_pos(self, agent:ap.Agent):
        return np.array(self.positions[agent])
    
    def get_trash_at_pos(self, pos):
        trash = list(filter(lambda ag: ag.type == "TrashAgent", list(self.grid[pos[0]][pos[1]][0])))
        return trash

class TrashAgent(ap.Agent):
    def setup(self, env):
      self.env = env

    def get_pos(self):
        return self.env.get_agent_pos(self)

class CleanerAgent(ap.Agent):
    def setup(self, env):
        self.picked_count:int = 0
        self.env:CleanerEnvironment = env

    def get_pos(self):
        return self.env.get_agent_pos(self)
    
    def clean(self, trash):
        self.env.remove_agents(trash)
        self.picked_count += len(trash)

    def execute(self):
        trash = self.env.get_trash_at_pos(self.get_pos())
        if len(trash) != 0:
            self.clean(trash)

        else :
            dir = (randint(-1, 1), randint(-1, 1))
            self.env.move_by(self, dir)
        

class CleanerModel(ap.Model):
    def setup(self):
        self.environment = CleanerEnvironment(self, (m, n), track_empty=True)
        
        #add trash
        self.environment.add_agents([TrashAgent(self, self.environment) for _ in range(floor(m * n * trash_pc))], random=True)

        #add cleaners
        self.environment.add_agents([CleanerAgent(self, self.environment) for _ in range(cleaner_count)], [(1, 1) for _ in range(cleaner_count)])

    def step(self):
        for ag in list(self.environment.agents):
            if ag.type == "CleanerAgent":
                ag.execute()
    
def my_plot(model, ax):
    grid = np.zeros(model.environment.shape)
    print(model.environment.positions)
    for agent, pos in model.environment.positions.items():
        grid[pos] = agent.id
    #sns.heatmap(ax=ax, grid, annot=True)
    ax.imshow(grid, cmap='Greys')

fig, ax = plt.subplots()
parameters = {'print': False, 'steps':max_steps}
cleaner_model = CleanerModel(parameters)

anim = ap.animate(cleaner_model, fig, ax, my_plot)
anim.save("animation.gif")

# animation = ap.animate(cleaner_model, fig, ax, my_plot)
# plt.show()