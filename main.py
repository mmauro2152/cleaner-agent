import numpy as np
import matplotlib.pyplot as plt
import agentpy as ap

trash_count = 20
cleaner_count = 1

class CleanerEnvironment(ap.Grid):

    def agent_at_pos(self, pos):
        for ag in self.agents:
            if pos == self.positions[ag]:
                return ag
            
    def get_agent_pos(self, agent:ap.Agent):
        return np.array(self.positions[agent])
    
    def vector(self, source:ap.Agent, objective:ap.Agent):
        source_pos = self.get_agent_pos(source)
        obj_pos = self.get_agent_pos(objective)

        x = obj_pos[0] - source_pos[0] 
        y = obj_pos[1] - source_pos[1] 
        return np.array([x, y])


class TrashAgent(ap.Agent):
    def setup(self, env):
      self.env = env

    def get_pos(self):
        return self.env.get_agent_pos(self)

class CleanerAgent(ap.Agent):
    def setup(self, env):
        self.picked_count:int = 0
        self.env:CleanerEnvironment = env
        self.objective:TrashAgent = None

    #vector normalizado a basura objetivo con movimiento vertical o horizontal
    def get_dir(self, trash:TrashAgent):
        v = self.env.vector(self, trash)
        magnitude = np.linalg.norm(v)
        
        v = v / magnitude

        if (abs(v[0]) >= abs(v[1])):
            v = (1, 0) if v[0] > 0 else (-1, 0)
        else:
            v = (0, 1) if v[1] > 0 else (0, -1)

        return v

    def get_pos(self):
        return self.env.get_agent_pos(self)
    
    def clean(self):
        self.env.remove_agents(self.objective)
        self.objective = None
        self.picked_count += 1

    def distance_to(self, trash:TrashAgent):
        v = self.env.vector(self, trash)
        return np.linalg.norm(v)

    def set_objective(self):
        trash = list(filter(lambda ag: ag.type == "TrashAgent", self.env.agents))

        if (len(trash) == 0):
            self.model.stop()
            return False

        for t in trash:
            self.objective = t if self.objective == None or self.distance_to(t) < self.distance_to(self.objective) else self.objective

        return True

    def execute(self):
        if self.objective == None:
            if not self.set_objective():
                return 

        if np.array_equal(self.objective.get_pos(), self.get_pos()):
            self.clean()

        else :
            trash_dir = self.get_dir(self.objective)
            print(trash_dir)
            self.env.move_by(self, trash_dir)
        

class CleanerModel(ap.Model):
    def setup(self):
        self.environment = CleanerEnvironment(self, (20, 20), track_empty=True)
        
        #add trash
        self.environment.add_agents([TrashAgent(self, self.environment) for _ in range(trash_count)], random=True)

        #add cleaners
        self.environment.add_agents([CleanerAgent(self, self.environment) for _ in range(cleaner_count)], random=True)

    def step(self):
        for ag in list(self.environment.agents):
            if ag.type == "CleanerAgent":
                ag.execute()
    
    def update(self):
        return super().update()
    

def my_plot(model, ax):
    grid = np.zeros(model.environment.shape)
    print(model.environment.positions)
    for agent, pos in model.environment.positions.items():
        grid[pos] = agent.id
    #sns.heatmap(ax=ax, grid, annot=True)
    ax.imshow(grid, cmap='Greys')

fig, ax = plt.subplots()
parameters = {'print': False, 'steps':150}
cleaner_model = CleanerModel(parameters)

anim = ap.animate(cleaner_model, fig, ax, my_plot)
anim.save("animation.gif")

# animation = ap.animate(cleaner_model, fig, ax, my_plot)
# plt.show()