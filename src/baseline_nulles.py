# baseline agent ne prenant que des actions nulles

from citylearn.agents.base import BaselineAgent
from citylearn.citylearn import CityLearnEnv
from IPython.display import display

env = CityLearnEnv('citylearn_challenge_2023_phase_2_local_evaluation', central_agent=False) # notre environnement d'évaluation
agent = BaselineAgent(env) # agent baseline ne prenant que des actions nulles

obs, _ = env.reset() # obs est un liste par bâtiment
while not (env.terminated or env.truncated):
    actions = agent.predict(obs)
    obs, reward, done, trunc, info = env.step(actions) # on avance d'un step

# résultats
kpis = env.evaluate()
kpis = kpis.pivot(index='cost_function', columns='name', values='value').round(3)
kpis = kpis.dropna(how='all')
display(kpis)

env.close() # on ferme l'environnement proprement