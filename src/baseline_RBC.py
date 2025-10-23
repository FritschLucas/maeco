# baseline agent utilisant une stratégie de régulation basique (RBC)

from citylearn.agents.rbc import BasicRBC
from citylearn.citylearn import CityLearnEnv
from IPython.display import display

env = CityLearnEnv('citylearn_challenge_2023_phase_2_local_evaluation', central_agent=False) # notre environnement d'évaluation
model = BasicRBC(env) # agent RBC de base

"""
obs, _ = env.reset() # obs est un liste par bâtiment
while not (env.terminated or env.truncated):
    actions = model.predict(obs)
    obs, reward, done, trunc, info = env.step(actions) # on avance d'un step
"""
model.learn(episodes=1,deterministic_finish=True)

# résultats
kpis = model.env.evaluate()
kpis = kpis.pivot(index='cost_function', columns='name', values='value').round(3)
kpis = kpis.dropna(how='all')
display(kpis)

env.close() # on ferme l'environnement proprement