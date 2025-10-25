# policy gradient agent pour CityLearn

import agents.policy_gradient as pg
from citylearn.citylearn import CityLearnEnv
from IPython.display import display

env = CityLearnEnv('citylearn_challenge_2023_phase_2_local_evaluation', central_agent=True) # notre environnement d'évaluation
model = pg.PolicyGradient(env) # agent Policy Gradient

"""
obs, _ = env.reset() # obs est un liste par bâtiment
while not (env.terminated or env.truncated):
    actions = model.predict(obs)
    obs, reward, done, trunc, info = env.step(actions) # on avance d'un step
"""
model.learn(episodes=9,deterministic_finish=True)

# résultats
kpis = model.env.evaluate()
kpis = kpis.pivot(index='cost_function', columns='name', values='value').round(3)
kpis = kpis.dropna(how='all')
display(kpis)

env.close() # on ferme l'environnement proprement