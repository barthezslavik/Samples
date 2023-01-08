import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define the fuzzy variables
score = ctrl.Antecedent(np.arange(0, 11, 1), 'score')
# print(score)
win_prob = ctrl.Consequent(np.arange(0, 101, 1), 'win_prob')
# print(win_prob)

# Define the fuzzy membership functions
score['low'] = fuzz.trimf(score.universe, [0, 0, 5])
score['medium'] = fuzz.trimf(score.universe, [0, 5, 10])
score['high'] = fuzz.trimf(score.universe, [5, 10, 10])
win_prob['low'] = fuzz.trimf(win_prob.universe, [0, 0, 50])
win_prob['medium'] = fuzz.trimf(win_prob.universe, [0, 50, 100])
win_prob['high'] = fuzz.trimf(win_prob.universe, [50, 100, 100])

# Define the fuzzy rules
rule1 = ctrl.Rule(score['low'] & score['low'], win_prob['low'])
rule2 = ctrl.Rule(score['low'] & score['medium'], win_prob['medium'])
rule3 = ctrl.Rule(score['low'] & score['high'], win_prob['high'])
rule4 = ctrl.Rule(score['medium'] & score['low'], win_prob['medium'])
rule5 = ctrl.Rule(score['medium'] & score['medium'], win_prob['medium'])
rule6 = ctrl.Rule(score['medium'] & score['high'], win_prob['high'])
rule7 = ctrl.Rule(score['high'] & score['low'], win_prob['high'])
rule8 = ctrl.Rule(score['high'] & score['medium'], win_prob['high'])
rule9 = ctrl.Rule(score['high'] & score['high'], win_prob['high'])

# Create the control system
win_prob_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
win_prob_sim = ctrl.ControlSystemSimulation(win_prob_ctrl)

# Input the values for score
win_prob_sim.input['score'] = 10

# Compute the output
win_prob_sim.compute()

# Print the output
print(win_prob_sim.output['win_prob'])