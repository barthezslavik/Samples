import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Load the dataset
df = pd.read_csv('data/fuzzy.csv')

# Define the fuzzy variables
diff = ctrl.Antecedent(np.arange(-5, 6, 1), 'diff')
outcome = ctrl.Consequent(np.arange(0, 6, 1), 'outcome')

# Define the fuzzy membership functions
diff['BL'] = fuzz.trimf(diff.universe, [-5, -5, -3])
diff['SL'] = fuzz.trimf(diff.universe, [-2, -2, -1])
diff['D'] = fuzz.trimf(diff.universe, [-0.5, 0, 0.5])
diff['SW'] = fuzz.trimf(diff.universe, [1, 2, 2])
diff['BW'] = fuzz.trimf(diff.universe, [3, 5, 5])
outcome['BL'] = fuzz.trimf(outcome.universe, [0, 0, 1])
outcome['SL'] = fuzz.trimf(outcome.universe, [0, 1, 2])
outcome['D'] = fuzz.trimf(outcome.universe, [1, 2, 3])
outcome['SW'] = fuzz.trimf(outcome.universe, [2, 3, 4])
outcome['BW'] = fuzz.trimf(outcome.universe, [3, 4, 5])

# Define the fuzzy rules
rule1 = ctrl.Rule(diff['BL'] & diff['BL'], outcome['BL'])
rule2 = ctrl.Rule(diff['BL'] & diff['SL'], outcome['BL'])
rule3 = ctrl.Rule(diff['BL'] & diff['D'], outcome['BL'])
rule4 = ctrl.Rule(diff['BL'] & diff['SW'], outcome['SL'])
rule5 = ctrl.Rule(diff['BL'] & diff['BW'], outcome['SL'])
rule6 = ctrl.Rule(diff['SL'] & diff['BL'], outcome['BL'])
rule7 = ctrl.Rule(diff['SL'] & diff['SL'], outcome['SL'])
rule8 = ctrl.Rule(diff['SL'] & diff['D'], outcome['SL'])
rule9 = ctrl.Rule(diff['SL'] & diff['SW'], outcome['D'])
rule10 = ctrl.Rule(diff['SL'] & diff['BW'], outcome['D'])
rule11 = ctrl.Rule(diff['D'] & diff['BL'], outcome['BL'])
rule12 = ctrl.Rule(diff['D'] & diff['SL'], outcome['SL'])
rule13 = ctrl.Rule(diff['D'] & diff['D'], outcome['D'])
rule14 = ctrl.Rule(diff['D'] & diff['SW'], outcome['SW'])
rule15 = ctrl.Rule(diff['D'] & diff['BW'], outcome['SW'])
rule16 = ctrl.Rule(diff['SW'] & diff['BL'], outcome['SL'])
rule17 = ctrl.Rule(diff['SW'] & diff['SL'], outcome['D'])
rule18 = ctrl.Rule(diff['SW'] & diff['D'], outcome['SW'])
rule19 = ctrl.Rule(diff['SW'] & diff['SW'], outcome['BW'])
rule20 = ctrl.Rule(diff['SW'] & diff['BW'], outcome['BW'])
rule21 = ctrl.Rule(diff['BW'] & diff['BL'], outcome['SL'])
rule22 = ctrl.Rule(diff['BW'] & diff['SL'], outcome['D'])
rule23 = ctrl.Rule(diff['BW'] & diff['D'], outcome['SW'])
rule24 = ctrl.Rule(diff['BW'] & diff['SW'], outcome['BW'])
rule25 = ctrl.Rule(diff['BW'] & diff['BW'], outcome['BW'])

# Create the control system
outcome_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
                                   rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20,
                                   rule21, rule22, rule23, rule24, rule25])

# Create the simulation object
outcome_sim = ctrl.ControlSystemSimulation(outcome_ctrl)

# Train the model
total_error = 0
for index, row in df.iterrows():
    outcome_sim.input['diff'] = row['diff']
    outcome_sim.compute()
    predicted_outcome = outcome_sim.output['outcome']
    actual_outcome = row['outcome']
    error = (predicted_outcome - actual_outcome) ** 2
    total_error += error

# Calculate the mean squared error
mse = total_error / len(df)
print(f'Mean Squared Error: {mse:.2f}')
