from skfuzzy import control
import numpy as np

# Define universe of variables
diff = np.arange(-5, 6, 1)
outcome = np.arange(-5, 6, 1)

# Create fuzzy variables
diff = control.Antecedent(diff, 'diff')
outcome = control.Consequent(outcome, 'outcome')

# Create membership functions
diff.automf(names=['BL', 'SL', 'D', 'SW', 'BW'])
outcome.automf(names=['BL', 'SL', 'D', 'SW', 'BW'])

# Create rules
rules = [control.Rule(diff[antecedent], outcome[consequent])
         for antecedent in ['BL', 'SL', 'D', 'SW', 'BW']
         for consequent in ['BL', 'SL', 'D', 'SW', 'BW']]

# Create control system
cs = control.ControlSystem(rules)

# Create control system simulation
outcome_sim = control.ControlSystemSimulation(cs)

# Set input variables
outcome_sim.input['diff'] = 3

# Calculate output
outcome_sim.compute()

# Print output
print(outcome_sim.output['outcome'])
