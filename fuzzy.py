import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define the input and output variables
temperature = ctrl.Antecedent(np.arange(-20, 41, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
comfort = ctrl.Consequent(np.arange(-20, 41, 1), 'comfort')

# Define the membership functions for the input variables
temperature['cold'] = fuzz.trimf(temperature.universe, [-20, -20, 0])
temperature['cool'] = fuzz.trimf(temperature.universe, [-20, 0, 20])
temperature['warm'] = fuzz.trimf(temperature.universe, [0, 20, 40])
temperature['hot'] = fuzz.trimf(temperature.universe, [20, 40, 40])
humidity['dry'] = fuzz.trimf(humidity.universe, [0, 0, 50])
humidity['comfortable'] = fuzz.trimf(humidity.universe, [0, 50, 100])
humidity['humid'] = fuzz.trimf(humidity.universe, [50, 100, 100])

# Define the membership functions for the output variable
comfort['uncomfortable'] = fuzz.trimf(comfort.universe, [-20, -20, 0])
comfort['moderate'] = fuzz.trimf(comfort.universe, [-20, 0, 20])
comfort['comfortable'] = fuzz.trimf(comfort.universe, [0, 20, 40])

# Define the fuzzy rules
rule1 = ctrl.Rule(temperature['cold'] & humidity['humid'], comfort['uncomfortable'])
rule2 = ctrl.Rule(temperature['cool'] & humidity['comfortable'], comfort['moderate'])
rule3 = ctrl.Rule(temperature['warm'] & humidity['dry'], comfort['comfortable'])

# Create the control system
comfort_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

# Create the control system simulation
comfort_sim = ctrl.ControlSystemSimulation(comfort_ctrl)

# Set the input variable values
comfort_sim.input['temperature'] = 30
comfort_sim.input['humidity'] = 60

# Compute the output value
comfort_sim.compute()

# Print the result
print(comfort_sim.output['comfort'])