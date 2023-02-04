import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define the input and output variables
x1 = ctrl.Antecedent(np.arange(0, 6, 1), 'x1')
x2 = ctrl.Antecedent(np.arange(0, 6, 1), 'x2')
x3 = ctrl.Antecedent(np.arange(0, 6, 1), 'x3')
x4 = ctrl.Antecedent(np.arange(0, 6, 1), 'x4')
x5 = ctrl.Antecedent(np.arange(0, 6, 1), 'x5')
z1 = ctrl.Antecedent(np.arange(0, 6, 1), 'z1')
y = ctrl.Consequent(np.arange(0, 6, 1), 'y')

# Defining the membership functions for all input variables
input_vars = [x1, x2, x3, x4, x5, z1]
for var in input_vars:
    var['BL'] = fuzz.trimf(var.universe, [0, 0, 0])
    var['BW'] = fuzz.trimf(var.universe, [0, 1, 2])
    var['SW'] = fuzz.trimf(var.universe, [1, 2, 3])
    var['SL'] = fuzz.trimf(var.universe, [2, 3, 4])
    var['D'] = fuzz.trimf(var.universe, [3, 4, 5])

# Define the membership functions for the output variable
y['BL'] = fuzz.trimf(y.universe, [0, 0, 0])
y['BW'] = fuzz.trimf(y.universe, [0, 1, 2])
y['SW'] = fuzz.trimf(y.universe, [1, 2, 3])
y['SL'] = fuzz.trimf(y.universe, [2, 3, 4])
y['D'] = fuzz.trimf(y.universe, [3, 4, 5])

# Define the fuzzy rules using the 'ctrl.Rule' class
rule1 = ctrl.Rule(x1['BL'] & x2['BL'] & x3['BL'] & x4['BL'] & x5['BL'], z1['BL'])
rule2 = ctrl.Rule(x1['BW'] & x2['SL'] & x3['BL'] & x4['SL'] & x5['BW'], z1['BL'])
rule3 = ctrl.Rule(x1['SW'] & x2['BL'] & x3['SL'] & x4['SL'] & x5['SW'], z1['BL'])
rule4 = ctrl.Rule(x1['SL'] & x2['SL'] & x3['SL'] & x4['SL'] & x5['SL'], z1['SL'])
rule5 = ctrl.Rule(x1['D'] & x2['SL'] & x3['SL'] & x4['D'] & x5['D'], z1['SL'])
rule6 = ctrl.Rule(x1['SW'] & x2['D'] & x3['SL'] & x4['SL'] & x5['SW'], z1['SL'])
rule7 = ctrl.Rule(x1['D'] & x2['D'] & x3['D'] & x4['D'] & x5['D'], z1['D'])
rule8 = ctrl.Rule(x1['SL'] & x2['SW'] & x3['SW'] & x4['D'] & x5['SL'], z1['D'])
rule9 = ctrl.Rule(x1['D'] & x2['D'] & x3['SW'] & x4['SW'] & x5['D'], z1['D'])
rule10 = ctrl.Rule(x1['SW'] & x2['SW'] & x3['SW'] & x4['SW'] & x5['SW'], z1['SW'])
rule11 = ctrl.Rule(x1['D'] & x2['BW'] & x3['BW'] & x4['SW'] & x5['D'], z1['SW'])
rule12 = ctrl.Rule(x1['SL'] & x2['SW'] & x3['SW'] & x4['BW'] & x5['SL'], z1['SW'])
rule13 = ctrl.Rule(x1['BW'] & x2['BW'] & x3['BW'] & x4['BW'] & x5['BW'], z1['BW'])
rule14 = ctrl.Rule(x1['SL'] & x2['BW'] & x3['SW'] & x4['BW'] & x5['SL'], z1['BW'])
rule15 = ctrl.Rule(x1['BL'] & x2['SW'] & x3['BW'] & x4['SW'] & x5['BL'], z1['BW'])

# Create the control system using the rules
soccer_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, 
rule9, rule10, rule11, rule12, rule13, rule14, rule15])

# Create the control system simulation
soccer_sim = ctrl.ControlSystemSimulation(soccer_ctrl)

# Set the input variable values
soccer_sim.input['x1'] = 0.5
soccer_sim.input['x2'] = 0.5
soccer_sim.input['x3'] = 0.5
soccer_sim.input['x4'] = 0.5
soccer_sim.input['x5'] = 0.5

# Compute the output value
soccer_sim.compute()

# Print the result
print(soccer_sim.output['y'])
