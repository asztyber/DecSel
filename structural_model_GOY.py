from wdn import EpanetConverter, fdt_helpers
import wntr
from spectral.sensor_placement import place_sensors
import os
import json
import numpy as np

n_sensors = 4 # choose number of pressure sensors here
n_leaks = 23 # choose number of considered leaks here
input_file_name = 'data/networks/GOY_v1.inp'
output_folder = 'data/mtes_results'
short_output_name = 'GOY'
output_name = short_output_name + '_' + str(n_sensors) + '_0_' + str(n_leaks)

np.random.RandomState(13)

wn = wntr.network.WaterNetworkModel(input_file_name)
sensors = place_sensors(wn, n_sensors)

epn_conv = EpanetConverter.EpanetConverter(input_file_name, sensors, [], sensor_faults=False)
epn_conv.structural_from_epanet()
epn_conv.save_files(output_folder, short_output_name)

mtes, unambiguity_groups, mso = fdt_helpers.calculate_tests(epn_conv.model, mso=True)
print('Number of mtes: ', len(mtes))
print(unambiguity_groups)
if mso:
    print('Number of msos: ', len(mso))

file_name = output_name + '_unambiguity_groups.json'
with open(os.path.join(output_folder, file_name), 'w') as f:
    json.dump(unambiguity_groups, f)

if mso:
    file_name = output_name + '_msos.json'
    with open(os.path.join(output_folder, file_name), 'w') as f:
        json.dump(mso, f)

file_name = output_name + '_mtes.json'
with open(os.path.join(output_folder, file_name), 'w') as f:
    json.dump(mtes, f)

result_stats = {
    'junctions': len(wn.node_name_list),
    'pipes': len(wn.link_name_list),
    'psensors': len(sensors),
    'faults': n_leaks,
    'mtes': len(mtes)
}

if mso:
    result_stats['msos'] = len(mso)

file_name = output_name + '_result_stats.json'
with open(os.path.join(output_folder, file_name), 'w') as f:
    json.dump(result_stats, f)
