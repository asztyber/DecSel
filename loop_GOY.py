from wdn import network_diagnoser
import os


input_file_name = 'data/networks/GOY_v1.inp'
input_name = 'GOY_4_0_23'
input_folder = 'data/mtes_results'
output_folder = 'data/decsel_results'
output_name = 'GOY'

dec_params = {'ufactor': 50, 'contig': False, 'ncuts': 100, 'recursive': True, 'niter': 100}
constraints = [['e0', 'e24', 'e22'], ['e10', 'e42'], ['e17', 'e39']]

n_subsystems = 3
flow_small_weights = True


nd = network_diagnoser.NetworkDiagnoser(input_file_name, input_name, input_folder, output_name, output_folder, mso=True)
nd.setup_decomposition(n_subsystems)
print('Starting loop')
nd.run_loop(constraints=constraints, flow_small_weights=flow_small_weights, **dec_params)
nd.save_results()
nd.read_steps_from_file()
nd.wdn_network_plots(node_size=40, link_width=2)
nd.plot_constraints(constraints, node_size=40, link_width=3)
nd.plot_steps()

