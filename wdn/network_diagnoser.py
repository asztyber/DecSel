import wntr
import os
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from gdec import decompose, graph_creation
import networkx as nx
from itertools import permutations
import matplotlib.colors


class NetworkDiagnoser():

    def __init__(self, input_network_file, input_name, input_folder, output_name, output_folder, mso=False):
        self.input_network_file = input_network_file
        self.input_name = input_name
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.output_name = output_name
        self.wn = wntr.network.WaterNetworkModel(input_network_file)
        self.mso = mso
        self.load_files()

    def load_files(self):
        if self.mso:
            file_name = self.input_name + '_msos.json'
            with open(os.path.join(self.input_folder, file_name)) as f:
                self.msos = json.load(f)
        else:
            file_name = self.input_name + '_mtes.json'
            with open(os.path.join(self.input_folder, file_name)) as f:
                self.msos = json.load(f)

        file_name = self.input_name + '_eq_name_map.json'
        with open(os.path.join(self.input_folder, file_name)) as f:
            self.eq_name_map = json.load(f)

        file_name = self.input_name + '_f_name_map.json'
        with open(os.path.join(self.input_folder, file_name)) as f:
            self.f_name_map = json.load(f)

        file_name = self.input_name + '_sm_data.json'
        with open(os.path.join(self.input_folder, file_name)) as f:
            self.sm_data = json.load(f)

    def reduce_tests(self, fmsos_eqs):
        # non-isolable faults are merged into one fault
        tests = list()
        for mso in fmsos_eqs:
            faults = tuple(float(v[1:]) for e in mso for v in self.sm_data["model"][e] if v in self.sm_data['faults'])
            tests.append(faults)

        print('Number of different fault signatures: ', len(set(tests)))
        print('Minimal cardinality of fault support: ', min([len(test) for test in tests]))

        max_isol = decompose.max_isolability(tests, len(self.sm_data['faults']))

        fault_enc = dict()
        f_nr = 0
        for group in max_isol:
            fault_enc[group[0]] = f_nr
            if len(group) > 1:
                for f in group[1:]:
                    fault_enc[f] = f_nr
            f_nr = f_nr + 1
        tests_red = [tuple(set(fault_enc[f] for f in t)) for t in tests]

        self.tests_red = tests_red
        self.fault_enc = fault_enc

    def setup_decomposition(self, n_subsystems):
        self.n_subsystems = n_subsystems
        fmsos_eqs = [['e' + str(e) for e in el] for el in self.msos]
        self.reduce_tests(fmsos_eqs)

    def run_loop(self, constraints=None, flow_small_weights=False, **kwargs):
        sm = self.sm_data['model']
        fmsos_eqs = [['e' + str(e) for e in el] for el in self.msos]
        n_faults = len(set(self.fault_enc.values()))

        small_weights = None
        if flow_small_weights:
            small_weights = [key for key, e in self.eq_name_map.items() if e[1] == 'q']

        # initial decomposition
        subsystems, edgecuts = decompose.decompose_from_fmsos(fmsos_eqs, sm, self.sm_data, self.n_subsystems,
                                                              constraints=constraints,
                                                              small_weights=small_weights)
        print('Initial decomposition: ', edgecuts)
        fmsos, subsystems_res, sel_tests, steps, costs = decompose.loop(subsystems, edgecuts, sm, self.sm_data,
                                                                        fmsos_eqs, self.tests_red, n_faults,
                                                                        self.n_subsystems, constraints=constraints,
                                                                        small_weights=small_weights,
                                                                        **kwargs)
        self.fmsos = fmsos
        self.subsystems_res = subsystems_res
        self.steps = steps
        self.costs = costs
        self.calc_shared_vars()

    def calc_shared_vars(self):
        subs_vars = dict()
        for key, subsystem in self.subsystems_res.items():
            subs_vars[key] = set([v for e in subsystem for v in self.sm_data["model"][e]])
        shared_vars = dict()
        for subsystem1 in self.subsystems_res:
            for subsystem2 in self.subsystems_res:
                if subsystem1 < subsystem2:
                    shareds = str(subs_vars[subsystem1].intersection(subs_vars[subsystem2]))
                    shared_vars[str((subsystem1, subsystem2))] = shareds
        self.shared_vars = shared_vars

    def save_results(self):
        file_name = self.output_name + '_steps.json'
        with open(os.path.join(self.output_folder, file_name), 'w') as f:
            json.dump(self.steps, f)

        file_name = self.output_name + '_costs.json'
        with open(os.path.join(self.output_folder, file_name), 'w') as f:
            json.dump(self.costs, f)

        file_name = self.output_name + '_shared_vars.json'
        with open(os.path.join(self.output_folder, file_name), 'w') as f:
            json.dump(self.shared_vars, f)

    def plot_constraints(self, constraints, node_size=200, link_width=3, node_labels=False, link_labels=False):
        cmap = cm.get_cmap('Dark2', self.n_subsystems)

        # junction_constraints = {key: 0 for key in self.wn.node_name_list}
        junction_constraints = dict()
        # link_constraints = {key: 0 for key in self.wn.link_name_list}
        link_constraints = dict()
        for i, constr in enumerate(constraints):
            junction_constr = {self.eq_name_map[eq][2:]: i + 1 for eq in constr if self.eq_name_map[eq][1] == 'p'}
            junction_constraints.update(junction_constr)
            link_constr = {self.eq_name_map[eq][2:]: i + 1 for eq in constr if self.eq_name_map[eq][1] == 'q'}
            link_constraints.update(link_constr)

        wntr.graphics.network.plot_network(self.wn, node_attribute=junction_constraints, add_colorbar=False,
                                           node_size=node_size, link_width=link_width, node_labels=node_labels,
                                           link_labels=link_labels, link_attribute=link_constraints,
                                           link_cmap=cmap, node_cmap=cmap)

        file_name = self.output_name + '_wdn_constraints.png'
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, file_name))

    def wdn_network_plots(self, node_size=200, link_width=3, node_labels=False, link_labels=False):

        node_link_cmap = cm.get_cmap('Dark2', self.n_subsystems)

        wntr.graphics.network.plot_network(self.wn)
        file_name = self.output_name + '_network.png'
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, file_name))

        eq_assignment = decompose.subsystems_to_eq_assignment(self.subsystems_res)
        wdn_parts = {self.eq_name_map[key]: part for key, part in eq_assignment.items()}
        junction_wdn_parts = {key[2:]: part for key, part in wdn_parts.items() if key[1] == 'p'}
        link_wdn_parts = {key[2:]: part for key, part in wdn_parts.items() if key[1] == 'q'}
        measurement_wdn_parts = {key[3:]: part for key, part in wdn_parts.items() if key[1:3] == 'mp'}
        junction_wdn_parts.update(measurement_wdn_parts)
        measurement_ind = dict()
        for key in measurement_wdn_parts.keys():
            measurement_ind[key] = 1

        ax = wntr.graphics.network.plot_network(self.wn, node_attribute=junction_wdn_parts, add_colorbar=False,
                                                node_size=node_size, link_width=link_width, node_labels=node_labels,
                                                link_labels=link_labels, link_attribute=link_wdn_parts,
                                                link_cmap=node_link_cmap, node_cmap=node_link_cmap)

        handles = [mpatches.Patch(color=node_link_cmap(i),
                   label='$\Sigma_' + str(i) + '$') for i in range(self.n_subsystems)]
        ax.legend(handles=handles)

        file_name = self.output_name + '_wdn_partitions.png'
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, file_name))

        cmap = matplotlib.colors.ListedColormap(['cyan'])
        wntr.graphics.network.plot_network(self.wn, node_attribute=measurement_ind, add_colorbar=False,
                                           node_size=node_size + 30, node_labels=node_labels, link_labels=link_labels,
                                           node_cmap=cmap)
        file_name = self.output_name + '_sensor_locations.png'
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, file_name))

        leak_isolation = dict()
        for key, item in self.f_name_map.items():
            leak_isolation[item[2:]] = self.fault_enc.get(int(key[1:]), int(key[1:]))
        wntr.graphics.network.plot_network(self.wn, node_attribute=leak_isolation, add_colorbar=False,
                                           node_size=node_size + 30, node_labels=node_labels, link_labels=link_labels,
                                           node_cmap='nipy_spectral')
        file_name = self.output_name + '_leak_isolation.png'
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, file_name))

        leaks = [leak[2:] for key, leak in self.f_name_map.items()]
        leak_ind = dict()
        for leak in leaks:
            leak_ind[leak] = 1
        cmap = matplotlib.colors.ListedColormap(['red'])
        wntr.graphics.network.plot_network(self.wn, node_attribute=leak_ind, add_colorbar=False,
                                           node_size=node_size + 10, node_labels=node_labels, link_labels=link_labels,
                                           node_cmap=cmap)
        file_name = self.output_name + '_leak_locations.png'
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, file_name))

    def plot_eqation_graph(self, eg, pos, parts, cmap, nr):
        plt.figure(figsize=(5, 5))
        nx.draw_networkx(eg, pos=pos, with_labels=False, node_color=parts, node_size=40, cmap=cmap)
        plt.axis('off')

        handles = [mpatches.Patch(color=cmap(i), label='$\Sigma_' + str(i) + '$') for i in range(self.n_subsystems)]
        plt.legend(handles=handles)
        file_name = self.output_name + '_step_' + str(nr) + '_dec.png'
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, file_name))

    def encode_for_max_similarity(self, parts, old_parts):
        elems = list(set(parts))
        elems.sort()
        perms = permutations(elems)

        encoded_parts = None
        best_cost = None
        for perm in perms:
            perm_parts = [perm[parts[i] - 1] for i in range(len(parts))]
            cost = sum(i != j for i, j in zip(perm_parts, old_parts))
            if best_cost is None or cost < best_cost:
                best_cost = cost
                encoded_parts = perm_parts

        return encoded_parts

    def read_steps_from_file(self):
        file_name = self.output_name + '_steps.json'
        with open(os.path.join(self.output_folder, file_name), 'r') as f:
            data = json.load(f)

        fmsos = None
        subsystems_res = None
        best_cost = None
        fmsos_c = None
        subsystems_c = None
        i = 0
        fmsos_steps = []
        subsystems_steps = []
        while i < len(data):
            if i % 4 == 2:
                fmsos_c = data[i]
                print('number of fmsos: ', len(fmsos_c))
                cost = data[i + 1]
                print(cost)
                if best_cost is None or cost <= best_cost:
                    best_cost = cost
                    fmsos = fmsos_c
                    subsystems_res = subsystems_c
                    fmsos_steps.append(fmsos)
            else:
                subsystems_c = data[i]
                cost = data[i + 1]
                print(cost)
                if best_cost is None or cost <= best_cost:
                    best_cost = cost
                    subsystems_res = subsystems_c
                    fmsos = fmsos_c
                    subsystems_steps.append({int(key): value for key, value in subsystems_c.items()})
            i += 2
        if subsystems_res:
            subsystems_res = {int(key): value for key, value in subsystems_res.items()}

        self.fmsos = fmsos
        self.subsystems_res = subsystems_res
        self.fmsos_steps = fmsos_steps
        self.subsystems_steps = subsystems_steps
        self.best_cost = best_cost

    def plot_steps(self, **kwargs):
        fmsos_eqs = [['e' + str(e) for e in el] for el in self.msos]
        eg = graph_creation.graph_from_fmsos(fmsos_eqs, self.sm_data["model"], self.sm_data)
        cmap = cm.get_cmap('Dark2', self.n_subsystems)
        pos = nx.kamada_kawai_layout(eg)
        edgecuts, parts = decompose.split_equations(eg, self.n_subsystems, **kwargs)
        self.plot_eqation_graph(eg, pos, parts, cmap, 0)

        for i, subsystems in enumerate(self.subsystems_steps):
            print(i)
            eq_assignment = decompose.subsystems_to_eq_assignment(subsystems)
            old_parts = parts
            parts = decompose.parts_from_eq_assignment(eq_assignment)
            parts = self.encode_for_max_similarity(parts, old_parts)
            self.plot_eqation_graph(eg, pos, parts, cmap, i + 1)
