# DecSel

This repository contains the code for the paper:

Anna Sztyber-Betley, Elodie Chanthery, Louise Travé-Massuyès, Carlos Gustavo Pérez-Zuñiga
*Diagnosis test selection for distributed systems under communication and privacy constraints*, submitted to Applied Intelligence.

## Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

BILP solver uses Gurobi. To run the code you need to have active Gurobi license and place the licence file in the required folder. 
Instructions can be found [here](https://support.gurobi.com/hc/en-us/articles/13232844297489-How-do-I-set-up-a-Web-License-Service-WLS-license).

To get structural model for the GOY water distrubution network run:
```python
python structural_model_GOY.py
```

Other networks can be found in : [wdn-sa-benchmark](https://github.com/asztyber/wdn-sa-benchmark).

Detailed description of the convertion algorithm can be found in the paper:

Anna Sztyber, Elodie Chanthery, Louise Travé-Massuyès, Carlos Gustavo Pérez-Zuñiga. *Water network benchmarks for structural analysis algorithms in fault diagnosis*. 33rd International Workshop on Principle of Diagnosis – DX 2022,  Sep 2022, Toulouse, France. [hal-03773713](https://hal.science/hal-03773713/)

Run DecSel algorithm loop:
```python
python loop_GOY.py
```