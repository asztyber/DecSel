import faultdiagnosistoolbox as fdt


def calculate_tests(data, mso=False):
    sm = data['model']
    relsX = [sm[e] for e in sorted(sm.keys(), key=lambda x: int(x[1:]))]
    model_def = {
        'type': 'VarStruc',
        'x': data['unknown'],
        'z': data['known'],
        'f': data['faults'],
        'rels': relsX
        }

    model = fdt.DiagnosisModel(model_def)
    model.Lint()

    mtes = model.MTES()

    isol = model.IsolabilityAnalysisArrs(mtes)
    mtes = [[int(x) for x in list(m)] for m in mtes]
    unambiguity_groups = [str(el) for el in list(isol.sum(axis=1))]

    msos = None
    if mso:
        msos = model.MSO()
        msos = [[int(x) for x in list(m)] for m in msos]

    return mtes, unambiguity_groups, msos
