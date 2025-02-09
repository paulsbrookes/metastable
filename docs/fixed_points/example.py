from metastable.eom import EOM, Params


params = Params(epsilon=1000.0, delta=1.0, chi=1.0)
eom = EOM(params)

print(eom.y_dot_classical_expr)
...