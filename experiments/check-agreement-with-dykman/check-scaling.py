from metastable.eom import EOM, Params


eom = EOM(
    params=Params(
        kappa=2.0,
        chi=0.0,
        delta=0.0,
        epsilon=0.0,
    )
)

print(eom.y_dot_classical_expr)
