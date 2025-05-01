import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lagrangian_string_to_phase_plot(L_str, params, q_var='q', t_max=20, y0=[1.0, 0.0]):
    # Define symbolic variables
    t = sp.Symbol('t')
    q = sp.Function(q_var)(t)
    dq = q.diff(t)

    # Create symbolic parameter symbols for substitution
    param_syms = {sp.Symbol(k): v for k, v in params.items()}

    # Replace string Lagrangian with symbolic form
    L_expr = sp.sympify(L_str.replace('q', f'{q}').replace('v', f'{dq}'))

    # Apply parameter substitutions to Lagrangian
    L_expr = L_expr.subs(param_syms)

    # Euler-Lagrange
    dL_dq = sp.diff(L_expr, q)
    dL_dv = sp.diff(L_expr, dq)
    dd_dt_dL_dv = sp.diff(dL_dv, t).doit()
    EL = dd_dt_dL_dv - dL_dq

    # Substitute q(t) → q, dq/dt → v
    q_s = sp.Symbol('q')
    v_s = sp.Symbol('v')
    EL_sub = EL.subs({q: q_s, dq: v_s, sp.Derivative(v_s, t): sp.Symbol('a')})
    
    # Solve for acceleration
    a_sym = sp.Symbol('a')
    accel_expr = sp.solve(sp.Eq(EL_sub, 0), a_sym)[0]

    # Momentum: p = ∂L/∂v
    p_expr = sp.diff(L_expr, dq).subs({q: q_s, dq: v_s})

    # Convert to numeric functions
    accel_func = sp.lambdify((q_s, v_s), accel_expr, modules='numpy')
    p_func = sp.lambdify((v_s,), p_expr, modules='numpy')

    # Define ODE system
    def ode(t, y):
        q_val, v_val = y
        return [v_val, accel_func(q_val, v_val)]

    # Time vector and integration
    t_eval = np.linspace(0, t_max, 1000)
    sol = solve_ivp(ode, [0, t_max], y0, t_eval=t_eval)
    q_vals, v_vals = sol.y

    # Compute momentum
    p_vals = p_func(v_vals)

    # Plot phase space
    plt.figure(figsize=(8, 6))
    plt.plot(q_vals, p_vals)
    plt.xlabel("Position q")
    plt.ylabel("Momentum p")
    plt.title("Phase Space Diagram")
    plt.grid(True)
    plt.show()

# === Try it with parameters ===
L_string = "0.5*m*v**2 - m*g*L*cos(q)"
params = {'m': 1.0, 'g': 9.81, 'L': 1.0}
lagrangian_string_to_phase_plot(L_string, params, y0=[0.5, 0.0])

