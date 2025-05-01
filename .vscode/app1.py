import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import streamlit as st
import os

# Configure the Streamlit page
st.set_page_config(page_title="Get Phase-Space", page_icon=":bar_chart:", layout="wide")

def lagrangian_phase_plot(L_str, params, t_max=20, y0=None, save_path=None):
    t = sp.Symbol('t')

    # Define time-dependent functions
    q1 = sp.Function('q1')(t)
    q2 = sp.Function('q2')(t)
    dq1 = q1.diff(t)
    dq2 = q2.diff(t)

    q1_sym, q2_sym = sp.symbols('q1 q2')
    v1_sym, v2_sym = sp.symbols('v1 v2')

    # Replace q1, v1, q2, v2 in user input
    L_expr = sp.sympify(
        L_str.replace('q1', f'({q1})')
             .replace('v1', f'({dq1})')
             .replace('q2', f'({q2})')
             .replace('v2', f'({dq2})')
    )

    # Substitute parameter values
    param_syms = {sp.Symbol(k): v for k, v in params.items()}
    L_expr = L_expr.subs(param_syms)

    # Detect DOFs
    dof_vars = []
    if 'q1' in L_str or 'v1' in L_str:
        dof_vars.append(('q1', q1, dq1, q1_sym, v1_sym, 'a1'))
    if 'q2' in L_str or 'v2' in L_str:
        dof_vars.append(('q2', q2, dq2, q2_sym, v2_sym, 'a2'))

    if len(dof_vars) == 0:
        st.error("No degrees of freedom detected in the Lagrangian.")
        return

    # Default initial conditions
    if y0 is None:
        y0 = [1.0 if i % 2 == 0 else 0.0 for i in range(len(dof_vars) * 2)]

    # Build ODEs
    subs_dict = {}
    ode_eqs = []
    p_funcs = []

    for i, (name, q, dq, q_sym, v_sym, acc_label) in enumerate(dof_vars):
        dL_dq = sp.diff(L_expr, q)
        dL_ddq = sp.diff(L_expr, dq)
        ddt_dL_ddq = sp.diff(dL_ddq, t).doit()
        EL = ddt_dL_ddq - dL_dq

        acc = sp.Symbol(acc_label)
        subs_dict[q] = q_sym
        subs_dict[dq] = v_sym
        subs_dict[sp.Derivative(v_sym, t)] = acc

        EL_sub = EL.subs(subs_dict)

        # ✅ Safer solve
        try:
            acc_expr = sp.solve(sp.Eq(EL_sub, 0), acc)
            if not acc_expr:
                raise ValueError("No solution found for acceleration.")
            ode_eqs.append(sp.lambdify(
                [*map(lambda x: x[3], dof_vars), *map(lambda x: x[4], dof_vars)],
                acc_expr[0], 'numpy'
            ))
        except Exception as e:
            st.error(f"❌ Failed to solve for acceleration {acc_label}")
            st.code(f"Equation: {sp.Eq(EL_sub, 0)}")
            st.exception(e)
            return

        # Get momentum function
        p_expr = sp.diff(L_expr, dq).subs(subs_dict)
        p_func = sp.lambdify(v_sym, p_expr, 'numpy')
        p_funcs.append(p_func)

    def ode(t, y):
        coords = y[::2]
        vels = y[1::2]
        accs = [f(*coords, *vels) for f in ode_eqs]
        dydt = []
        for v, a in zip(vels, accs):
            dydt.append(v)
            dydt.append(a)
        return dydt

    # Integrate the system
    t_eval = np.linspace(0, t_max, 1000)
    sol = solve_ivp(ode, [0, t_max], y0, t_eval=t_eval)
    sol_y = sol.y

    # Plot phase-space
    fig, axs = plt.subplots(1, len(dof_vars), figsize=(6 * len(dof_vars), 5))
    if len(dof_vars) == 1:
        axs = [axs]

    for i, (name, _, _, _, _, _) in enumerate(dof_vars):
        q_vals = sol_y[i * 2]
        v_vals = sol_y[i * 2 + 1]
        p_vals = p_funcs[i](v_vals)

        axs[i].plot(q_vals, p_vals)
        axs[i].set_xlabel(f"{name}")
        axs[i].set_ylabel(f"p_{name}")
        axs[i].set_title(f"Phase Space: {name} vs p_{name}")
        axs[i].grid(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        st.pyplot(fig)

# 🔘 Streamlit UI
def streamlit_app():
    st.title("Phase-Space Trajectory Generator")

    L_str = st.text_input("Enter Lagrangian (in terms of q1, v1, q2, v2):",
                          "0.5*m*v1**2 + 0.5*m*v2**2 - 0.5*k*q1**2 - 0.5*k*q2**2 - 0.5*k_c*(q1 - q2)**2")

    m = st.number_input("Mass (m)", value=1.0)
    k = st.number_input("Spring Constant (k)", value=1.0)
    k_c = st.number_input("Coupling Constant (k_c)", value=0.3)
    params = {'m': m, 'k': k, 'k_c': k_c}

    col1, col2 = st.columns(2)
    with col1:
        q1_0 = st.number_input("Initial q1", value=1.0)
        v1_0 = st.number_input("Initial v1", value=0.0)
    with col2:
        q2_0 = st.number_input("Initial q2", value=-1.0)
        v2_0 = st.number_input("Initial v2", value=0.0)

    t_max = st.number_input("Max Time", value=20.0)

    if st.button("Generate Phase-Space Plot"):
        y0 = []
        if 'q1' in L_str or 'v1' in L_str:
            y0 += [q1_0, v1_0]
        if 'q2' in L_str or 'v2' in L_str:
            y0 += [q2_0, v2_0]
        lagrangian_phase_plot(L_str, params, t_max=t_max, y0=y0)

if __name__ == "__main__":
    streamlit_app()
