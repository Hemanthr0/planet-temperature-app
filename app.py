import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import streamlit as st

# Function to generate phase-space plot from Lagrangian
def lagrangian_2dof_phase_plot(L_str, params, t_max=20, y0=[1.0, 0.0, -1.0, 0.0], save_path=None):
    t = sp.Symbol('t')

    # Define generalized coordinates and velocities
    q1 = sp.Function('q1')(t)
    q2 = sp.Function('q2')(t)
    dq1 = q1.diff(t)
    dq2 = q2.diff(t)

    q1_sym, q2_sym = sp.symbols('q1 q2')
    v1_sym, v2_sym = sp.symbols('v1 v2')

    # Parse Lagrangian string
    L_expr = sp.sympify(L_str.replace('q1', f'{q1}')
                                .replace('q2', f'{q2}')
                                .replace('v1', f'{dq1}')
                                .replace('v2', f'{dq2}'))

    param_syms = {sp.Symbol(k): v for k, v in params.items()}
    L_expr = L_expr.subs(param_syms)

    def get_eom(q, dq):
        dL_dq = sp.diff(L_expr, q)
        dL_ddq = sp.diff(L_expr, dq)
        ddt_dL_ddq = sp.diff(dL_ddq, t).doit()
        return ddt_dL_ddq - dL_dq

    EL1 = get_eom(q1, dq1)
    EL2 = get_eom(q2, dq2)

    subs_dict = {
        q1: q1_sym, q2: q2_sym,
        dq1: v1_sym, dq2: v2_sym,
        sp.Derivative(v1_sym, t): sp.Symbol('a1'),
        sp.Derivative(v2_sym, t): sp.Symbol('a2')
    }

    EL1_sub = EL1.subs(subs_dict)
    EL2_sub = EL2.subs(subs_dict)

    a1 = sp.Symbol('a1')
    a2 = sp.Symbol('a2')
    a1_expr = sp.solve(sp.Eq(EL1_sub, 0), a1)[0]
    a2_expr = sp.solve(sp.Eq(EL2_sub, 0), a2)[0]

    p1_expr = sp.diff(L_expr, dq1).subs(subs_dict)
    p2_expr = sp.diff(L_expr, dq2).subs(subs_dict)

    a1_func = sp.lambdify((q1_sym, q2_sym, v1_sym, v2_sym), a1_expr, 'numpy')
    a2_func = sp.lambdify((q1_sym, q2_sym, v1_sym, v2_sym), a2_expr, 'numpy')
    p1_func = sp.lambdify((v1_sym,), p1_expr, 'numpy')
    p2_func = sp.lambdify((v2_sym,), p2_expr, 'numpy')

    def ode(t, y):
        q1, v1, q2, v2 = y
        return [
            v1,
            a1_func(q1, q2, v1, v2),
            v2,
            a2_func(q1, q2, v1, v2)
        ]

    t_eval = np.linspace(0, t_max, 1000)
    sol = solve_ivp(ode, [0, t_max], y0, t_eval=t_eval)
    t_vals = sol.t
    q1_vals, v1_vals, q2_vals, v2_vals = sol.y
    p1_vals = p1_func(v1_vals)
    p2_vals = p2_func(v2_vals)

    # === 🖼 Plotting ===
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(q1_vals, p1_vals)
    axs[0].set_xlabel("q1")
    axs[0].set_ylabel("p1")
    axs[0].set_title("Phase Space: q1 vs p1")
    axs[0].grid(True)

    axs[1].plot(q2_vals, p2_vals)
    axs[1].set_xlabel("q2")
    axs[1].set_ylabel("p2")
    axs[1].set_title("Phase Space: q2 vs p2")
    axs[1].grid(True)

    plt.tight_layout()

    # Save to file if path is given
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        return save_path
    else:
        # Display plot in Streamlit
        st.pyplot(fig)

# Streamlit interface for Lagrangian input and display
def streamlit_app():
    st.title("Phase-Space Trajectory Generator")

    # Input the Lagrangian Equation
    L_str = st.text_input("Enter the Lagrangian (in terms of q1, q2, v1, v2):",
                          "0.5*m*v1**2 + 0.5*m*v2**2 - 0.5*k*q1**2 - 0.5*k*q2**2 - 0.5*k_c*(q1 - q2)**2")

    # Input parameters
    m = st.number_input("Mass (m)", value=1.0)
    k = st.number_input("Spring Constant (k)", value=1.0)
    k_c = st.number_input("Coupling Constant (k_c)", value=0.3)

    params = {'m': m, 'k': k, 'k_c': k_c}

    # Input initial conditions
    q1_0 = st.number_input("Initial q1", value=1.0)
    v1_0 = st.number_input("Initial v1", value=0.0)
    q2_0 = st.number_input("Initial q2", value=-1.0)
    v2_0 = st.number_input("Initial v2", value=0.0)

    initial_conditions = [q1_0, v1_0, q2_0, v2_0]

    # Input maximum time (t_max)
    t_max = st.number_input("Max Time (t_max)", value=20.0)

    # Button to generate the plot
    if st.button("Generate Phase-Space Plot"):
        # Generate the phase-space plot and display it
        lagrangian_2dof_phase_plot(L_str, params, t_max=t_max, y0=initial_conditions)

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()
