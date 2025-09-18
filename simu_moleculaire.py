import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- paramètres physiques ---
M = 1.0
temperature = 2.0

# potentiel
def V(q):
    return (q**2)/2

# force
def F(q):
    return -q

# energie cinetique
def W(p):
    return (p**2)/2*M

def euler_explicite(q, p, force, dt, gamma): 
    if gamma is not None:
        sigma = np.sqrt(2 * gamma * temperature * dt)
        pn = p + force(q) * dt - gamma * p * dt
    else:
        pn = p + force(q) * dt
    qn = q + pn * dt
    return qn, pn

def verlet(q, p, force, dt, gamma):
    if gamma is not None:
        g = np.random.normal(0, 1)
        p_quarter = p - gamma*p*dt + np.sqrt(2*gamma*temperature*dt) * g
        p_half = p_quarter + 0.5 * force(q) * dt 
        q_new = q + p_half * dt * M
        p_new = p_half + 0.5*force(q_new)*dt 
    else:
        p_half = p + 0.5 * force(q) * dt
        q_new = q + p_half * dt
        p_new = p_half + 0.5 * force(q_new) * dt
    return q_new, p_new

def euler_symplectiqueA(q, p, force, dt, gamma=None):
    qn = q + p*M*dt
    pn = p + force(qn)*dt
    return pn, qn  

def euler_symplectiqueB(q, p, force, dt, gamma=None):
    pn = p + force(q)*dt 
    qn = q + pn*M*dt 
    return pn, qn  


def film(q0, p0, force, n_steps, dt, gamma, schema): # animation
    q, p = q0, p0
    t = 0

    fig, ax = plt.subplots()

    xlim = 3.0
    X = np.linspace(-xlim, xlim, 400)

    courbe_potentiel, = ax.plot(X, V(X), color='black')
    atom, = ax.plot([q0], [V(q0)], 'o', ms=12, color='red')

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(0.0, 5.0)

    energy_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        atom.set_data([], [])
        energy_text.set_text('')
        return atom, energy_text

    def animate(frame):
        nonlocal q, p, force, t, dt
        q, p = schema(q, p, force, dt, gamma)
        t += dt

        atom.set_data([q], [V(q)])

        total_energy = V(q) + W(p)
        energy_text.set_text(f'H = {total_energy:.4f}, t = {t:.2f}')

        return atom, energy_text

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=n_steps,
        init_func=init,
        interval=20,
        blit=True,
        repeat=False
    )
    plt.show()
    return ani

# --- paramètres de simulation ---
dt = 0.01
n_steps = 500
gamma = 1

# --- conditions initiales ---
q0 = 0.0
sigma_p = np.sqrt(temperature * M)
p0 = np.random.randn() * sigma_p

# --- simulation : uncomment schéma d'intégration souhaité ---

#schema = euler_explicite
schema = verlet
#schema = euler_symplectiqueA 
#schema = euler_symplectiqueB 


# --- simulation with recording ---

q_vals = [q0]
p_vals = [p0]
q, p = q0, p0
for _ in range(n_steps):
    q, p = schema(q, p, F, dt, gamma)
    q_vals.append(q)
    p_vals.append(p)

ani = film(q0, p0, F, n_steps, dt, gamma, schema)

# --- tracé de l'énergie au cours du temps --- 

time = np.linspace(0, n_steps*dt, n_steps+1)
q_arr = np.array(q_vals)
p_arr = np.array(p_vals)

plt.figure(figsize=(6,4))
plt.plot(time, V(q_arr) + 0.5 * p_arr**2, color="black", label="énergie totale")
plt.plot(time, 0.5 * p_arr**2, color="red", label="énergie cinétique")
plt.plot(time, V(q_arr), color="blue", label="énergie potentielle")
plt.plot(time, q_arr, color="green", label="q")
plt.legend()
plt.title("énergie du système au cours du temps")
plt.xlabel('Temps')
plt.ylabel('Energie')
plt.show()

plt.hist(q_arr, bins = 1000)
plt.show()
