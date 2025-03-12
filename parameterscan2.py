import numpy as np
import subprocess
import matplotlib.pyplot as plt
import pdb
import os

# Parameters
# TODO adapt to what you need (folder path executable input filename)
executable = 'Ex2_2025_student'  # Name of the executable (NB: .exe extension is required on Windows)
repertoire = r"/home/chatelin/Desktop/MyFiles/Aiguille_aimantee_physNum/"
os.chdir(repertoire)


'global_params = {}'
input_filename = 'configuration.in'  # Name of the input file

nsteps_values = [100]  # Nombre de pas par période
'''
def read_config(filename):
    params = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.split('//')[0].strip()
            if '=' in line:
                key, value = line.split('=')
                params[key.strip()] = float(value.strip()) if '.' in value or 'e' in value else int(value.strip())
    return params

global_params = read_config(input_filename)

# Extraction des paramètres
Omega = global_params.get('Omega', 1.0)
kappa = global_params.get('kappa', 0.0)
m = global_params.get('m', 0.075)
L = global_params.get('L', 0.08)
B1 = global_params.get('B1', 0.0)
B0 = global_params.get('B0', 0.01)
mu = global_params.get('mu', 0.2)
theta0 = global_params.get('theta0', 1e-6)
thetadot0 = global_params.get('thetadot0', 0.0)
N_period = global_params.get('Nperiod', 3)
'''

'Omega = 1.0'
m = 0.075
L = 0.08
B0 = 0.01
B1 = 0.02
mu = 0.2
theta0 = 1e-6
thetadot0 = 0
N_period = 3
N_excit = 100
kappa = 0




paramstr = 'nsteps'  # Paramètre à scanner
param = nsteps_values


outputs = []  # Liste pour stocker les fichiers de sortie
errors = []  # Liste pour stocker les erreurs

omega_0 = np.sqrt(12*B0*mu/(m*L*L))

Omega = 2*omega_0
T0 = 2 * np.pi / Omega  # Période théorique
tfin = N_period * T0

A = np.sqrt(theta0**2 + m * L * L * thetadot0**2 / (12 * B0 * mu))

phi = np.pi/2

if (thetadot0 != 0):
	phi = np.arctan(-theta0/thetadot0*omega_0)

for i, nsteps in enumerate(nsteps_values):
    output_file = f"{paramstr}={nsteps}.out"
    outputs.append(output_file)
    cmd = f"./{executable} {input_filename} {paramstr}={nsteps} output={output_file}"
    print(cmd)
    subprocess.run(cmd, shell=True)
    print('Simulation terminée.')

    # Chargement des données
    data = np.loadtxt(output_file)
    t = data[:, 0]
    theta = data[:, 1]
    thetadot = data[:, 2]
    Emec = data[:, 3]
    Pnc = data[:, 4]

    # Solution analytique
    
    theta_a = A*np.sin(np.sqrt(12*B0*mu/(m*L*L))*t+phi)
    thetadot_a = A*np.sqrt(12*B0*mu/(m*L*L))*np.cos(np.sqrt(12*B0*mu/(m*L*L))*t+phi)

    # Calcul de l'erreur à tfin
    delta = np.sqrt(omega_0**2 * (theta[-1] - theta_a[-1])**2 + (thetadot[-1] - thetadot_a[-1])**2)
    errors.append(delta)
    
    # Tracé de θ(t)
    plt.figure()
    plt.plot(t, theta_a, linestyle='dashed', label='Théorique')
    plt.xlabel('Temps [s]')
    plt.ylabel('θ [rad]')
    plt.legend()
    plt.title(f'Évolution de θ(t) pour nsteps={nsteps}')
    plt.grid()
    

    # Tracé de l'orbite dans l’espace de phase
    plt.figure()
    plt.plot(theta, thetadot, label='Simulation')
    plt.xlabel('θ [rad]')
    plt.ylabel('˙θ [rad/s]')
    plt.legend()
    plt.title(f'Espace de phase (θ, ˙θ) pour nsteps={nsteps}')
    plt.grid()
    

    # Tracé de l'énergie mécanique Emec(t)
    plt.figure()
    plt.plot(t, Emec, label='Emec')
    plt.xlabel('Temps [s]')
    plt.ylabel('Énergie mécanique [J]')
    plt.legend()
    plt.title(f'Évolution de l\'énergie mécanique pour nsteps={nsteps}')
    plt.grid()
    

    # Comparaison dEmec/dt avec Pnc(t)
    dEmec_dt = m*L*L*thetadot*np.gradient(thetadot,t)/12-mu*Omega*B1*np.cos(Omega*t)+kappa*np.gradient(thetadot,t)
    plt.figure()
    plt.plot(t, dEmec_dt, label='dEmec/dt')
    plt.plot(t, Pnc, label='Pnc')
    plt.xlabel('Temps [s]')
    plt.ylabel('Puissance [W]')
    plt.legend()
    plt.title(f'Comparaison dEmec/dt et Pnc(t) pour nsteps={nsteps}')
    plt.grid()
    

n_order = 2

# Tracé de l'étude de convergence
plt.figure()
dt_values = T0 / np.array(nsteps_values)
plt.loglog(dt_values, errors, marker='o', linestyle='-', label="Erreur numérique")
plt.xlabel("Δt [s]")
plt.ylabel("Erreur δ")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Convergence de l'erreur en fonction de Δt")


plt.figure()
dt_values = T0 / np.array(nsteps_values)
plt.plot(dt_values**n_order, errors, marker='o', linestyle='-', label="Erreur numérique")
plt.xlabel("Δt [s]")
plt.ylabel("Erreur δ")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Convergence de l'erreur en fonction de Δt")





plt.show()



















































alpha = np.array([0])
vy0 = -5.28 - np.geomspace(1e-2,1,5) # np.linspace(-6,-5,50)

nsimul = len(alpha)  # Number of simulations to perform

tfin = 259200  # TODO: Verify that the value of tfin is EXACTLY the same as in the input file

paramstr = 'alpha'  # Parameter name to scan
param = alpha  # Parameter values to scan

# Simulations
outputs = []  # List to store output file names
convergence_list = []

for i in range(nsimul):
    output_file = f"{paramstr}={param[i]}.out"
    outputs.append(output_file)
    cmd = f"./{repertoire}{executable} {input_filename} {paramstr}={param[i]:.15g} output={output_file}"
    cmd = f"./{executable} {input_filename} {paramstr}={param[i]:.15g} output={output_file}"
    print(cmd)
    subprocess.run(cmd, shell=True)
    print('Done.')

error = np.zeros(nsimul)

lw = 1.5
fs = 16

fig, ax = plt.subplots(constrained_layout=True)
plt.grid(True, linestyle="--", alpha=0.3)
for i in range(nsimul):  # Iterate through the results of all simulations
    data = np.loadtxt(outputs[i])  # Load the output file of the i-th simulation

    dist_sl = np.sqrt((data[:,3]-3.80321e+08)**2 + data[:,4]**2)
    t = data[:, 0]
    #convergence_list.append(En)
    # TODO compute the error for each simulation
    #convergence_list = np.array(convergence_list)
    ax.plot(t, dist_sl)
ax.axhline(1737100)
ax.set_xlabel('$t$ [s]', fontsize=fs)
ax.set_ylabel('$d_{SL}$ [m]', fontsize=fs) 


fig, ax = plt.subplots(constrained_layout=True)
#plt.grid(True, linestyle="--", alpha=0.3)
fig2, ax2 = plt.subplots(constrained_layout=True)
#plt.grid(True, linestyle="--", alpha=0.3)
fig3, ax3 = plt.subplots(constrained_layout=True)
#plt.grid(True, linestyle="--", alpha=0.3)

for i in range(nsimul):  # Iterate through the results of all simulations
    data = np.loadtxt(outputs[i])  # Load the output file of the i-th simulation
    t = data[:, 0]

    vx = data[-1, 1]  # final position, velocity, energy
    vy = data[-1, 2]
    xx = data[-1, 3]
    yy = data[-1, 4]
    En = data[-1, 5]
    # TODO compute the error for each simulation
    error[i] =  np.abs(xx)
    lw = 1.5
    fs = 16
    convergence_list = np.array(convergence_list)

    ax.plot(data[:, 3], data[:, 4])
    ax2.plot(data[:, 3], data[:, 4])
    #fig3, ax3 = plt.subplots(constrained_layout=True)
    plt.grid(True, linestyle="-", alpha=1)
    ax3.plot(t,data[:,5])
    ax3.set_xlabel('$t$ [s]', fontsize=fs)
    ax3.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax3.xaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    # Appliquer la notation scientifique à l'axe X
    ax3.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax3.set_ylabel('E$_{mec}$/m$_s$ [J/kg]', fontsize=fs)


# Ajouter un disque
disque = plt.Circle((3.80321e+08, 0), 1737100, color='black')
disque2 = plt.Circle((3.80321e+08, 0), 1737100, color='black')
disque3 = plt.Circle((341878931, 0), 500000, color='red')
# Ajouter le disque au graphe
ax.add_patch(disque)
ax.add_patch(disque3)
ax2.add_patch(disque2)
ax.set_xlabel('$x\'$ [m]', fontsize=fs)
ax.set_ylabel('$y\'$ [m]', fontsize=fs)
ax2.set_xlabel('$x\'$ [m]', fontsize=fs)
ax2.set_ylabel('$y\'$ [m]', fontsize=fs)

# Ajuster les limites du graphique
ax.set_xlim(3.4e8,3.9e8)
ax.set_ylim(-1e7,0.31e8)
ax2.set_xlim(3.7e8,3.85e8)
ax2.set_ylim(-6e6,6e6)
ax.set_aspect('equal')  # Assurer un aspect circulaire
ax2.set_aspect('equal')  

plt.show()

