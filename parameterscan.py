import numpy as np
import subprocess
import matplotlib.pyplot as plt
import pdb
import os

# Parameters
# TODO adapt to what you need (folder path executable input filename)
executable = 'Ex2_2025_student'  # Name of the executable (NB: .exe extension is required on Windows)
repertoire = r"/Users/Sayu/Desktop/cham_mag_os"
os.chdir(repertoire)


CONFIG_FILE = os.path.join(os.path.dirname(__file__), "configuration.in.example")

input_filename = 'configuration.in.example'  # Name of the input file

nsteps_values = [50, 100, 200, 500, 700, 1000, 2000, 5000, 10000]  # Nombre de pas par période

def lire_configuration():
    config_path = os.path.join(os.path.dirname(__file__), "configuration.in.example")
    configuration = {}
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Le fichier {config_path} n'existe pas.")
    
    with open(config_path, "r", encoding="utf-8") as fichier:
        for ligne in fichier:
            ligne = ligne.strip()
            if ligne and "=" in ligne and not ligne.startswith("#"):
                cle, valeur = ligne.split("=", 1)
                configuration[cle.strip()] = valeur.strip()
    
    return configuration

def ecrire_configuration(nouvelles_valeurs):
    """Écrit les nouvelles valeurs dans le fichier de configuration."""
    if not os.path.exists(CONFIG_FILE):
        raise FileNotFoundError(f"Le fichier {CONFIG_FILE} n'existe pas.")

    lignes_modifiees = []
    
    with open(CONFIG_FILE, "r", encoding="utf-8") as fichier:
        for ligne in fichier:
            ligne_strippée = ligne.strip()
            if ligne_strippée and "=" in ligne_strippée and not ligne_strippée.startswith("#"):
                cle, _ = ligne_strippée.split("=", 1)
                cle = cle.strip()
                if cle in nouvelles_valeurs:
                    ligne = f"{cle} = {nouvelles_valeurs[cle]}\n"
            lignes_modifiees.append(ligne)

    with open(CONFIG_FILE, "w", encoding="utf-8") as fichier:
        fichier.writelines(lignes_modifiees)


valeurs = lire_configuration()

Omega = float(valeurs.get("Omega"))
kappa = float(valeurs.get("kappa"))
m = float(valeurs.get("m"))
L = float(valeurs.get("L"))
B1 = float(valeurs.get("B1"))
B0 = float(valeurs.get("B0"))
mu = float(valeurs.get("mu"))
theta0 = float(valeurs.get("theta0"))
thetadot0 = float(valeurs.get("thetadot0"))
sampling = float(valeurs.get("sampling"))
N_excit = float(valeurs.get("N_excit"))
Nperiod = float(valeurs.get("Nperiod"))
nsteps = float(valeurs.get("nsteps"))

T0 = 2 * np.pi / Omega  # Période théorique

tfin = Nperiod * T0


paramstr = 'nsteps'  # Paramètre à scanner
param = nsteps_values


outputs = []  # Liste pour stocker les fichiers de sortie
errors = []  # Liste pour stocker les erreurs

omega_0 = np.sqrt(12*B0*mu/(m*L*L))

Omega = 2*omega_0

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

    # Solution analytique
    
    theta_a = A*np.sin(np.sqrt(12*B0*mu/(m*L*L))*t+phi)
    thetadot_a = A*np.sqrt(12*B0*mu/(m*L*L))*np.cos(np.sqrt(12*B0*mu/(m*L*L))*t+phi)

    # Calcul de l'erreur à tfin
    delta = np.sqrt(omega_0**2 * (theta[-1] - theta_a[-1])**2 + (thetadot[-1] - thetadot_a[-1])**2)
    errors.append(delta)

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

