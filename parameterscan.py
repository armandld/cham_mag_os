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

Omega = 0.0
kappa = 0.0
m = 0.0
L = 0.0
B1 = 0.0
B0 = 0.0
mu = 0.0
theta0 = 0.0
thetadot0 = 0.0
sampling = 0.0
N_excit = 0.0
Nperiod = 0.0
nsteps = 0.0

nsteps_values = [50, 100, 200, 500, 700, 1000, 2000, 5000, 10000]  # Nombre de pas par période

valeurs = lire_configuration()

def actualise_valeur():
    global Omega, kappa, m, L, B1, B0, mu, theta0, thetadot0, sampling, N_excit, Nperiod, nsteps
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

def ecrire_valeur(nom,valeur):
    global valeurs
    valeurs[nom] = valeur
    ecrire_configuration(valeurs)
    actualise_valeur()




# Question 1

paramstr = 'nsteps'  # Paramètre à scanner
param = nsteps_values


ecrire_valeur("B1",0)
ecrire_valeur("kappa",0)
ecrire_valeur("theta0",1e-6)
ecrire_valeur("thetadot0",0)
ecrire_valeur("N_excit",0)

outputs = []  # Liste pour stocker les fichiers de sortie
errors = []  # Liste pour stocker les erreurs

omega_0 = np.sqrt(12*B0*mu/(m*L*L))

T0 = 2 * np.pi / Omega  # Période théorique

tfin = Nperiod * T0

A = np.sqrt(theta0**2 + m * L * L * thetadot0**2 / (12 * B0 * mu))

phi = np.pi/2

if (thetadot0 != 0):
	phi = np.arctan(-theta0/thetadot0*omega_0)

'''
# Question 1

for i, nsteps in enumerate(param):
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

# Tracé de l'étude de convergence
plt.figure()
dt_values = T0 / np.array(nsteps_values)
plt.loglog(dt_values, errors, marker='o', linestyle='-', label="Erreur numérique")
plt.xlabel("Δt [s]")
plt.ylabel("Erreur δ")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Convergence de l'erreur en fonction de Δt")

n_order = 2

plt.figure()
dt_values = T0 / np.array(nsteps_values)
plt.plot(dt_values**n_order, errors, marker='o', linestyle='-', label="Erreur numérique")
plt.xlabel("Δt [s]")
plt.ylabel("Erreur δ")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Convergence de l'erreur en fonction de Δt")


# Question 2

ecrire_valeur("Omega",2*omega_0)
ecrire_valeur("B1",0.002)
ecrire_valeur("kappa",0)
ecrire_valeur("N_excit",100)
ecrire_valeur("theta0",1e-3)
ecrire_valeur("thetadot0",0)

nsteps_values = [100,500,2000,5000]

errors = []

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

    # Tracé de θ(t)
    plt.figure()
    plt.plot(t, theta, linestyle='dashed')
    plt.xlabel('Temps [s]')
    plt.ylabel('$\Theta$ [rad]')
    plt.legend()
    plt.title(f'Évolution de θ(t) pour nsteps={nsteps}')
    plt.grid()
    

    # Tracé de l'orbite dans l’espace de phase
    plt.figure()
    plt.plot(theta, thetadot, label='Simulation')
    plt.xlabel('$\Theta$ [rad]')
    plt.ylabel('$\dot{\Theta}$ [rad/s]')
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

    # Solution analytique
    theta_a = A*np.sin(np.sqrt(12*B0*mu/(m*L*L))*t+phi)
    thetadot_a = A*np.sqrt(12*B0*mu/(m*L*L))*np.cos(np.sqrt(12*B0*mu/(m*L*L))*t+phi)

    # Calcul de l'erreur à tfin
    delta = np.sqrt(omega_0**2 * (theta[-1] - theta_a[-1])**2 + (thetadot[-1] - thetadot_a[-1])**2)
    errors.append(delta)

# Tracé de l'étude de convergence
plt.figure()
dt_values = T0 / np.array(nsteps_values)
plt.loglog(dt_values, errors, marker='o', linestyle='-', label="Erreur numérique")
plt.xlabel("Δt [s]")
plt.ylabel("Erreur δ")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Convergence de l'erreur en fonction de Δt")

n_order = 2

plt.figure()
dt_values = T0 / np.array(nsteps_values)
plt.plot(dt_values**n_order, errors, marker='o', linestyle='-', label="Erreur numérique")
plt.xlabel("Δt [s]")
plt.ylabel("Erreur δ")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Convergence de l'erreur en fonction de Δt")


# Question 3

ecrire_valeur("Omega",2*omega_0)
ecrire_valeur("B1",0.002)
ecrire_valeur("kappa",0)
ecrire_valeur("N_excit",10000)
ecrire_valeur("theta0",1e-3)
ecrire_valeur("thetadot0",0)

nsteps_values = [10,20,50,100]


for i, nsteps in enumerate(nsteps_values):
    couples_phase = []
    output_file = f"{paramstr}={nsteps}.out"
    outputs.append(output_file)
    cmd = f"./{executable} {input_filename} {paramstr}={nsteps} output={output_file}"
    print(cmd)
    subprocess.run(cmd, shell=True)
    print('Simulation terminée.')

    # Chargement des données
    data = np.loadtxt(output_file)
    tfin = data[-1, 0]
    for k in range(0,len(data[:,1]),nsteps):
        k = int(k)
        theta = ((data[k, 1]+np.pi) % (2*np.pi))-np.pi
        thetadot = data[k, 2]
        couples_phase.append([theta,thetadot])

    couples_phase = np.array(couples_phase)
        # Tracé de la section de Poincaré
    plt.figure()
    plt.scatter(couples_phase[:,0], couples_phase[:,1], label='Simulation',s = 2)
    plt.xlabel('$\Theta$ [rad]')
    plt.ylabel('$\dot{\Theta}$ [rad/s]')
    plt.legend()
    plt.title(f'Sections de Poincaré pour nsteps ={nsteps}')
    plt.grid()

'''
# Question 5

ecrire_valeur("Omega",2*omega_0)
ecrire_valeur("B1",0.018)
ecrire_valeur("kappa",2e-5)
ecrire_valeur("B0",0.01)

ecrire_valeur("N_excit",10000)

nsteps_values = [10]

theta0s = np.linspace(0, 1, 2)  # Plage des valeurs de theta0
thetadot0s = np.linspace(0, 20, 2)  # Plage des valeurs de thetadot0

# Boucle sur les différentes valeurs de theta0 et thetadot0
for theta0 in theta0s:
    for thetadot0 in thetadot0s:
        ecrire_valeur("theta0", theta0)
        ecrire_valeur("thetadot0", thetadot0)

        couples_phase = []
        output_file = f"{paramstr}_theta0={theta0}_thetadot0={thetadot0}.out"
        outputs.append(output_file)
        cmd = f"./{executable} {input_filename} {paramstr}={nsteps} output={output_file}"
        print(cmd)
        subprocess.run(cmd, shell=True)
        print('Simulation terminée.')

        # Chargement des données
        data = np.loadtxt(output_file)
        tfin = data[-1, 0]
        
        # Récupération des couples (theta, thetadot)
        for k in range(0, len(data[:, 1])):
            theta = ((data[k, 1] + np.pi) % (2 * np.pi)) - np.pi  # Pour ajuster theta dans l'intervalle [-pi, pi]
            thetadot = data[k, 2]
            couples_phase.append([theta, thetadot])

        couples_phase = np.array(couples_phase)

        # Tracé de la section de Poincaré sur le même graphique
        plt.scatter(couples_phase[:, 0], couples_phase[:, 1],s=2)

# Ajout des labels, titre et grille une seule fois après la boucle
plt.xlabel('$\\Theta$ [rad]')
plt.ylabel('$\\dot{\\Theta}$ [rad/s]')
plt.legend()
plt.title(f'Sections de Poincaré pour différents $\\theta_0$ et $\\dot{{\\theta}}_0$')
plt.grid()

plt.show()
