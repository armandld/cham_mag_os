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
C = 0.0
alpha = 0.0
beta = 0.0
gamma = 0.0

nsteps_values = [50, 100, 200, 500, 700, 1000, 2000, 5000, 10000]  # Nombre de pas par période

valeurs = lire_configuration()

def actualise_valeur():
    global Omega, kappa, m, L, B1, B0, mu, theta0, thetadot0, sampling, N_excit, Nperiod, nsteps, C, alpha, beta, gamma
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
    C = float(valeurs.get("C"))
    alpha = float(valeurs.get("alpha"))
    beta = float(valeurs.get("beta"))
    gamma = float(valeurs.get("gamma"))

def ecrire_valeur(nom,valeur):
    global valeurs
    valeurs[nom] = valeur
    ecrire_configuration(valeurs)
    actualise_valeur()

def lancer_simulation(theta0, output_file):
    ecrire_configuration({"theta0": theta0})
    cmd = f"./{executable} {input_filename} output={output_file}"
    subprocess.run(cmd, shell=True)

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
ecrire_valeur("C",0)
ecrire_valeur("alpha",0)
ecrire_valeur("beta",0)
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
plt.loglog(dt_values, errors, marker='v',markersize = 3, color = "black", linestyle='-')
plt.xlabel("Δt [s]")
plt.ylabel("Erreur δ [rad/s]")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Convergence de l'erreur en fonction de Δt")

n_order = 2

plt.figure()
dt_values = T0 / np.array(nsteps_values)
plt.plot(dt_values**n_order, errors, marker='v',markersize = 3, color = "black", linestyle='-')
plt.xlabel("Δt [s]")
plt.ylabel("Erreur δ [rad/s]")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.title("Convergence de l'erreur en fonction de Δt")

plt.show()


# Question 2

ecrire_valeur("C",0)
ecrire_valeur("alpha",0)
ecrire_valeur("beta",0)

ecrire_valeur("Omega",2*omega_0)
ecrire_valeur("B1",0.002)
ecrire_valeur("kappa",0)
ecrire_valeur("N_excit",100)
ecrire_valeur("theta0",1e-3)
ecrire_valeur("thetadot0",0)

nsteps_values = [5000]

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
    plt.plot(t, theta,color = "black")
    plt.xlabel('Temps [s]')
    plt.ylabel('$\\theta$ [rad]')
    plt.legend()
    plt.title(f"Évolution de $\\theta (t)$ pour nsteps={nsteps}")
    plt.grid()
    

    # Tracé de l'orbite dans l’espace de phase
    plt.figure()
    plt.plot(theta, thetadot,color = "black")
    plt.xlabel('$\\theta$ [rad]')
    plt.ylabel('$\\dot{{\\theta}}$ [rad/s]')
    plt.legend()
    plt.title(f"Espace de phase ($\\theta$,$\\dot{{\\theta}}$) pour nsteps={nsteps}")
    plt.grid()
    

    # Tracé de l'énergie mécanique Emec(t)
    plt.figure()
    plt.plot(t, Emec,color = "black")
    plt.xlabel('Temps [s]')
    plt.ylabel('Énergie mécanique [J]')
    plt.legend()
    plt.title(f"Évolution de l\'énergie mécanique pour nsteps={nsteps}")
    plt.grid()

    # Comparaison dEmec/dt avec Pnc(t)
    dEmec_dt = m*L*L*thetadot*np.gradient(thetadot,t)/12+mu*B0*np.sin(theta)*thetadot
    plt.figure()
    plt.plot(t, dEmec_dt, label='dE$_{mec}$/dt',color = "black")
    plt.plot(t, Pnc,linestyle = "-.", label='Pnc',color = "purple")
    plt.xlabel('Temps [s]')
    plt.ylabel('Puissance [W]')
    plt.legend()
    plt.title(f"Comparaison $\\frac{{\\text{{d}}E_{{mec}}}}{{\\text{{d}}t}}$ et $P_{{nc}}(t)$ pour nsteps={nsteps}")
    plt.grid()

plt.show()

# Question 3

ecrire_valeur("C",0)
ecrire_valeur("alpha",0)
ecrire_valeur("beta",0)

ecrire_valeur("Omega",2*omega_0)
ecrire_valeur("B1",0.002)
ecrire_valeur("kappa",0)
ecrire_valeur("N_excit",10000)

nsteps_values = [10]

theta0s = [-2,0,1e-3,1e-1,0.5,1]
thetadot0s = np.linspace(0,20,5)

plt.figure()
for theta in theta0s:
    for thetadot in thetadot0s:
        ecrire_valeur("theta0",theta)
        ecrire_valeur("thetadot0",thetadot)
        couples_phase= []
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
                theta = ((data[k, 1]+np.pi) % (2*np.pi))-np.pi
                thetadot = data[k, 2]
                couples_phase.append([theta,thetadot])

            couples_phase = np.array(couples_phase)
        # Tracé de la section de Poincaré

        plt.scatter(couples_phase[:,0], couples_phase[:,1],s = 0.1)

plt.xlabel('$\Theta$ [rad]')
plt.ylabel('$\dot{\Theta}$ [rad/s]')
plt.legend()
plt.title(f'Sections de Poincaré pour nsteps ={nsteps}')
plt.grid()

plt.show()

# Question 4

ecrire_valeur("C",0)
ecrire_valeur("alpha",0)
ecrire_valeur("beta",0)

ecrire_valeur("Omega",2*omega_0)
ecrire_valeur("B1",0.002)
ecrire_valeur("kappa",0)
ecrire_valeur("N_excit",10000)

ecrire_valeur("nsteps",1000)

ecrire_valeur("N_excit",100)

# Simulation avec deux conditions initiales proches

ecrire_valeur("theta0",1)
thetadot0s = [5,15] # 5 pas chaotique et 15 chaotique
plt.figure()
for thetadot in thetadot0s:
    ecrire_valeur("thetadot0",thetadot) 

    lancer_simulation(theta0, "angle_a.out")
    lancer_simulation(theta0+1e-6, "angle_b.out")

    data_a = np.loadtxt("angle_a.out")
    data_b = np.loadtxt("angle_b.out")

    t = data_a[:, 0]
    theta_a = data_a[:, 1]
    thetadot_a = data_a[:, 2]
    theta_b = data_b[:, 1]
    thetadot_b = data_b[:, 2]

    # Calcul de la distance delta_ab
    delta_ab = np.sqrt(omega_0**2 * (((theta_b - theta_a)+np.pi)%(2*np.pi)-np.pi)**2 + (thetadot_b - thetadot_a)**2)
    
    # Tracé des résultats
    plt.plot(t, delta_ab)
    plt.xlabel("Temps [s]")
    plt.ylabel("Distance $\\delta_{{ab}}$")
    plt.title("Évolution de la distance entre trajectoires d'angles proches")
    plt.grid()

plt.show()

'''
# Question 5

ecrire_valeur("C",0)
ecrire_valeur("alpha",0)
ecrire_valeur("beta",0)

ecrire_valeur("Omega",2*omega_0)
ecrire_valeur("B1",0.018)
ecrire_valeur("kappa",2e-5)
ecrire_valeur("B0",0.01)

ecrire_valeur("N_excit",10000)

ecrire_valeur("theta0",-2)
ecrire_valeur("thetadot0",50)

nsteps_values = [100]

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
        theta = (data[k, 1]% (2*np.pi))-np.pi
        thetadot = data[k, 2]
        couples_phase.append([theta,thetadot])

    couples_phase = np.array(couples_phase)
        # Tracé de la section de Poincaré
    plt.figure()
    plt.scatter(couples_phase[:,0], couples_phase[:,1],s = 2,color = "black")
    plt.xlabel('$\Theta$ [rad]')
    plt.ylabel('$\dot{\Theta}$ [rad/s]')
    plt.legend()
    plt.title(f'Sections de Poincaré pour nsteps ={nsteps}')
    plt.grid()

plt.show()

'''
# Question Facultatif:

ecrire_valeur("Omega",2*omega_0)
ecrire_valeur("B1",0.002)
ecrire_valeur("B0",0.01)
ecrire_valeur("kappa",0)
ecrire_valeur("N_excit",1000)
ecrire_valeur("nsteps",50)

ecrire_valeur("theta0", np.pi)
ecrire_valeur("thetadot0",0)

ecrire_valeur("C", 50)
ecrire_valeur("alpha",0) #'Tres bonne stabilisation à e-12 de précision'
ecrire_valeur("beta", 0)
ecrire_valeur("gamma", 0)

nsteps_values = [50]  # Nombre de pas par période

for i in (nsteps_values):
	ecrire_valeur("theta0",np.pi)
	lancer_simulation(np.pi, "stabilisation.out")

	data_stab = np.loadtxt("stabilisation.out")
	t_stab = data_stab[:, 0]
	theta_stab = data_stab[:, 1]

	plt.figure()
	plt.plot(t_stab, theta_stab,color = "black")
	plt.xlabel("Temps [s]")
	plt.ylabel("θ [rad]")
	plt.title(f"Stabilisation non-linéaire de θeq = π, nsteps={i}")
	plt.grid()
	
plt.show()
'''