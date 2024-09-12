import numpy as np 
from matplotlib import pyplot as plt

hbar = 6.5821220E-16 # [eV s]
electron_charge = 1.602E-19 # [C]
mass_electron = 9.11E-31 # [kg]

def GetWaveFn(x, width):
    wavefunction = []
    for i in range(1, 11):
        wavefunction.append(np.sqrt(2/width) * np.sin(np.pi * i * x/ width))     
    
    return wavefunction

# Energies for particle in a box for n = 1, 2, 3, ... 10 #
def GetEnergies(width):
    energy = []
    for i in range(1, 11):
        energy.append(i**2 * hbar**2 * np.pi**2 / (2 * mass_electron * width))
    return energy

def EnergyCorrection(wavefunction, potential, x):
    integrand = wavefunction * potential * wavefunction
    return np.trapz(integrand, x)

# %%
def GetWaveCorrection(wavefunction, x, potential, energy):
    correction = []
    
    for n in range(0, 10):
        temp = 0
        for m in range(0, 10):
            if(n != m):
                numerator = np.trapz(wavefunction[m] * potential * wavefunction[n], x)
                denominator = energy[n] - energy[m]
                temp = temp + numerator / denominator * wavefunction[m]
        correction.append(temp)
        
    return correction

def MakeGraph(title, x, y1, color1, y2, color2, label1, label2, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y1, color = color1, label = label1)
    plt.plot(x, y2, color = color2, label = label2)
    plt.minorticks_on()
    plt.tick_params(which = 'minor', direction = 'in', bottom = True, top = True, left = True, right = True, length = 4)
    plt.tick_params(length = 7, bottom = True, top = True, left = True, right = True)
    plt.tick_params(axis = 'both', direction = 'in')
    plt.ticklabel_format(style = "scientific", useMathText=True, axis="x", scilimits=(0,0))
    plt.legend()
    plt.show()
    
    return 0

def MakeEnergyGraph(x, energy, corrected_energy, potential):
    # Plots the energy inside the well #
    for i in range(0, 10):
        plt.plot(x, energy[i] + x*0, color = 'black')
        plt.plot(x, corrected_energy[i] + x*0, color = 'red')

    # Plots the Well #
    well_height = np.linspace(0, 25, 1000) # Used only for visual of the well
    plt.plot(0 * x, well_height, color = 'blue')
    plt.plot(0 * x + 10, well_height, color = 'blue')
    plt.plot(x, potential)

    # Only here to show the labels correctly for all Corrected and normal Energies  - Avoid redundant labelling #
    plt.plot(x, 0*x, color = 'red', label = f'$E_n^{(1)}$') 
    plt.plot(x, 0*x, color = 'black', label = f'$E_n^{(0)}$') 

    # Graph Style #
    plt.title('Energies inside of the Perturbed Potential Well')
    plt.xlabel("Width")
    plt.ylabel(f"$E_n$")
    plt.minorticks_on()
    plt.tick_params(which = 'minor', direction = 'in', bottom = True, top = True, left = True, right = True, length = 4)
    plt.tick_params(length = 7, bottom = True, top = True, left = True, right = True)
    plt.tick_params(axis = 'both', direction = 'in')
    plt.ticklabel_format(style = "scientific", useMathText=True, axis="x", scilimits=(0,0))
    plt.legend()
    plt.show()
    
    return 0

# Width of the potential well #
wellWidth = 10
x = np.linspace(0, wellWidth, 1000)

# Generate the potential and wavefunction #
wavefunction = GetWaveFn(x, wellWidth)

widthOfPotential = 0.05 # Width of Gaussian Perturbed Potential 
PotentialShift = 5      # Position (Left or Right of Origin) of Gaussian Perturbed Potential
potential = -1 / (widthOfPotential * np.sqrt(2*np.pi)) * np.exp(-(x-PotentialShift)**2 / (2 * widthOfPotential**2))


# Get the energy and first order corrected energy, add them for the total new energy #
energy = GetEnergies(wellWidth)
energy_firstOrder = EnergyCorrection(wavefunction, potential, x)
corrected_energy = energy + energy_firstOrder

# Find the corrected wavefunction
wavefunction_firstOrder = GetWaveCorrection(wavefunction, x, potential, energy)
corrected_wavefunction = []
for i in range(0, 10):
    corrected_wavefunction.append(wavefunction_firstOrder[i] + wavefunction[i])

# Normalize #
for i in range(0, 10):
    corrected_wavefunction[i] = corrected_wavefunction[i] / (np.sqrt(np.trapz(corrected_wavefunction[i]**2, x)))


# Get Probability Distributions #
waveFnProb = []
CorrectedWaveFnProb = []
for i in range(0, 10):
    waveFnProb.append(wavefunction[i]**2)
    CorrectedWaveFnProb.append(corrected_wavefunction[i]**2)


# Create Wavefunction Graphs #
for i in range(0, 10):
    MakeGraph(
        f'Infinite Square Well vs Perturbed Well Wavefunction for n = {i+1}',
        x,
        wavefunction[i],
        'black',
        corrected_wavefunction[i],
        'red',
        r'$\psi_n^{(0)}$',
        r'$\psi_n^{(1)}$',
        'width',
        f'$\Psi_{i}(x, t)$'
    )
    
    
# Create Probability Density Graphs
for i in range(0, 10):
    MakeGraph(
        f'Infinite Square Well vs Perturbed Well Probabilty for n = {i+1}',
        x,
        waveFnProb[i],
        'black',
        CorrectedWaveFnProb[i],
        'red',
        r'$\Psi_n^{(0)}$',
        r'$\Psi_n^{(1)}$',
        'width',
        f'$\Psi_{i}(x, t)$'
    )


# Create Energy Graph #
MakeEnergyGraph(x, energy, corrected_energy, potential)

# Width of the potential well #
wellWidth = 10
x = np.linspace(0, wellWidth, 1000)


# Generate the potential and wavefunction #
wavefunction = GetWaveFn(x, wellWidth)

widthOfPotential = 0.05 # Width of Gaussian Perturbed Potential 
PotentialShift = 8      # Position (Left or Right of Origin) of Gaussian Perturbed Potential
potential = -1 / (widthOfPotential * np.sqrt(2*np.pi)) * np.exp(-(x-PotentialShift)**2 / (2 * widthOfPotential**2))


# Get the energy and first order corrected energy, add them for the total new energy #
energy = GetEnergies(wellWidth)
energy_firstOrder = EnergyCorrection(wavefunction, potential, x)
corrected_energy = energy + energy_firstOrder


# Find the corrected wavefunction
wavefunction_firstOrder = GetWaveCorrection(wavefunction, x, potential, energy)
corrected_wavefunction = []
for i in range(0, 10):
    corrected_wavefunction.append(wavefunction_firstOrder[i] + wavefunction[i])


# Normalize #
for i in range(1, 10):
    corrected_wavefunction[i] = corrected_wavefunction[i] / (np.sqrt(np.trapz(corrected_wavefunction[i]**2, x)))


# Get Probability Distributions #
waveFnProb = []
CorrectedWaveFnProb = []
for i in range(0, 10):
    waveFnProb.append(wavefunction[i]**2)
    CorrectedWaveFnProb.append(corrected_wavefunction[i]**2)


# Create Wavefunction Graphs #
for i in range(0, 10):
    MakeGraph(
        f'Infinite Square Well vs Perturbed Well Wavefunction for n = {i+1}',
        x,
        wavefunction[i],
        'black',
        corrected_wavefunction[i],
        'red',
        r'$\psi_n^{(0)}$',
        r'$\psi_n^{(1)}$',
        'width',
        f'$\Psi_{i}(x, t)$'
    )


# Create Probability Density Graphs
for i in range(0, 10):
    MakeGraph(
        f'Infinite Square Well vs Perturbed Well Probabilty for n = {i+1}',
        x,
        waveFnProb[i],
        'black',
        CorrectedWaveFnProb[i],
        'red',
        r'$\Psi_n^{(0)}$',
        r'$\Psi_n^{(1)}$',
        'width',
        f'$\Psi_{i}(x, t)$'
    )


# Create Energy Graph #
MakeEnergyGraph(x, energy, corrected_energy, potential)

# Width of the potential well #
wellWidth = 10
x = np.linspace(0, wellWidth, 1000)


# Generate the potential and wavefunction #
wavefunction = GetWaveFn(x, wellWidth)

widthOfPotential = 1 # Width of Gaussian Perturbed Potential 
PotentialShift = 5      # Position (Left or Right of Origin) of Gaussian Perturbed Potential
potential = -1 / (widthOfPotential * np.sqrt(2*np.pi)) * np.exp(-(x-PotentialShift)**2 / (2 * widthOfPotential**2))


# Get the energy and first order corrected energy, add them for the total new energy #
energy = GetEnergies(wellWidth)
energy_firstOrder = EnergyCorrection(wavefunction, potential, x)
corrected_energy = energy + energy_firstOrder


# Find the corrected wavefunction
wavefunction_firstOrder = GetWaveCorrection(wavefunction, x, potential, energy)
corrected_wavefunction = []
for i in range(0, 10):
    corrected_wavefunction.append(wavefunction_firstOrder[i] + wavefunction[i])


# Normalize #
for i in range(1, 10):
    corrected_wavefunction[i] = corrected_wavefunction[i] / (np.sqrt(np.trapz(corrected_wavefunction[i]**2, x)))


# Get Probability Distributions #
waveFnProb = []
CorrectedWaveFnProb = []
for i in range(0, 10):
    waveFnProb.append(wavefunction[i]**2)
    CorrectedWaveFnProb.append(corrected_wavefunction[i]**2)


# Create Wavefunction Graphs #
for i in range(0, 10):
    MakeGraph(
        f'Infinite Square Well vs Perturbed Well Wavefunction for n = {i+1}',
        x,
        wavefunction[i],
        'black',
        corrected_wavefunction[i],
        'red',
        r'$\psi_n^{(0)}$',
        r'$\psi_n^{(1)}$',
        'width',
        f'$\Psi_{i}(x, t)$'
    )


# Create Probability Density Graphs
for i in range(0, 10):
    MakeGraph(
        f'Infinite Square Well vs Perturbed Well Probabilty for n = {i+1}',
        x,
        waveFnProb[i],
        'black',
        CorrectedWaveFnProb[i],
        'red',
        r'$\Psi_n^{(0)}$',
        r'$\Psi_n^{(1)}$',
        'width',
        f'$\Psi_{i}(x, t)$'
    )


# Create Energy Graph #
MakeEnergyGraph(x, energy, corrected_energy, potential)

# Width of the potential well #
wellWidth = 10
x = np.linspace(0, wellWidth, 1000)


# Generate the potential and wavefunction #
wavefunction = GetWaveFn(x, wellWidth)

# New Potential
potential = -1 * np.sqrt(x)


# Get the energy and first order corrected energy, add them for the total new energy #
energy = GetEnergies(wellWidth)
energy_firstOrder = EnergyCorrection(wavefunction, potential, x)
corrected_energy = energy + energy_firstOrder


# Find the corrected wavefunction
wavefunction_firstOrder = GetWaveCorrection(wavefunction, x, potential, energy)
corrected_wavefunction = []
for i in range(0, 10):
    corrected_wavefunction.append(wavefunction_firstOrder[i] + wavefunction[i])


# Normalize #
for i in range(1, 10):
    corrected_wavefunction[i] = corrected_wavefunction[i] / (np.sqrt(np.trapz(corrected_wavefunction[i]**2, x)))


# Get Probability Distributions #
waveFnProb = []
CorrectedWaveFnProb = []
for i in range(0, 10):
    waveFnProb.append(wavefunction[i]**2)
    CorrectedWaveFnProb.append(corrected_wavefunction[i]**2)


# Create Wavefunction Graphs #
for i in range(0, 10):
    MakeGraph(
        f'Infinite Square Well vs Perturbed Well Wavefunction for n = {i+1}',
        x,
        wavefunction[i],
        'black',
        corrected_wavefunction[i],
        'red',
        r'$\psi_n^{(0)}$',
        r'$\psi_n^{(1)}$',
        'width',
        f'$\Psi_{i}(x, t)$'
    )


# Create Probability Density Graphs
for i in range(0, 10):
    MakeGraph(
        f'Infinite Square Well vs Perturbed Well Probabilty for n = {i+1}',
        x,
        waveFnProb[i],
        'black',
        CorrectedWaveFnProb[i],
        'red',
        r'$\Psi_n^{(0)}$',
        r'$\Psi_n^{(1)}$',
        'width',
        f'$\Psi_{i}(x, t)$'
    )


# Create Energy Graph #
MakeEnergyGraph(x, energy, corrected_energy, potential)


# Width of the potential well #
wellWidth = 10
x = np.linspace(0, wellWidth, 1000)


# Generate the potential and wavefunction #
wavefunction = GetWaveFn(x, wellWidth)

# -x
potential = -x


# Get the energy and first order corrected energy, add them for the total new energy #
energy = GetEnergies(wellWidth)
energy_firstOrder = EnergyCorrection(wavefunction, potential, x)
corrected_energy = energy + energy_firstOrder


# Find the corrected wavefunction
wavefunction_firstOrder = GetWaveCorrection(wavefunction, x, potential, energy)
corrected_wavefunction = []
for i in range(0, 10):
    corrected_wavefunction.append(wavefunction_firstOrder[i] + wavefunction[i])


# Normalize #
for i in range(1, 10):
    corrected_wavefunction[i] = corrected_wavefunction[i] / (np.sqrt(np.trapz(corrected_wavefunction[i]**2, x)))


# Get Probability Distributions #
waveFnProb = []
CorrectedWaveFnProb = []
for i in range(0, 10):
    waveFnProb.append(wavefunction[i]**2)
    CorrectedWaveFnProb.append(corrected_wavefunction[i]**2)


# Create Wavefunction Graphs #
for i in range(0, 10):
    MakeGraph(
        f'Infinite Square Well vs Perturbed Well Wavefunction for n = {i+1}',
        x,
        wavefunction[i],
        'black',
        corrected_wavefunction[i],
        'red',
        r'$\psi_n^{(0)}$',
        r'$\psi_n^{(1)}$',
        'width',
        f'$\Psi_{i}(x, t)$'
    )


# Create Probability Density Graphs
for i in range(0, 10):
    MakeGraph(
        f'Infinite Square Well vs Perturbed Well Probabilty for n = {i+1}',
        x,
        waveFnProb[i],
        'black',
        CorrectedWaveFnProb[i],
        'red',
        r'$\Psi_n^{(0)}$',
        r'$\Psi_n^{(1)}$',
        'width',
        f'$\Psi_{i}(x, t)$'
    )


# Create Energy Graph #
MakeEnergyGraph(x, energy, corrected_energy, potential)

# %%
# Width of the potential well #
wellWidth = 10
x = np.linspace(0, wellWidth, 1000)


# Generate the potential and wavefunction #
wavefunction = GetWaveFn(x, wellWidth)

# sin(2x)
potential = np.sin(2*x)


# Get the energy and first order corrected energy, add them for the total new energy #
energy = GetEnergies(wellWidth)
energy_firstOrder = EnergyCorrection(wavefunction, potential, x)
corrected_energy = energy + energy_firstOrder


# Find the corrected wavefunction
wavefunction_firstOrder = GetWaveCorrection(wavefunction, x, potential, energy)
corrected_wavefunction = []
for i in range(0, 10):
    corrected_wavefunction.append(wavefunction_firstOrder[i] + wavefunction[i])


# Normalize #
for i in range(1, 10):
    corrected_wavefunction[i] = corrected_wavefunction[i] / (np.sqrt(np.trapz(corrected_wavefunction[i]**2, x)))


# Get Probability Distributions #
waveFnProb = []
CorrectedWaveFnProb = []
for i in range(0, 10):
    waveFnProb.append(wavefunction[i]**2)
    CorrectedWaveFnProb.append(corrected_wavefunction[i]**2)


# Create Wavefunction Graphs #
for i in range(0, 10):
    MakeGraph(
        f'Infinite Square Well vs Perturbed Well Wavefunction for n = {i+1}',
        x,
        wavefunction[i],
        'black',
        corrected_wavefunction[i],
        'red',
        r'$\psi_n^{(0)}$',
        r'$\psi_n^{(1)}$',
        'width',
        f'$\Psi_{i}(x, t)$'
    )


# Create Probability Density Graphs
for i in range(0, 10):
    MakeGraph(
        f'Infinite Square Well vs Perturbed Well Probabilty for n = {i+1}',
        x,
        waveFnProb[i],
        'black',
        CorrectedWaveFnProb[i],
        'red',
        r'$\Psi_n^{(0)}$',
        r'$\Psi_n^{(1)}$',
        'width',
        f'$\Psi_{i}(x, t)$'
    )


# Create Energy Graph #
MakeEnergyGraph(x, energy, corrected_energy, potential)

# %%
# Width of the potential well #
wellWidth = 10
x = np.linspace(0, wellWidth, 1000)


# Generate the potential and wavefunction #
wavefunction = GetWaveFn(x, wellWidth)

# -log(x)
potential = - np.log(x+0.00001) #Adjust slightly to avoid undefined behavior at log(0)


# Get the energy and first order corrected energy, add them for the total new energy #
energy = GetEnergies(wellWidth)
energy_firstOrder = EnergyCorrection(wavefunction, potential, x)
corrected_energy = energy + energy_firstOrder


# Find the corrected wavefunction
wavefunction_firstOrder = GetWaveCorrection(wavefunction, x, potential, energy)
corrected_wavefunction = []
for i in range(0, 10):
    corrected_wavefunction.append(wavefunction_firstOrder[i] + wavefunction[i])


# Normalize #
for i in range(1, 10):
    corrected_wavefunction[i] = corrected_wavefunction[i] / (np.sqrt(np.trapz(corrected_wavefunction[i]**2, x)))


# Get Probability Distributions #
waveFnProb = []
CorrectedWaveFnProb = []
for i in range(0, 10):
    waveFnProb.append(wavefunction[i]**2)
    CorrectedWaveFnProb.append(corrected_wavefunction[i]**2)


# Create Wavefunction Graphs #
for i in range(0, 10):
    MakeGraph(
        f'Infinite Square Well vs Perturbed Well Wavefunction for n = {i+1}',
        x,
        wavefunction[i],
        'black',
        corrected_wavefunction[i],
        'red',
        r'$\psi_n^{(0)}$',
        r'$\psi_n^{(1)}$',
        'width',
        f'$\Psi_{i}(x, t)$'
    )


# Create Probability Density Graphs
for i in range(0, 10):
    MakeGraph(
        f'Infinite Square Well vs Perturbed Well Probabilty for n = {i+1}',
        x,
        waveFnProb[i],
        'black',
        CorrectedWaveFnProb[i],
        'red',
        r'$\Psi_n^{(0)}$',
        r'$\Psi_n^{(1)}$',
        'width',
        f'$\Psi_{i}(x, t)$'
    )


# Create Energy Graph #
MakeEnergyGraph(x, energy, corrected_energy, potential)
