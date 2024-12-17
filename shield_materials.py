"""
Shield materials module containing implementations of common shielding materials
and their properties for nuclear radiation protection.
"""

from typing import Dict, Optional, List
import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass

from core_physics import Material, PhysicalConstants, Vector3D

@dataclass
class MaterialProperties:
    """Material properties for radiation shielding"""
    density: float              # Density [kg/m³]
    atomic_number: float        # Atomic number Z
    atomic_mass: float         # Atomic mass [g/mol]
    melting_point: float       # Melting point [K]
    thermal_conductivity: float  # Thermal conductivity [W/(m·K)]
    specific_heat: float       # Specific heat capacity [J/(kg·K)]
    electron_density: float    # Electron density [m^-3]
    radiation_length: float    # Radiation length [m]
    nuclear_collision_length: float  # Nuclear collision length [m]

# Malzeme veritabanı
MATERIALS = {
    "tungsten": MaterialProperties(
        density=19350.0,
        atomic_number=74.0,
        atomic_mass=183.84,
        melting_point=3695.0,
        thermal_conductivity=173.0,
        specific_heat=134.0,
        electron_density=4.5e29,
        radiation_length=0.0035,
        nuclear_collision_length=0.0096
    ),
    "lead": MaterialProperties(
        density=11340.0,
        atomic_number=82.0,
        atomic_mass=207.2,
        melting_point=600.61,
        thermal_conductivity=35.3,
        specific_heat=129.0,
        electron_density=3.3e29,
        radiation_length=0.0056,
        nuclear_collision_length=0.0177
    ),
    "iron": MaterialProperties(
        density=7874.0,
        atomic_number=26.0,
        atomic_mass=55.845,
        melting_point=1811.0,
        thermal_conductivity=80.4,
        specific_heat=449.0,
        electron_density=2.2e29,
        radiation_length=0.0176,
        nuclear_collision_length=0.0320
    ),
    "concrete": MaterialProperties(
        density=2300.0,
        atomic_number=11.0,  # Ortalama
        atomic_mass=24.0,    # Ortalama
        melting_point=1800.0,
        thermal_conductivity=1.7,
        specific_heat=880.0,
        electron_density=7.0e28,
        radiation_length=0.0340,
        nuclear_collision_length=0.0520
    ),
    "water": MaterialProperties(
        density=1000.0,
        atomic_number=7.23,  # Ortalama (H2O)
        atomic_mass=18.015,
        melting_point=273.15,
        thermal_conductivity=0.6,
        specific_heat=4186.0,
        electron_density=3.3428e29,
        radiation_length=0.0360,
        nuclear_collision_length=0.0850
    )
}

class ShieldMaterial(Material):
    """Base class for shield materials with common functionality"""
    def __init__(self, name: str):
        if name not in MATERIALS:
            raise ValueError(f"Material '{name}' not found in database. Available materials: {list(MATERIALS.keys())}")
        
        properties = MATERIALS[name]
        super().__init__(name, properties.density, properties.atomic_number)
        self.properties = properties
        self.constants = PhysicalConstants()
        self.thickness = 0.0  # Default thickness [m]
        
        # Initialize energy-dependent cross section tables
        self._initialize_cross_sections()
    
    @property
    def thickness(self) -> float:
        """Get material thickness"""
        return self._thickness
    
    @thickness.setter
    def thickness(self, value: float):
        """Set material thickness"""
        if value < 0:
            raise ValueError("Thickness cannot be negative")
        self._thickness = value
    
    def _initialize_cross_sections(self):
        """Initialize energy-dependent cross section tables"""
        # Energy grid for cross section tables [MeV]
        self.energy_grid = np.logspace(-3, 4, 1000)
        
        # Initialize interpolators for each particle type
        self._setup_interpolators()
    
    def _setup_interpolators(self):
        """Set up interpolation functions for cross sections"""
        self.interpolators = {}
        
        # Energy grid for interpolation
        energies = self.energy_grid
        
        # Create basic cross section data for each particle type
        for particle_type in ['neutron', 'gamma', 'beta', 'alpha']:
            self.interpolators[particle_type] = {}
            
            # Elastic scattering
            elastic_xs = self._calculate_elastic_xs(particle_type, energies)
            self.interpolators[particle_type]['elastic'] = \
                interp1d(energies, elastic_xs, bounds_error=False, fill_value='extrapolate')
            
            # Capture/absorption
            capture_xs = self._calculate_capture_xs(particle_type, energies)
            self.interpolators[particle_type]['capture'] = \
                interp1d(energies, capture_xs, bounds_error=False, fill_value='extrapolate')
            
            # Total
            total_xs = elastic_xs + capture_xs
            self.interpolators[particle_type]['total'] = \
                interp1d(energies, total_xs, bounds_error=False, fill_value='extrapolate')
    
    def _calculate_elastic_xs(self, particle_type: str, energies: np.ndarray) -> np.ndarray:
        """Calculate elastic scattering cross sections"""
        if particle_type == 'neutron':
            # Simple A^(2/3) dependence for neutrons
            return 2 * np.pi * (1.2e-15 * self.properties.atomic_mass**(1/3))**2 * \
                   np.ones_like(energies) * self.constants.barn
        elif particle_type == 'gamma':
            # Approximation of Compton scattering
            return 0.665 * self.constants.barn * self.properties.atomic_number * \
                   np.exp(-energies/10) * np.ones_like(energies)
        else:
            # Coulomb scattering for charged particles
            return np.pi * (2.82e-15)**2 * self.properties.atomic_number**2 * \
                   (1/energies)**2 * self.constants.barn
    
    def _calculate_capture_xs(self, particle_type: str, energies: np.ndarray) -> np.ndarray:
        """Calculate capture/absorption cross sections"""
        if particle_type == 'neutron':
            # 1/v law for thermal neutrons
            return 0.1 * self.constants.barn * self.properties.atomic_number * \
                   (0.025/energies)**0.5
        elif particle_type == 'gamma':
            # Photoelectric effect approximation
            return 1e-3 * self.constants.barn * self.properties.atomic_number**4 * \
                   (1/energies)**3
        else:
            # Negligible capture for charged particles
            return np.zeros_like(energies)
    
    def get_stopping_power(self, particle_type: str, energy: float) -> float:
        """Calculate stopping power for a given particle type and energy"""
        if particle_type not in ['alpha', 'beta', 'proton']:
            return 0.0
        
        # Bethe formula for stopping power
        if particle_type == 'alpha':
            z_projectile = 2
            m_projectile = 4 * self.constants.m_p
        elif particle_type == 'proton':
            z_projectile = 1
            m_projectile = self.constants.m_p
        else:  # beta
            z_projectile = -1
            m_projectile = self.constants.m_e
        
        # Convert energy to kinetic energy in Joules
        E_k = energy * self.constants.mev_to_joule
        
        # Calculate beta and gamma
        total_energy = E_k + m_projectile * self.constants.c**2
        gamma = total_energy / (m_projectile * self.constants.c**2)
        beta = np.sqrt(1 - 1/gamma**2)
        
        # Mean excitation energy (approximation)
        I = 16 * self.properties.atomic_number**0.9 * self.constants.e
        
        # Calculate stopping power using Bethe formula
        prefactor = 4 * np.pi * self.constants.e**4 * z_projectile**2
        main_term = (
            self.properties.electron_density / 
            (m_projectile * self.constants.c**2 * beta**2)
        )
        log_term = np.log(
            2 * m_projectile * self.constants.c**2 * beta**2 * gamma**2 / I
        ) - beta**2
        
        dE_dx = prefactor * main_term * log_term
        
        # Convert to MeV/m and return
        return dE_dx / self.constants.mev_to_joule
    
    def get_scattering_cross_section(self, particle_type: str, energy: float) -> float:
        """Calculate scattering cross section for a given particle type and energy"""
        if particle_type not in self.interpolators:
            return 0.0
            
        # Use pre-calculated interpolation tables
        return float(self.interpolators[particle_type]['elastic'](energy))
    
    def get_absorption_cross_section(self, particle_type: str, energy: float) -> float:
        """Calculate absorption cross section for a given particle type and energy"""
        if particle_type not in self.interpolators:
            return 0.0
            
        # Use pre-calculated interpolation tables
        return float(self.interpolators[particle_type]['capture'](energy))

class CompositeShield:
    """A shield composed of multiple layers of different materials"""
    def __init__(self, layers: List[ShieldMaterial]):
        self.layers = layers
    
    def get_total_stopping_power(self, particle_type: str, energy: float) -> float:
        """Calculate total stopping power through all layers"""
        return sum(layer.get_stopping_power(particle_type, energy) for layer in self.layers)
    
    def get_total_thickness(self) -> float:
        """Calculate total shield thickness"""
        return sum(layer.thickness for layer in self.layers)
    
    def get_total_mass(self) -> float:
        """Calculate total shield mass"""
        return sum(layer.density * layer.thickness for layer in self.layers) 

class MaterialDatabase:
    def __init__(self):
        self.materials = {
            'tungsten': {
                'density': 19.3,  # g/cm³
                'melting_point': 3422,  # °C
                'thermal_conductivity': 173,  # W/(m·K)
                'radiation_resistance': 0.95,  # arbitrary units
                'cost': 100  # arbitrary units
            },
            'lead': {
                'density': 11.34,
                'melting_point': 327.5,
                'thermal_conductivity': 35.3,
                'radiation_resistance': 0.75,
                'cost': 50
            },
            'concrete': {
                'density': 2.4,
                'melting_point': 1500,
                'thermal_conductivity': 1.7,
                'radiation_resistance': 0.45,
                'cost': 10
            }
        }
    
    def get_material(self, name: str) -> Optional[Dict]:
        """Get material properties by name"""
        return self.materials.get(name.lower())
    
    def add_material(self, name: str, properties: Dict) -> None:
        """Add new material to database"""
        self.materials[name.lower()] = properties
    
    def list_materials(self) -> List[str]:
        """List all available materials"""
        return list(self.materials.keys())
    
    def get_best_material(self, priority: str) -> str:
        """Get best material based on given priority property"""
        if not self.materials:
            return None
            
        return max(self.materials.items(), 
                  key=lambda x: x[1].get(priority, 0))[0] 