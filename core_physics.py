"""
Core physics module for nuclear blast and radiation shield simulation.
This module provides the fundamental physics interfaces and base classes.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
from scipy.constants import c, e, m_e, m_p, mu_0, epsilon_0, k as k_B, hbar

@dataclass
class PhysicalConstants:
    """Physical constants used throughout the simulation"""
    c: float = c                    # Speed of light [m/s]
    e: float = e                    # Elementary charge [C]
    m_e: float = m_e               # Electron mass [kg]
    m_p: float = m_p               # Proton mass [kg]
    mu_0: float = mu_0             # Vacuum permeability [H/m]
    epsilon_0: float = epsilon_0    # Vacuum permittivity [F/m]
    k_B: float = k_B               # Boltzmann constant [J/K]
    N_A: float = 6.02214076e23    # Avogadro's number [mol^-1]
    
    # Nuclear physics constants
    barn: float = 1e-28            # Barn [m²]
    mev_to_joule: float = 1.60218e-13  # MeV to Joule conversion
    
    # Derived constants
    alpha: float = e**2 / (4*np.pi*epsilon_0*hbar*c)  # Fine structure constant

class Vector3D:
    """3D vector class with common vector operations"""
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def magnitude(self) -> float:
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self) -> 'Vector3D':
        """Return a normalized copy of this vector"""
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x/mag, self.y/mag, self.z/mag)
    
    def dot(self, other: 'Vector3D') -> float:
        """Calculate dot product with another vector"""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other: 'Vector3D') -> 'Vector3D':
        """Calculate cross product with another vector"""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

class PhysicsObject(ABC):
    """Base class for all physics objects in the simulation"""
    def __init__(self, position: Vector3D, time: float = 0):
        self.position = position
        self.time = time
        self.constants = PhysicalConstants()
    
    @abstractmethod
    def update(self, dt: float):
        """Update the object's state for the next time step"""
        pass

class Force(ABC):
    """Base class for all forces in the simulation"""
    @abstractmethod
    def get_force(self, obj: PhysicsObject) -> Vector3D:
        """Calculate the force on a physics object"""
        pass

class Field(ABC):
    """Base class for all fields (electromagnetic, gravitational, etc.)"""
    @abstractmethod
    def get_field_at(self, position: Vector3D, time: float = 0) -> Vector3D:
        """Calculate the field value at a given position and time"""
        pass
    
    @abstractmethod
    def get_potential_at(self, position: Vector3D, time: float = 0) -> float:
        """Calculate the field potential at a given position and time"""
        pass

class Material(ABC):
    """Base class for materials used in the shield"""
    def __init__(self, name: str, density: float, atomic_number: float):
        self.name = name
        self.density = density  # [kg/m³]
        self.atomic_number = atomic_number  # Z number
        self._properties: Dict[str, float] = {}
    
    @abstractmethod
    def get_stopping_power(self, particle_type: str, energy: float) -> float:
        """Calculate stopping power for a given particle type and energy"""
        pass
    
    @abstractmethod
    def get_scattering_cross_section(self, particle_type: str, energy: float) -> float:
        """Calculate scattering cross section for a given particle type and energy"""
        pass
    
    @abstractmethod
    def get_absorption_cross_section(self, particle_type: str, energy: float) -> float:
        """Calculate absorption cross section for a given particle type and energy"""
        pass

class Interaction(ABC):
    """Base class for particle interactions"""
    @abstractmethod
    def compute_interaction(self, particle1: PhysicsObject, particle2: PhysicsObject) -> Tuple[Vector3D, Vector3D]:
        """Compute the interaction between two particles"""
        pass

class ParticleGenerator(ABC):
    """Base class for particle generation"""
    @abstractmethod
    def generate_particles(self, n: int, time: float = 0) -> List[PhysicsObject]:
        """Generate n particles at a given time"""
        pass

class DetectorResponse(ABC):
    """Base class for detector response calculations"""
    @abstractmethod
    def calculate_response(self, particle: PhysicsObject) -> float:
        """Calculate detector response for a given particle"""
        pass

class PhysicsModel(ABC):
    """Base class for physics models"""
    @abstractmethod
    def initialize(self):
        """Initialize the physics model"""
        pass
    
    @abstractmethod
    def step(self, dt: float):
        """Advance the model by one time step"""
        pass
    
    @abstractmethod
    def get_observables(self) -> Dict[str, float]:
        """Return observable quantities from the model"""
        pass 

class Particle(PhysicsObject):
    """Concrete implementation of PhysicsObject for particles"""
    def __init__(self, position: Vector3D, velocity: Optional[Vector3D] = None,
                 mass: float = 0, charge: float = 0, energy: float = 0,
                 particle_type: str = "unknown", time: float = 0):
        super().__init__(position, time)
        self.velocity = velocity if velocity is not None else Vector3D(0, 0, 0)
        self.mass = mass
        self.charge = charge
        self.energy = energy
        self.type = particle_type
        self.forces = []
    
    def update(self, dt: float, force: Optional[Vector3D] = None):
        """Update particle state using velocity Verlet algorithm"""
        if force is None:
            force = Vector3D(0, 0, 0)
        
        # Update position
        self.position = Vector3D(
            self.position.x + self.velocity.x * dt + 0.5 * force.x * dt**2 / self.mass,
            self.position.y + self.velocity.y * dt + 0.5 * force.y * dt**2 / self.mass,
            self.position.z + self.velocity.z * dt + 0.5 * force.z * dt**2 / self.mass
        )
        
        # Update velocity
        self.velocity = Vector3D(
            self.velocity.x + force.x * dt / self.mass,
            self.velocity.y + force.y * dt / self.mass,
            self.velocity.z + force.z * dt / self.mass
        )
        
        # Update energy (kinetic)
        v_mag = self.velocity.magnitude()
        if self.mass > 0:
            gamma = 1 / np.sqrt(1 - (v_mag/self.constants.c)**2)
            self.energy = (gamma - 1) * self.mass * self.constants.c**2
        else:
            self.energy = self.energy  # Photons maintain their energy
        
        # Update time
        self.time += dt

class PhysicsEngine:
    """Main physics engine class that coordinates all physics calculations"""
    def __init__(self):
        self.constants = PhysicalConstants()
        self.particles: List[Particle] = []
        self.fields: List[Field] = []
        self.materials: List[Material] = []
        self.time: float = 0.0
        self.dt: float = 1e-9  # Default time step: 1 ns
        self.gravity = Vector3D(0, 0, -9.81)
        self.time_step = 0.001
        
    def add_particle(self, particle: Particle):
        """Add a particle to the simulation"""
        self.particles.append(particle)
    
    def add_field(self, field: Field):
        """Add a field to the simulation"""
        self.fields.append(field)
    
    def add_material(self, material: Material):
        """Add a material to the simulation"""
        self.materials.append(material)
    
    def calculate_forces(self, particle: Particle) -> Vector3D:
        """Calculate total force on a particle"""
        total_force = Vector3D(0, 0, 0)
        
        # Field forces
        for field in self.fields:
            if isinstance(particle, Particle) and particle.charge != 0:
                # Electromagnetic force
                E = field.get_field_at(particle.position, self.time)
                B = field.get_field_at(particle.position, self.time)  # Assuming B-field component
                v = particle.velocity
                
                # F = q(E + v × B)
                E_force = E * particle.charge
                B_force = v.cross(B) * particle.charge
                total_force = total_force + E_force + B_force
        
        # Material interactions
        for material in self.materials:
            if isinstance(particle, Particle):
                # Calculate stopping power
                dE_dx = material.get_stopping_power(particle.type, particle.energy)
                v_dir = particle.velocity.normalize()
                stopping_force = v_dir * (-dE_dx)
                total_force = total_force + stopping_force
        
        return total_force
    
    def step(self):
        """Advance simulation by one time step"""
        for particle in self.particles:
            force = self.calculate_forces(particle)
            particle.update(self.dt, force)
        self.time += self.dt
    
    def run(self, steps: int):
        """Run simulation for specified number of steps"""
        for _ in range(steps):
            self.step()
    
    def get_particle_positions(self) -> List[Vector3D]:
        """Get current positions of all particles"""
        return [p.position for p in self.particles]
    
    def get_particle_energies(self) -> List[float]:
        """Get current energies of all particles"""
        return [p.energy for p in self.particles]
    
    def get_particle_trajectories(self) -> List[List[Vector3D]]:
        """Get particle trajectories"""
        return [[p.position] for p in self.particles]  # Simplified, actual implementation would track history
    
    def clear(self):
        """Clear all particles and reset time"""
        self.particles.clear()
        self.time = 0.0
    
    def calculate_trajectory(self, initial_pos: Vector3D, initial_vel: Vector3D, 
                           time: float) -> List[Vector3D]:
        positions = []
        current_pos = initial_pos
        current_vel = initial_vel
        t = 0
        
        while t < time:
            positions.append(current_pos)
            # Basic Euler integration
            current_vel = current_vel + self.gravity * self.time_step
            current_pos = current_pos + current_vel * self.time_step
            t += self.time_step
        
        return positions