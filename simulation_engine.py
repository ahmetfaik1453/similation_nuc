"""
Simulation engine module that coordinates physics models and runs the simulation.
This is the main simulation controller that ties all components together.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import time
import numba
from scipy.integrate import solve_ivp
import traceback

from core_physics import (
    PhysicsModel, PhysicsObject, Vector3D, ParticleGenerator,
    Material, PhysicalConstants, Particle
)
from nuclear_blast import NuclearBlast, BlastParameters, NuclearParticleGenerator
from shield_materials import ShieldMaterial, MATERIALS
from shield_geometry import ShieldGeometry, ShieldLayer
from magnetic_field import MagneticField

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SimulationParameters:
    """Parameters for controlling the simulation"""
    # Time control
    dt: float = 1e-9              # Time step [s]
    total_time: float = 1e-6      # Total simulation time [s]
    output_interval: float = 1e-7  # Time between outputs [s]
    
    # Particle tracking
    max_particles: int = 10000    # Maximum number of particles to track
    min_energy: float = 1e-3      # Minimum energy to track [MeV]
    
    # Physics control
    enable_magnetic_fields: bool = True
    enable_plasma_effects: bool = True
    enable_neutrino_physics: bool = True
    
    # Numerical parameters
    position_tolerance: float = 1e-6  # Position comparison tolerance [m]
    energy_tolerance: float = 1e-6    # Energy comparison tolerance [MeV]
    max_iterations: int = 1000        # Maximum iterations for numerical methods

class SimulationState:
    """Class to hold the current state of the simulation"""
    def __init__(self):
        self.time: float = 0.0
        self.step: int = 0
        
        # Numpy arrays for vectorized operations
        self.positions: np.ndarray = np.array([])
        self.velocities: np.ndarray = np.array([])
        self.charges: np.ndarray = np.array([])
        self.masses: np.ndarray = np.array([])
        self.energies: np.ndarray = np.array([])
        self.particle_types: List[str] = []
        
        # Statistics
        self.particle_counts: Dict[str, int] = {}
        self.energy_deposition: Dict[str, float] = {}
        self.radiation_dose: Dict[str, float] = {}
        
        # Performance metrics
        self.computation_time: float = 0.0
        self.num_steps: int = 0

class SimulationEngine:
    """Main simulation engine class"""
    def __init__(self, params: SimulationParameters):
        self.params = params
        self.state = SimulationState()
        self.constants = PhysicalConstants()
        
        # Components
        self.blast: Optional[NuclearBlast] = None
        self.shield: Optional[ShieldGeometry] = None
        self.magnetic_field: Optional[MagneticField] = None
        self.particle_generator: Optional[ParticleGenerator] = None
        
        # Data storage
        self.output_dir = Path("simulation_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Performance monitoring
        self.performance_stats = {
            'particle_updates': 0,
            'field_calculations': 0,
            'collision_checks': 0,
            'computation_time': 0.0
        }
    
    def _convert_particles_to_arrays(self, particles: List[Particle]) -> None:
        """Convert particle objects to numpy arrays for vectorized operations"""
        try:
            n_particles = len(particles)
            
            self.state.positions = np.zeros((n_particles, 3))
            self.state.velocities = np.zeros((n_particles, 3))
            self.state.charges = np.zeros(n_particles)
            self.state.masses = np.zeros(n_particles)
            self.state.energies = np.zeros(n_particles)
            self.state.particle_types = []
            
            for i, particle in enumerate(particles):
                self.state.positions[i] = [particle.position.x, particle.position.y, particle.position.z]
                self.state.velocities[i] = [particle.velocity.x, particle.velocity.y, particle.velocity.z]
                self.state.charges[i] = particle.charge
                self.state.masses[i] = particle.mass
                self.state.energies[i] = particle.energy
                self.state.particle_types.append(particle.type)
        except Exception as e:
            logger.error(f"Error converting particles to arrays: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _convert_arrays_to_particles(self) -> List[Particle]:
        """Convert numpy arrays back to particle objects"""
        try:
            particles = []
            for i in range(len(self.state.positions)):
                particle = Particle(
                    position=Vector3D(*self.state.positions[i]),
                    velocity=Vector3D(*self.state.velocities[i]),
                    mass=self.state.masses[i],
                    charge=self.state.charges[i],
                    energy=self.state.energies[i],
                    particle_type=self.state.particle_types[i],
                    time=self.state.time
                )
                particles.append(particle)
            return particles
        except Exception as e:
            logger.error(f"Error converting arrays to particles: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _calculate_forces(self, positions: np.ndarray, velocities: np.ndarray,
                         charges: np.ndarray, B_field: np.ndarray) -> np.ndarray:
        """Calculate forces on particles using vectorized operations"""
        try:
            # Lorentz force calculation
            v_cross_B = np.cross(velocities, B_field)
            forces = charges[:, np.newaxis] * v_cross_B
            return forces
        except Exception as e:
            logger.error(f"Error calculating forces: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _process_particles(self) -> None:
        """Process all particles in a single step"""
        try:
            # Calculate magnetic field for each position
            B_field = np.array([
                self.magnetic_field.get_field_at(Vector3D(*pos)).to_array()
                for pos in self.state.positions
            ])
            
            # Calculate forces
            forces = self._calculate_forces(
                self.state.positions,
                self.state.velocities,
                self.state.charges,
                B_field
            )
            
            # Separate handling for massive and massless particles
            dt = self.params.dt
            massive_mask = self.state.masses > 0
            
            # Update massive particles with forces
            if np.any(massive_mask):
                self.state.positions[massive_mask] += (
                    self.state.velocities[massive_mask] * dt +
                    0.5 * forces[massive_mask] / self.state.masses[massive_mask, np.newaxis] * dt**2
                )
                self.state.velocities[massive_mask] += (
                    forces[massive_mask] / self.state.masses[massive_mask, np.newaxis] * dt
                )
            
            # Update massless particles (like photons) with constant velocity
            massless_mask = ~massive_mask
            if np.any(massless_mask):
                self.state.positions[massless_mask] += (
                    self.state.velocities[massless_mask] * dt
                )
                # Velocities remain constant for massless particles
            
        except Exception as e:
            logger.error(f"Error processing particles: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _update_energies(self):
        """Update particle energies considering relativistic effects"""
        try:
            # Calculate relativistic gamma factor
            v_squared = np.sum(self.state.velocities**2, axis=1)
            gamma = 1 / np.sqrt(1 - (v_squared / self.constants.c**2))
            
            # Update energies for massive particles
            massive_mask = self.state.masses > 0
            self.state.energies[massive_mask] = (gamma[massive_mask] - 1) * \
                self.state.masses[massive_mask] * self.constants.c**2
            
            # Handle energy loss in shield if present
            if self.shield is not None:
                for i in range(len(self.state.positions)):
                    pos = Vector3D(*self.state.positions[i])
                    material = self.shield.get_material_at(pos)
                    if material is not None:
                        dE = material.calculate_stopping_power(
                            self.state.particle_types[i],
                            self.state.energies[i]
                        )
                        self.state.energies[i] = max(0.0, self.state.energies[i] - dE)
        except Exception as e:
            logger.error(f"Error updating energies: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def run_simulation(self, particles: List[Particle]) -> Dict[str, Any]:
        """Run the complete simulation"""
        try:
            start_time = time.time()
            
            # Convert particles to arrays for vectorized operations
            self._convert_particles_to_arrays(particles)
            
            # Initialize magnetic field grid if present
            if self.magnetic_field is not None:
                self.magnetic_field.initialize_field_grid()
            
            # Main simulation loop
            while self.state.time < self.params.total_time:
                try:
                    # Process all particles
                    self._process_particles()
                    
                    # Update energies and check for termination
                    self._update_energies()
                    active_particles = self.state.energies >= self.params.min_energy
                    
                    if not np.any(active_particles):
                        break
                    
                    # Keep only active particles
                    self.state.positions = self.state.positions[active_particles]
                    self.state.velocities = self.state.velocities[active_particles]
                    self.state.charges = self.state.charges[active_particles]
                    self.state.masses = self.state.masses[active_particles]
                    self.state.energies = self.state.energies[active_particles]
                    self.state.particle_types = [
                        ptype for i, ptype in enumerate(self.state.particle_types)
                        if active_particles[i]
                    ]
                    
                    # Update time and step counter
                    self.state.time += self.params.dt
                    self.state.step += 1
                    
                    # Save intermediate results if needed
                    if self.state.step % int(self.params.output_interval / self.params.dt) == 0:
                        self._save_intermediate_results()
                        
                except Exception as e:
                    logger.error(f"Error in simulation step {self.state.step}: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise
            
            # Convert final state back to particles
            final_particles = self._convert_arrays_to_particles()
            
            # Calculate statistics
            stats = self._calculate_statistics(final_particles)
            stats['computation_time'] = time.time() - start_time
            
            return {
                'results': final_particles,
                'statistics': stats
            }
            
        except Exception as e:
            logger.error(f"Error running simulation: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _save_intermediate_results(self):
        """Save intermediate simulation results"""
        try:
            output_file = self.output_dir / f"step_{self.state.step}.npz"
            np.savez(
                output_file,
                time=self.state.time,
                positions=self.state.positions,
                velocities=self.state.velocities,
                energies=self.state.energies,
                particle_types=self.state.particle_types
            )
        except Exception as e:
            logger.error(f"Error saving intermediate results: {str(e)}")
            logger.error(traceback.format_exc())
            # Don't raise the error here to allow simulation to continue
    
    def _calculate_statistics(self, particles: List[Particle]) -> Dict[str, Any]:
        """Calculate simulation statistics"""
        try:
            stats = {
                'total_particles': len(particles),
                'particle_types': {},
                'average_final_energy': 0.0,
                'energy_absorption': 0.0,
                'penetration_depths': [],
                'max_penetration': 0.0,
                'average_penetration': 0.0
            }
            
            if not particles:
                return stats
            
            # Calculate statistics
            total_initial_energy = max(1e-10, np.sum(self.state.energies))  # Avoid division by zero
            total_final_energy = 0.0
            
            for particle in particles:
                # Count particle types
                ptype = particle.type
                stats['particle_types'][ptype] = stats['particle_types'].get(ptype, 0) + 1
                
                # Energy statistics
                total_final_energy += max(0.0, particle.energy)
                
                # Penetration depth (distance from origin)
                depth = max(0.0, particle.position.magnitude())
                if np.isfinite(depth):  # Only include valid depths
                    stats['penetration_depths'].append(depth)
            
            # Calculate averages and totals
            valid_depths = [d for d in stats['penetration_depths'] if np.isfinite(d)]
            
            if valid_depths:
                stats['max_penetration'] = max(valid_depths)
                stats['average_penetration'] = sum(valid_depths) / len(valid_depths)
            
            stats['average_final_energy'] = total_final_energy / max(1, len(particles))
            stats['energy_absorption'] = ((total_initial_energy - total_final_energy) / 
                                        total_initial_energy * 100.0)  # Convert to percentage
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def prepare_simulation(self, config: Dict[str, Any]) -> None:
        """Prepare simulation with given configuration
        
        Args:
            config: Dictionary containing simulation configuration parameters
        """
        try:
            # Update simulation parameters
            self.params.dt = config.get('time_step', self.params.dt)
            self.params.total_time = config.get('total_time', self.params.total_time)
            self.params.enable_magnetic_fields = config.get('enable_magnetic_fields', True)
            self.params.enable_plasma_effects = config.get('enable_plasma_effects', True)
            self.params.enable_neutrino_physics = config.get('enable_neutrino_physics', True)
            
            # Initialize components based on configuration
            if self.params.enable_magnetic_fields:
                field_type = config.get('field_type', 'solenoid')
                field_strength = config.get('field_strength', 5.0)
                field_params = config.get('field_params', {})
                
                if field_type == 'solenoid':
                    self.magnetic_field = SolenoidField(
                        center=Vector3D(0, 0, 0),
                        radius=field_params.get('radius', 5.0),
                        length=field_params.get('length', 10.0),
                        n_turns=field_params.get('n_turns', 1000),
                        current=field_params.get('current', 1000),
                        axis=Vector3D(0, 0, 1)
                    )
                elif field_type == 'toroidal':
                    self.magnetic_field = ToroidalField(
                        center=Vector3D(0, 0, 0),
                        major_radius=field_params.get('major_radius', 5.0),
                        minor_radius=field_params.get('minor_radius', 1.0),
                        n_coils=field_params.get('n_coils', 16),
                        current=field_params.get('current', 1000)
                    )
            
            # Initialize shield geometry if parameters provided
            if 'shield_params' in config:
                shield_params = config['shield_params']
                self.shield = ShieldGeometry(
                    radius=shield_params.get('radius', 5.0),
                    height=shield_params.get('height', 10.0),
                    materials=[
                        Material(name=mat['name'], density=mat['density'], 
                               atomic_number=mat['atomic_number'])
                        for mat in shield_params.get('materials', [])
                    ]
                )
            
            # Initialize blast parameters if provided
            if 'blast_params' in config:
                blast_params = config['blast_params']
                self.blast = NuclearBlast(BlastParameters(
                    yield_kt=blast_params.get('yield_kt', 20.0),
                    distance=blast_params.get('distance', 1000.0),
                    altitude=blast_params.get('altitude', 0.0),
                    detonation_time=blast_params.get('detonation_time', 0.0)
                ))
            
            # Initialize particle generator if parameters provided
            if 'particle_params' in config:
                particle_params = config['particle_params']
                self.particle_generator = NuclearParticleGenerator(
                    self.blast,
                    num_particles=particle_params.get('num_particles', 1000)
                )
            
            # Reset simulation state
            self.state = SimulationState()
            
            logger.info("Simulation prepared successfully")
            
        except Exception as e:
            logger.error(f"Error preparing simulation: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get simulation performance statistics"""
        return {
            'computation_time': self.performance_stats['computation_time'],
            'particle_updates': self.performance_stats['particle_updates'],
            'field_calculations': self.performance_stats['field_calculations'],
            'collision_checks': self.performance_stats['collision_checks'],
            'active_particles': len(self.state.positions),
            'time_steps': self.state.step
        }
    
    def get_simulation_results(self) -> Dict[str, Any]:
        """Get simulation results and statistics"""
        return {
            'time': self.state.time,
            'particle_counts': self.state.particle_counts.copy(),
            'energy_deposition': self.state.energy_deposition.copy(),
            'radiation_dose': self.state.radiation_dose.copy(),
            'performance': self.get_performance_stats()
        }