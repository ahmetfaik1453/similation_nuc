"""
Nuclear blast physics module for simulating nuclear explosion effects.
This module implements specific models for nuclear explosions and their effects.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.special import gamma
from dataclasses import dataclass

from core_physics import (
    PhysicsModel, PhysicsObject, Vector3D, ParticleGenerator,
    Material, PhysicalConstants, Particle
)

@dataclass
class BlastParameters:
    """Parameters characterizing a nuclear blast"""
    yield_kt: float           # Blast yield in kilotons TNT
    distance: float          # Distance from ground zero in meters
    altitude: float          # Altitude above ground in meters
    detonation_time: float   # Time of detonation in seconds
    ambient_pressure: float = 101325.0    # Ambient atmospheric pressure in Pa
    ambient_temperature: float = 288.15   # Ambient temperature in K
    relative_humidity: float = 0.5        # Relative humidity (0-1)
    
    @property
    def yield_joules(self) -> float:
        """Convert yield from kt to joules"""
        return self.yield_kt * 4.184e12  # 1 kt = 4.184e12 joules

class NuclearBlast(PhysicsModel):
    """Model for nuclear blast physics"""
    def __init__(self, params: BlastParameters):
        """Initialize nuclear blast model
        
        Args:
            params: Blast parameters
        """
        self.params = params
        self.constants = PhysicalConstants()
        self.time = 0.0
        self._initialize_derived_parameters()
    
    def _initialize_derived_parameters(self):
        """Calculate derived blast parameters"""
        # Scaling factors based on yield
        self.W = self.params.yield_kt ** (1/3)  # Cube root of yield
        
        # Characteristic times
        self.t_unit = self.W * 0.001  # Characteristic time unit (seconds)
        
        # Characteristic distances
        self.r_fireball_max = 90 * self.W  # Maximum fireball radius (meters)
        self.r_shock_formation = 15 * self.W  # Shock wave formation radius (meters)
        
        # Energy partitioning
        self._calculate_energy_partition()
    
    def _calculate_energy_partition(self):
        """Calculate energy partition into different effects"""
        # Energy partition factors (approximate)
        self.energy_partition = {
            'blast': 0.5,      # Blast wave
            'thermal': 0.35,   # Thermal radiation
            'nuclear': 0.15,   # Nuclear radiation (prompt + delayed)
        }
        
        # Further partition nuclear radiation
        nuclear_energy = self.params.yield_joules * self.energy_partition['nuclear']
        self.nuclear_partition = {
            'prompt_neutron': 0.03,
            'prompt_gamma': 0.03,
            'fission_products': 0.07,
            'residual': 0.02
        }
    
    def initialize(self):
        """Initialize the blast model"""
        self.time = 0.0
        self._initialize_derived_parameters()
    
    def _calculate_fireball_radius(self, t: float) -> float:
        """Calculate fireball radius at time t"""
        if t <= 0:
            return 0.0
        
        # Simplified model based on scaling laws
        R_max = self.r_fireball_max
        tau = t / self.t_unit
        
        # Growth phase
        if tau < 1:
            return R_max * (tau ** 0.4)
        # Peak and decay phase
        else:
            return R_max * np.exp(-0.5 * (tau - 1))
    
    def _calculate_temperature(self, r: float, t: float) -> float:
        """Calculate temperature at radius r and time t"""
        if t <= 0:
            return self.params.ambient_temperature
        
        R_f = self._calculate_fireball_radius(t)
        if r > R_f:
            return self.params.ambient_temperature
        
        # Initial temperature (millions K)
        T_0 = 1e6 * self.params.yield_kt ** 0.3
        
        # Temperature decay with radius and time
        tau = t / self.t_unit
        r_norm = r / R_f
        
        T = T_0 * np.exp(-tau) * (1 - r_norm**2)
        return max(T, self.params.ambient_temperature)
    
    def _calculate_pressure(self, r: float, t: float) -> float:
        """Calculate pressure at radius r and time t"""
        if t <= 0:
            return self.params.ambient_pressure
        
        # Simplified Sedov-Taylor solution for blast wave
        R_s = self._calculate_shock_radius(t)
        if r > R_s:
            return self.params.ambient_pressure
        
        # Peak overpressure at shock front
        P_peak = self._calculate_peak_overpressure(R_s)
        
        # Pressure profile behind shock front
        r_norm = r / R_s
        P = P_peak * (1 - r_norm) * np.exp(-2 * (1 - r_norm))
        return max(P + self.params.ambient_pressure, self.params.ambient_pressure)
    
    def _calculate_shock_radius(self, t: float) -> float:
        """Calculate shock wave radius at time t"""
        if t <= 0:
            return 0.0
        
        # Sedov-Taylor solution
        E = self.params.yield_joules * self.energy_partition['blast']
        rho = self.params.ambient_pressure / (self.constants.k_B * self.params.ambient_temperature)
        
        return (E / rho) ** 0.2 * t ** 0.4
    
    def _calculate_peak_overpressure(self, r: float) -> float:
        """Calculate peak overpressure at radius r"""
        if r <= 0:
            return float('inf')
        
        # Empirical formula based on nuclear test data
        scaled_range = r / self.W
        
        if scaled_range < 3:
            return 1e6  # Very close to ground zero
        else:
            # Brode's formula (simplified)
            return 6.7 / scaled_range**3 + 1 / scaled_range * 1e5
    
    def step(self, dt: float):
        """Advance the simulation by dt seconds"""
        self.time += dt
    
    def get_observables(self) -> Dict[str, float]:
        """Return current observable quantities"""
        return {
            'time': self.time,
            'fireball_radius': self._calculate_fireball_radius(self.time),
            'shock_radius': self._calculate_shock_radius(self.time),
        }
    
    def get_radiation_flux(self, position: Vector3D, particle_type: str) -> float:
        """Calculate radiation flux at given position"""
        r = position.magnitude
        t = self.time
        
        if particle_type not in ['neutron', 'gamma', 'beta', 'alpha']:
            return 0.0
        
        # Get relevant energy fraction
        if particle_type in ['neutron', 'gamma']:
            energy_fraction = self.nuclear_partition.get(f'prompt_{particle_type}', 0.0)
        else:
            energy_fraction = self.nuclear_partition['fission_products'] / 2
        
        # Total energy in this radiation type
        E_total = self.params.yield_joules * self.energy_partition['nuclear'] * energy_fraction
        
        # Time-dependent flux calculation
        if t <= 0:
            return 0.0
        
        # Inverse square law with time decay
        if particle_type in ['neutron', 'gamma']:
            # Prompt radiation (exponential decay)
            flux = (E_total / (4 * np.pi * r**2)) * np.exp(-t / 1e-6)
        else:
            # Delayed radiation (t^-1.2 decay - Way-Wigner law)
            flux = (E_total / (4 * np.pi * r**2)) * (t ** -1.2)
        
        # Apply atmospheric attenuation
        mu = self._get_attenuation_coefficient(particle_type)
        flux *= np.exp(-mu * r)
        
        return max(flux, 0.0)
    
    def _get_attenuation_coefficient(self, particle_type: str) -> float:
        """Get atmospheric attenuation coefficient for given particle type"""
        # Approximate values in air at STP
        coefficients = {
            'neutron': 0.01,   # m^-1
            'gamma': 0.005,    # m^-1
            'beta': 0.1,       # m^-1
            'alpha': 1.0       # m^-1
        }
        return coefficients.get(particle_type, 0.0)

class NuclearParticleGenerator(ParticleGenerator):
    """Generator for nuclear blast particles"""
    def __init__(self, blast: NuclearBlast, num_particles: int = 1000):
        self.blast = blast
        self.num_particles = num_particles
        self.constants = PhysicalConstants()
        
        # Particle type distributions (example values)
        self.particle_ratios = {
            'neutron': 0.3,
            'gamma': 0.4,
            'beta': 0.2,
            'alpha': 0.1
        }
        
        # Energy distributions (mean energies in MeV)
        self.energy_means = {
            'neutron': 2.0,
            'gamma': 1.5,
            'beta': 0.5,
            'alpha': 5.0
        }
    
    def generate_particles(self, n: Optional[int] = None, time: float = 0) -> List[Particle]:
        """Generate n particles at given time"""
        if n is None:
            n = self.num_particles
        
        particles = []
        
        # Calculate particle counts for each type
        type_counts = {
            ptype: int(n * ratio)
            for ptype, ratio in self.particle_ratios.items()
        }
        
        # Generate particles for each type
        for ptype, count in type_counts.items():
            particles.extend(self._generate_type_particles(ptype, count, time))
        
        return particles
    
    def _generate_type_particles(self, particle_type: str, count: int, time: float) -> List[Particle]:
        """Generate particles of a specific type"""
        particles = []
        
        # Set particle properties based on type
        if particle_type == 'neutron':
            mass = 1.674927471e-27  # kg
            charge = 0
        elif particle_type == 'gamma':
            mass = 0
            charge = 0
        elif particle_type == 'beta':
            mass = 9.1093837015e-31  # kg
            charge = -1
        else:  # alpha
            mass = 6.644657230e-27  # kg
            charge = 2
        
        # Generate particles
        for _ in range(count):
            # Random position on blast surface
            theta = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            
            x = self.blast.params.distance * np.sin(phi) * np.cos(theta)
            y = self.blast.params.distance * np.sin(phi) * np.sin(theta)
            z = self.blast.params.distance * np.cos(phi) + self.blast.params.altitude
            
            position = Vector3D(x, y, z)
            
            # Direction towards center
            direction = Vector3D(0, 0, 0).normalize()
            
            # Random energy from exponential distribution
            energy = np.random.exponential(self.energy_means[particle_type])
            
            # Calculate velocity based on energy
            if mass > 0:
                # Relativistic velocity calculation
                total_energy = energy * self.constants.mev_to_joule + mass * self.constants.c**2
                gamma = total_energy / (mass * self.constants.c**2)
                v_mag = self.constants.c * np.sqrt(1 - 1/gamma**2)
            else:
                # Photons travel at speed of light
                v_mag = self.constants.c
            
            velocity = Vector3D(
                -direction.x * v_mag,
                -direction.y * v_mag,
                -direction.z * v_mag
            )
            
            # Create particle
            particle = Particle(
                position=position,
                velocity=velocity,
                mass=mass,
                charge=charge,
                energy=energy,
                particle_type=particle_type,
                time=time
            )
            
            particles.append(particle)
        
        return particles

class BlastSimulator:
    """Main simulator class for nuclear blast effects"""
    def __init__(self, blast_params: Optional[BlastParameters] = None):
        # Initialize with default parameters if none provided
        if blast_params is None:
            blast_params = BlastParameters(
                yield_kt=20.0,      # 20 kiloton yield
                distance=1000.0,    # 1 km from ground zero
                altitude=0.0,       # Surface burst
                detonation_time=0.0 # Immediate detonation
            )
        
        # Create blast model
        self.blast = NuclearBlast(blast_params)
        
        # Create particle generator
        self.particle_generator = NuclearParticleGenerator(self.blast)
        
        # Simulation state
        self.time = 0.0
        self.dt = 1e-6  # 1 microsecond timestep
        self.particles: List[Particle] = []
        self.particle_trajectories: Dict[int, List[Vector3D]] = {}
        
        # Performance metrics
        self.metrics = {
            'total_particles': 0,
            'escaped_particles': 0,
            'absorbed_particles': 0,
            'total_energy_deposited': 0.0
        }
    
    def initialize_simulation(self, num_particles: int = 1000):
        """Initialize simulation with given number of particles"""
        self.time = 0.0
        self.blast.initialize()
        self.particles = self.particle_generator.generate_particles(num_particles, self.time)
        self.particle_trajectories = {id(p): [p.position] for p in self.particles}
        self.metrics = {key: 0 for key in self.metrics}
        self.metrics['total_particles'] = num_particles
    
    def step(self):
        """Advance simulation by one time step"""
        # Update blast model
        self.blast.step(self.dt)
        
        # Update particles
        active_particles = []
        for particle in self.particles:
            # Get radiation flux at particle position
            flux = self.blast.get_radiation_flux(particle.position, particle.type)
            
            # Update particle state
            particle.update(self.dt)
            
            # Store trajectory
            self.particle_trajectories[id(particle)].append(particle.position)
            
            # Check if particle is still active
            if self._is_particle_active(particle):
                active_particles.append(particle)
            else:
                self._process_inactive_particle(particle)
        
        # Update active particles list
        self.particles = active_particles
        
        # Update time
        self.time += self.dt
    
    def _is_particle_active(self, particle: Particle) -> bool:
        """Check if particle is still active in simulation"""
        # Check if particle has enough energy
        if particle.energy <= 0.001:  # 1 keV threshold
            return False
        
        # Check if particle is too far from blast
        distance = particle.position.magnitude()
        if distance > self.blast.params.distance * 10:  # Arbitrary cutoff
            self.metrics['escaped_particles'] += 1
            return False
        
        return True
    
    def _process_inactive_particle(self, particle: Particle):
        """Process a particle that has become inactive"""
        # Update metrics
        if particle.energy <= 0.001:
            self.metrics['absorbed_particles'] += 1
            self.metrics['total_energy_deposited'] += particle.energy
    
    def run(self, total_time: float):
        """Run simulation for specified time"""
        num_steps = int(total_time / self.dt)
        for _ in range(num_steps):
            if not self.particles:  # Stop if no active particles
                break
            self.step()
    
    def get_results(self) -> Dict:
        """Get simulation results"""
        results = {
            'time': self.time,
            'metrics': self.metrics.copy(),
            'blast_observables': self.blast.get_observables(),
            'particle_trajectories': {
                str(pid): [pos for pos in traj]
                for pid, traj in self.particle_trajectories.items()
            }
        }
        
        # Calculate additional metrics
        if self.metrics['total_particles'] > 0:
            results['metrics']['absorption_ratio'] = (
                self.metrics['absorbed_particles'] / self.metrics['total_particles']
            )
            results['metrics']['escape_ratio'] = (
                self.metrics['escaped_particles'] / self.metrics['total_particles']
            )
        
        return results
    
    def get_particle_positions(self) -> List[Vector3D]:
        """Get current positions of all active particles"""
        return [p.position for p in self.particles]
    
    def get_particle_energies(self) -> List[float]:
        """Get current energies of all active particles"""
        return [p.energy for p in self.particles]
    
    def get_blast_radius(self) -> float:
        """Get current blast fireball radius"""
        return self.blast._calculate_fireball_radius(self.time)
    
    def get_shock_radius(self) -> float:
        """Get current shock wave radius"""
        return self.blast._calculate_shock_radius(self.time)
    
    def clear(self):
        """Clear simulation state"""
        self.particles.clear()
        self.particle_trajectories.clear()
        self.time = 0.0
        self.metrics = {key: 0 for key in self.metrics}