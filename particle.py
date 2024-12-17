import numpy as np

class Particle:
    """Temel parçacık sınıfı"""
    def __init__(self, position, velocity, energy, mass, charge):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.energy = float(energy)  # MeV
        self.mass = float(mass)  # kg
        self.charge = float(charge)  # C
        self.trajectory = [self.position.copy()]
        self.energies = [self.energy]
        self.velocities = [self.velocity.copy()]
        
    def update(self, dt, force=None):
        """Parçacığın konumunu ve hızını güncelle"""
        if force is not None and np.any(force != 0):
            acceleration = force / self.mass
            self.velocity += acceleration * dt
        
        self.position += self.velocity * dt
        self.trajectory.append(self.position.copy())
        self.velocities.append(self.velocity.copy())
        self.energies.append(self.energy)
    
    def update_energy(self, dE_dt, dt):
        """Parçacığın enerjisini güncelle"""
        self.energy += dE_dt * dt
        if self.energy < 0:
            self.energy = 0
        
        # Enerji değişimine göre hızı güncelle
        if self.mass > 0:
            v_mag = np.sqrt(2 * abs(self.energy) * 1e6 * 1.602e-19 / self.mass)
            v_dir = self.velocity / np.linalg.norm(self.velocity)
            self.velocity = v_dir * v_mag

class AlphaParticle(Particle):
    """Alfa parçacığı"""
    def __init__(self, position, velocity, energy):
        m_p = 1.672e-27  # Proton kütlesi [kg]
        e = 1.602e-19    # Elektron yükü [C]
        super().__init__(position, velocity, energy, 4*m_p, 2*e)
        self.type = 'alpha'

class BetaParticle(Particle):
    """Beta parçacığı (elektron)"""
    def __init__(self, position, velocity, energy):
        m_e = 9.109e-31  # Elektron kütlesi [kg]
        e = 1.602e-19    # Elektron yükü [C]
        super().__init__(position, velocity, energy, m_e, -e)
        self.type = 'beta'

class Neutron(Particle):
    """Nötron"""
    def __init__(self, position, velocity, energy):
        m_n = 1.674e-27  # Nötron kütlesi [kg]
        super().__init__(position, velocity, energy, m_n, 0)
        self.type = 'neutron'

class GammaRay(Particle):
    """Gama ışını"""
    def __init__(self, position, velocity, energy):
        c = 2.998e8  # Işık hızı [m/s]
        super().__init__(position, velocity, energy, 0, 0)
        self.type = 'gamma'
        # Gama ışınları her zaman ışık hızında hareket eder
        self.velocity = self.velocity / np.linalg.norm(self.velocity) * c

def create_particle(particle_type, position, energy, direction):
    """Belirtilen tipte parçacık oluştur"""
    # Yön vektörünü normalize et
    direction = np.array(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    
    # Fiziksel sabitler
    e = 1.602e-19    # Elektron yükü [C]
    m_e = 9.109e-31  # Elektron kütlesi [kg]
    m_p = 1.672e-27  # Proton kütlesi [kg]
    c = 2.998e8      # Işık hızı [m/s]
    
    if particle_type == 'alpha':
        # Alfa parçacığı için hız hesapla (klasik)
        v_mag = np.sqrt(2 * energy * 1e6 * e / (4*m_p))
        velocity = direction * v_mag
        return AlphaParticle(position, velocity, energy)
        
    elif particle_type == 'beta':
        # Beta parçacığı için hız hesapla (klasik)
        v_mag = np.sqrt(2 * energy * 1e6 * e / m_e)
        velocity = direction * v_mag
        return BetaParticle(position, velocity, energy)
        
    elif particle_type == 'neutron':
        # Nötron için hız hesapla (klasik)
        v_mag = np.sqrt(2 * energy * 1e6 * e / m_p)
        velocity = direction * v_mag
        return Neutron(position, velocity, energy)
        
    elif particle_type == 'gamma':
        # Gama ışını her zaman ışık hızında hareket eder
        velocity = direction * c
        return GammaRay(position, velocity, energy)
        
    else:
        raise ValueError(f"Geçersiz parçacık tipi: {particle_type}") 

class ParticleManager:
    """Manages particle generation and tracking for the simulation"""
    def __init__(self):
        self.particles = []
        self.particle_types = ['alpha', 'beta', 'neutron', 'gamma']
        self.particle_counts = {ptype: 0 for ptype in self.particle_types}
        self.total_energy = 0.0
    
    def generate_particles(self, num_particles, source_position, energy_range, direction=None):
        """Generate particles with given parameters
        
        Args:
            num_particles (int): Number of particles to generate
            source_position (np.ndarray): Source position [x, y, z]
            energy_range (tuple): (min_energy, max_energy) in MeV
            direction (np.ndarray, optional): Preferred direction [x, y, z]
        """
        new_particles = []
        min_energy, max_energy = energy_range
        
        for _ in range(num_particles):
            # Randomly select particle type
            particle_type = np.random.choice(self.particle_types)
            
            # Random energy within range
            energy = np.random.uniform(min_energy, max_energy)
            
            # Generate random direction if not specified
            if direction is None:
                phi = np.random.uniform(0, 2*np.pi)
                theta = np.random.uniform(0, np.pi)
                direction = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
            
            # Create particle
            particle = create_particle(particle_type, source_position, energy, direction)
            new_particles.append(particle)
            
            # Update statistics
            self.particle_counts[particle_type] += 1
            self.total_energy += energy
        
        self.particles.extend(new_particles)
        return new_particles
    
    def update_particles(self, dt, magnetic_field=None, electric_field=None):
        """Update all particles' positions and energies
        
        Args:
            dt (float): Time step
            magnetic_field (callable, optional): Function B(r) returning magnetic field at position r
            electric_field (callable, optional): Function E(r) returning electric field at position r
        """
        active_particles = []
        
        for particle in self.particles:
            # Calculate Lorentz force if fields are present and particle is charged
            force = np.zeros(3)
            if particle.charge != 0:
                if electric_field is not None:
                    E = electric_field(particle.position)
                    force += particle.charge * E
                
                if magnetic_field is not None:
                    B = magnetic_field(particle.position)
                    v_cross_B = np.cross(particle.velocity, B)
                    force += particle.charge * v_cross_B
            
            # Update particle state
            particle.update(dt, force)
            
            # Keep only active particles (energy > 0)
            if particle.energy > 0:
                active_particles.append(particle)
            else:
                self.particle_counts[particle.type] -= 1
                self.total_energy -= particle.energy
        
        self.particles = active_particles
    
    def get_particle_positions(self):
        """Get current positions of all particles"""
        return np.array([p.position for p in self.particles])
    
    def get_particle_velocities(self):
        """Get current velocities of all particles"""
        return np.array([p.velocity for p in self.particles])
    
    def get_particle_energies(self):
        """Get current energies of all particles"""
        return np.array([p.energy for p in self.particles])
    
    def get_trajectories(self):
        """Get trajectories of all particles"""
        return {i: p.trajectory for i, p in enumerate(self.particles)}
    
    def get_statistics(self):
        """Get particle statistics"""
        return {
            'total_particles': len(self.particles),
            'particle_counts': self.particle_counts.copy(),
            'total_energy': self.total_energy,
            'average_energy': self.total_energy / len(self.particles) if self.particles else 0
        }
    
    def clear(self):
        """Clear all particles and reset statistics"""
        self.particles.clear()
        self.particle_counts = {ptype: 0 for ptype in self.particle_types}
        self.total_energy = 0.0