import numpy as np
from typing import List, Dict, Tuple, Optional
from core_physics import Vector3D, PhysicsObject, PhysicalConstants
from magnetic_field import MagneticField
from scipy.integrate import solve_ivp
import numba
from radiation_shield import NuclearBlastShieldSimulation

class PlasmaParameters:
    def __init__(self, temperature: float, density: float, ionization_degree: float):
        self.temperature = temperature  # Kelvin
        self.density = density  # particles/m³
        self.ionization_degree = ionization_degree  # 0 to 1
        self.constants = PhysicalConstants()
        
        # Derived parameters
        self._update_derived_parameters()
    
    def _update_derived_parameters(self):
        """Türetilmiş plazma parametrelerini güncelle"""
        # Debye uzunluğu
        self.debye_length = np.sqrt(
            self.constants.epsilon_0 * self.constants.k_B * self.temperature /
            (self.density * self.ionization_degree * self.constants.e**2)
        )
        
        # Plazma frekansı
        self.plasma_frequency = np.sqrt(
            self.density * self.ionization_degree * self.constants.e**2 /
            (self.constants.m_e * self.constants.epsilon_0)
        )
        
        # Elektron termal hızı
        self.electron_thermal_velocity = np.sqrt(
            2 * self.constants.k_B * self.temperature / self.constants.m_e
        )
        
        # İyon akustik hızı
        self.ion_acoustic_speed = np.sqrt(
            self.constants.k_B * self.temperature / self.constants.m_p
        )
        
        # Çarpışma frekansı
        self.collision_frequency = self.density * self.ionization_degree * np.pi * \
            self.debye_length**2 * self.electron_thermal_velocity

class PlasmaShield:
    def __init__(self, parameters: PlasmaParameters, magnetic_field: 'MagneticField'):
        self.params = parameters
        self.magnetic_field = magnetic_field
        self.constants = PhysicalConstants()
        
        # Plazma durumu
        self.electron_density = np.zeros((50, 50, 50))  # Basit 3D ızgara
        self.ion_density = np.zeros((50, 50, 50))
        self.current_density = np.zeros((50, 50, 50, 3))
        self.electric_field = np.zeros((50, 50, 50, 3))
        
        # PIC parametreleri
        self.grid_size = 0.1  # metre
        self.time_step = 1e-12  # saniye
        self.particle_weight = 1e10  # makro-parçacık ağırlığı
    
    @numba.jit(nopython=True)
    def calculate_particle_deflection(self, particle: PhysicsObject) -> Tuple[float, Vector3D]:
        """Parçacık sapmasını hesapla"""
        if particle.charge == 0:
            return 0.0, particle.velocity
        
        # Manyetik alan
        B = self.magnetic_field.get_field_at(particle.position)
        v = particle.velocity.magnitude()
        
        # Siklotron frekansı
        omega_c = abs(particle.charge * B.magnitude() / particle.mass)
        
        # Etkileşim süresi
        interaction_time = 2 * self.params.debye_length / v
        
        # Sapma açısı
        deflection_angle = np.arctan(omega_c * interaction_time)
        
        # Yeni hız vektörü
        B_unit = B.normalize()
        v_parallel = particle.velocity.dot(B_unit) * B_unit
        v_perp = particle.velocity - v_parallel
        
        if v_perp.magnitude() > 0:
            v_perp_unit = v_perp.normalize()
            v_perp_new = v_perp.magnitude() * (
                v_perp_unit * np.cos(deflection_angle) +
                B_unit.cross(v_perp_unit) * np.sin(deflection_angle)
            )
        else:
            v_perp_new = Vector3D(0, 0, 0)
        
        new_velocity = v_parallel + v_perp_new
        
        return deflection_angle, new_velocity
    
    def calculate_energy_loss(self, particle: PhysicsObject) -> float:
        """Plazma etkileşimleri nedeniyle enerji kaybını hesapla"""
        if particle.charge == 0:
            return 0.0
        
        # Bethe formülü
        beta = particle.velocity.magnitude() / self.constants.c
        gamma = 1 / np.sqrt(1 - beta**2)
        
        # Ortalama uyarılma enerjisi
        I = 10 * self.constants.e  # 10 eV
        
        dE_dx = (
            4 * np.pi * self.constants.e**4 * self.params.density *
            self.params.ionization_degree / (self.constants.m_e * self.constants.c**2 * beta**2)
        ) * (
            np.log(2 * self.constants.m_e * self.constants.c**2 * beta**2 * gamma**2 / I) -
            beta**2
        )
        
        # Plazma yoğunluğu düzeltmesi
        density_factor = self._get_local_density_factor(particle.position)
        
        return dE_dx * self.params.debye_length * density_factor
    
    def _get_local_density_factor(self, position: Vector3D) -> float:
        """Yerel plazma yoğunluğu faktörünü hesapla"""
        # Izgara indekslerini hesapla
        x_idx = int((position.x + 2.5) / self.grid_size) % 50
        y_idx = int((position.y + 2.5) / self.grid_size) % 50
        z_idx = int((position.z + 2.5) / self.grid_size) % 50
        
        # Normalize edilmiş yoğunluk
        return (self.electron_density[x_idx, y_idx, z_idx] + 
                self.ion_density[x_idx, y_idx, z_idx]) / (2 * self.params.density)

class PlasmaDynamics:
    def __init__(self, shield: PlasmaShield):
        self.shield = shield
        self.constants = PhysicalConstants()
        
        # Zaman entegrasyonu için parametreler
        self.dt = 1e-12  # saniye
        self.substeps = 10
    
    def update_particle_state(self, particle: PhysicsObject, dt: float):
        """Parçacık durumunu güncelle"""
        # Alt adımlar için Runge-Kutta 4 integrasyonu
        t_span = (0, dt)
        y0 = [
            particle.position.x, particle.position.y, particle.position.z,
            particle.velocity.x, particle.velocity.y, particle.velocity.z
        ]
        
        def deriv(t, y):
            pos = Vector3D(y[0], y[1], y[2])
            vel = Vector3D(y[3], y[4], y[5])
            
            # Toplam kuvvet
            F = self._calculate_total_force(particle, pos, vel)
            
            return [
                y[3], y[4], y[5],  # dx/dt = v
                F.x/particle.mass, F.y/particle.mass, F.z/particle.mass  # dv/dt = F/m
            ]
        
        # Sayısal integrasyon
        sol = solve_ivp(deriv, t_span, y0, method='RK45', rtol=1e-8, atol=1e-8)
        
        # Son durumu güncelle
        y = sol.y[:,-1]
        particle.position = Vector3D(y[0], y[1], y[2])
        particle.velocity = Vector3D(y[3], y[4], y[5])
        
        # Enerji kaybı
        energy_loss = self.shield.calculate_energy_loss(particle)
        particle.energy = max(0.0, particle.energy - energy_loss * dt)
    
    def _calculate_total_force(self, particle: PhysicsObject, 
                             position: Vector3D, velocity: Vector3D) -> Vector3D:
        """Toplam kuvveti hesapla"""
        # Lorentz kuvveti
        B = self.shield.magnetic_field.get_field_at(position)
        E = self._get_local_electric_field(position)
        
        F_lorentz = Vector3D(0, 0, 0)
        if particle.charge != 0:
            # Elektrik alan kuvveti
            F_E = Vector3D(
                particle.charge * E[0],
                particle.charge * E[1],
                particle.charge * E[2]
            )
            
            # Manyetik alan kuvveti
            v_cross_B = velocity.cross(B)
            F_B = Vector3D(
                particle.charge * v_cross_B.x,
                particle.charge * v_cross_B.y,
                particle.charge * v_cross_B.z
            )
            
            F_lorentz = F_E + F_B
        
        # Plazma sürtünme kuvveti
        v_mag = velocity.magnitude()
        if v_mag > 0:
            drag_coef = self._calculate_plasma_drag(particle, position, v_mag)
            F_drag = velocity * (-drag_coef / v_mag)
        else:
            F_drag = Vector3D(0, 0, 0)
        
        return F_lorentz + F_drag
    
    def _get_local_electric_field(self, position: Vector3D) -> np.ndarray:
        """Yerel elektrik alanını al"""
        x_idx = int((position.x + 2.5) / self.shield.grid_size) % 50
        y_idx = int((position.y + 2.5) / self.shield.grid_size) % 50
        z_idx = int((position.z + 2.5) / self.shield.grid_size) % 50
        
        return self.shield.electric_field[x_idx, y_idx, z_idx]
    
    def _calculate_plasma_drag(self, particle: PhysicsObject, 
                             position: Vector3D, velocity: float) -> float:
        """Plazma sürtünme katsayısını hesapla"""
        if particle.charge == 0:
            return 0.0
        
        # Coulomb logaritması
        lambda_D = self.shield.params.debye_length
        b_min = abs(particle.charge * self.constants.e) / \
                (4 * np.pi * self.constants.epsilon_0 * particle.mass * velocity**2)
        ln_Lambda = np.log(lambda_D / b_min)
        
        # Sürtünme katsayısı
        return (
            4 * np.pi * particle.charge**2 * self.constants.e**4 * 
            self.shield.params.density * ln_Lambda /
            (self.constants.m_e * velocity**3)
        )

class PlasmaInstabilityAnalyzer:
    def __init__(self, shield: PlasmaShield):
        self.shield = shield
    
    def calculate_growth_rates(self) -> Dict[str, float]:
        """Plazma kararsızlıklarının büyüme hızlarını hesapla"""
        B = self.shield.magnetic_field.get_field_at(Vector3D(0, 0, 0)).magnitude()
        
        growth_rates = {
            'kink': self._kink_instability_rate(B),
            'sausage': self._sausage_instability_rate(B),
            'drift': self._drift_instability_rate(),
            'two_stream': self._two_stream_instability_rate(),
            'weibel': self._weibel_instability_rate(),
            'fire_hose': self._fire_hose_instability_rate(B)
        }
        
        return growth_rates
    
    def _kink_instability_rate(self, B: float) -> float:
        """Kink kararsızlığı büyüme hızı"""
        return np.sqrt(
            self.shield.params.plasma_frequency**2 *
            (1 - B**2 / (self.shield.constants.mu_0 * self.shield.params.density * 
             self.shield.params.temperature))
        )
    
    def _sausage_instability_rate(self, B: float) -> float:
        """Sausage kararsızlığı büyüme hızı"""
        return 0.5 * self._kink_instability_rate(B)
    
    def _drift_instability_rate(self) -> float:
        """Drift kararsızlığı büyüme hızı"""
        return 0.1 * self.shield.params.plasma_frequency
    
    def _two_stream_instability_rate(self) -> float:
        """Two-stream kararsızlığı büyüme hızı"""
        return 0.7 * self.shield.params.plasma_frequency
    
    def _weibel_instability_rate(self) -> float:
        """Weibel kararsızlığı büyüme hızı"""
        T_perp = self.shield.params.temperature  # Dik sıcaklık
        T_para = 0.5 * T_perp  # Paralel sıcaklık (varsayılan)
        
        anisotropy = T_perp/T_para - 1
        return self.shield.params.plasma_frequency * np.sqrt(anisotropy)
    
    def _fire_hose_instability_rate(self, B: float) -> float:
        """Fire-hose kararsızlığı büyüme hızı"""
        beta = (
            2 * self.shield.constants.mu_0 * self.shield.params.density *
            self.shield.constants.k_B * self.shield.params.temperature
        ) / B**2
        
        return self.shield.params.plasma_frequency * np.sqrt(beta - 1)

class PlasmaPhysicsEngine:
    """Main engine for plasma physics calculations and simulations"""
    def __init__(self, magnetic_field: Optional['MagneticField'] = None):
        # Initialize plasma parameters with default values
        self.parameters = PlasmaParameters(
            temperature=1e6,  # 1 million Kelvin
            density=1e20,     # particles/m³
            ionization_degree=0.9  # 90% ionized
        )
        
        # Initialize magnetic field if provided
        self.magnetic_field = magnetic_field
        
        # Create plasma shield
        self.shield = PlasmaShield(self.parameters, self.magnetic_field)
        
        # Initialize dynamics calculator
        self.dynamics = PlasmaDynamics(self.shield)
        
        # Initialize instability analyzer
        self.instability_analyzer = PlasmaInstabilityAnalyzer(self.shield)
        
        # Simulation state
        self.time = 0.0
        self.dt = 1e-12  # 1 picosecond timestep
        self.particles = []
        
    def set_parameters(self, temperature: float, density: float, ionization_degree: float):
        """Update plasma parameters"""
        self.parameters = PlasmaParameters(temperature, density, ionization_degree)
        self.shield = PlasmaShield(self.parameters, self.magnetic_field)
        self.dynamics = PlasmaDynamics(self.shield)
        self.instability_analyzer = PlasmaInstabilityAnalyzer(self.shield)
    
    def add_particle(self, particle: PhysicsObject):
        """Add a particle to the simulation"""
        self.particles.append(particle)
    
    def step(self, dt: Optional[float] = None):
        """Advance simulation by one time step"""
        if dt is not None:
            self.dt = dt
            
        # Update each particle
        for particle in self.particles:
            # Calculate plasma effects
            deflection_angle, new_velocity = self.shield.calculate_particle_deflection(particle)
            particle.velocity = new_velocity
            
            # Update particle state with plasma dynamics
            self.dynamics.update_particle_state(particle, self.dt)
        
        # Update time
        self.time += self.dt
        
        # Check for instabilities
        self.check_instabilities()
    
    def check_instabilities(self):
        """Check for plasma instabilities"""
        growth_rates = self.instability_analyzer.calculate_growth_rates()
        
        # Flag dangerous instabilities
        dangerous_modes = []
        for mode, rate in growth_rates.items():
            if rate > 0.1 * self.parameters.plasma_frequency:
                dangerous_modes.append((mode, rate))
        
        return dangerous_modes
    
    def get_particle_energy_loss(self, particle: PhysicsObject) -> float:
        """Calculate energy loss for a particle"""
        return self.shield.calculate_energy_loss(particle)
    
    def get_deflection_angle(self, particle: PhysicsObject) -> float:
        """Calculate deflection angle for a particle"""
        angle, _ = self.shield.calculate_particle_deflection(particle)
        return angle
    
    def get_plasma_parameters(self) -> Dict[str, float]:
        """Get current plasma parameters"""
        return {
            'temperature': self.parameters.temperature,
            'density': self.parameters.density,
            'ionization_degree': self.parameters.ionization_degree,
            'debye_length': self.parameters.debye_length,
            'plasma_frequency': self.parameters.plasma_frequency,
            'electron_thermal_velocity': self.parameters.electron_thermal_velocity,
            'ion_acoustic_speed': self.parameters.ion_acoustic_speed,
            'collision_frequency': self.parameters.collision_frequency
        }
    
    def get_field_strength(self, position: Vector3D) -> float:
        """Get magnetic field strength at position"""
        if self.magnetic_field:
            return self.magnetic_field.get_field_at(position).magnitude()
        return 0.0
    
    def clear(self):
        """Clear all particles and reset time"""
        self.particles.clear()
        self.time = 0.0