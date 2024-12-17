import numpy as np
from scipy.constants import G, hbar, c, k as k_B
from scipy.integrate import quad

class RelicNeutrinoParameters:
    """Relic nötrinolar için parametreler"""
    def __init__(self):
        # Kozmolojik parametreler (ΛCDM modeli)
        self.H0 = 67.4  # Hubble sabiti [km/s/Mpc]
        self.Omega_m = 0.315  # Madde yoğunluk parametresi
        self.Omega_Lambda = 0.685  # Karanlık enerji yoğunluk parametresi
        self.T_nu = 1.945  # Nötrino sıcaklığı [K]
        self.n_nu = 336  # Nötrino yoğunluğu [cm^-3]
        
        # Nötrino kütleleri [eV]
        self.m_nu = np.array([0.0, 0.009, 0.048])  # Normal hiyerarşi
        
        # Fiziksel sabitler
        self.G_F = 1.1663787e-5  # Fermi sabiti [GeV^-2]
        self.sin2_theta_w = 0.23122  # Zayıf karışım açısı
        self.m_Z = 91.1876  # Z bozon kütlesi [GeV]
        self.k_B = k_B  # Boltzmann sabiti [J/K]
        
        # Birim dönüşümleri
        self.GeV_to_eV = 1e9
        self.cm3_to_m3 = 1e6
        
    def get_neutrino_density(self, flavor):
        """Belirli bir nötrino türü için yoğunluğu hesapla"""
        # Her tür için eşit dağılım varsayılıyor
        return self.n_nu / 6  # 3 tür × 2 (nötrino + antinötrino)
    
    def get_neutrino_momentum_distribution(self, p, T=None):
        """Nötrino momentum dağılımını hesapla (Fermi-Dirac)"""
        if T is None:
            T = self.T_nu
        
        # Fermi-Dirac dağılımı
        E = np.sqrt(p*p + self.m_nu[0]**2)  # En hafif nötrino için
        return 1 / (np.exp(E/(k_B*T)) + 1)

class NeutrinoInteractions:
    """Relic nötrino etkileşimlerini hesaplayan sınıf"""
    def __init__(self, params=None):
        if params is None:
            params = RelicNeutrinoParameters()
        self.params = params
        
        # Etkileşim kesitleri için ön faktörler
        self.prefactor = (self.params.G_F**2 * self.params.m_Z**4 /
                         (4*np.pi * (hbar*c)**2))
    
    def _weak_form_factors(self, Q2):
        """Zayıf form faktörlerini hesapla"""
        sin2_theta_w = self.params.sin2_theta_w
        
        # Vektör ve aksiyal vektör form faktörleri
        g_V = -1/2 + 2*sin2_theta_w
        g_A = -1/2
        
        return g_V, g_A
    
    def differential_cross_section(self, E_e, cos_theta, E_nu):
        """Diferansiyel tesir kesitini hesapla
        
        Args:
            E_e: Elektron enerjisi [GeV]
            cos_theta: Saçılma açısının kosinüsü
            E_nu: Nötrino enerjisi [GeV]
        
        Returns:
            dsigma/dcos_theta [cm^2]
        """
        # Kinematik değişkenler
        s = 2*E_e*E_nu*(1 - cos_theta)  # Mandelstam değişkeni
        Q2 = s  # Momentum transferi
        
        # Form faktörleri
        g_V, g_A = self._weak_form_factors(Q2)
        
        # Diferansiyel tesir kesiti
        dsigma = self.prefactor * (
            (g_V + g_A)**2 * (1 + cos_theta)**2 +
            (g_V - g_A)**2 * (1 - cos_theta)**2
        )
        
        # GeV^-2 to cm^2
        return dsigma * 0.389379e-27
    
    def total_cross_section(self, E_e, E_nu):
        """Toplam tesir kesitini hesapla"""
        # cos_theta üzerinden integre et
        result, _ = quad(lambda x: self.differential_cross_section(E_e, x, E_nu),
                        -1, 1)
        return result
    
    def interaction_rate(self, electron_energy):
        """Etkileşim oranını hesapla [s^-1]"""
        E_e = electron_energy / self.params.GeV_to_eV  # eV to GeV
        
        # Nötrino enerjisi dağılımı üzerinden integre et
        def integrand(E_nu):
            # Tesir kesiti × nötrino akısı
            sigma = self.total_cross_section(E_e, E_nu)
            flux = self.params.get_neutrino_density('e') * c  # [m^-2 s^-1]
            return sigma * flux * 1e-4  # cm^2 to m^2
        
        # Tipik nötrino enerjileri için integre et
        E_nu_min = 1e-9  # GeV
        E_nu_max = 1e-6  # GeV
        
        result, _ = quad(integrand, E_nu_min, E_nu_max)
        return result
    
    def energy_loss_rate(self, electron_energy):
        """Enerji kaybı oranını hesapla [GeV/s]"""
        E_e = electron_energy / self.params.GeV_to_eV  # eV to GeV
        
        def integrand(E_nu, cos_theta):
            # Enerji transferi × diferansiyel tesir kesiti × nötrino akısı
            dE = E_e * (1 - cos_theta)  # Enerji transferi
            dsigma = self.differential_cross_section(E_e, cos_theta, E_nu)
            flux = self.params.get_neutrino_density('e') * c * 1e-4  # [m^-2 s^-1]
            return dE * dsigma * flux
        
        # Çift integral (enerji ve açı)
        E_nu_min = 1e-9
        E_nu_max = 1e-6
        
        result = 0
        for E_nu in np.linspace(E_nu_min, E_nu_max, 100):
            partial, _ = quad(lambda x: integrand(E_nu, x), -1, 1)
            result += partial * (E_nu_max - E_nu_min) / 100
        
        return result * self.params.GeV_to_eV  # GeV to eV
    
    def get_deflection_angle(self, electron_energy, dt):
        """Saçılma açısını hesapla
        
        Args:
            electron_energy: Elektron enerjisi [eV]
            dt: Zaman adımı [s]
        
        Returns:
            (theta, phi): Saçılma ve azimut açıları [radyan]
        """
        # Etkileşim olasılığı
        P_int = 1 - np.exp(-self.interaction_rate(electron_energy) * dt)
        
        if np.random.random() < P_int:
            # Etkileşim gerçekleşirse
            E_e = electron_energy / self.params.GeV_to_eV
            E_nu = np.random.uniform(1e-9, 1e-6)  # Tipik nötrino enerjisi
            
            # Diferansiyel tesir kesitine göre cos_theta seç
            cos_theta_vals = np.linspace(-1, 1, 1000)
            dsigma_vals = [self.differential_cross_section(E_e, cos_theta, E_nu)
                          for cos_theta in cos_theta_vals]
            dsigma_vals = np.array(dsigma_vals)
            dsigma_vals /= np.sum(dsigma_vals)
            
            cos_theta = np.random.choice(cos_theta_vals, p=dsigma_vals)
            theta = np.arccos(cos_theta)
            phi = np.random.uniform(0, 2*np.pi)
            
            return theta, phi
        else:
            # Etkileşim yoksa açı değişimi de yok
            return 0.0, 0.0 

class NeutrinoPhysicsEngine:
    """Engine for handling neutrino physics calculations in the simulation"""
    def __init__(self):
        self.params = RelicNeutrinoParameters()
        self.interactions = NeutrinoInteractions(self.params)
        self.active_neutrinos = []
        self.interaction_history = []
    
    def initialize(self):
        """Initialize the neutrino physics engine"""
        self.active_neutrinos.clear()
        self.interaction_history.clear()
    
    def update(self, dt, particles):
        """Update neutrino interactions with particles
        
        Args:
            dt: Time step [s]
            particles: List of particles to check for interactions
        """
        for particle in particles:
            if hasattr(particle, 'energy') and hasattr(particle, 'charge'):
                # Only consider charged particles for neutrino interactions
                if particle.charge != 0:
                    # Calculate interaction effects
                    energy_loss = self.interactions.energy_loss_rate(particle.energy) * dt
                    theta, phi = self.interactions.get_deflection_angle(particle.energy, dt)
                    
                    # Apply effects to particle
                    if energy_loss > 0:
                        particle.energy = max(0, particle.energy - energy_loss)
                    
                    if theta != 0 or phi != 0:
                        # Rotate velocity vector by calculated angles
                        v_mag = np.linalg.norm(particle.velocity)
                        sin_theta = np.sin(theta)
                        cos_theta = np.cos(theta)
                        sin_phi = np.sin(phi)
                        cos_phi = np.cos(phi)
                        
                        # New velocity components after deflection
                        vx = v_mag * sin_theta * cos_phi
                        vy = v_mag * sin_theta * sin_phi
                        vz = v_mag * cos_theta
                        
                        particle.velocity = np.array([vx, vy, vz])
                        
                        # Record interaction
                        self.interaction_history.append({
                            'time': dt,
                            'particle_type': particle.type,
                            'energy_loss': energy_loss,
                            'deflection_angle': theta
                        })
    
    def get_neutrino_density(self, position):
        """Get neutrino density at given position
        
        Args:
            position: numpy array [x, y, z] in meters
        
        Returns:
            Neutrino density in cm^-3
        """
        # For now, return uniform background density
        return self.params.n_nu
    
    def get_neutrino_flux(self, position, direction):
        """Get neutrino flux at given position and direction
        
        Args:
            position: numpy array [x, y, z] in meters
            direction: numpy array [dx, dy, dz] (normalized)
        
        Returns:
            Neutrino flux in cm^-2 s^-1
        """
        density = self.get_neutrino_density(position)
        return density * c * 100  # Convert m/s to cm/s
    
    def get_interaction_statistics(self):
        """Get statistics about neutrino interactions
        
        Returns:
            Dictionary with interaction statistics
        """
        if not self.interaction_history:
            return {
                'total_interactions': 0,
                'total_energy_loss': 0,
                'average_deflection': 0
            }
        
        total_interactions = len(self.interaction_history)
        total_energy_loss = sum(event['energy_loss'] for event in self.interaction_history)
        average_deflection = sum(event['deflection_angle'] for event in self.interaction_history) / total_interactions
        
        return {
            'total_interactions': total_interactions,
            'total_energy_loss': total_energy_loss,
            'average_deflection': average_deflection
        }
    
    def clear(self):
        """Clear all neutrino physics data"""
        self.initialize()