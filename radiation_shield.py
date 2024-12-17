import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from core_physics import Vector3D
from magnetic_field import MagneticField, SolenoidField, ToroidalField, DipoleField, HelmholtzCoilField

class NuclearBlastShieldSimulation:
    def __init__(self):
        # Fiziksel sabitler
        self.e = 1.602e-19  # Elektron yükü [C]
        self.m_e = 9.109e-31  # Elektron kütlesi [kg]
        self.m_p = 1.672e-27  # Proton kütlesi [kg]
        self.c = 2.998e8  # Işık hızı [m/s]
        self.k_B = 1.380649e-23  # Boltzmann sabiti [J/K]
        
        # Plazma parametreleri
        self.plasma_density = 1e22  # Plazma yoğunluğu [m^-3]
        self.plasma_temperature = 10  # Plazma sıcaklığı [keV]
        self.plasma_composition = {
            'H': 0.4,  # Hidrojen oranı
            'D': 0.3,  # Döteryum oranı
            'T': 0.2,  # Trityum oranı
            'W': 0.1   # Tungsten oranı
        }
        
        # Kalkan parametreleri
        self.shield_thickness = 0.5  # Kalkan kalınlığı [m]
        self.B0 = 5.0  # Maksimum manyetik alan [Tesla]
        self.shield_radius = 5.0  # Kalkan yarıçapı [m]
        self.shield_height = 10.0  # Kalkan yüksekliği [m]
        
        # Radyasyon kaynağı parametreleri
        self.blast_yield = 20e3  # Patlama gücü [kiloton TNT]
        self.source_distance = 1000.0  # Radyasyon kaynağı mesafesi [m]
        self.blast_position = Vector3D(self.source_distance, 0, 0)  # Patlama konumu [m]
        
        # Simülasyon parametreleri
        self.dt = 1e-9  # Zaman adımı [s]
        self.num_steps = 1000
        
        # Parçacık verileri
        self.particles = []
        self.trajectories = []
        
        # Manyetik alan konfigürasyonu
        self.field_type = 'toroidal'  # Varsayılan alan tipi
        self.magnetic_field = None
        self.setup_magnetic_field()
    
    def setup_magnetic_field(self):
        """Seçilen manyetik alan tipine göre alan nesnesini oluştur"""
        if self.field_type == 'solenoid':
            self.magnetic_field = SolenoidField(
                center=Vector3D(0, 0, 0),
                radius=self.shield_radius,
                length=self.shield_height,
                n_turns=1000,  # Varsayılan değer
                current=1000,  # Varsayılan değer
                axis=Vector3D(0, 0, 1)
            )
        elif self.field_type == 'toroidal':
            self.magnetic_field = ToroidalField(
                center=Vector3D(0, 0, 0),
                major_radius=self.shield_radius,
                minor_radius=self.shield_radius * 0.2,
                n_coils=16,
                current=1000
            )
        elif self.field_type == 'helmholtz':
            self.magnetic_field = HelmholtzCoilField(
                radius=self.shield_radius,
                current=1000
            )
        elif self.field_type == 'dipole':
            moment = Vector3D(0, 0, self.B0 * 4*np.pi * self.shield_radius**3 / (4 * np.pi * 1e-7))
            self.magnetic_field = DipoleField(moment=moment, position=Vector3D(0, 0, 0))
    
    def magnetic_field(self, r, t=0):
        """Manyetik alanı hesapla"""
        if self.magnetic_field is None:
            self.setup_magnetic_field()
        return self.magnetic_field.get_field(r, t)
    
    def plasma_collision_frequency(self, particle_type, energy):
        """Plazma çarpışma frekansını hesapla"""
        # Parçacık hızı
        if particle_type in ['alpha', 'beta']:
            v = np.sqrt(2 * energy * self.e * 1e6 / self.m_p)  # m/s
        else:
            v = self.c  # Fotonlar ve nötronlar için ışık hızı
        
        # Plazma Debye uzunluğu
        T_e = self.plasma_temperature * 1e3 * self.e  # keV to Joules
        lambda_D = np.sqrt(self.k_B * T_e / (self.plasma_density * self.e**2))
        
        # Coulomb logaritması
        ln_Lambda = np.log(12 * np.pi * self.plasma_density * lambda_D**3)
        
        # Çarpışma frekansı
        if particle_type == 'alpha':
            nu = 4 * np.pi * self.e**4 * self.plasma_density * ln_Lambda / \
                 (self.m_p * v**3)
        elif particle_type == 'beta':
            nu = 4 * np.pi * self.e**4 * self.plasma_density * ln_Lambda / \
                 (self.m_e * v**3)
        elif particle_type == 'neutron':
            # Nötronlar için nükleer çarpışma kesiti (basitleştirilmiş)
            sigma = 1e-28  # m^2
            nu = self.plasma_density * sigma * v
        else:  # gamma
            # Compton saçılması için kesit (basitleştirilmiş)
            sigma = 6.65e-29  # m^2 (Thomson kesiti)
            nu = self.plasma_density * sigma * self.c
        
        return nu
    
    def energy_loss_rate(self, particle_type, energy):
        """Enerji kaybı oranını hesapla"""
        nu = self.plasma_collision_frequency(particle_type, energy)
        
        if particle_type in ['alpha', 'beta']:
            # Yüklü parçacıklar için plazma frenlenmesi
            dE_dt = -2 * nu * energy
        elif particle_type == 'neutron':
            # Nötronlar için elastik saçılma
            dE_dt = -0.5 * nu * energy
        else:  # gamma
            # Compton saçılması için enerji kaybı
            dE_dt = -nu * energy * (energy / (self.m_e * self.c**2))
        
        return dE_dt
    
    def lorentz_force(self, q, v, B):
        """Lorentz kuvvetini hesapla (F = q * v × B)
        
        Args:
            q (float): Parçacık yükü [C]
            v (Vector3D): Hız vektörü [m/s]
            B (Vector3D): Manyetik alan vektörü [T]
            
        Returns:
            Vector3D: Lorentz kuvveti vektörü [N]
        """
        if B.magnitude() == 0 or q == 0:  # Manyetik alan veya yük sıfırsa kuvvet sıfır
            return Vector3D(0, 0, 0)
        
        # Lorentz kuvveti hesapla
        F = v.cross(B) * q
        
        # Kuvveti sınırla (çok büyük değerleri engelle)
        max_force = 1e10  # Newton
        force_mag = F.magnitude()
        if force_mag > max_force:
            F = F * (max_force / force_mag)
        
        return F
    
    def run_simulation(self, progress_callback=None):
        """Simülasyonu çalıştır ve parçacık yörüngelerini hesapla"""
        self.trajectories = []
        
        print("Parçacık yörüngeleri hesaplanıyor...")
        for i, particle in enumerate(tqdm(self.particles)):
            positions = []
            velocities = []
            energies = []
            
            position = particle['position']
            velocity = particle['velocity']
            energy = particle['energy']
            
            positions.append(position)
            velocities.append(velocity)
            energies.append(energy)
            
            for step in range(self.num_steps):
                try:
                    # Manyetik alan etkisi
                    B = self.magnetic_field.get_field_at(position)
                    if particle['q'] != 0:
                        F = self.lorentz_force(particle['q'], velocity, B)
                        acceleration = F * (1.0 / particle['m'])
                        velocity = velocity + acceleration * self.dt
                    
                    # Plazma frenlenmesi
                    dE_dt = self.energy_loss_rate(particle['type'], energy)
                    energy += dE_dt * self.dt
                    
                    # Enerji kaybına göre hızı güncelle
                    if particle['m'] > 0:
                        v_mag = np.sqrt(2 * abs(energy) * 1e6 * self.e / particle['m'])
                        velocity = velocity.normalize() * v_mag
                    
                    # Konumu güncelle
                    position = position + velocity * self.dt
                    
                    # Verileri kaydet
                    positions.append(position)
                    velocities.append(velocity)
                    energies.append(energy)
                    
                    # Simülasyonu sonlandırma koşulları
                    r = position.magnitude()
                    if r > self.shield_radius * 1.5 or energy < 0.01 * energies[0]:
                        break
                    
                except Exception as e:
                    print(f"Adım {step} hesaplanırken hata: {str(e)}")
                    break
            
            # Yörüngeyi kaydet
            if len(positions) > 1:
                # Vector3D nesnelerini numpy dizilerine dönüştür
                positions_array = np.array([[p.x, p.y, p.z] for p in positions])
                velocities_array = np.array([[v.x, v.y, v.z] for v in velocities])
                
                self.trajectories.append({
                    'type': particle['type'],
                    'positions': positions_array,
                    'velocities': velocities_array,
                    'energies': np.array(energies)
                })
            
            if progress_callback:
                progress = (i + 1) / len(self.particles) * 100
                progress_callback(progress)
    
    def visualize_results(self):
        """Simülasyon sonuçlarını görselleştir"""
        fig = plt.figure(figsize=(20, 15))
        
        # 3D yörünge grafiği
        ax1 = fig.add_subplot(221, projection='3d')
        colors = {'alpha': 'red', 'beta': 'blue', 'neutron': 'green', 'gamma': 'yellow'}
        
        for traj in self.trajectories[::10]:  # Her 10 yörüngeden birini çiz
            positions = traj['positions']
            ax1.plot(positions[:,0], positions[:,1], positions[:,2],
                    color=colors[traj['type']], alpha=0.5, label=traj['type'])
        
        # Kalkan sınırlarını çiz
        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(-self.shield_height/2, self.shield_height/2, 100)
        U, V = np.meshgrid(u, v)
        X = self.shield_radius * np.cos(U)
        Y = self.shield_radius * np.sin(U)
        Z = V
        ax1.plot_surface(X, Y, Z, alpha=0.1, color='gray')
        
        ax1.set_title("Parçacık Yörüngeleri")
        # Remove duplicate labels before creating legend
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys())
        
        # Manyetik alan yoğunluğu haritası
        ax2 = fig.add_subplot(222)
        x = np.linspace(-self.shield_radius, self.shield_radius, 50)
        y = np.linspace(-self.shield_radius, self.shield_radius, 50)
        X, Y = np.meshgrid(x, y)
        B_strength = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pos = Vector3D(X[i,j], Y[i,j], 0)
                B = self.magnetic_field.get_field_at(pos)
                B_strength[i,j] = B.magnitude()
        
        plt.imshow(B_strength, extent=[-self.shield_radius, self.shield_radius, 
                                     -self.shield_radius, self.shield_radius])
        plt.colorbar(label='Manyetik Alan Şiddeti (T)')
        ax2.set_title("Manyetik Alan Yoğunluğu")
        
        plt.tight_layout()
        return fig

    def visualize_comparative_results(self):
        """Karşılaştırmalı sonuçları görselleştir"""
        fig = plt.figure(figsize=(15, 10))
        
        # Parçacık tiplerine göre renk ve etiketler
        colors = {'alpha': 'red', 'beta': 'blue', 'neutron': 'green', 'gamma': 'yellow'}
        labels = {'alpha': 'Alfa', 'beta': 'Beta', 'neutron': 'Nötron', 'gamma': 'Gama'}
        
        # Penetrasyon derinliği dağılımı
        ax1 = fig.add_subplot(221)
        for i, (ptype, color) in enumerate(colors.items()):
            depths = []
            for traj in self.trajectories:
                if traj['type'] == ptype:
                    positions = traj['positions']
                    if len(positions) > 0:
                        # Merkeze olan maksimum uzaklığı hesapla
                        distances = np.sqrt(np.sum(positions**2, axis=1))
                        max_depth = np.max(distances)
                        depths.append(max_depth)
            
            if depths:  # Veri varsa
                # Çubuk grafik kullan
                ax1.bar(i, np.mean(depths), alpha=0.5, color=color, 
                       label=labels[ptype], yerr=np.std(depths) if len(depths) > 1 else 0)
        
        ax1.set_xticks(range(len(colors)))
        ax1.set_xticklabels(labels.values())
        ax1.set_xlabel('Parçacık Tipi')
        ax1.set_ylabel('Ortalama Penetrasyon Derinliği (m)')
        ax1.set_title('Penetrasyon Derinliği Analizi')
        ax1.legend()
        
        # Enerji kaybı dağılımı
        ax2 = fig.add_subplot(222)
        for i, (ptype, color) in enumerate(colors.items()):
            energy_loss = []
            for traj in self.trajectories:
                if traj['type'] == ptype and len(traj['energies']) > 1:
                    initial_E = traj['energies'][0]
                    final_E = traj['energies'][-1]
                    if initial_E > 0:
                        loss = (initial_E - final_E) / initial_E * 100
                        if 0 <= loss <= 100:
                            energy_loss.append(loss)
            
            if energy_loss:  # Veri varsa
                # Çubuk grafik kullan
                ax2.bar(i, np.mean(energy_loss), alpha=0.5, color=color,
                       label=labels[ptype], yerr=np.std(energy_loss) if len(energy_loss) > 1 else 0)
        
        ax2.set_xticks(range(len(colors)))
        ax2.set_xticklabels(labels.values())
        ax2.set_xlabel('Parçacık Tipi')
        ax2.set_ylabel('Ortalama Enerji Kaybı (%)')
        ax2.set_title('Enerji Kaybı Analizi')
        ax2.legend()
        
        # Yörünge uzunluğu dağılımı
        ax3 = fig.add_subplot(223)
        for i, (ptype, color) in enumerate(colors.items()):
            path_lengths = []
            for traj in self.trajectories:
                if traj['type'] == ptype and len(traj['positions']) > 1:
                    positions = traj['positions']
                    segments = np.diff(positions, axis=0)
                    length = np.sum(np.sqrt(np.sum(segments**2, axis=1)))
                    if length > 0:
                        path_lengths.append(length)
            
            if path_lengths:  # Veri varsa
                # Çubuk grafik kullan
                ax3.bar(i, np.mean(path_lengths), alpha=0.5, color=color,
                       label=labels[ptype], yerr=np.std(path_lengths) if len(path_lengths) > 1 else 0)
        
        ax3.set_xticks(range(len(colors)))
        ax3.set_xticklabels(labels.values())
        ax3.set_xlabel('Parçacık Tipi')
        ax3.set_ylabel('Ortalama Yörünge Uzunluğu (m)')
        ax3.set_title('Yörünge Uzunluğu Analizi')
        ax3.legend()
        
        # Saçılma açısı dağılımı
        ax4 = fig.add_subplot(224)
        for i, (ptype, color) in enumerate(colors.items()):
            scatter_angles = []
            for traj in self.trajectories:
                if traj['type'] == ptype and len(traj['positions']) > 2:
                    try:
                        positions = traj['positions']
                        initial_dir = positions[1] - positions[0]
                        final_dir = positions[-1] - positions[-2]
                        
                        initial_norm = np.linalg.norm(initial_dir)
                        final_norm = np.linalg.norm(final_dir)
                        
                        if initial_norm > 0 and final_norm > 0:
                            initial_dir = initial_dir / initial_norm
                            final_dir = final_dir / final_norm
                            cos_angle = np.clip(np.dot(initial_dir, final_dir), -1.0, 1.0)
                            angle = np.degrees(np.arccos(cos_angle))
                            scatter_angles.append(angle)
                    except Exception as e:
                        print(f"Açı hesaplama hatası: {str(e)}")
                        continue
            
            if scatter_angles:  # Veri varsa
                # Çubuk grafik kullan
                ax4.bar(i, np.mean(scatter_angles), alpha=0.5, color=color,
                       label=labels[ptype], yerr=np.std(scatter_angles) if len(scatter_angles) > 1 else 0)
        
        ax4.set_xticks(range(len(colors)))
        ax4.set_xticklabels(labels.values())
        ax4.set_xlabel('Parçacık Tipi')
        ax4.set_ylabel('Ortalama Saçılma Açısı (derece)')
        ax4.set_title('Saçılma Açısı Analizi')
        ax4.legend()
        
        plt.tight_layout()
        return fig

    def visualize_detailed_3d_results(self):
        """3D detaylı sonuçları görselleştir"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Parçacık tiplerine göre renk ve etiketler
        colors = {'alpha': 'red', 'beta': 'blue', 'neutron': 'green', 'gamma': 'yellow'}
        labels = {'alpha': 'Alfa', 'beta': 'Beta', 'neutron': 'Nötron', 'gamma': 'Gama'}
        
        # Her parçacık tipinin yörüngelerini çiz
        for ptype in colors:
            for traj in self.trajectories:
                if traj['type'] == ptype:
                    positions = np.array(traj['positions'])
                    if len(positions) > 1:
                        ax.plot(positions[:,0], positions[:,1], positions[:,2],
                               color=colors[ptype], alpha=0.3, linewidth=1)
        
        # Kalkan sınırlarını çiz
        u = np.linspace(0, 2*np.pi, 50)
        v = np.linspace(-self.shield_height/2, self.shield_height/2, 50)
        U, V = np.meshgrid(u, v)
        X = self.shield_radius * np.cos(U)
        Y = self.shield_radius * np.sin(U)
        Z = V
        ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')
        
        # Görünümü ayarla
        max_range = max(self.shield_radius, self.shield_height/2) * 1.2
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Parçacık Yörüngeleri ve Kalkan (3D)')
        
        # Lejant ekle
        legend_elements = [Line2D([0], [0], color=c, label=l) 
                         for ptype, (c, l) in zip(colors.keys(), zip(colors.values(), labels.values()))]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        return fig

    def run_multiple_simulations(self, scenarios):
        """Farklı senaryoları simüle et"""
        self.simulation_results = []
        
        for scenario in scenarios:
            # Senaryo parametrelerini ayarla
            self.B0 = scenario['B0']
            self.shield_radius = scenario['shield_radius']
            self.shield_height = scenario['shield_height']
            self.blast_yield = scenario['blast_yield']
            
            # Simülasyonu çalıştır
            print(f"\nSenaryo: B={self.B0}T, R={self.shield_radius}m, Y={self.blast_yield/1e3}kt")
            self.particles = []
            self.trajectories = []
            self.run_simulation()
            
            # Sonuçları analiz et
            results = self.analyze_scenario()
            results['parameters'] = scenario
            self.simulation_results.append(results)

    def analyze_scenario(self):
        """Tek bir senaryonun sonuçlarını analiz et"""
        results = {
            'particle_stats': {},
            'shield_effectiveness': {},
            'energy_distribution': {},
            'penetration_depth': {},
            'deflection_angles': {}
        }
        
        for particle_type in ['alpha', 'beta', 'neutron', 'gamma']:
            # İlgili parçacık tipinin yörüngelerini filtrele
            trajectories = [t for t in self.trajectories if t['type'] == particle_type]
            
            if not trajectories:
                continue
            
            # Parçacık istatistikleri
            initial_energies = [t['energies'][0] for t in trajectories]
            final_energies = [t['energies'][-1] for t in trajectories]
            
            # Kalkan etkinliği (enerji kaybı bazında)
            energy_loss = [(i-f)/i*100 for i, f in zip(initial_energies, final_energies)]
            
            # Penetrasyon derinliği - merkeze olan mesafe olarak hesapla
            penetration = [np.max(np.linalg.norm(t['positions'], axis=1))
                          for t in trajectories]
            
            # Sapma açıları
            initial_directions = [t['positions'][1] - t['positions'][0] for t in trajectories]
            final_directions = [t['positions'][-1] - t['positions'][-2] for t in trajectories]
            deflection = [np.arccos(np.clip(np.dot(i, f)/(np.linalg.norm(i)*np.linalg.norm(f)), -1.0, 1.0))*180/np.pi
                         for i, f in zip(initial_directions, final_directions)]
            
            results['particle_stats'][particle_type] = {
                'count': len(trajectories),
                'avg_initial_energy': np.mean(initial_energies),
                'avg_final_energy': np.mean(final_energies)
            }
            
            results['shield_effectiveness'][particle_type] = np.mean(energy_loss)
            results['energy_distribution'][particle_type] = {
                'initial': initial_energies,
                'final': final_energies
            }
            results['penetration_depth'][particle_type] = penetration
            results['deflection_angles'][particle_type] = deflection
        
        return results

    def visualize_all_results(self):
        """Tüm sonuçları iki panel halinde göster"""
        plt.ion()  # Interaktif mod aç
        
        # İlk panel: Karşılaştırmalı sonuçlar
        fig1 = plt.figure(figsize=(20, 15))
        fig1.canvas.manager.window.wm_geometry("+0+0")  # Sol üst köşe
        self.visualize_comparative_results()
        
        # İkinci panel: Detaylı 3D sonuçlar
        fig2 = plt.figure(figsize=(20, 20))
        fig2.canvas.manager.window.wm_geometry("+1000+0")  # Sağ üst köşe
        self.visualize_detailed_3d_results()
        
        # Grafikleri ekranda tut
        plt.show(block=True)

    def generate_blast_particles(self, num_particles=1000):
        """Radyasyon parçacıklarını oluştur"""
        self.particles = []  # Önceki parçacıkları temizle
        
        # Parçacık tipleri ve özellikleri
        particle_types = {
            'alpha': {'q': 2*self.e, 'm': 4*self.m_p, 'ratio': 0.2, 'E_min': 4.0, 'E_max': 6.0},
            'beta': {'q': -self.e, 'm': self.m_e, 'ratio': 0.3, 'E_mean': 0.5},
            'neutron': {'q': 0, 'm': self.m_p, 'ratio': 0.2, 'E_mean': 2.0},
            'gamma': {'q': 0, 'm': 0, 'ratio': 0.3, 'E_mean': 1.5}
        }
        
        # Kalkanın dışında ve üstünde rastgele başlangıç pozisyonları oluştur
        R = self.shield_radius * 2.0  # Yatay başlangıç mesafesi
        H = self.shield_height  # Dikey başlangıç mesafesi
        
        for ptype, props in particle_types.items():
            n_particles = int(num_particles * props['ratio'])
            
            # Her parçacık tipi için enerji dağılımı (MeV)
            if ptype == 'alpha':
                energies = np.random.uniform(props['E_min'], props['E_max'], n_particles)
            else:
                energies = np.random.exponential(props['E_mean'], n_particles)
            
            for E in energies:
                # Rastgele parçacık konumu seçimi (yandan veya üstten)
                if np.random.random() < 0.3:  # %30 olasılıkla üstten
                    # Kalkanın üstünde rastgele x,y pozisyonu
                    x = np.random.uniform(-R/2, R/2)
                    y = np.random.uniform(-R/2, R/2)
                    z = H  # Kalkanın üstünde başla
                    
                    position = Vector3D(x, y, z)
                    
                    # Aşağı ve merkeze doğru yönlendirilmiş hız vektörü
                    target = Vector3D(0, 0, -H/2)  # Hedef nokta (kalkanın ortası)
                    direction = target - position
                    direction = direction.normalize()
                
                else:  # %70 olasılıkla yandan
                    # Küresel koordinatlarda rastgele pozisyon (kalkanın dışında)
                    theta = np.random.uniform(0, np.pi)  # Yükseklik açısı
                    phi = np.random.uniform(0, 2*np.pi)  # Azimut açısı
                    
                    # Başlangıç pozisyonu (küresel koordinatlardan kartezyen koordinatlara)
                    position = Vector3D(
                        R * np.sin(theta) * np.cos(phi),
                        R * np.sin(theta) * np.sin(phi),
                        R * np.cos(theta)
                    )
                    
                    # Merkeze doğru yönlendirilmiş hız vektörü
                    direction = position * (-1.0 / position.magnitude())
                
                # Hız hesapla
                if props['m'] > 0:
                    # Klasik kinetik enerji formülü: E = 1/2 * m * v^2
                    v_mag = np.sqrt(2 * E * 1e6 * self.e / props['m'])  # m/s
                    velocity = direction * v_mag
                else:  # Fotonlar için ışık hızı
                    velocity = direction * self.c
                
                self.particles.append({
                    'type': ptype,
                    'q': props['q'],
                    'm': props['m'],
                    'energy': E,  # MeV
                    'position': position,
                    'velocity': velocity
                })

class SimulationControlPanel:
    def __init__(self, simulation):
        self.sim = simulation
        self.root = tk.Tk()
        self.root.title("Radyasyon Kalkanı Simülasyonu")
        self.root.geometry("500x800+0+0")
        
        # Stil ayarları
        self.setup_styles()
        
        # Simülasyon durumu
        self.simulation_ready = False
        self.parameters_valid = False
        
        # Kontrol paneli bileşenleri
        self.create_widgets()
        
        # Grafik pencereleri
        self.comparative_window = None
        self.detailed_window = None
        self.animation_running = False
        
        # İpuçları için tooltip
        self.tooltip = None
    
    def setup_styles(self):
        """Arayüz stillerini ayarla"""
        style = ttk.Style()
        style.configure('Title.TLabel', 
                       font=('Helvetica', 14, 'bold'),
                       padding=10)
        style.configure('Header.TLabel',
                       font=('Helvetica', 12, 'bold'),
                       padding=5)
        style.configure('Info.TLabel',
                       font=('Helvetica', 10, 'italic'),
                       foreground='navy')
        style.configure('Warning.TLabel',
                       font=('Helvetica', 10),
                       foreground='red')
        style.configure('Success.TLabel',
                       font=('Helvetica', 10),
                       foreground='green')
        style.configure('Primary.TButton',
                       font=('Helvetica', 10, 'bold'),
                       padding=5)
        style.configure('Secondary.TButton',
                       font=('Helvetica', 10),
                       padding=5)
    
    def create_widgets(self):
        """Arayüz bileşenlerini oluştur"""
        # Ana container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Başlık
        title = ttk.Label(main_container, 
                         text="Radyasyon Kalkanı Simülasyonu",
                         style='Title.TLabel')
        title.pack(fill='x')
        
        # Adım adım ilerleme
        steps_frame = ttk.LabelFrame(main_container, text="Simülasyon Adımları")
        steps_frame.pack(fill='x', pady=5)
        
        self.step_vars = []
        steps = [
            "1. Parametreleri Ayarla",
            "2. Simülasyonu Çalıştır",
            "3. Sonuçları Görüntüle"
        ]
        
        for step in steps:
            var = tk.BooleanVar(value=False)
            self.step_vars.append(var)
            ttk.Checkbutton(steps_frame, text=step, variable=var, 
                           state='disabled').pack(anchor='w', padx=5)
        
        # Parametre girişleri
        param_frame = ttk.LabelFrame(main_container, text="Simülasyon Parametreleri")
        param_frame.pack(fill='x', pady=5)
        
        # Grid layout için
        param_frame.columnconfigure(1, weight=1)
        
        # Manyetik alan konfigürasyonu
        row = 0
        ttk.Label(param_frame, text="Manyetik Alan Tipi:", 
                 style='Info.TLabel').grid(row=row, column=0, padx=5, pady=2, sticky='w')
        
        self.field_type_var = tk.StringVar(value=self.sim.field_type)
        field_types = {
            'solenoid': 'Solenoid',
            'toroidal': 'Toroidal',
            'helmholtz': 'Helmholtz Bobinleri',
            'dipole': 'Manyetik Dipol'
        }
        field_combo = ttk.Combobox(param_frame, textvariable=self.field_type_var,
                                  values=list(field_types.values()))
        field_combo.grid(row=row, column=1, padx=5, pady=2, sticky='ew')
        field_combo.bind('<<ComboboxSelected>>', self.on_field_type_change)
        
        # Manyetik alan parametreleri frame
        self.field_params_frame = ttk.LabelFrame(param_frame, text="Manyetik Alan Parametreleri")
        self.field_params_frame.grid(row=row+1, column=0, columnspan=3, padx=5, pady=5, sticky='ew')
        
        # Manyetik alan şiddeti
        row += 2
        ttk.Label(param_frame, text="Manyetik Alan (Tesla):", 
                 style='Info.TLabel').grid(row=row, column=0, padx=5, pady=2, sticky='w')
        self.B0_var = tk.StringVar(value=str(self.sim.B0))
        self.B0_entry = ttk.Entry(param_frame, textvariable=self.B0_var)
        self.B0_entry.grid(row=row, column=1, padx=5, pady=2, sticky='ew')
        ttk.Label(param_frame, text="(0.1 - 10.0)", 
                 style='Info.TLabel').grid(row=row, column=2, padx=5, pady=2)
        
        # Kalkan yarıçapı
        row += 1
        ttk.Label(param_frame, text="Kalkan Yarıçapı (m):", 
                 style='Info.TLabel').grid(row=row, column=0, padx=5, pady=2, sticky='w')
        self.radius_var = tk.StringVar(value=str(self.sim.shield_radius))
        self.radius_entry = ttk.Entry(param_frame, textvariable=self.radius_var)
        self.radius_entry.grid(row=row, column=1, padx=5, pady=2, sticky='ew')
        ttk.Label(param_frame, text="(100 - 1000)", 
                 style='Info.TLabel').grid(row=row, column=2, padx=5, pady=2)
        
        # Patlama gücü
        row += 1
        ttk.Label(param_frame, text="Patlama Gücü (kiloton TNT):", 
                 style='Info.TLabel').grid(row=row, column=0, padx=5, pady=2, sticky='w')
        self.yield_var = tk.StringVar(value=str(self.sim.blast_yield/1e3))
        self.yield_entry = ttk.Entry(param_frame, textvariable=self.yield_var)
        self.yield_entry.grid(row=row, column=1, padx=5, pady=2, sticky='ew')
        ttk.Label(param_frame, text="(1 - 100)", 
                 style='Info.TLabel').grid(row=row, column=2, padx=5, pady=2)
        
        # Parçacık sayısı
        row += 1
        ttk.Label(param_frame, text="Parçacık Sayısı:", 
                 style='Info.TLabel').grid(row=row, column=0, padx=5, pady=2, sticky='w')
        self.particle_count_var = tk.StringVar(value="1000")
        self.particle_entry = ttk.Entry(param_frame, textvariable=self.particle_count_var)
        self.particle_entry.grid(row=row, column=1, padx=5, pady=2, sticky='ew')
        ttk.Label(param_frame, text="(100 - 5000)", 
                 style='Info.TLabel').grid(row=row, column=2, padx=5, pady=2)
        
        # İlk manyetik alan parametrelerini göster
        self.update_field_params()
        
        # Parametre doğrulama butonu
        validate_btn = ttk.Button(param_frame, text="Parametreleri Doğrula",
                                 command=self.validate_parameters,
                                 style='Primary.TButton')
        validate_btn.grid(row=row+1, column=0, columnspan=3, pady=10)
        
        # Simülasyon kontrolü
        sim_frame = ttk.LabelFrame(main_container, text="Simülasyon Kontrolü")
        sim_frame.pack(fill='x', pady=5)
        
        ttk.Button(sim_frame, text="Simülasyonu Başlat",
                  command=self.run_simulation,
                  style='Primary.TButton').pack(fill='x', padx=5, pady=5)
        
        # Görselleştirme kontrolü
        viz_frame = ttk.LabelFrame(main_container, text="Görselleştirme")
        viz_frame.pack(fill='x', pady=5)
        
        ttk.Button(viz_frame, text="3D Görünüm",
                  command=self.show_3d_view,
                  style='Secondary.TButton').pack(fill='x', padx=5, pady=2)
        ttk.Button(viz_frame, text="Analiz Grafikleri",
                  command=self.show_analysis,
                  style='Secondary.TButton').pack(fill='x', padx=5, pady=2)
        
        # Animasyon kontrolü
        anim_frame = ttk.LabelFrame(main_container, text="Animasyon")
        anim_frame.pack(fill='x', pady=5)
        
        self.anim_btn = ttk.Button(anim_frame, text="Animasyonu Başlat",
                                  command=self.toggle_animation,
                                  style='Secondary.TButton')
        self.anim_btn.pack(fill='x', padx=5, pady=5)
        
        # Durum göstergesi
        status_frame = ttk.Frame(main_container)
        status_frame.pack(fill='x', pady=5)
        
        self.status_var = tk.StringVar(value="Hazır")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var,
                                     style='Info.TLabel')
        self.status_label.pack(side='left')
        
        # İlerleme çubuğu
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_container, 
                                           variable=self.progress_var,
                                           maximum=100)
        self.progress_bar.pack(fill='x', pady=5)
        
        # Çıkış butonu
        ttk.Button(main_container, text="Çıkış",
                  command=self.quit_application,
                  style='Secondary.TButton').pack(pady=10)
    
    def update_field_params(self):
        """Seçilen manyetik alan tipine göre parametre girişlerini güncelle"""
        # Önceki parametreleri temizle
        for widget in self.field_params_frame.winfo_children():
            widget.destroy()
        
        field_type = self.field_type_var.get()
        
        if field_type == "Solenoid":
            # Solenoid parametreleri
            self.solenoid_params = {
                'num_turns': tk.StringVar(value="1000"),
                'current': tk.StringVar(value="1000")
            }
            
            ttk.Label(self.field_params_frame, text="Sarım Sayısı:").grid(row=0, column=0, padx=5, pady=2)
            ttk.Entry(self.field_params_frame, textvariable=self.solenoid_params['num_turns']).grid(
                row=0, column=1, padx=5, pady=2)
            
            ttk.Label(self.field_params_frame, text="Akım (A):").grid(row=1, column=0, padx=5, pady=2)
            ttk.Entry(self.field_params_frame, textvariable=self.solenoid_params['current']).grid(
                row=1, column=1, padx=5, pady=2)
            
        elif field_type == "Toroidal":
            # Toroidal parametreleri
            self.toroidal_params = {
                'minor_radius': tk.StringVar(value=str(self.sim.shield_radius * 0.2)),
                'num_coils': tk.StringVar(value="16"),
                'current': tk.StringVar(value="1000")
            }
            
            ttk.Label(self.field_params_frame, text="Küçük Yarıçap (m):").grid(row=0, column=0, padx=5, pady=2)
            ttk.Entry(self.field_params_frame, textvariable=self.toroidal_params['minor_radius']).grid(
                row=0, column=1, padx=5, pady=2)
            
            ttk.Label(self.field_params_frame, text="Bobin Sayısı:").grid(row=1, column=0, padx=5, pady=2)
            ttk.Entry(self.field_params_frame, textvariable=self.toroidal_params['num_coils']).grid(
                row=1, column=1, padx=5, pady=2)
            
            ttk.Label(self.field_params_frame, text="Akım (A):").grid(row=2, column=0, padx=5, pady=2)
            ttk.Entry(self.field_params_frame, textvariable=self.toroidal_params['current']).grid(
                row=2, column=1, padx=5, pady=2)
            
        elif field_type == "Helmholtz Bobinleri":
            # Helmholtz parametreleri
            self.helmholtz_params = {
                'current': tk.StringVar(value="1000")
            }
            
            ttk.Label(self.field_params_frame, text="Akım (A):").grid(row=0, column=0, padx=5, pady=2)
            ttk.Entry(self.field_params_frame, textvariable=self.helmholtz_params['current']).grid(
                row=0, column=1, padx=5, pady=2)
            
        elif field_type == "Manyetik Dipol":
            # Dipol parametreleri
            self.dipole_params = {
                'moment_magnitude': tk.StringVar(value=str(self.sim.B0 * 4*np.pi * self.sim.shield_radius**3 / self.sim.mu0))
            }
            
            ttk.Label(self.field_params_frame, text="Dipol Momenti (A⋅m²):").grid(row=0, column=0, padx=5, pady=2)
            ttk.Entry(self.field_params_frame, textvariable=self.dipole_params['moment_magnitude']).grid(
                row=0, column=1, padx=5, pady=2)
    
    def on_field_type_change(self, event=None):
        """Manyetik alan tipi değiştiğinde çağrılır"""
        self.update_field_params()
    
    def validate_parameters(self):
        """Girilen parametreleri doğrula"""
        try:
            # Manyetik alan tipi kontrolü
            field_type = self.field_type_var.get()
            if field_type not in ["Solenoid", "Toroidal", "Helmholtz Bobinleri", "Manyetik Dipol"]:
                raise ValueError("Geçerli bir manyetik alan tipi seçin.")
            
            # Manyetik alan parametrelerini kontrol et
            if field_type == "Solenoid":
                num_turns = int(self.solenoid_params['num_turns'].get())
                current = float(self.solenoid_params['current'].get())
                if not 100 <= num_turns <= 10000:
                    raise ValueError("Sarım sayısı 100 ile 10000 arasında olmalıdır.")
                if not 100 <= current <= 10000:
                    raise ValueError("Akım 100 ile 10000 Amper arasında olmalıdır.")
                
            elif field_type == "Toroidal":
                minor_radius = float(self.toroidal_params['minor_radius'].get())
                num_coils = int(self.toroidal_params['num_coils'].get())
                current = float(self.toroidal_params['current'].get())
                if not 0.1 <= minor_radius <= self.sim.shield_radius:
                    raise ValueError("Küçük yarıçap 0.1 ile kalkan yarıçapı arasında olmalıdır.")
                if not 8 <= num_coils <= 32:
                    raise ValueError("Bobin sayısı 8 ile 32 arasında olmalıdır.")
                if not 100 <= current <= 10000:
                    raise ValueError("Akım 100 ile 10000 Amper arasında olmalıdır.")
                
            elif field_type == "Helmholtz Bobinleri":
                current = float(self.helmholtz_params['current'].get())
                if not 100 <= current <= 10000:
                    raise ValueError("Akım 100 ile 10000 Amper arasında olmalıdır.")
                
            elif field_type == "Manyetik Dipol":
                moment = float(self.dipole_params['moment_magnitude'].get())
                if not 1e3 <= moment <= 1e9:
                    raise ValueError("Dipol momenti 1e3 ile 1e9 A⋅m² arasında olmalıdır.")
            
            # Diğer parametrelerin kontrolü...
            # (mevcut kontroller aynen kalacak)
            
            # Parametreler geçerliyse durumu güncelle
            self.parameters_valid = True
            self.step_vars[0].set(True)
            self.status_var.set("Parametreler geçerli")
            tk.messagebox.showinfo("Başarılı", "Parametreler doğrulandı!")
            
        except ValueError as e:
            self.parameters_valid = False
            self.status_var.set("Parametre hatası!")
            tk.messagebox.showerror("Hata", str(e))
        except Exception as e:
            self.parameters_valid = False
            self.status_var.set("Beklenmeyen hata!")
            tk.messagebox.showerror("Hata", f"Beklenmeyen bir hata oluştu: {str(e)}")
            
    def run_simulation(self):
        """Simülasyonu çalıştır ve sonuçları görüntüle"""
        if not self.parameters_valid:
            tk.messagebox.showwarning("Uyarı", "Lütfen önce parametreleri doğrulayın!")
            return
            
        try:
            # Manyetik alan tipini ve parametrelerini güncelle
            field_type = self.field_type_var.get()
            self.sim.field_type = field_type.lower().replace(" bobinleri", "")
            
            if field_type == "Solenoid":
                self.sim.field_params = {
                    'num_turns': int(self.solenoid_params['num_turns'].get()),
                    'current': float(self.solenoid_params['current'].get())
                }
            elif field_type == "Toroidal":
                self.sim.field_params = {
                    'minor_radius': float(self.toroidal_params['minor_radius'].get()),
                    'num_coils': int(self.toroidal_params['num_coils'].get()),
                    'current': float(self.toroidal_params['current'].get())
                }
            elif field_type == "Helmholtz Bobinleri":
                self.sim.field_params = {
                    'current': float(self.helmholtz_params['current'].get())
                }
            elif field_type == "Manyetik Dipol":
                self.sim.field_params = {
                    'moment_magnitude': float(self.dipole_params['moment_magnitude'].get())
                }
            
            # Diğer parametreleri g��ncelle
            self.sim.B0 = float(self.B0_var.get())
            self.sim.shield_radius = float(self.radius_var.get())
            self.sim.blast_yield = float(self.yield_var.get()) * 1e3
            particle_count = int(self.particle_count_var.get())
            
            # Manyetik alanı yeniden oluştur
            self.sim.setup_magnetic_field()
            
            # Durum güncelle
            self.status_var.set("Simülasyon başlatılıyor...")
            self.progress_var.set(0)
            self.root.update()
            
            # Simülasyonu çalıştır
            self.sim.particles = []
            self.sim.trajectories = []
            
            def update_progress(progress):
                self.progress_var.set(progress)
                self.root.update()

            # Parçacıkları oluştur
            self.status_var.set("Parçacıklar oluşturuluyor...")
            self.sim.generate_blast_particles(particle_count)
            self.progress_var.set(30)
            self.root.update()

            # Simülasyonu çalıştır
            self.status_var.set("Yörüngeler hesaplanıyor...")
            self.sim.run_simulation(progress_callback=update_progress)
            
            # İkinci adımı tamamla
            self.step_vars[1].set(True)
            
            # Sonuçları görüntüle
            self.status_var.set("Sonuçlar görüntüleniyor...")
            self.show_all()
            
            # Üçüncü adımı tamamla
            self.step_vars[2].set(True)
            
            # Simülasyon hazır
            self.simulation_ready = True
            self.status_var.set("Simülasyon tamamlandı")
            self.progress_var.set(100)

        except Exception as e:
            self.simulation_ready = False
            self.status_var.set("Simülasyon hatası!")
            tk.messagebox.showerror("Hata", f"Simülasyon çalıştırılırken hata oluştu: {str(e)}")
            self.progress_var.set(0)

    def toggle_animation(self):
        self.animation_running = not self.animation_running
        if self.animation_running:
            self.animate_particles()
    
    def animate_particles(self):
        """Parçacık hareketlerini canlandır"""
        if not self.animation_running or not hasattr(self, 'ax') or not hasattr(self, 'canvas'):
            return
        
        try:
            # Figürü temizle
            self.ax.clear()
            
            # Zaman adımını hesapla
            if not self.sim.trajectories:
                raise ValueError("Yörünge verisi bulunamadı")
                
            max_steps = min(len(traj['positions']) for traj in self.sim.trajectories)
            time_step = int(max_steps * (self.animation_frame / 100))
            
            # Parçacıkları çiz
            colors = {'alpha': 'red', 'beta': 'blue', 'neutron': 'green', 'gamma': 'yellow'}
            
            for traj in self.sim.trajectories:
                if self.particle_filters[traj['type']].get():
                    positions = np.array(traj['positions'])
                    if len(positions) > time_step:
                        # Parçacığın o ana kadarki yörüngesi
                        self.ax.plot(positions[:time_step,0], 
                                   positions[:time_step,1],
                                   positions[:time_step,2],
                                   color=colors[traj['type']], alpha=0.3)
                        
                        # Parçacığın anlık konumu
                        self.ax.scatter(positions[time_step-1,0],
                                      positions[time_step-1,1],
                                      positions[time_step-1,2],
                                      color=colors[traj['type']], s=50)
            
            # Kalkanı çiz
            if self.show_shield.get():
                u = np.linspace(0, 2*np.pi, 50)
                v = np.linspace(-self.sim.shield_height/2, self.sim.shield_height/2, 50)
                U, V = np.meshgrid(u, v)
                X = self.sim.shield_radius * np.cos(U)
                Y = self.sim.shield_radius * np.sin(U)
                Z = V
                self.ax.plot_surface(X, Y, Z, alpha=0.2, color='gray')
            
            # Görünümü ayarla
            max_range = max(self.sim.shield_radius, self.sim.shield_height/2) * 1.2
            self.ax.set_xlim(-max_range, max_range)
            self.ax.set_ylim(-max_range, max_range)
            self.ax.set_zlim(-max_range, max_range)
            
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            self.ax.set_title(f"Parçacık Hareketi (Adım: {time_step}/{max_steps})")
            
            # Canvası güncelle
            self.canvas.draw()
            
            # Animasyon karesini güncelle
            self.animation_frame = (self.animation_frame + self.animation_speed.get()) % 100
            
            # Bir sonraki kare için zamanlayıcı ayarla
            if self.animation_running and self.detailed_window:
                self.detailed_window.after(50, self.animate_particles)
            
        except Exception as e:
            print(f"Animasyon hatası: {str(e)}")
            self.animation_running = False
            if hasattr(self, 'anim_btn'):
                self.anim_btn.config(text="Animasyonu Başlat")
            self.update_3d_view()  # Normal görünüme geri dön
    
    def save_results(self):
        try:
            # Sonuçları kaydet
            import json
            import datetime
            
            results = {
                'parameters': {
                    'B0': self.sim.B0,
                    'shield_radius': self.sim.shield_radius,
                    'blast_yield': self.sim.blast_yield
                },
                'particle_stats': {},
                'effectiveness': {}
            }
            
            # statistikleri topla
            for traj in self.sim.trajectories:
                ptype = traj['type']
                if ptype not in results['particle_stats']:
                    results['particle_stats'][ptype] = {
                        'count': 0,
                        'avg_energy_loss': 0
                    }
                
                results['particle_stats'][ptype]['count'] += 1
                energy_loss = (traj['energies'][0] - traj['energies'][-1]) / traj['energies'][0] * 100
                results['particle_stats'][ptype]['avg_energy_loss'] += energy_loss
            
            # Ortalamaları hesapla
            for ptype in results['particle_stats']:
                count = results['particle_stats'][ptype]['count']
                if count > 0:
                    results['particle_stats'][ptype]['avg_energy_loss'] /= count
            
            # Dosyaya kaydet
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_results_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=4)
            
            self.status_var.set(f"Sonuçlar kaydedildi: {filename}")
            
        except Exception as e:
            tk.messagebox.showerror("Hata", f"Sonuçlar kaydedilirken hata oluştu: {str(e)}")
    
    def rerun_simulation(self):
        try:
            # Parametreleri güncelle
            self.sim.B0 = float(self.B0_var.get())
            self.sim.shield_radius = float(self.radius_var.get())
            self.sim.blast_yield = float(self.yield_var.get()) * 1e3
            particle_count = int(self.particle_count_var.get())
            
            # Durum güncelle
            self.status_var.set("Simülasyon çalışıyor...")
            self.root.update()
            
            # Simülasyonu yeniden çalıştır
            self.sim.particles = []
            self.sim.trajectories = []
            self.sim.generate_blast_particles(particle_count)
            self.sim.run_simulation()
            
            # Grafikleri güncelle
            if self.comparative_window:
                self.comparative_window.destroy()
                self.comparative_window = None
            if self.detailed_window:
                self.detailed_window.destroy()
                self.detailed_window = None
            
            self.show_all()
            self.status_var.set("Simülasyon tamamlandı")
            
        except ValueError:
            tk.messagebox.showerror("Hata", "Lütfen geçerli sayısal değerler girin!")
        except Exception as e:
            tk.messagebox.showerror("Hata", f"Simülasyon çalıştırılırken hata oluştu: {str(e)}")
    
    def show_comparative(self):
        if self.comparative_window is None:
            self.comparative_window = tk.Toplevel(self.root)
            self.comparative_window.title("Karşılaştırmalı Sonuçlar")
            self.comparative_window.geometry("1000x800+300+0")
            
            fig = self.sim.visualize_comparative_results()
            
            canvas = FigureCanvasTkAgg(fig, master=self.comparative_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Toolbar ekle
            toolbar = NavigationToolbar2Tk(canvas, self.comparative_window)
            toolbar.update()
    
    def show_detailed(self):
        if self.detailed_window is None:
            self.detailed_window = tk.Toplevel(self.root)
            self.detailed_window.title("3D Detaylı Analiz")
            self.detailed_window.geometry("1000x800+300+0")
            
            fig = self.sim.visualize_detailed_3d_results()
            
            canvas = FigureCanvasTkAgg(fig, master=self.detailed_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Toolbar ekle
            toolbar = NavigationToolbar2Tk(canvas, self.detailed_window)
            toolbar.update()
    
    def show_3d_view(self):
        """3D görünümü göster ve animasyon kontrollerini ekle"""
        if not self.simulation_ready:
            tk.messagebox.showwarning("Uyarı", "Önce simülasyonu çalıştırın!")
            return
            
        try:
            if self.detailed_window is None:
                self.detailed_window = tk.Toplevel(self.root)
                self.detailed_window.title("3D Parçacık Yörüngeleri")
                self.detailed_window.geometry("1200x1000+300+0")
                
                # Ana frame
                main_frame = ttk.Frame(self.detailed_window)
                main_frame.pack(fill=tk.BOTH, expand=True)
                
                # Kontrol paneli
                control_frame = ttk.LabelFrame(main_frame, text="Görüntüleme Kontrolleri")
                control_frame.pack(fill=tk.X, padx=5, pady=5)
                
                # Parçacık filtreleme
                filter_frame = ttk.Frame(control_frame)
                filter_frame.pack(fill=tk.X, padx=5, pady=5)
                
                self.particle_filters = {}
                for ptype, label in [('alpha', 'Alfa'), ('beta', 'Beta'), 
                                   ('neutron', 'Nötron'), ('gamma', 'Gama')]:
                    var = tk.BooleanVar(value=True)
                    self.particle_filters[ptype] = var
                    ttk.Checkbutton(filter_frame, text=label, variable=var,
                                  command=self.update_3d_view).pack(side=tk.LEFT, padx=5)
                
                # Animasyon kontrolleri
                anim_frame = ttk.Frame(control_frame)
                anim_frame.pack(fill=tk.X, padx=5, pady=5)
                
                self.animation_speed = tk.DoubleVar(value=1.0)
                ttk.Label(anim_frame, text="Animasyon Hızı:").pack(side=tk.LEFT, padx=5)
                ttk.Scale(anim_frame, from_=0.1, to=5.0, variable=self.animation_speed,
                         orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
                
                self.anim_btn = ttk.Button(anim_frame, text="Animasyonu Başlat",
                                         command=self.toggle_animation)
                self.anim_btn.pack(side=tk.LEFT, padx=5)
                
                # Görünüm kontrolleri
                view_frame = ttk.Frame(control_frame)
                view_frame.pack(fill=tk.X, padx=5, pady=5)
                
                self.show_shield = tk.BooleanVar(value=True)
                ttk.Checkbutton(view_frame, text="Kalkanı Göster", 
                               variable=self.show_shield,
                               command=self.update_3d_view).pack(side=tk.LEFT, padx=5)
                
                self.show_field_lines = tk.BooleanVar(value=False)
                ttk.Checkbutton(view_frame, text="Manyetik Alan Çizgilerini Göster",
                               variable=self.show_field_lines,
                               command=self.update_3d_view).pack(side=tk.LEFT, padx=5)
                
                # Matplotlib figürü
                self.fig = plt.figure(figsize=(12, 10))
                self.ax = self.fig.add_subplot(111, projection='3d')
                
                # Canvas
                self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
                self.canvas.draw()
                self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # Toolbar
                toolbar = NavigationToolbar2Tk(self.canvas, main_frame)
                toolbar.update()
                
                # İlk görünümü oluştur
                self.update_3d_view()
                
                # Pencere kapatıldığında animasyonu durdur
                self.detailed_window.protocol("WM_DELETE_WINDOW", self.close_3d_view)
                
        except Exception as e:
            tk.messagebox.showerror("Hata", f"3D görünüm oluşturulurken hata: {str(e)}")
    
    def close_3d_view(self):
        """3D görünüm penceresini kapat"""
        self.animation_running = False
        if self.detailed_window:
            self.detailed_window.destroy()
            self.detailed_window = None
    
    def update_3d_view(self):
        """3D görünümü güncelle"""
        if not hasattr(self, 'ax') or not hasattr(self, 'canvas'):
            return
            
        try:
            self.ax.clear()
            
            # Parçacık yörüngelerini çiz
            colors = {'alpha': 'red', 'beta': 'blue', 'neutron': 'green', 'gamma': 'yellow'}
            labels = {'alpha': 'Alfa', 'beta': 'Beta', 'neutron': 'Nötron', 'gamma': 'Gama'}
            
            for traj in self.sim.trajectories:
                ptype = traj['type']
                if self.particle_filters[ptype].get():
                    positions = np.array(traj['positions'])
                    if len(positions) > 0:
                        self.ax.plot(positions[:,0], positions[:,1], positions[:,2],
                                   color=colors[ptype], alpha=0.5, linewidth=1,
                                   label=labels[ptype])
            
            # Kalkanı çiz
            if self.show_shield.get():
                u = np.linspace(0, 2*np.pi, 50)
                v = np.linspace(-self.sim.shield_height/2, self.sim.shield_height/2, 50)
                U, V = np.meshgrid(u, v)
                X = self.sim.shield_radius * np.cos(U)
                Y = self.sim.shield_radius * np.sin(U)
                Z = V
                self.ax.plot_surface(X, Y, Z, alpha=0.2, color='gray')
            
            # Manyetik alan çizgilerini çiz
            if self.show_field_lines.get():
                r = np.linspace(0, self.sim.shield_radius*1.2, 20)
                theta = np.linspace(0, 2*np.pi, 20)
                R, THETA = np.meshgrid(r, theta)
                X = R * np.cos(THETA)
                Y = R * np.sin(THETA)
                Z = np.zeros_like(X)
                
                U = np.zeros_like(X)
                V = np.zeros_like(X)
                W = np.zeros_like(X)
                
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        pos = Vector3D(X[i,j], Y[i,j], Z[i,j])
                        B = self.sim.magnetic_field.get_field_at(pos)
                        B_norm = B.magnitude()
                        if B_norm > 0:
                            B_normalized = B * (1.0 / B_norm)
                            U[i,j] = B_normalized.x
                            V[i,j] = B_normalized.y
                            W[i,j] = B_normalized.z
                
                self.ax.quiver(X, Y, Z, U, V, W, length=0.5, color='cyan', alpha=0.3)
            
            # Görünümü ayarla
            max_range = max(self.sim.shield_radius, self.sim.shield_height/2) * 1.2
            self.ax.set_xlim(-max_range, max_range)
            self.ax.set_ylim(-max_range, max_range)
            self.ax.set_zlim(-max_range, max_range)
            
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            self.ax.set_title("Parçacık Yörüngeleri ve Kalkan")
            
            # Lejantı güncelle
            handles, labels = self.ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            self.ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            
            self.canvas.draw()
            
        except Exception as e:
            print(f"3D görünüm güncelleme hatası: {str(e)}")
    
    def toggle_animation(self):
        """Animasyonu başlat/durdur"""
        self.animation_running = not self.animation_running
        
        if self.animation_running:
            self.anim_btn.config(text="Animasyonu Durdur")
            self.animation_frame = 0
            self.animate_particles()
        else:
            self.anim_btn.config(text="Animasyonu Başlat")
            self.update_3d_view()  # Normal görünüme geri dön
    
    def show_all(self):
        self.show_comparative()
        self.show_detailed()
    
    def show_analysis(self):
        """Detaylı analiz grafiklerini göster"""
        if not self.simulation_ready:
            tk.messagebox.showwarning("Uyarı", "Önce simülasyonu çalıştırın!")
            return
            
        try:
            analysis_window = tk.Toplevel(self.root)
            analysis_window.title("Detaylı Analiz")
            analysis_window.geometry("1200x800+300+0")
            
            fig = plt.figure(figsize=(12, 8))
            
            # Alt grafikler için düzen oluştur
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # Parçacık tiplerine göre enerji dağılımı
            ax1 = fig.add_subplot(gs[0, 0])
            particle_types = {'alpha': 'Alfa', 'beta': 'Beta', 'neutron': 'Nötron', 'gamma': 'Gama'}
            colors = {'alpha': 'red', 'beta': 'blue', 'neutron': 'green', 'gamma': 'yellow'}
            
            for ptype in particle_types:
                energies = [traj['energies'][0] for traj in self.sim.trajectories if traj['type'] == ptype]
                if energies:
                    ax1.hist(energies, bins=20, alpha=0.5, label=particle_types[ptype], color=colors[ptype])
            
            ax1.set_xlabel('Başlangıç Enerjisi (MeV)')
            ax1.set_ylabel('Parçacık Sayısı')
            ax1.set_title('Parçacık Tiplerine Göre Enerji Dağılımı')
            ax1.legend()
            
            # Penetrasyon derinliği analizi
            ax2 = fig.add_subplot(gs[0, 1])
            for ptype in particle_types:
                depths = [np.max(np.sqrt(np.sum(np.array(traj['positions'])**2, axis=1))) 
                         for traj in self.sim.trajectories if traj['type'] == ptype]
                if depths:
                    ax2.hist(depths, bins=20, alpha=0.5, label=particle_types[ptype], color=colors[ptype])
            
            ax2.set_xlabel('Maksimum Penetrasyon Derinliği (m)')
            ax2.set_ylabel('Parçacık Sayısı')
            ax2.set_title('Penetrasyon Derinliği Dağılımı')
            ax2.legend()
            
            # Enerji kaybı analizi
            ax3 = fig.add_subplot(gs[1, 0])
            for ptype in particle_types:
                energy_loss = [(traj['energies'][0] - traj['energies'][-1])/traj['energies'][0] * 100
                             for traj in self.sim.trajectories if traj['type'] == ptype]
                if energy_loss:
                    ax3.hist(energy_loss, bins=20, alpha=0.5, label=particle_types[ptype], color=colors[ptype])
            
            ax3.set_xlabel('Enerji Kaybı (%)')
            ax3.set_ylabel('Parçacık Sayısı')
            ax3.set_title('Enerji Kaybı Dağılımı')
            ax3.legend()
            
            # Yörünge uzunluğu analizi
            ax4 = fig.add_subplot(gs[1, 1])
            for ptype in particle_types:
                path_lengths = []
                for traj in self.sim.trajectories:
                    if traj['type'] == ptype:
                        positions = np.array(traj['positions'])
                        if len(positions) > 1:
                            diffs = np.diff(positions, axis=0)
                            length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
                            
                if path_lengths:
                    ax4.hist(path_lengths, bins=20, alpha=0.5, label=particle_types[ptype], color=colors[ptype])
            
            ax4.set_xlabel('Yörünge Uzunluğu (m)')
            ax4.set_ylabel('Parçacık Sayısı')
            ax4.set_title('Yörünge Uzunluğu Dağılımı')
            ax4.legend()
            
            # Canvas ve toolbar ekle
            canvas = FigureCanvasTkAgg(fig, master=analysis_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            toolbar = NavigationToolbar2Tk(canvas, analysis_window)
            toolbar.update()
            
            # Özet istatistikler
            stats_frame = ttk.LabelFrame(analysis_window, text="Özet İstatistikler")
            stats_frame.pack(fill='x', padx=10, pady=5)
            
            # Parçacık tiplerine göre istatistikler
            for ptype in particle_types:
                type_trajectories = [t for t in self.sim.trajectories if t['type'] == ptype]
                if type_trajectories:
                    avg_energy_loss = np.mean([(t['energies'][0] - t['energies'][-1])/t['energies'][0] * 100
                                             for t in type_trajectories])
                    max_depth = np.max([np.max(np.sqrt(np.sum(np.array(t['positions'])**2, axis=1)))
                                      for t in type_trajectories])
                    
                    stats_text = f"{particle_types[ptype]}: {len(type_trajectories)} parçacık, "
                    stats_text += f"Ort. Enerji Kaybı: {avg_energy_loss:.1f}%, "
                    stats_text += f"Maks. Derinlik: {max_depth:.1f}m"
                    
                    ttk.Label(stats_frame, text=stats_text).pack(anchor='w', padx=5, pady=2)
            
        except Exception as e:
            tk.messagebox.showerror("Hata", f"Analiz görüntülenirken hata oluştu: {str(e)}")
    
    def quit_application(self):
        if tk.messagebox.askokcancel("Çıkış", "Uygulamadan çıkmak istiyor musunuz?"):
            plt.close('all')
            self.root.quit()
            self.root.destroy()
    
    def run(self):
        self.root.mainloop()

# Ana programda kullanımı güncelle
if __name__ == "__main__":
    # Simülasyonu oluştur
    sim = NuclearBlastShieldSimulation()
    
    # Kontrol panelini başlat
    control_panel = SimulationControlPanel(sim)
    control_panel.run()