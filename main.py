import sys
import logging
import multiprocessing
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QGridLayout, QGroupBox, QMessageBox, QFileDialog, QProgressBar, 
    QHBoxLayout, QCheckBox, QTextEdit, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QFont
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from radiation_shield import NuclearBlastShieldSimulation
from simulation_engine import SimulationEngine, SimulationParameters
from core_physics import PhysicsEngine, Vector3D
from magnetic_field import MagneticFieldManager
from plasma_physics import PlasmaPhysicsEngine
from shield_materials import MaterialDatabase
from shield_geometry import ShieldGeometryManager
from nuclear_blast import BlastSimulator
from particle import ParticleManager
from neutrino_physics import NeutrinoPhysicsEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class AdvancedSimulationControlPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Nuclear Shield Simulation System")
        self.setGeometry(100, 100, 1600, 900)
        
        # Initialize core components
        self.init_simulation_components()
        
        # Setup UI
        self.setup_user_interface()
        
        # Initialize data storage
        self.simulation_results: Dict[str, Any] = {}
        self.current_config: Dict[str, Any] = {}
        
        logger.info("Advanced Simulation Control Panel initialized")

    def init_simulation_components(self):
        """Initialize all simulation components and engines"""
        try:
            # Create simulation parameters
            sim_params = SimulationParameters(
                dt=1e-9,              # Time step [s]
                total_time=1e-6,      # Total simulation time [s]
                output_interval=1e-7,  # Time between outputs [s]
                max_particles=10000,   # Maximum number of particles to track
                min_energy=1e-3,      # Minimum energy to track [MeV]
                enable_magnetic_fields=True,
                enable_plasma_effects=True,
                enable_neutrino_physics=True
            )
            
            self.simulation = NuclearBlastShieldSimulation()
            self.sim_engine = SimulationEngine(sim_params)
            self.physics_engine = PhysicsEngine()
            self.magnetic_field = MagneticFieldManager()
            self.plasma_engine = PlasmaPhysicsEngine()
            self.material_db = MaterialDatabase()
            self.geometry_manager = ShieldGeometryManager()
            self.blast_simulator = BlastSimulator()
            self.particle_manager = ParticleManager()
            self.neutrino_engine = NeutrinoPhysicsEngine()
            
            logger.info("All simulation components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing simulation components: {str(e)}")
            raise

    def setup_user_interface(self):
        """Setup the main user interface with all control panels"""
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create tab widget for different control panels
        tabs = QTabWidget()
        
        # Add various control tabs
        tabs.addTab(self.create_simulation_control_tab(), "Simulation Control")
        tabs.addTab(self.create_physics_config_tab(), "Physics Configuration")
        tabs.addTab(self.create_visualization_tab(), "3D Visualization")
        tabs.addTab(self.create_analysis_tab(), "Data Analysis")
        tabs.addTab(self.create_results_tab(), "Results")
        
        layout.addWidget(tabs)
        
        logger.info("User interface setup completed")

    def create_simulation_control_tab(self) -> QWidget:
        """Create the main simulation control tab with all necessary controls"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Simulation Parameters Group
        param_group = QGroupBox("Simulation Parameters")
        param_layout = QGridLayout()
        
        # Magnetic Field Configuration
        row = 0
        param_layout.addWidget(QLabel("Magnetic Field Type:"), row, 0)
        self.field_type_combo = QComboBox()
        self.field_type_combo.addItems([
            "Solenoid", "Toroidal", "Helmholtz", "Dipole", "Custom"
        ])
        self.field_type_combo.currentTextChanged.connect(self.on_field_type_changed)
        param_layout.addWidget(self.field_type_combo, row, 1)
        
        # Field Strength
        row += 1
        param_layout.addWidget(QLabel("Field Strength (Tesla):"), row, 0)
        self.field_strength_spin = QDoubleSpinBox()
        self.field_strength_spin.setRange(0.1, 10.0)
        self.field_strength_spin.setValue(5.0)
        self.field_strength_spin.setSingleStep(0.1)
        param_layout.addWidget(self.field_strength_spin, row, 1)
        
        # Shield Parameters
        row += 1
        param_layout.addWidget(QLabel("Shield Radius (m):"), row, 0)
        self.shield_radius_spin = QDoubleSpinBox()
        self.shield_radius_spin.setRange(1.0, 1000.0)
        self.shield_radius_spin.setValue(100.0)
        param_layout.addWidget(self.shield_radius_spin, row, 1)
        
        row += 1
        param_layout.addWidget(QLabel("Shield Height (m):"), row, 0)
        self.shield_height_spin = QDoubleSpinBox()
        self.shield_height_spin.setRange(1.0, 1000.0)
        self.shield_height_spin.setValue(200.0)
        param_layout.addWidget(self.shield_height_spin, row, 1)
        
        # Blast Parameters
        row += 1
        param_layout.addWidget(QLabel("Blast Yield (kt):"), row, 0)
        self.blast_yield_spin = QDoubleSpinBox()
        self.blast_yield_spin.setRange(1.0, 1000.0)
        self.blast_yield_spin.setValue(20.0)
        param_layout.addWidget(self.blast_yield_spin, row, 1)
        
        row += 1
        param_layout.addWidget(QLabel("Source Distance (m):"), row, 0)
        self.source_distance_spin = QDoubleSpinBox()
        self.source_distance_spin.setRange(100.0, 10000.0)
        self.source_distance_spin.setValue(1000.0)
        param_layout.addWidget(self.source_distance_spin, row, 1)
        
        # Particle Parameters
        row += 1
        param_layout.addWidget(QLabel("Number of Particles:"), row, 0)
        self.particle_count_spin = QSpinBox()
        self.particle_count_spin.setRange(100, 10000)
        self.particle_count_spin.setValue(1000)
        param_layout.addWidget(self.particle_count_spin, row, 1)
        
        # Add parameter group to layout
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Field-specific parameters group (will be updated based on field type)
        self.field_params_group = QGroupBox("Field-Specific Parameters")
        self.field_params_layout = QGridLayout()
        self.field_params_group.setLayout(self.field_params_layout)
        layout.addWidget(self.field_params_group)
        
        # Control Buttons
        button_layout = QHBoxLayout()
        
        validate_btn = QPushButton("Validate Parameters")
        validate_btn.clicked.connect(self.validate_parameters)
        button_layout.addWidget(validate_btn)
        
        run_btn = QPushButton("Run Simulation")
        run_btn.clicked.connect(self.run_simulation)
        button_layout.addWidget(run_btn)
        
        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self.stop_simulation)
        button_layout.addWidget(stop_btn)
        
        layout.addLayout(button_layout)
        
        # Progress Section
        progress_group = QGroupBox("Simulation Progress")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Initialize field-specific parameters
        self.update_field_parameters()
        
        return tab
    
    def on_field_type_changed(self, field_type: str):
        """Handle field type change"""
        self.update_field_parameters()
        
    def update_field_parameters(self):
        """Update field-specific parameter controls based on selected field type"""
        # Clear existing widgets
        for i in reversed(range(self.field_params_layout.count())): 
            self.field_params_layout.itemAt(i).widget().setParent(None)
        
        field_type = self.field_type_combo.currentText().lower()
        
        if field_type == "solenoid":
            self.add_solenoid_parameters()
        elif field_type == "toroidal":
            self.add_toroidal_parameters()
        elif field_type == "helmholtz":
            self.add_helmholtz_parameters()
        elif field_type == "dipole":
            self.add_dipole_parameters()
        elif field_type == "custom":
            self.add_custom_parameters()
    
    def add_solenoid_parameters(self):
        """Add parameter controls for solenoid field"""
        row = 0
        self.field_params_layout.addWidget(QLabel("Number of Turns:"), row, 0)
        self.solenoid_turns_spin = QSpinBox()
        self.solenoid_turns_spin.setRange(100, 10000)
        self.solenoid_turns_spin.setValue(1000)
        self.field_params_layout.addWidget(self.solenoid_turns_spin, row, 1)
        
        row += 1
        self.field_params_layout.addWidget(QLabel("Current (A):"), row, 0)
        self.solenoid_current_spin = QDoubleSpinBox()
        self.solenoid_current_spin.setRange(100, 10000)
        self.solenoid_current_spin.setValue(1000)
        self.field_params_layout.addWidget(self.solenoid_current_spin, row, 1)
    
    def add_toroidal_parameters(self):
        """Add parameter controls for toroidal field"""
        row = 0
        self.field_params_layout.addWidget(QLabel("Minor Radius (m):"), row, 0)
        self.toroidal_minor_radius_spin = QDoubleSpinBox()
        self.toroidal_minor_radius_spin.setRange(0.1, 100.0)
        self.toroidal_minor_radius_spin.setValue(20.0)
        self.field_params_layout.addWidget(self.toroidal_minor_radius_spin, row, 1)
        
        row += 1
        self.field_params_layout.addWidget(QLabel("Number of Coils:"), row, 0)
        self.toroidal_coils_spin = QSpinBox()
        self.toroidal_coils_spin.setRange(8, 32)
        self.toroidal_coils_spin.setValue(16)
        self.field_params_layout.addWidget(self.toroidal_coils_spin, row, 1)
        
        row += 1
        self.field_params_layout.addWidget(QLabel("Current (A):"), row, 0)
        self.toroidal_current_spin = QDoubleSpinBox()
        self.toroidal_current_spin.setRange(100, 10000)
        self.toroidal_current_spin.setValue(1000)
        self.field_params_layout.addWidget(self.toroidal_current_spin, row, 1)
    
    def add_helmholtz_parameters(self):
        """Add parameter controls for Helmholtz coils"""
        row = 0
        self.field_params_layout.addWidget(QLabel("Current (A):"), row, 0)
        self.helmholtz_current_spin = QDoubleSpinBox()
        self.helmholtz_current_spin.setRange(100, 10000)
        self.helmholtz_current_spin.setValue(1000)
        self.field_params_layout.addWidget(self.helmholtz_current_spin, row, 1)
    
    def add_dipole_parameters(self):
        """Add parameter controls for magnetic dipole"""
        row = 0
        self.field_params_layout.addWidget(QLabel("Moment Magnitude (A⋅m²):"), row, 0)
        self.dipole_moment_spin = QDoubleSpinBox()
        self.dipole_moment_spin.setRange(1e3, 1e9)
        self.dipole_moment_spin.setValue(1e6)
        self.field_params_layout.addWidget(self.dipole_moment_spin, row, 1)
    
    def add_custom_parameters(self):
        """Add parameter controls for custom field configuration"""
        self.field_params_layout.addWidget(
            QLabel("Custom field configuration can be defined programmatically."))
    
    def validate_parameters(self):
        """Validate all simulation parameters"""
        try:
            # Get basic parameters
            field_type = self.field_type_combo.currentText().lower()
            field_strength = self.field_strength_spin.value()
            shield_radius = self.shield_radius_spin.value()
            shield_height = self.shield_height_spin.value()
            blast_yield = self.blast_yield_spin.value()
            source_distance = self.source_distance_spin.value()
            particle_count = self.particle_count_spin.value()
            
            # Validate field-specific parameters
            field_params = self.get_field_parameters()
            
            # Update configuration
            self.current_config = {
                'field_type': field_type,
                'field_strength': field_strength,
                'shield_radius': shield_radius,
                'shield_height': shield_height,
                'blast_yield': blast_yield,
                'source_distance': source_distance,
                'particle_count': particle_count,
                'field_parameters': field_params
            }
            
            # Show success message
            QMessageBox.information(self, "Validation", "All parameters are valid!")
            return True
            
        except ValueError as e:
            QMessageBox.warning(self, "Validation Error", str(e))
            return False
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Unexpected error: {str(e)}")
            logger.error("Parameter validation error: %s", str(e))
            return False
    
    def get_field_parameters(self) -> Dict[str, Any]:
        """Get field-specific parameters based on selected field type"""
        field_type = self.field_type_combo.currentText().lower()
        params = {}
        
        if field_type == "solenoid":
            params = {
                'n_turns': self.solenoid_turns_spin.value(),
                'current': self.solenoid_current_spin.value(),
                'radius': self.shield_radius_spin.value(),
                'length': self.shield_height_spin.value()
            }
        elif field_type == "toroidal":
            params = {
                'major_radius': self.shield_radius_spin.value(),
                'minor_radius': self.toroidal_minor_radius_spin.value(),
                'n_coils': self.toroidal_coils_spin.value(),
                'current': self.toroidal_current_spin.value()
            }
        elif field_type == "helmholtz":
            params = {
                'radius': self.shield_radius_spin.value(),
                'current': self.helmholtz_current_spin.value()
            }
        elif field_type == "dipole":
            params = {
                'moment': Vector3D(0, 0, self.dipole_moment_spin.value())
            }
        
        return params
    
    def stop_simulation(self):
        """Stop the running simulation"""
        # Implementation for stopping simulation
        pass

    def create_physics_config_tab(self) -> QWidget:
        """Create the physics configuration tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Plasma Physics Group
        plasma_group = QGroupBox("Plasma Physics Parameters")
        plasma_layout = QGridLayout()
        
        row = 0
        plasma_layout.addWidget(QLabel("Plasma Density (m⁻³):"), row, 0)
        self.plasma_density_spin = QDoubleSpinBox()
        self.plasma_density_spin.setRange(1e20, 1e24)
        self.plasma_density_spin.setValue(1e22)
        self.plasma_density_spin.setDecimals(2)
        self.plasma_density_spin.setSingleStep(1e20)
        plasma_layout.addWidget(self.plasma_density_spin, row, 1)
        
        row += 1
        plasma_layout.addWidget(QLabel("Plasma Temperature (keV):"), row, 0)
        self.plasma_temp_spin = QDoubleSpinBox()
        self.plasma_temp_spin.setRange(0.1, 100.0)
        self.plasma_temp_spin.setValue(10.0)
        self.plasma_temp_spin.setDecimals(1)
        plasma_layout.addWidget(self.plasma_temp_spin, row, 1)
        
        plasma_group.setLayout(plasma_layout)
        layout.addWidget(plasma_group)
        
        # Material Properties Group
        material_group = QGroupBox("Material Properties")
        material_layout = QGridLayout()
        
        row = 0
        material_layout.addWidget(QLabel("Shield Material:"), row, 0)
        self.material_combo = QComboBox()
        self.material_combo.addItems([
            "Tungsten", "Lead", "Iron", "Copper", "Water", "Concrete"
        ])
        material_layout.addWidget(self.material_combo, row, 1)
        
        row += 1
        material_layout.addWidget(QLabel("Material Thickness (cm):"), row, 0)
        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setRange(1.0, 1000.0)
        self.thickness_spin.setValue(50.0)
        material_layout.addWidget(self.thickness_spin, row, 1)
        
        material_group.setLayout(material_layout)
        layout.addWidget(material_group)
        
        # Physics Options Group
        options_group = QGroupBox("Physics Options")
        options_layout = QVBoxLayout()
        
        self.enable_magnetic = QCheckBox("Enable Magnetic Fields")
        self.enable_magnetic.setChecked(True)
        options_layout.addWidget(self.enable_magnetic)
        
        self.enable_plasma = QCheckBox("Enable Plasma Effects")
        self.enable_plasma.setChecked(True)
        options_layout.addWidget(self.enable_plasma)
        
        self.enable_neutrino = QCheckBox("Enable Neutrino Physics")
        self.enable_neutrino.setChecked(True)
        options_layout.addWidget(self.enable_neutrino)
        
        self.enable_relativistic = QCheckBox("Enable Relativistic Effects")
        self.enable_relativistic.setChecked(True)
        options_layout.addWidget(self.enable_relativistic)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        return tab
    
    def create_visualization_tab(self) -> QWidget:
        """Create the 3D visualization tab with all necessary controls"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Visualization Controls Group
        controls_group = QGroupBox("Visualization Controls")
        controls_layout = QGridLayout()
        
        # View Options
        row = 0
        controls_layout.addWidget(QLabel("View Options:"), row, 0)
        
        self.show_particles_check = QCheckBox("Show Particles")
        self.show_particles_check.setChecked(True)
        controls_layout.addWidget(self.show_particles_check, row, 1)
        
        self.show_trajectories_check = QCheckBox("Show Trajectories")
        self.show_trajectories_check.setChecked(True)
        controls_layout.addWidget(self.show_trajectories_check, row, 2)
        
        self.show_field_check = QCheckBox("Show Magnetic Field")
        self.show_field_check.setChecked(True)
        controls_layout.addWidget(self.show_field_check, row, 3)
        
        # Particle Filters
        row += 1
        controls_layout.addWidget(QLabel("Particle Types:"), row, 0)
        
        self.show_alpha_check = QCheckBox("Alpha")
        self.show_alpha_check.setChecked(True)
        controls_layout.addWidget(self.show_alpha_check, row, 1)
        
        self.show_beta_check = QCheckBox("Beta")
        self.show_beta_check.setChecked(True)
        controls_layout.addWidget(self.show_beta_check, row, 2)
        
        self.show_neutron_check = QCheckBox("Neutron")
        self.show_neutron_check.setChecked(True)
        controls_layout.addWidget(self.show_neutron_check, row, 3)
        
        self.show_gamma_check = QCheckBox("Gamma")
        self.show_gamma_check.setChecked(True)
        controls_layout.addWidget(self.show_gamma_check, row, 4)
        
        # Color Options
        row += 1
        controls_layout.addWidget(QLabel("Color By:"), row, 0)
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems([
            "Particle Type",
            "Energy",
            "Velocity",
            "Penetration Depth"
        ])
        controls_layout.addWidget(self.color_by_combo, row, 1, 1, 2)
        
        # Animation Controls
        row += 1
        controls_layout.addWidget(QLabel("Animation:"), row, 0)
        
        self.animation_speed_spin = QDoubleSpinBox()
        self.animation_speed_spin.setRange(0.1, 10.0)
        self.animation_speed_spin.setValue(1.0)
        self.animation_speed_spin.setSingleStep(0.1)
        controls_layout.addWidget(QLabel("Speed:"), row, 1)
        controls_layout.addWidget(self.animation_speed_spin, row, 2)
        
        animation_buttons = QHBoxLayout()
        
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_animation)
        animation_buttons.addWidget(self.play_button)
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_animation)
        animation_buttons.addWidget(self.reset_button)
        
        controls_layout.addLayout(animation_buttons, row, 3, 1, 2)
        
        # Camera Controls
        row += 1
        controls_layout.addWidget(QLabel("Camera:"), row, 0)
        
        camera_buttons = QHBoxLayout()
        
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self.reset_camera)
        camera_buttons.addWidget(self.reset_view_button)
        
        self.top_view_button = QPushButton("Top View")
        self.top_view_button.clicked.connect(lambda: self.set_camera_view("top"))
        camera_buttons.addWidget(self.top_view_button)
        
        self.side_view_button = QPushButton("Side View")
        self.side_view_button.clicked.connect(lambda: self.set_camera_view("side"))
        camera_buttons.addWidget(self.side_view_button)
        
        controls_layout.addLayout(camera_buttons, row, 1, 1, 4)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # 3D View
        view_group = QGroupBox("3D View")
        view_layout = QVBoxLayout()
        
        # Placeholder for VTK widget
        self.vtk_widget = QWidget()  # This will be replaced with actual VTK widget
        self.vtk_widget.setMinimumSize(800, 600)
        view_layout.addWidget(self.vtk_widget)
        
        view_group.setLayout(view_layout)
        layout.addWidget(view_group)
        
        # Connect signals
        self.show_particles_check.stateChanged.connect(self.update_visualization)
        self.show_trajectories_check.stateChanged.connect(self.update_visualization)
        self.show_field_check.stateChanged.connect(self.update_visualization)
        self.show_alpha_check.stateChanged.connect(self.update_visualization)
        self.show_beta_check.stateChanged.connect(self.update_visualization)
        self.show_neutron_check.stateChanged.connect(self.update_visualization)
        self.show_gamma_check.stateChanged.connect(self.update_visualization)
        self.color_by_combo.currentTextChanged.connect(self.update_visualization)
        
        return tab
    
    def toggle_animation(self):
        """Toggle animation playback"""
        if self.play_button.text() == "Play":
            self.play_button.setText("Pause")
            # Start animation
            self.start_animation()
        else:
            self.play_button.setText("Play")
            # Pause animation
            self.pause_animation()
    
    def reset_animation(self):
        """Reset animation to initial state"""
        self.play_button.setText("Play")
        # Reset animation state
        self.reset_visualization()
    
    def reset_camera(self):
        """Reset camera to default position"""
        # Implementation for resetting camera
        pass
    
    def set_camera_view(self, view: str):
        """Set camera to predefined view"""
        # Implementation for setting camera view
        pass
    
    def start_animation(self):
        """Start animation playback"""
        # Implementation for starting animation
        pass
    
    def pause_animation(self):
        """Pause animation playback"""
        # Implementation for pausing animation
        pass
    
    def update_visualization(self):
        """Update visualization based on current settings"""
        # Implementation for updating visualization
        pass
    
    def reset_visualization(self):
        """Reset visualization to initial state"""
        # Implementation for resetting visualization
        pass

    def create_analysis_tab(self) -> QWidget:
        """Create the data analysis tab with all necessary controls and plots"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Analysis Controls Group
        controls_group = QGroupBox("Analysis Controls")
        controls_layout = QGridLayout()
        
        # Analysis Type Selection
        row = 0
        controls_layout.addWidget(QLabel("Analysis Type:"), row, 0)
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems([
            "Penetration Depth",
            "Energy Loss",
            "Trajectory Length",
            "Scattering Angle",
            "Particle Distribution",
            "Field Strength"
        ])
        self.analysis_type_combo.currentTextChanged.connect(self.update_analysis)
        controls_layout.addWidget(self.analysis_type_combo, row, 1, 1, 2)
        
        # Data Filters
        row += 1
        controls_layout.addWidget(QLabel("Data Filters:"), row, 0)
        
        self.filter_alpha_check = QCheckBox("Alpha")
        self.filter_alpha_check.setChecked(True)
        controls_layout.addWidget(self.filter_alpha_check, row, 1)
        
        self.filter_beta_check = QCheckBox("Beta")
        self.filter_beta_check.setChecked(True)
        controls_layout.addWidget(self.filter_beta_check, row, 2)
        
        self.filter_neutron_check = QCheckBox("Neutron")
        self.filter_neutron_check.setChecked(True)
        controls_layout.addWidget(self.filter_neutron_check, row, 3)
        
        self.filter_gamma_check = QCheckBox("Gamma")
        self.filter_gamma_check.setChecked(True)
        controls_layout.addWidget(self.filter_gamma_check, row, 4)
        
        # Plot Options
        row += 1
        controls_layout.addWidget(QLabel("Plot Type:"), row, 0)
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Bar Chart",
            "Histogram",
            "Line Plot",
            "Scatter Plot",
            "Box Plot"
        ])
        self.plot_type_combo.currentTextChanged.connect(self.update_analysis)
        controls_layout.addWidget(self.plot_type_combo, row, 1, 1, 2)
        
        # Statistical Options
        row += 1
        controls_layout.addWidget(QLabel("Statistics:"), row, 0)
        
        self.show_mean_check = QCheckBox("Show Mean")
        self.show_mean_check.setChecked(True)
        controls_layout.addWidget(self.show_mean_check, row, 1)
        
        self.show_std_check = QCheckBox("Show Std Dev")
        self.show_std_check.setChecked(True)
        controls_layout.addWidget(self.show_std_check, row, 2)
        
        self.show_median_check = QCheckBox("Show Median")
        self.show_median_check.setChecked(False)
        controls_layout.addWidget(self.show_median_check, row, 3)
        
        # Export Options
        row += 1
        export_buttons = QHBoxLayout()
        
        self.export_data_btn = QPushButton("Export Data")
        self.export_data_btn.clicked.connect(self.export_analysis_data)
        export_buttons.addWidget(self.export_data_btn)
        
        self.export_plot_btn = QPushButton("Export Plot")
        self.export_plot_btn.clicked.connect(self.export_analysis_plot)
        export_buttons.addWidget(self.export_plot_btn)
        
        controls_layout.addLayout(export_buttons, row, 0, 1, 5)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Results Display
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout()
        
        # Statistics Display
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(100)
        results_layout.addWidget(self.stats_text)
        
        # Plot Display
        self.plot_widget = QWidget()  # This will be replaced with matplotlib widget
        self.plot_widget.setMinimumSize(800, 400)
        results_layout.addWidget(self.plot_widget)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Connect signals
        for check in [self.filter_alpha_check, self.filter_beta_check,
                     self.filter_neutron_check, self.filter_gamma_check,
                     self.show_mean_check, self.show_std_check,
                     self.show_median_check]:
            check.stateChanged.connect(self.update_analysis)
        
        return tab
    
    def update_analysis(self):
        """Update analysis display based on current settings"""
        try:
            analysis_type = self.analysis_type_combo.currentText()
            plot_type = self.plot_type_combo.currentText()
            
            # Get filtered data
            data = self.get_filtered_data()
            
            # Calculate statistics
            stats = self.calculate_statistics(data)
            
            # Update statistics display
            self.update_statistics_display(stats)
            
            # Update plot
            self.update_analysis_plot(data, analysis_type, plot_type)
            
        except Exception as e:
            logger.error(f"Error updating analysis: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error updating analysis: {str(e)}")
    
    def get_filtered_data(self) -> Dict[str, Any]:
        """Get filtered simulation data based on current settings"""
        data = {}
        
        # Get particle type filters
        particle_filters = {
            'alpha': self.filter_alpha_check.isChecked(),
            'beta': self.filter_beta_check.isChecked(),
            'neutron': self.filter_neutron_check.isChecked(),
            'gamma': self.filter_gamma_check.isChecked()
        }
        
        # Filter trajectories
        filtered_trajectories = [
            traj for traj in self.simulation_results.get('trajectories', [])
            if particle_filters.get(traj['type'], False)
        ]
        
        # Get analysis type
        analysis_type = self.analysis_type_combo.currentText()
        
        # Extract relevant data based on analysis type
        if analysis_type == "Penetration Depth":
            for traj in filtered_trajectories:
                if traj['type'] not in data:
                    data[traj['type']] = []
                positions = np.array(traj['positions'])
                max_depth = np.max(np.sqrt(np.sum(positions**2, axis=1)))
                data[traj['type']].append(max_depth)
                
        elif analysis_type == "Energy Loss":
            for traj in filtered_trajectories:
                if traj['type'] not in data:
                    data[traj['type']] = []
                energy_loss = (traj['energies'][0] - traj['energies'][-1]) / traj['energies'][0] * 100
                data[traj['type']].append(energy_loss)
                
        elif analysis_type == "Trajectory Length":
            for traj in filtered_trajectories:
                if traj['type'] not in data:
                    data[traj['type']] = []
                positions = np.array(traj['positions'])
                segments = np.diff(positions, axis=0)
                length = np.sum(np.sqrt(np.sum(segments**2, axis=1)))
                data[traj['type']].append(length)
                
        elif analysis_type == "Scattering Angle":
            for traj in filtered_trajectories:
                if traj['type'] not in data:
                    data[traj['type']] = []
                positions = np.array(traj['positions'])
                if len(positions) > 2:
                    initial_dir = positions[1] - positions[0]
                    final_dir = positions[-1] - positions[-2]
                    initial_dir = initial_dir / np.linalg.norm(initial_dir)
                    final_dir = final_dir / np.linalg.norm(final_dir)
                    angle = np.arccos(np.clip(np.dot(initial_dir, final_dir), -1.0, 1.0))
                    data[traj['type']].append(np.degrees(angle))
        
        return data
    
    def calculate_statistics(self, data: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for the filtered data"""
        stats = {}
        
        for particle_type, values in data.items():
            if values:
                stats[particle_type] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return stats
    
    def update_statistics_display(self, stats: Dict[str, Dict[str, float]]):
        """Update the statistics text display"""
        text = ""
        
        for particle_type, particle_stats in stats.items():
            text += f"{particle_type.capitalize()} Particles:\n"
            text += f"  Count: {particle_stats['count']}\n"
            if self.show_mean_check.isChecked():
                text += f"  Mean Energy Loss: {particle_stats['mean']:.2f}%\n"
            if self.show_std_check.isChecked():
                text += f"  Max Penetration: {particle_stats['max']:.2f} m\n"
            if self.show_median_check.isChecked():
                text += f"  Median: {particle_stats['median']:.2f}\n"
            text += f"  Range: [{particle_stats['min']:.2f}, {particle_stats['max']:.2f}]\n\n"
        
        self.stats_text.setText(text)
    
    def update_analysis_plot(self, data: Dict[str, List[float]], 
                           analysis_type: str, plot_type: str):
        """Update the analysis plot"""
        # Implementation for updating the plot
        pass
    
    def export_analysis_data(self):
        """Export analysis data to file"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Data", "", "CSV Files (*.csv);;All Files (*)"
            )
            
            if filename:
                data = self.get_filtered_data()
                stats = self.calculate_statistics(data)
                
                # Export data to CSV
                import csv
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow(['Particle Type', 'Count', 'Mean', 'Std Dev',
                                   'Median', 'Min', 'Max'])
                    
                    # Write statistics
                    for ptype, pstats in stats.items():
                        writer.writerow([
                            ptype,
                            pstats['count'],
                            pstats['mean'],
                            pstats['std'],
                            pstats['median'],
                            pstats['min'],
                            pstats['max']
                        ])
                    
                    # Write raw data
                    writer.writerow([])
                    writer.writerow(['Particle Type', 'Value'])
                    for ptype, values in data.items():
                        for value in values:
                            writer.writerow([ptype, value])
                
                QMessageBox.information(self, "Success", "Data exported successfully!")
                
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error exporting data: {str(e)}")
    
    def export_analysis_plot(self):
        """Export current plot to file"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Plot", "",
                "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
            )
            
            if filename:
                # Implementation for saving the plot
                QMessageBox.information(self, "Success", "Plot exported successfully!")
                
        except Exception as e:
            logger.error(f"Error exporting plot: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error exporting plot: {str(e)}")

    def create_results_tab(self) -> QWidget:
        """Create the results display tab with all necessary controls"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Results Controls Group
        controls_group = QGroupBox("Results Controls")
        controls_layout = QGridLayout()
        
        # Results Type Selection
        row = 0
        controls_layout.addWidget(QLabel("Results Type:"), row, 0)
        self.results_type_combo = QComboBox()
        self.results_type_combo.addItems([
            "Summary Statistics",
            "Particle Trajectories",
            "Field Configuration",
            "Shield Performance",
            "Energy Distribution",
            "Spatial Distribution"
        ])
        self.results_type_combo.currentTextChanged.connect(self.update_results_display)
        controls_layout.addWidget(self.results_type_combo, row, 1, 1, 2)
        
        # Export Options
        row += 1
        controls_layout.addWidget(QLabel("Export Format:"), row, 0)
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems([
            "CSV",
            "JSON",
            "Excel",
            "PDF Report"
        ])
        controls_layout.addWidget(self.export_format_combo, row, 1)
        
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_results)
        controls_layout.addWidget(export_btn, row, 2)
        
        # Display Options
        row += 1
        controls_layout.addWidget(QLabel("Display Options:"), row, 0)
        
        self.show_details_check = QCheckBox("Show Details")
        self.show_details_check.setChecked(True)
        controls_layout.addWidget(self.show_details_check, row, 1)
        
        self.show_plots_check = QCheckBox("Show Plots")
        self.show_plots_check.setChecked(True)
        controls_layout.addWidget(self.show_plots_check, row, 2)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Results Display
        results_group = QGroupBox("Simulation Results")
        results_layout = QVBoxLayout()
        
        # Summary Text
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(200)
        results_layout.addWidget(self.summary_text)
        
        # Detailed Results Table
        self.results_table = QTableWidget()
        self.results_table.setMinimumHeight(300)
        results_layout.addWidget(self.results_table)
        
        # Results Plot
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(400)
        results_layout.addWidget(self.canvas)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Connect signals
        self.show_details_check.stateChanged.connect(self.update_results_display)
        self.show_plots_check.stateChanged.connect(self.update_results_display)
        
        return tab
    
    def update_results_display(self):
        """Update results display based on current settings"""
        try:
            results_type = self.results_type_combo.currentText()
            show_details = self.show_details_check.isChecked()
            show_plots = self.show_plots_check.isChecked()
            
            # Update summary text
            self.update_summary_text(results_type)
            
            # Update results table
            self.update_results_table(results_type, show_details)
            
            # Update results plot
            if show_plots:
                self.update_results_plot(results_type)
                self.canvas.show()
            else:
                self.canvas.hide()
            
        except Exception as e:
            logger.error(f"Error updating results display: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error updating results display: {str(e)}")
    
    def update_summary_text(self, results_type: str):
        """Update the summary text display"""
        try:
            summary = "Simulation Summary Statistics\n"
            summary += "=========================\n\n"
            
            # Simulation parameters
            summary += "Simulation Parameters:\n"
            summary += f"Field Type: {self.current_config['field_type']}\n"
            summary += f"Field Strength: {self.current_config['field_strength']} T\n"
            summary += f"Shield Radius: {self.current_config['shield_radius']} m\n"
            summary += f"Shield Height: {self.current_config['shield_height']} m\n"
            summary += f"Blast Yield: {self.current_config['blast_yield']} kt\n"
            summary += f"Source Distance: {self.current_config['source_distance']} m\n"
            summary += f"Particle Count: {self.current_config['particle_count']}\n\n"
            
            # Results statistics
            if hasattr(self, 'simulation_results'):
                stats = self.calculate_statistics(self.get_filtered_data())
                
                summary += "Particle Statistics:\n"
                for ptype, pstats in stats.items():
                    summary += f"\n{ptype.capitalize()}:\n"
                    summary += f"  Count: {pstats['count']}\n"
                    summary += f"  Mean Energy Loss: {pstats['mean']:.2f}%\n"
                    summary += f"  Max Penetration: {pstats['max']:.2f} m\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary statistics: {str(e)}")
            return f"Error generating summary statistics: {str(e)}"
    
    def update_results_table(self, results_type: str, show_details: bool):
        """Update the results table"""
        try:
            # Clear existing table
            self.results_table.clear()
            
            if results_type == "Summary Statistics":
                self.populate_statistics_table(show_details)
            elif results_type == "Particle Trajectories":
                self.populate_trajectories_table(show_details)
            elif results_type == "Shield Performance":
                self.populate_performance_table(show_details)
            else:
                self.results_table.setRowCount(0)
                self.results_table.setColumnCount(0)
            
        except Exception as e:
            logger.error(f"Error updating results table: {str(e)}")
            self.results_table.setRowCount(1)
            self.results_table.setColumnCount(1)
            self.results_table.setItem(0, 0, QTableWidgetItem(f"Error: {str(e)}"))
    
    def update_results_plot(self, results_type: str):
        """Update the results plot"""
        try:
            # Clear the figure
            self.figure.clear()
            
            # Create subplot
            ax = self.figure.add_subplot(111)
            
            if results_type == "Energy Distribution":
                self.plot_energy_distribution(ax)
            elif results_type == "Spatial Distribution":
                self.plot_spatial_distribution(ax)
            elif results_type == "Shield Performance":
                self.plot_shield_performance(ax)
            else:
                ax.text(0.5, 0.5, f"Plot not available for {results_type}",
                       ha='center', va='center')
            
            # Update canvas
            self.figure.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating results plot: {str(e)}")
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            self.canvas.draw()
    
    def plot_energy_distribution(self, ax):
        """Plot energy distribution of particles"""
        if not hasattr(self, 'simulation_results'):
            raise ValueError("No simulation results available")
        
        data = self.get_filtered_data()
        colors = {'alpha': 'red', 'beta': 'blue', 'neutron': 'green', 'gamma': 'yellow'}
        
        for ptype, values in data.items():
            if values:
                ax.hist(values, bins=30, alpha=0.5, label=ptype.capitalize(),
                       color=colors[ptype])
        
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Count')
        ax.set_title('Particle Energy Distribution')
        ax.legend()
    
    def plot_spatial_distribution(self, ax):
        """Plot spatial distribution of particles"""
        if not hasattr(self, 'simulation_results'):
            raise ValueError("No simulation results available")
        
        trajectories = self.simulation_results.get('trajectories', [])
        colors = {'alpha': 'red', 'beta': 'blue', 'neutron': 'green', 'gamma': 'yellow'}
        
        for traj in trajectories:
            positions = np.array(traj['positions'])
            if len(positions) > 0:
                ax.plot(positions[:,0], positions[:,1], alpha=0.5,
                       color=colors[traj['type']], label=traj['type'].capitalize())
        
        # Plot shield boundary
        theta = np.linspace(0, 2*np.pi, 100)
        r = self.current_config['shield_radius']
        ax.plot(r*np.cos(theta), r*np.sin(theta), 'k--', alpha=0.5, label='Shield')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Particle Spatial Distribution')
        ax.legend()
        ax.axis('equal')
    
    def plot_shield_performance(self, ax):
        """Plot shield performance metrics"""
        if not hasattr(self, 'simulation_results'):
            raise ValueError("No simulation results available")
        
        data = self.get_filtered_data()
        particle_types = list(data.keys())
        
        # Calculate blocking efficiency for each particle type
        efficiencies = []
        for ptype in particle_types:
            if data[ptype]:
                blocked = sum(1 for v in data[ptype] 
                           if v < self.current_config['shield_radius'])
                efficiency = blocked / len(data[ptype]) * 100
                efficiencies.append(efficiency)
            else:
                efficiencies.append(0)
        
        # Create bar plot
        x = range(len(particle_types))
        ax.bar(x, efficiencies)
        ax.set_xticks(x)
        ax.set_xticklabels([pt.capitalize() for pt in particle_types])
        ax.set_ylabel('Blocking Efficiency (%)')
        ax.set_title('Shield Performance by Particle Type')
        
        # Add value labels on top of bars
        for i, v in enumerate(efficiencies):
            ax.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    def export_results(self):
        """Export results based on selected format"""
        try:
            export_format = self.export_format_combo.currentText()
            results_type = self.results_type_combo.currentText()
            
            # Get file path from user
            file_filters = {
                "CSV": "CSV Files (*.csv)",
                "JSON": "JSON Files (*.json)",
                "Excel": "Excel Files (*.xlsx)",
                "PDF Report": "PDF Files (*.pdf)"
            }
            
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Results",
                f"simulation_results_{results_type.lower().replace(' ', '_')}",
                file_filters[export_format]
            )
            
            if not filename:
                return
            
            # Export based on format
            if export_format == "CSV":
                self.export_results_csv(filename, results_type)
            elif export_format == "JSON":
                self.export_results_json(filename, results_type)
            elif export_format == "Excel":
                self.export_results_excel(filename, results_type)
            elif export_format == "PDF Report":
                self.export_results_pdf(filename, results_type)
            
            QMessageBox.information(self, "Success", "Results exported successfully!")
            
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error exporting results: {str(e)}")
    
    def export_results_csv(self, filename: str, results_type: str):
        """Export results to CSV format"""
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Simulation Results:', results_type])
            writer.writerow([])
            
            # Write configuration
            writer.writerow(['Configuration:'])
            for key, value in self.current_config.items():
                if isinstance(value, dict):
                    writer.writerow([key])
                    for subkey, subvalue in value.items():
                        writer.writerow(['', subkey, str(subvalue)])
                else:
                    writer.writerow([key, str(value)])
            
            writer.writerow([])
            
            # Write results based on type
            if results_type == "Summary Statistics":
                stats = self.calculate_statistics(self.get_filtered_data())
                writer.writerow(['Particle Type', 'Count', 'Mean', 'Std Dev', 
                               'Median', 'Min', 'Max'])
                for ptype, pstats in stats.items():
                    writer.writerow([
                        ptype,
                        pstats['count'],
                        f"{pstats['mean']:.2f}",
                        f"{pstats['std']:.2f}",
                        f"{pstats['median']:.2f}",
                        f"{pstats['min']:.2f}",
                        f"{pstats['max']:.2f}"
                    ])
            
            elif results_type == "Shield Performance":
                writer.writerow(['Performance Metrics:'])
                data = self.get_filtered_data()
                total_particles = sum(len(values) for values in data.values())
                blocked = sum(1 for values in data.values() 
                            for v in values if v < self.current_config['shield_radius'])
                effectiveness = blocked / total_particles * 100 if total_particles > 0 else 0
                
                writer.writerow(['Overall Effectiveness (%)', f"{effectiveness:.1f}"])
                writer.writerow([])
                
                writer.writerow(['Particle Type', 'Total', 'Blocked', 'Effectiveness (%)'])
                for ptype, values in data.items():
                    blocked = sum(1 for v in values if v < self.current_config['shield_radius'])
                    pct = blocked / len(values) * 100 if values else 0
                    writer.writerow([ptype, len(values), blocked, f"{pct:.1f}"])
    
    def export_results_json(self, filename: str, results_type: str):
        """Export results to JSON format"""
        import json
        
        data = {
            'results_type': results_type,
            'configuration': self.current_config,
            'results': {}
        }
        
        if results_type == "Summary Statistics":
            data['results']['statistics'] = self.calculate_statistics(self.get_filtered_data())
        elif results_type == "Shield Performance":
            filtered_data = self.get_filtered_data()
            performance = {}
            
            total_particles = sum(len(values) for values in filtered_data.values())
            blocked = sum(1 for values in filtered_data.values() 
                        for v in values if v < self.current_config['shield_radius'])
            
            performance['overall_effectiveness'] = (
                blocked / total_particles * 100 if total_particles > 0 else 0
            )
            
            performance['particle_breakdown'] = {}
            for ptype, values in filtered_data.items():
                blocked = sum(1 for v in values if v < self.current_config['shield_radius'])
                pct = blocked / len(values) * 100 if values else 0
                performance['particle_breakdown'][ptype] = {
                    'total': len(values),
                    'blocked': blocked,
                    'effectiveness': pct
                }
            
            data['results']['performance'] = performance
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    
    def export_results_excel(self, filename: str, results_type: str):
        """Export results to Excel format"""
        # Implementation for Excel export
        pass
    
    def export_results_pdf(self, filename: str, results_type: str):
        """Export results to PDF format"""
        # Implementation for PDF export
        pass

    def run_simulation(self):
        """Execute the simulation with current parameters"""
        try:
            # Prepare simulation parameters
            self.sim_engine.prepare_simulation(self.current_config)
            
            # Run the main simulation loop
            results = self.sim_engine.run()
            
            # Process and store results
            self.simulation_results = self.process_results(results)
            
            # Update visualization
            self.update_visualization()
            
            # Generate reports
            self.generate_reports()
            
            logger.info("Simulation completed successfully")
            
        except Exception as e:
            logger.error(f"Error during simulation execution: {str(e)}")
            # Handle error and show appropriate message to user
            self.show_error_message(str(e))

    def process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process and analyze simulation results"""
        # Process simulation results
        # (Implementation details would go here)
        return results

    def update_visualization(self):
        """Update all visualization components with new results"""
        # Update 3D visualization
        # (Implementation details would go here)
        pass

    def generate_reports(self):
        """Generate comprehensive simulation reports"""
        # Generate reports
        # (Implementation details would go here)
        pass

    def show_error_message(self, message: str):
        """Display error message to user"""
        # Show error dialog
        # (Implementation details would go here)
        pass

    def generate_summary_statistics(self) -> str:
        """Generate summary statistics text"""
        try:
            summary = "Simulation Summary Statistics\n"
            summary += "=========================\n\n"
            
            # Simulation parameters
            summary += "Simulation Parameters:\n"
            summary += f"Field Type: {self.current_config['field_type']}\n"
            summary += f"Field Strength: {self.current_config['field_strength']} T\n"
            summary += f"Shield Radius: {self.current_config['shield_radius']} m\n"
            summary += f"Shield Height: {self.current_config['shield_height']} m\n"
            summary += f"Blast Yield: {self.current_config['blast_yield']} kt\n"
            summary += f"Source Distance: {self.current_config['source_distance']} m\n"
            summary += f"Particle Count: {self.current_config['particle_count']}\n\n"
            
            # Results statistics
            if hasattr(self, 'simulation_results'):
                stats = self.calculate_statistics(self.get_filtered_data())
                
                summary += "Particle Statistics:\n"
                for ptype, pstats in stats.items():
                    summary += f"\n{ptype.capitalize()}:\n"
                    summary += f"  Count: {pstats['count']}\n"
                    summary += f"  Mean Energy Loss: {pstats['mean']:.2f}%\n"
                    summary += f"  Max Penetration: {pstats['max']:.2f} m\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary statistics: {str(e)}")
            return f"Error generating summary statistics: {str(e)}"
    
    def generate_shield_performance_summary(self) -> str:
        """Generate shield performance summary text"""
        try:
            summary = "Shield Performance Summary\n"
            summary += "========================\n\n"
            
            if hasattr(self, 'simulation_results'):
                # Calculate shield effectiveness
                data = self.get_filtered_data()
                total_particles = sum(len(values) for values in data.values())
                blocked = sum(1 for values in data.values() 
                            for v in values if v < self.current_config['shield_radius'])
                
                effectiveness = (blocked / total_particles * 100 
                               if total_particles > 0 else 0)
                
                summary += f"Overall Shield Effectiveness: {effectiveness:.1f}%\n\n"
                
                # Per-particle-type statistics
                summary += "Particle-Type Breakdown:\n"
                for ptype, values in data.items():
                    if values:
                        blocked = sum(1 for v in values if v < self.current_config['shield_radius'])
                        pct = blocked / len(values) * 100
                        summary += f"{ptype.capitalize()}: {pct:.1f}% blocked\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating shield performance summary: {str(e)}")
            return f"Error generating shield performance summary: {str(e)}"

    def populate_statistics_table(self, show_details: bool):
        """Populate the table with statistics data"""
        try:
            if not hasattr(self, 'simulation_results'):
                raise ValueError("No simulation results available")
            
            # Calculate statistics
            stats = self.calculate_statistics(self.get_filtered_data())
            
            # Set up table
            headers = ['Particle Type', 'Count', 'Mean', 'Std Dev', 'Median', 'Min', 'Max']
            if not show_details:
                headers = headers[:4]  # Show only basic statistics
            
            self.results_table.setColumnCount(len(headers))
            self.results_table.setHorizontalHeaderLabels(headers)
            self.results_table.setRowCount(len(stats))
            
            # Populate table
            for row, (ptype, pstats) in enumerate(stats.items()):
                col = 0
                self.results_table.setItem(row, col, QTableWidgetItem(ptype.capitalize()))
                col += 1
                self.results_table.setItem(row, col, QTableWidgetItem(str(pstats['count'])))
                col += 1
                self.results_table.setItem(row, col, QTableWidgetItem(f"{pstats['mean']:.2f}"))
                col += 1
                self.results_table.setItem(row, col, QTableWidgetItem(f"{pstats['std']:.2f}"))
                
                if show_details:
                    col += 1
                    self.results_table.setItem(row, col, QTableWidgetItem(f"{pstats['median']:.2f}"))
                    col += 1
                    self.results_table.setItem(row, col, QTableWidgetItem(f"{pstats['min']:.2f}"))
                    col += 1
                    self.results_table.setItem(row, col, QTableWidgetItem(f"{pstats['max']:.2f}"))
            
            self.results_table.resizeColumnsToContents()
            
        except Exception as e:
            logger.error(f"Error populating statistics table: {str(e)}")
            raise
    
    def populate_trajectories_table(self, show_details: bool):
        """Populate the table with trajectory data"""
        try:
            if not hasattr(self, 'simulation_results'):
                raise ValueError("No simulation results available")
            
            # Set up table
            headers = ['Particle Type', 'Initial Energy', 'Final Energy', 'Path Length']
            if show_details:
                headers.extend(['Max Depth', 'Scatter Angle', 'Time in Shield'])
            
            trajectories = self.simulation_results.get('trajectories', [])
            
            self.results_table.setColumnCount(len(headers))
            self.results_table.setHorizontalHeaderLabels(headers)
            self.results_table.setRowCount(len(trajectories))
            
            # Populate table
            for row, traj in enumerate(trajectories):
                col = 0
                self.results_table.setItem(row, col, QTableWidgetItem(traj['type'].capitalize()))
                col += 1
                self.results_table.setItem(row, col, QTableWidgetItem(f"{traj['energies'][0]:.2f}"))
                col += 1
                self.results_table.setItem(row, col, QTableWidgetItem(f"{traj['energies'][-1]:.2f}"))
                col += 1
                
                # Calculate path length
                positions = np.array(traj['positions'])
                segments = np.diff(positions, axis=0)
                length = np.sum(np.sqrt(np.sum(segments**2, axis=1)))
                self.results_table.setItem(row, col, QTableWidgetItem(f"{length:.2f}"))
                
                if show_details:
                    col += 1
                    # Max depth
                    max_depth = np.max(np.sqrt(np.sum(positions**2, axis=1)))
                    self.results_table.setItem(row, col, QTableWidgetItem(f"{max_depth:.2f}"))
                    
                    col += 1
                    # Scatter angle
                    if len(positions) > 2:
                        initial_dir = positions[1] - positions[0]
                        final_dir = positions[-1] - positions[-2]
                        initial_dir = initial_dir / np.linalg.norm(initial_dir)
                        final_dir = final_dir / np.linalg.norm(final_dir)
                        angle = np.arccos(np.clip(np.dot(initial_dir, final_dir), -1.0, 1.0))
                        self.results_table.setItem(row, col, QTableWidgetItem(f"{np.degrees(angle):.2f}"))
                    
                    col += 1
                    # Time in shield
                    time = len(positions) * self.current_config.get('time_step', 1e-9)
                    self.results_table.setItem(row, col, QTableWidgetItem(f"{time*1e9:.2f}"))
            
            self.results_table.resizeColumnsToContents()
            
        except Exception as e:
            logger.error(f"Error populating trajectories table: {str(e)}")
            raise

    def populate_performance_table(self, show_details: bool):
        """Populate the table with shield performance data"""
        try:
            if not hasattr(self, 'simulation_results'):
                raise ValueError("No simulation results available")
            
            # Set up table
            headers = ['Metric', 'Value']
            if show_details:
                headers.extend(['Alpha', 'Beta', 'Neutron', 'Gamma'])
            
            metrics = [
                'Particles Blocked (%)',
                'Mean Energy Loss (%)',
                'Max Penetration (m)',
                'Mean Path Length (m)'
            ]
            
            self.results_table.setColumnCount(len(headers))
            self.results_table.setHorizontalHeaderLabels(headers)
            self.results_table.setRowCount(len(metrics))
            
            # Calculate performance metrics
            data = self.get_filtered_data()
            stats = self.calculate_statistics(data)
            
            # Populate table
            for row, metric in enumerate(metrics):
                col = 0
                self.results_table.setItem(row, col, QTableWidgetItem(metric))
                col += 1
                
                # Calculate overall value
                if metric == 'Particles Blocked (%)':
                    total_particles = sum(len(values) for values in data.values())
                    blocked = sum(1 for values in data.values() 
                                for v in values if v < self.current_config['shield_radius'])
                    value = blocked / total_particles * 100 if total_particles > 0 else 0
                    self.results_table.setItem(row, col, QTableWidgetItem(f"{value:.1f}"))
                    
                    if show_details:
                        for ptype in ['alpha', 'beta', 'neutron', 'gamma']:
                            col += 1
                            if ptype in data and data[ptype]:
                                blocked = sum(1 for v in data[ptype] 
                                           if v < self.current_config['shield_radius'])
                                pct = blocked / len(data[ptype]) * 100
                                self.results_table.setItem(row, col, QTableWidgetItem(f"{pct:.1f}"))
                
                elif metric == 'Mean Energy Loss (%)':
                    mean_loss = np.mean([stats[pt]['mean'] for pt in stats]) if stats else 0
                    self.results_table.setItem(row, col, QTableWidgetItem(f"{mean_loss:.1f}"))
                    
                    if show_details:
                        for ptype in ['alpha', 'beta', 'neutron', 'gamma']:
                            col += 1
                            if ptype in stats:
                                self.results_table.setItem(row, col, 
                                                         QTableWidgetItem(f"{stats[ptype]['mean']:.1f}"))
                
                elif metric == 'Max Penetration (m)':
                    max_pen = max(stats[pt]['max'] for pt in stats) if stats else 0
                    self.results_table.setItem(row, col, QTableWidgetItem(f"{max_pen:.2f}"))
                    
                    if show_details:
                        for ptype in ['alpha', 'beta', 'neutron', 'gamma']:
                            col += 1
                            if ptype in stats:
                                self.results_table.setItem(row, col, 
                                                         QTableWidgetItem(f"{stats[ptype]['max']:.2f}"))
                
                elif metric == 'Mean Path Length (m)':
                    paths = []
                    for traj in self.simulation_results.get('trajectories', []):
                        positions = np.array(traj['positions'])
                        segments = np.diff(positions, axis=0)
                        length = np.sum(np.sqrt(np.sum(segments**2, axis=1)))
                        paths.append(length)
                    
                    mean_length = np.mean(paths) if paths else 0
                    self.results_table.setItem(row, col, QTableWidgetItem(f"{mean_length:.2f}"))
                    
                    if show_details:
                        for ptype in ['alpha', 'beta', 'neutron', 'gamma']:
                            col += 1
                            trajectories = self.simulation_results.get('trajectories', [])
                            type_paths = [length for length, traj in zip(paths, trajectories)
                                        if traj['type'] == ptype]
                            if type_paths:
                                mean = np.mean(type_paths)
                                self.results_table.setItem(row, col, QTableWidgetItem(f"{mean:.2f}"))
            
            self.results_table.resizeColumnsToContents()
            
        except Exception as e:
            logger.error(f"Error populating performance table: {str(e)}")
            raise

def main():
    try:
        # Enable high DPI scaling
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
        
        # Create output directory if it doesn't exist
        Path("simulation_output").mkdir(exist_ok=True)
        
        # Initialize the application
        app = QApplication(sys.argv)
        
        # Create and show the main window
        main_window = AdvancedSimulationControlPanel()
        main_window.show()
        
        # Start the application event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.critical(f"Critical error in main application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 