"""
Magnetic field module for defining and managing magnetic fields in the simulation.
"""

import numpy as np
from core_physics import Vector3D, Field, PhysicalConstants
from typing import List, Tuple, Optional, Dict, Any
from scipy.spatial import cKDTree
import warnings
import logging

logger = logging.getLogger(__name__)

class MagneticField(Field):
    """Base magnetic field class"""
    def __init__(self, strength: float, direction: Vector3D):
        """
        Args:
            strength: Magnetic field strength (Tesla)
            direction: Magnetic field direction (unit vector)
        """
        self.strength = strength
        self.direction = direction.normalize()
        
        # Field grid parameters
        self.grid_points = 20  # Reduced from 50 to save memory
        self.grid_size = 10.0  # meters
        
        # Cache for field values
        self._cache_size = 100  # Reduced cache size
        self._cache = {}
        self._cache_tree = None
        self._cached_points = []
        self._cached_fields = []
    
    def get_field_at(self, position: Vector3D, time: float = 0) -> Vector3D:
        """Calculate magnetic field at a given position"""
        # Try to get value from cache
        pos_key = (round(position.x, 3), round(position.y, 3), round(position.z, 3))
        if pos_key in self._cache:
            return self._cache[pos_key]
        
        # Calculate field
        field = self._compute_field(position, time)
        
        # Update cache
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[pos_key] = field
        return field
    
    def _compute_field(self, position: Vector3D, time: float = 0) -> Vector3D:
        """Compute base magnetic field"""
        # Basic uniform field implementation
        return Vector3D(
            self.direction.x * self.strength,
            self.direction.y * self.strength,
            self.direction.z * self.strength
        )
    
    def get_potential_at(self, position: Vector3D, time: float = 0) -> float:
        """Calculate magnetic vector potential"""
        # Simple implementation for uniform field
        r = position
        B = self.get_field_at(position, time)
        A_vec = B.cross(r)
        return 0.5 * A_vec.magnitude()
    
    def initialize_field_grid(self):
        """Initialize a sparse field grid for interpolation"""
        # Create a sparse grid of points
        x = np.linspace(-self.grid_size, self.grid_size, self.grid_points)
        y = np.linspace(-self.grid_size, self.grid_size, self.grid_points)
        z = np.linspace(-self.grid_size, self.grid_size, self.grid_points)
        
        # Generate points only where field is significant
        points = []
        fields = []
        
        for xi in x[::2]:  # Use every other point to reduce memory
            for yi in y[::2]:
                for zi in z[::2]:
                    pos = Vector3D(xi, yi, zi)
                    field = self._compute_field(pos)
                    
                    # Only store points with significant field
                    if field.magnitude() > 1e-6 * self.strength:
                        points.append([xi, yi, zi])
                        fields.append([field.x, field.y, field.z])
        
        if points:
            self._cached_points = np.array(points)
            self._cached_fields = np.array(fields)
            self._cache_tree = cKDTree(self._cached_points)
        else:
            warnings.warn("No significant field points found during grid initialization")

class DipoleField(MagneticField):
    """Magnetic dipole field"""
    def __init__(self, moment: Vector3D, position: Vector3D = Vector3D(0, 0, 0)):
        super().__init__(0, Vector3D(0, 0, 0))
        self.moment = moment
        self.position = position
    
    def _compute_field(self, position: Vector3D, time: float = 0) -> Vector3D:
        r = position - self.position
        r_mag = r.magnitude()
        
        if r_mag < 1e-10:
            return Vector3D(0, 0, 0)
        
        r_hat = r * (1.0 / r_mag)
        m_dot_r = self.moment.dot(r_hat)
        
        # B = (μ0/4π) * (3(m·r̂)r̂ - m)/r³
        factor = PhysicalConstants.mu_0 / (4 * np.pi * r_mag**3)
        return (r_hat * m_dot_r * 3.0 - self.moment) * factor

class SolenoidField(MagneticField):
    """Solenoid magnetic field"""
    def __init__(self, center: Vector3D, radius: float, length: float, 
                 n_turns: int, current: float, axis: Vector3D = Vector3D(0, 0, 1)):
        super().__init__(0, Vector3D(0, 0, 0))
        self.center = center
        self.radius = radius
        self.length = length
        self.n_turns = n_turns
        self.current = current
        self.axis = axis.normalize()
    
    def _compute_field(self, position: Vector3D, time: float = 0) -> Vector3D:
        # Convert position to cylindrical coordinates relative to solenoid axis
        r = position - self.center
        z = r.dot(self.axis)
        rho = (r - self.axis * z).magnitude()
        
        if abs(z) > self.length/2:
            # Field outside the solenoid ends
            return self._external_field(position)
        
        # Field inside the solenoid
        B_mag = PhysicalConstants.mu_0 * self.n_turns * self.current / self.length
        return self.axis * B_mag
    
    def _external_field(self, position: Vector3D) -> Vector3D:
        # Approximate field outside solenoid using dipole approximation
        moment = self.axis * (np.pi * self.radius**2 * self.n_turns * self.current)
        dipole = DipoleField(moment, self.center)
        return dipole._compute_field(position)

class ToroidalField(MagneticField):
    """Toroidal magnetic field"""
    def __init__(self, center: Vector3D, major_radius: float, minor_radius: float,
                 n_coils: int, current: float):
        super().__init__(0, Vector3D(0, 0, 0))
        self.center = center
        self.R = major_radius
        self.a = minor_radius
        self.n_coils = n_coils
        self.current = current
    
    def _compute_field(self, position: Vector3D, time: float = 0) -> Vector3D:
        # Convert to toroidal coordinates
        r = position - self.center
        R = np.sqrt(r.x**2 + r.y**2)  # Distance from vertical axis
        phi = np.arctan2(r.y, r.x)
        
        # Calculate field strength (approximation valid near R = R0)
        B_phi = PhysicalConstants.mu_0 * self.n_coils * self.current / (2 * np.pi * R)
        
        # Convert back to Cartesian coordinates
        return Vector3D(
            -B_phi * np.sin(phi),
            B_phi * np.cos(phi),
            0
        )

class CustomField(MagneticField):
    """Composite magnetic field from multiple components"""
    def __init__(self, field_components: List[MagneticField]):
        super().__init__(0, Vector3D(0, 0, 0))
        self.components = field_components
    
    def _compute_field(self, position: Vector3D, time: float = 0) -> Vector3D:
        total_field = Vector3D(0, 0, 0)
        for component in self.components:
            total_field += component.get_field_at(position, time)
        return total_field

class HelmholtzCoilField(MagneticField):
    """Helmholtz coil magnetic field configuration"""
    def __init__(self, radius: float, current: float):
        """
        Args:
            radius: Radius of the Helmholtz coils (meters)
            current: Current flowing through the coils (Amperes)
        """
        self.radius = radius
        self.current = current
        self.mu0 = PhysicalConstants.MU_0  # Vacuum permeability
        
        # Calculate field strength at center
        self.B0 = (8 * self.mu0 * self.current) / (np.sqrt(125) * self.radius)
        super().__init__(strength=self.B0, direction=Vector3D(0, 0, 1))
    
    def get_field_at(self, position: Vector3D) -> Vector3D:
        """Calculate magnetic field at given position"""
        x, y, z = position.x, position.y, position.z
        r = np.sqrt(x*x + y*y)
        
        if r < 1e-10:  # On axis
            Bz = self.B0
            return Vector3D(0, 0, Bz)
            
        # Off-axis field calculation using approximation
        Bz = self.B0 * (1 - (r/(4*self.radius))**2)
        Br = -self.B0 * (r/(4*self.radius)) * (z/self.radius)
        
        # Convert to Cartesian components
        Bx = Br * (x/r)
        By = Br * (y/r)
        
        return Vector3D(Bx, By, Bz)

class MagneticFieldManager:
    """Manager class for handling different magnetic field configurations"""
    def __init__(self):
        self.available_field_types = {
            'solenoid': SolenoidField,
            'toroidal': ToroidalField,
            'helmholtz': HelmholtzCoilField,
            'dipole': DipoleField,
            'custom': CustomField
        }
        self.active_field: Optional[MagneticField] = None
        self.field_parameters: Dict[str, Any] = {}
        
        logger.info("MagneticFieldManager initialized with available field types: %s", 
                   list(self.available_field_types.keys()))
    
    def create_field(self, field_type: str, parameters: Dict[str, Any]) -> MagneticField:
        """Create a magnetic field of specified type with given parameters
        
        Args:
            field_type: Type of magnetic field to create
            parameters: Dictionary of parameters for field initialization
            
        Returns:
            MagneticField: Initialized magnetic field object
            
        Raises:
            ValueError: If field_type is not supported or parameters are invalid
        """
        try:
            if field_type not in self.available_field_types:
                raise ValueError(f"Unsupported field type: {field_type}")
            
            field_class = self.available_field_types[field_type]
            
            # Validate parameters
            self._validate_parameters(field_type, parameters)
            
            # Create field instance
            if field_type == 'solenoid':
                field = field_class(
                    center=parameters.get('center', Vector3D(0, 0, 0)),
                    radius=parameters['radius'],
                    length=parameters['length'],
                    n_turns=parameters['n_turns'],
                    current=parameters['current'],
                    axis=parameters.get('axis', Vector3D(0, 0, 1))
                )
            elif field_type == 'toroidal':
                field = field_class(
                    center=parameters.get('center', Vector3D(0, 0, 0)),
                    major_radius=parameters['major_radius'],
                    minor_radius=parameters['minor_radius'],
                    n_coils=parameters['n_coils'],
                    current=parameters['current']
                )
            elif field_type == 'helmholtz':
                field = field_class(
                    radius=parameters['radius'],
                    current=parameters['current']
                )
            elif field_type == 'dipole':
                field = field_class(
                    moment=parameters['moment'],
                    position=parameters.get('position', Vector3D(0, 0, 0))
                )
            elif field_type == 'custom':
                field = field_class(parameters['field_components'])
            
            logger.info("Created %s field with parameters: %s", field_type, parameters)
            return field
            
        except KeyError as e:
            raise ValueError(f"Missing required parameter for {field_type} field: {str(e)}")
        except Exception as e:
            logger.error("Error creating magnetic field: %s", str(e))
            raise
    
    def set_active_field(self, field_type: str, parameters: Dict[str, Any]) -> None:
        """Set the active magnetic field configuration
        
        Args:
            field_type: Type of magnetic field to activate
            parameters: Dictionary of parameters for field initialization
        """
        try:
            self.active_field = self.create_field(field_type, parameters)
            self.field_parameters = parameters.copy()
            logger.info("Active field set to %s", field_type)
        except Exception as e:
            logger.error("Error setting active field: %s", str(e))
            raise
    
    def get_field_at(self, position: Vector3D, time: float = 0) -> Vector3D:
        """Get magnetic field vector at specified position
        
        Args:
            position: Position to calculate field at
            time: Time at which to calculate field (for time-varying fields)
            
        Returns:
            Vector3D: Magnetic field vector at specified position
        """
        if self.active_field is None:
            return Vector3D(0, 0, 0)
        return self.active_field.get_field_at(position, time)
    
    def _validate_parameters(self, field_type: str, parameters: Dict[str, Any]) -> None:
        """Validate parameters for specified field type
        
        Args:
            field_type: Type of magnetic field
            parameters: Dictionary of parameters to validate
            
        Raises:
            ValueError: If parameters are invalid
        """
        required_params = {
            'solenoid': ['radius', 'length', 'n_turns', 'current'],
            'toroidal': ['major_radius', 'minor_radius', 'n_coils', 'current'],
            'helmholtz': ['radius', 'current'],
            'dipole': ['moment'],
            'custom': ['field_components']
        }
        
        # Check required parameters
        for param in required_params[field_type]:
            if param not in parameters:
                raise ValueError(f"Missing required parameter '{param}' for {field_type} field")
        
        # Validate parameter values
        if field_type == 'solenoid':
            if parameters['radius'] <= 0:
                raise ValueError("Solenoid radius must be positive")
            if parameters['length'] <= 0:
                raise ValueError("Solenoid length must be positive")
            if parameters['n_turns'] <= 0:
                raise ValueError("Number of turns must be positive")
        elif field_type == 'toroidal':
            if parameters['major_radius'] <= 0:
                raise ValueError("Major radius must be positive")
            if parameters['minor_radius'] <= 0:
                raise ValueError("Minor radius must be positive")
            if parameters['n_coils'] <= 0:
                raise ValueError("Number of coils must be positive")
        elif field_type == 'helmholtz':
            if parameters['radius'] <= 0:
                raise ValueError("Helmholtz coil radius must be positive")
        elif field_type == 'custom':
            if not isinstance(parameters['field_components'], list):
                raise ValueError("Field components must be a list of MagneticField objects")
            if not all(isinstance(f, MagneticField) for f in parameters['field_components']):
                raise ValueError("All field components must be MagneticField instances")
        
        # Validate current for all field types that use it
        if 'current' in parameters:
            if not isinstance(parameters['current'], (int, float)):
                raise ValueError("Current must be a number")
            if parameters['current'] <= 0:
                raise ValueError("Current must be positive")