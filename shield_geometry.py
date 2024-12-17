"""
Shield geometry module for defining and managing shield configurations.
This module provides classes for different shield geometries and their properties.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

from core_physics import Vector3D, Material, PhysicsObject
from shield_materials import ShieldMaterial

@dataclass
class ShieldLayer:
    """A single layer in a shield configuration"""
    material: ShieldMaterial
    thickness: float  # [m]
    position: Vector3D  # Layer's center position
    temperature: float = 293.15  # [K]
    pressure: float = 101325.0  # [Pa]

class ShieldGeometry:
    """Class for managing shield geometry and layer configuration"""
    def __init__(self, layers: List[ShieldLayer]):
        self.layers = layers
        self._calculate_boundaries()
    
    def _calculate_boundaries(self):
        """Calculate the boundaries of each layer"""
        current_radius = 0
        self.boundaries = []
        
        for layer in self.layers:
            inner_radius = current_radius
            outer_radius = current_radius + layer.thickness
            self.boundaries.append((inner_radius, outer_radius))
            current_radius = outer_radius
    
    def add_layer(self, layer: ShieldLayer):
        """Add a new layer to the shield"""
        self.layers.append(layer)
        self._calculate_boundaries()
    
    def get_intersection(self, position: Vector3D, direction: Vector3D) -> List[Tuple[float, ShieldLayer]]:
        """Get intersection points of a ray with shield layers"""
        intersections = []
        direction = direction.normalize()
        
        for i, (inner_r, outer_r) in enumerate(self.boundaries):
            # Calculate intersection with spherical shells
            a = direction.dot(direction)
            b = 2 * position.dot(direction)
            
            # Inner intersection
            c_inner = position.dot(position) - inner_r**2
            discriminant_inner = b**2 - 4*a*c_inner
            
            if discriminant_inner >= 0:
                t1 = (-b - np.sqrt(discriminant_inner)) / (2*a)
                t2 = (-b + np.sqrt(discriminant_inner)) / (2*a)
                if t1 > 0:
                    intersections.append((t1, self.layers[i]))
                if t2 > 0:
                    intersections.append((t2, self.layers[i]))
            
            # Outer intersection
            c_outer = position.dot(position) - outer_r**2
            discriminant_outer = b**2 - 4*a*c_outer
            
            if discriminant_outer >= 0:
                t1 = (-b - np.sqrt(discriminant_outer)) / (2*a)
                t2 = (-b + np.sqrt(discriminant_outer)) / (2*a)
                if t1 > 0:
                    intersections.append((t1, self.layers[i]))
                if t2 > 0:
                    intersections.append((t2, self.layers[i]))
        
        # Sort intersections by distance
        intersections.sort(key=lambda x: x[0])
        return intersections
    
    def get_material_at(self, position: Vector3D) -> Optional[ShieldLayer]:
        """Get shield layer at a given position"""
        r = position.magnitude()
        
        for i, (inner_r, outer_r) in enumerate(self.boundaries):
            if inner_r <= r <= outer_r:
                return self.layers[i]
        
        return None
    
    def get_total_mass(self) -> float:
        """Calculate total mass of the shield"""
        total_mass = 0.0
        for layer, (inner_r, outer_r) in zip(self.layers, self.boundaries):
            volume = 4/3 * np.pi * (outer_r**3 - inner_r**3)
            total_mass += volume * layer.material.density
        return total_mass
    
    def get_volume(self) -> float:
        """Calculate total volume of the shield"""
        total_volume = 0.0
        for inner_r, outer_r in self.boundaries:
            volume = 4/3 * np.pi * (outer_r**3 - inner_r**3)
            total_volume += volume
        return total_volume
    
    def get_layer_properties(self) -> List[Dict]:
        """Get properties of all layers"""
        properties = []
        for layer, (inner_r, outer_r) in zip(self.layers, self.boundaries):
            volume = 4/3 * np.pi * (outer_r**3 - inner_r**3)
            mass = volume * layer.material.density
            
            properties.append({
                'material': layer.material.name,
                'thickness': layer.thickness,
                'inner_radius': inner_r,
                'outer_radius': outer_r,
                'volume': volume,
                'mass': mass,
                'temperature': layer.temperature,
                'pressure': layer.pressure
            })
        
        return properties 

class ShieldGeometryManager:
    """Manager class for handling shield geometries and configurations"""
    def __init__(self):
        self.geometries: Dict[str, ShieldGeometry] = {}
        self.active_geometry: Optional[str] = None
        self.default_materials = {
            'outer': 'tungsten',
            'middle': 'lead',
            'inner': 'concrete'
        }
    
    def create_spherical_shield(self, name: str, radius: float, 
                              layer_thicknesses: List[float], 
                              materials: Optional[List[str]] = None) -> ShieldGeometry:
        """Create a new spherical shield configuration"""
        if materials is None:
            materials = list(self.default_materials.values())
        
        if len(layer_thicknesses) != len(materials):
            raise ValueError("Number of layers must match number of materials")
        
        layers = []
        current_radius = 0
        
        for thickness, material_name in zip(layer_thicknesses, materials):
            material = ShieldMaterial(material_name)
            material.thickness = thickness
            
            layer = ShieldLayer(
                material=material,
                thickness=thickness,
                position=Vector3D(0, 0, 0)  # Center at origin
            )
            
            layers.append(layer)
            current_radius += thickness
        
        geometry = ShieldGeometry(layers)
        self.geometries[name] = geometry
        
        if self.active_geometry is None:
            self.active_geometry = name
        
        return geometry
    
    def create_cylindrical_shield(self, name: str, radius: float, height: float,
                                layer_thicknesses: List[float],
                                materials: Optional[List[str]] = None) -> ShieldGeometry:
        """Create a new cylindrical shield configuration"""
        if materials is None:
            materials = list(self.default_materials.values())
        
        if len(layer_thicknesses) != len(materials):
            raise ValueError("Number of layers must match number of materials")
        
        # For cylindrical geometry, we approximate with spherical layers
        # but adjust the thickness to maintain equivalent protection
        adjusted_thicknesses = []
        for thickness in layer_thicknesses:
            # Adjust thickness to maintain equivalent mass/protection
            adjusted_thickness = thickness * np.sqrt((radius * height) / 
                                                   (4/3 * np.pi * radius**3))
            adjusted_thicknesses.append(adjusted_thickness)
        
        return self.create_spherical_shield(name, radius, adjusted_thicknesses, materials)
    
    def get_geometry(self, name: str) -> Optional[ShieldGeometry]:
        """Get a shield geometry by name"""
        return self.geometries.get(name)
    
    def set_active_geometry(self, name: str):
        """Set the active shield geometry"""
        if name not in self.geometries:
            raise ValueError(f"Geometry '{name}' not found")
        self.active_geometry = name
    
    def get_active_geometry(self) -> Optional[ShieldGeometry]:
        """Get the currently active shield geometry"""
        if self.active_geometry is None:
            return None
        return self.geometries[self.active_geometry]
    
    def optimize_shield(self, name: str, target_mass: float, 
                       min_thickness: float = 0.1) -> ShieldGeometry:
        """Optimize shield layer thicknesses for a target mass"""
        geometry = self.get_geometry(name)
        if geometry is None:
            raise ValueError(f"Geometry '{name}' not found")
        
        current_mass = geometry.get_total_mass()
        scale_factor = (target_mass / current_mass) ** (1/3)
        
        # Scale layer thicknesses
        new_layers = []
        for layer in geometry.layers:
            new_thickness = max(layer.thickness * scale_factor, min_thickness)
            new_layer = ShieldLayer(
                material=layer.material,
                thickness=new_thickness,
                position=layer.position,
                temperature=layer.temperature,
                pressure=layer.pressure
            )
            new_layers.append(new_layer)
        
        # Create new optimized geometry
        optimized_name = f"{name}_optimized"
        optimized_geometry = ShieldGeometry(new_layers)
        self.geometries[optimized_name] = optimized_geometry
        
        return optimized_geometry
    
    def calculate_shield_effectiveness(self, geometry: ShieldGeometry, 
                                    particle_type: str, energy: float) -> float:
        """Calculate shield effectiveness for a given particle type and energy"""
        total_attenuation = 1.0
        
        for layer in geometry.layers:
            # Get relevant cross sections
            scatter_xs = layer.material.get_scattering_cross_section(particle_type, energy)
            absorb_xs = layer.material.get_absorption_cross_section(particle_type, energy)
            
            # Calculate attenuation
            total_xs = scatter_xs + absorb_xs
            attenuation = np.exp(-total_xs * layer.material.density * layer.thickness)
            total_attenuation *= attenuation
        
        return 1.0 - total_attenuation  # Return effectiveness (0 to 1)
    
    def get_shield_properties(self, name: str) -> Dict:
        """Get comprehensive shield properties"""
        geometry = self.get_geometry(name)
        if geometry is None:
            raise ValueError(f"Geometry '{name}' not found")
        
        properties = {
            'name': name,
            'total_mass': geometry.get_total_mass(),
            'total_volume': geometry.get_volume(),
            'number_of_layers': len(geometry.layers),
            'layer_properties': geometry.get_layer_properties(),
            'outer_radius': geometry.boundaries[-1][1],
            'materials_used': [layer.material.name for layer in geometry.layers]
        }
        
        return properties
    
    def clear(self):
        """Clear all shield geometries"""
        self.geometries.clear()
        self.active_geometry = None