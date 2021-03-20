# Sara Ahanchi
# ASTR 302 Final Project
# Due March 19, 2021
# The goal of this project is to graph a given ray and its relection or refraction while hitting a flatt mirror, absorber or medium change.
# You can choose the type of surface using a widget and vary the x and y coordinates of the incident ray.
# For medium change, n_1 and n_2 are also interactive. 
# Keep in mind the size and position of the surface and the interaction point are fixed

# Import the needed Modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import ipywidgets as widgets
from ipywidgets import interact

# A function used in classes and other functions that will take any vector and return a unit vector with the same direction.
def unit_vector(v):
    return(v / np.linalg.norm(v))

# CLASSES

class Ray:
    # Defined a class for rays to interact with an object.
    def __init__(self, direction, activity = True, history =[]):
        self.direction = np.array(unit_vector(direction))
        self.activity = activity
        self.history = history
        

class mirror_flat:
    # Defined a class for flat mirrors.
    def __init__(self, endpoint_1, endpoint_2):
        self.endpoint_1 = np.array(endpoint_1)  
        self.endpoint_2 = np.array(endpoint_2) 
    def param_func(self, t):
        t_point = t * self.endpoint_1 + (1 -t) * self.endpoint_2
        return(t_point)

class absorber:
    # Defined a class for absorbers where rays do not go through.
    def __init__(self, endpoint_1, endpoint_2):
        # The attributes are the two endpoints of the flat black screen.
        self.endpoint_1 = np.array(endpoint_1)  
        self.endpoint_2 = np.array(endpoint_2)   
    def param_func(self, t):
        t_point = t * self.endpoint_1 + (1 -t) * self.endpoint_2
        return(t_point)
    
class medium_change:
    # Defined a class for ray going through the medium change.
    def __init__(self, endpoint_1, endpoint_2):
        # The attributes are the two endpoints of the surface.
        self.endpoint_1 = np.array(endpoint_1)  
        self.endpoint_2 = np.array(endpoint_2)   
    def param_func(self, t):
        t_point = t * self.endpoint_1 + (1 -t) * self.endpoint_2
        return(t_point)
    
    
# FUNCTIONS
               

def make_normal(surface, t):
    # Function make_normal is able to find the normal vector to a surface at point t.
    # The cross product between the tangent line to the surface at point t and a vector perpendicular
    # to the page will give a vector in the same direction as the normal.
    normal = np.cross((surface(t+0.001) - surface(t)), [0, 0, 1])
    # here we just make sure that the vector is a unit vector.
    normal = unit_vector(normal)
    # Since the cross product is in 3d we have to take out the z coordinate.
    return(normal[:-1])
               
def reflect(my_ray, my_surface, intersect_t):
    # Finds the consequences of reflection of the given ray at point intercect_t of my_surface.
    
    # Get intersection point, normal vector and initial direction.
    intersect_point =  intersect_t
    normal_vector = make_normal(my_surface, intersect_t)
    initial_direction = my_ray.direction
    
    # Find the new direction for the emergent ray.
    new_direction = initial_direction - 2 * np.dot(initial_direction, normal_vector) * normal_vector
    return(new_direction)
               
    

def rotation_matrix(theta):
    # Defined a function that returns the matrix which is able to get amount of refraction.
    c, s = np.cos(theta), np.sin(theta)
    rotation_mat = np.array([[c, -s], [s, c]])
    return(rotation_mat)

def refract(v1, surface, n1, n2):
    # Get unit vectors.
    v1 = unit_vector(v1.direction)
    normal = make_normal(surface, 0.5)
    # Make the vector parallel to the surface.
    parallel = np.array([-normal[1], normal[0]])
    # Get the sine of the incident angle, and the incident angle.
    sin_q1 = np.abs(np.dot(v1, parallel))
    q1 = np.arcsin(sin_q1)
    # Snell's Law: get the sine of the refracted angle.
    sin_q2= (n1 / n2) * sin_q1
    # Get the refracted angle.
    q2 = np.arcsin(sin_q2)
    # Rotate by q1 - q2.
    v2 = np.matmul(rotation_matrix(q1 - q2), v1)
    return(v2)
               
def absorb_ray(ray):
    # kill a ray if it hits an absorber and print a message.
    ray.activity = False
    print('No emergent ray')
               

        
# PLOTTING

def my_plot():
    surface = medium_change([-2,0], [2,0])
    # Added a function for the surface and incident ray widgets.
    def g(surface_type, inc_x, inc_y):
        
        def param_ray_x(t, start_point, line_direction, length):
            return start_point[0] + (line_direction[0] * length * t)
        def param_ray_y(t, start_point, line_direction, length):
            return start_point[1] + (line_direction[1] * length *t)
        
        ray = Ray([inc_x, inc_y])
        # Get the x and y list of the incident ray depending on what values are given
        t_list = np.arange(0, 1, 0.001)
        start_x = - np.sqrt(8 / ((ray.direction[1]/ray.direction[0])**2 + 1))
        start_y = (ray.direction[1]/ray.direction[0]) * start_x
        my_inc_x = param_ray_x(t_list, np.array([start_x, start_y]), unit_vector(ray.direction), np.sqrt(8))
        my_inc_y = param_ray_y(t_list, np.array([start_x, start_y]), unit_vector(ray.direction), np.sqrt(8))

    
        if surface_type == 'Flat Mirror':
            # Plot the surface.
            x_list = np.arange(-2, 2, 0.001)
            y_list = np.array([0] * 4000)
            plt.plot(x_list, y_list, 'b')

           # Make the lists of x and y coordinates of the ray and plot.
            plt.plot(my_inc_x, my_inc_y, 'm')

            my_em_x = param_ray_x(t_list, np.array([0, 0]), reflect(ray, surface.param_func, 0.5), np.sqrt(8))
            my_em_y = param_ray_y(t_list, np.array([0, 0]), reflect(ray, surface.param_func, 0.5), np.sqrt(8))
            
            plt.plot(my_em_x, my_em_y, 'g')
            print('The direction of the emergent ray is', reflect(ray, surface.param_func, 0.5))

        elif surface_type == 'Medium Change':
            # Added widgets for the indices of refraction.
            def f(n_1, n_2):
                # Plot the surface.
                x_list = np.arange(-2, 2, 0.001)
                y_list = np.array([0] * 4000)
                plt.plot(x_list, y_list, 'b')

                # Make the lists of x and y coordinates for the ray and plot.
                plt.plot(my_inc_x, my_inc_y, 'm')
                
                my_em_x = param_ray_x(t_list, np.array([0, 0]), refract(ray, surface.param_func, n_1, n_2), np.sqrt(8))
                my_em_y = param_ray_y(t_list, np.array([0, 0]), refract(ray, surface.param_func, n_1, n_2), np.sqrt(8))
                plt.plot(my_em_x, my_em_y, 'g')
                print('The direction of the emergent ray is', refract(ray, surface.param_func, n_1, n_2))

            interact(f, n_1 = widgets.FloatSlider(value=1.0, min=0.10, max=1.00, step=0.01, description='n_1:',
                                                  disabled=False, continuous_update=False, orientation='horizontal',
                                                  readout=True),
                    n_2 = widgets.FloatSlider(value=1.0, min=1.00, max=4.00, step=0.01, description='n_2:',
                                              disabled=False, continuous_update=False, orientation='horizontal',
                                              readout=True))

        elif surface_type == 'Absorber':
            # Plot the surface
            x_list = np.arange(-2, 2, 0.001)
            y_list = np.array([0] * 4000)
            plt.plot(x_list, y_list, 'b')

            # make the plot
            plt.plot(my_inc_x, my_inc_y, 'm')
            absorb_ray(ray)
   
    interact(g, 
             surface_type = widgets.Dropdown(options=['Absorber', 'Flat Mirror', 'Medium Change'], value='Absorber', description='Surface:',
                                             disabled=False),
             inc_x = widgets.BoundedFloatText(value=1.0, min=1.0, max=10.0, step=1, description='x_incident:', disabled=False),
             inc_y= widgets.BoundedFloatText(value= -1.0, min= -10.0, max=10.0, step=1, description='y_incident:', disabled=False)
            )
    