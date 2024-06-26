import numpy as np
import math
from abc import ABC, abstractmethod
from PIL import Image

c = 1

speedCap = 0.9


Gamma = 0.80
IntensityMax = 255

def freq_to_rgb(freq):
    return wave_length_to_rgb(c/freq)


def wave_length_to_rgb(wavelength):
    """Convert a wavelength in the range 380-780 nm to an RGB color."""
    factor = 0.0
    red, green, blue = 0.0, 0.0, 0.0

    if wavelength<380:
        red = 0.0
        green = 0.0
        blue = 1.0

    if 380 <= wavelength < 440:
        red = -(wavelength - 440) / (440 - 380)
        green = 0.0
        blue = 1.0
    elif 440 <= wavelength < 490:
        red = 0.0
        green = (wavelength - 440) / (490 - 440)
        blue = 1.0
    elif 490 <= wavelength < 510:
        red = 0.0
        green = 1.0
        blue = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength < 580:
        red = (wavelength - 510) / (580 - 510)
        green = 1.0
        blue = 0.0
    elif 580 <= wavelength < 645:
        red = 1.0
        green = -(wavelength - 645) / (645 - 580)
        blue = 0.0
    elif 645 <= wavelength:
        red = 1.0
        green = 0.0
        blue = 0.0
    factor = 1.0

    rgb = [
        0 if red == 0.0 else int(round(IntensityMax * (red * factor) ** Gamma)),
        0 if green == 0.0 else int(round(IntensityMax * (green * factor) ** Gamma)),
        0 if blue == 0.0 else int(round(IntensityMax * (blue * factor) ** Gamma)),
    ]
    rgb = np.array(rgb)
    factor = 0.3
    return (1 - factor) * rgb + factor

class FourVector:
    def __init__(self, t, x, y, z):
        self.ct = c*t
        self.x = x
        self.y = y
        self.z = z
        self.vector = np.array([c*t, x, y, z])
        self.three_vector = np.array([x, y, z])

    def squared_magnitude(self):
        t, x, y, z = self.vector
        return (t**2 - x**2 - y**2 - z**2)

    def magnitude(self):
        t, x, y, z = self.vector
        return np.sqrt(t**2 - x**2 - y**2 - z**2)

    def __add__(self, other):
        return FourVector(self.ct+other.ct,self.x+other.x,self.y+other.y,self.z+other.z)
    
    def __sub__(self, other):
        return FourVector(self.ct-other.ct,self.x-other.x,self.y-other.y,self.z-other.z)

    def __str__(self):
        return f"({self.vector[0]}, {self.vector[1]}, {self.vector[2]}, {self.vector[3]})"
    
    def lorentz_boost(self, velocity):

        #return self
        if -0.001<=np.linalg.norm(velocity)<=0.001:
            return self
        
        gamma = 1/np.sqrt(1-(np.linalg.norm(velocity)/c)**2)
        n = velocity/np.linalg.norm(velocity)
        r = np.array([self.vector[1], self.vector[2], self.vector[3]])
        ct_prime = gamma*(self.ct-np.dot(velocity,r)/c)
        r_prime = r + (gamma-1)*(np.dot(r,n)*n) - gamma*self.ct*velocity/c
        return type(self)(ct_prime/c, r_prime[0],r_prime[1],r_prime[2])

class FourVelocity(FourVector):
    def __init__(self, t, x, y, z):
        self.three_velocity = np.array([x,y,z])
        self.three_magnitude = np.linalg.norm(self.three_velocity)
        if (self.three_magnitude>=c):
            self.three_velocity=(self.three_velocity/self.three_magnitude)*speedCap*c
            self.three_magnitude = np.linalg.norm(self.three_velocity)
        gamma = 1/np.sqrt(1-(self.three_magnitude/c)**2)

        super().__init__(gamma, self.three_velocity[0], self.three_velocity[1], self.three_velocity[2])
    

class FourVelocity_lightlike(FourVector):
    def __init__(self, t, x, y, z):
        three_magnitude = x**2 + y**2 + z**2
        if three_magnitude == 0:
            raise ValueError("The spatial components cannot all be zero for a lightlike 4-vector.")
        # Calculate the scaling constant
        constant = np.sqrt((c*t)**2 / three_magnitude)
        # Scale the spatial components
        self.three_velocity = np.array([x,y,z])*constant
        super().__init__(t, x * constant, y * constant, z * constant)

class Object:
    def __init__(self, four_velocity: FourVelocity, four_position: FourVector):
        self.four_velocity = four_velocity
        self.four_position = four_position

class Material:
    def __init__(self, reflectance, diffusion, freq,color,lighting=0):
        self.reflectance = reflectance
        self.diffusion = diffusion
        self.freq = freq
        self.color = np.array(color)
        self.lighting=lighting
        return
    def scatter(self, ray, normal):
        ray = ray/np.linalg.norm(ray)
        direction = np.random.rand(3)
        direction = direction/np.linalg.norm(direction)
        if np.dot(direction,normal)<0:
            direction = -direction
        direction += normal
        reflected = ray - 2*np.dot(ray,normal)*normal
        
        direction += normal
        reflected = ray - 2*np.dot(ray,normal)*normal
        reflected += self.diffusion*direction
        return FourVelocity_lightlike(self.freq,reflected[0],reflected[1],reflected[2])
        
    
class Objects():
    def __init__(self,objectList):
        self.objects=objectList

    def hit(self, light_velocity: FourVelocity_lightlike, observer_position: FourVector):

        vectorized_hit = np.vectorize(lambda obj: obj.intersection(light_velocity, observer_position))

        hits = vectorized_hit(self.objects)
        none_mask = hits == None
        hits = hits[~none_mask]

        if not np.any(hits):
            return None
        
        vectorized_t = np.vectorize(lambda rec: -rec.time if rec else np.inf)
        hitvalues = vectorized_t(hits)
        
       
        min_index = np.argmin(hitvalues)
        
        return hits[min_index]
    
class Hit():
    def __init__(self, new_four_velocity_light: FourVelocity_lightlike, new_fourposition: FourVector,time, material:Material):
        self.four_velocity_light = new_four_velocity_light
        self.fourposition = new_fourposition
        self.time = time
        self.material = material

class Sphere(Object):
    def __init__(self, four_velocity: FourVelocity, four_position: FourVector, radius, material: Material):
        self.radius = radius
        self.material = material
        super().__init__(four_velocity,four_position)

    def intersection(self, light_velocity: FourVelocity_lightlike, observer_position: FourVector):
        center = self.four_position.lorentz_boost(self.four_velocity.three_velocity)
        camera = observer_position.lorentz_boost(self.four_velocity.three_velocity)
        ray = light_velocity.lorentz_boost(self.four_velocity.three_velocity)


        ray_direction = ray.three_vector/np.linalg.norm(ray.three_vector)
        
        oc = ( camera.three_vector-center.three_vector)

        a = np.dot(ray_direction, ray_direction)
        h = np.dot(oc, ray_direction)
        k = np.dot(oc, oc) - self.radius**2
        discriminant = h**2 - a*k

        
        
        if discriminant < 0:
            return None

        sqrt_discriminant = np.sqrt(discriminant)
        
        root = (-h - sqrt_discriminant) / a

        
        if not 0.01<= root <=1000:
            root = (-h + sqrt_discriminant) /a

            if not 0.01<= root <=1000:
                return None
        
        
        
        time = observer_position.ct - root
        
        point = camera.three_vector + root * ray_direction
        
        outward_normal = (point - center.three_vector) / self.radius
        
        new_fourposition = FourVector(time/c, point[0],point[1],point[2]).lorentz_boost(-self.four_velocity.three_vector)
       
        new_four_velocity_light = self.material.scatter(ray_direction, outward_normal).lorentz_boost(-self.four_velocity.three_vector)
        
        return Hit(new_four_velocity_light, new_fourposition,time,self.material)
    
class Camera:
    maxdepth = 10
    def __init__(self):
        self.image_height = 256
        self.image_width = 256

        focal_length = 1.0
        viewport_width = 2.0
        viewport_height = 2.0

        viewport_u = np.array([viewport_width,0,0])
        viewport_v = np.array([0,-viewport_height,0])

        self.pixel_delta_u = viewport_u/self.image_width
        self.pixel_delta_v = viewport_v/self.image_width

        viewport_upper_left = -np.array([0,0,focal_length])-viewport_u/2 - viewport_v/2
        pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v)
        pixel00_locations = np.tile(pixel00_loc, (self.image_height, self.image_width, 1))
        indices_i = np.repeat(np.tile(np.arange(self.image_width), (self.image_height, 1))[:, :, np.newaxis], 3, axis=2)
        indices_j = np.repeat(np.tile(np.arange(self.image_height)[:, np.newaxis],  (1,self.image_width))[:, :, np.newaxis], 3, axis=2)
        #finally setting up our camera location through this
        self.pixel_centers = pixel00_locations+indices_i*self.pixel_delta_u+ indices_j*self.pixel_delta_v
        return
    def raytrace(self, four_velocity_lightlike: FourVelocity_lightlike, four_position: FourVector, objects: Objects, depth):
        hit = objects.hit(four_velocity_lightlike,four_position)
        
        if(hit and depth>0):
            if(hit.material.lighting!=0):
                return np.array([1,1,1])
            return freq_to_rgb(hit.four_velocity_light.ct)*self.raytrace(hit.four_velocity_light,hit.fourposition, objects, depth-1)
           
        
        unit_direction = four_velocity_lightlike.three_vector/np.linalg.norm(four_velocity_lightlike.three_velocity)
        a = 0.7*(unit_direction[1] + 1.0)
        return ((1.0-a)*np.array([1.0,1.0,1.0]) + a*np.array([0.5, 0.5, 1.0]))

    def render(self, objects):
        pixel_array = np.apply_along_axis(lambda vector: self.raytrace(FourVelocity_lightlike(1,vector[0],vector[1],vector[2]), FourVector(0,0,0,0),objects,10), 2, self.pixel_centers.astype(np.float32))
        pixel_array = np.clip(pixel_array, 0, 1)
        return np.sqrt(pixel_array)
    

print(FourVector(1,1,0,0).lorentz_boost(np.array([0.5,0,0])).lorentz_boost(np.array([-0.5,0,0])))

raytracer = Camera()

image_array = raytracer.render(Objects([Sphere(FourVelocity(1,0,0,0.5), FourVector(0,0,0,-7), 3, Material(1,0,c/480,[0.7,0.4,0.2])),Sphere(FourVelocity(1,0,0,-0.3), FourVector(0,3,-1,-4), 2, Material(1,0,c/720,[0.1,0.36,0.7])),Sphere(FourVelocity(1,0,0,0), FourVector(0,0,-1003.5,-1), 1000, Material(1,0,c/800,[0.9,0.8,0.8]))]))
image_array*=255#, Sphere(FourVelocity(1,0,0,0), FourVector(0,0,-130,-1), 100, Material(1,1,c/720,[0.9,0.4,0.8]))
image = Image.fromarray(image_array.astype(np.uint8))

# Save the image
image.save('output_image.jpg')



image.show()


