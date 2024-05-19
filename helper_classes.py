import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    norm_axis = normalize(axis) 
    v = np.array([0,0,0])
    return vector - 2 * np.dot(vector, norm_axis) * norm_axis
    

## Lights


class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        
        self.direction = normalize(direction)

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self,intersection_point):
        new_ray = Ray(intersection_point, -self.direction)
        return new_ray

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        dis = np.inf
        return dis

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self,intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.direction = np.array(direction)
        #self.direction = normalize(self.direction)
        self.kc = kc
        self.kl = kl
        self.kq = kq
        


    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self, intersection):
        
        #print(type(self.direction))  # Should show <class 'numpy.ndarray'>

        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(self.position - intersection)

    def get_intensity(self, intersection):
        # Calculate the distance from the light source to the intersection
        distance = self.get_distance_from_light(intersection)
        # Distance attenuation factor
        distance_attenuation = self.intensity / (self.kc + self.kl * distance + self.kq * distance**2)

        # Check alignment of the light direction with the direction to the intersection
        direction_to_intersection = normalize(self.get_light_ray(intersection).direction)
        cos_theta = np.dot(-self.direction, direction_to_intersection)  # Negative because the light direction is outgoing
        return self.intensity * distance_attenuation * max(cos_theta,0)
        

        
        


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        #intersections = None
        nearest_object = None
        min_distance = np.inf
        for obj in objects:
            distance, obj = obj.intersect(self)
            if distance < min_distance:
                min_distance = distance
                nearest_object = obj
        return nearest_object, min_distance


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess 
        self.reflection = reflection


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = np.dot(v, self.normal) / (np.dot(self.normal, ray.direction) + 1e-6)
        if t > 0:
            return t, self
        else:
            return np.inf, None 


class Triangle(Object3D):
    """
        C
        /\
       /  \
    A /____\ B

    The fornt face of the triangle is A -> B -> C.
    
    """
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal()
        

    # computes normal to the trainagle surface. Pay attention to its direction!
    def compute_normal(self):
        AB = self.b - self.a
        AC = self.c - self.a
        normal = np.cross(AB, AC)
        return normalize(normal)
        
    def intersect(self, ray: Ray):
        firts_edge = self.b - self.a
        secomd_edge = self.c - self.a
        h = np.cross(ray.direction, secomd_edge)
        a = np.dot(firts_edge, h)

        if a > -1e-6 and a < 1e-6:
            return np.inf, None
        
        f = 1.0 / a
        s = ray.origin - self.a
        u = f * np.dot(s, h)

        if u < 0 or u > 1:
            return np.inf, None
        q = np.cross(s, firts_edge)
        v = f * np.dot(ray.direction, q)

        if v < 0 or u + v > 1:
            return np.inf, None
        
        t = f * np.dot(secomd_edge, q)
        if t > 1e-6:
            return t, self
        else:
            return np.inf, None
        


        
        

class Pyramid(Object3D):
    """     
            D
            /\*\
           /==\**\
         /======\***\
       /==========\***\
     /==============\****\
   /==================\*****\
A /&&&&&&&&&&&&&&&&&&&&\ B &&&/ C
   \==================/****/
     \==============/****/
       \==========/****/
         \======/***/
           \==/**/
            \/*/
             E 
    
    Similar to Traingle, every from face of the diamond's faces are:
        A -> B -> D
        B -> C -> D
        A -> C -> B
        E -> B -> A
        E -> C -> B
        C -> E -> A
    """
    def __init__(self, v_list):
        self.v_list = v_list
        self.triangle_list = self.create_triangle_list()


    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection

    def create_triangle_list(self):
        l = []
        t_idx = [
                [0,1,3],
                [1,2,3],
                [0,3,2],
                 [4,1,0],
                 [4,2,1],
                 [2,4,0]]
        
        l = [Triangle(self.v_list[i[0]], self.v_list[i[1]], self.v_list[i[2]]) for i in t_idx]
        return l
        

    def apply_materials_to_triangles(self):
        for t in self.triangle_list:
            t.set_material(self.ambient, self.diffuse, self.specular, self.shininess, self.reflection)
        

    def intersect(self, ray: Ray):
        min_distance = np.inf
        nearest_obj = None
        for t in self.triangle_list:
            distance, obj = t.intersect(ray)
            if distance < min_distance:
                min_distance = distance
                nearest_obj = obj
        return min_distance, nearest_obj
       

class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        L = self.center - ray.origin
        tca = np.dot(L, ray.direction)
        if tca < 0:
            return np.inf, None
        d2 = np.dot(L, L) - tca * tca
        r2 = self.radius ** 2
        if d2 > r2:
            return np.inf, None
        thc = np.sqrt(r2 - d2)
        t0 = tca - thc
        t1 = tca + thc
        if t0 > t1:
            t0, t1 = t1, t0
        if t0 < 0:
            t0 = t1
            if t0 < 0:
                return np.inf, None
        return t0, self
    
        


class CustomTransparentSphere(Sphere):
    def __init__(self, center, radius, ambient, diffuse, specular, shininess, reflection, refractive_index, transparency):
        super().__init__(center, radius)
        self.ambient = np.array(ambient)
        self.diffuse = np.array(diffuse)
        self.specular = np.array(specular)
        self.shininess = shininess
        self.reflection = reflection
        self.refractive_index = refractive_index
        self.transparency = transparency

    def normal_at(self, point):
        return normalize(point - self.center)

    def calculate_refraction(self, incoming_ray, hit_point):
        normal = self.normal_at(hit_point)
        cos_theta_i = -np.dot(incoming_ray.direction, normal)
        eta_i = 1.0  
        eta_t = self.refractive_index

        if cos_theta_i < 0:
            normal = -normal
            cos_theta_i = -cos_theta_i
            eta_i, eta_t = eta_t, eta_i

        eta = eta_i / eta_t
        sin_theta_t2 = eta ** 2 * (1 - cos_theta_i ** 2)
        if sin_theta_t2 > 1:
            return None  

        cos_theta_t = np.sqrt(1 - sin_theta_t2)
        refracted_direction = eta * incoming_ray.direction + (eta * cos_theta_i - cos_theta_t) * normal
        return Ray(hit_point - normal * 0.001, refracted_direction)  

class CustomReflectivePolygon(Triangle):
    def __init__(self, a, b, c, ambient, diffuse, specular, shininess, reflection, refractive_index, transparency):
        super().__init__(a, b, c)
        self.ambient = np.array(ambient)
        self.diffuse = np.array(diffuse)
        self.specular = np.array(specular)
        self.shininess = shininess
        self.reflection = reflection
        self.refractive_index = refractive_index
        self.transparency = transparency

    def normal_at(self, point):
        return self.normal  

    def calculate_refraction(self, incoming_ray, hit_point):
        normal = self.normal_at(hit_point)
        cos_theta_i = -np.dot(incoming_ray.direction, normal)
        eta_i = 1.0  
        eta_t = self.refractive_index

        if cos_theta_i < 0:
            normal = -normal
            cos_theta_i = -cos_theta_i
            eta_i, eta_t = eta_t, eta_i

        eta = eta_i / eta_t
        sin_theta_t2 = eta ** 2 * (1 - cos_theta_i ** 2)
        if sin_theta_t2 > 1:
            return None  

        cos_theta_t = np.sqrt(1 - sin_theta_t2)
        refracted_direction = eta * incoming_ray.direction + (eta * cos_theta_i - cos_theta_t) * normal
        return Ray(hit_point - normal * 0.001, refracted_direction)  