from helper_classes import *
import matplotlib.pyplot as plt

def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)
            color = np.zeros(3)
            min_object, min_distance = ray.nearest_intersected_object(objects)
            color = color if min_object is None else calculate_color(ray, min_object, min_distance, objects, lights, ambient, max_depth, 1)
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image


def calculate_color(ray, obj, distance, objects, lights, ambient, depth, level):
    color = np.zeros(3)
    if isinstance(obj, Sphere):
        normal = normalize((ray.origin + ray.direction * distance) - obj.center)
    else:
        #print('the object is :', obj.__class__.__name__)
        normal = obj.normal  
    intersection = ray.origin + ray.direction * distance + 1e-4 * normal  
    color += (ambient * obj.ambient).astype(np.float64)
    
    for light in lights:
        light_ray = light.get_light_ray(intersection)
        light_distance = light.get_distance_from_light(intersection)
        shadow_ray = Ray(intersection, light_ray.direction) 
        shadow_object, shadow_distance = shadow_ray.nearest_intersected_object(objects)
        if shadow_object is None or shadow_distance > light_distance:
            # Diffuse component
            max_dot = max(np.dot(normal, light_ray.direction), 0)
            color += obj.diffuse * light.get_intensity(intersection) * max_dot
            # Specular component
            viewer_direction = normalize(-ray.direction)
            reflection = reflected(light_ray.direction, normal)
            specular_intensity = np.dot(viewer_direction, reflection) ** obj.shininess
            color += obj.specular * light.get_intensity(intersection) * specular_intensity

    if level < depth:
        reflected_direction = reflected(ray.direction, normal)
        reflection_ray = Ray(intersection, reflected_direction)
        reflection_object, reflection_distance = reflection_ray.nearest_intersected_object(objects)
        if reflection_object is not None:
            reflection_color = calculate_color(reflection_ray, reflection_object, reflection_distance, objects, lights, ambient, depth, level + 1)
            color += obj.reflection * reflection_color

    return color





def your_own_scene():
        # Camera setup
    camera = np.array([0, 0, 1])

    # Adjusting lighting setup so the light comes directly from the camera's perspective
    light_right = SpotLight(intensity=np.array([0.8, 1.2, 0.8]), position=np.array([0, 0, 1]), direction=np.array([0, 0, -1]), kc=0.1, kl=0.1, kq=0.1)
    light_left = PointLight(intensity=np.array([1, 1, 1]), position=np.array([0, 0, 1]), kc=0.1, kl=0.1, kq=0.1)

    # Objects setup
    sphere_main = Sphere(np.array([0, -0.5, -3]), 1)
    sphere_main.set_material(np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([0.8, 0.8, 0.8]), 500, 0.9)

    sphere_small1 = Sphere(np.array([1.5, 0, -2]), 0.3)
    sphere_small1.set_material(np.array([0.9, 0.1, 0.1]), np.array([0.9, 0.1, 0.1]), np.array([0.6, 0.6, 0.6]), 300, 0.7)

    sphere_small2 = Sphere(np.array([-1.5, 0, -2]), 0.3)
    sphere_small2.set_material(np.array([0.1, 0.1, 0.9]), np.array([0.1, 0.1, 0.9]), np.array([0.6, 0.6, 0.6]), 300, 0.7)

    v_list = np.array([
        [0, 0, -3],     # Base center
        [1, 0, -2],     # Front right
        [-1, 0, -2],    # Front left
        [0, 1, -2],     # Back middle
        [0, 0.5, -1]    # Apex
    ])
    pyramid = Pyramid(v_list)
    pyramid.set_material(np.array([0.5, 0.3, 0.7]), np.array([0.5, 0.3, 0.7]), np.array([0.5, 0.5, 0.5]), 100, 0.5)
    pyramid.apply_materials_to_triangles()

    plane = Plane(np.array([0, 1, 0]), np.array([0, -1.5, 0]))
    plane.set_material(np.array([0.3, 0.3, 0.3]), np.array([0.3, 0.3, 0.3]), np.array([0.5, 0.5, 0.5]), 1000, 0.1)

    lights = [light_left, light_right]
    objects = [sphere_main, sphere_small1, sphere_small2, pyramid, plane]

    return camera, lights, objects





def render_scene_with_refraction(camera, ambient_light, lights, objects, screen_size, max_depth):
    width, height = screen_size
    aspect_ratio = float(width) / height
    screen = (-1, 1 / aspect_ratio, 1, -1 / aspect_ratio)

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)

            color = np.zeros(3)
            closest_obj, min_dist = ray.nearest_intersected_object(objects)
            if closest_obj is not None:
                color = compute_color(closest_obj, ray, min_dist, ambient_light, lights, objects, max_depth, 1)

            image[i, j] = np.clip(color, 0, 1)

    return image

def compute_color(obj, ray, min_distance, ambient_light, lights, objects, max_depth, current_depth):
    base_color = (obj.ambient * ambient_light).astype(np.float64)
    hit_point = ray.origin + min_distance * ray.direction
    normal = obj.normal_at(hit_point) if isinstance(obj, Sphere) else obj.normal  
    hit_point += 0.001 * normal  # Offset to prevent self-intersection

    for light in lights:
        light_dir = normalize(light.position - hit_point)
        light_dist = light.get_distance_from_light(hit_point)
        shadow_ray = Ray(hit_point, light_dir)
        shadow_obj, shadow_dist = shadow_ray.nearest_intersected_object(objects)
        if shadow_obj is None or shadow_dist > light_dist:
            diffuse_intensity = max(np.dot(normal, light_dir), 0)
            base_color += obj.diffuse * light.get_intensity(hit_point) * diffuse_intensity
            half_vector = normalize(light_dir - ray.direction)
            specular_intensity = max(np.dot(normal, half_vector), 0) ** obj.shininess
            base_color += obj.specular * light.get_intensity(hit_point) * specular_intensity

    if current_depth < max_depth:
        reflection_direction = reflect(ray.direction, normal)
        reflection_ray = Ray(hit_point, reflection_direction)
        reflection_obj, reflection_dist = reflection_ray.nearest_intersected_object(objects)
        if reflection_obj is not None:
            reflection_color = compute_color(reflection_obj, reflection_ray, reflection_dist, ambient_light, lights, objects, max_depth, current_depth + 1)
            base_color += obj.reflection * reflection_color
        
        if obj.transparency > 0:
            refraction_ray = obj.calculate_refraction(ray, hit_point)
            if refraction_ray is not None:
                refraction_obj, refraction_dist = refraction_ray.nearest_intersected_object(objects)
                if refraction_obj is not None:
                    refraction_color = compute_color(refraction_obj, refraction_ray, refraction_dist, ambient_light, lights, objects, max_depth, current_depth + 1)
                    base_color += obj.transparency * refraction_color

    return np.clip(base_color, 0, 1)

def reflect(direction, normal):
    return direction - 2 * np.dot(direction, normal) * normal

def calculate_refraction(direction, normal, eta_i, eta_t):
    n = normalize(normal)
    i = normalize(direction)
    cos_i = -np.dot(n, i)
    sin_t2 = (eta_i / eta_t) ** 2 * (1 - cos_i ** 2)
    if sin_t2 > 1:
        return None
    cos_t = np.sqrt(1 - sin_t2)
    return (eta_i / eta_t) * i + (eta_i / eta_t * cos_i - cos_t) * n



def setup_scene_with_refraction():
    camera = np.array([0, 0, 1])

    # Lighting setup
    light = PointLight(
        intensity=np.array([1, 1, 1]), 
        position=np.array([5, 5, 5]), 
        kc=0.1, 
        kl=0.01, 
        kq=0.01
    )

    # Objects setup
    glass_sphere = CustomTransparentSphere(
        center=np.array([0, 0, -5]), 
        radius=1,
        ambient=[0.1, 0.1, 0.1], 
        diffuse=[0.1, 0.1, 0.1], 
        specular=[1, 1, 1], 
        shininess=100, 
        reflection=0.1, 
        refractive_index=1.5, 
        transparency=0.95  # High transparency
    )

    background_plane = CustomReflectivePolygon(
        a=np.array([-10, -10, -10]), 
        b=np.array([10, -10, -10]), 
        c=np.array([0, 10, -10]),
        ambient=[0.8, 0.2, 0.2],  # Brightly colored background
        diffuse=[0.8, 0.2, 0.2], 
        specular=[1, 1, 1], 
        shininess=100, 
        reflection=0.1,
        refractive_index=1.0, 
        transparency=0.0
    )

    objects = [glass_sphere, background_plane]
    lights = [light]

    return camera, lights, objects

def setup_scene_without_refraction():
    camera = np.array([0, 0, 1])

    # Lighting setup
    light = PointLight(
        intensity=np.array([1, 1, 1]), 
        position=np.array([5, 5, 5]), 
        kc=0.1, 
        kl=0.01, 
        kq=0.01
    )

    # Objects setup
    opaque_sphere = Sphere(
        center=np.array([0, 0, -5]), 
        radius=1
    )
    opaque_sphere.set_material(
        ambient=[0.1, 0.1, 0.1], 
        diffuse=[0.1, 0.1, 0.1], 
        specular=[1, 1, 1], 
        shininess=100, 
        reflection=0.1
    )

    background_triangle = Triangle(
        a=np.array([-10, -10, -10]), 
        b=np.array([10, -10, -10]), 
        c=np.array([0, 10, -10])
    )
    background_triangle.set_material(
        ambient=[0.8, 0.2, 0.2],  # Brightly colored background
        diffuse=[0.8, 0.2, 0.2], 
        specular=[1, 1, 1], 
        shininess=100, 
        reflection=0.1
    )

    objects = [opaque_sphere, background_triangle]
    lights = [light]

    return camera, lights, objects
