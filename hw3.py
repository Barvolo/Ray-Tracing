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
    
    # Lighting setup
    light_right = SpotLight(intensity= np.array([0.8, 1.2, 0.8]),position=np.array([2, 2, 2]), direction=([0,0,1]),
                    kc=0.1,kl=0.1,kq=0.1)
    light_left = PointLight(intensity=np.array([1.2, 0.8, 0.8]), position=np.array([-2, 2, 2]), kc=0.1, kl=0.05, kq=0.01)
    #light_right = PointLight(intensity=np.array([0.8, 1.2, 0.8]), position=np.array([2, 2, 2]), kc=0.1, kl=0.05, kq=0.01)
    
    # Objects setup
    # Main reflective sphere with high shininess to act as a central focal point
    sphere_main = Sphere(np.array([0, -0.5, -3]), 1)
    sphere_main.set_material(np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([0.8, 0.8, 0.8]), 500, 0.9)

    # Smaller colorful spheres
    sphere_small1 = Sphere(np.array([1.5, 0, -2]), 0.3)
    sphere_small1.set_material(np.array([0.9, 0.1, 0.1]), np.array([0.9, 0.1, 0.1]), np.array([0.6, 0.6, 0.6]), 300, 0.7)

    sphere_small2 = Sphere(np.array([-1.5, 0, -2]), 0.3)
    sphere_small2.set_material(np.array([0.1, 0.1, 0.9]), np.array([0.1, 0.1, 0.9]), np.array([0.6, 0.6, 0.6]), 300, 0.7)


    
    # Pyramid with defined vertices
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
    # Ground plane with a subtle texture
    plane = Plane(np.array([0, 1, 0]), np.array([0, -1.5, 0]))
    plane.set_material(np.array([0.3, 0.3, 0.3]), np.array([0.3, 0.3, 0.3]), np.array([0.5, 0.5, 0.5]), 1000, 0.1)

    # Collect lights and objects
    lights = [light_left, light_right]
    objects = [sphere_main, sphere_small1, sphere_small2, pyramid, plane]

    return camera, lights, objects
