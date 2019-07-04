""" Reference: https://github.com/Maluuba/FigureQA/blob/master/figureqa/generation/source_data_generation.py """

import numpy as np
import randomcolor
rand_color = randomcolor.RandomColor()

from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

#--------------------------------------------------------------------------
# Load anchor colors for generating colors based on training/testing phase
#--------------------------------------------------------------------------
train_colors = []
with open('resources/color_split1.txt', 'r') as file:
    for line in file:
        k, v = line.strip().split(', ')
        train_colors.append(v)
    
test_colors = []
with open('resources/color_split2.txt', 'r') as file:
    for line in file:
        k, v = line.strip().split(', ')
        test_colors.append(v)
        
def hex2rgb(h):
    h = h.lstrip('#')
    try:
        return tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))
    except:
        return (0, 0, 0)
    
def rgb2hex(R, G, B):
    return '#{:02x}{:02x}{:02x}'.format(R, G, B)

def delta_e(color1, color2):
    """ Compute human percpetron color distance.
    Args:
        `color1` and `color2` (tuple): (R, G, B)
    Returns:
        `delta_e`: Human perceptron color distance.
    """
    
    if type(color1) == str:
        color1 = hex2rgb(color1)
    if type(color2) == str:
        color2 = hex2rgb(color2)
    
    color1_rgb = sRGBColor(*color1)
    color2_rgb = sRGBColor(*color2)

    # Convert from RGB to Lab Color Space
    color1_lab = convert_color(color1_rgb, LabColor)
    color2_lab = convert_color(color2_rgb, LabColor)

    # Find the color difference
    delta_e = delta_e_cie2000(color1_lab, color2_lab)
    return delta_e
        
def is_valid_color(color, phase='train'):
    """ Check whether given color is close to train/test anchor colors. """

    train_deltas, test_deltas = [], []
        
    for train_anchor_color, test_anchor_color in zip(train_colors, test_colors):
        train_deltas.append(delta_e(color, train_anchor_color))
        test_deltas.append(delta_e(color, test_anchor_color))
                
    if phase == 'train' and min(train_deltas) < min(test_deltas):
        return True
    
    if phase == 'test' and min(train_deltas) > min(test_deltas):
        return True
        """
        Create more difficult test color by setting threshold of color distance.
        if min(train_deltas) - min(test_deltas) >= 5:
            return True
        else:
            return False
        """
    
    return False

def generate_bg_color(luminosity=None, phase='train'):
    """ For generating background colors.
        Note: we use `randomcolor` package to generate foreground colors.
        Args:
        - `luminosity`: [None, 'bright', 'dark']
    """
        
    stop = False
    
    while not stop:
    
        if not luminosity:
            luminosity = np.random.choice(['bright', 'dark'])

        if luminosity == 'bright':
            rgb_min, rgb_max = 230, 255
        else:
            rgb_min, rgb_max = 0, 100

        R_bins = np.linspace(rgb_min, rgb_max, dtype='uint8')
        G_bins = np.linspace(rgb_min, rgb_max, dtype='uint8')
        B_bins = np.linspace(rgb_min, rgb_max, dtype='uint8')

        R = np.random.choice(R_bins)
        G = np.random.choice(G_bins)
        B = np.random.choice(B_bins)
        rgb_color = (R, G, B) # cast to list of tuples
        hex_color = rgb2hex(*rgb_color)
        
        if is_valid_color(hex_color, phase):
            stop = True

    return hex_color, luminosity

        
def generate_fg_colors(num_colors, hue, luminosity, phase='train'):
    """ Generate color close to <phase>_colors. 
    
        Args:
        - `hue`: If `hue` is not none, ignore train/test check.
    """

    colors = []
    stop = False
    
    if not hue:
        while not stop:
            color = rand_color.generate(count=1, hue=hue, luminosity=luminosity)[0]

            if is_valid_color(color, phase):
                colors.append(color)

            if len(colors) == num_colors:
                stop = True
    else:
        colors = rand_color.generate(count=num_colors, hue=hue, luminosity=luminosity)
        
    return colors


def generate_value(value_scale):
    if value_scale == 'linear':
        return np.random.randint(1, 10)
    elif value_scale == 'percent':
        return np.random.choice(np.arange(10, 100+10, 10), 1)[0]
    elif value_scale == 'exp':
        return np.random.uniform(0, 10**10)
    

def generate_data_by_shape(x_range, y_range, n, x_distn, shape):
    x = []

    if x_distn == "random":
        x = (x_range[1] - x_range[0]) * np.random.random(n) + x_range[0]

    elif x_distn == "linear":
        x = np.linspace(x_range[0], x_range[1], n)

    elif x_distn == "normal":
        mean = (x_range[1] - x_range[0]) * np.random.random(1) + x_range[0]
        points = (x_range[1] - x_range[0]) *  np.random.normal(0, 1/6.0, 3*n) + mean

        final_points = []
        for point in points:
            if point >= x_range[0] and point <= x_range[1]:
                final_points.append(point)
            if len(final_points) == n:
                break
        x = final_points

    x = sorted(x)
    y = []

    max_slope = (y_range[1] - y_range[0]) / float(x_range[1] - x_range[0])

    if shape == "random":
        y = (y_range[1] - y_range[0]) * np.random.random(n) + y_range[0]

    elif shape == "linear":
        # Decide slope direction randomly
        slope_direction = 1 if np.random.random() > 0.5 else -1
        offset = y_range[0] if slope_direction >= 0 else y_range[1]
        y = np.clip(slope_direction*max_slope*np.random.random()*np.array(x[:]) + offset, y_range[0], y_range[1]).tolist()

    elif shape == "linear_with_noise":
        # Decide slope direction randomly
        slope_direction = 1 if np.random.random() > 0.5 else -1
        offset = y_range[0] if slope_direction >= 0 else y_range[1]
        y = np.clip(slope_direction*max_slope*np.random.random()*np.array(x[:]) + offset, y_range[0], y_range[1]).tolist()

        # Add some noise then reclip
        noise_multiplier = 0.05 * (y_range[1] - y_range[0])
        for i in range(len(y)):
            y[i] += noise_multiplier * (2*np.random.random() - 1)

        y = np.clip(y, y_range[0], y_range[1]).tolist()
    
    elif shape == "linear_inc":
        y = np.clip(max_slope*np.random.random()*np.array(x[:]) + y_range[0], y_range[0], y_range[1]).tolist()

    elif shape == "linear_dec":
        y = np.clip(-max_slope*np.random.random()*np.array(x[:]) + y_range[1], y_range[0], y_range[1]).tolist()

    elif shape == "cluster":
        mean = (y_range[1] - y_range[0]) * np.random.random() + y_range[0]

        points = (y_range[1] - y_range[0]) *  np.random.normal(0, 1/6.0, 3*n) + mean
        
        final_points = []
        got_all_points = False

        while True:

            points = (y_range[1] - y_range[0]) *  np.random.normal(0, 1/6.0, n) + mean

            for point in points:

                if point >= y_range[0] and point <= y_range[1]:
                    final_points.append(point)

                if len(final_points) == n:
                    got_all_points = True
                    break

            if got_all_points:
                break

        y = final_points

    elif shape == "quadratic":
        # Use vertex form: y = a(x-h)^2 + k
        h = (x_range[1] - x_range[0])/2 * np.random.random() + x_range[0]
        k = (y_range[1] - y_range[0])/2 * np.random.random() + y_range[0]

        dist_from_mid = np.abs((y_range[1] - y_range[0])/2 + y_range[0])

        # Decide a direction based on k
        if k < (y_range[1] - y_range[0])/2 + y_range[0]:
            a = -1 * dist_from_mid
        else:
            a = 1 * dist_from_mid

        a *= np.random.random()*0.00005
        y = np.clip(np.array([a*(xx-h)**2 + k for xx in x]), y_range[0], y_range[1]).tolist()

    return x, y

def pick_random_int_range(the_range):
    range_start, range_end = the_range
    start = np.random.randint(range_start, range_end - 1)
    end = np.random.randint(start + 1, range_end)
    return start, end

# Data generation functions
def generate_scatter_data_continuous(x_range, y_range, x_distns, shapes, n_classes, n_points, class_distn_mean=0, fix_x_range=False, fix_y_range=False):
    if not fix_x_range:
        x_range = pick_random_int_range(x_range)
    if not fix_y_range:
        y_range = pick_random_int_range(y_range)

    point_sets = []
    for i in range(0, n_classes):
        x_distn = np.random.choice(x_distns)
        shape = np.random.choice(shapes)

        x, y = generate_data_by_shape(x_range, y_range, n_points, x_distn, shape)

        if type(x) != type([]):
            x = x.tolist()
        if type(y) != type([]):
            y = y.tolist()

        point_sets.append({ 'class': i, 'x': x, 'y': y })

    return {'type': "scatter_base", 'data': point_sets, 'n_points': n_points, 'n_classes': n_classes}