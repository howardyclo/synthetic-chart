import random
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_utils import generate_bg_color, generate_fg_colors, generate_value

#------------------------------------------
# Load vocabs
#------------------------------------------
train_vocab, test_vocab = [], []
with open('resources/train_vocab.txt', 'r') as file:
    for line in file:
        train_vocab.append(line.strip())

with open('resources/test_vocab.txt', 'r') as file:
    for line in file:
        test_vocab.append(line.strip())
        
#------------------------------------------
# Load legend position splits
#------------------------------------------
with open('resources/legend_position_splits.pkl', 'rb') as file:
    legend_position_splits = pickle.load(file)

def generate_bar(category_range=(2, 6), 
                 series_range=(1, 5),
                 value_scale=None,
                 orientation='vertical', # or 'horizontal'
                 has_texture=None,
                 has_bar_border=None,
                 has_bar_space=None,
                 has_label_rotation=None,
                 fg_colors=None,
                 bg_color=None,
                 fg_hue=None,
                 fg_luminosity=None,
                 bg_luminosity=None,
                 legend_position='',
                 legend_position_offset='random',
                 filepath=None,
                 factors=None,
                 phase='train'):

    if not factors:
        factors = ['color', 'vocab', 'legend_position']
    
    vocab = train_vocab
    if 'vocab' in factors and phase == 'test':
        vocab = test_vocab
            
    # Define choices
    if bg_luminosity == None:
        AXES_STYLE_CHOICES = ['whitegrid', 'white', 'darkgrid', 'dark', 'ticks']
    elif bg_luminosity == 'dark':
        AXES_STYLE_CHOICES = ['darkgrid', 'dark', 'ticks']
    else:
        AXES_STYLE_CHOICES = ['whitegrid', 'white']
        
    if bg_color:
        AXES_STYLE_CHOICES = ['darkgrid', 'dark', 'ticks']

    VALUE_SCALE_CHOICES     = ['linear', 'percent', 'exp'] # linear [0-1]; percent [0-100]; exp [1-10^10]
    ORIENTATION_CHOICES     = ['vertical', 'horizontal']
    TEXTURE_CHOICES         = ['/', '\\', '|', '-', '+', 'x', '.']
    LEGEND_POSITION_CHOICES = [None, 'top', 'right', 'left', 'bottom']
    
    # Format arguments
    if value_scale == None:        value_scale = np.random.choice(VALUE_SCALE_CHOICES)
    if orientation == None:        orientation = np.random.choice(ORIENTATION_CHOICES)
    if has_texture == None:        has_texture = np.random.choice([True, False])
    if has_bar_border == None:     has_bar_border = np.random.choice([True, False])
    if has_bar_space == None:      has_bar_space = np.random.choice([True, False])
    if has_label_rotation == None: has_label_rotation = np.random.choice([True, False])

    #----------------------------------------------------------------------------------------
    # Random generate parameters.
    #----------------------------------------------------------------------------------------
    # Random generate axes style.
    axes_style = np.random.choice(AXES_STYLE_CHOICES)
    sns.set(style=axes_style) # Set axis style. (This must be placed before plotting.)
    
    # Random generate axes background colors.
    if not bg_color:
        if 'color' in factors:
            axes_facecolor, bg_luminosity = generate_bg_color(bg_luminosity, phase) if axes_style in ['darkgrid', 'dark', 'ticks'] else ('#ffffff', 'bright')
        else:
            axes_facecolor, bg_luminosity = generate_bg_color(bg_luminosity, 'train') if axes_style in ['darkgrid', 'dark', 'ticks'] else ('#ffffff', 'bright')
    else:
        axes_facecolor = bg_color
        bg_luminosity = np.random.choice(['bright', 'dark']) # no need?
    
    # Random generate legend position.
    if legend_position == '':
        legend_position = np.random.choice(LEGEND_POSITION_CHOICES)
    
    # Random generate chart parameters.
    if not fg_colors:
        num_categories = random.randint(*category_range)
        num_series = random.randint(*series_range)
    else:
        num_categories = random.randint(*category_range)
        num_series = len(fg_colors)
        
    # Random generate colors and shuffle color palette order.
    num_colors = num_series if num_series > 1 else num_categories
    
    if not fg_colors:
        if 'color' in factors:
            hex_colors = generate_fg_colors(num_colors, fg_hue, fg_luminosity, phase)
        else:
            hex_colors = generate_fg_colors(num_colors, fg_hue, fg_luminosity, 'train')
    else:
        hex_colors = fg_colors

    # Random generate data and labels.
    data = []
    category_title = np.random.choice(vocab)
    category_labels = [np.random.choice(vocab) for _ in range(num_categories)]
    value_title = np.random.choice(vocab)
    legend_labels = [np.random.choice(vocab) for _ in range(num_series)] if num_series > 1 else []
    
    # Adjust style parameters based on sampled chart parameters.
    if orientation == 'horizontal':
        has_label_rotation = False
    if num_categories > len(TEXTURE_CHOICES):
        has_texture = False
    if num_series == 1:
        legend_position = None
        
    # Create dataframe from generated data and labels.        
    for i_categories in range(num_categories):
        category_label = category_labels[i_categories]
        for i_series in range(num_series):
            data.append({category_title: category_label,
                         value_title: generate_value(value_scale),
                         'series': i_series+1})
    df = pd.DataFrame(data)
    
    #----------------------------------------------------------------------------------------
    # Plot
    #----------------------------------------------------------------------------------------
    x, y = (category_title, value_title) if orientation == 'vertical' else (value_title, category_title)
    ax = sns.barplot(x=x, y=y, hue=None if num_series == 1 else 'series',
                     data=df, palette=sns.color_palette(hex_colors))
    
    #----------------------------------------------------------------------------------------
    # Set styles.
    #----------------------------------------------------------------------------------------
    if orientation == 'vertical' and value_scale == 'exp':
        ax.set_yscale('log')
    elif orientation == 'horizontal' and value_scale == 'exp':
        ax.set_xscale('log')
    
    # Set background color.
    ax.set_facecolor(axes_facecolor)
    
    # Set texture.
    # https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html
    if has_texture:
        try:
            texture_choices = TEXTURE_CHOICES.copy()
            random.shuffle(texture_choices)
            num_textures = num_series if num_series > 1 else num_categories

            for i, patch in enumerate(ax.patches):
                # First iterate same colors first, then another colors and vice versa.
                # If 'i // num_categories', then same texture in same color.
                # If 'i % num_categories' , then same texture in same category.
                i_ = i if num_series == 1 else i // num_categories
                # Set texture to each bar.
                patch.set_hatch(texture_choices[i_])
        except:
            has_texture = False
    elif not has_texture:
        texture_choices = []
        num_textures = 0
        
    # Set label rotations. (Vertical bar)
    label_rotation =  np.random.choice(np.arange(-45, 90+5, 5), 1)[0] if has_label_rotation else 0

    # Set axis scale
    if orientation == 'vertical':
        for xticklabel in ax.get_xticklabels():
            xticklabel.set_rotation(label_rotation)
    else:
        for yticklabel in ax.get_yticklabels():
            yticklabel.set_rotation(label_rotation)
        
    # Set bar width and diff.
    bar_scale = random.uniform(0.5, 1) if has_bar_space else 0.0
    if bar_scale:
        for i, patch in enumerate(ax.patches):            
            # Re-center the bar.
            if orientation == 'vertical':
                current_width = patch.get_width()
                new_width = current_width * bar_scale
                bar_diff = current_width - new_width
                patch.set_width(new_width) # Reset width.
                patch.set_x(patch.get_x() + bar_diff * .5)
            else:
                current_height = patch.get_height()
                new_height = current_height * bar_scale
                bar_diff = current_height - new_height
                patch.set_height(new_height) # Reset width.
                patch.set_y(patch.get_y() + bar_diff * .5)

    # Set bar border.
    for i, patch in enumerate(ax.patches):
        # Set border to each bar.
        if has_bar_border:
            patch.set_linewidth(1)
            patch.set_edgecolor('black')
        else:
            patch.set_linewidth(0)
                
    # Set legend style
    # Legend position: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    # Legend style   : https://jakevdp.github.io/PythonDataScienceHandbook/04.06-customizing-legends.html
    # bbox_to_anchor: (x, y, w, h)
    if legend_position:
        if legend_position == 'top':
            if legend_position_offset == 'random':
                if phase == 'train' or 'legend_position' not in factors:
                    y = np.random.choice(legend_position_splits['train']['top']['y'])
                    w = np.random.choice(legend_position_splits['train']['top']['w'])
                else:
                    y = np.random.choice(legend_position_splits['test']['top']['y'])
                    w = np.random.choice(legend_position_splits['test']['top']['w'])
            else: y, w = 1.02, 1
            legend = ax.legend(bbox_to_anchor=(0, y, w, 2.2), loc="lower center", borderaxespad=0, ncol=6)
            
        elif legend_position == 'right':
            if legend_position_offset == 'random':
                if phase == 'train' or 'legend_position' not in factors:
                    x = np.random.choice(legend_position_splits['train']['right']['x'])
                    y = np.random.choice(legend_position_splits['train']['right']['y'])
                else:
                    x = np.random.choice(legend_position_splits['test']['right']['x'])
                    y = np.random.choice(legend_position_splits['test']['right']['y'])
            else: x, y = 1.04, 0.5
            legend = ax.legend(bbox_to_anchor=(x, y), loc="center left", borderaxespad=0)
            
        elif legend_position == 'left':
            if legend_position_offset == 'random':
                if phase == 'train' or 'legend_position' not in factors:
                    x = np.random.choice(legend_position_splits['train']['left']['x'])
                    y = np.random.choice(legend_position_splits['train']['left']['y'])
                else:
                    x = np.random.choice(legend_position_splits['test']['left']['x'])
                    y = np.random.choice(legend_position_splits['test']['left']['y'])
            else: x, y = -0.5, 0.5
            legend = ax.legend(bbox_to_anchor=(x, y), loc="center left", borderaxespad=0)
            
        elif legend_position == 'bottom':
            if legend_position_offset == 'random':
                if phase == 'train' or 'legend_position' not in factors:
                    x = np.random.choice(legend_position_splits['train']['bottom']['x'])
                    y = np.random.choice(legend_position_splits['train']['bottom']['y'])
                else:
                    x = np.random.choice(legend_position_splits['test']['bottom']['x'])
                    y = np.random.choice(legend_position_splits['test']['bottom']['y'])    
            else: x, y = 0.5, -0.1
            legend = ax.legend(bbox_to_anchor=(x, y), loc="lower center", bbox_transform=plt.gcf().transFigure, ncol=6)
        
        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('white')
        
        # Set legend texts with random vocabulary
        texts = legend.get_texts()
        for i, text in enumerate(texts):
            label = legend_labels[i]
            text.set_text(label)
            
    elif num_series > 1:
        ax.legend_.remove()
        
    #----------------------------------------------------------------------------------------
    # Save
    #----------------------------------------------------------------------------------------
    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
    
    #----------------------------------------------------------------------------------------
    # Format returns
    #----------------------------------------------------------------------------------------
    style = {
        'fg_luminosity': fg_luminosity,
        'bg_luminosity': bg_luminosity,
        'num_categories': num_categories,
        'num_series': num_series,
        'value_scale': value_scale,
        'orientation': orientation,
        'hex_colors': hex_colors,
        'axes_style': axes_style,
        'axes_facecolor': axes_facecolor,
        'textures': texture_choices[:num_textures],
        'has_bar_border': has_bar_border,
        'bar_scale': bar_scale,
        'legend_labels': legend_labels,
        'legend_position': legend_position,
        'category_title': category_title,
        'category_labels': category_labels,
        'value_title': value_title,
        'label_rotation': label_rotation,
    }
    
    return df, style

def render_bar(df, style, filepath=None):
    """ This is for testing whether dataframe is correctly generated. 
        If you don't want to render texture, just set:
        `style['textures'] = None`
    """
    
    # Extract styles
    fg_luminosity = style['fg_luminosity']
    bg_luminosity = style['bg_luminosity']
    num_categories = style['num_categories']
    num_series = style['num_series']
    value_scale = style['value_scale']
    orientation = style['orientation']
    hex_colors = style['hex_colors']
    axes_style = style['axes_style']
    axes_facecolor = style['axes_facecolor']
    textures = style['textures']
    has_bar_border = style['has_bar_border']
    bar_scale = style['bar_scale']
    legend_labels = style['legend_labels']
    legend_position = style['legend_position']
    category_title = style['category_title']
    category_labels = style['category_labels']
    value_title = style['value_title']
    label_rotation = style['label_rotation']
    
    # Render axis style
    sns.set(style=axes_style)
    
    # Generate bar chart object
    x, y = (category_title, value_title) if orientation == 'vertical' else (value_title, category_title)
    ax = sns.barplot(x=x, y=y, hue=None if num_series == 1 else 'series',
                     data=df, palette=sns.color_palette(hex_colors))
    
    # Set axis scale
    if orientation == 'vertical' and value_scale == 'exp':
        ax.set_yscale('log')
    elif orientation == 'horizontal' and value_scale == 'exp':
        ax.set_xscale('log')
    
    # Set background color
    ax.set_facecolor(axes_facecolor)

    # Set textures.
    # https://matplotlib.org/api/_as_gen/matplotlib.patches.Patch.html
    if textures:
        for i, patch in enumerate(ax.patches):
            # First iterate same colors first, then another colors and vice versa.
            # If 'i // num_categories', then same texture in same color.
            # If 'i % num_categories' , then same texture in same category.
            i_ = i if num_series == 1 else i // num_categories
            patch.set_hatch(textures[i_])
            
    # Set label rotations.
    if orientation == 'vertical':
        for xticklabel in ax.get_xticklabels():
            xticklabel.set_rotation(label_rotation)
    else:
        for yticklabel in ax.get_yticklabels():
            yticklabel.set_rotation(label_rotation)
            
    # Set bar width and diff.
    if bar_scale:
        for i, patch in enumerate(ax.patches):
            # Re-center the bar.
            if orientation == 'vertical':
                current_width = patch.get_width()
                new_width = current_width * bar_scale
                bar_diff = current_width - new_width
                patch.set_width(new_width) # Reset width.
                patch.set_x(patch.get_x() + bar_diff * .5)
            else:
                current_height = patch.get_height()
                new_height = current_height * bar_scale
                bar_diff = current_height - new_height
                patch.set_height(new_height) # Reset width.
                patch.set_y(patch.get_y() + bar_diff * .5)
                
    # Set bar border.
    for i, patch in enumerate(ax.patches):
        # Set border to each bar.
        if has_bar_border:
            patch.set_linewidth(1)
            patch.set_edgecolor('black')
        else:
            patch.set_linewidth(0)
    
    # Set legend style
    # Legend position: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    # Legend style   : https://jakevdp.github.io/PythonDataScienceHandbook/04.06-customizing-legends.html
    if legend_position:
        if legend_position == 'top':
            legend = ax.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower center", borderaxespad=0, ncol=6)
        elif legend_position == 'right':
            legend = ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        elif legend_position == 'left':
            legend = ax.legend(bbox_to_anchor=(-0.5, 0.5), loc="center left", borderaxespad=0)
        elif legend_position == 'bottom':
            legend = ax.legend(bbox_to_anchor=(0.5, -0.1), loc="lower center", bbox_transform=plt.gcf().transFigure, ncol=6)

        frame = legend.get_frame()
        frame.set_facecolor('white')
        frame.set_edgecolor('white')
        
        # Set legend texts with random vocabulary
        texts = legend.get_texts()
        for text, label in zip(texts, legend_labels):
            text.set_text(label)
    elif num_series > 1:
        ax.legend_.remove()

    # Save
    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()