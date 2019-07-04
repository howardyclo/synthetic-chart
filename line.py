import numpy as np
import pandas as pd
import seaborn as sns
import pickle

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from data_utils import generate_bg_color, generate_fg_colors, generate_scatter_data_continuous

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
        
def generate_line(category_range=(1, 7+1),
                  series_range=(5, 20+1),
                  fg_colors=None,
                  bg_color=None,
                  fg_hue=None,
                  fg_luminosity=None,
                  bg_luminosity=None,
                  filepath=None,
                  phase='train',
                  factors=None,
                  legend_position='',
                  legend_position_offset='random'):
    
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
        
    LEGEND_POSITION_CHOICES = [None, 'top', 'right', 'left', 'bottom']
    
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
    
    # Random generate legend position.
    if legend_position == '':
        legend_position = np.random.choice(LEGEND_POSITION_CHOICES)
    
    # Random generate labels.
    x_title, y_title, legend_title = np.random.choice(vocab, size=3, replace=False) # Do not sample same label (replace=False).

    # Random generate data.
    num_categories = np.random.randint(*category_range)
    num_series = np.random.randint(*series_range)

    x_range = (0, 100)
    y_range = (0, 100)
    
    data = generate_scatter_data_continuous(x_range, y_range,
                                            x_distns=['linear'],
                                            shapes=['linear', 'linear_with_noise', 'quadratic'],
                                            n_classes=num_categories,
                                            n_points=num_series,
                                            fix_x_range=True)
    
    # Create dataframe from generated data and labels.
    cs, xs, ys = [], [], []
    legend_labels = np.random.choice(vocab, size=len(data['data']), replace=False) # Do not sample same label (replace=False).
    for line, legend_label in zip(data['data'], legend_labels):
        xs.extend(line['x'])
        ys.extend(line['y'])
        cs.extend([legend_label for _ in range(len(line['x']))])
        
    df = pd.DataFrame({
        legend_title: cs,
        x_title: xs,
        y_title: ys
    })
    
    # Random generate colors and shuffle color palette order.
    if not fg_colors:
        num_colors = num_categories
        if 'color' in factors:
            hex_colors = generate_fg_colors(num_colors, fg_hue, fg_luminosity, phase)
        else:
            hex_colors = generate_fg_colors(num_colors, fg_hue, fg_luminosity, 'train')
    else:
        hex_colors = fg_colors
    
    #----------------------------------------------------------------------------------------
    # Plot
    #----------------------------------------------------------------------------------------
    ax = sns.lineplot(x=x_title, y=y_title, hue=legend_title, data=df, palette=sns.color_palette(hex_colors))
    
    #----------------------------------------------------------------------------------------
    # Set styles.
    #----------------------------------------------------------------------------------------
    # Set grid style.
    if axes_style in ['whitegrid', 'darkgrid']:
        if bg_luminosity == 'dark':
            ax.grid(color='#ffffff', linestyle='-', linewidth=1)
        else:
            ax.grid(color='.8', linestyle='-', linewidth=1) # plot light gray (same as Seaborn)
            
    # Set background color.
    ax.set_facecolor(axes_facecolor)
    
    # Set legend style
    # Legend position: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    # Legend style   : https://jakevdp.github.io/PythonDataScienceHandbook/04.06-customizing-legends.html
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
    else:
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
        'hex_colors': hex_colors,
        'axes_style': axes_style,
        'axes_facecolor': axes_facecolor,
        'legend_title': legend_title,
        'legend_labels': legend_labels,
        'legend_position': legend_position,
        'x_title': x_title,
        'y_title': y_title
    }
    
    return df, style

def render_line(df, style, filepath=None):
    # Extract styles
    fg_luminosity = style['fg_luminosity']
    bg_luminosity = style['bg_luminosity']
    num_categories = style['num_categories']
    num_series = style['num_series']
    hex_colors = style['hex_colors']
    axes_style = style['axes_style']
    axes_facecolor = style['axes_facecolor']
    legend_title = style['legend_title']
    legend_labels = style['legend_labels']
    legend_position = style['legend_position']
    x_title = style['x_title']
    y_title = style['y_title']
    
    #----------------------------------------------------------------------------------------
    # Plot
    #----------------------------------------------------------------------------------------
    ax = sns.lineplot(x=x_title, y=y_title, hue=legend_title, data=df, palette=sns.color_palette(hex_colors))
    
    #----------------------------------------------------------------------------------------
    # Set styles.
    #----------------------------------------------------------------------------------------
    # Set grid style.
    if axes_style in ['whitegrid', 'darkgrid']:
        if bg_luminosity == 'dark':
            ax.grid(color='#ffffff', linestyle='-', linewidth=1)
        else:
            ax.grid(color='.8', linestyle='-', linewidth=1) # plot light gray (same as Seaborn)
            
    # Set background color.
    ax.set_facecolor(axes_facecolor)
    
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
    else:
        ax.legend_.remove()
    
    #----------------------------------------------------------------------------------------
    # Save
    #----------------------------------------------------------------------------------------
    if filepath:
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
    
    