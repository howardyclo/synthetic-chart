import random
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from data_utils import generate_bg_color, generate_fg_colors

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

def generate_pie(category_range=(2, 7+1),
                 has_texture=None,
                 has_pie_border=None,
                 fg_hue=None,
                 fg_luminosity=None,
                 bg_luminosity=None,
                 fg_colors=None,
                 bg_color=None,
                 filepath=None,
                 factors=None,
                 legend_position='',
                 legend_position_offset='random',
                 phase='train'):
    
    if not factors:
        factors = ['color', 'vocab', 'legend_position']
    
    vocab = train_vocab
    if 'vocab' in factors and phase == 'test':
        vocab = test_vocab
    
    # Define choices.
    if bg_luminosity == None:
        AXES_STYLE_CHOICES = ['white', 'dark']
    elif bg_luminosity == 'dark':
        AXES_STYLE_CHOICES = ['dark']
    else:
        AXES_STYLE_CHOICES = ['white']
        
    if bg_color:
        AXES_STYLE_CHOICES = ['dark'] # Pie does not have grid and ticks style.
        
    PIE_TEXT_STYLE_CHOICES  = [None, 'label', 'number', 'label_number']
    TEXTURE_CHOICES         = ['/', '\\', '|', '-', '+', 'x', '.']
    LEGEND_POSITION_CHOICES = [None, 'top', 'right', 'left', 'bottom']
    
    # Format arguments
    if has_texture == None: has_texture = np.random.choice([True, False])
    if has_pie_border == None: has_pie_border = np.random.choice([True, False])
    
    #----------------------------------------------------------------------------------------
    # Random generate parameters.
    #----------------------------------------------------------------------------------------
    # Random generate axes style.
    axes_style = np.random.choice(AXES_STYLE_CHOICES)
    
    # Random generate axes background colors.
    if not bg_color:
        if 'color' not in factors:
            axes_facecolor, bg_luminosity = generate_bg_color(bg_luminosity, phase) if axes_style == 'dark' else ('#ffffff', 'bright')
        else:
            axes_facecolor, bg_luminosity = generate_bg_color(bg_luminosity, 'train') if axes_style == 'dark' else ('#ffffff', 'bright')
    else:
        axes_facecolor = bg_color
    
    # Random generate pie text style.
    pie_text_style = np.random.choice(PIE_TEXT_STYLE_CHOICES)
    
    # Random generate legend position.
    if legend_position == '':
        legend_position = np.random.choice(LEGEND_POSITION_CHOICES)
        
    # Random generate data.
    num_categories = np.random.randint(*category_range)
    data = np.random.randint(low=1, high=10, size=num_categories)
    data = (100 * data / data.sum())
    
    # Random generate labels.
    legend_title = np.random.choice(vocab)
    legend_labels = [np.random.choice(vocab) for _ in range(num_categories)]
       
    # Random generate colors and shuffle color palette order.
    if not fg_colors:
        num_colors = num_categories
        if 'color' in factors:
            hex_colors = generate_fg_colors(num_colors, fg_hue, fg_luminosity, phase)
        else:
            hex_colors = generate_fg_colors(num_colors, fg_hue, fg_luminosity, 'train')
    else:
        assert len(fg_colors) >= num_categories
        hex_colors = fg_colors[:num_categories] # Ensure same number of colors as category. 
    
    # Adjust style parameters based on sampled chart parameters.
    if pie_text_style in ['label', 'label_number']:
        legend_position = None
    if num_categories > len(TEXTURE_CHOICES):
        has_texture = False
        
    #----------------------------------------------------------------------------------------
    # Plot
    #----------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(aspect="equal"))
    
    def _number_text(pct):
        return "{:.1f}%".format(pct) if pie_text_style in ['number', 'label_number'] else ""
    
    _legend_labels = legend_labels if pie_text_style in ['label', 'label_number'] else None
    linewidth = 1 if has_pie_border else 0
    wedges, label_texts, number_texts = ax.pie(data,
                                               labels=_legend_labels,
                                               colors=hex_colors,
                                               autopct=lambda pct: _number_text(pct),
                                               # https://matplotlib.org/api/_as_gen/matplotlib.patches.Wedge.html
                                               wedgeprops={"edgecolor":"0",
                                                           'linewidth': linewidth,
                                                           'linestyle': '-',
                                                           'antialiased': True})
    
    #----------------------------------------------------------------------------------------
    # Set styles
    #----------------------------------------------------------------------------------------
    # Set background color.
    fig.patch.set_facecolor(axes_facecolor)
    
    # Set pie label text color.
    if bg_luminosity == 'dark':
        for text in label_texts:
            text.set_color('white')
    
    # Set pie number text size.
    # plt.setp(number_texts, size=10, weight='bold')
    
    # Set pie number text color.
    for text in number_texts:
        text.set_color('white')
        
    # Set pie texture.
    if has_texture:
        texture_choices = TEXTURE_CHOICES.copy()
        random.shuffle(texture_choices)
        num_textures = num_categories
        for i, wedge in enumerate(wedges):
            wedge.set_hatch(texture_choices[i])
    else:
        texture_choices = []
        num_textures = 0
        
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
            legend = ax.legend(wedges, legend_labels,
                               title=legend_title, bbox_to_anchor=(0, y, w, 2.2), loc="lower center", borderaxespad=0, ncol=6)
            
        elif legend_position == 'right':
            if legend_position_offset == 'random':
                if phase == 'train' or 'legend_position' not in factors:
                    x = np.random.choice(legend_position_splits['train']['right']['x'])
                    y = np.random.choice(legend_position_splits['train']['right']['y'])
                else:
                    x = np.random.choice(legend_position_splits['test']['right']['x'])
                    y = np.random.choice(legend_position_splits['test']['right']['y'])    
            else: x, y = 1.04, 0.5
            legend = ax.legend(wedges, legend_labels,
                               title=legend_title, bbox_to_anchor=(x, y), loc="center left", borderaxespad=0)
            
        elif legend_position == 'left':
            if legend_position_offset == 'random':
                if phase == 'train' or 'legend_position' not in factors:
                    x = np.random.choice(legend_position_splits['train']['left']['x'])
                    y = np.random.choice(legend_position_splits['train']['left']['y'])
                else:
                    x = np.random.choice(legend_position_splits['test']['left']['x'])
                    y = np.random.choice(legend_position_splits['test']['left']['y'])
            else: x, y = -0.5, 0.5
            legend = ax.legend(wedges, legend_labels,
                               title=legend_title, bbox_to_anchor=(x, y), loc="center left", borderaxespad=0)
            
        elif legend_position == 'bottom':
            if legend_position_offset == 'random':
                if phase == 'train' or 'legend_position' not in factors:
                    x = np.random.choice(legend_position_splits['train']['bottom']['x'])
                    y = np.random.choice(legend_position_splits['train']['bottom']['y'])
                else:
                    x = np.random.choice(legend_position_splits['test']['bottom']['x'])
                    y = np.random.choice(legend_position_splits['test']['bottom']['y'])
            else: x, y = 0.5, -0.1
            legend = ax.legend(wedges, legend_labels,
                               title=legend_title, bbox_to_anchor=(x, y), loc="lower center", bbox_transform=plt.gcf().transFigure, ncol=6)
        
        # Set legend background color
        frame = legend.get_frame()
        frame.set_facecolor(axes_facecolor)
        frame.set_edgecolor(axes_facecolor)
        
        # Set legend text color to white when chart has dark background color.
        if bg_luminosity == 'dark':
            legend.get_title().set_color('w')
            plt.setp(legend.get_texts(), color='w')
    
    #----------------------------------------------------------------------------------------
    # Save
    #----------------------------------------------------------------------------------------
    if filepath:
        plt.savefig(filepath, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        
    #----------------------------------------------------------------------------------------
    # Format returns
    #----------------------------------------------------------------------------------------
    style = {
        'fg_luminosity': fg_luminosity,
        'bg_luminosity': bg_luminosity,
        'num_categories': num_categories,
        'hex_colors': hex_colors,
        'axes_style': axes_style,
        'axes_facecolor': axes_facecolor,
        'textures': texture_choices[:num_textures],
        'legend_title': legend_title,
        'legend_labels': legend_labels,
        'legend_position': legend_position,
        'pie_text_style': pie_text_style,
        'has_pie_border': has_pie_border
    }
        
    return data, style

def render_pie(data, style, filepath=None):
    # Extract styles
    fg_luminosity = style['fg_luminosity']
    bg_luminosity = style['bg_luminosity']
    num_categories = style['num_categories']
    hex_colors = style['hex_colors']
    axes_style = style['axes_style']
    axes_facecolor = style['axes_facecolor']
    textures = style['textures']
    legend_title = style['legend_title']
    legend_labels = style['legend_labels']
    legend_position = style['legend_position']
    pie_text_style = style['pie_text_style']
    has_pie_border = style['has_pie_border']
    
    #----------------------------------------------------------------------------------------
    # Plot
    #----------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(aspect="equal"))
    
    def _number_text(pct):
        return "{:.1f}%".format(pct) if pie_text_style in ['number', 'label_number'] else ""
    
    _legend_labels = legend_labels if pie_text_style in ['label', 'label_number'] else None
    linewidth = 1 if has_pie_border else 0
    wedges, label_texts, number_texts = ax.pie(data,
                                               labels=_legend_labels,
                                               colors=hex_colors,
                                               autopct=lambda pct: _number_text(pct),
                                               # https://matplotlib.org/api/_as_gen/matplotlib.patches.Wedge.html
                                               wedgeprops={"edgecolor":"0",
                                                           'linewidth': linewidth,
                                                           'linestyle': '-',
                                                           'antialiased': True})
    
    #----------------------------------------------------------------------------------------
    # Set styles
    #----------------------------------------------------------------------------------------
    # Set background color.
    fig.patch.set_facecolor(axes_facecolor)
    
    # Set pie label text color.
    if bg_luminosity == 'dark':
        for text in label_texts:
            text.set_color('white')
    
    # Set pie number text size.
    # plt.setp(number_texts, size=10, weight='bold')
    
    # Set pie number text color.
    for text in number_texts:
        text.set_color('white')
        
    # Set pie texture.
    if textures:
        for i, wedge in enumerate(wedges):
            wedge.set_hatch(textures[i])

    # Set legend style
    # Legend position: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    # Legend style   : https://jakevdp.github.io/PythonDataScienceHandbook/04.06-customizing-legends.html
    if legend_position:
        if legend_position == 'top':
            legend = ax.legend(wedges, legend_labels,
                               title=legend_title, bbox_to_anchor=(0,1.02,1,0.2), loc="lower center", borderaxespad=0, ncol=6)
        elif legend_position == 'right':
            legend = ax.legend(wedges, legend_labels,
                               title=legend_title, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        elif legend_position == 'left':
            legend = ax.legend(wedges, legend_labels,
                               title=legend_title, bbox_to_anchor=(-0.5, 0.5), loc="center left", borderaxespad=0)
        elif legend_position == 'bottom':
            legend = ax.legend(wedges, legend_labels,
                               title=legend_title, bbox_to_anchor=(0.5, -0.1), loc="lower center",
                               bbox_transform=plt.gcf().transFigure, ncol=6)
        
        # Set legend background color
        frame = legend.get_frame()
        frame.set_facecolor(axes_facecolor)
        frame.set_edgecolor(axes_facecolor)
        
        # Set legend text color to white when chart has dark background color.
        if bg_luminosity == 'dark':
            legend.get_title().set_color('w')
            plt.setp(legend.get_texts(), color='w')
    
    #----------------------------------------------------------------------------------------
    # Save
    #----------------------------------------------------------------------------------------
    if filepath:
        plt.savefig(filepath, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()