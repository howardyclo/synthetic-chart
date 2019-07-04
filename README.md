# Synthetic-Chart
![](https://i.imgur.com/wbNVmNnr.png)
- Code to generate style-enriched chart images and annotations based on Matplotlib and Seaborn. See `demo.ipynb`.
- Combines the best of [FigureQA](https://arxiv.org/abs/1710.07300) (multiple chart types) and [DVQA](https://arxiv.org/abs/1801.08163) (enriched styles).
- Support vertical bar, horizontal bar, pie and line charts.
- Built on open-source visualization libraries, Seaborn and Matplotlib.

# Annotations
- Visual styles (shared across different types of charts): Axes style, background color, foreground colors, has border, textures, legend position, other minor styles.
- Text labels
- Numerical data

# Train & Test Splits Generation
- Note that there is a `phase` argument for generating charts, it is mainly generating training and testing data with some differences:
  - Generate foreground colors based on two different color splits.
  - Generate labels based on two different vocabulary splits.
