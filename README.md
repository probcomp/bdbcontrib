# `bdbcontrib`
 A set of convenience utilities for BayesDB (SQLite backend)

 **Requires**

 - Matplotlib
 - Seaborn
 - Pandas
 - numpy

 ## CrossCat wrapper

 ```python
import bdbcontrib.facade

# load data from csv
lient = facade.BayesDBClient.from_csv('my_database.bdb', 'my_table', 'data.csv', 
                                      csv_code_filename='codebook.csv')

# Intialize and analyze models
client('INITIALIZE 10 MODELS FOR my_table')
client('ANALYZE my_table FOR 100 ITERATIONS WAIT')
```

Do a query and get the results as a pandas DataFrame
``` python
df = cc_client('SELECT column_0 from my_table').as_df()
```

Render a vizualization of a given crosscat state
``` python
import matplotlib.pyplot as plt
cc_client.plot_state('my_table', 0)
plt.show()
 ```

## Plotting
`pairplot` works similarly to `seaborn.pairplot` but handles combinations of data types better. Pairwise combinations of data types are plotted as follows:

- Numerical-Numerical: **Scatter with KDE overlay**
- Categorical-Numerical: **Violin plot**
- Categorical-Categorical: **Heatmap**

## Example `draw_state`
Render a visualization of a Crosscat state hilighting the 'age' column in blue

```python
from bdbcontrib.draw_cc_state import daw_state
modelno = 0
draw_state(client.bdb, 'my_table', modelno, hilight_cols=['age'], hilight_cols_colors=['red'])
```

View the 'blank state'---the raw data without any Crosscat structure

```python
modelno = 0
draw_state(client.bdb, 'my_table', modelno, blank_state=True, 
           hilight_cols=['age'], hilight_cols_colors=['red'])
```

### Example `pairplot`
Plot data straight from the table:

```python
# ...continued from above
import bdbcontrib.plotutils as pu

df = client('SELECT col_1, col_2, col_3, col_4 from my_table').as_df()
pu.pairplot(client.bdb, 'my_table', df)
plt.show()
```

Plot something arbitrary
```python
# ...continued from above
df = client('SELECT col_1 + col_2, col_3 || col_4, col_5, "Ward Cleaver" from my_table')
pu.pairplot(client.bdb, 'my_table', df)
plt.show()
```

### Example `zmatrix`

```python
df = client('ESTIMATE PAIRWISE DEPENDENCE PROBABILITY FROM my_table').as_df()

# kwargs sent to seaborn.clustermap
clustermap_kws = {
    'linewidths': 0,
    'cmap': 'PuBu'
}
pu.zmatrix(df, clustermap_kws)
```
