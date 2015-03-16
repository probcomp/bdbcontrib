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
cc_client = facade.BayesDBCrossCat.from_csv('my_database.bdb', 'my_table', 'data.csv', 
                                            csv_code_filename='codebook.csv')

# Intialize and analyze models
cc_client('INITIALIZE 10 MODELS FOR my_table')
cc_client('ANALYZE my_table FOR 100 ITERATIONS WAIT')

# do a query and get the results as a pandas DataFrame
df = cc_client('SELECT column_0 from my_table').as_df()

# render a vizualization of a given crosscat state
import matplotlib.pyplot as plt
cc_client.plot_state('my_table', 0)
plt.show()
 ```

## Plotting
`pairplot` works similarly to `seaborn.pairplot` but handles combinations of data types better. Pairwise combinations of data types are plotted as follows:

- Numerical-Numerical: **Scatter with KDE overlay**
- Categorical-Numerical: **Violin plot**
- Categorical-Categorical: **Heatmap**

### Example
Plot data straight from the table:
```python
# ...continued from above
import bdbcontrib.plotutils as pu

df = cc_client('SELECT col_1, col_2, col_3, col_4 from my_table')
pu.pairplot_cols(cc_client.bdb, 'my_table', df)
plt.show()
```

Plot something arbitrary
```python
# ...continued from above
df = cc_client('SELECT col_1 + col_2, col_3 || col_4, col_5, "Ward Cleaver" from my_table')
pu.pairplot_cols(cc_client.bdb, 'my_table', df)
plt.show()
```
