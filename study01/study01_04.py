#O'REILLY Pythonではじめる機械学習 p.10～11
import pandas as pd
from IPython import display

data = {'Name':["John","Anna","Peter","Linda"],
        'Location':["New York","Paris","Berlin","London"],
        'Age':[24,13,53,33]}
data_pandas = pd.DataFrame(data)
#IPhtyon.displayを用いるとDataFrameを
#Jupyter notebook上できれいに表示することができる
#Jupyter notebookの起動方法
#>cd d:\python\venv\309AI\.309AI\scripts
#>jupyter notebook
#Webブラウザから
display.display(data_pandas)
#print(data_pandas)
display.display(data_pandas[data_pandas.Age>30])
