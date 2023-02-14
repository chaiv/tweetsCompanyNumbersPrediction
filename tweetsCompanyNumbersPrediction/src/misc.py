'''
Created on 05.02.2023

@author: vital
'''
import pandas as pd
df =  pd.DataFrame(
                  [
                  (1.0),
                  (1.0),
                  (2.0),
                  (2.0),
                  (2.0)
                  ],
                  columns=["class"]
                  )
predictions = [1,1,1,1,2]
df['predicted_class'] = predictions
correctly_predicted_df = df.loc[df['class'] == df['predicted_class']]
print(correctly_predicted_df)
