'''
Created on 28.12.2024

@author: vital
'''
import matplotlib.pyplot as plt

def createDummyPieChart():
        labels = ['Non-duplicates', 'Duplicates']
        sizes = [90.7, 9.3]
        colors = [ 'white','lightgray'] 
        
        # Create the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes, 
            labels=labels, 
            colors=colors, 
            autopct='%1.1f%%', 
            startangle=90, 
            textprops={'fontname': 'Times New Roman', 'fontsize':16},
            wedgeprops={'edgecolor': 'black'}  
        )
        plt.axis('equal')  

        plt.show()
        
        
createDummyPieChart()