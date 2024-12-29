'''
Created on 28.12.2024

@author: vital
'''
import matplotlib.pyplot as plt

def createSentimentDummyPieChart():
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [40.9, 41.6, 17.5]  # Example values
        colors = ['#d9d9d9', 'white', '#a6a6a6' ]  # Shades of gray
        
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

def createCompanyDummyPieChart():
        labels = ['TSLA', 'AAPL', 'GOOGL (GOOG)', 'MSFT', 'AMZN', ]
        sizes = [25.3, 32.9, 16.7, 8.7, 16.6]  # Example values
        colors = ['#d9d9d9', 'white', '#a6a6a6', '#8c8c8c', '#737373', ]  # Shades of gray
        
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

def createDuplicatesDummyPieChart():
        labels = ['Non-duplicates', 'Duplicates']
        sizes = [80.5, 19.5]
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
        
        
createSentimentDummyPieChart()