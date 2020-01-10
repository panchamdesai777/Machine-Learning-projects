# Mars Crater Predictions (Ensemble Methods)


## Problem Statement :
Determine if the instance is a crater or not a crater. 1=Crater, 0=Not Crater 

## About Dataset :
Using the technique described by L. Bandeira (Bandeira, Ding, Stepinski. 2010.Automatic Detection of Sub-km Craters Using Shape and Texture Information) we identify crater candidates in the image using the pipeline depicted in the figure below.</br> 
Each crater candidate image block is normalized to a standard scale of 48 pixels. Each of the nine kinds of image masks probes the normalized image block in four different scales of 12 pixels, 24 pixels, 36 pixels, and 48 pixels, with a step of a third of the mask size (meaning 2/3 overlap).</br> 
We totally extract 1,090 Haar-like attributes using nine types of masks as the attribute vectors to represent each crater candidate.</br> 
The dataset was converted to the Weka ARFF format by Joseph Paul Cohen in 2012.</br>

![Alt txt](https://storage.googleapis.com/ga-commit-live-prod-live-data/account/b92/11111111-1111-1111-1111-000000000000/b-534/23d79bc9-0a25-44b7-8f11-e19429e4ce74/file.png)

## Attribute Information:
We construct a attribute vector for each crater candidate using Haar-like attributes described by Papageorgiou 1998.</br>  
These attributes are simple texture attributes which are calculated using Haar-like image masks that were used by Viola in 2004 for face detection consisting only black and white sectors.</br>  
The value of an attribute is the difference between the sum of gray pixel values located within the black sector and the white sector of an image mask. The figure below shows nine image masks used in our case study.</br> 
The first five masks focus on capturing diagonal texture gradient changes while the remaining four masks on horizontal or vertical textures.</br> 

## Approach taken to solve the project :
* Load the data </br>  
* Preprocessing of Data : split it into train and test set and standardize the data.</br>  
* Predict the values after building a Machine learning model.</br>  
* Decision Tree
* Can we improve our model's performance with Random forrest algorithm?
* Bagginng or Bootstrap aggregation
