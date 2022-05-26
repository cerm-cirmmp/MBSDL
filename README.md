# MBSDL
Metal Binding Sites Deep Learning

This repo contains the code to run the physiological-adventitious MBS classifier.
The outcome of the classifier is a csv file containing the predictions for each MBS given as input.



## How to run the predictor on new data


### Data preparation
- For each site, create a dictionary containing the features as described in the paper and save it as pickle file. 
Zinc and iron sites used in this work are available as examples.
 
- Create a folder containing the MBS data as described in the paper.

- Create a csv file with header \['site', 'length'\] containg the names of the files and their lengths.
- Run the following command
```
python Classify.py --data_path <data_to_test_folder>  --list_path <data_to_test_folder/mbs_list.csv>
```

  
## How to run the predictor to reproduce the performances
  
- create two empty folder 'zinc' and 'iron'
- download and extract the MBS data in the respective folders.

Link to data <url>

To classify zinc data:
```
python ClassifyAllSites.py --metal zinc
```
  
To classify iron data:  
```
python ClassifyAllSites.py --metal iron
```
