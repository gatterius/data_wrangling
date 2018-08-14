# data_wrangling
A small case study of processing real estate data from the websites

The aim is to create a unionized dataset of target schema, using given data sources of different schemata, additional datasets which need to be joined to the main data, and reference data which is used to curate the main data. 

The algorithm includes following stages:
1.	Data extraction and profiling – load the datasets from files, extract the data schemata; perform basic profiling (null calculations);
2.	Schema matching – use schema-level matching to find correspondences between the source data schemata and given target schema;
3.	Schema mapping – using the found matches, extract matched columns and unionize them into one dataset of target schema, joining it with the given additional datasets; 
4.	Data repair – using reference data or externally discovered functional dependencies, update, clean data or fill in empty values to improve the quality of the result;
5.	Format transformation – change data values to maintain same format over entire dataset 
6.	Entity resolution – using techniques of vectorizing (hash vectorizing), dimensionality reduction (PCA) and clustering (Mean Shift), find and remove row duplicates from the dataset.

