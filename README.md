# Machine Learning on Agrochemical-likeness

### Required Packages: 
- numpy
- pandas
- rdkit
- scikit-learn
- keras
- tensorflow
- matplotlib

### Note on files: 
1. __overlap\_with_toxins__ - folder that contains all the previous runs on old data, which consists of compounds which are both toxic and are agrochemicals.
2. __removed\_overlap\_with\_toxins__ - folder that contains the newer runs on new data. All compounds which are both toxic and used as agrochemicals are completely removed. 
  - __binary\_classification__ - folder for runs that only classify agrochemical and non-agrochemical.
  - __multiclass\_classification__ - folder for runs that further classify sub-agrochemical classes. 
    - __includes\_toxin__ - classify all compounds into sub-agrochemical or non-agrochemical classes (no toxin removed).
    - __no\_toxin__ - only classify agrochemicals into their sub-agrochemical classes (toxin molecules removed).

__dataset1.pkl__ - Dataset that contains 26,543 compounds (11,543 agrochemicals and 15,000 toxins)    
__dataset2.pkl__ - Dataset that contains all 499,017 extracted compounds (11,543 agrochemicals and 487,474 toxins)


Run files in order of file numbering.

### Data source:
[Chembl\_24\_sqlite.tar.gz -> chembl\_24.db] `https://www.ebi.ac.uk/chembl/`

If retrieval of data as separate csv files is desired, use the following command:
```
sqlite3 chembl_24.db
sqlite> .headers on
sqlite> .mode csv
sqlite> .output toxins.csv
sqlite> select * from assays where description like "%<keyword>%"; [note the two % before and after keyword]
```
where \<keyword\> used in this project are 'nematicidal', 'herbacidal', 'insecticidal', 'fungicidal' and 'toxin'. 


### For the code to work on CUDA-8.0, some packages need to be of specific versions (python 3.6.8) :
- tensorflow 1.3.0
- keras 2.0.5
- numpy 1.15.0
- matplotlib 2.2.2


