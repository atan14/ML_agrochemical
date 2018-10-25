# Machine Learning on Agrochemical-likeness

Required Packages:
- numpy
- pandas
- rdkit
- scikit-learn
- keras

Run files in order of file numbering.
[1\_extract-data.ipynb -> 2\_conventional\_machine\_learning.ipynb]

Data source:
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


