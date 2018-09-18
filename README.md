# Machine Learning on Agrochemical-likeness

Required Packages:
- numpy
- pandas
- rdkit

Data source:
[Chembl\_24\_sqlite] `https://www.ebi.ac.uk/chembl/`

Data retrieval command: 
```
sqlite3 chembl_24.db
sqlite> .headers on
sqlite> .mode csv
sqlite> .output toxins.csv
sqlite> select * from assays where description like "%<keyword>%";
```
where \<keyword\> used are 'nematicide', 'herbacide', 'insecticide', 'fungicide' and 'toxin'.

