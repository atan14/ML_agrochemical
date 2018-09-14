# Machine Learning on Agrochemical-likeness

Required Packages:
- numpy
- pandas
- rdkit

Data source:
Chembl_24_sqlite
Data retrieval command: 
```
sqlite> .headers on
sqlite> .mode csv
sqlite> .output toxins.csv
sqlite> select * from assays where description like "%<keyword>%";
```
where <keyword> used are 'nematicide', 'herbacide', 'insecticide', 'fungicide' and 'toxin'.

