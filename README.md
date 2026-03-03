# pygee
This repository aims to provide codes and demos for producing landcover maps as in **Guerou et al (submitted)**.  
It is based on **python code** to runs analysis on **Google Earth Engine (GEE) python API** through local notebooks.

## Requirements

### Hardware requirements
No specific computer requirements are needed for GEE processing.  
Local visualisation of the produced dataset might however require "standard" computer memory usage (16 GB+ RAM)

### Software requirements
This software should be compatible with:
* macOS
* Windows
* Linux

It has been tested on **Linux: Ubuntu 22.04.5 LTS**

### GEE requirements
One needs to create a Google Earth Engine (GEE) account and install GEE python API module 

* **Google Earth Engine (GEE) account**: https://developers.google.com/earth-engine/guides/access
* **GEE python API**: https://developers.google.com/earth-engine/guides/python_install  


A python environment is required beforehand, see **Installation guide** if needed.


## Installation guide

### Python environment
A **python3** environment is required.  
Multiple choices are possible depending on your preferences.   
All have been created with the help of **[anaconda](https://www.anaconda.com/download)**  

**A.** Exact clone of the python environment used for this study  

One can create such an environment (after download of <mark>pygee</mark>):
```
cd pygee
conda env create -f environment_pygee.yml
```

**B.** A simpler python environment can be created instead with basic modules:
* First edit the <mark>basic_requirement_pygee.yml</mark> file with the desired
  * **env_name**
  * **prefix**
* Execute the command lines
    ```
    cd pygee
    conda env create -f basic_requirement_pygee.yml
    ```  

**C.** Eventually, for a quicker set up of an already existing environment  

Here is a list of the required modules for <mark>pygee</mark>
* importlib
* numpy
* operator
* setuptools
* pandas
* geopandas
* matplotlib
* xarray
* rioxarray
* dask
* earthengine-api
* eemont
* geemap


### Installation (pygee)

Clone or download  <mark>pygee</mark>
```
git clone https://github.com/adguerou/pygee.git
cd pygee
python3 setupy.py install
```


## Runs
The **landcover maps** of the European Alps **postglacial areas** produced in 
**Guerou et al (submitted)** can be reproduced by executing the notebooks founds in <mark>notebooks/</mark>

Exports and savings of the various outputs can be skipped. All necessary datasets have indeed been shared to avoid long GEE computational runs. They can be however visualized without the need to export them.