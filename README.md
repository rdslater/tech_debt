# Description
This was a model given to me by junior engineers.  It needs to go into a PACs system at UW Health

The .ipynb was the initial model.

- FinalClassify_andCropFF.ipynb (Notebook I received)
- FinalClassify_andCropFF.py (Saved as py)
- main.py (main function--created from the notebook, edited for PEP8)
- models.py (split model definitions into here)

Yes I still have some long lines due to model paths.  I'll live for now

## TODO
main.py right now accepts and excel file and saves a jpg.  We need a dicom output so output will be modified.  Also we probably need to return the image rather than saving it, but that depends on the final details of the pacs system

Every time this model runs, we load up the model--creates overhead.  For now is acceptable.  Will not scale (Model Serving or Triton required).  This is only proof of concept so higher priority is formating.

Ivan Drago says about bout print statements "I must break them"  (aka USE LOGGING PEOPLE!)

