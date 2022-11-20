# HyParamOptRL

# The structure of the project

```
|——> dataset.py                                               <— describes utils for dataset loading of AI Fairness model
|——> data_preprocessing.py                                    <— |
|——>   descrete_part_model_defect_recognition.ipynb           <— describes the main run file of the defect classification model
|——>   main.ipynb                                             <— describes the main run file of the AI Fairness model
|——>   models.py                                              <— file consists of a units of AI Fairness model
|——>   model_defect_recognition_data_Loader.py                <— describes main class of data loader for defect classification model
|——>   model_defect_recognition_data_path_info.py             <— describes additional class for the data loader class of the defect classification model
|——>   model_defect_recognition_storage_schema_constants.py   <— describes additional class for correct loading of data of the defect classification model   
|——>   README.md
|——>   RL Project Final.ipynb                                 <— the main file which describes a class of Greedy hyperparameters search algorithm
|——>   RLOpt.py                                               <— the main file which describes a class of Sarsa algorithm with modofications
|——>   shell_model_defect_recognition.py                      <— file wich provide functionality for ranning defect classification model on Sarsa algorithm
|——>   simple_models_test.ipynb                               <— file that consists of 4 additional simple models for testing
|——>   trainer.py                                             <— file that describes the functionality for modified model training 
└——>   utils.py                                               <— file that describes additional functionality for Sarsa algorithm 
```
# Run the project

1) To run this project clone it first on your computer to the specific folder
```
$ git clone https://github.com/Reinforcement-Learning-F22/HyParamOptRL
```
or download ```.zip``` file

2) Open or download IDE like "pyCharm" after go to ```Project -> open -> HyParamOptRL``` folder

3) To run Greedy hyperparameters search algorithm use ``` RL Project Final.ipynb ``` — with face classification model

4) To run modified Sarsa algorithm you can use:

    ``` main.ipynb ``` — with AI Fairness model

    ``` descrete_part_model_defect_recognition.ipynb ``` — with defect classification model

Contact the project developers if you want to download the datasets

