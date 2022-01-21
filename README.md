# tf_object_detection
This repository helps use the pretrained models from the Tensorflow model zoo.

This repository was made from the tutorial by "TheCodingBug" https://www.youtube.com/watch?v=2yQqg_mXuPQ&list=PLUE9cBml08yiahlgN1BDv_8dAJFeCIR1u&index=1&ab_channel=TheCodingBug

To use this repository download and extract the zip and create a test folder in the same folder and add a test image or video to it and paste its location to the imagePath or videoPath variable in main.py, then run the main.py file in the environment created for the repository.


Creating the environment:
1 - install anaconda distribution                                                                                                                                                   
2 - create environment using :  conda create -n tf_gpu python==3.9                                                                                                                 
3 - install the cuda toolkit and cudnn in this environment using the command : conda install cudatoolkit=11.2 cudnn=8.1 -c=conda-forge                                             

select the appropriate versoins of cuda toolkit cudnn and python version using this link : https://www.tensorflow.org/install/source_windows
