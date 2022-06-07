# MAgNet: Mesh-Agnostic Neural PDE Solver
This is the official repository to the paper "MAgNet: Mesh-Agnostic Neural PDE Solver". In this paper, we aim to address the problem of learning solutions to Pratial Differential Equations (PDE) while also generalizing to any mesh or resolution at test-time. This effectively enables us to generate predictions at any point of the PDE domain.  


![MAgNet](assets/magnet.jpg "MAgNet: Mesh-Agnostic Neural PDE Solver")

![Predictions](assets/predictions.JPG "Predictions vs Ground-Truth for different resolutions")
# Requirements

Start by installing the required modules:
```
pip install -r requirements.txt
```
# Dataset
The dataset is available for download at the following link: [magnet dataset](https://www.dropbox.com/sh/5d8vq03vmw21dhf/AAD1nK5ElGTiQ3dkoGjstthHa?dl=0).

The structure of the dataset is as follows:
```
├───E1
│   ├───irregular
│   │       CE_test_E1_graph_100.h5
│   │       CE_test_E1_graph_200.h5
│   │       CE_test_E1_graph_40.h5
│   │       CE_test_E1_graph_50.h5
│   │       CE_train_E1_graph_30.h5
│   │       CE_train_E1_graph_50.h5
│   │       CE_train_E1_graph_70.h5
│   │       
│   └───regular
│           CE_test_E1_100.h5
│           CE_test_E1_200.h5
│           CE_test_E1_40.h5
│           CE_test_E1_50.h5
│           CE_train_E1_50.h5
│           
├───E2
│   └───regular
│           CE_train_E2_50.h5
│           CE_test_E2_100.h5
│           CE_test_E2_200.h5
│           CE_test_E2_40.h5
│           CE_test_E2_50.h5
│           
└───E3
    └───regular
            CE_test_E3_100.h5
            CE_test_E3_200.h5
            CE_test_E3_40.h5
            CE_test_E3_50.h5
            CE_train_E3_50.h5
```

Each file is formatted as follows: `CE_{mode}_{dataset}_{resolution}.h5` where `mode` can be `train` or `test` and `dataset` can be `E1`, `E2` or `E3` and `resolution` denotes the resolution of the dataset
# Experiments
