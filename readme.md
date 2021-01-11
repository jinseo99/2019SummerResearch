## Getting Started

These instructions will get you a copy of the project up and running on your local machine for testing.

### Prerequisites

You will need to have the following package structure:
```
root
├── run.py
├── env # this is the directory for venv
├── HeatMapAlgorithms
│   └── CAM.py
│   └── Grad_CAM.py
│   └── Grad_CAMPP.py
├── Utility
│   └── load_data.py
│   └── load_model.py
│   └── Model_Prediction.py
│   └── process_image.py
│   └── save_image.py
│   └── train_model.py
│   └── Validate_Activation_Map.py
│   └── visualize_image.py
│   └── visualize_intermediate.py
├── MLBrainData
│   └── some_npy.py
└── Models
    └── some_model.json
    └── some_model.h5

```

## Running the tests

### Step 1:

Terminal 1 - Initialize venv:
```
source env/bin/activate
```
### Step 2:
fill in the following variables to run run.py:
```
filename0
filename1
filename2
last_conv_layer
```

### Step 3:
Run run.py
```
python run.py
```

### Step 4:

Test the code.
You can also import other functions for testing other components of the package inside the run.py file.