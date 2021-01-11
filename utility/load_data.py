"""
By: Jinseo Lee

Loading numpy array patient MS data
"""

import numpy as np 


def load_data(X_dir = 'MLBrainData/patientImagesCorTriple.npy', y_dir = 'MLBrainData/patientLabelsTriple.npy'):
    X = np.load(X_dir)
    y = np.load(y_dir)
    return X, y

"""
# sample test
X, y = load_data()

patient_class = []
for i in range(38):
    MRI_type = ['SPMS', 'RRMS', 'CTRL']
    patient_class.append(MRI_type[y[i*135]])
    
for i in range(38):
    print('patient '+ str(i+1) +': ' +patient_class[i])
    
plt.imshow(data[0], cmap = 'jet')
"""