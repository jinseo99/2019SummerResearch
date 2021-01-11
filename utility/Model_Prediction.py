"""
Different ways to use Model prediction
"""
import numpy as np

def compare_prediction_with_real(prediction, model_index):
    y_predicted =[]
    for i in range(0,len(prediction)):
        y_predicted.append(np.argmax(prediction[i]))
    y_predicted = np.array(y_predicted)
    
    dif = np.where(np.equal(y,  y_predicted)== False)[0]
    
    print('Number of Wrong predictions for model ' +str(model_index) +': ' + str(len(dif)))
    
    print('list of wrong predictions:')
    print('     patient     slice')
    for data_index in dif:
        patient_index = int(data_index/135)
        slice_index = data_index - patient_index*135
        print('       '+str(patient_index)+'         '+str(slice_index))

X = np.load("MLBrainData/38_FT1T2_data.npy")
y = np.load('MLBrainData/labels_categorical.npy')
for model_index in range(8):
    print('saving prediction for model ' +str(model_index))
    model = loadInitialModel(model_index)
    model.load_weights('final_Models/model_' + str(model_index)+'_135_data_best_loss.h5')
    prediction = model.predict(X)
    save_dir = 'final_numpy/Prediction/model_'+str(model_index)+'/Prediction_model_' +str(model_index)+'.npy'
    print(save_dir)
    np.save(save_dir, prediction)

    if model_index in [2,3]:
        compare_prediction_with_real(prediction, model_index)
