"""
By: Jinseo Lee

Load model for visualization

Important Directories:
    Weights for CAM, GradCAM, GradCAM++: /JModels --> move to /Models
    Models with random initialized weights for training: /Models/InitialModels
    
"""

from keras.models import model_from_json

def loadModel(file):
	json_file = open('Models/'+file+'.json' , 'r') # delete Tumor

	model_json = json_file.read()
	json_file.close()
	#group = Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
	model = model_from_json(model_json)
	model.load_weights('Models/'+file+".h5")

	return model

"""
loading 
No Weights 
"""
# def loadModel(file):
# 	json_file = open('Models/'+file+'.json' , 'r') # delete Tumor

# 	model_json = json_file.read()
# 	json_file.close()
# 	#group = Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
# 	model = model_from_json(model_json)

# 	return model

def loadInitialModel(argument = -1):
    switcher = { 
        0: "AlexNet", 
        1: "ResNet50", 
        2: "Vgg16", 
        3: "Vgg19",
        4: "myVgg19",
        5: "myVGG19b",
        6: "myResNet50",
        7: 'ResNet18'
    }

    name = 'InitialModels/' + switcher.get(argument, "nothing") 
    model = loadModel(name)
    return model

"""
# sample run
model = loadInitialModel(3)
model.load_weights('JModels/test_1_model_3_best_loss_weights.h5')

model = loadInitialModel(7)
"""