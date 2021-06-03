from config import*

class modelVGG16:
    
    model = VGG16()
    def __init__(self):
        
        self.model = Model(inputs = self.model.inputs, outputs = self.model.layers[-2].output)
    
    def getFeatureSet(self, reshaped_img):
     
        img = preprocess_input(reshaped_img)
        features = self.model.predict(img, use_multiprocessing=True)
        return features