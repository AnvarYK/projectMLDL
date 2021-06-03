from config import*
from featureSet import modelVGG16
# import util
parent_dir = "C:/MLDL2021Spring/ca2/ca2-AnvarYK-master/dataset/"
class C100Dataset:
    """
    X is a feature vector
    Y is the predictor variable
    """
    tr_x = [] # X (data) of training set.
    tr_y = []  # Y (label) of training set.
    ts_x = [] # X (data) of test set.
    #ts_y = [] # Y (label) of test set. Not needed, since we don't have them    
    reshaped_img = [] #using for getting feature set. It requires different dimension of image. That's why I can't use tr_x

    def __init__(self, filename):
        ## read the csv for dataset (cifar100.csv, cifar100_lt.csv or cifar100_nl.csv), 
        # 
        # Format:
        #   image file path,classname
        filepath = []
        classname = []
        with open(filename) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            # print(dictionary)
            line_count = -1
            for row in csv_reader:
                line_count += 1   
                if len(row) == 1:
                    self.ts_x.append(row[0]) 
                    continue  
                # 
                # Format:
                #   image file path,classname
                filepath.append(parent_dir+row[0])
                classname.append(row[1]) 
                ### TODO: Read the csv file and make the training and testing set
                # convert from 'PIL.Image.Image' to numpy array
                img = load_img(filepath[line_count], target_size=(32,32,3))
                imgArray = np.array(img)
                self.tr_x.append(img) 
                self.tr_y.append(classname[line_count])
                #for feature data
                reshape = np.resize(imgArray,(1,224,224,3))
                reshape = np.array(reshape)
                self.reshaped_img.append(reshape)

            print(f'Processed {line_count} lines.') 
        # ts_y = None ### TODO: YOUR CODE HERE Remark: There is no ts_y

    def getDataset(self):
        return [self.tr_x, self.tr_y, self.ts_x, self.ts_y]
    
    def getReshape(self):
        return self.reshaped_img

def main():
    #./dataset/cifar100_nl/data/cifar100_nl.csv
    cifar = C100Dataset('./dataset/cifar100_nl/data/cifar100_nl.csv')
    featureSet = modelVGG16()
    reshape = cifar.getReshape()
    # print(reshape)
    feature = featureSet.getFeatureSet(reshape)
    print(feature)

main()