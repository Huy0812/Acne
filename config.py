import torch

BATCH_SIZE = 8 # increase / decrease according to GPU memeory
RESIZE_TO = 416 # resize the image for training and transforms
NUM_EPOCHS = 10 # number of epochs to train for
NUM_WORKERS = 4

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
TRAIN_DIR = 'C:\\Users\\Admin\\Desktop\\Elcom\\faster-rcnn\\20211129_A_Simple_Pipeline_to_Train_PyTorch_Faster_RCNN_Object_Detection_Model\\data\\train'
# validation images and XML files directory
VALID_DIR = 'C:\\Users\\Admin\\Desktop\\Elcom\\faster-rcnn\\20211129_A_Simple_Pipeline_to_Train_PyTorch_Faster_RCNN_Object_Detection_Model\\data\\valid'

# classes: 0 index is reserved for background
CLASSES = [
    'background', 'whitehead and blackhead', 'pustules', 'papules', 'nodules and cysts' 
]

NUM_CLASSES = len(CLASSES)

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True

# location to save model and plots
OUT_DIR = 'outputs'