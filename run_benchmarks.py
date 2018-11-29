import argparse
import os
import shutil
import time
import sys
import sklearn
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import scipy
import sklearn.metrics
from dataloader import *
from torch.nn.parameter import Parameter
from sklearn.metrics.pairwise import cosine_similarity
from models import *
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt



USR_THRESHOLD = True
NUM_MODELS = 9
# MODEL_LIST = ['AlexNet', 'ResNet18', 'VGG16', 'SqueezeNet', 'DenseNet121', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152' , 'DenseNet169', 'DenseNet201']
MODEL_LIST = ['AlexNet', 'ResNet18', 'VGG16', 'SqueezeNet', 'DenseNet121', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152','CustomAlexnet']
def to_data(x, normalize =False):

    x = x.data.cpu().numpy()
    if normalize:
        for i in range(x.shape[0]):

            x[i] = (x[i] - min(x[i]))/float(max(x[i]) - min(x[i]))

        np.savetxt('normalized_Values.txt',x,delimiter = ',')
    return x


best_prec1 = 0

def parse_arguments():
    parser = argparse.ArgumentParser(description='PreTrained Model Analysis :: Bossanova')
    
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
    parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

    parser.add_argument('--lr', default=0.1, type = float,
                    help = 'Learning Rate')
    parser.add_argument('--e',default = 100, type = int,
                    help = 'Number of Epochs')
    parser.add_argument('--m',default = 0.9, type = float,
                    help = 'Momentum')
    parser.add_argument('--b',default = 32, type = int,
                    help = 'Batch Size')
    parser.add_argument('--w', default = 1e-4, type = float,
                    help = 'Weight Decay')
    parser.add_argument('--arch', default ='Alexnet', type = str,
                    help = 'Architecture Name')
    parser.add_argument('--thres', default = 0.5, type = float,
                    help = 'Threshold for classification')


    return parser.parse_args()

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(args,threshold,model,MODEL_COUNT):


    args = parse_arguments()
    args.distributed = args.world_size > 1


    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tranformations = transforms.Compose([
            transforms.Resize((112,80)),
            transforms.ToTensor(),
            normalize,
        ])

    batch_size = args.b

    if model_name == 'DenseNet' or model_name == 'SqueezeNet' or model_name == 'Inception':
        batch_size = 5
    elif int(MODEL_COUNT)%2 == 0 or int(MODEL_COUNT)%8 == 0 or int(MODEL_COUNT)%9 == 0:
        batch_size = 5

    test1 = Dataloader(data_dir = '/home/sreena/Documents/benchmarks/data',
                        filename = 'test.txt',
                        transform = tranformations,
                        flag = 'part1')
    test2 = Dataloader(data_dir = '/home/sreena/Documents/benchmarks/data',
                        filename = 'test.txt',
                        transform = tranformations,
                        flag = 'part2')

    test1_loader = torch.utils.data.DataLoader(
                        test1, 
                        batch_size = batch_size,
                        shuffle = False,
                        num_workers = args.workers,
                        pin_memory = True)
    test2_loader = torch.utils.data.DataLoader(
                        test2, 
                        batch_size = batch_size,
                        shuffle = False,
                        num_workers = args.workers,
                        pin_memory = True)


    loader = zip(test1_loader,test2_loader)
    cosine_scores = []
    true_labels = []

    total_time = 0

    # Generate features by (pretrained) model
    for i, [data1,data2] in enumerate(loader):

        start_time = time.time()

        if data1[2][0] == data2[2][0] :
            if model_name == 'ResNet':


                out1 = resnet(model,data1)
                out2 = resnet(model,data2)


            elif model_name == 'SqueezeNet':

                out1 = squeezenet(model,data1)
                out2 = squeezenet(model,data2)
            elif model_name == 'DenseNet':

                out1 = densenet(model,data1)
                out2 = densenet(model,data2)

            elif model_name == 'Inception':

                out1 = inception(model,data1)
                out2 = inception(model,data2)
            elif model_name == 'Custom_Alexnet':

                out1 = custom_alexnet(model,data1)
                out2 = custom_alexnet(model,data2)
            else:

                out1 = alex_vgg(model,data1)
                out2 = alex_vgg(model,data2)
        else:

            break
        end_time = time.time()
        total_time = total_time + end_time - start_time

        

        metric_score = np.array(cosine_similarity(to_data(out1, normalize =True),to_data(out2 , normalize = True)))
        metric_score = metric_score.diagonal()
        cosine_scores.extend(metric_score)
        true_labels.extend(to_data(data1[1]))

    

    

    true_labels = np.array(true_labels)
    cosine_scores = np.array(cosine_scores,dtype = np.float32)
    predicted = np.zeros((true_labels.shape))

    np.savetxt('cosine_scores.txt',cosine_scores,delimiter = ',')

    #Thresholding
    if USR_THRESHOLD:
        predicted[cosine_scores > args.thres] = 1
    else:
        predicted[cosine_scores>threshold] = 1


    Accuracy = sum(true_labels == predicted)/float(len(true_labels))



    precision = precision_score(true_labels, predicted)

    recall= recall_score(true_labels, predicted)


    true_pos_rate = sum(true_labels*predicted)/float(sum(true_labels))

    neg_labels = np.array((true_labels==0),dtype = np.uint8)
    fal_pos_rate = sum(neg_labels*predicted)/float(sum(neg_labels))



    print('{}  Threshold : {}'.format(model_name,threshold))

    return Accuracy,precision,recall,true_pos_rate,fal_pos_rate,total_time




if __name__ == '__main__':

    #model
   

    MODEL_COUNT = 0    


    sys.setrecursionlimit(10000)
    args = parse_arguments()
    print('What would you like to do?')
    user_in = int(raw_input(" 1. Analyze a specific model\n 2. All model Comparisions\n 3. Analyize a model for all thresholds\n"))
    if user_in == 1 or user_in == 3:

        USR_THRESHOLD = True
        print('\nWhich Pre-trained model would you like to use\n')
        model_decision = raw_input(" 1. Alexnet \n 2. Resnet\n 3. VGG16\n 4. Squeezenet\n 5. DenseNet\n 6. ResNet34\n 7. ResNet50\n 8. ResNet101\n 9. ResNet152\n 10. Custom_AlexNet\n")
        
        if user_in == 1:
            model,model_name = pretrained_models(model_decision)
            model.cuda()
            a,p,r,tr,nr,tt = main(sys.argv,0.5,model,model_decision)
            print('********Performance for {} ******'.format(model_name))
            print('Accuracy :: {}'.format(a))
            print('Precision :: {}'.format(p))
            print('Recall :: {}'.format(r))
            print('TRue positive rate :: {}'.format(tr))
        else:
            USR_THRESHOLD = False
            precision = []
            recall = []
            true_pos_rate = []
            fal_pos_rate = []
            Accuracy = []
            index = []
            total_time = []
            model,model_name = pretrained_models(model_decision)
            for i in range(10):
                a,p,r,tr,nr,tt = main(sys.argv,i/10.0,model,MODEL_COUNT)
                Accuracy.append(a)
                precision.append(p)
                recall.append(r)
                index.append(i/10.0)
                true_pos_rate.append(tr)
                fal_pos_rate.append(nr)
                total_time.append(tt)
            for i in range(995,1001):
                a,p,r,tr,nr,tt = main(sys.argv,i/1000.0,model,MODEL_COUNT)
                Accuracy.append(a)
                precision.append(p)
                recall.append(r)
                index.append(i/1000.0)
                true_pos_rate.append(tr)
                fal_pos_rate.append(nr)
                total_time.append(tt)
            np.save('acc_{}.npy'.format(model_name), np.array(Accuracy))
            np.save('precision_{}.npy'.format(model_name),np.array(precision))
            np.save('recall_{}.npy'.format(model_name),np.array(recall))
            np.save('true_pos_{}.npy'.format(model_name),np.array(true_pos_rate))
            np.save('fal_pos_{}.npy'.format(model_name),np.array(fal_pos_rate))
            


    else:

        USR_THRESHOLD = False
        model_accuracy = []
        model_precision = []
        model_recall = []
        model_tr = []
        model_nr = []
        model_tt =[]
        print(MODEL_LIST)
        print(NUM_MODELS)

        for model_decision in range(NUM_MODELS):

            MODEL_COUNT = MODEL_COUNT + 1
            precision = []
            recall = []
            true_pos_rate = []
            fal_pos_rate = []
            Accuracy = []
            index = []
            total_time = []

            for i in range(10):

                model,model_name = pretrained_models(model_decision +1)
                model.cuda()
                a,p,r,tr,nr,tt = main(sys.argv,i/10.0,model,MODEL_COUNT)
                Accuracy.append(a)
                precision.append(p)
                recall.append(r)
                index.append(i/10.0)
                true_pos_rate.append(tr)
                fal_pos_rate.append(nr)
                total_time.append(tt)

            for i in range(995,1001):
                model,model_name = pretrained_models(model_decision +1)
                model.cuda()
                a,p,r,tr,nr,tt = main(sys.argv,i/1000.0,model,MODEL_COUNT)
                Accuracy.append(a)
                precision.append(p)
                recall.append(r)
                index.append(i/1000.0)
                true_pos_rate.append(tr)
                fal_pos_rate.append(nr)
                total_time.append(tt)


            print('{} :: done'.format(model_name))

            model_accuracy.append(np.array(Accuracy))
            model_precision.append(np.array(precision))
            model_recall.append(np.array(recall))
            model_tr.append(np.array(true_pos_rate))
            model_nr.append(np.array(fal_pos_rate))
            model_tt.append(np.array(total_time))


        model_accuracy = np.array(model_accuracy)
        for i in range(NUM_MODELS):
            plt.plot(index, model_accuracy[i])
        plt.xlabel('Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy plot')
        plt.legend(MODEL_LIST,loc = 'upper left')
        plt.grid(True)
        plt.show()
        np.save('acc_old_data.npy',model_accuracy)

        model_tt = np.array(model_tt)
        for i in range(NUM_MODELS):
            plt.plot(index, model_tt[i])
        plt.xlabel('Threshold')
        plt.ylabel('Total Time')
        plt.title('Time plot')
        plt.legend(MODEL_LIST,loc = 'upper left')
        plt.grid(True)
        plt.show()
        # np.save('total_time_old_data.npy',model_tt)


        model_precision = np.array(model_precision)
        model_recall = np.array(model_recall)
        for i in range(NUM_MODELS):
            plt.plot(model_recall[i], model_precision[i])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(MODEL_LIST,loc = 'upper left')
        plt.grid(True)
        plt.show()
        np.save('precision_old_data.npy',model_precision)
        np.save('recall_old_data.npy',model_recall)


        model_tr = np.array(model_tr)
        model_nr = np.array(model_nr)
        for i in range(NUM_MODELS):
            plt.plot(model_nr[i], model_tr[i])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC Curve ')
        plt.legend(MODEL_LIST,loc = 'upper left')
        plt.grid(True)
        plt.show()
        np.save('tr_old_data.npy',model_tr)
        np.save('nr_old_data.npy',model_nr)

        print('----Timings-----')
        for i in range(NUM_MODELS):
            print('{} : {}'.format(MODEL_LIST[i],(model_tt[i][:-1])/1895.0))


