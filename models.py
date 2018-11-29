import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
def pretrained_models(model_num = 1):

    if int(model_num) == 1:
        model = models.alexnet(pretrained = True)
        model_name = 'AlexNet'
    elif int(model_num) == 2  :
        model = models.resnet18(pretrained = True)
        model_name = 'ResNet'
    elif int(model_num) == 3 :
        model = models.vgg16(pretrained = True)
        model_name = 'VGG16'
    elif int(model_num) == 4 :
        model = models.squeezenet1_0(pretrained = True)
        model_name = 'SqueezeNet'
    elif int(model_num) == 5 :
        model = models.densenet121(pretrained = True)
        model_name = 'DenseNet'
    elif int(model_num) == 6:
        model = models.resnet34(pretrained = True)
        model_name = 'ResNet'
    elif int(model_num) == 7:
        model = models.resnet50(pretrained = True)
        model_name = 'ResNet'
    elif int(model_num) == 8:
        model = models.resnet101(pretrained = True)
        model_name = 'ResNet'
    elif int(model_num) == 9:
        model = models.resnet152(pretrained = True)
        model_name = 'ResNet'
    elif int(model_num) == 10 :
        model = Custom_Alexnet()

        model_dict = model.state_dict() 
        pretrained_dict = torch.load('model_best_pretrainedAlexnet.pth.tar')
        pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model_name = 'Custom_Alexnet'


    return model,model_name


class Custom_Alexnet(nn.Module):

  def __init__(self):
    super(Custom_Alexnet,self).__init__()
    
    model = models.alexnet()
    self.features = model.features
    self.classifier_1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096))

def densenet(model,data):
    model.features[0] = torch.nn.DataParallel(model.features[0])
    out = model.features[0](data[0])
    out = model.features[1:-1](out)
    out = out.view(out.shape[0],-1)
    return out

def resnet(model,data):

    model.conv1 = torch.nn.DataParallel(model.conv1)

    out = model.maxpool(model.relu(model.bn1(model.conv1(data[0]))))

    out = model.layer1(out)

    out = model.layer2(out)

    out = model.layer3(out)

    out = model.layer4(out)

    out = model.avgpool(out)
    out = out.view(out.shape[0],-1)

    return out



def squeezenet(model,data):

    model.features = torch.nn.DataParallel(model.features)

    out = model.features(data[0])

    out = model.classifier[:-3](out)

    out = out.view(out.shape[0],-1)
    return out


def inception(model,data):
    model.Conv2d_1a_3x3 = torch.nn.DataParallel(model.Conv2d_1a_3x3)

    out = model.Conv2d_1a_3x3(data[0])
    out = model.Conv2d_2a_3x3(out)
    out = model.Conv2d_2b_3x3(out)
    out = model.Conv2d_3b_1x1(out)
    out = model.Conv2d_4a_3x3(out)
    out = model.Mixed_5b(out)
    out = model.Mixed_5c(out)
    out = model.Mixed_5d(out)
    out = model.Mixed_6a(out)
    out = model.Mixed_6b(out)
    out = model.Mixed_6c(out)
    out = model.Mixed_6d(out)
    out = model.Mixed_6e(out)
    out = model.AuxLogits.conv0(out)
    out = model.AuxLogits.conv1(out)
    out = out.view(out.shape[0],-1)
    return out

def alex_vgg(model,data):

    model.features = torch.nn.DataParallel(model.features)

    out = model.features(data[0])
    out = out.view(out.shape[0],-1)
    out = model.classifier[:-1](out)

    return out

def custom_alexnet(model,data):

    # model.features = torch.nn.DataParallel(model.features)

    out = model.features(data[0])

    out = out.view(out.shape[0],-1)
    out = model.classifier_1(out)

    return out
