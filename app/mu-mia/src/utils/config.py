'''
   model_arch = ["vgg11_bn", "vgg13_bn", "vgg16_bn",
   "vgg19_bn", "resnet18", "resnet34", "resnet50", "densenet121",
   "densenet161", "densenet169", "mobilenet_v2", "googlenet", "inception_v3"]

   net = ['vgg16', 'vgg13', 'vgg11', 'vgg19',
   'densenet121', 'densenet161', 'densenet169', 'densenet201', 'googlenet',
   'inceptionv3', 'inceptionv4', 'inceptionresnetv2', 'xception', 'resnet18',
   'resnet34', 'resnet50', 'resnet101', 'resnet152', 'preactresnet18', 'preactresnet34',
   'preactresnet50', 'preactresnet101', 'preactresnet152', 'resnext50', 'resnext101', 'resnext152',
   'shufflenet', 'shufflenetv2', 'squeezenet', 'mobilenet', 'mobilenetv2', 'nasnet', 'attention56',
   'attention92', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'wideresnet',
   'stochasticdepth18', 'stochasticdepth34', 'stochasticdepth50', 'stochasticdepth101']

   When using the kid34k dataset,
   set data_type='kid34k' and n_classes=2.
   (the default architecture for the target/shadow model is resnet50.)
'''

device = 1
model_arch = "vgg11_bn" # for cifar10
net = 'vgg11' # for cifar100
n_classes = 100# [2, 10, 100]
data_type = f'cifar{n_classes}' # [cifar, kid34k]
attack_type = "nn_cls" # [nn, samia, nn_cls, siamese, (pre; not use)]

data_dir = f'/data/datasets/cifar{n_classes}-data'
save_dir = f'/miadata'

logger = 'tensorboard'
test_phase = 0
dev = 0

# Target/Shadow Trainer args (cifar10)
precision = 32
batch_size = 64
max_epochs = 200
num_workers = 10

learning_rate = 1e-2
weight_decay = 1e-2

# Target/Shadow Trainer args (cifar100)
warm = 1


#total training epoches
MILESTONES = [60, 120, 160]

# attack model params
atk_epochs = 100
m=10
epsilon=1e-3
a_lr_ = 1e-3
a_wd_ = 0

# unlearning params
test_batch_size = 1000
epochs = 15

gamma = 0.7
no_cuda = False
dry_run = False
seed = 0
log_interval = 10
save_model = False
pgd_eps = 2.0
pgd_alpha = 0.1
pgd_iter = 100
unlearn_label = 9
unlearn_k = 10
num_adv_images = None  # default is None, can be set later
reg_lamb = 10.0



# kr_celeb_unlenar_parms
dataset = 'VGGFace2'
data_path = '/miadata/mu/privacy_demo/data'
mode = ''

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
org_epochs = 8
lr = 1e-3
momentum = 0.9
weight_decay = 1e-4
evaluate = True
save_root_dir = './'
model_load_path = './VGGFace2_original_model.pth'
forget_class_idx = 9 #Dong-won
eps = 32.0
alpha = 1.0
iters = 10
denorm = True
unlearn_epochs = 8
unlearn_lr = 6e-4
unlearn_momentum = 0.9
unlearn_weight_decay = 1e-4