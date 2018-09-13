
class trainConfig(object):

    dataset = 'brainweb'    # DataSet to use
    arch = 'unet'           # Architecture to use
    visdom = False          # Show visualization(s) on visdom
    use_gpu = True          # use GPU or not
    gpu_idx = [1]           # set GPU id
    load_model_path = None  # load the path of pre-trained model
    # resume = '/home/jwliu/disk/kxie/semseg-pytorch/unet_brainweb_best_model.pkl'
    resume = None
    # Path to previous saved model to restart from

    n_epoch = 100          # number of epochs
    batch_size = 1          # batch size
    l_rate = 1e-6           # initial learning rate
    num_workers = 4         # number of workers for loading data
    lr_decay = 0.95         # learning rate decay: when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4     # Weight decay
    momentum = 0.99         # Momentum
    print_freq = 100        # print info every N batch

    debug_file = '/tmp/debug'   # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

class testConfig(object):

    # model name
    arch = 'unet'
    # img_norm
    img_norm = True
    # set GPU id
    gpu_idx = [2]
    # Enable DenseCRF based post-processing
    dcrf = False
    # Path to the saved model
    model_path = '/home/jwliu/disk/kxie/semseg-pytorch/trained-model/unet_brainweb_0611_1e06_1000.pkl'
    # Path of the input image
    img_path = '/home/jwliu/disk/kxie/semseg-pytorch/img/s04_t093.bmp'
    # Path of the output segmap
    out_path = '/home/jwliu/disk/kxie/semseg-pytorch/results/pred_s04_t093.bmp'
    # Dataset to use
    dataset = 'brainweb'

