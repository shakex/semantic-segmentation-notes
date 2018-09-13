import os
import visdom
from tqdm import tqdm
from utils.loss import *
from torch.utils import data
from torch.autograd import Variable
from data import get_loader, get_data_path
from config import trainConfig
from utils.metrics import runningScore
from models import get_model

def train():
    hparam = trainConfig()

    # Print training information
    print('Training configuration:')
    print('-------------------------')
    print('Model: %s' % hparam.arch)
    print('Dataset: %s' % hparam.dataset)
    print('GPU: %s' % hparam.use_gpu)
    print('Epoches: %s' % hparam.n_epoch)
    print('Batch size: %s' % hparam.batch_size)
    print('Learning rate: %s' % hparam.l_rate)

    # Setup Augmentations
    print('\nInitialization')
    print('-------------------------')
    """
    print('Setup augmentations...', end='')
    data_aug = Compose([RandomRotate(10),
                        RandomHorizontallyFlip()])
    print('done.')
    """

    # Setup Dataloader
    print('Loading data...', end='')

    data_loader = get_loader(hparam.dataset)
    data_path = get_data_path(hparam.dataset)
    t_loader = data_loader(data_path, split='train', is_transform=True, augmentations=None)
    v_loader = data_loader(data_path, split='val', is_transform=True, augmentations=None)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=hparam.batch_size, num_workers=hparam.num_workers, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=hparam.batch_size, num_workers=hparam.num_workers)

    print('done.')

    # Setup Metrics
    print('Setup metrics...', end='')

    running_metrics = runningScore(n_classes)

    print('done.')

    # Setup visdom for visualization
    if hparam.visdom:
        print('Setup visdom...', end='')
        vis = visdom.Visdom()

        loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Loss']))
        print('done.')

    # Setup Model
    print('Setup model...')

    model = get_model(hparam.arch, n_classes)
    if hparam.use_gpu:
        model = torch.nn.DataParallel(model, device_ids=hparam.gpu_idx)
        model.cuda(hparam.gpu_idx[0])
        print('\tusing gpu: %d' % hparam.gpu_idx[0])

    # Check if model has custom optimizer / loss

    if hasattr(model.modules, 'optimizer'):
        optimizer = model.modules.optimizer
        print('\t[using custom optimizer]: todo')
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=hparam.l_rate, momentum=hparam.momentum, weight_decay=hparam.weight_decay)
        print('\tusing default optimizer: SGD')

    if hasattr(model.modules, 'loss'):
        loss_fn = model.module.loss
        print('\tusing custom loss: todo')
    else:
        loss_fn = cross_entropy2d
        print('\tusing default loss: Cross entropy')

    if hparam.resume is not None:
        if os.path.isfile(hparam.resume):
            print("\tloading model and optimizer from checkpoint '{}'...".format(hparam.resume))
            checkpoint = torch.load(hparam.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("\tloaded checkpoint '{}' (epoch {})"
                  .format(hparam.resume, checkpoint['epoch']))
        else:
            print("\tno checkpoint found at '{}'".format(hparam.resume))

    print('\tdone.')


    # Training
    print('\nTraining')
    print('-------------------------')

    best_iou = -100.0
    tpeList = []

    for epoch in range(hparam.n_epoch):
        print('Epoch[%d/%d]\t' % (epoch+1,hparam.n_epoch))
        # epoch_start = time.time()
        model.train()

        for i, (images, labels) in enumerate(trainloader):
            if hparam.use_gpu:
                images = Variable(images.cuda(hparam.gpu_idx[0]))
                labels = Variable(labels.cuda(hparam.gpu_idx[0]))
            else:
                images = Variable(images)
                labels = Variable(labels)

            optimizer.zero_grad()
            outputs = model(images)

            h_offset = int(np.abs(labels.size(1) - outputs.size(2)) / 2)
            w_offset = int(np.abs(labels.size(2) - outputs.size(3)) / 2)

            labels_resize = labels[:, h_offset:labels.size(1)-1-h_offset, w_offset:labels.size(2)-1-w_offset]
            loss = F.cross_entropy(input=outputs, target=labels_resize)
            # print('done.')

            # print('\t\tbackward pass...', end='')
            loss.backward()
            # print('done.')

            # print('\t\tupdate parameters...', end='')
            optimizer.step()
            # print('done.')

            if hparam.visdom:
                vis.line(
                    X=torch.ones(1).cpu() * i,
                    Y=torch.Tensor([loss.data[0]]).cpu(),
                    win=loss_window,
                    update='append')

            if (i+1) % hparam.print_freq == 0:
                print("\tloss: %.8f, (%d / %d)" % (loss.data[0], i+1, trainloader.sampler.num_samples))

        # print('\ttraining epoch[%d/%d] done.\n' % (epoch+1, hparam.n_epoch))

        model.eval()
        print('\tmodel(%s) now in evaluation mode.' % hparam.arch)
        for i, (images, labels) in tqdm(enumerate(valloader)):
            images = Variable(images.cuda(hparam.gpu_idx[0]), volatile=True)
            labels = Variable(labels.cuda(hparam.gpu_idx[0]), volatile=True)

            outputs = model(images)
            pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy())
            gt = np.squeeze(labels.data.cpu().numpy())

            h_offset = int(np.abs(gt.shape[0] - pred.shape[0]) / 2)
            w_offset = int(np.abs(gt.shape[1] - pred.shape[1]) / 2)

            gt_resize = gt[h_offset:gt.shape[0]-1-h_offset, w_offset:gt.shape[1]-1-w_offset]
            running_metrics.update(gt_resize, pred)

        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            print(k, v)
        running_metrics.reset()

        # time print

        # epoch_end = time.time()
        # tpe = epoch_end - epoch_start
        # print('\truntime: %.3fs (~ %dmin)' % (tpe, tpe/60))
        # tpeList.append(tpe)

        if score['Mean IoU : \t'] >= best_iou:
            best_iou = score['Mean IoU : \t']
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),}
            torch.save(state, "{}_{}_best_model.pkl".format(hparam.arch, hparam.dataset))

        # state = {'epoch': epoch+1,
        #          'model_state': model.state_dict(),
        #          'optimizer_state' : optimizer.state_dict(),}
        # torch.save(state, "{}_{}_{}_model.pkl".format(epoch, hparam.arch, hparam.dataset))

    # runTime = 0
    # for i, time in enumerate(tpeList):
    #     runTime += time
    # print('Total runing time: %.3fs (~ %dmin)' % (runTime, runTime/60))


from skimage import io
import matplotlib.pyplot as plt

def debug():

    hparam = trainConfig()
    data_loader = get_loader(hparam.dataset)
    data_path = get_data_path(hparam.dataset)
    t_loader = data_loader(data_path, split='train', is_transform=True, augmentations=None)
    v_loader = data_loader(data_path, split='val', is_transform=True, augmentations=None)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=hparam.batch_size, num_workers=hparam.num_workers, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=hparam.batch_size, num_workers=hparam.num_workers)

    img, lbl = t_loader.getitem(10)

    io.imshow(lbl)
    plt.show()

train()
# debug()






