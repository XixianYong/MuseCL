import argparse
import copy
import csv
import os
import random
import time

import torch
import torchvision.transforms.functional as TF
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from models.inceptionv3 import PlaceImageSkipGram
from utils.image_dataset import PlaceImagePairDataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters of UR training')
    parser.add_argument('--embedding_dim',
                        type=int,
                        default=128)
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.0005)
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0)
    parser.add_argument('--margin',
                        type=int,
                        default=5)
    parser.add_argument('--batch_size',
                        type=int,
                        default=16)
    parser.add_argument('--num_epochs',
                        type=int,
                        default=20)
    parser.add_argument('--lr_decay_rate',
                        type=float,
                        default=0.7)
    parser.add_argument('--lr_decay_epochs',
                        type=int,
                        default=10)

    return parser.parse_args()


def RandomRotationNew(image):
    angle = random.choice([0, 90, 180, 270])
    image = TF.rotate(image, angle)
    return image


def metrics(stats):
    accuracy = (stats['T'] + 0.00001) * 1.0 / (stats['T'] + stats['F'] + 0.00001)
    return accuracy


def train_embedding(model, model_name, dataloaders, criterion, optimizer, num_epochs, metrics, verbose=True,
                    return_best=True,
                    if_early_stop=True, early_stop_epochs=20, scheduler=None,
                    save_dir=None, save_epochs=1, dist=None):
    since = time.time()
    training_log = dict()
    # embs = []

    training_log['train_loss_history'] = []
    training_log['val_loss_history'] = []
    training_log['val_metric_value_history'] = []
    training_log['current_epoch'] = -1
    current_epoch = training_log['current_epoch'] + 1

    best_model_wts = copy.deepcopy(model.state_dict())
    best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
    best_log = copy.deepcopy(training_log)

    best_metric_value = 0.
    nodecrease = 0  # to count the epochs that val loss doesn't decrease
    early_stop = False

    for epoch in range(current_epoch, current_epoch + num_epochs):
        if verbose:
            print('-' * 20)
            print('     Epoch {}/{}'.format(epoch, num_epochs + current_epoch - 1))
            print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            stats = {'T': 0, 'F': 0}

            # Iterate over data.
            for anc_images, pos_images, neg_images in tqdm(dataloaders[phase]):
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # print(anc_images)
                    anc_images = anc_images.to(device)  # anc_images
                    pos_images = pos_images.to(device)  # pos_images
                    neg_images = neg_images.to(device)  # neg_images

                    outputs1 = model(anc_images)
                    outputs2 = model(pos_images)
                    outputs3 = model(neg_images)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # Get model outputs and calculate loss
                    distance1 = dist(outputs1, outputs2)
                    distance2 = dist(outputs1, outputs3)

                    labels = torch.ones(len(distance1))  # [1, ...]
                    labels = labels.to(device)
                    loss = criterion(distance2, distance1, target=labels)
                    print('batch loss:', loss.item())
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    del anc_images
                    del pos_images
                    del outputs1
                    del outputs2
                    del outputs3
                    del labels
                # statistics
                # running_loss += loss.item() * fips.size(0)
                running_loss += loss.item()
                stats['T'] += torch.sum(distance1 < distance2).cpu().item()
                stats['F'] += torch.sum(distance1 > distance2).cpu().item()
                # print(stats['F'])
                del distance1
                del distance2
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_metric_value = metrics(stats)
            # embs.append(outputs1.detach().cpu().numpy())

            if verbose:
                print('{} Loss: {:.4f} Metrics: {:.4f}'.format(phase, epoch_loss, epoch_metric_value))

            training_log['current_epoch'] = epoch
            if phase == 'val':
                training_log['val_metric_value_history'].append(epoch_metric_value)
                training_log['val_loss_history'].append(epoch_loss)
                # deep copy the model
                if epoch_metric_value > best_metric_value:
                    best_metric_value = epoch_metric_value
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
                    best_log = copy.deepcopy(training_log)
                    nodecrease = 0
                else:
                    nodecrease += 1
            else:
                training_log['train_loss_history'].append(epoch_loss)
                if scheduler is not None:
                    scheduler.step()

            if nodecrease >= early_stop_epochs:
                early_stop = True

        if save_dir and epoch % save_epochs == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_log': training_log
            }
            torch.save(checkpoint,
                       os.path.join(save_dir, model_name + '_' + str(training_log['current_epoch']) + '.tar'))

        if if_early_stop and early_stop:
            print('Early stopped!')
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation metric value: {:4f}'.format(best_metric_value))

    # load best model weights
    if return_best:
        model.load_state_dict(best_model_wts)
        optimizer.load_state_dict(best_optimizer_wts)
        training_log = best_log

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_log': training_log
    }

    if save_dir:
        torch.save(checkpoint,
                   os.path.join(save_dir, model_name + '_' + str(training_log['current_epoch']) + '_last.tar'))
    return model, training_log, best_metric_value


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(299),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize(299),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

if __name__ == "__main__":
    print('BEGIN!!!')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)
    args = parse_arguments()
    data_dir = ""
    media_dir = ""

    train_csv_path = ""
    train_csv = csv.reader(open(train_csv_path, encoding='UTF-8-sig'))
    train_path_list = list(train_csv)  # [6666/name.jpg, 6667/name.jpg, 6668/name.jpg]

    val_path_path = ""
    val_csv = csv.reader(open(val_path_path, encoding='UTF-8-sig'))
    val_path_list = list(val_csv)

    verbose = True
    return_best = True
    if_early_stop = True
    early_stop_epochs = 20

    pdist = torch.nn.PairwiseDistance(p=2.0)

    datasets = {'train': PlaceImagePairDataset(data_dir, train_path_list, data_transforms['train']),
                'val': PlaceImagePairDataset(data_dir, val_path_list, data_transforms['val'])}

    dataloaders_dict = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                      shuffle=True, num_workers=0) for x in ['train', 'val']}

    model = PlaceImageSkipGram(embedding_dim=args.embedding_dim)
    model.load_CNN_params("")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=args.weight_decay, amsgrad=True)

    loss_fn = torch.nn.MarginRankingLoss(margin=args.margin)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_epochs, gamma=args.lr_decay_rate)

    _, training_log, best_value = train_embedding(model, model_name="NY_SV_128", dataloaders=dataloaders_dict,
                                                  criterion=loss_fn,
                                                  optimizer=optimizer, num_epochs=args.num_epochs,
                                                  metrics=metrics,
                                                  verbose=verbose, return_best=return_best,
                                                  if_early_stop=if_early_stop,
                                                  early_stop_epochs=early_stop_epochs, scheduler=scheduler,
                                                  save_dir=media_dir, save_epochs=1, dist=pdist)

    media_dir1 = os.path.join(media_dir, "training_log.txt")
    with open(media_dir1, "w") as file:
        for k in range(len(training_log["val_metric_value_history"])):
            file.write("epoch:" + str(k) + "\n")
            file.write("val_metric_value_history:" + str(training_log["val_metric_value_history"][k]) + "\n")
            file.write("val_loss_history:" + str(training_log["val_loss_history"][k]) + "\n")
            file.write("train_loss_history:" + str(training_log["train_loss_history"][k]) + "\n")

    print("DONE!!!")
