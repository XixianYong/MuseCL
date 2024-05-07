import argparse
import copy
import csv
import os
import random
import time

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from models.resnet18 import RestNet18
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
                        default=32)
    parser.add_argument('--num_epochs',
                        type=int,
                        default=201)
    parser.add_argument('--lr_decay_rate',
                        type=float,
                        default=0.7)
    parser.add_argument('--lr_decay_epochs',
                        type=int,
                        default=10)
    parser.add_argument('--model_ckpt_path',
                        type=str,
                        default="")

    return parser.parse_args()


def RandomRotationNew(image):
    angle = random.choice([0, 90, 180, 270])
    image = TF.rotate(image, angle)
    return image


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_resnet18(pretrained, output_dim, ckpt_path):
    model = torchvision.models.resnet18(pretrained=pretrained)
    model.fc = Identity()
    checkpoint = torch.load(ckpt_path)
    checkpoint.pop('fc.weight')
    checkpoint.pop('fc.bias')
    model.load_state_dict(checkpoint)
    model.fc = torch.nn.Linear(512, output_dim)
    return model


def nt_xent_loss(x, y, temperature=0.5):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    features = torch.cat([x, y], dim=0)
    similarities = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
    mask = torch.eye(2 * len(x)).bool()
    similarities = similarities[~mask].reshape(2 * len(x), -1)
    numerator = torch.exp(F.cosine_similarity(x, y) / temperature)
    denominator = torch.exp(similarities / temperature).sum(dim=1)[:len(x)].reshape(len(x), ) + 1e-6
    loss = -torch.log(numerator / denominator).mean()

    return loss


def train_embedding(model, model_name, dataloaders, criterion, optimizer, num_epochs, verbose=True, return_best=True,
                    if_early_stop=False, early_stop_epochs=20, scheduler=None, save_dir=None, save_epochs=10, dist=None):
    since = time.time()
    training_log = dict()

    training_log['train_loss_history'] = []
    training_log['val_loss_history'] = []
    training_log['current_epoch'] = -1
    current_epoch = training_log['current_epoch'] + 1

    best_model_wts = copy.deepcopy(model.state_dict())
    best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
    best_log = copy.deepcopy(training_log)

    best_val_loss = float('inf')
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

            # Iterate over data.
            for anc_images, pos_images, neg_images in tqdm(dataloaders[phase]):
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        anc_images = RandomRotationNew(anc_images)  # anc_images
                        pos_images = RandomRotationNew(pos_images)  # pos_images
                        neg_images = RandomRotationNew(neg_images)  # neg_images

                    anc_images = anc_images.to(device)
                    pos_images = pos_images.to(device)
                    neg_images = neg_images.to(device)

                    outputs1 = model(anc_images)
                    outputs2 = model(pos_images)
                    outputs3 = model(neg_images)
                    optimizer.zero_grad()
                    distance1 = dist(outputs1, outputs2)
                    distance2 = dist(outputs1, outputs3)
                    labels = torch.ones(len(distance1))
                    labels = labels.to(device)
                    loss = criterion(distance2, distance1, target=labels)
                    print('batch loss:', loss.item())
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    del anc_images
                    del pos_images
                    del neg_images
                    del outputs1
                    del outputs2
                    del outputs3

                # statistics
                running_loss += loss.item()
                del distance1
                del distance2
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if verbose:
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            training_log['current_epoch'] = epoch
            if phase == 'val':
                training_log['val_loss_history'].append(epoch_loss)
                # deep copy the model
                if best_val_loss > epoch_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_optimizer_wts = copy.deepcopy(optimizer.state_dict())
                    best_log = copy.deepcopy(training_log)
                    nodecrease = 0
                else:
                    nodecrease += 1
            else:  # train phase
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
    return model, training_log


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ]),
}

if __name__ == "__main__":
    print('BEGIN!!!!!')
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
    if_early_stop = False
    early_stop_epochs = 20

    pdist = torch.nn.PairwiseDistance(p=2.0)
    datasets = {'train': PlaceImagePairDataset(data_dir, train_path_list, data_transforms['train']),
                'val': PlaceImagePairDataset(data_dir, val_path_list, data_transforms['val'])}
    dataloaders_dict = {x: DataLoader(datasets[x], batch_size=args.batch_size,
                                      shuffle=True, num_workers=0) for x in ['train', 'val']}

    model = get_resnet18(pretrained=True, output_dim=args.embedding_dim, ckpt_path=args.model_ckpt_path)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08,
                           weight_decay=args.weight_decay, amsgrad=True)

    loss_fn = torch.nn.MarginRankingLoss(margin=args.margin)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_epochs, gamma=args.lr_decay_rate)

    _, training_log = train_embedding(model, model_name="NY_RV_128_triplet", dataloaders=dataloaders_dict,
                                      criterion=loss_fn,
                                      optimizer=optimizer, num_epochs=args.num_epochs,
                                      verbose=verbose, return_best=return_best,
                                      if_early_stop=if_early_stop,
                                      early_stop_epochs=early_stop_epochs, scheduler=scheduler,
                                      save_dir=media_dir, save_epochs=10, dist=pdist)

    media_dir1 = os.path.join(media_dir, "training_log.txt")
    with open(media_dir1, "w") as file:
        for k in range(len(training_log["train_loss_history"])):
            file.write("epoch:" + str(k) + "\n")
            file.write("val_loss_history:" + str(training_log["val_loss_history"][k]) + "\n")
            file.write("train_loss_history:" + str(training_log["train_loss_history"][k]) + "\n")

    print("DONE!!!")
