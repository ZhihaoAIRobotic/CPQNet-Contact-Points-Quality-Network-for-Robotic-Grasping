import torch
from cpnet import CPNET
from gcmap_dataset import Gcmap
import torch.optim as optim
import argparse
import tensorboardX

def parse_args():
    parser = argparse.ArgumentParser(description='Train GG-CNN')
    # Network
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batches_per_epoch', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=15000)
    parser.add_argument('--val_batches', type=int, default=150)


    args = parser.parse_args()
    return args


def validate(net, device, val_data, batches_per_epoch):

    net.eval()
    results = {
        'loss': 0,
    }

    with torch.no_grad():
        batch_idx = 0
        while batch_idx < batches_per_epoch:
            data_num=0
            for x, y, didx, rot, zoom_factor in val_data:
                batch_idx += 1
                data_num=data_num+1
                if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                    break
                xc = x.to(device)
                yc = y.to(device)
                lossd = net.compute_loss(xc, yc)
                loss = lossd['loss']
                results['loss'] += loss.item()/batches_per_epoch

    return results




def train(net, device, train_data, optimizer, batches_per_epoch):
    results = {
        'loss': 0,
    }

    net.train()

    batch_idx = 0
    while batch_idx < batches_per_epoch:
        for x, y, a_, b_, c_ in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            xc = x.to(device)
            yc = y.to(device)
            lossd = net.compute_loss(xc, yc)

            loss = lossd['loss']
            results['loss'] += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    results['loss'] /= 16
    results['loss'] /= batch_idx

    return results



def run():
    args = parse_args()
    writer = tensorboardX.SummaryWriter('log_lr0001_bs16_d1')  # 括号里为数据存放的地址
    Path_dataset = "grasp_map_480640_c.hdf5"

    # train dataset
    train_dataset = Gcmap(Path_dataset, start=0.0, end=0.9)
    # val dataset
    val_dataset = Gcmap(Path_dataset, start=0.9, end=1)

    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= 16 ,#args.batch_size,
        shuffle=True,
    )
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
    )

    net = CPNET()

    #net.load_state_dict(torch.load('model_test/model3_lr0001_bs16_d1.pkl'))
    device = torch.device("cuda:0")
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(),lr=0.001)
    print('Done')

    for epoch in range(args.epochs):
        train_results = train(epoch, net, device, train_data, optimizer,args.batches_per_epoch )#
        test_results = validate(net, device, val_data, args.val_batches)#
        torch.save(net.state_dict(), 'model/model.pkl')
        writer.add_scalars('Accu',{'v':test_results['loss'],'t':train_results['loss']},epoch)
        print(train_results)
        print(test_results)




if __name__ == '__main__':
    run()

