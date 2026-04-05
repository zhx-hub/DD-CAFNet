import os
import torch
import torch.nn.functional as F
#import sys
#from dataset_SOD_T_v3 import customDataset
from dataset import  SalObjDataset,TestDataset
#sys.path.append('./models')
import numpy as np
from datetime import datetime
from models.DDCAFNet_swinv2_B_384 import Net
from torchvision.utils import make_grid
from tools.data import get_loader, test_dataset
from tools.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging

# RGBD
# RGBD
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=300, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
parser.add_argument('--load_pre', type=str, default='./DDCAFNet_RGBD_cpts/DDCAFNet_epoch_best.pth', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
parser.add_argument("--backbonet_path", type=str, default="./swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth")
parser.add_argument('--mean', default=(0.485, 0.456, 0.406))
parser.add_argument('--std', default=(0.229, 0.224, 0.225))
parser.add_argument('--save_path', type=str, default="./ck/", help='the path to save models and logs')
opt = parser.parse_args()

# set the device for training
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
#cudnn.benchmark = True

save_path = opt.save_path
dataset_train = SalObjDataset('./SOD-T/train_2985/',384 )
dataset_val = TestDataset('./SOD-T/test_groups/DUT-RGBD/',384 )
train_loader = torch.utils.data.DataLoader(
    dataset_train, 
    batch_size=opt.batchsize,
    num_workers=opt.num_workers,
    pin_memory=True,
    shuffle=True,

)

test_loader = torch.utils.data.DataLoader(
    dataset_val, 
    batch_size=int(1.5 * opt.batchsize),
    num_workers=opt.num_workers,
    pin_memory=True,
    shuffle=True,
)


logging.basicConfig(filename=save_path + 'RGBD.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("DDCAFNet-Train_4_pairs")

# build the model
model = Net(ckpt=opt.backbonet_path,img_size=(384,384))
model.init_weights()
num_parms = 0




#model.cuda()

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path
if not os.path.exists(save_path):
    os.makedirs(save_path)


# load data
print('load data...')
total_step = len(train_loader)

logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
CE = torch.nn.BCEWithLogitsLoss()
def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

    
step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0
model=model.to('cuda')

backbone_params = []
other_params = []

for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if "backbone" in name:
        backbone_params.append(p)
    else:
        other_params.append(p)

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": opt.lr * 0.1},
    {"params": other_params, "lr": opt.lr},
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=opt.epoch, eta_min=1e-6
)

CE = torch.nn.BCEWithLogitsLoss()
scaler = torch.cuda.amp.GradScaler()


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()

    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts, depth, edge, _) in enumerate(train_loader, start=1):
            images = images.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            depth = depth.cuda(non_blocking=True)
            edge = edge.cuda(non_blocking=True)
    
            optimizer.zero_grad(set_to_none=True)
    
            with torch.cuda.amp.autocast():
                s, e = model(images, depth)
    
                sal_loss = structure_loss(s, gts)
                edge_loss = structure_loss(e, edge)
                
                if not torch.isfinite(sal_loss):
                    print(f"Step {i}: sal_loss invalid")
                    raise RuntimeError("sal_loss invalid")
            
                if not torch.isfinite(edge_loss):
                    print(f"Step {i}: edge_loss invalid")
                    print("e min/max:", e.detach().min().item(), e.detach().max().item())
                    print("edge min/max:", edge.detach().min().item(), edge.detach().max().item())
                    raise RuntimeError("edge_loss invalid")

                loss = sal_loss + 0.3 * edge_loss
            
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
    
            step += 1
            epoch_step += 1
            loss_all += loss.item()
    
        
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}||sal_loss:{:4f} ||edge_loss:{:4f} ||sal_loss1:{:4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           optimizer.state_dict()['param_groups'][0]['lr'], sal_loss.data, edge_loss.data,
                           sal_loss.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f} ||edge_loss:{:4f}||sal_loss1:{:4f} , mem_use:{:.0f}MB'.
                    format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'],
                           sal_loss.data, edge_loss.data, sal_loss.data, memory_used))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = s[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('res', torch.tensor(res), step, dataformats='HW')
        scheduler.step()
        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 20== 0 or (epoch)>=295:
           
            torch.save(model.state_dict(), save_path + 'DDCAFNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'DDCAFNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        #for i in range(test_loader.size):
        for i, (images, gts, depth, edge,_) in enumerate(test_loader, start=1):
            image, gt, depth,  = images, gts, depth,#test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            res,e = model(image, depth)
            #res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / len(dataset_val)
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'DDCAFNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):
        # if (epoch % 50 ==0 and epoch < 60):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)

        test(test_loader, model, epoch, save_path)