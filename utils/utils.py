
'''
@author LeslieZhao
@date 20221221
'''
import torch 
from dataloader.AlignLoader import AlignData
from dataloader.BlendLoader import BlendData
import random
import numpy as np
import cv2


def requires_grad(model, flag=True):
    if model is None:
        return 
    for p in model.parameters():
        p.requires_grad = flag
def need_grad(x):
    x = x.detach()
    x.requires_grad_()
    return x

def init_weights(m,init_type='normal', gain=0.02):
        
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, gain)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            torch.nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        elif init_type == 'kaiming':
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight.data, gain=gain)
        elif init_type == 'none':  # uses pytorch's default init method
            m.reset_parameters()
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)
def setup_seed(seed):
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_data_loader(args):
    if args.model == 'align' or args.model == 'finetune':
        train_data = AlignData(dist=args.dist,
            size=args.size,
            root=args.train_root,
            eval=False)
        
        test_data = AlignData(dist=args.dist,
            size=args.size,
            root=args.val_root,
            eval=True)

    elif args.model == 'blend':
        train_data = BlendData(dist=args.dist,
            size=args.size,
            root=args.train_root,
            landscope_root=args.landscope_root,
            fabric_root=args.fabric_root,
            eval=False,
            use_affine_scale=True,
            use_affine_shift=True,)
        
        test_data = BlendData(dist=args.dist,
            size=args.size,
            root=args.val_root,
            landscope_root=args.landscope_root,
            fabric_root=args.fabric_root,
            eval=True,
            use_affine_scale=True,
            use_affine_shift=True)
    

    train_loader = torch.utils.data.DataLoader(
                        train_data,
                        batch_size=args.batch_size,
                        num_workers=args.nDataLoaderThread,
                        pin_memory=False,
                        drop_last=True
                    )
    test_loader = None if test_data is None else \
        torch.utils.data.DataLoader(
                        test_data,
                        batch_size=args.batch_size,
                        num_workers=args.nDataLoaderThread,
                        pin_memory=False,
                        drop_last=True
                    )
    return train_loader,test_loader,len(train_loader) 



def merge_args(args,params):
   for k,v in vars(params).items():
      setattr(args,k,v)
   return args

def convert_img(img,unit=False):
   
    img = (img + 1) * 0.5
    if unit:
        return torch.clamp(img*255+0.5,0,255)
    
    return torch.clamp(img,0,1)


def convert_flow_to_deformation(flow):
    r"""convert flow fields to deformations.

    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        deformation (tensor): The deformation used for warpping
    """
    b,c,h,w = flow.shape
    flow_norm = 2 * torch.cat([flow[:,:1,...]/(w-1),flow[:,1:,...]/(h-1)], 1)
    grid = make_coordinate_grid(flow)
    deformation = grid + flow_norm.permute(0,2,3,1)
    return deformation

def make_coordinate_grid(flow):
    r"""obtain coordinate grid with the same size as the flow filed.

    Args:
        flow (tensor): Flow field obtained by the model
    Returns:
        grid (tensor): The grid with the same size as the input flow
    """    
    b,c,h,w = flow.shape

    x = torch.arange(w).to(flow)
    y = torch.arange(h).to(flow)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    meshed = meshed.expand(b, -1, -1, -1)
    return meshed    

    
def warp_image(source_image, deformation):
    r"""warp the input image according to the deformation

    Args:
        source_image (tensor): source images to be warpped
        deformation (tensor): deformations used to warp the images; value in range (-1, 1)
    Returns:
        output (tensor): the warpped images
    """ 
    _, h_old, w_old, _ = deformation.shape
    _, _, h, w = source_image.shape
    if h_old != h or w_old != w:
        deformation = deformation.permute(0, 3, 1, 2)
        deformation = torch.nn.functional.interpolate(deformation, size=(h, w), mode='bilinear')
        deformation = deformation.permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(source_image, deformation) 


def add_list(x):
    r = None
   
    for v in x:
        r = v if r is None else r + v
    return r 

def compute_cosine(x):
    mm = np.matmul(x,x.T)
    norm = np.linalg.norm(x,axis=-1,keepdims=True)
    dis = mm / np.matmul(norm,norm.T)
    return  dis - np.eye(*dis.shape)

def compute_graph(cos_dis):
    index = np.where(np.triu(cos_dis) >= 0.68)
   
    # dd存放最终的图
    # vis存放各节点以及他们的root
    dd = {}
    vis = {}


    for i in np.unique(index[0]):
        
        # 此步用来存放根，因为是上三角，如果存在vis中，必不为root
        if i not in vis:
            vis[i] = i
            dd[vis[i]] = [i]
        
        for j in index[1][index[0]==i]:
            # 遍历行，不存在vis中的才为没有加入的，要将其root指向最终的root
            # 如果i为root，则vis[vis[i]] 为本身，如果i为节点，则vis[vis[i]]必为root
            # vis存放的k:val 只有两种形式 root:root, val:root
            if j not in vis:
                vis[j] = vis[vis[i]]
                dd[vis[i]] = dd.get(vis[i],[]) + [j]
            
            # 如果两簇有关联，进行合并
            elif j in vis and vis[vis[j]] != vis[vis[i]]:
                old_root = vis[vis[j]]
                for v in dd[vis[vis[j]]]:
                    dd[vis[vis[i]]] += [v]
                    vis[v] = vis[vis[i]] 
                del dd[old_root]

    for k,v in dd.items():
        dd[k] = list(set(v+[k]))
    return dd,index

def color_transfer2(background, face, center_ratio=1.0, mask=None):
    '''
    根据background 校正 face
    可选方式：
    0. default: 不填, 对整个crop区域校正
    1. center_ratio: 只用中心肤色区域校正，减少周围头发背景颜色差异的干扰
    2. mask: （255）
    '''
    source = cv2.cvtColor(
        np.rint(background).astype("uint8"),
        cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(
        np.rint(face).astype("uint8"),
        cv2.COLOR_BGR2LAB).astype("float32")

    (l_mean_src, l_std_src,
     a_mean_src, a_std_src,
     b_mean_src, b_std_src) = _image_stats(source,center_ratio, mask)
    (l_mean_tar, l_std_tar,
     a_mean_tar, a_std_tar,
     b_mean_tar, b_std_tar) = _image_stats(target,center_ratio, mask)

    (light, col_a, col_b) = cv2.split(target)
    light -= l_mean_tar
    col_a -= a_mean_tar
    col_b -= b_mean_tar

    light = (l_std_src / l_std_tar) * light
    col_a = (a_std_src / a_std_tar) * col_a
    col_b = (b_std_src / b_std_tar) * col_b

    light += l_mean_src
    col_a += a_mean_src
    col_b += b_mean_src

    light = np.clip(light, 0, 255)
    col_a = np.clip(col_a, 0, 255)
    col_b = np.clip(col_b, 0, 255)

    transfer = cv2.merge([light, col_a, col_b])
    transfer = cv2.cvtColor(
        transfer.astype("uint8"),
        cv2.COLOR_LAB2BGR)
    return transfer

def _image_stats(image, center_ratio, mask):
    if mask is None:
        h,w = image.shape[:2]
        t = int((1-center_ratio)/2.0*h)
        l = int((1-center_ratio)/2.0*w)
        image = image[t:h-t, l:w-l,:]
        (light, col_a, col_b) = cv2.split(image)
    else:
        (light, col_a, col_b) = cv2.split(image)
        mask = mask[:,:,0]
        light = np.masked_array(light, mask=mask<=240)  #位置： 1 无效， 0 有效
        col_a = np.masked_array(col_a, mask=mask<=240)  
        col_b = np.masked_array(col_b, mask=mask<=240)  
            
    (l_mean, l_std) = (light.mean(), light.std())
    (a_mean, a_std) = (col_a.mean(), col_a.std())
    (b_mean, b_std) = (col_b.mean(), col_b.std())

    return (l_mean, l_std, a_mean, a_std, b_mean, b_std)
