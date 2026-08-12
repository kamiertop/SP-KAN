from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime


def configure_cuda_visible_devices(argv: list[str]) -> str | None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--gpu_id", "--gpu", dest="gpu_id", choices=["0", "1"])
    pre_args, _ = pre_parser.parse_known_args(argv)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if pre_args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = pre_args.gpu_id
    return pre_args.gpu_id


_requested_gpu_id = configure_cuda_visible_devices(sys.argv[1:])

from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.dataset_resize import *
from util.metrics import *
from util.utils import *
from torch.utils.tensorboard import SummaryWriter
from util.train_helpers import Net, align_prediction_to_target, save_checkpoint, weights_init_kaiming
from util.data_split import split_train_validation

parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
parser.add_argument("--model_names", default=['SP_KAN'], nargs='+', help="Models to train")
parser.add_argument("--dataset_names", default=['SIRST3'],
                    nargs='+')  # SIRST3： NUAA NUDT-SIRST IRSTD-1K  SIRST3   SIRST3_Enhance
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: AdamW, Adam, Adagrad, SGD")
parser.add_argument("--epochs", default=1000, type=int, help="numbers of epoch")

parser.add_argument("--begin_validation", "--begin_test", dest="begin_validation", default=1, type=int,
                    help="Epoch at which validation starts (legacy alias: --begin_test)")
parser.add_argument("--every_validation", "--every_test", dest="every_validation", default=1, type=int,
                    help="Validate every N epochs (legacy alias: --every_test)")
parser.add_argument("--every_print", default=10, type=int)

parser.add_argument("--dataset_dir", default=r'./datasets')
parser.add_argument("--batchSize", type=int, default=8, help="Training batch sizse")
parser.add_argument("--gpu_id", "--gpu", dest="gpu_id", choices=["0", "1"], default=_requested_gpu_id,
                    help="Physical GPU id exposed to this process via CUDA_VISIBLE_DEVICES")
# ******************* Others   *******************
parser.add_argument("--patchSize", type=int, default=512, help="Training patch size")
parser.add_argument("--patchSize_eva", type=int, default=512, help="Evaluation patch size")
parser.add_argument("--save", default=r'./runs', type=str, help="Root directory for timestamped run artifacts")
parser.add_argument("--log_dir", type=str, default=None,
                    help='Optional separate TensorBoard root (default: <run_dir>/tensorboard)')
parser.add_argument("--img_norm_cfg", default=None)
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use")
parser.add_argument("--max_train_steps", type=int, default=None,
                    help="Optional cap on training batches per epoch (useful for smoke tests)")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
parser.add_argument("--val_ratio", type=float, default=0.1,
                    help="Fraction of train_*.txt reserved for validation")
parser.add_argument("--validation_source", choices=['train_split', 'official_test'], default='train_split',
                    help="Evaluation data: hold out part of train list, or reuse the official test list per epoch")
parser.add_argument("--early_stopping_patience", type=int, default=50,
                    help="Stop after this many validation checks without mIoU improvement (0 disables)")
parser.add_argument("--early_stopping_min_delta", type=float, default=1e-5,
                    help="Minimum validation mIoU improvement counted by early stopping")
parser.add_argument("--min_epochs", type=int, default=1,
                    help="Do not early-stop before this many epochs")
parser.add_argument("--auto_test", action=argparse.BooleanOptionalAction, default=True,
                    help="Run official test.py automatically after training")
parser.add_argument("--auto_test_save_img", action=argparse.BooleanOptionalAction, default=False,
                    help="Save prediction images during automatic final test")
parser.add_argument("--resume", default=False, help="Resume from an existing checkpoint")
parser.add_argument("--loss_name", choices=['target_aware', 'bce'], default='target_aware',
                    help="Training loss; target_aware is the TABDS innovation, bce reproduces the baseline")
parser.add_argument("--loss_boundary_weight", type=float, default=2.0,
                    help="Boundary emphasis used by target_aware loss")
parser.add_argument("--loss_dice_weight", type=float, default=1.0,
                    help="Dice term weight used by target_aware loss")
parser.add_argument("--loss_max_pos_weight", type=float, default=20.0,
                    help="Maximum per-batch foreground reweighting")
parser.add_argument("--cross_view", action=argparse.BooleanOptionalAction, default=False,
                    help="Enable clean/noisy cross-view alignment and Top-K background fusion")
parser.add_argument("--cross_view_noise_std", type=float, default=0.03)
parser.add_argument("--cross_view_topk", type=float, default=0.2,
                    help="Fraction of lowest-response locations used as background prototypes")
parser.add_argument("--cross_view_consistency_weight", type=float, default=0.1,
                    help="Weight of clean/noisy prediction consistency loss")
parser.add_argument("--mamba_branch", action=argparse.BooleanOptionalAction, default=False,
                    help="Enable MiM-ISTD local/global selective state-space block at deepest stage")

global opt
opt = parser.parse_args()
seed_pytorch(opt.seed)
print("---------------------------------------------------------------")
print('batchSize: {0} -- begin_validation: {1} -- every_print: {2} -- every_validation: {3}'.format(opt.batchSize, opt.begin_validation,
                                                                                        opt.every_print,
                                                                                        opt.every_validation))


def resolve_training_and_validation_names(
        dataset_dir: str, dataset_name: str, validation_source: str,
        val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    """Read the train list and choose the periodic evaluation list."""
    index_dir = os.path.join(dataset_dir, dataset_name, 'img_idx')
    with open(os.path.join(index_dir, 'train_' + dataset_name + '.txt'), encoding='utf-8') as list_file:
        train_names = list_file.read().splitlines()

    if validation_source == 'official_test':
        with open(os.path.join(index_dir, 'test_' + dataset_name + '.txt'), encoding='utf-8') as list_file:
            return train_names, list_file.read().splitlines()
    return split_train_validation(train_names, val_ratio, seed)


def train() -> None:
    # *******************************************************************************************************
    #                                             Train
    # *******************************************************************************************************
    train_names, validation_names = resolve_training_and_validation_names(
        opt.dataset_dir, opt.dataset_name, opt.validation_source, opt.val_ratio, opt.seed)
    if not train_names or not validation_names:
        raise ValueError('training and periodic evaluation lists must both contain at least one sample')
    run_config = {}
    for key, value in vars(opt).items():
        if key == 'f':
            continue
        try:
            json.dumps(value)
        except TypeError:
            continue
        run_config[key] = value
    run_config.update({
        'run_id': opt.run_id,
        'run_dir': opt.run_dir,
        'checkpoint_dir': opt.checkpoint_dir,
        'tensorboard_dir': opt.log_dir,
        'train_samples': len(train_names),
        'validation_samples': len(validation_names),
        'validation_source': opt.validation_source,
    })
    with open(opt.params_path, 'w', encoding='utf-8') as params_file:
        json.dump(run_config, params_file, ensure_ascii=True, indent=2)
        params_file.write('\n')
    train_set = TrainSetLoader_Re_Pad(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name,
                               patch_size=opt.patchSize, img_norm_cfg=opt.img_norm_cfg,
                               sample_list=train_names, augment=True)
    worker_options = {'persistent_workers': True} if opt.threads else {}
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                              shuffle=True, **worker_options)

    validation_set = TestSetLoader_Re_Pad(
        opt.dataset_dir, opt.dataset_name, opt.dataset_name, opt.patchSize_eva,
        img_norm_cfg=opt.img_norm_cfg, sample_list=validation_names)
    validation_loader = DataLoader(dataset=validation_set, num_workers=opt.threads, batch_size=1,
                             shuffle=False, **worker_options)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net(model_name=opt.model_name, mode='train', loss_name=opt.loss_name,
              boundary_weight=opt.loss_boundary_weight, dice_weight=opt.loss_dice_weight,
              max_pos_weight=opt.loss_max_pos_weight, cross_view=opt.cross_view,
              cross_view_noise_std=opt.cross_view_noise_std,
              cross_view_topk=opt.cross_view_topk,
              cross_view_consistency_weight=opt.cross_view_consistency_weight,
              mamba_branch=opt.mamba_branch).to(device)
    net.apply(weights_init_kaiming)
    net.train()
    total_loss_list = []

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)
    writer = SummaryWriter(opt.log_dir)

    ### Default settings of SP_KAN
    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 0.001}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs': opt.epochs, 'eta_min': 1e-5, 'last_epoch': -1}


    opt.nEpochs = opt.scheduler_settings['epochs']
    if opt.early_stopping_patience < 0:
        raise ValueError('early_stopping_patience must be non-negative')
    if opt.early_stopping_min_delta < 0:
        raise ValueError('early_stopping_min_delta must be non-negative')
    if opt.min_epochs < 1:
        raise ValueError('min_epochs must be at least 1')
    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings,
                                         opt.scheduler_settings)

    best_mIOU = [0, -float('inf')]
    best_Pd_Fa = [0, 1]
    best_validation_miou = -float('inf')
    validation_checks_without_improvement = 0
    opt.best_checkpoint_path = None

    for idx_epoch in range(0, opt.nEpochs):
        epoch_started = time.time()
        epoch_loss_values = []
        net.train()
        results1 = [0, 0]
        results2 = [0, 1]
        optimizer_steps = 0
        for img, gt_mask in train_loader:
            img, gt_mask = Variable(img).to(device), Variable(gt_mask).to(device)
            pred = net.forward(img)
            loss = net.loss(pred, gt_mask)
            epoch_loss_values.append(float(loss.detach().cpu()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            if opt.max_train_steps is not None and optimizer_steps >= opt.max_train_steps:
                break
        if not epoch_loss_values:
            raise RuntimeError('No optimization steps completed in this epoch.')
        scheduler.step()

        epoch_loss = float(np.mean(epoch_loss_values))
        total_loss_list.append(epoch_loss)
        epoch_seconds = time.time() - epoch_started
        epoch_log = time.ctime()[4:-5] + ' Epoch---%d, train_loss---%f, lr---%f, epoch_seconds---%.3f' \
            % (idx_epoch + 1, epoch_loss, scheduler.get_last_lr()[0], epoch_seconds)
        print(epoch_log)
        opt.f.write(epoch_log + '\n')
        opt.f.flush()

        if (idx_epoch + 1) % opt.every_print == 0:  # tensorboard : write train loss
            # Log the scalar values
            writer.add_scalar('loss', epoch_loss, idx_epoch + 1)
            writer.add_scalar('lr', scheduler.get_last_lr()[0], idx_epoch + 1)

        if (idx_epoch + 1) >= opt.begin_validation and (
                idx_epoch + 1) % opt.every_validation == 0:  # TensorBoard: validation metrics
            # *******************************************************************************************************
            #                                          Validation
            # *******************************************************************************************************
            net.eval()
            with torch.no_grad():
                eval_mIoU = mIoU()
                eval_PD_FA = PD_FA()
                Metric = F1(opt.threshold)
                validation_loss_values = []
                for idx_iter, (img, gt_mask, target_size, org_size, _) in enumerate(validation_loader):
                    img = Variable(img).to(device)
                    gt_mask = gt_mask.to(device)
                    pred = net.forward(img)
                    if isinstance(pred, tuple):
                        pred = pred[-1]
                    elif isinstance(pred, list):
                        pred = pred[-1]
                    else:
                        pred = pred

                    pred = align_prediction_to_target(pred, gt_mask, target_size, org_size)
                    validation_loss_values.append(float(net.loss(pred, gt_mask).detach().cpu()))
                    eval_mIoU.update((pred > opt.threshold).cpu(), gt_mask.cpu())
                    eval_PD_FA.update((pred[0, 0, :, :] > opt.threshold).cpu(),
                                      gt_mask[0, 0, :, :].cpu(), org_size)
                    Metric.update(labels=gt_mask.cpu(), preds=pred.cpu())

                results1 = eval_mIoU.get()
                results2 = eval_PD_FA.get()
                f1_score = Metric.get()
                val_loss = float(np.mean(validation_loss_values))
                validation_improved = results1[1] > best_validation_miou + opt.early_stopping_min_delta
                if validation_improved:
                    best_validation_miou = results1[1]
                    validation_checks_without_improvement = 0
                else:
                    validation_checks_without_improvement += 1
                writer.add_scalar('val_loss', val_loss, idx_epoch + 1)
                writer.add_scalar('mIOU', results1[-1], idx_epoch + 1)
                writer.add_scalar('F1', f1_score, idx_epoch + 1)
                writer.add_scalar('Pd', results2[0], idx_epoch + 1)
                writer.add_scalar('Fa', results2[1], idx_epoch + 1)
                print('Validation---Epoch---%d, val_loss---%f, mIoU---%f, F1---%f'
                      % (idx_epoch + 1, val_loss, results1[1], f1_score))
                opt.f.write('Validation---Epoch---%d, val_loss---%f, mIoU---%f, F1---%f\n'
                            % (idx_epoch + 1, val_loss, results1[1], f1_score))
                opt.f.flush()

            # IOU
            if results1[1] > best_mIOU[1]:
                best_mIOU = results1
                print('------save the best model epoch', opt.model_name, '_%d ------' % (idx_epoch + 1))
                opt.f.write("the best model epoch \t" + str(idx_epoch + 1) + '\n')
                print("mIoU, F1:\t" + str(results1[1]) + ", " + str(f1_score))
                # print("testloss:\t" + str(test_loss[-1]))
                print("PD, FA :\t" + str(results2))
                opt.f.write("mIoU: " + str(results1[1]) + '\n')
                opt.f.write("PD, FA :\t" + str(results2) + '\n')
                best_IOU = format(results1[1], '.4f')
                best_Pd = format(results2[0], '.4f')
                best_Fa = format(results2[1], '.6f')

                save_pth = opt.checkpoint_dir + '/' + opt.model_name + '_' + str(
                    idx_epoch + 1) + '_' + str(best_IOU) + "_" + str(best_Pd) + "_" + str(best_Fa) + "_" + '.pth.tar'
                save_checkpoint({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                    'total_loss': total_loss_list,
                }, save_pth)
                opt.best_checkpoint_path = save_pth

            elif results2[0] > 0.978 and results2[1] < 1e-5:
                # best_Pd = results2
                print('------save the best model epoch', opt.model_name, '_%d ------PdPd' % (idx_epoch + 1))
                opt.f.write("the best model epoch \t" + str(idx_epoch + 1) + '\n')
                print("mIoU, F1:\t" + str(results1[1]) + ", " + str(f1_score))
                print("PD, FA:\t" + str(results2))
                opt.f.write("mIoU: " + str(results1[1]) + '\n')
                opt.f.write("PD, FA :\t" + str(results2) + '\n')
                best_IOU = format(results1[1], '.4f')
                best_Pd = format(results2[0], '.4f')
                best_Fa = format(results2[1], '.6f')
                save_pth = opt.checkpoint_dir + '/' + opt.model_name + 'PdPd' + '_Epoch' + str(
                    idx_epoch + 1) + '_' + str(best_IOU) + '_' + str(best_Pd) + "_" + str(best_Fa) + "_" + '.pth.tar'
                save_checkpoint({
                    'epoch': idx_epoch + 1,
                    'state_dict': net.state_dict(),
                    'total_loss': total_loss_list,
                }, save_pth)
                opt.best_checkpoint_path = save_pth

        evaluated = (idx_epoch + 1) >= opt.begin_validation and (
            idx_epoch + 1) % opt.every_validation == 0
        record = {
            'run_id': opt.run_id,
            'epoch': idx_epoch + 1,
            'lr': float(scheduler.get_last_lr()[0]),
            'train_loss': epoch_loss,
            'val_loss': float(val_loss) if evaluated else None,
            'miou': float(results1[1]) if evaluated else None,
            'pd': float(results2[0]) if evaluated else None,
            'fa': float(results2[1]) if evaluated else None,
            'f1': float(f1_score) if evaluated else None,
            'epoch_seconds': round(epoch_seconds, 3),
        }
        with open(opt.metrics_path, 'a') as metrics_file:
            metrics_file.write(json.dumps(record, ensure_ascii=True) + '\n')

        if evaluated and opt.early_stopping_patience > 0 and (idx_epoch + 1) >= opt.min_epochs \
                and validation_checks_without_improvement >= opt.early_stopping_patience:
            print('Early stopping at epoch %d: validation mIoU did not improve for %d validation checks.'
                  % (idx_epoch + 1, validation_checks_without_improvement))
            break
    writer.close()


def build_final_test_command(test_script: str) -> list[str]:
    """Build a test.py command matching the trained model structure."""
    command = [
        sys.executable, test_script,
        '--model_names', opt.model_name,
        '--dataset_names', opt.dataset_name,
        '--dataset_dir', os.path.abspath(opt.dataset_dir),
        '--pth_dirs', os.path.abspath(opt.best_checkpoint_path),
        '--save_log', os.path.abspath(opt.run_dir),
        '--save_img_dir', os.path.join(os.path.abspath(opt.run_dir), 'results'),
        '--threshold', str(opt.threshold),
        '--no-save_img' if not opt.auto_test_save_img else '--save_img',
    ]
    if opt.cross_view:
        command.extend(['--cross_view', '--cross_view_topk', str(opt.cross_view_topk)])
    if opt.mamba_branch:
        command.append('--mamba_branch')
    if opt.gpu_id is not None:
        command.extend(['--gpu_id', opt.gpu_id])
    return command


def run_final_test() -> None:
    """Evaluate the selected checkpoint on the official test list."""
    if not opt.best_checkpoint_path or not os.path.isfile(opt.best_checkpoint_path):
        print('No validation checkpoint was saved; skipping automatic final test.')
        return
    test_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test.py')
    command = build_final_test_command(test_script)
    print('Running automatic official test with:', opt.best_checkpoint_path)
    subprocess.run(command, check=True)

if __name__ == '__main__':
    save_root = opt.save
    tensorboard_root = opt.log_dir
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            opt.run_id = f'{opt.dataset_name}_{opt.model_name}_{run_timestamp}'
            opt.run_dir = os.path.join(save_root, opt.run_id)
            opt.checkpoint_dir = os.path.join(opt.run_dir, 'checkpoints')
            opt.log_dir = (os.path.join(opt.run_dir, 'tensorboard') if tensorboard_root is None
                           else os.path.join(tensorboard_root, opt.run_id))
            os.makedirs(opt.checkpoint_dir, exist_ok=True)
            os.makedirs(opt.log_dir, exist_ok=True)
            opt.params_path = os.path.join(opt.run_dir, 'train_config.json')
            opt.f = open(os.path.join(opt.run_dir, 'train.log'), 'w', encoding='utf-8', buffering=1)
            opt.metrics_path = os.path.join(opt.run_dir, 'metrics.jsonl')
            with open(opt.metrics_path, 'w') as metrics_file:
                json.dump({
                    'run_id': opt.run_id,
                    'dataset': opt.dataset_name,
                    'model': opt.model_name,
                    'epochs': opt.epochs,
                    'batch_size': opt.batchSize,
                    'patch_size': opt.patchSize,
                    'seed': opt.seed,
                    'val_ratio': opt.val_ratio,
                    'validation_source': opt.validation_source,
                    'early_stopping_patience': opt.early_stopping_patience,
                    'early_stopping_min_delta': opt.early_stopping_min_delta,
                    'min_epochs': opt.min_epochs,
                    'run_dir': opt.run_dir,
                    'checkpoint_dir': opt.checkpoint_dir,
                    'tensorboard_dir': opt.log_dir,
                }, metrics_file, ensure_ascii=True)
                metrics_file.write('\n')
            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            if opt.auto_test:
                run_final_test()
            print('\n')
            opt.f.close()
