import os
import torch
from tqdm import tqdm
from utils.eval_model import eval
from torch.autograd import Variable


def train(model,
          device,
          trainloader,
          valloader,
          metric_loss,
          criterion,
          optimizer,
          scheduler,
          save_path,
          start_epoch,
          end_epoch,
          best_val_acc):
    best_acc = best_val_acc
    for epoch in range(start_epoch + 1, end_epoch + 1):
        f = open(os.path.join(save_path, 'log.txt'), 'a')
        model.train()
        print('Training %d epoch' % epoch)

        lr = next(iter(optimizer.param_groups))['lr']
        turn = True
        for _, data in enumerate(tqdm(trainloader)):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            embeddings, logits = model(images)
            m_loss = metric_loss(embeddings, labels)
            ce_loss = criterion(logits, labels)
            total_loss = ce_loss + m_loss

            total_loss.backward()
            optimizer.step()
            
        scheduler.step()
        
        f.write('\nEPOCH' + str(epoch) + '\n')
        # eval valset
        val_ce_loss_avg, val_metric_loss_avg, val_accuracy = eval(model, device, valloader, metric_loss, criterion, split='val')
        print('Validation set: Avg Val CE Loss: {:.4f}; Avg Val Metric Loss: {:.4f}; Val accuracy: {:.2f}%'.format(val_ce_loss_avg, val_metric_loss_avg, 100. * val_accuracy))
        f.write('Validation set: Avg Val CE Loss: {:.4f}; Avg Val Metric Loss: {:.4f}; Val accuracy: {:.2f}% \n'.format(val_ce_loss_avg, val_metric_loss_avg, 100. * val_accuracy))
       
        
        # save checkpoint
        print('Saving checkpoint')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'learning_rate': lr,
            'val_acc': val_accuracy
        }, os.path.join(save_path, 'current_model' + '.pth'))

        if val_accuracy > best_acc:
            print('Saving best model')
            f.write('\nSaving best model!\n')
            best_acc = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'learning_rate': lr,
                'val_acc': val_accuracy
            }, os.path.join(save_path, 'best_model' + '.pth'))
        f.close()