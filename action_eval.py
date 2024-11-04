# def train_epoch_action(model, train_loader, criterion, optimizer, epoch, writer, opts):

#     losses_action = [] 
#     model.train()
    
#     loop = tqdm((train_loader), total = len(train_loader))
#     for data in loop:
#         inputs = data[0].cuda()
#         labels = data[1].cuda(non_blocking=True)
            
#         optimizer.zero_grad()

#         outputs = model((inputs).permute(0,2,1,3,4))

#         loss_action = criterion(outputs,labels) 

#         losses_action.append(loss_action.item())
#         loss_action.backward()
        
#         optimizer.step()
#         loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
#         loop.set_postfix(loss = loss_action.item())
#     del loss_action, inputs, outputs, labels

#     print('Training Epoch: %d, loss_action: %.4f' % (epoch, np.mean(losses_action)))
#     writer.add_scalar(f'Training loss_action {opts.src}', np.mean(losses_action), epoch)
#     writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)