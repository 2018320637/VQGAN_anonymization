# def train_epoch_privacy(model, train_loader, criterion, optimizer, epoch, writer, opts):

#     losses_privacy = [] 
#     model.train()
#     scaler = GradScaler()
    
#     loop = tqdm((train_loader), total = len(train_loader))
#     for data in loop:
#         inputs = data[0].cuda()
#         labels = data[2].cuda(non_blocking=True)
#         labels = labels.unsqueeze(1).expand(-1, inputs.size(1), -1)
#         labels = labels.reshape(-1, labels.size(2))

#         optimizer.zero_grad()

#         B, T, C, H, W = inputs.shape
#         inputs = inputs.reshape(-1, C, H, W)
        
#         with autocast():
#             outputs = model(inputs)
#             loss_privacy = criterion(outputs, labels)

#         scaler.scale(loss_privacy).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         losses_privacy.append(loss_privacy.item())
#         writer.add_scalar('Train loss privacy step', loss_privacy.item(), epoch)
        
#         loop.set_description(f'Epoch [{epoch}/{opts.num_epochs}]')
#         loop.set_postfix(loss = loss_privacy.item())
#     del loss_privacy, inputs, outputs, labels

#     print('Training Epoch: %d, loss_privacy: %.4f' % (epoch, np.mean(losses_privacy)))
#     writer.add_scalar(f'Training loss_privacy {opts.src}', np.mean(losses_privacy), epoch)

#     return model