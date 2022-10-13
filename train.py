import json
import icl_dataset
import torch
import transformer_model
import tqdm
import numpy as np
import generate_dataset
def model_train(num_epochs, batch_size, lr, model_file):
    set_length = 10000
    model = transformer_model.SimpleTransformer(torch.device('cuda'))

    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9,0.999), eps = 1e-9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=num_epochs, steps_per_epoch = int(np.ceil(float(set_length) / float(batch_size))), pct_start=0.1)
    train_losses = []



    for epoch in tqdm.trange(num_epochs, desc='training', unit='epoch'):
        train_losses_this_epoch = []
        data = generate_dataset.generate_dataset(set_length)
        dataset = icl_dataset.ICLDataset(data)
        train_length = int(len(dataset) * 0.9)
        valid_length = len(dataset) - train_length
        train_set, valid_set = torch.utils.data.random_split(dataset, [train_length, valid_length])

        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                       collate_fn=icl_dataset.ICLDataset.collate)
        valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True,
                                                       collate_fn=icl_dataset.ICLDataset.collate)

        with tqdm.tqdm(train_dataloader,desc ='epoch ' + str(epoch+1), unit='batch', total=len(train_dataloader)) as batch_iterator:
            model.train()
            for i, batch in enumerate(batch_iterator, start=1):
                optimizer.zero_grad()
                loss = model.calculate_loss(batch)
                loss.backward()
                optimizer.step()
                scheduler.step()
                batch_iterator.set_postfix(loss=loss.item())
                train_losses.append(loss.item())
                train_losses_this_epoch.append(loss.item())
        print("average training loss this epoch:" + str(np.mean(train_losses_this_epoch)))
        model.eval()


        # pred_correct = 0
        # with tqdm.tqdm(train_dataloader, desc='train epoch ' + str(epoch + 1), unit="batch",
        #                total=len(train_dataloader)) as valid_batch_iterator:
        #     for i, batch in enumerate(valid_batch_iterator, start=1):
        #         logits, labels = model.forward(batch)
        #         pred = logits.argmax(dim=-1)
        #         pred_correct += torch.sum(torch.eq(pred, labels))
        #
        # print("train accuracy:" + str(pred_correct / len(train_set)))

        pred_correct =0
        with tqdm.tqdm(valid_dataloader, desc = 'validation epoch ' + str(epoch+1), unit="batch", total=len(valid_dataloader)) as valid_batch_iterator:
            for i, batch in enumerate(valid_batch_iterator, start=1):
                logits, labels = model.forward(batch)
                pred = logits.argmax(dim=-1)
                pred_correct += torch.sum(torch.eq(pred,labels))

        print("valid accuracy:" + str(pred_correct/len(valid_set)))

        model_save_file_name = model_file + '_epoch_'+ str(epoch+1) + '.pt'
        # torch.save(model.state_dict(), model_save_file_name)
        # print("model saved:"+model_save_file_name)

