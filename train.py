import pandas as pd
from tqdm import tqdm
import numpy as np
from random import seed
import os
import argparse
from datetime import datetime

from sklearn.metrics import coverage_error, mean_absolute_error, accuracy_score, label_ranking_average_precision_score
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier

import torch.nn as nn

from encoders import VarBrownianEncoder, EarlyStopper

from transformers import get_linear_schedule_with_warmup    
from datasets import VarNytDataset, VarS2gDataset

# Setting up the device for GPU usage
from torch import cuda

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type =str,
                        help='Path to dataset directory')
    parser.add_argument('-bs','--batchsize', default=32, type=int,
                        help='Batch size')
    parser.add_argument('-ep','--epochs', default=100, type=int,
                        help='Epochs')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--no-finetune', dest='finetune', action='store_false')
    parser.set_defaults(finetune=False)
    parser.add_argument('-lr','--learningrate', default=5e-4, type=float,
                        help='Learning rate')
    parser.add_argument('-l','--loss', default="BCE_var", type=str,
                        help='Loss (either L2, BCE_var, BCE_time or BCE')
    parser.add_argument('-e','--encoder', default="DistilBERT", type=str,
                        help='Language encoder')
    parser.add_argument('-ls','--latentsize', default=100, type=int,
                        help='Size of the latent representation')
    parser.add_argument('-hs','--hidden_size', default=512, type=int,
                        help='Hidden size')
    parser.add_argument('-ax','--axis', default='authors', type=str,
                        help='Axis defining trajectories (either authors or topics)')
    parser.add_argument('-t','--timeprecision', default='Y', type=str,
                        help='Time precision for date classification (Y, M or D)')
    parser.add_argument('-dy','--dynamic', default='local', type=str,
                        help='Type of dynamic for each author : local or global')
    args = parser.parse_args()

    data_dir = args.dataset
    DATASET = data_dir.split(os.sep)[-2]
    BATCH_SIZE = args.batchsize
    EPOCHS = args.epochs
    LEARNING_RATE = args.learningrate
    ENCODER = args.encoder
    LOSS = args.loss
    FINETUNE  = args.finetune
    HIDDEN_DIM = args.hidden_size
    LATENT_SIZE = args.latentsize
    AXIS = args.axis
    TIME = args.timeprecision
    EVAL_STEPS = args.eval_steps
    DYNAMIC = args.dynamic

    data_dir = "C:\\Users\\EnzoT\\Documents\\datasets\\s2g\\imputation\\corpus.json"
    DATASET = data_dir.split(os.sep)[-3]
    TEMPORALITY = data_dir.split(os.sep)[-2]
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 5e-4
    ENCODER = "DistilBERT"
    LOSS = "BCE_var"
    FINETUNE  = False
    HIDDEN_DIM = 512
    LATENT_SIZE = 128
    AXIS = "authors"
    TIME = "Y"
    DYNAMIC = "local"

    print(DATASET, ENCODER)

    if DATASET == "nytg":
        dataset_train = VarNytDataset(data_dir = data_dir, encoder=ENCODER, train=True, seed=42, axis=AXIS, time_precision=TIME)
        dataset_test = VarNytDataset(data_dir = data_dir, encoder=ENCODER, train=False, seed=42, axis=AXIS, time_precision=TIME)
        dataset_val = VarNytDataset(data_dir = data_dir, encoder=ENCODER, train=False, val=True, seed=42, axis=AXIS, time_precision=TIME)
    elif DATASET == "s2g":
        dataset_train = VarS2gDataset(data_dir = data_dir, encoder=ENCODER, train=True, seed=42, axis=AXIS, time_precision=TIME)
        dataset_test = VarS2gDataset(data_dir = data_dir, encoder=ENCODER, train=False, seed=42, axis=AXIS, time_precision=TIME)
        dataset_val = VarS2gDataset(data_dir = data_dir, encoder=ENCODER, train=False, val=True, seed=42, axis=AXIS, time_precision=TIME)

    if LOSS != "L2":
        dataset_train.sample_negative(10)
        dataset_test.sample_negative(10)
        dataset_val.sample_negative(10)

    dataset_train.interpolate_axis(DYNAMIC, n_steps=100)

    method = "%s_%s_%s_%s" % (ENCODER, TEMPORALITY, LOSS, DYNAMIC)

    n_axis = dataset_train.n_axis
    max_date = max(dataset_test.data.ddelta.max(), dataset_train.data.ddelta.max(), dataset_val.data.ddelta.max())

    test_ids = list(dataset_test.data.id)

    model = VarBrownianEncoder(hidden_dim=HIDDEN_DIM, latent_dim=LATENT_SIZE, loss=LOSS,
                            tokenizer = ENCODER, finetune = FINETUNE, dynamic=DYNAMIC, 
                            method = method, n_axis = n_axis, L=10, test_ids=test_ids).to(device)

    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    batch = next(iter(dataloader_train))

    total_steps = len(dataloader_train) * EPOCHS

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    
    def get_evaluation_data(dataset_train, dataset_test, model):

        corpus_train = list(dataset_train.data.texts)
        docs_train = []
        for c in tqdm(chunks(corpus_train, BATCH_SIZE), total=len(list(range(0, len(corpus_train), BATCH_SIZE)))):

            with torch.no_grad():
                input_ids, attention_masks = dataset_train.tokenize_caption(c, device)
                z = model.encode(input_ids, attention_masks)
                docs_train.append(z.cpu().numpy())

        docs_train = np.vstack(docs_train)

        with torch.no_grad():
            z_s = model.z_estart.weight.cpu().numpy()
            z_d = model.z_eend.weight.cpu().numpy()

        corpus_test = list(dataset_test.data.texts)
        docs_test = []

        for c in tqdm(chunks(corpus_test, BATCH_SIZE), total=len(list(range(0, len(corpus_test), BATCH_SIZE)))):

            with torch.no_grad():
                input_ids, attention_masks = dataset_test.tokenize_caption(c, device)
                z = model.encode(input_ids, attention_masks)
                docs_test.append(z.cpu().numpy())

        docs_test = np.vstack(docs_test)

        time_train = dataset_train.data.timestep
        time_test = dataset_test.data.timestep

        label_train = dataset_train.data.axis_id
        label_test = dataset_test.data.axis_id

        return docs_train, docs_test, label_train, label_test, time_train, time_test, z_s, z_d
    
    from sklearn.model_selection import GridSearchCV

    def eval_fn(docs_test, labels, time_test, z_s, z_e):

        start = datetime.now()
        axis_embds = z_s + z_e

        aa = normalize(axis_embds, axis=1)
        dd = normalize(docs_test, axis=1)
        y_score = normalize(dd @ aa.transpose(), norm="l1")

        aut_doc_test = np.zeros(y_score.shape)
        aut_doc_test[[i for i in range(len(docs_test))], labels] = 1

        ce = coverage_error(aut_doc_test,y_score)/n_axis

        if DATASET=="s2g":
            acc = label_ranking_average_precision_score(aut_doc_test, y_score)
        else:
            acc = accuracy_score(labels, np.argmax(y_score, axis=1))
        print("[Evaluation for AA took %s]" % (str(datetime.now() - start)), flush=True)

        start = datetime.now()
        n_steps = 100
        t = [(1-i/n_steps) * z_s + i/n_steps * z_e for i in range(n_steps+1)]
        t=np.stack(t, axis=1)

        author_train=[]
        for i in range(dataset_train.n_axis):
            author_train.append(t[i,:,:])

        author_train = np.vstack(author_train)

        author_time = dataset_train.interpolation_df.timestep.values

        tuned_parameters={'n_neighbors':[3, 5, 10], "weights":['distance'], 'metric':['cosine', 'euclidean']}

        grid_search = GridSearchCV(KNeighborsClassifier(), tuned_parameters, scoring=["accuracy", "neg_mean_absolute_error"],  n_jobs=-1, cv=5, verbose=0, refit="accuracy")

        grid_search.fit(author_train, author_time)

        time_pred = grid_search.predict(docs_test)

        act = accuracy_score(time_test, time_pred)
        mae = mean_absolute_error(time_test, time_pred)
        print("[Evaluation for DD took %s]" % (str(datetime.now() - start)), flush=True)

        return ce, acc, act, mae
    
    docs_train, docs_test, label_train, label_test, time_train, time_test, z_s, z_e = get_evaluation_data(dataset_train, dataset_test, model)

    print("Beginning evaluation for dataset %s in %s" % (DATASET, TEMPORALITY), flush=True)
    start = datetime.now()
    ce, acc, act, mae = eval_fn(docs_test, label_test, time_test, z_s, z_e)
    print("[Evaluation at beginning in %s] Coverage : %.3f  | Accuracy : %.3f | MAE : %.3f | Time Accuracy : %.3f \n" % (str(datetime.now() - start), ce, acc, mae, act), flush=True)

    if LOSS=="L2":
        criterion = nn.MSELoss(reduction="none")
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='sum')

    early_stopper = EarlyStopper(patience=3, min_delta=1e-2)
    for epoch in range(1,EPOCHS+1):
        
        start = datetime.now()

        model.train()

        vloss = 0
        ploss = 0
        for batch in tqdm(dataloader_train):

            texts = batch['y_t']
            
            # Random masking of date token for the 10 first epochs following a decreasing probability
            texts = [t if p<1-(epoch-1)*0.2 else t[7:] for t, p in zip(texts,np.random.uniform(0,1,len(texts)))]

            axis = batch['axis'].to(device)

            label = batch['label'].to(device)

            start_pin = batch['start_pin'].to(device)
            t = batch['t'].to(device)
            end_pin = batch['end_pin'].to(device)

            input_ids, attention_mask = dataset_train.tokenize_caption(texts, device)

            loss, vloss, ploss = model(input_ids, attention_mask, axis, label, start_pin, t, end_pin, max_date, criterion)

            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            scheduler.step()

            vloss+=vloss

        ploss /= len(dataloader_train)
        vloss /= len(dataloader_train)

        training_time = datetime.now() - start

        if not os.path.isdir(os.path.join("model", DATASET, AXIS)):
            os.mkdir(os.path.join("model", DATASET, AXIS))

        model.eval()

        vloss_val = 0.0
        ploss_val = 0.0
        for batch in tqdm(dataloader_val):

            texts = batch['y_t']
            axis = batch['axis'].to(device)

            label = batch['label'].to(device)

            start_pin = batch['start_pin'].to(device)
            t = batch['t'].to(device)
            end_pin = batch['end_pin'].to(device)

            input_ids, attention_mask = dataset_train.tokenize_caption(texts, device)
            
            with torch.no_grad():
                loss, vloss, ploss = model(input_ids, attention_mask, axis, label, start_pin, t, end_pin, max_date, criterion)

            vloss_val+= vloss
            ploss_val+= ploss

        vloss_val/=len(dataloader_val)

        print("[%d/%d] in %s Validation loss : %.4f + %.4f  |  Training loss : %.4f + %.4f" % (epoch, EPOCHS, str(training_time), vloss_val.item(), ploss_val.item(), vloss, ploss), flush=True)

        if early_stopper.early_stop(vloss_val):             
            break

    start = datetime.now()
    docs_train, docs_test, label_train, label_test, time_train, time_test, z_s, z_e = get_evaluation_data(dataset_train, dataset_test, model)
    ce, acc, act, mae = eval_fn(docs_test, label_test, time_test, z_s, z_e)

    print("[Evaluation in %s] Coverage : %.3f  | Accuracy : %.3f | MAE : %.3f | Time Accuracy : %.3f\n" % (str(datetime.now() - start), ce, acc, mae, act), flush=True)
   
    print("We're finished !")

    torch.save(model.state_dict(), os.path.join("model", DATASET, "%s_%d_ckpt.pt" % (model.method, epoch)))