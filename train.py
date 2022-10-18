import os, sys
import json
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix,f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from utils import get_dataset, create_data_loader,train,evaluate



def plot_bar(value,name,x_name,y_name,title):
    fig, ax = plt.subplots(tight_layout=True)

    ax.set(xlabel=x_name, ylabel=y_name,title=title)

    ax.barh(name, value)
   
  
    return ax.figure 

def evaluate_full(args,model, data_loader,device,type_):
    if type_ not in ['valid','test']:
        raise Exception(f"`type_` must be either `test` or `valid`!")


    # Get acc, f1 and confusion matrix
    model.eval()
    with torch.no_grad():
        acc=[]

        preds, targets = [],[]
        for input, target in data_loader:


            input, target = input.to(device), target.to(device)


            # calculate accuracy
            prediction = model(input).logits
            predicted_index = prediction.argmax(1)

            preds.extend(predicted_index.cpu().numpy().tolist())
            targets.extend(target.cpu().numpy().tolist())

            train_acc = torch.sum(predicted_index == target).cpu().item()
            final_train_acc = train_acc/input.shape[0]
            acc.append(final_train_acc)
    
    final_acc = sum(acc)/len(acc) 
    f1_scores = f1_score(targets, preds, average='weighted').tolist() 


    # Creating  a confusion matrix,which compares the y_test and y_pred
    cm = confusion_matrix(targets, preds)

    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(cm)
                        
    #Plotting the confusion matrix
    plt.figure()
    sns.heatmap(cm_df, annot=True)
    plt.title(f'Confusion Matrix ({args.filename.upper()}) ({type_.upper()})')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.savefig(os.path.join(args.CM_PATH,f'{args.filename.lower()}_confusion_matrix_{type_.upper()}.png'))

    data = {'acc':final_acc,'f1':f1_scores}
    METRICS_FILE = os.path.join(args.CM_PATH,f'{args.filename.lower()}_METRICS_{type_.upper()}.json')
    with open(METRICS_FILE,'w+') as file_:
        json.dump(data,file_)

    return final_acc






def main(args):
    EXPERIMENT_FOLDER = args.experiment_directory

    if not os.path.exists(EXPERIMENT_FOLDER):
        raise Exception(f"Experiment directory must exist. You specified `{EXPERIMENT_FOLDER}` which does not exist") 
   
    CM_PATH = os.path.join(EXPERIMENT_FOLDER,args.metrics_folder)
    args.CM_PATH = CM_PATH
    os.makedirs(CM_PATH,exist_ok=True)


    model_checkpoint = "facebook/wav2vec2-base"
    batch_size = args.batch_size
    num_labels = 10

    label2id, id2label = dict(), dict()
    labels = ['0','1','2','3','4','5','6','7','8','9']

    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label



    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    SAMPLE_RATE = args.sample_rate
    FILE_NAME = args.filename
    SAVE_PATH = os.path.join(args.save_model_path,f'{FILE_NAME}.pth')
    EVAL_STEP=args.eval_step
    train_path = args.train_path
    test_path = args.test_path

    LOSS_JSON_FILE = os.path.join(EXPERIMENT_FOLDER,f'loss_{FILE_NAME}.json')
    ACC_JSON_FILE = os.path.join(EXPERIMENT_FOLDER,f'val_acc_{FILE_NAME}.json')

    LEARNING_RATE = 3e-5
    EPOCHS = args.num_epochs

    #Preprocessing the data

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
    max_duration = 2.0  # seconds

    usd,valid_dataset,test_dataset=  get_dataset(args,feature_extractor,
                                                16000,
                                                device,
                                                train_path,
                                                test_path)
    # Create dataloader
    train_dataloader = create_data_loader(usd, batch_size)
    valid_dataloader = create_data_loader(valid_dataset, batch_size)
    test_dataloader = create_data_loader(test_dataset, batch_size)



    # construct model and assign it to device

    model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint, 
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    ).to(device)
    MULTILINGUAL_MODEL_PATH = args.multilingual_model_path
    if args.multilingual_model_path is not None and os.path.exists(MULTILINGUAL_MODEL_PATH):
        print('Using pretrained multilingual model.......................')
        model_saved = torch.load(MULTILINGUAL_MODEL_PATH)
        model.load_state_dict(model_saved)


    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(),
                                    lr=LEARNING_RATE)


    # train model
    train(args,model, train_dataloader,
            valid_dataloader, loss_fn, 
            optimiser, device, EPOCHS,
            SAVE_PATH,LOSS_JSON_FILE,
            ACC_JSON_FILE,feature_extractor)

    _ = evaluate_full(args,model, train_dataloader,device,'valid')
    if test_dataloader is not None:
        test_acc = evaluate_full(args,model, test_dataloader,device,'test')
        print(f"Test accuracy is {test_acc}")
    else:
        print(f"No test dataset was provided! So not performing evaluation on test.")
    if args.push_to_hub:
        model.push_to_hub(f"chrisjay/{FILE_NAME.lower()}")
        feature_extractor.push_to_hub(f"chrisjay/{FILE_NAME.lower()}")
    print('ALL DONE')




if __name__=="__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser('Afro Digits Speech Finetuning with WaV2Vec')
    parser.add_argument('--audio_homepath', type=str, default='crowd-speech-africa/data',
        help='Path to afro digit audio files.')
    parser.add_argument('--experiment_directory', type=str, default='./',
        help='Direcory for experiment. Deafults to current working directory')
    parser.add_argument('--metrics_folder', type=str, default='metrics',
        help='Direcory name where metrics files will be saved (default: %(default)s)')
    parser.add_argument('--batch_size', type=int, default=64,
        help='Batch size (default: %(default)s)')
    parser.add_argument('--sample_rate', type=int, default=48_000,
        help='Sample rate of your training audio data (default: %(default)s)')
    parser.add_argument('--num_epochs', type=int, default=1,
        help='Number of epochs (default: %(default)s)')
    parser.add_argument('--filename', type=str, default='afro-digits-speech',
        help='name of experiment. Model, and metric files will be named with this filename (default: %(default)s)')
    parser.add_argument('--save_model_path', type=str, default='./',
        help='Path to save model checkpoints. Default is current working directory')
    parser.add_argument('--eval_step', type=int, default=1,
        help='Number of steps before evaluation (default: %(default)s)')
    parser.add_argument('--save_validation_data',type=bool,default=False,
        help='Whether or not to save the validation data')
    parser.add_argument('--save_test_data',type=bool,default=False,
        help='Whether or not to save the test data')
    parser.add_argument('--push_to_hub',type=bool,default=False,
        help='Whether or not to push the best model to the HF hub')
    parser.add_argument('--train_path',type=str,default=None,
        help='Path to training CSV file')
    parser.add_argument('--valid_path',type=str,default=None,
        help='Path to validation CSV file. If `save_validation_data` is True, then this becomes the path to save the validation file')
    parser.add_argument('--test_path',type=str,default=None,
        help='Path to test CSV file. If `save_test_data` is True, then this becomes the path to save the test file')
    parser.add_argument('--multilingual_model_path',type=str,default=None,
        help='Path to multilingual model checkpoint')

    args = parser.parse_args()


    main(args)



