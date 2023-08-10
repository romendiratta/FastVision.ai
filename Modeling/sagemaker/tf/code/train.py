import argparse
import codecs
import json
import logging
import numpy as np
import pandas as pd
import os
import re

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from model import get_model
from utils import get_dataset


logging.getLogger().setLevel(logging.INFO)


class CustomTensorBoardCallback(TensorBoard):
    def on_batch_end(self, batch, logs=None):
        pass


def save_history(path, history):

    history_for_json = {}
    # transform float values that aren't json-serializable
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            history_for_json[key] == history.history[key].tolist()
        elif type(history.history[key]) == list:
            if type(history.history[key][0]) == np.float32 or type(history.history[key][0]) == np.float64:
                history_for_json[key] = list(map(float, history.history[key]))

    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(history_for_json, f, separators=(',', ':'), sort_keys=True, indent=4) 


def save_model(model, output):

    # create a TensorFlow SavedModel for deployment to a SageMaker endpoint with TensorFlow Serving
    tf.contrib.saved_model.save_keras_model(model, args.model_dir)
    logging.info("Model successfully saved at: {}".format(output))
    return


def main(args):

    if 'sourcedir.tar.gz' in args.tensorboard_dir:
        tensorboard_dir = re.sub('source/sourcedir.tar.gz', 'model', args.tensorboard_dir)
    else:
        tensorboard_dir = args.tensorboard_dir

    logging.info("Writing TensorBoard logs to {}".format(tensorboard_dir))

    logging.info("getting data")

    train_dataset = get_dataset(args.epochs,
                                args.batch_size,
                                args.data_dir,
                                'train')
    val_dataset = get_dataset(args.epochs,
                              args.batch_size,
                              args.data_dir,
                              'val')
    test_dataset = get_dataset(args.epochs,
                               args.batch_size,
                               args.data_dir,
                               'test')

    logging.info("configuring model")
    logging.info(f"Hosts: {os.environ.get('SM_HOSTS')}")

    size = len(args.hosts)

    model = get_model(args.learning_rate, args.weight_decay, args.optimizer, args.momentum, size)
    callbacks = []
    if args.current_host == args.hosts[0]:
        callbacks.append(ModelCheckpoint(args.output_data_dir + '/checkpoint-{epoch}.h5'))
        callbacks.append(CustomTensorBoardCallback(log_dir=tensorboard_dir))

    logging.info("Starting training")

    history = model.fit(train_dataset,
                        steps_per_epoch=(num_examples_per_epoch('train') // args.batch_size) // size,
                        epochs=args.epochs,
                        validation_data=val_dataset,
                        validation_steps=(num_examples_per_epoch('val') // args.batch_size) // size, 
                        callbacks=callbacks)

    score = model.evaluate(test_dataset,
                           steps=num_examples_per_epoch('test') // args.batch_size,
                           verbose=0)

    logging.info('Test loss:{}'.format(score[0]))
    logging.info('Test accuracy:{}'.format(score[1]))

    # PS: Save model and history only on worker 0
    if args.current_host == args.hosts[0]:
        save_history(args.model_dir + "/ps_history.p", history)
        save_model(model, args.model_dir)


def num_examples_per_epoch(subset='train'):
    if subset == 'train':
        return 600
    elif subset == 'val':
        return 70
    elif subset == 'test':
        return 70
    else:
        raise ValueError('Invalid data subset "%s"' % subset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--hosts',type=list,default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host',type=str,default=os.environ.get('SM_CURRENT_HOST'))
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument('--model_dir',type=str,required=True,help='The directory where the model will be stored.')
    parser.add_argument('--model_output_dir',type=str,default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output_data_dir',type=str,default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--output-dir',type=str,default=os.environ.get('SM_OUTPUT_DIR'))
    parser.add_argument('--tensorboard-dir',type=str,default=os.environ.get('SM_MODULE_DIR'))
    parser.add_argument('--weight-decay',type=float,default=2e-4,help='Weight decay for convolutions.')
    parser.add_argument('--learning-rate',type=float,default=0.001,help='Initial learning rate.')
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--data-config',type=json.loads,default=os.environ.get('SM_INPUT_DATA_CONFIG'))
    parser.add_argument('--fw-params',type=json.loads,default=os.environ.get('SM_FRAMEWORK_PARAMS'))
    parser.add_argument('--optimizer',type=str,default='adam')
    parser.add_argument('--momentum',type=float,default='0.9')
    
    args = parser.parse_args()

    main(args)