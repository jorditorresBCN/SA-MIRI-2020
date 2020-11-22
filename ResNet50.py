import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import argparse
import time
import sys

sys.path.append('/gpfs/projects/nct00/nct00002/cifar-utils')
from cifar import load_cifar

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--n_gpus', type=int, default=1)

args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs
n_gpus = args.n_gpus

train_ds, test_ds = load_cifar(batch_size)

#list all devices gpus
device_type = 'GPU'
devices = tf.config.experimental.list_physical_devices(device_type)
devices_names = [d.name.split("e:")[1] for d in devices]

strategy = tf.distribute.MirroredStrategy(devices=devices_names[:n_gpus])
with strategy.scope():
	model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=True, weights=None,
            input_shape=(128, 128, 3), classes=10)

	opt = tf.keras.optimizers.SGD(0.01*n_gpus)

	model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print(f" Model ResNet50 {n_gpus} GPUs","\n-----------------------")
print('Batch_size:', batch_size)
print('N_replicas:', strategy.num_replicas_in_sync,"\n")

start = time.time()
model.fit(train_ds, epochs=epochs, verbose=2)
end = time.time()
print('Avg per epoch:', round((end - start)/(epochs), 2),"\n\n")
