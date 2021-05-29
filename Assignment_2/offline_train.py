import os
import sys

os.system("nohup bash -c '" +
 sys.executable + " lstm_ae_MNIST_offline.py  >result.txt" +
 "' &")

# os.system("nohup bash -c '" +
#  sys.executable + " dgcnn_classification.py --bit 1  >result.txt" +
#  "' &")
