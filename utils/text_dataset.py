import numpy as np
import os
import json
from skimage import io

class TextDataSet():
  """TextDataSet
  process text input file dataset 
  text file format:
    image_path xmin1 ymin1 xmax1 ymax1 class1 xmin2 ymin2 xmax2 ymax2 class2
  """

  def __init__(self, dataset_params):
    """
    Args:
      dataset_params: A dict
    """
    #process params
    self.train_data_path = str(dataset_params['train_path'])
    self.test_data_path = str(dataset_params['test_path'])
    self.batch_size = int(dataset_params['batch_size'])
    self.epoch =  int(dataset_params['epoch'])
    self._epochs_completed = 0
    self._index_in_epoch = 0

  """ 讀取training data，必須在batch前呼叫才有東西
      Args:
      Returns:
  """
  def loadTrainData(self):
    # 讀training data 跟testing data
    tr_paths = self.ReadDirFiles(self.train_data_path)
    self.x_train, self.y_train = self.LoadData(tr_paths)
    self._num_examples = self.x_train.shape[0]

  """ 讀取testing data，必須在getTestData前呼叫才有東西
      Args:
      Returns:
  """
  def loadTestData(self):
    # 讀training data 跟testing data
    ts_paths = self.ReadDirFiles(self.test_data_path)
    self.x_test, self.y_test = self.LoadData(ts_paths)

  """ 得到testing data
     Args:
     Returns:
  """
  def getTestData(self):
    # 讀training data 跟testing data
    return self.x_test, self.y_test

  """ 讀取圖檔
    Args:
      fname: 檔案路徑 
    Returns:
      灰階image: (width, height) 
  """
  def LoadImage(self, fname):
      image = io.imread(fname)
      return image[:,:,0] / 255.

  def LoadAnnotation(self, fname):
    with open(fname) as data_file:
      data = json.load(data_file)

    left = min(data["objects"][0]["points"]["exterior"][0][0], data["objects"][0]["points"]["exterior"][1][0])
    right = max(data["objects"][0]["points"]["exterior"][0][0], data["objects"][0]["points"]["exterior"][1][0])

    top = min(data["objects"][0]["points"]["exterior"][0][1], data["objects"][0]["points"]["exterior"][1][1])
    bottom = max(data["objects"][0]["points"]["exterior"][0][1], data["objects"][0]["points"]["exterior"][1][1])

    return [left, top, right, bottom]

  """ 根據path傳回image路徑與json路徑
    Args:
      path: 路徑 
    Returns:
      all_paths: list [("image路徑","json路徑"), ..]
  """
  def ReadDirFiles(self, dname):
    paths = []
    for file in os.listdir(os.path.join(dname, "img")):
      bname = os.path.basename(file).split(".")[0]

      img_name = os.path.join(dname, "img", file)
      ann_name = os.path.join(dname, "ann", bname + ".json")
      paths.append((img_name, ann_name))
    return paths

  """ 根據path讀取image與json
    Args:
      path: 路徑 
    Returns:
      xs: nparray (num, width, height)
      ys: nparray (num, top, left, right, bottom)
  """
  def LoadData(self, paths):
    xs = []
    ys = []
    for ex_paths in paths:
      img_path = ex_paths[0]
      ann_path = ex_paths[1]
      xs.append(self.LoadImage(img_path))
      ys.append(self.LoadAnnotation(ann_path))

    return np.array(xs), np.array(ys)

  def batch(self):
    start = self._index_in_epoch
    self._index_in_epoch += self.batch_size
    if self._index_in_epoch > self._num_examples:
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self.x_train = self.x_train[perm]
      self.y_train = self.y_train[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = self.batch_size
      assert self.batch_size <= self._num_examples
    end = self._index_in_epoch
    return self.x_train[start:end], self.y_train[start:end]

  def epoch_completed(self):
    return self._epochs_completed

  def epoch(self):
    return self.epoch