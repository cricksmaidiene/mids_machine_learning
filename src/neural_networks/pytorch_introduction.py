# Databricks notebook source
# MAGIC %md
# MAGIC # PyTorch
# MAGIC
# MAGIC * Developed at Facebook AI in 2016
# MAGIC * Preferable to quickly prototype, and by researchers
# MAGIC * More Pythonic, and easy to tinker with low level details than Tensorflow, according to most users
# MAGIC * [Building the Same Model with PyTorch & TensorFlow](https://www.youtube.com/watch?v=ay1E1f8VqP8)
# MAGIC * [Crash Course Notebook](https://colab.research.google.com/drive/1eiUBpmQ4m7Lbxqi2xth1jBaL61XTKdxp?usp=sharing)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Components

# COMMAND ----------

# DBTITLE 1,torch.NN import
from torch import nn

# COMMAND ----------

# MAGIC %md
# MAGIC * Every model created in Pytorch inherits from `pytorch.nn` called `nn.module`
# MAGIC * We define the inputs and outputs of the model along with the definitions of the layers of the network within this class

# COMMAND ----------

# DBTITLE 1,Extending NN class
class SimpleNeuralNet(nn.Module):
    ...

# COMMAND ----------

# MAGIC %md
# MAGIC * Every Pytorch inherited neural net needs to implement the `__init__` method and `forward` method.
# MAGIC * The `__init__` can define the layers of the model, and `forward` defines how data is passed from one layer to the next. 
# MAGIC > `__init__` is like the static declaration, while `forward` is like `main()` or where execution comes forth across the static components

# COMMAND ----------

# DBTITLE 1,Defining required methods
class SimpleNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        ...

# COMMAND ----------

# MAGIC %md
# MAGIC Declare a GPU torch device and pass the model object to the GPU

# COMMAND ----------

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# COMMAND ----------

simple_neural_net_model = SimpleNeuralNet().to(device=device)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tensors

# COMMAND ----------

# DBTITLE 1,Tensor
t = torch.Tensor([1,2,3])
print(t.shape)
print(t.shape == t.size())
t

# COMMAND ----------

# DBTITLE 1,Argmax function
torch.argmax(t)

# COMMAND ----------

# DBTITLE 1,Multi-Dim Tensor
t2 = torch.Tensor([[1, 2, 3], [1, 2, 4], [2, 5, 7]])
print(t2.shape)
t2

# COMMAND ----------

# Note different method of getting argmax
t2.argmax()

# COMMAND ----------

# MAGIC %md
# MAGIC When you call `t2.argmax()`, it treats the tensor as if it's a flat array (because no dimension is specified) and returns the index of the maximum value in this flat array. 

# COMMAND ----------

t2.argmax(dim=0)

# COMMAND ----------

print(torch.empty(3, 3, 3))

# COMMAND ----------

print(torch.rand(3, 3, 3))

# COMMAND ----------

print(torch.zeros(3, 3, 3))

# COMMAND ----------

print(torch.ones(3, 3, 3))

# COMMAND ----------


