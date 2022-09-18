# 第一次接觸 Flower 就上手
> Flower 官方網站 [在這裡](https://flower.dev/).
## 硬體設備及環境
- 作業系統 Windows10 專業版 21H1
- 中央處理器 Intel ® Core i7-2600 CPU @ 3.40GHz
- 記憶體 12GB
- 開發工具 PyCharm Community Edition 2022.2.1
- 程式語言 Python 3.9.13 (需要 Python 3.7 或以上)


## 初次使用安裝指令
```
python -m pip install --upgrade pip 
pip install flwr
pip install tensorflow
```

## 使用 tensorflow 搭建一個 FL 吧!
[範例1](https://flower.dev/docs/quickstart-tensorflow.html) ，使用 tensorflow 及 flower 建構一個自己的 federated learning 系統吧
### Client 端
首先新增檔名為 `client.py`，並匯入 flower 及 tensorflow 套件
```python=
import flwr as fl
import tensorflow as tf
```
我們使用 cifar10 資料集，並切分成 training 及 testing dataset
```python=
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
```

接下來我們需要一個模型，我們使用 MobilNetV2 ，設定 10 個輸出類別
```python=
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
```

有了 model 以後，要用 client 來控制這個 model 來 training 或做其他操作，並能夠將 weights 傳遞給 server。
```python=
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": float(accuracy)}
```
Flower client 有三個必需的 function
- get_parameters(): 用來取得 weights。
- fit(): 用來訓練剛剛的模型。
- evaluate(): 使用該 client 的 dataset 評估模型。

到了這裡，Flower 的 client 算是架設完成了。接下來用下面程式碼來啟動客戶端

```python=2
fl.client.start_numpy_client("localhost:7001", client=CifarClient())
```
這部分使用 `localhost` 與官方文件不同，若有錯誤請改回使用 `[::]`

### Server 端
Server 的建構又更簡單了，新增檔名為 `server.py` 後，只需要匯入 flower 套件並啟用 server 即可。

```python=
import flwr as fl

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3))
```
### 執行
看到這邊先恭喜你成功架設了第一個 FL 系統了，我們趕快來啟動它吧！
要注意這個專案需要執行兩個 client 端才能執行喔
