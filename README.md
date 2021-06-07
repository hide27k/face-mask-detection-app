# Face Mask Detector

# How to run it in the local environment

Set-up virtual environment

```sh
$  virtualenv -p python3 venv v
$ source venv/bin/activate
```

Install all dependencies by `pip`.

```sh
$ pip install -r requirements.txt
```

Chose the running model

The default source doe use ResNet34 as the running model.
If you want to run the VGG19, download it from [here](https://drive.google.com/file/d/1-3xhCa4yDJluwzAeOq1UXZD_puo6QU2B/view?usp=sharing). We put the model in google drive because it could not fit in the GitHub.

After downloading, put it into the model folder and update the filepath as below.

```py
device = torch.device("cpu")
model = ResNet34().to(device)
model.load_state_dict(
    torch.load("./model/resnet34_model_v0.pth", map_location=lambda storage, loc: storage)
)
model.eval()
```

Run the command below

```sh
$ python app.py
```

The web app starts running on `http://127.0.0.1:5000/`.

Exit from virtual environment 

```sh
$ deactivate
```