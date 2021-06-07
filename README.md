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

Download the VGG19 model from [here](https://drive.google.com/file/d/1-3xhCa4yDJluwzAeOq1UXZD_puo6QU2B/view?usp=sharing) (We couldn't upload it into the GitHub because it is too large.)

After downloading, put it into the `model` folder

Run the command below

```sh
$ python app.py
```

The web app starts running on `http://127.0.0.1:5000/`.

Exit from virtual environment 

```sh
$ deactivate
```