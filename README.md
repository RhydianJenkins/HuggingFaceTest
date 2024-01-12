If you haven't already, start a virtual environment

```bash
python3 -m venv myenv
source myenv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Download the model (this can take some time)

```bash
git-lfs clone https://huggingface.co/facebook/bart-large-cnn
```

Run the example

```bash
python3 ./index.py
```
