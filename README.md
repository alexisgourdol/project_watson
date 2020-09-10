# Data analysis
- Document here the project: project_watson
- Description: NLI model that assigns labels of 0, 1, or 2 (corresponding to entailment, neutral, and contradiction) to pairs of premises and hypotheses
- Data Source: https://www.kaggle.com/c/contradictory-my-dear-watson/data
- Type of analysis: text data, multiclass classification, categorizationaccuracy

Please document the project the better you can.

# Info on the data
- train.csv, test.csv, sample_submission.csv from Kaggle competition
- train_trans_to_en.csv from Kaggle Notebook : https://www.kaggle.com/tuckerarrants/contradictorytranslatedtrain?select=train_en.csv
- test_trans_to_en.csv generated using functions in translation.py (in a jupyter notebook :/, bad style)

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
  $ sudo apt-get install virtualenv python-pip python-dev
  $ deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
  $ make clean install test
```

Check for project_watson in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/project_watson`
- Then populate it:

```bash
  $ ##   e.g. if group is "{group}" and project_name is "project_watson"
  $ git remote add origin git@gitlab.com:{group}/project_watson.git
  $ git push -u origin master
  $ git push -u origin --tags
```

Functionnal test with a script:
```bash
  $ cd /tmp
  $ project_watson-run
```
# Install
Go to `gitlab.com/{group}/project_watson` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:
```bash
  $ sudo apt-get install virtualenv python-pip python-dev
  $ deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:
```bash
  $ git clone gitlab.com/{group}/project_watson
  $ cd project_watson
  $ pip install -r requirements.txt
  $ make clean install test                # install and test
```
Functionnal test with a script:
```bash
  $ cd /tmp
  $ project_watson-run
```

# Continus integration
## Github
Every push of `master` branch will execute `.github/workflows/pythonpackages.yml` docker jobs.
## Gitlab
Every push of `master` branch will execute `.gitlab-ci.yml` docker jobs.
