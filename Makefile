# ----------------------------------
#             PARAMS
# ----------------------------------

BUCKET_NAME=wagon-watson-perso

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

TRAIN_PATH=project_watson/data/train.csv
TEST_PATH=project_watson/data/test.csv
UPLOADED_TRAIN_NAME=train.csv
UPLOADED_TEST_NAME=test.csv

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# will store the packages uploaded to GCP for the training
BUCKET_TRAINING_FOLDER=trainings

##### Project  - - - - - - - - - - - - - - - - - - - - - - -

PROJECT_ID=wagon-bootcamp-288408

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

##### Machine configuration - - - - - - - - - - - - - - - -

REGION=europe-west4

##### Paython params  - - - - - - - - - - - - - - - - - - -

PYTHON_VERSION=3.7
FRAMEWORK=
RUNTIME_VERSION=2.1

##### Package params  - - - - - - - - - - - - - - - - - - -

PACKAGE_NAME=project_watson
FILENAME=trainer

##### Job - - - - - - - - - - - - - - - - - - - - - - - - -

JOB_NAME=training_bert_$(shell date +'%Y%m%d_%H%M%S')



# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* project_watson/*.py

black:
	@black scripts/* project_watson/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit=$(VIRTUAL_ENV)/lib/python*

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr project_watson-*.dist-info
	@rm -fr project_watson.egg-info

install:
	@pip install . -U

all: clean install test black check_code


uninstal:
	@python setup.py install --record files.txt
	@cat files.txt | xargs rm -rf
	@rm -f files.txt

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      LOCAL AND GOOGLE CLOUD
# ----------------------------------

run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs \
		--scale-tier=BASIC_TPU \
		-- \
		--distribution_strategy=tpu \
		--worker-machine-type=cloud_tpu

# Create model version based on that SavedModel directory
create_model_version:
	gcloud ai-platform versions create $MODEL_VERSION \
	  --model $MODEL_NAME \
	  --runtime-version 1.15 \
	  --python-version 3.7 \
	  --framework tensorflow \
	  --origin $SAVED_MODEL_PATH

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ __pycache__
	@rm -fr build dist *.dist-info *.egg-info
	@rm -fr */*.pyc
	@rm model.joblib

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u lologibus2

pypi:
	@twine upload dist/* -u lologibus2


# ----------------------------------
#          GCP project setup
# ----------------------------------

create_bucket:
	-@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

upload_data:
	# -@gsutil cp train_1k.csv gs://wagon-ml-my-bucket-name/data/train_1k.csv
	-@gsutil cp ${TRAIN_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${UPLOADED_TRAIN_NAME}
	-@gsutil cp ${TEST_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${UPLOADED_TEST_NAME}

create_model:
	-@gcloud ai-platform models create XLMBERT \
  	--regions ${REGION}



run_streamlit:
   @streamlit run app.py
