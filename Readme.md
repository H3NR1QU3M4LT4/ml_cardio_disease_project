# Cardiovascular Disease Prediction 

## How to run
* pip install requirements.txt
* python main.py --PATH_CSV=data/heart_cleveland_upload.csv --SAVE_REPORT_PATH=data/
* to be able to see the graphs from training and evaluation: tensorboard --logdir logs/fit

## Data Information
### There are 13 attributes
* age: age in years
* sex: sex (1 = male; 0 = female)
* cp: chest pain type 
  * value 0: typical angina
  * value 1: atypical angina
  * value 2: non-anginal pain
  * value 3: asymptomatic
* trestbps: resting blood pressure (in mm Hg on admission to the hospital)
* chol: serum cholestoral in mg/dl
* fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
* restecg: resting electrocardiographic results
    * value 0: normal
    * value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    * value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
* thalach: maximum heart rate achieved
* exang: exercise induced angina (1 = yes; 0 = no)
* oldpeak = ST depression induced by exercise relative to rest
* slope: the slope of the peak exercise ST segment
    * value 0: upsloping
    * value 1: flat
    * value 2: downsloping
* ca: number of major vessels (0-3) colored by flourosopy
* thal: 0 = normal; 1 = fixed defect; 2 = reversable defect and the label
* condition: 0 = no disease, 1 = disease
