import flask
import pickle
from sklearn.externals import joblib
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import warnings; warnings.simplefilter('ignore')
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Use pickle to load in the pre-trained model.
def tune_model(input_variables, no_of_trees):
	print("=====No of estimator===")
	print(no_of_trees)
	if no_of_trees == 20:
		with open(f'model/random-forest-classifier.pkl', 'rb') as f:
			rf_model = pickle.load(f)
	else:
		print("---Tune Model----")
		bankdata = pd.read_csv("bill_authentication.csv")
		X = bankdata.drop('Class', axis=1)  
		y = bankdata['Class']		
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)		
		classfier = RandomForestClassifier(n_estimators=20, random_state= 0)
		classfier.fit(X_train, y_train)
		y_pred = classfier.predict(X_test)
		with open('model/random-forest-classifier1.pkl', 'wb') as file:
			pickle.dump(classfier, file)
		with open(f'model/random-forest-classifier1.pkl', 'rb') as f:
			rf_model = pickle.load(f)

	prediction =rf_model.predict(input_variables)[0]
	print("-----Prediction------")
	print(prediction)
	return prediction


# with open(f'model/svm-classifier.pkl', 'rb') as f:
    # svm_model = pickle.load(f)

# with open(f'model/decision-tree-classifier.pkl', 'rb') as f:
    # dt_model = pickle.load(f)
#categorical_engineered_features = []

def clean_data(new_data):
	print("Inside Function")
	print(new_data)
	
	feature_names = ['OverallGrade', 'Obedient', 'ResearchScore', 'ProjectScore']
	numeric_feature_names = ['ResearchScore', 'ProjectScore']
	categoricial_feature_names = ['OverallGrade', 'Obedient']
	
	#--data preparation
	prediction_features = new_data[feature_names]

	#--scaling
	ss = StandardScaler()
			# fit scaler on numeric features
	ss.fit(prediction_features[numeric_feature_names])

	# scale numeric features now
	prediction_features[numeric_feature_names] = ss.transform(prediction_features[numeric_feature_names])
	# view updated feature-set
	print(prediction_features)
	prediction_features[numeric_feature_names] = ss.transform(prediction_features[numeric_feature_names])

	#--engineering categorical variables
	prediction_features = pd.get_dummies(prediction_features, columns=categoricial_feature_names)

	#--view feature set
	print(prediction_features)


	categorical_engineered_features = ['OverallGrade_F', 'OverallGrade_A', 'Obedient_N', 'Obedient_Y', 'OverallGrade_C', 'OverallGrade_E', 'OverallGrade_B']

	print(categorical_engineered_features)
	# add missing categorical feature columns
	current_categorical_engineered_features = set(prediction_features.columns) - set(numeric_feature_names)


	missing_features = set(categorical_engineered_features) - current_categorical_engineered_features
    
    # add zeros since feature is absent in these data samples
	for feature in missing_features:
		prediction_features[feature] = [0] * len(prediction_features)

	print(prediction_features)

	#Load the Model

	model = joblib.load(r'Model/regression_model.pickle')
	#model = joblib.load(r'Model/dt_regression_model.pickle')
	predictions = model.predict(prediction_features)

	##--display results
	new_data['Recommend'] = predictions
	print(new_data)

	return predictions


app = flask.Flask(__name__, template_folder='templates')
app.debug = True
@app.route('/', methods=['GET', 'POST'])
def main():
	if flask.request.method == 'GET':
		return(flask.render_template('main.html'))

	if flask.request.method == 'POST':

		model_choice = flask.request.form['model']
		print(model_choice)
		variance = flask.request.form['variance']
		skewness = flask.request.form['skewness']
		curtosis = flask.request.form['curtosis']
		entropy  = flask.request.form['entropy']
		estimator = flask.request.form['estimator']

		print(f"--Estimaor---{estimator}")

		input_variables = pd.DataFrame([[variance, skewness, curtosis, entropy]], columns=['variance', 'skewness', 'curtosis' ,'entropy'], dtype=float)

		print(input_variables)

		if model_choice == "svm_model":
			model = "SVM Classifier"
			prediction = svm_model.predict(input_variables)[0]
		elif model_choice == "decision_tree":
			model = "Decision Tree"
			prediction = dt_model.predict(input_variables)[0]
		else:
			model = "Random Forest Algorithm"
			prediction = tune_model(input_variables, estimator)

		print(f"Prediction-{prediction}")
		return flask.render_template('main.html', result='True', original_input={
			'variance' : variance,
			'skewness' : skewness,
			'curtosis' : curtosis,
			'entropy'  : entropy
			},
			Prediction= prediction, Model=model)


UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/build', methods=['GET', 'POST'])
def build():
	if flask.request.method == "GET":
		msg = "Upload a file in csv format only.\n File Should Contain OverallGrade, Obedient, ResearchScore, ProjectScore Columns Only"		

	if flask.request.method == "POST":
		input_file = flask.request.files['input_file']
		print(input_file)
		if input_file:
			df = pd.read_csv(input_file)
			print(df.head())
			feature_names = ['OverallGrade', 'Obedient', 'ResearchScore', 'ProjectScore']
			training_features = df[feature_names]

			outcome_name = ['Recommend']
			outcome_labels = df[outcome_name]

			numeric_feature_names = ['ResearchScore', 'ProjectScore']
			categoricial_feature_names = ['OverallGrade', 'Obedient']

			ss = StandardScaler()
			# fit scaler on numeric features
			ss.fit(training_features[numeric_feature_names])

			# scale numeric features now
			training_features[numeric_feature_names] = ss.transform(training_features[numeric_feature_names])
			# view updated feature-set
			print(training_features)

			#--Engineering Categorical Features
			training_features = pd.get_dummies(training_features, columns=categoricial_feature_names)
			print(training_features)

			#--get list of new categorical features
			# global categorical_engineered_features
			categorical_engineered_features = list(set(training_features.columns) - set(numeric_feature_names))
			print(categorical_engineered_features)

			X_train, X_test, y_train, y_test  = train_test_split(training_features, outcome_labels, test_size = 0.25)
			model = LogisticRegression()
			#model = DecisionTreeRegressor()

			model.fit(X_train, y_train)

			#--simple evaluation on training data
			pred_labels = model.predict(training_features)
			actual_labels = np.array(outcome_labels['Recommend'])


			#print('Accuracy:', float(accuracy_score(actual_labels, pred_labels))*100, '%')
			print('Classification Stats:')
			print(classification_report(actual_labels, pred_labels))

			Accuracy = float(accuracy_score(actual_labels, pred_labels))*100
			msg = "Accuracy of this model is" +str(Accuracy)+ "%"
			joblib.dump(model, r'Model/dt_regression_model.pickle') 
			
	return flask.render_template('build.html', msg=msg)


@app.route("/test_model", methods=['GET','POST'])
def test_model():
	if flask.request.method == "GET":
		msg = "Please Input All required Valid Data"
		return flask.render_template('test_model.html', msg=msg)

	if flask.request.method == "POST":
		name = flask.request.form['name']
		overallgrade = flask.request.form['overallgrade']
		obedient = flask.request.form['obedient']
		projectscore = flask.request.form['projectscore']
		researchscore = flask.request.form['researchscore']
		my_dict = {}
		my_dict['Name'] = name
		my_dict['OverallGrade'] = overallgrade
		my_dict['Obedient'] = obedient
		my_dict['ProjectScore'] = projectscore
		my_dict['ResearchScore'] = researchscore
		print(my_dict)
		input_data = pd.DataFrame(my_dict, index=[0]);
		print(input_data)
		print(type(input_data))		
		prediction = clean_data(input_data)
		my_dict['result'] = prediction
		return flask.render_template('test_model.html', result = True, Result=my_dict)

if __name__ == '__main__':
	app.run(debug=True)