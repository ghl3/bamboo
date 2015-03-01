
Modeling Example
----------------


    from sklearn.ensemble import RandomForestClassifier
	from sklearn.linear_model import LogisticRegression
	import sklearn.datasets


	# Create a test dataset
	state = np.random.RandomState(10)
	features, classes = sklearn.datasets.make_classification(1000, 5, n_informative=2, class_sep=0.3, random_state=state)
	features = pd.DataFrame(features, columns=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'])
	classes = pd.Series(classes)
	
	# Create a ModelingData object
	data = bamboo.modeling.ModelingData(features, classes)
	
	# Plot a features
	data.hist('feature1', alpha=0.5, bins=np.arange(-1.0,1.0,.1))
![alt tag](https://raw.githubusercontent.com/ghl3/bamboo/master/images/modeling1.png)

	# One can also view all features at once
	data.hist_all(figsize=(12,8), alpha=0.5, normed=True)
![alt tag](https://raw.githubusercontent.com/ghl3/bamboo/master/images/modeling2.png)
	
	# Let's create training and testing data and train a classifier
	training, testing = data.train_test_split()
    
	training_balanced = training.get_balanced()

	clf = LogisticRegression()

	training_balanced.fit(clf)
	
	
	# And let's score the classifier on our testing set and view the score distributions by class
	testing.plot_proba(clf, 0, bins=np.arange(0.0, 1.0, 0.1), alpha=0.5, normalize=True)
![alt tag](https://raw.githubusercontent.com/ghl3/bamboo/master/images/modeling3.png)
	
	
	# Let's look deeper into the individual scores and some hypothetical performance data as a function of score threshold
	# Imagine that we're trying to classify the class '0'
	scores, summary = testing.get_classifier_performance_summary(clf, target=0)

	scores.head()	
		index	proba_0	proba_1	target
	0	674	0.853067	0.146933	0
	1	195	0.219421	0.780579	1
	2	136	0.369358	0.630642	0
	3	174	0.302821	0.697179	1
	4	756	0.428402	0.571598	1

	summary.head().T
	
	threshold	0.0	0.01	0.02	0.03	0.04
	accuracy	0.516000	0.516000	0.516000	0.516000	0.516000
	f1	0.680739	0.680739	0.680739	0.680739	0.680739
	false_negatives	0.000000	0.000000	0.000000	0.000000	0.000000
	false_positive_rate	1.000000	1.000000	1.000000	1.000000	1.000000
	false_positives	121.000000	121.000000	121.000000	121.000000	121.000000
	precision	0.516000	0.516000	0.516000	0.516000	0.516000
	recall	1.000000	1.000000	1.000000	1.000000	1.000000
	sensiticity	1.000000	1.000000	1.000000	1.000000	1.000000
	specificity	0.000000	0.000000	0.000000	0.000000	0.000000
	target	0.000000	0.000000	0.000000	0.000000	0.000000
	true_negatives	0.000000	0.000000	0.000000	0.000000	0.000000
	true_positive_rate	1.000000	1.000000	1.000000	1.000000	1.000000
	true_positives	129.000000	129.000000	129.000000	129.000000	129.000000



	summary.plot(x='false_positive_rate', y='true_positive_rate')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.savefig("images/modeling4.png")
![alt tag](https://raw.githubusercontent.com/ghl3/bamboo/master/images/modeling4.png)


	summary.plot(x='precision', y='recall')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.savefig("images/modeling5.png")
![alt tag](https://raw.githubusercontent.com/ghl3/bamboo/master/images/modeling5.png)
