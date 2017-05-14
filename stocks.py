import quandl, math, sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats.stats import pearsonr

class Result:
	def __init__(self, accuracy, classifier):
		self.accuracy = accuracy
		self.classifier = classifier


def regression(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
	clf = LinearRegression()
	clf.fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)
	return Result(accuracy, clf)


latest_n = 20 if len(sys.argv) < 2 else int(sys.argv[1])
verbose = True if '-v' in sys.argv else False

df = quandl.get('WIKI/ORCL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low','Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = 100*((df['Adj. High'] - df['Adj. Low'])/ df['Adj. Low'])
df['PCT_CHANGE'] = 100*(df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open']


df = df[['HL_PCT', 'PCT_CHANGE', 'Adj. Close', 'Adj. Volume']]


forecast_col = 'Adj. Close'
forecast_out = int(math.ceil(0.01*len(df)))


df['Label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop('Label', axis=1))
y = np.array(df['Label'])


X_scaled = preprocessing.scale(X)
reg_res_wproc = regression(X_scaled[:-latest_n], y[:-latest_n])

print 'accuracy ->',
print reg_res_wproc.accuracy
clf = reg_res_wproc.classifier
actual_price = y[-latest_n:]
predicted_price = clf.predict(X_scaled[-latest_n:])
print "Pearson correlation ->",
print pearsonr(actual_price, predicted_price)

if verbose:
	# print "Latest Data .."
	# print X[-latest_n:]
	print "Latest price"
	print actual_price
	print "Predicted price"
	print predicted_price





