import nltk
import pandas as pa#
from nltk.corpus import stopwords#
from sklearn.ensemble import RandomForestClassifier#
from sklearn.feature_extraction.text import CountVectorizer#
from bs4 import BeautifulSoup#
import numpy as np#
import re#
stemmer=nltk.stem.PorterStemmer()



trainData=pa.read_csv("/Users/madeshsivakumar/Desktop/saCode/labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)#
testData=pa.read_csv("/Users/madeshsivakumar/Desktop/saCode/testData.tsv",header=0,delimiter="\t",quoting=3)#
# forest=RandomForestClassifier(n_estimators=100)
print testData["id"][0]
print len(testData["review"])
def cleanData( raw_data ):#
    noTags=BeautifulSoup(raw_data).get_text()#
    letters_only=re.sub("[^a-zA-Z]", " ",noTags)#
    words=letters_only.lower().split()
    stop=set(stopwords.words("english"))
    noStopWords=[w for w in words if not w in stop]
    stemWords=[stemmer.stem(i) for i in noStopWords]
    return (" ".join( stemWords ))

cleanTrain=[]
print "Cleaning Reviews..................................\n"
for i in xrange(0,len(trainData["review"])):
    cleanTrain.append(cleanData(trainData["review"][i]))

###############
# output=pa.DataFrame( data={"id":trainData["id"],"review":cleanTrain})
# output.to_csv("cleanTrain.csv",index=False,quoting=3)
# print cleanTrain

print "Creating Bag of Words..................................\n"
vectorizer=CountVectorizer(analyzer="word",tokenizer = None, preprocessor = None, stop_words = None, max_features=5000)
features=vectorizer.fit_transform(cleanTrain)

Vdf=pa.DataFrame(features.A, columns=vectorizer.get_feature_names())
Vdf.describe()
# print features
# vocab=vectorizer.get_feature_names()
# print vocab

# output=pa.DataFrame( data={"id":trainData["id"],"features":features})
# output.to_csv("trainFeatures.csv",index=False,quoting=3,quotechar='')
# np.asarray(features)
features=features.toarray()
# print features.shape
#################
# np.savetxt("trainFeatures.csv", features)
#



print "Training Random Forest..................................\n"
forest = RandomForestClassifier(n_estimators = 100)
result=forest.fit(features,trainData["sentiment"])

cleanTest=[]
print "Cleaning Test Data..................................\n"
for i in xrange(0,len(testData["review"])):
    cleanTest.append(cleanData(testData["review"][i]))
testFeatures=vectorizer.transform(cleanTest)
# output=pa.DataFrame( data={"id":testData["id"],"features":testFeatures})
# output.to_csv("testFeatures.csv",index=False,quoting=3)
# np.asarray(testFeatures)
testFeatures=testFeatures.toarray()
#####################
# print testFeatures.shape
# np.savetxt("testFeatures.csv", testFeatures, delimiter="\t")


print "Predicting Output..................................\n"
result=forest.predict(testFeatures)
output=pa.DataFrame( data={"id":testData["id"],"sentiment":result})
output.to_csv("9_1_cost_result.csv",index=False,quoting=3)
