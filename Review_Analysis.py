"""
 CSI-535 || Artificial Intelligence Team project
 Team : Reviewers
 Topic : Sentiment analysis using product review data
 Authors: Rahul Chhapgar, Glen Sequeira, Nilesh Chakraborty
"""

from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.svm.libsvm import decision_function
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import naive_bayes
from sklearn import metrics
from util import *
import scipy

## ------------------------------------------------ Task-1 -------

def load(File):

    line_count = 0
    KEYS = []
    VALUES = []

    # Creating output file to write each user review in Dictionary format
    F2 = open("Amazon_Dict_Reviews.txt", "w")

    # Reading original Amazon Review data file
    F1 = open(File, "r")
    for line in F1:
        try:
            key, val = line.rstrip().split(": ")

            KEYS.append(key)
            if len(val) < 18:
                VALUES.append(val)
            else:
                VALUES.append(stopwords_removal(val))
            line_count += 1

            Dict = {}
            if line_count % 8 == 0:
                for num in range(len(KEYS)):
                    Dict[KEYS[num]] = VALUES[num]
                F2.writelines(json.dumps(Dict) + '\n')

            # if line_count == 5000:
            #     break
        except:
            continue

    # Number of lines in source file: 41,437,488    // 2.81 GB
    # Number of Reviews fetched: Approx: 5,179,686  \\ 0.5 GB
    # Reviews 2,800,500

def stopwords_removal(current_line):

    # Setting the value for Rating key @ Global Dictionary
    if current_line.__contains__(" out of 5 stars"):
        current_line = current_line.split()[0]
        return current_line
    # Setting the value for review key @ Global Dictionary
    else:
        updated_line = ' '
        for Word in current_line.lower().split():
            if Word not in stopwords:
                updated_line += Word + ' '
        return updated_line.strip()

## ------------------------------------------------ Task-2 -------

def Dict_Formation(Temp_Dict):

    # Formation of Global Dict for single words
    # Each word have unique dictionary which has its Rating score count per Rating

    for WORD in Temp_Dict:
        if WORD not in stopwords:
            # if word is already in Dictionary, increment its rating count
            if WORD in Global_dict.keys():
                Global_dict[WORD][Rating] += 1
            # if word is not in Dict, initialize all rating for it with 0
            #   Add count of 1 to word according to its Rating group
            else:
                Global_dict[WORD] = {'1.0': 0, '2.0': 0, '3.0': 0, '4.0': 0, '5.0': 0}
                Global_dict[WORD][Rating] = 1

def NOA_NOV_generation(RVW):

    # Defining local variables for Dictionary updates
    NOA_local = []
    NOV_local = []

    for i in range(len(RVW) - 3):
        if RVW[i][1] in ADV:
            if RVW[i + 1][1] in ADJ:
                NOA.append((RVW[i][0] + ' ' + RVW[i + 1][0]))
                NOA_local.append((RVW[i][0] + ' ' + RVW[i + 1][0]))
            elif RVW[i + 1][1] in VERB:
                NOV.append((RVW[i][0] + ' ' + RVW[i + 1][0]))
                NOV_local.append((RVW[i][0] + ' ' + RVW[i + 1][0]))
            elif RVW[i + 2][1] in ADJ:
                NOA.append((RVW[i][0] + ' ' + RVW[i + 1][0] + ' ' + RVW[i + 2][0]))
                NOA_local.append((RVW[i][0] + ' ' + RVW[i + 1][0] + ' ' + RVW[i + 2][0]))
            elif RVW[i + 2][1] in VERB:
                NOV.append((RVW[i][0] + ' ' + RVW[i + 1][0] + ' ' + RVW[i + 2][0]))
                NOV_local.append((RVW[i][0] + ' ' + RVW[i + 1][0] + ' ' + RVW[i + 2][0]))

    return NOA_local, NOV_local

def Cal_Sentiment_Score(Dict):

    # Calculating Gamma Values for each Dictionary word
    # Calculating Sentiment score for each word

    SS = {}     # Temporary Dict of Scores
    for Key in Dict:
        SumNum = 0
        SumDiv = 0

        # Individual Dictionary for each word
        RatingDict = Global_dict[Key]
        # For each rating group...
        for rating in RatingDict:

            # Checking and finding gamma value, to stop equation returning infinite value
            if Global_dict[Key][rating] == 0:
                gamma = 1
                SumDiv += gamma
            else:
                gamma = (Global_dict[Key]['5.0']) / (Global_dict[Key][rating])
                SumDiv += (gamma * Global_dict[Key][rating])

            # Calculation of numerator and denominator
            f = float(rating)
            SumNum += (f * gamma * Global_dict[Key][rating])

        # updating scores for each word with its sentiment score
        SS[Key] = SumNum / SumDiv

    # returning final sentiment scores
    return SS

def Global_30Plus():
    occuranceFile = open("occuranceFile.txt","w")
    # Creation of dictionary which has words which occurred more then 30 times
    for occurance in Global_dict:
        rating =Global_dict[occurance]
        if(rating['1.0'] + rating['2.0'] + rating['3.0'] + rating['4.0'] + rating['5.0'])>30:
            occuranceFile.writelines(json.dumps({occurance:Global_dict[occurance]})+'\n')



def COUNT(R, Count1, Count2, Count3, Count4, Count5):

    # Conversion of string value of rating to float
    rate = float(R)

    # incrementing count of rating variable based on occurrence
    if rate == 1.0:
        Count1 += 1
    elif rate == 2.0:
        Count2 += 1
    elif rate == 3.0:
        Count3 += 1
    elif rate == 4.0:
        Count4 += 1
    else:
        Count5 += 1

    return Count1, Count2, Count3, Count4, Count5

def PlotBar(C1, C2, C3, C4, C5):

    # All Rating value's final count for before plotting
    Count = [C1, C2, C3, C4, C5]
    stars = ('1-star', '2-star', '3-star', '4-star', '5-star')
    y_pos = np.arange(len(stars))

    plt.title('Review Categories')
    plt.bar(y_pos, Count, align='center')
    plt.xticks(y_pos, stars)
    plt.ylabel('Reviews')
    plt.show()

## ------------------------------------------------ Task-3 -------

def vec_fit(input_x):

    # creation of object for Count Vectorizer
    Vectorizer = CountVectorizer(min_df=1, tokenizer=lambda doc: doc, lowercase=False, stop_words='english')
    count_Vect = Vectorizer.fit_transform(input_x).toarray()
    tf_transform = TfidfTransformer()
    x_train_tft = tf_transform.fit_transform(count_Vect)

    return x_train_tft




## ------------------------------------------------------  _main_  -------

if __name__ == '__main__':

    start_time = time.time()

    ## ---------------------------------------------------- Task-1 -------
    FileName = "amazon_total.txt"
    # load(FileName)

    ## ---------------------------------------------------- Task-2 -------
    New_File = "Amazon_Dict_Reviews.txt"
    F = open(New_File, "r")
    X = []
    y = []

    count = 0
    for Line in F:
        count += 1
        data = json.loads(Line)

        # Fetching required data from Reviews
        Rating = data['rating'].encode('utf-8')

        Count_1, Count_2, Count_3, Count_4, Count_5 =\
            COUNT(Rating, Count_1, Count_2, Count_3, Count_4, Count_5)

        Review = data['review'].encode('utf-8')
        Review_token = nltk.tokenize.word_tokenize(Review)

        # Generation of X for Models
        X.append(Review)
        # print X

        # Generation of Y for Models
        Intensity = SentimentIntensityAnalyzer()
        answers = Intensity.polarity_scores(Review)
        Positivity, Negativity = answers['pos'], answers['neg']

        if Positivity > Negativity:
            y.append(1)
        else:
            y.append(0)

        # Dictionary generation from tokenize words
        Dict_Formation(Review_token)

        # generating tuples using POS Tagger; format: ( word , Token )
        RVWs = nltk.pos_tag(Review_token)

        # Generation of NOA and NOV list POS tagged words
        NOA_Local, NOV_Local = NOA_NOV_generation(RVWs)
        Dict_Formation(NOA_Local)
        Dict_Formation(NOV_Local)

        if count == 20:
            break

    PlotBar(Count_1, Count_2, Count_3, Count_4, Count_5)
    print Count_1, Count_2, Count_3, Count_4, Count_5

    # Checking each word/phrase occurrence
    Global_30Plus()

    # Calculation of Gamma value and Sentiment scores
    Sentiment_Score = Cal_Sentiment_Score(Global_dict)
    print Sentiment_Score
    ## ---------------------------------------------------- Task-3 -------
    #
    # # vectorizer = CountVectorizer()
    # Final_X = vec_fit(X)
    #
    # svm_classifier = svm.SVC()
    # svm_classifier.fit(Final_X, np.array(y)).decision_function(X)
    # predicted_label = svm_classifier.predict(X)
    # Acc_svm = metrics.accuracy_score(y, predicted_label)
    # print "Accuracy @ SVM = ", Acc_svm
    #
    # nb_classifier = naive_bayes.GaussianNB()
    # nb_classifier.fit(Final_X, np.array(y)).decision_function(X)
    # predicted_label = nb_classifier.predict(X)
    # Acc_nb = metrics.accuracy_score(y, predicted_label)
    # print "Accuracy @ DT = ", Acc_nb
    #
    # rf_classifier = RandomForestClassifier()
    # rf_classifier.fit(Final_X, np.array(y)).decision_function(X)
    # predicted_label = rf_classifier.predict(X)
    # Acc_rf = metrics.accuracy_score(y, predicted_label)
    # print "Accuracy @ RF = ", Acc_rf

    ## ------------------------------------------------------- END -------

    end_time = time.time()
    time = float("{0:.2f}".format((end_time - start_time)/60))
    print "Running time =", time, " Min"
