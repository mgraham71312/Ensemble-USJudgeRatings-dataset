# Ensemble-USJudgeRatings-dataset
Bagging and Random Forest analysis on USJudgeRatings data.

## Dataset
This dataset is a survey of lawyers on various aspects of US Superior Court judges.  All columns are a 1-10 scale except for `Contact`, which is the number of times the lawyer has had contact with the judge in question.  There are 43 observations over 12 variables.

The dependent variable is Retention (`RTEN`), which suggests whether or not the judge is deemed worth of retention.  The independent variables include number of contacts of lawyer with judge (`CONT`), judicial integrity (`INTG`), demeanor (`DMNR`), diligence (`DILG`), case flow managing (`CFMG`), prompt decisions (`DECI`), preparation for trial (`PREP`), familiarity with law (`FAMI`), sound oral rulings (`ORAL`), sound written rulings (`WRIT`), and physical ability (`PHYS`).

The dataset was retrieved April 9, 2022, from an [R repository](https://vincentarelbundock.github.io/Rdatasets/datasets.html).

_Note:_ Previous analyses of this dataset involved addressing multicollinearity issues.  In this analysis, all variables/columns remain as is in order to examine how the ensemble models handle the multicollinearity/redundancy.

## Model
Changes to the dependent variable (`RTEN`) include: values less than 8.0 are “unworthy” (0) and values at 8.0 or greater are “worthy” (1).  _Note:_ 8.0 is an arbitrary number used only for this analysis. 
 
The cross validation includes only 3 folds due to the small size of the dataset.  

Compare the bagging and random forests models to the performance of a single decision tree.

## Performance
| Type | Number of Trees | Accuracy |
| --- | :---: | :---: |
| Decision Tree | 1 | 81% |
| Bagging | 10 | 84% |
| Bagging | 100 | 86% |
| Bagging | 500 | 88% |
| Random Forest | 10 | 91% |
| Random Forest | 100 | 88% |
| Random Forest | 500 | 88% |

## Conclusion
Previous projects using this dataset (to be uploaded on GitHub) were limited to two or three independent variables due to multicollinearity issues; however, this project utilizes all variables.  It attempts to improve upon a baseline decision tree model with bagging and random forests models.  The collection of bagged trees will have the strong predictor in the top split and will generally look similar and, therefore, highly correlate.  Averaging highly correlated quantities does not lead to a large reduction in variance.  Random forests decorrelates the trees, which makes the average of trees less variable and, therefore, more reliable.  Model accuracy trends upward as the models move from decision tree to bagging to random forest.  As the random forest number of trees increased, the accuracy leveled off at about 88 percent.  Overall, the random forest with 10 decision trees held the best accuracy metric of 91 percent.

## References:
New Haven Register. (1977, January 14). _Lawyers' ratings of state judges in the US Superior Court._ Retrieved April 9, 2022, from https://vincentarelbundock.github.io/Rdatasets/datasets.html

## License
MIT

