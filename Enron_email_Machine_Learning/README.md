Intent:
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed
into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a
significant amount of typically confidential information entered into the public record, including
tens of thousands of emails and detailed financial data for top executives. 

The goal of this project is to construct, tune and validate a machine learning classifier for
identifying “persons of interest” (pois) in the Enron scandal. The data sources are publicly
available internal email communications and financial records. Tools used Python with
Scikit-Learn to scale the data, select features, create classifiers, and cross-validate test results.

This is an example of a supervised learning problem. The Enron scandal was very public and
lead to several arrests and prosecutions of pois, because of these we already have historical
data features and labels flagging who was and was not a poi. Using these data and our human
intuition about the nature of a financial scandal, machine learning can be applied to
mathematically pinpoint correlations between a poi and features like salary, stock value,
frequency of email communication etc...

poi_id.py contains the code focused on the Machine learning investigation. If you wish to download and run the scrip please download the and run enron_project.zip
